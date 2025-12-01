import json
import os
import re
import logging
import time
from typing import Any, Dict, List, Optional
from datasets import load_dataset, load_from_disk
from openai import OpenAI, AsyncOpenAI
import verifiers as vf
from verifiers.envs.tool_env import ToolEnv

logger = logging.getLogger(__name__)

def llm_tool_executor(tool_name: str, arguments: Dict[str, Any],
                      base_url: str, api_key_var: str, model: str) -> str:
    """Execute a tool call by proxying to an LLM with timeout."""
    try:
        client = OpenAI(
            base_url=base_url, 
            api_key=os.getenv(api_key_var, ""),
            timeout=30.0
        )
        system = (
            f"You are executing the tool '{tool_name}' with the given arguments.\n"
            "Return ONLY the direct result value. No explanation. Keep it short."
        )
        user = json.dumps({"tool": tool_name, "arguments": arguments})
        
        start = time.time()
        resp = client.chat.completions.create(
            model=model,
            temperature=0.0,
            max_tokens=100,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        elapsed = time.time() - start
        result = (resp.choices[0].message.content or "").strip()
        print(f"    -> Result ({elapsed:.1f}s): {result[:50]}...")
        return result
    except Exception as e:
        print(f"    -> ERROR: {e}")
        return f"Error: {str(e)[:50]}"

def _extract_tool_calls_from_text(text: str) -> List[Dict[str, Any]]:
    blocks = re.findall(r"<tool(?:_call)?>(.*?)</tool(?:_call)?>", text, re.DOTALL)
    calls: List[Dict[str, Any]] = []
    for block in blocks:
        try:
            parsed = json.loads(block.strip())
        except Exception:
            continue
        if isinstance(parsed, list):
            calls.extend([item for item in parsed if isinstance(item, dict) and "name" in item])
        elif isinstance(parsed, dict) and "name" in parsed:
            calls.append(parsed)
    return calls

def _get_completion_text(completion: Any) -> str:
    if isinstance(completion, list):
        return " ".join(
            str(message.get("content", ""))
            for message in completion
            if isinstance(message, dict)
        )
    return str(completion)

def _detect_hop_count(example: Dict[str, Any]) -> int:
    task = example.get("task", "")
    if isinstance(task, str):
        if "3_hop" in task or "3hop" in task:
            return 3
        if "6_hop" in task or "6hop" in task:
            return 6
        if "9_hop" in task or "9hop" in task:
            return 9
    answer = example.get("answer", "[]")
    expected_tools: List[Any] = []
    if isinstance(answer, str):
        try:
            expected_tools = json.loads(answer)
        except Exception:
            expected_tools = []
    elif isinstance(answer, list):
        expected_tools = answer
    num_tools = len(expected_tools) if isinstance(expected_tools, list) else 0
    if num_tools <= 3:
        return 3
    if num_tools <= 6:
        return 6
    return 9


class CustomToolEnv(ToolEnv):
    def __init__(self, dataset, system_prompt, tools, parser, max_turns,
                 exec_base_url, exec_api_key_var, exec_model, **kwargs):
        super().__init__(
            tools=tools,
            dataset=dataset,
            system_prompt=system_prompt,
            parser=parser,
            max_turns=max_turns,
            **kwargs,
        )
        self.exec_base_url = exec_base_url
        self.exec_api_key_var = exec_api_key_var
        self.exec_model = exec_model
        self._tool_call_count = 0
        self._last_response = ""
        self._repetition_count = 0

    async def setup_state(self, state, **kwargs):
        state = await super().setup_state(state, **kwargs)
        self._tool_call_count = 0
        self._last_response = ""
        self._repetition_count = 0
        return state

    def is_completed(self, messages, state, **kwargs) -> bool:
        if state.get("turns", 0) >= self.max_turns:
            return True
        if isinstance(messages, list) and messages:
            last_msg = messages[-1]
            if isinstance(last_msg, dict):
                content = last_msg.get("content", "")
                if "<answer>" in content and "</answer>" in content:
                    return True
                if content.strip() == self._last_response.strip() and len(content) > 50:
                    self._repetition_count += 1
                    if self._repetition_count >= 2:
                        return True
                else:
                    self._last_response = content
                    self._repetition_count = 0
        if self._tool_call_count >= 10:
            return True
        return False

    async def env_response(self, messages, state, **kwargs):
        try:
            if not messages:
                return [], state
            last_msg = messages[-1] if isinstance(messages, list) else messages
            content = last_msg.get("content", "") if isinstance(last_msg, dict) else str(last_msg)
            tool_calls = _extract_tool_calls_from_text(content)
            if tool_calls:
                results = []
                for tool_call in tool_calls:
                    self._tool_call_count += 1
                    tool_name = tool_call.get("name", "")
                    tool_args = tool_call.get("arguments", {})
                    if tool_name == "execute":
                        actual_tool = tool_args.get("tool_name", "")
                        actual_args = tool_args.get("arguments", {})
                    else:
                        actual_tool = tool_name
                        actual_args = tool_args
                    print(f"[TOOL {self._tool_call_count}] {actual_tool}")
                    result = llm_tool_executor(
                        actual_tool, actual_args,
                        base_url=self.exec_base_url,
                        api_key_var=self.exec_api_key_var,
                        model=self.exec_model,
                    )
                    results.append(f'<result name="{actual_tool}">{result}</result>')
                return [{"role": "user", "content": "\n".join(results)}], state
            return [], state
        except Exception as e:
            logger.error(f"env_response error: {e}")
            return [], state


def load_environment(
    dataset_name: str = "Anna4242/tool-n1-combined-3-6-9-hop-corrected-split",
    train_split: str = "train",
    eval_split: str = "eval",
    dataset_path: Optional[str] = None,
    eval_dataset_path: Optional[str] = None,
    use_local_default_path: bool = True,
    exec_base_url: str = "https://openrouter.ai/api/v1",
    exec_api_key_var: str = "OPENROUTER_API_KEY",
    exec_model: str = "x-ai/grok-4.1-fast:free",
    judge_base_url: str = "https://openrouter.ai/api/v1",
    judge_api_key_var: str = "OPENROUTER_API_KEY",
    judge_model: str = "x-ai/grok-4.1-fast:free",
    max_turns: int = 10,
    **kwargs,
) -> CustomToolEnv:
    # Load datasets
    if dataset_path and os.path.exists(dataset_path):
        dataset = load_from_disk(dataset_path)
    elif use_local_default_path and os.path.exists(
        "/workspace/anushka/verifiers/data/tool-n1-multihop/combined-3-6-9-hop-corrected"
    ):
        dataset = load_from_disk(
            "/workspace/anushka/verifiers/data/tool-n1-multihop/combined-3-6-9-hop-corrected"
        )
    else:
        dataset = load_dataset(dataset_name, split=train_split)

    eval_dataset = None
    if eval_dataset_path and os.path.exists(eval_dataset_path):
        eval_dataset = load_from_disk(eval_dataset_path)
    else:
        try:
            eval_dataset = load_dataset(dataset_name, split=eval_split)
        except Exception as e:
            logger.error(f"Failed to load eval: {e}")

    def process_example(example: Dict[str, Any]) -> Dict[str, Any]:
        prompt = example.get("prompt", [])
        if isinstance(prompt, str):
            try:
                prompt = json.loads(prompt)
            except:
                prompt = []
        example["prompt"] = prompt if isinstance(prompt, list) else []
        
        answer = example.get("answer", "[]")
        expected_tools = []
        if isinstance(answer, str):
            try:
                expected_tools = json.loads(answer)
            except:
                pass
        elif isinstance(answer, list):
            expected_tools = answer
        example["info"] = {
            "num_tools": len(expected_tools) if isinstance(expected_tools, list) else 0,
            "expected_tools": expected_tools,
            "hop_count": _detect_hop_count(example),
        }
        return example

    from datasets import Dataset
    
    train_rows = [process_example(ex) for ex in dataset]
    train_rows = [ex for ex in train_rows if ex.get("prompt") and isinstance(ex["prompt"], list)]
    dataset = Dataset.from_list(train_rows)

    if eval_dataset:
        eval_rows = [process_example(ex) for ex in eval_dataset]
        eval_rows = [ex for ex in eval_rows if ex.get("prompt") and isinstance(ex["prompt"], list)]
        eval_dataset = Dataset.from_list(eval_rows)

    parser = vf.XMLParser(
        fields=["think", "tool_call", "result", "answer"],
        answer_field="answer",
    )

    env = CustomToolEnv(
        dataset=dataset,
        eval_dataset=eval_dataset,
        system_prompt="",
        tools=[],
        parser=parser,
        max_turns=max_turns,
        exec_base_url=exec_base_url,
        exec_api_key_var=exec_api_key_var,
        exec_model=exec_model,
        **kwargs,
    )

    # Judge client for chaining reward
    judge_client = AsyncOpenAI(
        base_url=judge_base_url, 
        api_key=os.getenv(judge_api_key_var, ""),
        timeout=30.0
    )

    # ==================== REWARD 1: FORMAT (0.25) ====================
    def format_reward(completion, answer=None, **kwargs) -> float:
        text = _get_completion_text(completion)
        think_pattern = r"<think>(.*?)</think>"
        tool_pattern = r"<tool_call>(.*?)</tool_call>"
        think_matches = list(re.finditer(think_pattern, text, re.DOTALL))
        tool_matches = list(re.finditer(tool_pattern, text, re.DOTALL))
        
        if not tool_matches:
            print(f"[FORMAT] No tool calls found. Score: 0.0")
            return 0.0
        
        correct_format = 0
        for tool_match in tool_matches:
            tool_start = tool_match.start()
            has_think_before = any(
                think_match.end() < tool_start 
                for think_match in think_matches
            )
            if has_think_before:
                correct_format += 1
        
        score = correct_format / len(tool_matches)
        print(f"[FORMAT] {correct_format}/{len(tool_matches)} correct. Score: {score:.2f}")
        return score

    # ==================== REWARD 2: TOOL NAME (0.25) ====================
    def tool_name_reward(completion, answer=None, **kwargs) -> float:
        text = _get_completion_text(completion)
        expected_tools = []
        if isinstance(answer, str):
            try:
                expected_tools = json.loads(answer)
            except:
                pass
        elif isinstance(answer, list):
            expected_tools = answer
        
        if not expected_tools:
            print(f"[TOOL NAME] No expected tools. Score: 0.0")
            return 0.0
        
        expected_names = [t.get("name", "").lower().strip() for t in expected_tools]
        called_tools = _extract_tool_calls_from_text(text)
        called_names = [t.get("name", "").lower().strip() for t in called_tools]
        
        if not called_names:
            print(f"[TOOL NAME] No tools called. Score: 0.0")
            return 0.0
        
        matches = sum(1 for called in called_names if called in expected_names)
        score = min(1.0, matches / len(expected_names))
        print(f"[TOOL NAME] {matches}/{len(expected_names)} match. Score: {score:.2f}")
        return score

    # ==================== REWARD 3: ORDER (0.25) ====================
    def order_reward(completion, answer=None, **kwargs) -> float:
        text = _get_completion_text(completion)
        expected_tools = []
        if isinstance(answer, str):
            try:
                expected_tools = json.loads(answer)
            except:
                pass
        elif isinstance(answer, list):
            expected_tools = answer
        
        if not expected_tools:
            print(f"[ORDER] No expected tools. Score: 0.0")
            return 0.0
        
        expected_names = [t.get("name", "").lower().strip() for t in expected_tools]
        called_tools = _extract_tool_calls_from_text(text)
        called_names = [t.get("name", "").lower().strip() for t in called_tools]
        
        if not called_names:
            print(f"[ORDER] No tools called. Score: 0.0")
            return 0.0
        
        correct_order = 0
        for i, expected_name in enumerate(expected_names):
            if i < len(called_names) and called_names[i] == expected_name:
                correct_order += 1
            else:
                break
        
        score = correct_order / len(expected_names)
        print(f"[ORDER] {correct_order}/{len(expected_names)} in order. Score: {score:.2f}")
        return score

    # ==================== REWARD 4: CHAINING (0.25) ====================
    CHAINING_PROMPT = """Check if tool calls are CHAINED (output of tool N used in tool N+1).

Response:
{response}

Score 0.0-1.0 (1.0 = good chaining, 0.0 = no chaining).
Reply with ONLY a number:"""

    async def chaining_reward(prompt, completion, answer=None, **kwargs) -> float:
        text = _get_completion_text(completion)
        called_tools = _extract_tool_calls_from_text(text)
        if len(called_tools) < 2:
            print(f"[CHAINING] <2 tools, N/A. Score: 1.0")
            return 1.0
        
        try:
            resp = await judge_client.chat.completions.create(
                model=judge_model,
                temperature=0.3,
                max_tokens=10,
                messages=[{"role": "user", "content": CHAINING_PROMPT.format(response=text[:2000])}],
            )
            result = resp.choices[0].message.content.strip()
            numbers = re.findall(r'([0-9]*\.?[0-9]+)', result)
            if numbers:
                score = max(0.0, min(1.0, float(numbers[-1])))
                print(f"[CHAINING] Score: {score:.2f}")
                return score
        except Exception as e:
            print(f"[CHAINING] Error: {e}")
        return 0.5

    # ==================== RUBRIC ====================
    rubric = vf.JudgeRubric(
        parser=parser,
        judge_client=judge_client,
        judge_model=judge_model,
        judge_prompt="",
        judge_sampling_args={"temperature": 0.3, "max_tokens": 10},
    )
    
    rubric.add_reward_func(format_reward, weight=0.25)
    rubric.add_reward_func(tool_name_reward, weight=0.25)
    rubric.add_reward_func(order_reward, weight=0.25)
    rubric.add_reward_func(chaining_reward, weight=0.25)

    print("=" * 60)
    print("REWARDS: Format(0.25) + ToolName(0.25) + Order(0.25) + Chain(0.25)")
    print(f"Tool Exec: {exec_model}")
    print(f"Judge: {judge_model}")
    print("=" * 60)

    env.rubric = rubric
    return env


