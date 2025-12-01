# VF Nemotron Multi-hop Tool-use Environment

A Reinforcement Learning from Verifier Feedback (RLVR) environment for training language models on multi-hop (3-6-9) multi-turn tool-use tasks with judge-based rewards.

## What is RLVR?

**RLVR (Reinforcement Learning from Verifier Feedback)** is a training paradigm implemented by the [Verifiers framework](https://github.com/arcee-ai/verifiers). Instead of training on human feedback, models learn from automated verifiers that evaluate their outputs across multiple criteria.

### Key Concepts

- **Environments**: Define interaction protocols between models and tasks (multi-turn conversations, tool use, etc.)
- **Rubrics**: Composable reward functions that evaluate model performance
- **RL Training**: Uses GRPO (Group Relative Policy Optimization) to train models based on verifier scores

**Learn more:**
- [Verifiers GitHub](https://github.com/arcee-ai/verifiers)
- [Verifiers Documentation](https://verifiers.readthedocs.io/)
- [Verifiers Overview](https://github.com/arcee-ai/verifiers/blob/main/docs/source/overview.md)

## 🎯 Features

### Multi-hop Tool Execution
- Supports 3-hop, 6-hop, and 9-hop tool chaining tasks
- Automatic hop count detection from dataset
- LLM-based tool execution (configurable model)
- Multi-turn conversation support (up to 10 turns by default)

### 4-Component Reward System
1. **Format Reward (0.25)**: Ensures `<think>` tags precede `<tool_call>` tags
2. **Tool Name Reward (0.25)**: Matches called tool names with expected tools
3. **Order Reward (0.25)**: Validates tool call sequence matches expected order
4. **Chaining Reward (0.25)**: LLM judge evaluates if tool outputs are properly chained

### Production-Ready
- Dataset: `Anna4242/tool-n1-combined-3-6-9-hop-corrected-split`
- Separate train/eval splits
- Comprehensive error handling and logging
- Timeout protection for tool execution

## 📦 Installation

### Prerequisites

- Python >=3.11
- API key for tool execution (configurable)
- HuggingFace token for dataset access (if using HuggingFace datasets)

### Quick Install

```bash
cd /path/to/verifiers

# Set environment variables (replace with your actual keys)
export HF_TOKEN="your-hf-token"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
export OPENROUTER_API_KEY="your-openrouter-key"  # Or your preferred API key
export HF_HUB_ENABLE_HF_TRANSFER=0

# Login to HuggingFace (if using HF datasets)
huggingface-cli login --token $HF_TOKEN

# Install environment
uv run --active vf-install vf-nemotron-multihop -p ./environments
python -c "import vf_nemotron_multihop; print('OK')"
```

## 🚀 Usage

### Quick Evaluation

```bash
uv run vf-eval vf-nemotron-multihop \
  --api-base-url "https://openrouter.ai/api/v1" \
  --api-key-var OPENROUTER_API_KEY \
  --model "x-ai/grok-4.1-fast:free" \
  --num-examples 1 \
  --rollouts-per-example 1 \
  --temperature 0.0 \
  --max-tokens 2048 \
  --env-args '{"exec_model":"x-ai/grok-4.1-fast:free", "judge_model":"x-ai/grok-4.1-fast:free", "max_turns":10}' \
  --verbose
```

**Note**: Replace API keys and model names with your own credentials.

### Training Configuration

Example training config (`configs/nemotron_multihop_train.toml`):

```toml
[model]
name = "Qwen/Qwen2.5-7B-Instruct"

[training]
num_iterations = 1
num_train_epochs = 1
gradient_checkpointing = true

[training.sft]
learning_rate = 1e-5
per_device_train_batch_size = 2
gradient_accumulation_steps = 8
warmup_ratio = 0.1
max_seq_length = 4096

[training.grpo]
num_generations = 4
learning_rate = 5e-7
per_device_train_batch_size = 1
gradient_accumulation_steps = 16
warmup_ratio = 0.05
max_seq_length = 4096
max_new_tokens = 2048
max_grad_norm = 0.5
beta = 0.01
num_train_epochs = 1

[environment]
name = "vf-nemotron-multihop"

[environment.args]
dataset_name = "Anna4242/tool-n1-combined-3-6-9-hop-corrected-split"
train_split = "train"
eval_split = "eval"
use_local_default_path = true
exec_base_url = "https://openrouter.ai/api/v1"
exec_api_key_var = "OPENROUTER_API_KEY"
exec_model = "x-ai/grok-4.1-fast:free"
judge_base_url = "https://openrouter.ai/api/v1"
judge_api_key_var = "OPENROUTER_API_KEY"
judge_model = "x-ai/grok-4.1-fast:free"
max_turns = 10

[logging]
run_name = "qwen25-7b-multihop-grpo"
log_with = "wandb"
project_name = "multihop-tool-training"

[checkpointing]
output_dir = "./outputs/qwen25_7b_multihop_grpo"
save_strategy = "steps"
save_steps = 100
```

## 📊 Dataset

- **Dataset**: `Anna4242/tool-n1-combined-3-6-9-hop-corrected-split`
- **Splits**: `train` and `eval`
- **Task Types**: 3-hop, 6-hop, and 9-hop tool chaining
- **Format**: Each example contains:
  - `prompt`: List of messages (conversation history)
  - `answer`: Expected tool call sequence (JSON array)
  - `task`: Task description (may contain hop count info)

## 🔧 Configuration

### Tool Execution Model

The **tool execution model** is an LLM that executes tool calls by proxying them through a chat completion API. When the agent calls a tool, the environment sends the tool name and arguments to this model, which returns the tool's result.

- **Default**: `x-ai/grok-4.1-fast:free` (configurable)
- **Purpose**: Execute tools that the agent requests
- **Configuration**: Set via `exec_model` parameter
- **API**: Uses OpenAI-compatible API (OpenRouter, OpenAI, local vLLM, etc.)

### Judge Model

The **judge model** is an LLM that evaluates whether tool calls are properly chained (i.e., whether the output of tool N is used as input to tool N+1). This is used for the Chaining Reward component.

- **Default**: `x-ai/grok-4.1-fast:free` (configurable)
- **Purpose**: Evaluate tool chaining quality for reward calculation
- **Configuration**: Set via `judge_model` parameter
- **API**: Uses OpenAI-compatible API (OpenRouter, OpenAI, local vLLM, etc.)

### Environment Arguments

```python
load_environment(
    dataset_name="Anna4242/tool-n1-combined-3-6-9-hop-corrected-split",
    train_split="train",
    eval_split="eval",
    dataset_path=None,  # Optional: local dataset path
    eval_dataset_path=None,  # Optional: local eval dataset path
    use_local_default_path=True,  # Use local path if available
    exec_base_url="https://openrouter.ai/api/v1",  # API endpoint for tool execution
    exec_api_key_var="OPENROUTER_API_KEY",  # Environment variable name for API key
    exec_model="x-ai/grok-4.1-fast:free",  # Tool execution model
    judge_base_url="https://openrouter.ai/api/v1",  # API endpoint for judge
    judge_api_key_var="OPENROUTER_API_KEY",  # Environment variable name for judge API key
    judge_model="x-ai/grok-4.1-fast:free",  # Judge model for chaining reward
    max_turns=10,  # Maximum conversation turns
)
```

**Important**: Never commit API keys or tokens. Always use environment variables.

## 🎯 Reward System

The reward system uses **4 equal-weighted components** (each 25%) that evaluate different aspects of multi-hop tool use:

### Format Reward (25%)
- **What it checks**: Ensures proper structure - each `<tool_call>` must be preceded by a `<think>` tag
- **Score calculation**: `correct_format_count / total_tool_calls`
- **Example**: 
  - ✅ Good: `<think>I need to search...</think><tool_call>[...]</tool_call>`
  - ❌ Bad: `<tool_call>[...]</tool_call>` (missing think tag)

### Tool Name Reward (25%)
- **What it checks**: Whether the agent calls the correct tools (matches expected tool names)
- **Score calculation**: `min(1.0, matching_tools / expected_tools)`
- **Example**: If expected tools are `[search_user, get_posts]` and agent calls `[search_user, get_posts]`, score = 1.0

### Order Reward (25%)
- **What it checks**: Whether tools are called in the correct sequence
- **Score calculation**: `correct_sequence_length / expected_sequence_length`
- **Example**: 
  - ✅ Good: Expected `[A, B, C]`, called `[A, B, C]` → score = 1.0
  - ❌ Bad: Expected `[A, B, C]`, called `[A, C, B]` → score = 0.33 (only A matches)

### Chaining Reward (25%)
- **What it checks**: Whether tool outputs are properly chained (output of tool N used in tool N+1)
- **How it works**: Uses the **judge model** to evaluate chaining quality
- **Score calculation**: LLM judge returns 0.0-1.0 score based on chaining quality
- **Judge prompt**: "Check if tool calls are CHAINED (output of tool N used in tool N+1). Score 0.0-1.0"
- **Judge settings**: Temperature=0.3, max_tokens=10

### Total Reward

Final reward = `0.25 × Format + 0.25 × ToolName + 0.25 × Order + 0.25 × Chaining`

**Example reward breakdown:**
```
[FORMAT] 2/2 correct. Score: 1.00
[TOOL NAME] 2/2 match. Score: 1.00
[ORDER] 2/2 in order. Score: 1.00
[CHAINING] Score: 0.85
Total Reward: 0.25×1.00 + 0.25×1.00 + 0.25×1.00 + 0.25×0.85 = 0.9625
```

## 🔍 Tool Execution

Tools are executed via **LLM proxy** (the tool execution model):

1. Agent generates `<tool_call>` with tool name and arguments
2. Environment extracts tool call from agent's response
3. Environment sends tool name + arguments to **tool execution model**
4. Tool execution model returns the tool's result
5. Result is fed back to agent for next turn

**Tool Execution Model Settings:**
- **Model**: Configurable (default: `x-ai/grok-4.1-fast:free`)
- **Timeout**: 30 seconds per tool call
- **Max Tokens**: 100 (for concise results)
- **Temperature**: 0.0 (deterministic execution)
- **System Prompt**: "You are executing the tool '{tool_name}' with the given arguments. Return ONLY the direct result value. No explanation. Keep it short."

**Why LLM proxy?** This allows the environment to work with any tools without implementing each one separately. The LLM acts as a universal tool executor.

## 📁 Files

| File | Purpose |
|------|---------|
| `vf_nemotron_multihop.py` | Main environment implementation |
| `__init__.py` | Package initialization |
| `pyproject.toml` | Dependencies and entry points |
| `README.md` | This file |

## 🐛 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Dataset not found | Set `HF_TOKEN` environment variable and login to HuggingFace |
| Tool execution timeout | Check API key is set correctly and model is accessible |
| Judge errors | Verify judge model is accessible via your API endpoint |
| Import errors | Reinstall: `uv run --active vf-install vf-nemotron-multihop -p ./environments` |
| API key errors | Ensure environment variables are set (never hardcode keys in code) | |

### Debug Mode

Enable verbose logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 🔗 Integration

This environment is designed for the [Verifiers](https://github.com/arcee-ai/verifiers) RL training framework.

### Verifiers Resources

- **GitHub Repository**: https://github.com/arcee-ai/verifiers
- **Documentation**: https://verifiers.readthedocs.io/
- **Overview Guide**: https://github.com/arcee-ai/verifiers/blob/main/docs/source/overview.md
- **Environments Guide**: https://github.com/arcee-ai/verifiers/blob/main/docs/source/environments.md

### RLVR Training

To train models using this environment with RLVR:

```bash
# Install verifiers with RL support
uv add 'verifiers[rl]'

# Train with GRPO
uv run vf-rl --config configs/nemotron_multihop_train.toml
```

See the [Verifiers training documentation](https://verifiers.readthedocs.io/en/latest/training.html) for more details.

## 📝 Example Output

```
[TOOL 1] search_user
    -> Result (1.2s): User found: John Doe...
[TOOL 2] get_user_posts
    -> Result (0.8s): Posts retrieved: 5 posts...
[FORMAT] 2/2 correct. Score: 1.00
[TOOL NAME] 2/2 match. Score: 1.00
[ORDER] 2/2 in order. Score: 1.00
[CHAINING] Score: 0.85
============================================================
REWARDS: Format(0.25) + ToolName(0.25) + Order(0.25) + Chain(0.25)
Tool Exec: x-ai/grok-4.1-fast:free
Judge: x-ai/grok-4.1-fast:free
============================================================
```

## 🎓 Key Features

- ✅ Multi-hop tool chaining (3-6-9 hops)
- ✅ 4-component reward system with equal weights
- ✅ LLM-based tool execution
- ✅ LLM judge for chaining evaluation
- ✅ Automatic hop count detection
- ✅ Comprehensive error handling
- ✅ Timeout protection



---

**Version**: 0.1.0  
**Status**: Production Ready ✅


