# VF Nemotron Multi-hop Tool-use Environment

A Reinforcement Learning from Verifier Feedback (RLVR) environment for training language models on multi-hop (3-6-9) multi-turn tool-use tasks with judge-based rewards.

## What is RLVR?

**RLVR (Reinforcement Learning from Verifier Feedback)** is a training paradigm implemented by the [Verifiers framework](https://github.com/PrimeIntellect-ai/verifiers). Instead of training on human feedback, models learn from automated verifiers that evaluate their outputs across multiple criteria.



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

##  Usage

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
- **Task Types**: Multihop tool chaining
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





