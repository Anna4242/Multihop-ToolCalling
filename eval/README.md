# Nemotron Multi-hop Evaluation Guide

This directory contains scripts and instructions for evaluating models on the `vf-nemotron-multihop` environment using the Verifiers framework.

## Quick Start

### Option 1: Evaluate One Model

```bash
cd nemotron_multihop/eval
python eval_one_model.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --api-base-url "https://openrouter.ai/api/v1" \
    --api-key-var OPENROUTER_API_KEY \
    --num-examples 10 \
    --rollouts-per-example 1
```

### Option 2: Use Main Eval Script

```bash
cd nemotron_multihop/eval
python eval.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --num-examples 50 \
    --rollouts-per-example 3 \
    --save-results
```

## Installation & Setup

### 1. Install the Environment

```bash
cd /workspace/anushka/verifiers  # or your verifiers directory
uv run --active vf-install vf-nemotron-multihop -p ./environments
```

### 2. Set Environment Variables

```bash
# Replace with your actual API keys (never commit these!)
export OPENROUTER_API_KEY="your-openrouter-key"
export HF_TOKEN="your-hf-token"
export HUGGINGFACE_HUB_TOKEN="$HF_TOKEN"
```

### 3. Login to HuggingFace (if using HF models)

```bash
huggingface-cli login --token $HF_TOKEN
```

**⚠️ Security**: Never commit API keys or tokens. Always use environment variables.

## Evaluation Methods

### Method 1: Using vf-eval CLI (Recommended)

```bash
uv run vf-eval vf-nemotron-multihop \
    --model "x-ai/grok-4.1-fast:free" \
    --api-base-url "https://openrouter.ai/api/v1" \
    --api-key-var OPENROUTER_API_KEY \
    --num-examples 10 \
    --rollouts-per-example 1 \
    --temperature 0.0 \
    --max-tokens 2048 \
    --env-args '{"exec_model":"x-ai/grok-4.1-fast:free", "judge_model":"x-ai/grok-4.1-fast:free", "max_turns":10}' \
    --save-results \
    --verbose
```

### Method 2: Using eval.py Script

```bash
python eval.py \
    --model "Qwen/Qwen2.5-7B-Instruct" \
    --model-path "/path/to/local/model" \
    --num-examples 50 \
    --rollouts-per-example 3 \
    --temperature 0.0 \
    --max-tokens 4096 \
    --save-results
```

### Method 3: Using eval_one_model.py (Single Model)

```bash
python eval_one_model.py \
    --model "x-ai/grok-4.1-fast:free" \
    --api-base-url "https://openrouter.ai/api/v1" \
    --api-key-var OPENROUTER_API_KEY \
    --num-examples 10 \
    --rollouts-per-example 1 \
    --output-dir eval_results
```

## Local vLLM Evaluation

For evaluating local models with vLLM:

### Step 1: Start vLLM Server

```bash
CUDA_VISIBLE_DEVICES=0 uv run vf-vllm \
    --model "/path/to/model" \
    --served-model-name my-model \
    --port 8000 \
    --trust-remote-code \
    --dtype bfloat16 \
    --max-model-len 16384
```

### Step 2: Evaluate

```bash
uv run vf-eval vf-nemotron-multihop \
    --api-base-url "http://127.0.0.1:8000/v1" \
    --api-key-var EMPTY \
    --model "my-model" \
    --num-examples 10 \
    --rollouts-per-example 1 \
    --save-results
```

## Cloning Evaluation Setup

To clone this evaluation setup to another machine:

### 1. Copy Files

```bash
# Copy the entire eval directory
cp -r nemotron_multihop/eval /path/to/destination/
```

### 2. Install Dependencies

```bash
cd /path/to/destination/eval
# Ensure verifiers is installed
uv sync --extra rl
```

### 3. Install Environment

```bash
cd /workspace/anushka/verifiers
uv run --active vf-install vf-nemotron-multihop -p ./environments
```

### 4. Set Environment Variables

```bash
# Replace with your actual keys (never commit these!)
export OPENROUTER_API_KEY="your-key"
export HF_TOKEN="your-token"
```

## Logging Evaluation Results

### Automatic Logging

Results are automatically saved when using `--save-results`:

```bash
uv run vf-eval vf-nemotron-multihop \
    --save-results \
    --num-examples 10
```

Results are saved to: `./eval_results/vf-nemotron-multihop_<timestamp>.jsonl`

### Manual Logging

```bash
# Run evaluation and save output
uv run vf-eval vf-nemotron-multihop \
    --num-examples 10 \
    --save-results \
    2>&1 | tee eval_log_$(date +%Y%m%d_%H%M%S).txt
```

### Logging to WandB

```bash
uv run vf-eval vf-nemotron-multihop \
    --num-examples 10 \
    --wandb-project "nemotron-multihop-eval" \
    --wandb-run-name "qwen25-7b-eval"
```

## Evaluation Parameters

### Environment Arguments

Pass via `--env-args` (JSON string):

```json
{
    "exec_model": "x-ai/grok-4.1-fast:free",
    "judge_model": "x-ai/grok-4.1-fast:free",
    "exec_base_url": "https://openrouter.ai/api/v1",
    "judge_base_url": "https://openrouter.ai/api/v1",
    "exec_api_key_var": "OPENROUTER_API_KEY",
    "judge_api_key_var": "OPENROUTER_API_KEY",
    "max_turns": 10,
    "dataset_name": "Anna4242/tool-n1-combined-3-6-9-hop-corrected-split",
    "train_split": "train",
    "eval_split": "eval"
}
```

### Model Arguments

- `--model`: Model identifier (e.g., "gpt-4.1-mini", "Qwen/Qwen2.5-7B-Instruct")
- `--api-base-url`: API endpoint (OpenRouter, OpenAI, or local vLLM)
- `--api-key-var`: Environment variable name for API key
- `--temperature`: Sampling temperature (default: 0.0)
- `--max-tokens`: Maximum tokens per response (default: 2048)

### Evaluation Arguments

- `--num-examples`: Number of examples to evaluate (default: 5)
- `--rollouts-per-example`: Number of rollouts per example (default: 3)
- `--max-concurrent`: Maximum concurrent requests (default: 32)
- `--save-results`: Save results to JSONL file
- `--verbose`: Enable verbose output

## Results Format

Results are saved as JSONL with the following structure:

```json
{
    "example_id": 0,
    "prompt": [...],
    "completion": [...],
    "reward": 0.75,
    "metadata": {
        "turns": 3,
        "tool_calls": 5,
        "hop_count": 6
    }
}
```

## Batch Evaluation

To evaluate multiple models:

```bash
# Create a models list
cat > models.txt << EOF
Qwen/Qwen2.5-7B-Instruct
Qwen/Qwen2.5-3B-Instruct
x-ai/grok-4.1-fast:free
EOF

# Evaluate each model
while read model; do
    echo "Evaluating: $model"
    uv run vf-eval vf-nemotron-multihop \
        --model "$model" \
        --api-base-url "https://openrouter.ai/api/v1" \
        --api-key-var OPENROUTER_API_KEY \
        --num-examples 10 \
        --save-results \
        --output-dir "eval_results/${model//\//_}"
done < models.txt
```

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| Environment not found | Run `uv run --active vf-install vf-nemotron-multihop -p ./environments` |
| API key error | Check `OPENROUTER_API_KEY` is set correctly |
| Dataset not found | Set `HF_TOKEN` and login to HuggingFace |
| Timeout errors | Increase `--max-tokens` or reduce `--num-examples` |
| Memory errors | Use smaller models or reduce batch size |

### Debug Mode

```bash
# Enable verbose logging
uv run vf-eval vf-nemotron-multihop \
    --verbose \
    --num-examples 1 \
    --rollouts-per-example 1
```

## Example Evaluation Commands

### Quick Test (1 example)

```bash
uv run vf-eval vf-nemotron-multihop \
    -n 1 -r 1 -v -s
```

### Full Evaluation (50 examples, 3 rollouts)

```bash
uv run vf-eval vf-nemotron-multihop \
    --model "x-ai/grok-4.1-fast:free" \
    --api-base-url "https://openrouter.ai/api/v1" \
    --api-key-var OPENROUTER_API_KEY \
    -n 50 -r 3 -s \
    --env-args '{"max_turns":10}'
```

### Local Model Evaluation

```bash
# Start vLLM first, then:
uv run vf-eval vf-nemotron-multihop \
    --api-base-url "http://127.0.0.1:8000/v1" \
    --api-key-var EMPTY \
    --model "my-model" \
    -n 10 -r 1 -s
```

## Files

- `eval.py` - Main evaluation script with full options
- `eval_one_model.py` - Simplified script for single model evaluation
- `README.md` - This file

## Notes

- Each evaluation can take 10-30 minutes depending on model and number of examples
- Tool execution uses LLM proxy (configurable via `exec_model`)
- Judge uses LLM for chaining reward (configurable via `judge_model`)
- Results include reward breakdown: Format(0.25) + ToolName(0.25) + Order(0.25) + Chain(0.25)

