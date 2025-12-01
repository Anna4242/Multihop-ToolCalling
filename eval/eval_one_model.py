#!/usr/bin/env python3
"""
Evaluate a single model on vf-nemotron-multihop environment.

Usage:
    python eval_one_model.py --model "x-ai/grok-4.1-fast:free" --num-examples 10

For local models with vLLM:
    python eval_one_model.py --model "my-model" --api-base-url "http://127.0.0.1:8000/v1" --api-key-var EMPTY
"""
import os
import subprocess
import argparse
import json
from pathlib import Path
from datetime import datetime


def run_evaluation(
    model: str,
    api_base_url: str = "https://openrouter.ai/api/v1",
    api_key_var: str = "OPENROUTER_API_KEY",
    num_examples: int = 10,
    rollouts_per_example: int = 1,
    temperature: float = 0.0,
    max_tokens: int = 2048,
    max_concurrent: int = 32,
    output_dir: str = "eval_results",
    env_args: dict = None,
    save_results: bool = True,
    verbose: bool = False,
):
    """Run evaluation on vf-nemotron-multihop environment."""
    
    # Default environment arguments
    if env_args is None:
        env_args = {
            "exec_model": "x-ai/grok-4.1-fast:free",
            "judge_model": "x-ai/grok-4.1-fast:free",
            "max_turns": 10,
        }
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True, parents=True)
    
    # Build vf-eval command
    cmd = [
        "uv", "run", "vf-eval", "vf-nemotron-multihop",
        "--model", model,
        "--api-base-url", api_base_url,
        "--api-key-var", api_key_var,
        "--num-examples", str(num_examples),
        "--rollouts-per-example", str(rollouts_per_example),
        "--temperature", str(temperature),
        "--max-tokens", str(max_tokens),
        "--max-concurrent", str(max_concurrent),
        "--env-args", json.dumps(env_args),
    ]
    
    if save_results:
        cmd.append("--save-results")
    
    if verbose:
        cmd.append("--verbose")
    
    # Print evaluation info
    print("=" * 80)
    print("Nemotron Multi-hop Evaluation")
    print("=" * 80)
    print(f"Model: {model}")
    print(f"API Base URL: {api_base_url}")
    print(f"API Key Var: {api_key_var}")
    print(f"Examples: {num_examples}")
    print(f"Rollouts per example: {rollouts_per_example}")
    print(f"Temperature: {temperature}")
    print(f"Max Tokens: {max_tokens}")
    print(f"Environment Args: {json.dumps(env_args, indent=2)}")
    print("=" * 80)
    print(f"\nRunning: {' '.join(cmd)}\n")
    
    # Run evaluation
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=not verbose,
            text=True,
        )
        
        if verbose:
            print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr)
        
        print("\n" + "=" * 80)
        print("✓ Evaluation completed successfully!")
        print("=" * 80)
        
        if save_results:
            # Find the most recent results file
            results_files = list(output_path.glob("vf-nemotron-multihop_*.jsonl"))
            if results_files:
                latest = max(results_files, key=lambda p: p.stat().st_mtime)
                print(f"Results saved to: {latest}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        print("\n" + "=" * 80)
        print("✗ Evaluation failed")
        print("=" * 80)
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False
    except Exception as e:
        print(f"\n✗ Error running evaluation: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate a single model on vf-nemotron-multihop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate with OpenRouter model
  python eval_one_model.py --model "x-ai/grok-4.1-fast:free" --num-examples 10
  
  # Evaluate with local vLLM model
  python eval_one_model.py --model "my-model" \\
      --api-base-url "http://127.0.0.1:8000/v1" \\
      --api-key-var EMPTY
  
  # Custom environment arguments
  python eval_one_model.py --model "gpt-4.1-mini" \\
      --env-exec-model "x-ai/grok-4.1-fast:free" \\
      --env-judge-model "x-ai/grok-4.1-fast:free" \\
      --env-max-turns 15
        """
    )
    
    # Model arguments
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier (e.g., 'x-ai/grok-4.1-fast:free', 'Qwen/Qwen2.5-7B-Instruct')"
    )
    parser.add_argument(
        "--api-base-url",
        type=str,
        default="https://openrouter.ai/api/v1",
        help="API base URL (default: https://openrouter.ai/api/v1)"
    )
    parser.add_argument(
        "--api-key-var",
        type=str,
        default="OPENROUTER_API_KEY",
        help="Environment variable name for API key (default: OPENROUTER_API_KEY)"
    )
    
    # Evaluation arguments
    parser.add_argument(
        "--num-examples",
        type=int,
        default=10,
        help="Number of examples to evaluate (default: 10)"
    )
    parser.add_argument(
        "--rollouts-per-example",
        type=int,
        default=1,
        help="Number of rollouts per example (default: 1)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)"
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens per response (default: 2048)"
    )
    parser.add_argument(
        "--max-concurrent",
        type=int,
        default=32,
        help="Maximum concurrent requests (default: 32)"
    )
    
    # Environment arguments
    parser.add_argument(
        "--env-exec-model",
        type=str,
        default="x-ai/grok-4.1-fast:free",
        help="Model for tool execution (default: x-ai/grok-4.1-fast:free)"
    )
    parser.add_argument(
        "--env-judge-model",
        type=str,
        default="x-ai/grok-4.1-fast:free",
        help="Model for judging/chaining reward (default: x-ai/grok-4.1-fast:free)"
    )
    parser.add_argument(
        "--env-max-turns",
        type=int,
        default=10,
        help="Maximum conversation turns (default: 10)"
    )
    parser.add_argument(
        "--env-exec-base-url",
        type=str,
        default=None,
        help="Base URL for tool execution (default: same as --api-base-url)"
    )
    parser.add_argument(
        "--env-judge-base-url",
        type=str,
        default=None,
        help="Base URL for judge (default: same as --api-base-url)"
    )
    
    # Output arguments
    parser.add_argument(
        "--output-dir",
        type=str,
        default="eval_results",
        help="Output directory for results (default: eval_results)"
    )
    parser.add_argument(
        "--no-save-results",
        action="store_true",
        help="Don't save results to file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    args = parser.parse_args()
    
    # Build environment arguments
    env_args = {
        "exec_model": args.env_exec_model,
        "judge_model": args.env_judge_model,
        "max_turns": args.env_max_turns,
    }
    
    if args.env_exec_base_url:
        env_args["exec_base_url"] = args.env_exec_base_url
    else:
        env_args["exec_base_url"] = args.api_base_url
    
    if args.env_judge_base_url:
        env_args["judge_base_url"] = args.env_judge_base_url
    else:
        env_args["judge_base_url"] = args.api_base_url
    
    # Use same API key var for both by default
    env_args["exec_api_key_var"] = args.api_key_var
    env_args["judge_api_key_var"] = args.api_key_var
    
    # Run evaluation
    success = run_evaluation(
        model=args.model,
        api_base_url=args.api_base_url,
        api_key_var=args.api_key_var,
        num_examples=args.num_examples,
        rollouts_per_example=args.rollouts_per_example,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_concurrent=args.max_concurrent,
        output_dir=args.output_dir,
        env_args=env_args,
        save_results=not args.no_save_results,
        verbose=args.verbose,
    )
    
    exit(0 if success else 1)


if __name__ == "__main__":
    main()

