#!/usr/bin/env python3
"""
Main evaluation script for vf-nemotron-multihop environment.

This script provides a comprehensive interface for evaluating models on the
nemotron multi-hop tool-use environment with various configuration options.

Usage:
    python eval.py --model "x-ai/grok-4.1-fast:free" --num-examples 50
"""
import os
import subprocess
import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any


def check_environment_installed() -> bool:
    """Check if vf-nemotron-multihop environment is installed."""
    try:
        result = subprocess.run(
            ["uv", "run", "python", "-c", "import vf_nemotron_multihop; print('OK')"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return result.returncode == 0 and "OK" in result.stdout
    except Exception:
        return False


def run_vf_eval(
    model: str,
    api_base_url: str,
    api_key_var: str,
    num_examples: int,
    rollouts_per_example: int,
    temperature: float,
    max_tokens: int,
    max_concurrent: int,
    env_args: Dict[str, Any],
    save_results: bool,
    verbose: bool,
    output_dir: Optional[str] = None,
) -> bool:
    """Run vf-eval command with specified parameters."""
    
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
        if output_dir:
            # Note: vf-eval saves to current directory by default
            # We'll handle output directory separately if needed
            pass
    
    if verbose:
        cmd.append("--verbose")
    
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
    print(f"Max Concurrent: {max_concurrent}")
    print(f"\nEnvironment Arguments:")
    for key, value in env_args.items():
        print(f"  {key}: {value}")
    print("=" * 80)
    print(f"\nCommand: {' '.join(cmd)}\n")
    
    try:
        result = subprocess.run(
            cmd,
            check=True,
            capture_output=not verbose,
            text=True,
        )
        
        if verbose:
            if result.stdout:
                print(result.stdout)
            if result.stderr:
                print("STDERR:", result.stderr, file=sys.stderr)
        
        print("\n" + "=" * 80)
        print("✓ Evaluation completed successfully!")
        print("=" * 80)
        
        if save_results:
            # Results are saved by vf-eval to current directory
            results_dir = Path(".") / "eval_results"
            if results_dir.exists():
                results_files = list(results_dir.glob("vf-nemotron-multihop_*.jsonl"))
                if results_files:
                    latest = max(results_files, key=lambda p: p.stat().st_mtime)
                    print(f"Results saved to: {latest.absolute()}")
        
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
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        return False
    except Exception as e:
        print(f"\n✗ Error running evaluation: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate models on vf-nemotron-multihop environment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick test with OpenRouter model
  python eval.py --model "x-ai/grok-4.1-fast:free" -n 5 -r 1 -v
  
  # Full evaluation
  python eval.py --model "Qwen/Qwen2.5-7B-Instruct" -n 50 -r 3 -s
  
  # Local vLLM model
  python eval.py --model "my-model" \\
      --api-base-url "http://127.0.0.1:8000/v1" \\
      --api-key-var EMPTY \\
      -n 10 -r 1
  
  # Custom tool execution and judge models
  python eval.py --model "gpt-4.1-mini" \\
      --env-exec-model "x-ai/grok-4.1-fast:free" \\
      --env-judge-model "x-ai/grok-4.1-fast:free" \\
      -n 20 -r 2
        """
    )
    
    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model identifier (e.g., 'x-ai/grok-4.1-fast:free', 'Qwen/Qwen2.5-7B-Instruct')"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default=None,
        help="Path to local model (for vLLM, use --api-base-url instead)"
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
    
    # Evaluation parameters
    parser.add_argument(
        "-n", "--num-examples",
        type=int,
        default=10,
        help="Number of examples to evaluate (default: 10)"
    )
    parser.add_argument(
        "-r", "--rollouts-per-example",
        type=int,
        default=1,
        help="Number of rollouts per example (default: 1)"
    )
    parser.add_argument(
        "-t", "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature (default: 0.0)"
    )
    parser.add_argument(
        "-T", "--max-tokens",
        type=int,
        default=2048,
        help="Maximum tokens per response (default: 2048)"
    )
    parser.add_argument(
        "-c", "--max-concurrent",
        type=int,
        default=32,
        help="Maximum concurrent requests (default: 32)"
    )
    
    # Environment configuration
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
    parser.add_argument(
        "--env-exec-api-key-var",
        type=str,
        default=None,
        help="API key var for tool execution (default: same as --api-key-var)"
    )
    parser.add_argument(
        "--env-judge-api-key-var",
        type=str,
        default=None,
        help="API key var for judge (default: same as --api-key-var)"
    )
    parser.add_argument(
        "--env-dataset-name",
        type=str,
        default=None,
        help="Dataset name (default: Anna4242/tool-n1-combined-3-6-9-hop-corrected-split)"
    )
    parser.add_argument(
        "--env-train-split",
        type=str,
        default=None,
        help="Training split name (default: train)"
    )
    parser.add_argument(
        "--env-eval-split",
        type=str,
        default=None,
        help="Evaluation split name (default: eval)"
    )
    
    # Output options
    parser.add_argument(
        "-s", "--save-results",
        action="store_true",
        help="Save evaluation results to file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Output directory for results (default: ./eval_results)"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose output"
    )
    
    # Utility options
    parser.add_argument(
        "--check-install",
        action="store_true",
        help="Check if environment is installed and exit"
    )
    
    args = parser.parse_args()
    
    # Check installation if requested
    if args.check_install:
        if check_environment_installed():
            print("✓ vf-nemotron-multihop environment is installed")
            sys.exit(0)
        else:
            print("✗ vf-nemotron-multihop environment is not installed")
            print("\nInstall with:")
            print("  uv run --active vf-install vf-nemotron-multihop -p ./environments")
            sys.exit(1)
    
    # Check if environment is installed
    if not check_environment_installed():
        print("⚠ Warning: vf-nemotron-multihop environment may not be installed")
        print("Install with: uv run --active vf-install vf-nemotron-multihop -p ./environments")
        if not args.verbose:
            response = input("Continue anyway? (y/n): ")
            if response.lower() != 'y':
                sys.exit(1)
    
    # Build environment arguments
    env_args: Dict[str, Any] = {
        "exec_model": args.env_exec_model,
        "judge_model": args.env_judge_model,
        "max_turns": args.env_max_turns,
    }
    
    # Set base URLs
    env_args["exec_base_url"] = args.env_exec_base_url or args.api_base_url
    env_args["judge_base_url"] = args.env_judge_base_url or args.api_base_url
    
    # Set API key vars
    env_args["exec_api_key_var"] = args.env_exec_api_key_var or args.api_key_var
    env_args["judge_api_key_var"] = args.env_judge_api_key_var or args.api_key_var
    
    # Set dataset options if provided
    if args.env_dataset_name:
        env_args["dataset_name"] = args.env_dataset_name
    if args.env_train_split:
        env_args["train_split"] = args.env_train_split
    if args.env_eval_split:
        env_args["eval_split"] = args.env_eval_split
    
    # Run evaluation
    success = run_vf_eval(
        model=args.model,
        api_base_url=args.api_base_url,
        api_key_var=args.api_key_var,
        num_examples=args.num_examples,
        rollouts_per_example=args.rollouts_per_example,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_concurrent=args.max_concurrent,
        env_args=env_args,
        save_results=args.save_results,
        verbose=args.verbose,
        output_dir=args.output_dir,
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()

