"""
Main grid search runner for SAE benchmark training.

Processes LLM-by-LLM: for each LLM, caches activations then runs all 63 combinations.
"""

import json
import sys
from pathlib import Path
from typing import Any

# Add SAELens to path
SAELENS_PATH = Path(__file__).parent.parent / "SAELens-main"
if str(SAELENS_PATH) not in sys.path:
    sys.path.insert(0, str(SAELENS_PATH))

import torch
from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner

from benchmark_config import LLM_CONFIGS, OUTPUT_BASE_DIR
from utils import (
    create_training_config,
    is_combo_complete,
)


def load_combinations() -> list[dict[str, Any]]:
    """Load all combinations from JSON file."""
    combinations_path = Path(OUTPUT_BASE_DIR) / "combinations.json"
    if not combinations_path.exists():
        raise FileNotFoundError(
            f"Combinations file not found: {combinations_path}. "
            "Run generate_combinations.py first."
        )
    
    with open(combinations_path, "r") as f:
        return json.load(f)


def filter_combinations_for_llm(
    combinations: list[dict[str, Any]], llm_name: str
) -> list[dict[str, Any]]:
    """Filter combinations for a specific LLM."""
    return [c for c in combinations if c["llm"] == llm_name]


def run_benchmark_for_llm(
    llm_name: str, combinations: list[dict[str, Any]], device: str = "cuda"
) -> dict[str, Any]:
    """
    Run all combinations for a specific LLM.
    
    Returns:
        Dictionary with results: {"completed": [...], "failed": [...], "skipped": [...]}
    """
    llm_combinations = filter_combinations_for_llm(combinations, llm_name)
    
    # Exclude matching_pursuit (too slow, 10-100x slower than other architectures)
    llm_combinations = [c for c in llm_combinations if c["architecture"] != "matching_pursuit"]
    
    print(f"\n{'='*60}")
    print(f"Running benchmark for {llm_name}")
    print(f"Total combinations: {len(llm_combinations)}")
    print(f"{'='*60}\n")
    
    # Use streaming activations (no caching to save disk space)
    print(f"Using streaming activations for {llm_name} (no caching)\n")
    
    results = {
        "completed": [],
        "failed": [],
        "skipped": [],
    }
    
    for i, combo in enumerate(llm_combinations, 1):
        combo_id = (
            f"{combo['architecture']}_L{combo['layer']}_"
            f"{combo['width']}_{combo['sparsity_param']}{combo['sparsity_value']}"
        )
        
        # Check if already completed
        if is_combo_complete(llm_name, combo_id):
            print(f"\n[{i}/{len(llm_combinations)}] Skipping (already completed): {combo_id}")
            results["skipped"].append(combo_id)
            continue
        
        print(f"\n[{i}/{len(llm_combinations)}] Running: {combo_id}")
        print(f"  Architecture: {combo['architecture']}")
        print(f"  Width: {combo['width']} (d_sae={combo['d_sae']})")
        print(f"  Sparsity: {combo['sparsity_param']}={combo['sparsity_value']}")
        
        try:
            # Create training config
            config = create_training_config(combo, device=device)
            
            # Use streaming activations (no caching)
            config.use_cached_activations = False
            config.cached_activations_path = None
            
            # Run training
            runner = LanguageModelSAETrainingRunner(config)
            sae = runner.run()
            
            print(f"  ✓ Completed: {combo_id}")
            results["completed"].append(combo_id)
            
        except KeyboardInterrupt:
            print("\n\nInterrupted by user. Exiting...")
            raise
        except Exception as e:
            print(f"  ✗ Failed: {combo_id}")
            print(f"    Error: {e}")
            results["failed"].append({"combo": combo_id, "error": str(e)})
            # Continue with next combination
            continue
    
    return results


def save_results(llm_name: str, results: dict[str, Any]) -> None:
    """Save results to JSON file."""
    results_path = Path(OUTPUT_BASE_DIR) / llm_name / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ Saved results to {results_path}")


def print_summary(results: dict[str, Any]) -> None:
    """Print summary of results."""
    print(f"\n{'='*60}")
    print("Results Summary")
    print(f"{'='*60}")
    print(f"Completed: {len(results['completed'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Skipped: {len(results['skipped'])}")
    print(f"{'='*60}\n")
    
    if results["failed"]:
        print("Failed combinations:")
        for failure in results["failed"]:
            print(f"  - {failure['combo']}: {failure['error']}")


def run_all_benchmarks(device: str = "cuda", start_from_llm: str | None = None) -> None:
    """
    Run benchmarks for all LLMs, processing one LLM completely before moving to next.
    
    Args:
        device: Device to use for training
        start_from_llm: If specified, start from this LLM (useful for resuming)
    """
    # Load all combinations
    combinations = load_combinations()
    
    # Get list of LLMs
    llm_names = list(LLM_CONFIGS.keys())
    
    # Find starting index if resuming
    start_idx = 0
    if start_from_llm:
        try:
            start_idx = llm_names.index(start_from_llm)
            print(f"Resuming from LLM: {start_from_llm}")
        except ValueError:
            print(f"Warning: LLM '{start_from_llm}' not found, starting from beginning")
    
    # Process each LLM
    for llm_name in llm_names[start_idx:]:
        try:
            results = run_benchmark_for_llm(llm_name, combinations, device=device)
            save_results(llm_name, results)
            print_summary(results)
        except KeyboardInterrupt:
            print(f"\n\nInterrupted while processing {llm_name}.")
            print(f"To resume, run with --start-from-llm {llm_name}")
            raise
        except Exception as e:
            print(f"\n✗ Error processing {llm_name}: {e}")
            print("Continuing with next LLM...")
            continue
    
    print("\n" + "="*60)
    print("All benchmarks complete!")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run SAE benchmark training")
    parser.add_argument(
        "--llm",
        type=str,
        choices=list(LLM_CONFIGS.keys()),
        help="Run benchmark for a specific LLM only",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training",
    )
    parser.add_argument(
        "--start-from-llm",
        type=str,
        choices=list(LLM_CONFIGS.keys()),
        help="Resume from a specific LLM (useful if interrupted)",
    )
    
    args = parser.parse_args()
    
    if args.llm:
        # Run for single LLM
        combinations = load_combinations()
        results = run_benchmark_for_llm(args.llm, combinations, device=args.device)
        save_results(args.llm, results)
        print_summary(results)
    else:
        # Run for all LLMs
        run_all_benchmarks(device=args.device, start_from_llm=args.start_from_llm)

