"""
Runner for GPT-2 benchmark training.

Runs all 63 combinations for GPT-2 only.
"""

import sys
from pathlib import Path

# Add parent directory to path
PARENT_DIR = Path(__file__).parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

# Add SAELens to path
SAELENS_PATH = PARENT_DIR.parent / "SAELens-main"
if str(SAELENS_PATH) not in sys.path:
    sys.path.insert(0, str(SAELENS_PATH))

import torch
from sae_lens.llm_sae_training_runner import LanguageModelSAETrainingRunner

# Override OUTPUT_BASE_DIR to use absolute path (matching OPT-125M)
import benchmark_config
import os
data_dir = os.environ.get("DATA_DIR", "/data/user")
benchmark_config.OUTPUT_BASE_DIR = f"{data_dir}/sae/training"

from benchmark_config import OUTPUT_BASE_DIR
from cache_activations import cache_activations_for_llm
from utils import (
    check_if_cached,
    create_training_config,
    get_cached_activations_path,
    is_combo_complete,
    load_combinations,
    filter_combinations_for_llm,
    save_results,
    print_summary,
)


def run_gpt2_benchmark(device: str = "cuda") -> None:
    """Run all 63 combinations for GPT-2."""
    llm_name = "gpt2"
    
    print(f"\n{'='*60}")
    print(f"Running GPT-2 Benchmark")
    print(f"{'='*60}\n")
    
    # Load combinations
    combinations = load_combinations()
    llm_combinations = filter_combinations_for_llm(combinations, llm_name)
    
    # Exclude matching_pursuit (too slow, 10-100x slower than other architectures)
    llm_combinations = [c for c in llm_combinations if c["architecture"] != "matching_pursuit"]
    
    print(f"Total combinations for GPT-2: {len(llm_combinations)}\n")
    
    # Use streaming activations (no caching to save disk space)
    print(f"Using streaming activations for {llm_name} (no caching)\n")
    
    results = {
        "completed": [],
        "failed": [],
        "skipped": [],
    }
    
    # Run all combinations
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
            continue
    
    # Save results
    save_results(llm_name, results)
    print_summary(results)
    
    print(f"\n{'='*60}")
    print(f"GPT-2 benchmark complete!")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run GPT-2 SAE benchmark")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for training (e.g., 'cuda', 'cuda:0', 'cuda:2')",
    )
    
    args = parser.parse_args()
    run_gpt2_benchmark(device=args.device)

