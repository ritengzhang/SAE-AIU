"""
Pre-cache activations for each LLM and layer combination.

This allows efficient training sweeps without recomputing activations.
"""

import sys
from pathlib import Path

# Add SAELens to path
SAELENS_PATH = Path(__file__).parent.parent / "SAELens-main"
if str(SAELENS_PATH) not in sys.path:
    sys.path.insert(0, str(SAELENS_PATH))

import torch
from sae_lens.config import CacheActivationsRunnerConfig
from sae_lens.cache_activations_runner import CacheActivationsRunner

from benchmark_config import LLM_CONFIGS, OUTPUT_BASE_DIR, TRAINING_CONFIG
from utils import get_cached_activations_path, check_if_cached


def cache_activations_for_llm(llm_name: str, device: str = "cuda") -> None:
    """Cache activations for a specific LLM."""
    llm_config = LLM_CONFIGS[llm_name]
    layer = llm_config["middle_layer"]
    hook_name = llm_config["hook_name_template"].format(layer=layer)
    
    # Check if already cached
    if check_if_cached(llm_name, layer):
        print(f"Activations already cached for {llm_name} layer {layer}, skipping...")
        return
    
    print(f"\n{'='*60}")
    print(f"Caching activations for {llm_name} (layer {layer})")
    print(f"{'='*60}")
    
    # Calculate training tokens
    training_tokens = TRAINING_CONFIG["total_training_steps"] * TRAINING_CONFIG["train_batch_size_tokens"]
    
    # Set up cache path
    cache_path = get_cached_activations_path(llm_name, layer)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Remove existing cache directory if it exists (to avoid "not empty" error)
    if cache_path.exists():
        import shutil
        shutil.rmtree(cache_path)
        print(f"Removed existing incomplete cache at {cache_path}")
    
    # Create cache config
    # For tokenized datasets, set context_size to -1 (auto-detect) or skip the check
    context_size = -1 if llm_config["is_dataset_tokenized"] else TRAINING_CONFIG["context_size"]
    
    cache_cfg = CacheActivationsRunnerConfig(
        model_name=llm_config["model_name"],
        model_class_name=llm_config["model_class_name"],
        hook_name=hook_name,
        d_in=llm_config["d_in"],
        dataset_path=llm_config["dataset_path"],
        streaming=llm_config["streaming"],
        context_size=context_size,
        training_tokens=training_tokens,
        model_batch_size=TRAINING_CONFIG["store_batch_size_prompts"],
        new_cached_activations_path=str(cache_path),
        device=device,
        dtype=TRAINING_CONFIG["dtype"],
        prepend_bos=True,
        autocast_lm=TRAINING_CONFIG.get("autocast", True),
        compile_llm=TRAINING_CONFIG.get("compile_llm", True),
    )
    
    # Run caching
    runner = CacheActivationsRunner(cache_cfg)
    runner.run()
    
    print(f"✓ Cached activations for {llm_name} layer {layer} to {cache_path}")


def cache_all_activations(device: str = "cuda") -> None:
    """Cache activations for all LLMs."""
    print("\n" + "="*60)
    print("Caching Activations for All LLMs")
    print("="*60)
    
    for llm_name in LLM_CONFIGS.keys():
        try:
            cache_activations_for_llm(llm_name, device=device)
        except Exception as e:
            print(f"✗ Error caching activations for {llm_name}: {e}")
            print("Continuing with next LLM...")
            continue
    
    print("\n" + "="*60)
    print("Activation caching complete!")
    print("="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Cache activations for SAE benchmark")
    parser.add_argument(
        "--llm",
        type=str,
        choices=list(LLM_CONFIGS.keys()),
        help="Cache activations for a specific LLM only",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for caching",
    )
    
    args = parser.parse_args()
    
    if args.llm:
        cache_activations_for_llm(args.llm, device=args.device)
    else:
        cache_all_activations(device=args.device)

