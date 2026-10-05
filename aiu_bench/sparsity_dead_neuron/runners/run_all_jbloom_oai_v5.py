#!/usr/bin/env python3
"""
Sparsity Analysis Runner for jbloom OAI-v5 SAEs

This script calculates sparsity metrics for each jbloom OAI-v5 SAE:
1. Average active features (simple counting)
2. Effective active features (features summing to 75% and 90% of total activation)

Results are saved in results.json with config information.
"""

import sys
import os
from pathlib import Path
import json
import time

# Set cache directories (can be overridden via CACHE_DIR env var)
cache_base = os.environ.get("CACHE_DIR", os.environ.get("DATA_DIR", "/data/user"))
cache_dir = Path(cache_base) / "cache"
hf_cache = cache_dir / "huggingface"
os.environ['HF_HOME'] = str(hf_cache)
os.environ['HF_DATASETS_CACHE'] = str(hf_cache / "datasets")
os.environ['TRANSFORMERS_CACHE'] = str(hf_cache / "transformers")
os.environ['XDG_CACHE_HOME'] = str(cache_dir)
# Ensure cache directories exist
hf_cache.mkdir(parents=True, exist_ok=True)
(hf_cache / "datasets").mkdir(parents=True, exist_ok=True)
(hf_cache / "transformers").mkdir(parents=True, exist_ok=True)

# Add parent directories to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Import analysis functions
from aiu_bench.sparsity_dead_neuron import calculate_sparsity_metrics, collect_activations

from sae_family_load_convert import (
    setup_activations_store,
    load_base_model,
)

from sae_family_load_convert.jbloom_oai_v5 import (
    load_jbloom_oai_v5_sae,
    cleanup_sae_cache,
    cleanup_repo_cache,
)

from analysis2_encoder_decoder_activation_relationship.cleanup_utils import (
    cleanup_model_cache,
)


def run_analysis_jbloom_oai_v5(
    repo_id: str,
    layer: int,
    width: str,
    hook_type: str,
    device: str = None,
    n_batches: int = 15,
    output_dir: Path = None,
    model=None  # Optional: reuse existing model
):
    """
    Run sparsity analysis for jbloom OAI-v5 SAE.
    
    Args:
        repo_id: HuggingFace repo ID
        layer: Layer number (0-11)
        width: Width name ('32k' or '128k')
        hook_type: Hook type ('resid_post', 'resid_mid', 'mlp_out', 'attn_out')
        device: Device to use (default: auto-detect cuda:0, requires GPU)
        n_batches: Number of batches for activation collection
        output_dir: Output directory (default: /data/user/sae/benchmark/sparsity_dead_neuron/)
        model: Optional pre-loaded model to reuse (saves GPU memory)
    """
    # Auto-detect device if not specified - require GPU
    if device is None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU is not available. GPU is required for analysis.")
        device = "cuda:0"
        print(f"Using GPU: {device}")
    
    print("="*70)
    print("SPARSITY ANALYSIS - JBLOOM OAI-V5")
    print("="*70)
    
    # Setup output directory: jbloom_oai_v5/{width}/{hook_type}/layer_{layer}/
    if output_dir is None:
        data_dir = os.environ.get("DATA_DIR", "/data/user")
        output_dir = Path(data_dir) / "sae" / "benchmark" / "sparsity_dead_neuron"
    
    output_dir = output_dir / "jbloom_oai_v5" / width / hook_type / f"layer_{layer}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Step 1: Load SAE
    print("\n[1/4] Loading SAE...")
    sae, metadata = load_jbloom_oai_v5_sae(
        repo_id=repo_id,
        layer=layer,
        device=device
    )
    
    # Copy metadata to output folder
    metadata_file = output_dir / "metadata.json"
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✓ Saved metadata to: {metadata_file}")
    
    # Step 2: Load base model (reuse if provided)
    print("\n[2/4] Loading base model...")
    model_loaded_here = False
    if model is None:
        # Clear GPU cache before loading to avoid OOM
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            model = load_base_model(
                model_name=metadata['model_name'],
                device=device
            )
            model_loaded_here = True
        except Exception as e:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            raise
    else:
        print("  ✓ Reusing existing model (saves GPU memory)")
    
    # Step 3: Setup activation store and collect activations
    print("\n[3/4] Collecting activations...")
    # Override dataset_path to use working tokenized dataset (openwebtext uses deprecated scripts)
    metadata_with_dataset = metadata.copy()
    metadata_with_dataset['dataset_path'] = 'NeelNanda/pile-10k'
    store = setup_activations_store(
        model=model,
        sae=sae,
        metadata=metadata_with_dataset,
        device=device,
    )
    
    activations = collect_activations(store, n_batches=n_batches)
    
    # Step 4: Calculate sparsity metrics
    print("\n[4/4] Calculating sparsity metrics...")
    
    sparsity_results = calculate_sparsity_metrics(
        sae=sae,
        activations=activations,
        n_batches=n_batches,
        activation_threshold=1e-5,
        percentiles=[75.0, 90.0]
    )
    
    # Prepare results JSON with config and sparsity metrics
    results = {
        'config': {
            'repo_id': repo_id,
            'width': width,
            'hook_type': hook_type,
            'layer': layer,
            'model_name': metadata.get('model_name', 'gpt2-small'),
            'hook_name': metadata.get('hook_name', 'unknown'),
            'd_in': sparsity_results['d_in'],
            'd_sae': sparsity_results['d_sae'],
        },
        'sparsity_metrics': {
            'average_active_features': sparsity_results['average_active_features'],
            'effective_active_features_75': sparsity_results['effective_active_features_75'],
            'effective_active_features_90': sparsity_results['effective_active_features_90'],
            'n_samples': sparsity_results['n_samples'],
        }
    }
    
    # Add dead neuron metrics if calculated
    if 'dead_neuron_metrics' in sparsity_results:
        results['dead_neuron_metrics'] = sparsity_results['dead_neuron_metrics']
    
    # Save results JSON
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✓ Saved results to: {results_file}")
    
    # Clean up SAE cache after analysis
    print("\n[Cleanup] Removing cached SAE files...")
    cache_path_str = metadata.get('_cache_path')
    if cache_path_str:
        cleanup_sae_cache(cache_path=cache_path_str)
    else:
        cleanup_sae_cache(metadata=metadata)
    
    # Aggressive GPU cleanup
    import torch
    import gc
    if torch.cuda.is_available():
        del store
        del sae
        del activations
        # Only delete model if we loaded it here (not if it was passed in for reuse)
        if model_loaded_here:
            del model
            model = None
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print(f"✓ Cleared GPU cache (device: {device})")
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")
    
    # Return model if we loaded it (for reuse)
    return model if model_loaded_here else None


# jbloom OAI-v5 Configuration
JBLOOM_OAI_V5_CONFIG = {
    'repos': {
        '32k_resid_post': 'jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs',
        '128k_resid_post': 'jbloom/GPT2-Small-OAI-v5-128k-resid-post-SAEs',
        '32k_resid_mid': 'jbloom/GPT2-Small-OAI-v5-32k-resid-mid-SAEs',
        '128k_resid_mid': 'jbloom/GPT2-Small-OAI-v5-128k-resid-mid-SAEs',
        '32k_mlp_out': 'jbloom/GPT2-Small-OAI-v5-32k-mlp-out-SAEs',
        '128k_mlp_out': 'jbloom/GPT2-Small-OAI-v5-128k-mlp-out-SAEs',
        '32k_attn_out': 'jbloom/GPT2-Small-OAI-v5-32k-attn-out-SAEs',
        '128k_attn_out': 'jbloom/GPT2-Small-OAI-v5-128k-attn-out-SAEs',
    },
    'layers': list(range(12)),  # 0-11
    'model_name': 'gpt2-small',
}


def run_all_analyses(
    device: str = None,
    n_batches: int = 15,
    output_dir: Path = None,
    skip_existing: bool = True
):
    """
    Run sparsity analysis on all jbloom OAI-v5 SAEs.
    
    Args:
        device: Device to use (default: auto-detect cuda:0)
        n_batches: Number of batches for activation collection
        output_dir: Output directory
        skip_existing: Skip analyses that already have results.json
    """
    config = JBLOOM_OAI_V5_CONFIG
    
    total_analyses = len(config['repos']) * len(config['layers'])
    completed = 0
    skipped = 0
    failed = []
    
    print("="*70)
    print("BATCH SPARSITY ANALYSIS: ALL JBLOOM OAI-V5 SAEs")
    print("="*70)
    print(f"\nTotal analyses to run: {total_analyses}")
    print(f"  - Repos: {len(config['repos'])}")
    print(f"  - Layers: {len(config['layers'])}")
    print(f"\nDevice: {device or 'auto-detect'}")
    print(f"n_batches: {n_batches}")
    
    start_time = time.time()
    
    # Load model ONCE and reuse for all 96 SAEs (all use GPT-2 Small)
    print("\n" + "="*70)
    print("LOADING BASE MODEL (will reuse for all SAEs)")
    print("="*70)
    
    import torch
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear before loading
    
    model = None
    try:
        model = load_base_model(config['model_name'], device=device)
        print(f"✓ Loaded model: {config['model_name']} (will reuse for all {total_analyses} SAEs)")
    except Exception as e:
        print(f"⚠️  Failed to load model: {e}")
        raise
    
    # Process each repo
    for repo_key, repo_id in config['repos'].items():
        # Parse width and hook_type from repo_key
        parts = repo_key.split('_')
        width = parts[0]  # '32k' or '128k'
        hook_type = '_'.join(parts[1:])  # 'resid_post', 'resid_mid', 'mlp_out', 'attn_out'
        
        print("\n" + "="*70)
        print(f"PROCESSING {width.upper()} {hook_type.upper()}")
        print("="*70)
        
        for layer in config['layers']:
            # Check if already exists
            if skip_existing and output_dir:
                layer_dir = output_dir / "jbloom_oai_v5" / width / hook_type / f"layer_{layer}"
                if layer_dir.exists():
                    if (layer_dir / "results.json").exists():
                        print(f"  ⏭️  Layer {layer}: Already exists, skipping...")
                        skipped += 1
                        continue
            
            print(f"\n  🔄 Layer {layer}...")
            try:
                # Aggressive memory cleanup before each analysis
                import gc
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
                    # Check available memory
                    if device:
                        memory_allocated = torch.cuda.memory_allocated(device)
                        memory_reserved = torch.cuda.memory_reserved(device)
                        memory_total = torch.cuda.get_device_properties(device).total_memory
                        memory_free = memory_total - memory_reserved
                        print(f"    💾 GPU Memory: {memory_free/1024**3:.2f} GB free / {memory_total/1024**3:.2f} GB total")
                
                # Reuse model for all analyses
                run_analysis_jbloom_oai_v5(
                    repo_id=repo_id,
                    layer=layer,
                    width=width,
                    hook_type=hook_type,
                    device=device,
                    n_batches=n_batches,
                    output_dir=output_dir,
                    model=model  # Reuse model to save GPU memory
                )
                completed += 1
                print(f"  ✅ Layer {layer}: Completed ({completed}/{total_analyses})")
                
                # Aggressive cleanup after each successful analysis
                import gc
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
            except Exception as e:
                error_msg = str(e)[:200]
                print(f"  ❌ Layer {layer}: Failed - {error_msg}")
                failed.append({
                    'repo_key': repo_key,
                    'layer': layer,
                    'error': error_msg
                })
                
                # Aggressive cleanup after failure too
                import gc
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        
        # Clean up repo cache after all layers from this repo are processed
        print(f"\n🧹 Cleaning up repo cache for {width} {hook_type}...")
        cleanup_repo_cache(repo_id)
        print(f"✓ Cleaned up repo cache (including blobs) for {repo_key}")
    
    # Clean up model after all analyses
    if model is not None:
        print(f"\n🧹 Cleaning up model...")
        import gc
        del model
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        cleanup_model_cache(config['model_name'])
        print(f"✓ Cleaned up model and GPU cache")
    
    elapsed_time = time.time() - start_time
    
    print("\n" + "="*70)
    print("BATCH ANALYSIS COMPLETE")
    print("="*70)
    print(f"\nTotal analyses: {total_analyses}")
    print(f"✅ Completed: {completed}")
    print(f"⏭️  Skipped: {skipped}")
    print(f"❌ Failed: {len(failed)}")
    print(f"⏱️  Time elapsed: {elapsed_time/3600:.2f} hours ({elapsed_time/60:.1f} minutes)")
    
    if failed:
        print("\nFailed analyses:")
        for f in failed:
            print(f"  - {f['repo_key']} layer {f['layer']}: {f['error']}")
    
    return completed, skipped, len(failed)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Run sparsity analysis on all jbloom OAI-v5 SAEs"
    )
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (default: auto-detect cuda:0)')
    parser.add_argument('--n-batches', type=int, default=15,
                       help='Number of batches for activation collection (default: 15)')
    parser.add_argument('--output-dir', type=str, default=None,
                       help='Output directory (default: /data/user/sae/benchmark/sparsity_dead_neuron)')
    parser.add_argument('--no-skip-existing', action='store_true',
                       help='Do not skip existing results')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    run_all_analyses(
        device=args.device,
        n_batches=args.n_batches,
        output_dir=output_dir,
        skip_existing=not args.no_skip_existing
    )

