#!/usr/bin/env python3
"""
Batch runner for ALL jbloom OAI-v5 SAEs.

This script runs analysis5 (mean direction divergence) on:
- All 8 repos (32k/128k × 4 hook types)
- All 12 layers (0-11)
Total: 8 × 12 = 96 analyses

All SAEs use the same base model (GPT-2 Small), so we load it once and reuse.
"""

import sys
import os
from pathlib import Path
import time
import json
import shutil
import numpy as np

# Add parent directories to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Import analysis functions
from mean_direction_divergence import (
    compute_mean_direction_from_activations,
    compute_angular_divergences,
    create_divergence_plots,
)
from mean_direction_divergence.mean_direction_analysis import (
    compute_angular_distances_from_mean,
)

from sae_family_load_convert.jbloom_oai_v5 import (
    load_jbloom_oai_v5_sae,
    cleanup_sae_cache,
    cleanup_repo_cache,
)

from sae_family_load_convert import (
    setup_activations_store,
    load_base_model,
)

from encoder_decoder_activation_relationship.cleanup_utils import (
    cleanup_model_cache,
)


def collect_original_activations(store, n_batches: int = 15):
    """
    Collect original input activations (before SAE encoding).
    
    Args:
        store: ActivationsStore for collecting activations
        n_batches: Number of batches to collect
    
    Returns:
        all_activations: Original activations (n_samples, d_in)
    """
    print(f"\n📊 Collecting {n_batches} batches of original activations...")
    all_activations = []
    
    for batch_idx in range(n_batches):
        batch_activations = store.next_batch()  # [batch_size, d_in] - original activations
        all_activations.append(batch_activations.cpu().numpy())
        
        # Periodic memory cleanup
        if (batch_idx + 1) % 10 == 0:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    all_activations = np.concatenate(all_activations, axis=0)
    print(f"✓ Collected {all_activations.shape[0]:,} activation vectors")
    
    return all_activations


def run_analysis_jbloom_oai_v5(
    repo_id: str,
    layer: int,
    device: str = None,
    n_batches: int = 15,
    output_dir: Path = None,
    model=None  # Optional: reuse existing model
):
    """
    Run mean direction divergence analysis for jbloom OAI-v5 SAE.
    
    Args:
        repo_id: HuggingFace repo ID
        layer: Layer number (0-11)
        device: Device to use (default: auto-detect cuda:0, requires GPU)
        n_batches: Number of batches for activation collection
        output_dir: Output directory (default: /data/user/sae/mean_direction_divergence/)
        model: Optional pre-loaded model to reuse (saves GPU memory)
    """
    import torch
    import gc
    
    # Auto-detect device if not specified - require GPU
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU is not available. GPU is required for analysis.")
        device = "cuda:0"
        print(f"Using GPU: {device}")
    
    print("="*70)
    print("MEAN DIRECTION DIVERGENCE ANALYSIS - JBLOOM OAI-V5")
    print("="*70)
    
    # Setup output directory: jbloom_oai_v5/{repo_name}/layer_{layer}/
    if output_dir is None:
        output_dir = Path(os.environ.get("DATA_DIR", "/data/user")) / "sae" / "mean_direction_divergence"
    
    # Parse repo_id to get repo name
    repo_name = repo_id.split('/')[-1]  # e.g., "GPT2-Small-OAI-v5-32k-resid-post-SAEs"
    
    # Create output directory
    output_dir_final = output_dir / "jbloom_oai_v5" / repo_name / f"layer_{layer}"
    output_dir_final.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir_final}")
    
    # Step 1: Load SAE
    print("\n[1/4] Loading SAE...")
    sae, metadata = load_jbloom_oai_v5_sae(
        repo_id=repo_id,
        layer=layer,
        device=device
    )
    
    # Step 2: Load base model (reuse if provided)
    print("\n[2/4] Loading base model...")
    model_loaded_here = False
    if model is None:
        # Clear GPU cache before loading to avoid OOM
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
    
    # Step 3: Setup activation store and collect original activations
    print("\n[3/4] Collecting original activations...")
    store = setup_activations_store(
        model=model,
        sae=sae,
        metadata=metadata,
        device=device
    )
    
    original_activations = collect_original_activations(store, n_batches=n_batches)
    
    # Step 4: Run mean direction divergence analysis
    print("\n[4/4] Running mean direction divergence analysis...")
    
    try:
        # Compute mean direction from original activations
        mean_direction, mean_resultant_length = compute_mean_direction_from_activations(original_activations)
        
        # Get encoder and decoder vectors
        encoder_vectors = sae.W_enc.detach().cpu().numpy().T  # (d_sae, d_in)
        decoder_vectors = sae.W_dec.detach().cpu().numpy()     # (d_sae, d_in)
        
        # Compute angular divergences
        results, encoder_distances, decoder_distances = compute_angular_divergences(
            encoder_weights=encoder_vectors,
            decoder_weights=decoder_vectors,
            mean_direction=mean_direction
        )
        
        # Compute activation distances for comparison
        activation_distances = compute_angular_distances_from_mean(original_activations, mean_direction)
        
        # Create output subdirectory
        analysis_output_dir = output_dir_final / "mean_direction_divergence"
        
        # Create plots
        metadata['experiment_folder'] = f"{repo_name}_layer_{layer}"
        metadata['description'] = f"jbloom_oai_v5: {repo_name}"
        
        create_divergence_plots(
            encoder_distances=encoder_distances,
            decoder_distances=decoder_distances,
            activation_distances=activation_distances,
            results=results,
            output_dir=analysis_output_dir,
            metadata=metadata
        )
        
        # Save results JSON
        results_file = analysis_output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump({
                'mean_resultant_length': float(mean_resultant_length),
                'divergence_statistics': results,
            }, f, indent=2)
        print(f"✓ Saved results to: {results_file}")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        print(f"⚠️  Output directory may be incomplete: {output_dir_final}")
        raise
    
    # Clean up SAE cache after analysis
    print("\n[Cleanup] Removing cached SAE files...")
    cache_path_str = metadata.get('_cache_path')
    if cache_path_str:
        cleanup_sae_cache(cache_path=cache_path_str)
    else:
        cleanup_sae_cache(metadata=metadata)
    
    # Clean up GPU memory (but keep model if we loaded it here and it should be reused)
    if torch.cuda.is_available():
        del store
        del sae
        # Only delete model if we loaded it here (not if it was passed in for reuse)
        if model_loaded_here:
            del model
            model = None
        gc.collect()  # Explicit garbage collection
        torch.cuda.empty_cache()
        torch.cuda.synchronize()  # Wait for all operations to complete
        print(f"✓ Cleared GPU cache (device: {device})")
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir_final}")
    
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
    Run analysis5 on all jbloom OAI-v5 SAEs.
    
    Args:
        device: Device to use (default: auto-detect cuda:0)
        n_batches: Number of batches for activation collection
        output_dir: Output directory
        skip_existing: Skip analyses that already have results
    """
    import torch
    import gc
    
    total_analyses = len(JBLOOM_OAI_V5_CONFIG['repos']) * len(JBLOOM_OAI_V5_CONFIG['layers'])
    completed = 0
    skipped = 0
    failed = []
    
    print("="*70)
    print("BATCH ANALYSIS: ALL JBLOOM OAI-V5 SAEs (MEAN DIRECTION DIVERGENCE)")
    print("="*70)
    print(f"\nTotal analyses to run: {total_analyses}")
    print(f"  - Repos: {len(JBLOOM_OAI_V5_CONFIG['repos'])}")
    print(f"  - Layers: {len(JBLOOM_OAI_V5_CONFIG['layers'])}")
    print(f"\nDevice: {device or 'auto-detect'}")
    print(f"n_batches: {n_batches}")
    
    start_time = time.time()
    
    # Load model once and reuse (all SAEs use same base model: GPT-2 Small)
    print("\n📦 Loading base model (will reuse for all SAEs)...")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()  # Clear before loading
    
    try:
        model = load_base_model(
            model_name=JBLOOM_OAI_V5_CONFIG['model_name'],
            device=device
        )
        print(f"✓ Loaded model: {JBLOOM_OAI_V5_CONFIG['model_name']} (will reuse for all {total_analyses} SAEs)")
    except Exception as e:
        print(f"⚠️  Failed to load model: {e}")
        model = None
    
    for repo_name, repo_id in JBLOOM_OAI_V5_CONFIG['repos'].items():
        print("\n" + "="*70)
        print(f"PROCESSING {repo_name.upper()}")
        print("="*70)
        
        for layer in JBLOOM_OAI_V5_CONFIG['layers']:
            # Check if already exists
            if skip_existing and output_dir:
                layer_dir = output_dir / "jbloom_oai_v5" / repo_id.split('/')[-1] / f"layer_{layer}"
                png_exists = (layer_dir / "mean_direction_divergence" / "1_encoder_decoder_divergence_comparison.png").exists()
                json_exists = (layer_dir / "mean_direction_divergence" / "results.json").exists()
                if png_exists and json_exists:
                    print(f"  ⏭️  Layer {layer}: Already exists, skipping...")
                    skipped += 1
                    continue
            
            print(f"\n  🔄 Layer {layer}...")
            try:
                # Aggressive memory cleanup before each analysis
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                
                # Reuse model for all SAEs
                run_analysis_jbloom_oai_v5(
                    repo_id=repo_id,
                    layer=layer,
                    device=device,
                    n_batches=n_batches,
                    output_dir=output_dir,
                    model=model  # Reuse model to save GPU memory
                )
                completed += 1
                print(f"  ✅ Layer {layer}: Completed ({completed}/{total_analyses})")
                
                # Aggressive cleanup after each successful analysis
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
            except Exception as e:
                error_msg = str(e)[:200]
                print(f"  ❌ Layer {layer}: Failed - {error_msg}")
                failed.append({
                    'repo': repo_name,
                    'layer': layer,
                    'error': error_msg
                })
                # Aggressive cleanup after failure too
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
        
        # Clean up repo cache after all layers processed
        print(f"\n🧹 Cleaning up repo cache for {repo_name}...")
        cleanup_repo_cache(repo_id)
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        print(f"✓ Cleaned up repo cache (including blobs) for {repo_name}")
    
    # Clean up model cache after all repos processed
    if model is not None:
        print("\n🧹 Cleaning up model cache...")
        del model
        model = None
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        cleanup_model_cache(JBLOOM_OAI_V5_CONFIG['model_name'])
        print(f"✓ Cleaned up model cache and GPU cache")
    
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
            print(f"  - {f['repo']}/layer_{f['layer']}: {f['error']}")
    
    return completed, skipped, len(failed)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run analysis5 on all jbloom OAI-v5 SAEs")
    parser.add_argument('--device', type=str, default=None, help='Device to use (default: auto-detect cuda:0)')
    parser.add_argument('--n-batches', type=int, default=15, help='Number of batches for activation collection (default: 15)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--no-skip-existing', action='store_true', help='Re-run analyses even if results exist')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    run_all_analyses(
        device=args.device,
        n_batches=args.n_batches,
        output_dir=output_dir,
        skip_existing=not args.no_skip_existing
    )

