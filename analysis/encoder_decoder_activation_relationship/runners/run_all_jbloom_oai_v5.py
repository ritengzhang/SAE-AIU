#!/usr/bin/env python3
"""
Batch runner for ALL jbloom OAI-v5 SAEs.

This script runs analysis2 on:
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

# Add parent directories to path
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

# Import analysis functions
from encoder_decoder_activation_relationship import (
    analyze_activated_features_angular_distances,
    create_activated_features_plot,
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


def collect_activations(store, n_batches: int = 15):
    """Collect activations from the activation store."""
    print(f"\n📊 Collecting {n_batches} batches of activations...")
    all_activations = []
    
    for batch_idx in range(n_batches):
        batch_activations = store.next_batch()
        all_activations.append(batch_activations.cpu().numpy())
    
    import numpy as np
    all_activations = np.concatenate(all_activations, axis=0)
    print(f"✓ Collected {all_activations.shape[0]:,} activation vectors")
    
    return all_activations


def run_analysis_jbloom_oai_v5(
    repo_id: str,
    layer: int,
    width: str,
    hook_type: str,
    device: str = None,
    n_batches: int = 15,
    n_samples: int = 5000,
    output_dir: Path = None,
    model=None  # Optional: reuse existing model
):
    """
    Run encoder-decoder-activation analysis for jbloom OAI-v5 SAE.
    
    Args:
        repo_id: HuggingFace repo ID
        layer: Layer number (0-11)
        width: Width name ('32k' or '128k')
        hook_type: Hook type ('resid_post', 'resid_mid', 'mlp_out', 'attn_out')
        device: Device to use (default: auto-detect cuda:0, requires GPU)
        n_batches: Number of batches for activation collection
        n_samples: Number of samples for analysis
        output_dir: Output directory (default: /data/user/sae/encoder_decoder_activation_relationship/)
        model: Optional pre-loaded model to reuse (saves GPU memory)
    """
    import torch
    
    # Auto-detect device if not specified - require GPU
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU is not available. GPU is required for analysis.")
        device = "cuda:0"
        print(f"Using GPU: {device}")
    
    print("="*70)
    print("ENCODER-DECODER-ACTIVATION ANALYSIS - JBLOOM OAI-V5")
    print("="*70)
    
    # Setup output directory: jbloom_oai_v5/{width}/{hook_type}/layer_{layer}/
    if output_dir is None:
        output_dir = Path(os.environ.get("DATA_DIR", "/data/user")) / "sae" / "encoder_decoder_activation_relationship"
    
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
    
    # Step 4: Run analysis
    print("\n[4/4] Running angular distance analysis...")
    
    try:
        # Get encoder and decoder vectors
        encoder_vectors = sae.W_enc.detach().cpu().numpy().T  # (d_sae, d_in)
        decoder_vectors = sae.W_dec.detach().cpu().numpy()     # (d_sae, d_in)
        
        # Get SAE architecture
        if hasattr(sae.cfg, 'architecture') and callable(getattr(sae.cfg, 'architecture')):
            sae_architecture = sae.cfg.architecture()
        else:
            sae_architecture = 'standard'
        
        # Analyze activated features
        results = analyze_activated_features_angular_distances(
            sae=sae,
            activations=activations,
            encoder_vectors=encoder_vectors,
            decoder_vectors=decoder_vectors,
            sae_architecture=sae_architecture,
            n_samples=n_samples
        )
        
        # Create plot
        metadata['experiment_folder'] = f"{width}_{hook_type}_layer_{layer}"
        metadata['description'] = f"jbloom OAI-v5: {width} {hook_type} layer {layer}"
        
        plot_path = output_dir / "activated_features_analysis.png"
        create_activated_features_plot(
            results=results,
            output_path=plot_path,
            metadata=metadata
        )
        print(f"✓ Saved plot to: {plot_path}")
        
        # Save results JSON
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results['summary'], f, indent=2)
        print(f"✓ Saved results to: {results_file}")
        
    except Exception as e:
        print(f"\n❌ Analysis failed: {e}")
        print(f"⚠️  Output directory may be incomplete: {output_dir}")
        raise
    
    # Clean up SAE cache after analysis
    print("\n[Cleanup] Removing cached SAE files...")
    cache_path_str = metadata.get('_cache_path')
    if cache_path_str:
        cleanup_sae_cache(cache_path=cache_path_str)
    else:
        cleanup_sae_cache(metadata=metadata)
    
    # Clean up GPU memory (but keep model if we loaded it here and it should be reused)
    import gc
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
    n_samples: int = 5000,
    output_dir: Path = None,
    skip_existing: bool = True
):
    """
    Run analysis2 on all jbloom OAI-v5 SAEs.
    
    Args:
        device: Device to use (default: auto-detect cuda:0)
        n_batches: Number of batches for activation collection
        n_samples: Number of samples for analysis
        output_dir: Output directory
        skip_existing: Skip analyses that already have results
    """
    config = JBLOOM_OAI_V5_CONFIG
    
    total_analyses = len(config['repos']) * len(config['layers'])
    completed = 0
    skipped = 0
    failed = []
    
    print("="*70)
    print("BATCH ANALYSIS: ALL JBLOOM OAI-V5 SAEs")
    print("="*70)
    print(f"\nTotal analyses to run: {total_analyses}")
    print(f"  - Repos: {len(config['repos'])}")
    print(f"  - Layers: {len(config['layers'])}")
    print(f"\nDevice: {device or 'auto-detect'}")
    print(f"n_batches: {n_batches}, n_samples: {n_samples}")
    
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
                    png_exists = (layer_dir / "activated_features_analysis.png").exists()
                    json_exists = (layer_dir / "results.json").exists()
                    if png_exists and json_exists:
                        print(f"  ⏭️  Layer {layer}: Already exists, skipping...")
                        skipped += 1
                        continue
                    elif png_exists or json_exists:
                        print(f"  ⚠️  Layer {layer}: Incomplete analysis found, will retry...")
            
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
                    n_samples=n_samples,
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
    
    # Check for incomplete analyses
    if output_dir:
        print("\n" + "="*70)
        print("CHECKING FOR INCOMPLETE ANALYSES")
        print("="*70)
        incomplete_count = 0
        for repo_key in config['repos'].keys():
            parts = repo_key.split('_')
            width = parts[0]
            hook_type = '_'.join(parts[1:])
            for layer in config['layers']:
                layer_dir = output_dir / "jbloom_oai_v5" / width / hook_type / f"layer_{layer}"
                if layer_dir.exists():
                    png_exists = (layer_dir / "activated_features_analysis.png").exists()
                    json_exists = (layer_dir / "results.json").exists()
                    if not (png_exists and json_exists):
                        incomplete_count += 1
                        status = []
                        if not png_exists:
                            status.append("missing PNG")
                        if not json_exists:
                            status.append("missing JSON")
                        print(f"  ⚠️  Incomplete: {width}/{hook_type}/layer_{layer} ({', '.join(status)})")
        
        if incomplete_count == 0:
            print("  ✓ All analyses are complete!")
        else:
            print(f"\n  Found {incomplete_count} incomplete analyses.")
            print("  These will be retried on the next run (use --no-skip-existing to force retry now).")
    
    return completed, skipped, len(failed)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run analysis2 on all jbloom OAI-v5 SAEs")
    parser.add_argument('--device', type=str, default=None, help='Device to use (default: auto-detect cuda:0)')
    parser.add_argument('--n-batches', type=int, default=15, help='Number of batches for activation collection (default: 15)')
    parser.add_argument('--n-samples', type=int, default=5000, help='Number of samples for analysis')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--no-skip-existing', action='store_true', help='Re-run analyses even if results exist')
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    run_all_analyses(
        device=args.device,
        n_batches=args.n_batches,
        n_samples=args.n_samples,
        output_dir=output_dir,
        skip_existing=not args.no_skip_existing
    )

