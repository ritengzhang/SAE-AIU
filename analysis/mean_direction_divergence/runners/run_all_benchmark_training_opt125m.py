#!/usr/bin/env python3
"""
Batch runner for ALL Benchmark Training OPT-125m models for Mean Direction Divergence Analysis.

This script runs analysis5 (mean direction divergence) on all 63 OPT-125m combinations.

Usage:
    python -m mean_direction_divergence.runners.run_all_benchmark_training_opt-125m --device cuda:1 --output-dir /data/user/sae/mean_direction_divergence
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
from mean_direction_divergence.mean_direction_analysis import (
    compute_mean_direction_from_activations,
    compute_angular_divergences,
    compute_angular_distances_from_mean,
    create_divergence_plots,
)

# Import benchmark training utilities
from benchmark_training.load_benchmark_sae import (
    load_benchmark_sae,
    load_base_model_for_llm,
    parse_combo_id,
)
from benchmark_training.utils import load_combinations, filter_combinations_for_llm

# Import SAELens utilities
SAELENS_PATH = parent_dir / "SAELens-main"
if str(SAELENS_PATH) not in sys.path:
    sys.path.insert(0, str(SAELENS_PATH))

from sae_lens.training.activations_store import ActivationsStore
from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.saes.standard_sae import StandardTrainingSAEConfig

from encoder_decoder_activation_relationship.cleanup_utils import (
    cleanup_model_cache,
)


def setup_activations_store_for_benchmark(model, sae, metadata: dict, device: str = None, width_name: str = None) -> ActivationsStore:
    """Setup activations store for benchmark training SAEs."""
    if device is None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU is not available. GPU is required.")
        device = "cuda:0"
    
    print("Setting up activations store...")
    
    dummy_sae_cfg = StandardTrainingSAEConfig(
        d_in=sae.cfg.d_in,
        d_sae=sae.cfg.d_sae,
        device=device,
        dtype="float32",
    )
    
    # Adaptive buffer sizing
    if width_name == '16k':
        n_batches_in_buffer = 8
        store_batch_size_prompts = 4
    elif width_name == '65k':
        n_batches_in_buffer = 8
        store_batch_size_prompts = 4
    else:
        n_batches_in_buffer = 16
        store_batch_size_prompts = 8
    
    cfg = LanguageModelSAERunnerConfig(
        sae=dummy_sae_cfg,
        model_name=metadata.get('model_name', 'opt-125m'),
        hook_name=metadata.get('hook_name', 'blocks.6.hook_mlp_out'),
        dataset_path=metadata.get('dataset_path', 'apollo-research/Skylion007-openwebtext-tokenizer-opt-125m'),
        is_dataset_tokenized=True,
        streaming=True,
        train_batch_size_tokens=4096,
        context_size=512,
        n_batches_in_buffer=n_batches_in_buffer,
        training_tokens=4096 * 100,
        store_batch_size_prompts=store_batch_size_prompts,
        device=device,
        seed=42,
        dtype="float32",
    )
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            store = ActivationsStore.from_config(model, cfg)
            print(f"✓ Activations store ready")
            return store
        except Exception as e:
            if attempt < max_retries - 1:
                print(f"  ⚠️  Attempt {attempt + 1} failed: {str(e)[:100]}...")
                time.sleep(5)
            else:
                raise


def collect_original_activations(store, n_batches: int = 30):
    """Collect original input activations (before SAE encoding)."""
    print(f"\n📊 Collecting {n_batches} batches of original activations...")
    all_activations = []
    
    for batch_idx in range(n_batches):
        batch_activations = store.next_batch()  # [batch_size, d_in] - original activations
        all_activations.append(batch_activations.cpu().numpy())
        
        if (batch_idx + 1) % 10 == 0:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    all_activations = np.concatenate(all_activations, axis=0)
    print(f"✓ Collected {all_activations.shape[0]:,} activation vectors")
    
    return all_activations


def run_analysis_benchmark_training_opt125m(
    combo: dict,
    device: str = None,
    n_batches: int = 30,
    output_dir: Path = None,
    model=None
):
    """Run mean direction divergence analysis for a benchmark training OPT-125m SAE."""
    import torch
    import gc
    
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU is not available. GPU is required.")
        device = "cuda:0"
        print(f"Using GPU: {device}")
    
    print("="*70)
    print("MEAN DIRECTION DIVERGENCE ANALYSIS - BENCHMARK TRAINING OPT-125m")
    print("="*70)
    
    if output_dir is None:
        output_dir = Path(os.environ.get("DATA_DIR", "/data/user")) / "sae" / "mean_direction_divergence"
    
    combo_id = (
        f"{combo['architecture']}_L{combo['layer']}_"
        f"{combo['width']}_{combo['sparsity_param']}{combo['sparsity_value']}"
    )
    
    parsed = parse_combo_id(combo_id)
    width_name = parsed['width']
    architecture = parsed['architecture']
    
    base_output_dir = output_dir / "benchmark_training_opt-125m" / width_name / architecture
    base_output_dir.mkdir(parents=True, exist_ok=True)
    
    output_dir_final = base_output_dir / combo_id
    output_dir_final.mkdir(parents=True, exist_ok=True)
    
    results_file = output_dir_final / "results.json"
    if results_file.exists():
        print(f"\n⏭️  Skipping (already exists): {combo_id}")
        return model
    
    # Step 1: Load SAE
    print(f"\n[1/4] Loading SAE: {combo_id}...")
    sae, metadata = load_benchmark_sae(combo_id=combo_id, llm_name="opt-125m", device=device)
    
    # Get encoder and decoder weights
    if hasattr(sae, 'W_enc'):
        encoder_weights = sae.W_enc.detach().cpu().numpy().T  # [d_sae, d_in]
    else:
        raise ValueError("Could not find encoder weights")
    
    if hasattr(sae, 'W_dec'):
        decoder_weights = sae.W_dec.detach().cpu().numpy()  # [d_sae, d_in]
    else:
        raise ValueError("Could not find decoder weights")
    
    # Copy config.json
    sae_path = Path(metadata['sae_path'])
    config_json_path = sae_path / "cfg.json"
    if config_json_path.exists():
        output_config_path = output_dir_final / "config.json"
        shutil.copy2(config_json_path, output_config_path)
        print(f"✓ Copied config.json")
    
    # Step 2: Load base model (or reuse)
    print("\n[2/4] Loading base model...")
    if model is None:
        model = load_base_model_for_llm("opt-125m", device=device)
    else:
        print("  Reusing existing model")
    
    # Step 3: Collect original activations
    print("\n[3/4] Collecting original activations...")
    store = setup_activations_store_for_benchmark(
        model=model, sae=sae, metadata=metadata, device=device, width_name=width_name
    )
    
    activations = collect_original_activations(store, n_batches=n_batches)
    
    # Step 4: Compute mean direction and divergences
    print("\n[4/4] Computing mean direction and angular divergences...")
    
    # Compute mean direction from activations
    mean_direction, mean_resultant_length = compute_mean_direction_from_activations(activations)
    
    # Compute angular divergences
    divergences, encoder_distances, decoder_distances = compute_angular_divergences(
        encoder_weights=encoder_weights,
        decoder_weights=decoder_weights,
        mean_direction=mean_direction
    )
    
    # Compute activation distances for comparison
    activation_distances = compute_angular_distances_from_mean(activations, mean_direction)
    
    # Create plots
    print("\n[Plots] Creating divergence plots...")
    metadata_plot = {
        'experiment_folder': combo_id,
        'description': f"Benchmark Training OPT-125m: {combo_id}"
    }
    create_divergence_plots(
        encoder_distances=encoder_distances,
        decoder_distances=decoder_distances,
        activation_distances=activation_distances,
        results=divergences,
        output_dir=output_dir_final,
        metadata=metadata_plot
    )
    
    # Save results JSON
    with open(results_file, 'w') as f:
        json.dump({
            'config': {
                'combo_id': combo_id,
                'architecture': architecture,
                'width': width_name,
                'sparsity_param': combo['sparsity_param'],
                'sparsity_value': combo['sparsity_value'],
            },
            'mean_resultant_length': float(mean_resultant_length),
            'divergence_statistics': divergences,
        }, f, indent=2)
    print(f"✓ Saved results to: {results_file}")
    
    # Cleanup
    print("\n[Cleanup] Cleaning up...")
    del store
    del sae
    del activations
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir_final}")
    return model


def run_all_analyses(
    device: str = None,
    n_batches: int = 30,
    output_dir: Path = None,
    skip_existing: bool = True
):
    """Run mean direction divergence analysis on all benchmark training OPT-125m combinations."""
    combinations = load_combinations()
    opt125m_combinations = filter_combinations_for_llm(combinations, "opt-125m")
    
    total_analyses = len(opt125m_combinations)
    completed = 0
    skipped = 0
    failed = []
    
    print("="*70)
    print("BATCH MEAN DIRECTION DIVERGENCE ANALYSIS: ALL BENCHMARK TRAINING OPT-125m MODELS")
    print("="*70)
    print(f"\nTotal analyses to run: {total_analyses}")
    print(f"Device: {device or 'auto-detect'}")
    
    start_time = time.time()
    
    if device is None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU is not available. GPU is required.")
        device = "cuda:0"
    
    model = load_base_model_for_llm("opt-125m", device=device)
    
    try:
        for i, combo in enumerate(opt125m_combinations, 1):
            combo_id = (
                f"{combo['architecture']}_L{combo['layer']}_"
                f"{combo['width']}_{combo['sparsity_param']}{combo['sparsity_value']}"
            )
            
            if skip_existing and output_dir:
                parsed = parse_combo_id(combo_id)
                width_name = parsed['width']
                architecture = parsed['architecture']
                check_dir = output_dir / "benchmark_training_opt-125m" / width_name / architecture / combo_id
                if (check_dir / "results.json").exists():
                    print(f"\n[{i}/{total_analyses}] ⏭️  Skipping (already exists): {combo_id}")
                    skipped += 1
                    continue
            
            print(f"\n[{i}/{total_analyses}] 🔄 Processing: {combo_id}")
            try:
                model = run_analysis_benchmark_training_opt125m(
                    combo=combo,
                    device=device,
                    n_batches=n_batches,
                    output_dir=output_dir,
                    model=model
                )
                completed += 1
                print(f"  ✅ Completed ({completed}/{total_analyses})")
                
                import torch
                import gc
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                    
            except Exception as e:
                error_msg = str(e)[:200]
                print(f"  ❌ Failed: {error_msg}")
                failed.append({'combo_id': combo_id, 'error': error_msg})
                
                import torch
                import gc
                if torch.cuda.is_available():
                    gc.collect()
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
    finally:
        print("\n🧹 Final cleanup...")
        del model
        import torch
        import gc
        if torch.cuda.is_available():
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        cleanup_model_cache("opt-125m")
    
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
            print(f"  - {f['combo_id']}: {f['error']}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run mean direction divergence analysis on benchmark training OPT-125m models")
    parser.add_argument('--device', type=str, default=None, help='Device to use (default: auto-detect cuda:0)')
    parser.add_argument('--n-batches', type=int, default=30, help='Number of batches (default: 30)')
    parser.add_argument('--output-dir', type=str, default=None, help='Output directory')
    parser.add_argument('--no-skip-existing', action='store_true', help='Do not skip existing results')
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir) if args.output_dir else None
    
    run_all_analyses(
        device=args.device,
        n_batches=args.n_batches,
        output_dir=output_dir,
        skip_existing=not args.no_skip_existing
    )
