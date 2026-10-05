#!/usr/bin/env python3
"""
Example Runner: Encoder-Decoder-Activation Analysis

This script demonstrates how to use the helper functions to analyze
encoder-decoder-activation relationships for SAEs from different sources.

Example usage:
    # Analyze SAE Bench SAE
    python run_analysis_example.py \
        --source saebench \
        --repo-id adamkarvonen/saebench_pythia-160m-deduped_width-2pow14_date-0108 \
        --sae-type TopK_pythia-160m-deduped__0108 \
        --layer resid_post_layer_8 \
        --trainer trainer_0
"""

import sys
import os
import argparse
import numpy as np
from pathlib import Path

# Add parent directory to path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

from encoder_decoder_activation_relationship import (
    analyze_activated_features_angular_distances,
    create_activated_features_plot,
)

# Import SAE loading functions
from sae_family_load_convert.saebench import (
    load_saebench_sae,
    cleanup_sae_cache,
)
from sae_family_load_convert import (
    setup_activations_store,
    load_base_model,
)

from encoder_decoder_activation_relationship.cleanup_utils import (
    cleanup_model_cache,
)


def collect_activations(store, n_batches: int = 50):
    """
    Collect activations from the activation store.
    
    Args:
        store: ActivationsStore
        n_batches: Number of batches to collect
        
    Returns:
        numpy array of activations (n_samples, d_in)
    """
    print(f"\n📊 Collecting {n_batches} batches of activations...")
    all_activations = []
    
    for batch_idx in range(n_batches):
        batch_activations = store.next_batch()
        all_activations.append(batch_activations.cpu().numpy())
    
    all_activations = np.concatenate(all_activations, axis=0)
    print(f"✓ Collected {all_activations.shape[0]:,} activation vectors")
    
    return all_activations


def run_analysis_saebench(
    repo_id: str,
    sae_type_folder: str,
    layer_folder: str = "resid_post_layer_8",
    trainer_folder: str = "trainer_0",
    device: str = None,
    n_batches: int = 50,
    n_samples: int = 5000,
    output_dir: Path = None
):
    """
    Run encoder-decoder-activation analysis for SAE Bench SAE.
    
    Args:
        repo_id: HuggingFace repo ID
        sae_type_folder: SAE type folder name
        layer_folder: Layer folder name
        trainer_folder: Trainer folder name
        device: Device to use (default: auto-detect cuda:0, requires GPU)
        n_batches: Number of batches for activation collection
        n_samples: Number of samples for analysis
        output_dir: Output directory (default: /data/user/sae/encoder_decoder_activation_relationship/)
    """
    # Auto-detect device if not specified - require GPU
    if device is None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU is not available. GPU is required for analysis.")
        device = "cuda:0"
        print(f"Using GPU: {device}")
    
    print("="*70)
    print("ENCODER-DECODER-ACTIVATION ANALYSIS")
    print("="*70)
    
    # Setup output directory
    if output_dir is None:
        output_dir = Path(os.environ.get("DATA_DIR", "/data/user")) / "sae" / "encoder_decoder_activation_relationship"
    output_dir = output_dir / "saebench" / f"{sae_type_folder}_{layer_folder}_{trainer_folder}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nOutput directory: {output_dir}")
    
    # Step 1: Load SAE
    print("\n[1/4] Loading SAE...")
    sae, metadata = load_saebench_sae(
        repo_id=repo_id,
        sae_type_folder=sae_type_folder,
        layer_folder=layer_folder,
        trainer_folder=trainer_folder,
        device=device
    )
    
    # Step 2: Load base model
    print("\n[2/4] Loading base model...")
    model = load_base_model(
        model_name=metadata['model_name'],
        device=device
    )
    
    # Step 3: Setup activation store and collect activations
    print("\n[3/4] Collecting activations...")
    store = setup_activations_store(
        model=model,
        sae=sae,
        metadata=metadata,
        device=device
    )
    
    activations = collect_activations(store, n_batches=n_batches)
    
    # Step 4: Run analysis
    print("\n[4/4] Running angular distance analysis...")
    
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
    metadata['experiment_folder'] = f"{sae_type_folder}_{layer_folder}_{trainer_folder}"
    metadata['description'] = f"SAE Bench: {sae_type_folder}"
    
    plot_path = output_dir / "2_activated_features_analysis.png"
    create_activated_features_plot(
        results=results,
        output_path=plot_path,
        metadata=metadata
    )
    
    # Save results JSON
    import json
    results_file = output_dir / "results.json"
    with open(results_file, 'w') as f:
        json.dump(results['summary'], f, indent=2)
    print(f"✓ Saved results to: {results_file}")
    
    # Clean up SAE cache after analysis
    print("\n[Cleanup] Removing cached SAE files...")
    cleanup_sae_cache(metadata['_cache_path'])
    
    # Clean up LLM model cache after analysis
    print("\n[Cleanup] Removing cached LLM model files...")
    cleanup_model_cache(metadata['model_name'])
    
    print(f"\n✅ Analysis complete! Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(
        description="Run encoder-decoder-activation analysis for SAEs"
    )
    parser.add_argument(
        '--source',
        type=str,
        default='saebench',
        help='SAE source (default: saebench)'
    )
    parser.add_argument(
        '--repo-id',
        type=str,
        required=True,
        help='HuggingFace repo ID (for saebench)'
    )
    parser.add_argument(
        '--sae-type',
        type=str,
        required=True,
        dest='sae_type_folder',
        help='SAE type folder name (e.g., TopK_pythia-160m-deduped__0108)'
    )
    parser.add_argument(
        '--layer',
        type=str,
        default='resid_post_layer_8',
        dest='layer_folder',
        help='Layer folder name (default: resid_post_layer_8)'
    )
    parser.add_argument(
        '--trainer',
        type=str,
        default='trainer_0',
        dest='trainer_folder',
        help='Trainer folder name (default: trainer_0)'
    )
    parser.add_argument(
        '--device',
        type=str,
        default=None,
        help='Device to use (default: auto-detect cuda:0 if available, else cpu)'
    )
    parser.add_argument(
        '--n-batches',
        type=int,
        default=50,
        help='Number of batches for activation collection (default: 50)'
    )
    parser.add_argument(
        '--n-samples',
        type=int,
        default=5000,
        help='Number of samples for analysis (default: 5000)'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: /data/user/sae/encoder_decoder_activation_relationship/)'
    )
    
    args = parser.parse_args()
    
    # Auto-detect device if not specified - require GPU
    if args.device is None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU is not available. GPU is required for analysis.")
        args.device = "cuda:0"
        print(f"Using GPU: {args.device}")
    
    if args.source == 'saebench':
        output_dir = Path(args.output_dir) if args.output_dir else None
        run_analysis_saebench(
            repo_id=args.repo_id,
            sae_type_folder=args.sae_type_folder,
            layer_folder=args.layer_folder,
            trainer_folder=args.trainer_folder,
            device=args.device,
            n_batches=args.n_batches,
            n_samples=args.n_samples,
            output_dir=output_dir
        )
    else:
        raise ValueError(f"Unknown source: {args.source}")


if __name__ == "__main__":
    main()

