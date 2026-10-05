#!/usr/bin/env python3
"""
Run dominance-aware auto-interpretability evaluation on all Benchmark Training OPT-125M SAEs.

This script uses the exact same workflow as test_single_sae_api.py:
- Pass 0: Fast counting (for quantile sampling)
- Combined Pass: Collect all sequences, bin into 3 bins, sample from bins, decode only sampled sequences
- Explanation: Generate explanations using offline model (mistralai/Mistral-7B-Instruct-v0.2)
- Evaluation: Token-based and sentence-based multiple-choice questions using offline model (mistralai/Mistral-7B-Instruct-v0.2)

Configuration:
- Widths: 4k, 16k, 65k (all 3 widths)
- Architectures: standard, gated, jumprelu, topk, batchtopk, matryoshka_batchtopk (6 types)
- Sparsity levels: 3 per architecture
- Features: 20 per quantile (80 total per SAE)
- Total SAEs: 6 architectures × 3 widths × 3 sparsity = 54 SAEs

Output: /data/user/sae/benchmark/monosemanticity/dominance_aware_auto_inter_score/{model_family}/{width}/{architecture}/{sparsity}/

Usage:
    python run_all_benchmark_gpt2.py [--device cuda:0] [--skip-existing]
"""

import sys
import os
import asyncio
import argparse
import re
from pathlib import Path
from datetime import datetime
import logging
import json
import gc
import torch
import numpy as np
from collections import defaultdict
from typing import Optional, Dict, List, Any
import sys as sys_module
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

# Set HuggingFace cache directory to /data/user
os.environ['HF_HOME'] = '/data/user/.cache/huggingface'
os.environ['HF_DATASETS_CACHE'] = '/data/user/.cache/huggingface/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/data/user/.cache/huggingface/transformers'
# Ensure cache directory exists
Path('/data/user/.cache/huggingface').mkdir(parents=True, exist_ok=True)
Path('/data/user/.cache/huggingface/datasets').mkdir(parents=True, exist_ok=True)
Path('/data/user/.cache/huggingface/transformers').mkdir(parents=True, exist_ok=True)

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Add benchmark_training for loading SAEs
benchmark_training_dir = project_root / "benchmark_training"
sys.path.insert(0, str(benchmark_training_dir))

# Add SAELens to path for imports
saelens_dir = project_root / "SAELens-main"
sys.path.insert(0, str(saelens_dir))

# Add dominance_aware_auto_inter_score to path for core imports
dame_dir = Path(__file__).parent.parent
sys.path.insert(0, str(dame_dir))

from benchmark_training.load_benchmark_sae import load_benchmark_sae, parse_combo_id
from benchmark_training.benchmark_config import LLM_CONFIGS, ARCHITECTURES, WIDTHS
from transformer_lens import HookedTransformer
from sae_lens.training.activations_store import ActivationsStore
from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.saes.standard_sae import StandardTrainingSAEConfig
from core import ActivationCollector, CollectionConfig, ExplanationGenerator, TokenBasedEvaluator, SentenceBasedEvaluator, create_quantile_scores_plot
from core.visualization import create_dame_metric_plots
from tqdm import tqdm

# Setup directories
current_dir = Path(__file__).parent.parent
log_dir = current_dir / "log"
log_dir.mkdir(exist_ok=True)

# Output base directory: /data/user/sae/benchmark/mono/dame/
output_base_dir = Path("/data/user/sae/benchmark/mono/dame")
output_base_dir.mkdir(parents=True, exist_ok=True)

# Setup logging
log_file = log_dir / f"run_all_benchmark_opt125m_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file, mode='w'),
        logging.StreamHandler(sys_module.stdout)
    ],
    force=True
)
logger = logging.getLogger(__name__)

# Force unbuffered output
sys_module.stdout.reconfigure(line_buffering=True)
sys_module.stderr.reconfigure(line_buffering=True)

# Benchmark Training OPT-125M Configuration
BENCHMARK_CONFIG = {
    "llm_name": "opt-125m",
    "model_name": "opt-125m",
    "layer": 6,  # Middle layer
    "hook_name": "blocks.6.hook_mlp_out",
    "architectures": ["standard", "gated", "jumprelu", "topk", "batchtopk", "matryoshka_batchtopk"],
    "widths": ["4k", "16k", "65k"],
    "base_dir": Path("/data/user/sae/training"),
}

# Configuration: 20 features per quantile (80 total)
N_FEATURES_PER_QUANTILE = 20
N_BATCHES_FAST_COUNT = 10  # Fewer batches for fast counting (dominance estimation) - 10 is enough for good estimate
N_BATCHES_DISCOVERY = 50  # Full batches for actual sequence collection


def load_base_model(model_name: str, device: str):
    """Load base model using HookedTransformer."""
    # Use cache_dir to ensure caching to /data/user
    cache_dir = '/data/user/.cache/huggingface/transformers'
    return HookedTransformer.from_pretrained(
        model_name, 
        device=device,
        cache_dir=cache_dir
    )


def setup_activations_store(model, sae, metadata, device: str, dataset_path: str):
    """Setup activation store for collecting activations."""
    dummy_sae_cfg = StandardTrainingSAEConfig(
        d_in=sae.cfg.d_in,
        d_sae=sae.cfg.d_sae,
        device=device,
        dtype="float32",
    )
    
    hook_name = metadata.get('hook_name', BENCHMARK_CONFIG['hook_name'])
    
    sentence_size = 16
    store_batch_size_prompts = 32
    
    cfg = LanguageModelSAERunnerConfig(
        sae=dummy_sae_cfg,
        model_name=metadata.get('model_name', BENCHMARK_CONFIG['model_name']),
        hook_name=hook_name,
        dataset_path=dataset_path,
        is_dataset_tokenized=True,
        streaming=True,
        train_batch_size_tokens=4096,
        context_size=sentence_size,
        n_batches_in_buffer=32,
        training_tokens=4096 * 100,
        store_batch_size_prompts=store_batch_size_prompts,
        device=device,
        seed=42,
        dtype="float32",
    )
    
    return ActivationsStore.from_config(model, cfg)


def check_output_exists(combo_id: str, output_base_dir: Path) -> bool:
    """Check if output already exists for this SAE."""
    # Parse combo_id: {architecture}_L6_{width}_{sparsity}
    # New structure: {model_family}/{width}/{architecture}/{sparsity}/
    model_family = "benchmark_opt125m"
    
    # Parse combo_id using regex to handle multi-part architectures like "matryoshka_batchtopk"
    import re
    pattern = re.match(r'^(.+)_L6_(\d+k)_(.+)$', combo_id)
    if not pattern:
        # Fallback to old structure check
        output_dir = output_base_dir / combo_id
        results_file = output_dir / "results.json"
        return results_file.exists()
    
    architecture = pattern.group(1)
    width = pattern.group(2)
    sparsity = pattern.group(3)
    
    output_dir = output_base_dir / model_family / width / architecture / sparsity
    results_file = output_dir / "results.json"
    return results_file.exists()


def validate_sae_loaded_correctly(sae) -> None:
    """Validate that SAE weights are loaded correctly."""
    if not hasattr(sae, 'W_enc') or sae.W_enc is None:
        raise ValueError("SAE missing W_enc - loading may have failed")
    if not hasattr(sae, 'W_dec') or sae.W_dec is None:
        raise ValueError("SAE missing W_dec - loading may have failed")
    
    if sae.W_enc.shape[0] != sae.cfg.d_in or sae.W_enc.shape[1] != sae.cfg.d_sae:
        raise ValueError(f"W_enc shape mismatch: expected ({sae.cfg.d_in}, {sae.cfg.d_sae}), got {sae.W_enc.shape}")
    if sae.W_dec.shape[0] != sae.cfg.d_sae or sae.W_dec.shape[1] != sae.cfg.d_in:
        raise ValueError(f"W_dec shape mismatch: expected ({sae.cfg.d_sae}, {sae.cfg.d_in}), got {sae.W_dec.shape}")
    
    w_enc_norm = torch.norm(sae.W_enc).item()
    w_dec_norm = torch.norm(sae.W_dec).item()
    
    if w_enc_norm < 0.1:
        raise ValueError(f"W_enc norm too small ({w_enc_norm:.6f}) - appears uninitialized/untrained")
    if w_dec_norm < 0.1:
        raise ValueError(f"W_dec norm too small ({w_dec_norm:.6f}) - appears uninitialized/untrained")
    
    if torch.allclose(sae.W_enc, torch.zeros_like(sae.W_enc), atol=1e-6):
        raise ValueError("W_enc is all zeros - loading failed")
    if torch.allclose(sae.W_dec, torch.zeros_like(sae.W_dec), atol=1e-6):
        raise ValueError("W_dec is all zeros - loading failed")
    
    logger.info(f"✓ SAE weights validated: W_enc norm={w_enc_norm:.2f}, W_dec norm={w_dec_norm:.2f}")


def convert_to_json_serializable(obj):
    """Recursively convert numpy types to Python native types."""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {int(k) if isinstance(k, np.integer) else k: convert_to_json_serializable(v) 
               for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_json_serializable(item) for item in obj]
    elif isinstance(obj, tuple):
        return tuple(convert_to_json_serializable(item) for item in obj)
    else:
        return obj


def generate_all_combinations():
    """Generate all combinations for OPT-125M benchmark training SAEs."""
    combinations = []
    llm_name = BENCHMARK_CONFIG['llm_name']
    
    for arch_name in BENCHMARK_CONFIG['architectures']:
        if arch_name not in ARCHITECTURES:
            continue
        arch_config = ARCHITECTURES[arch_name]
        sparsity_param = arch_config['sparsity_param']
        sparsity_values = arch_config['sparsity_values']
        
        for width_name in BENCHMARK_CONFIG['widths']:
            for sparsity_value in sparsity_values:
                # Generate combo_id
                combo_id = f"{arch_name}_L{BENCHMARK_CONFIG['layer']}_{width_name}_{sparsity_param}{sparsity_value}"
                combinations.append({
                    'combo_id': combo_id,
                    'architecture': arch_name,
                    'width': width_name,
                    'sparsity_param': sparsity_param,
                    'sparsity_value': sparsity_value,
                })
    
    return combinations


async def run_evaluation_for_sae(
    combo_id: str,
    device: str,
    output_base_dir: Path,
    model_cache: Optional[Dict[str, Any]] = None,
) -> Dict:
    """Run dominance-aware evaluation for a single SAE."""
    # Parse combo_id: {architecture}_L6_{width}_{sparsity}
    # New structure: {model_family}/{width}/{architecture}/{sparsity}/
    model_family = "benchmark_opt125m"
    
    # Parse combo_id using regex to handle multi-part architectures like "matryoshka_batchtopk"
    import re
    pattern = re.match(r'^(.+)_L6_(\d+k)_(.+)$', combo_id)
    if not pattern:
        # Fallback to old structure
        output_dir = output_base_dir / combo_id
    else:
        architecture = pattern.group(1)
        width = pattern.group(2)
        sparsity = pattern.group(3)
        output_dir = output_base_dir / model_family / width / architecture / sparsity
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"\n{'='*70}")
    logger.info(f"Processing: {combo_id}")
    logger.info(f"Output: {output_dir}")
    logger.info(f"{'='*70}")
    sys_module.stdout.flush()
    
    start_time = datetime.now()
    
    try:
        # Step 1: Load SAE
        logger.info("[1/8] Loading SAE...")
        sys_module.stdout.flush()
        sae, sae_metadata = load_benchmark_sae(
            combo_id=combo_id,
            llm_name=BENCHMARK_CONFIG['llm_name'],
            device=device,
            base_dir=BENCHMARK_CONFIG['base_dir']
        )
        logger.info(f"✓ Loaded SAE: {sae.cfg.d_sae} features")
        
        # Validate SAE weights
        logger.info("Validating SAE weights...")
        validate_sae_loaded_correctly(sae)
        sys_module.stdout.flush()
        
        if sae_metadata is None:
            sae_metadata = {}
        sae_metadata['model_name'] = BENCHMARK_CONFIG['model_name']
        sae_metadata['hook_name'] = BENCHMARK_CONFIG['hook_name']
        sae_metadata['layer'] = BENCHMARK_CONFIG['layer']
        sae_metadata['combo_id'] = combo_id
        
        # Step 2: Load model (reuse if available)
        model_name = BENCHMARK_CONFIG['model_name']
        if model_cache is None:
            model_cache = {}
        
        if model_name not in model_cache or model_cache[model_name] is None:
            logger.info("[2/8] Loading model...")
            sys_module.stdout.flush()
            model = load_base_model(model_name=model_name, device=device)
            model_cache[model_name] = model
        else:
            logger.info("[2/8] Reusing cached model...")
            sys_module.stdout.flush()
            model = model_cache[model_name]
            # If cached model is on CPU, move it back to GPU
            if model is not None and next(model.parameters()).device.type == 'cpu':
                logger.info("  Moving cached model from CPU back to GPU...")
                model = model.to(device)
                model_cache[model_name] = model
        logger.info(f"✓ Loaded model: {model_name}")
        sys_module.stdout.flush()
        
        # Step 3: Setup activation store
        logger.info("[3/8] Setting up activation store...")
        sys_module.stdout.flush()
        store = setup_activations_store(
            model=model,
            sae=sae,
            metadata=sae_metadata,
            device=device,
            dataset_path="NeelNanda/pile-10k",
        )
        logger.info(f"✓ Setup activation store")
        sys_module.stdout.flush()
        
        # Detect SAE architecture
        sae_architecture = getattr(sae.cfg, 'architecture', 'standard')
        if callable(sae_architecture):
            sae_architecture = sae_architecture()
        architectures_with_signed = ['topk', 'batchtopk', 'topk_abs', 'batchtopk_abs']
        use_absolute_value = any(arch in sae_architecture.lower() for arch in architectures_with_signed)
        
        # PASS 0: Fast counting (using core method)
        # ====================================================================
        logger.info("\n[4/8] Pass 0: Feature discovery (fast counting)...")
        logger.info(f"  Step 1: Counting firing frequency across {N_BATCHES_DISCOVERY} batches (for accurate filtering)...")
        logger.info(f"  Step 2: Calculating dominance across {N_BATCHES_FAST_COUNT} batches (for ranking, fast estimate)...")
        sys_module.stdout.flush()
        
        # Create collector for Pass 0
        temp_collector = ActivationCollector(
            model=model,
            sae=sae,
            store=store,
            device=device,
            config=CollectionConfig()
        )
        
        # Step 1: Count firing frequency with FULL batches (50) for accurate filtering
        def progress_callback_firing(batch_idx, total):
            if batch_idx % 25 == 0:
                logger.info(f"    Firing count: Processed {batch_idx}/{total} batches...")
                sys_module.stdout.flush()
        
        logger.info("  Counting firing frequency...")
        sys_module.stdout.flush()
        active_features_firing, sequence_firing_counts, total_sequences_firing, _ = temp_collector.fast_count_features(
            n_batches=N_BATCHES_DISCOVERY,
            use_absolute_value=use_absolute_value,
            min_firing_count=80,
            progress_callback=progress_callback_firing
        )
        logger.info(f"  ✓ Firing count complete: {len(active_features_firing)} features fired ≥80 times across {total_sequences_firing:,} sequences")
        sys_module.stdout.flush()
        
        # Reset store to start from beginning for dominance calculation
        if hasattr(store, 'reset_input_dataset'):
            store.reset_input_dataset()
            if hasattr(store, 'iterable_sequences') and hasattr(store, '_iterate_tokenized_sequences'):
                store.iterable_sequences = store._iterate_tokenized_sequences()
        
        # Step 2: Calculate dominance with FEWER batches (10) for fast ranking
        def progress_callback_dominance(batch_idx, total):
            if batch_idx % 5 == 0 or batch_idx == total:
                logger.info(f"    Dominance calc: Processed {batch_idx}/{total} batches...")
                sys_module.stdout.flush()
        
        logger.info("  Calculating dominance scores...")
        sys_module.stdout.flush()
        active_features_dom, _, dominance_scores, total_sequences_dom, _ = temp_collector.fast_count_dominance(
            n_batches=N_BATCHES_FAST_COUNT,
            use_absolute_value=use_absolute_value,
            min_firing_count=0,  # Don't filter here, use firing counts from step 1
            progress_callback=progress_callback_dominance
        )
        logger.info(f"  ✓ Dominance calculation complete: {total_sequences_dom:,} sequences processed")
        sys_module.stdout.flush()
        
        # Combine: Use firing counts from 50 batches (accurate) + dominance from 10 batches (fast estimate)
        # Filter active features based on firing count from 50 batches
        active_features = [f for f in active_features_firing if sequence_firing_counts[f] >= 80]
        total_sequences = total_sequences_firing  # Use total from firing count (50 batches)
        
        logger.info(f"✓ Found {len(active_features)} active features out of {sae.cfg.d_sae}")
        logger.info(f"  Total sequences processed: {total_sequences:,}")
        if len(dominance_scores[dominance_scores > 0]) > 0:
            logger.info(f"  Dominance range: {dominance_scores[dominance_scores > 0].min():.6f} - {dominance_scores.max():.6f}")
            logger.info(f"  Mean dominance: {dominance_scores[dominance_scores > 0].mean():.6f}")
        sys_module.stdout.flush()
        
        # ====================================================================
        # Quantile Sampling by DOMINANCE (using core method)
        # ====================================================================
        logger.info("\n[5/8] Filtering and quantile sampling by DOMINANCE...")
        sys_module.stdout.flush()
        
        final_sampled_features, quantile_stats = ActivationCollector.sample_features_by_dominance_quantiles(
            active_features=active_features,
            dominance_scores=dominance_scores,
            sequence_firing_counts=sequence_firing_counts,
            total_sequences=total_sequences,
            n_features_per_quantile=N_FEATURES_PER_QUANTILE,
            min_firing_count=80
        )
        
        logger.info(f"✓ Sampled {len(final_sampled_features)} features total ({N_FEATURES_PER_QUANTILE} per quartile)")
        logger.info("\nDominance Quantile Statistics:")
        for qname, stats in quantile_stats.items():
            logger.info(f"  {qname}:")
            logger.info(f"    Features: {stats['n_sampled']}/{stats['n_features']} sampled")
            logger.info(f"    Dominance: {stats['min_dominance']:.6f} - {stats['max_dominance']:.6f} (mean: {stats['mean_dominance']:.6f})")
            logger.info(f"    Firing frequency: {stats['min_firing']} - {stats['max_firing']} (mean: {stats['mean_firing']:.1f})")
            logger.info(f"    Firing percentage: {stats['firing_percentage']:.4f}% of sequences")
        sys_module.stdout.flush()
        
        # ====================================================================
        # COMBINED PASS: Collect activations, bin, sample, and decode
        # ====================================================================
        logger.info("\n[6/8] Combined Pass: Collecting activations, binning, sampling, and decoding...")
        logger.info(f"  Processing {N_BATCHES_DISCOVERY} batches for {len(final_sampled_features)} features")
        logger.info(f"  Strategy: Collect all fired sequences, bin into 3 bins, sample from bins")
        sys_module.stdout.flush()
        
        # Reset store to start from beginning
        if hasattr(store, 'reset_input_dataset'):
            store.reset_input_dataset()
            if hasattr(store, 'iterable_sequences') and hasattr(store, '_iterate_tokenized_sequences'):
                store.iterable_sequences = store._iterate_tokenized_sequences()
        
        # Storage: Collect ALL sequences per feature
        feature_all_sequences_max = defaultdict(list)
        feature_all_sequences_sum = defaultdict(list)
        batch_tokens_cache = {}
        
        selected_features_set = set(final_sampled_features)
        
        logger.info("  Step 1: Collecting all sequences (including non-firing)...")
        sys_module.stdout.flush()
        
        for batch_idx in range(N_BATCHES_DISCOVERY):
            batch_tokens = store.get_batch_tokens()
            if batch_tokens is None:
                break
            
            batch_tokens_cache[batch_idx] = batch_tokens.clone()
            batch_size = batch_tokens.shape[0]
            
            with torch.no_grad():
                model_acts = store.get_activations(batch_tokens)
                seq_len = batch_tokens.shape[1]
                model_acts_flat = model_acts.reshape(-1, model_acts.shape[-1])
                # Only move to device if not already there (avoid redundant transfers)
                if model_acts_flat.device != torch.device(device):
                    model_acts_flat = model_acts_flat.to(device)
                
                sae_acts_flat = sae.encode(model_acts_flat)
                sae_acts = sae_acts_flat.reshape(batch_size, seq_len, sae.cfg.d_sae)
                
                if use_absolute_value:
                    sae_acts = torch.abs(sae_acts)
                
                # Process all selected features on GPU first, then transfer once
                selected_features_list = list(selected_features_set)
                selected_features_tensor = torch.tensor(selected_features_list, device=device, dtype=torch.long)
                
                # Extract activations for all selected features at once: [batch_size, seq_len, n_selected]
                selected_acts = sae_acts[:, :, selected_features_tensor]  # [batch_size, seq_len, n_selected]
                
                # Compute max, argmax, and sum for all features on GPU
                max_activations = selected_acts.max(dim=1)[0]  # [batch_size, n_selected]
                max_token_indices = selected_acts.argmax(dim=1)  # [batch_size, n_selected]
                sum_activations = selected_acts.sum(dim=1)  # [batch_size, n_selected]
                
                # Transfer to CPU once for entire batch
                max_acts_cpu = max_activations.cpu().numpy()
                max_tok_cpu = max_token_indices.cpu().numpy()
                sum_acts_cpu = sum_activations.cpu().numpy()
                
                # Now store results (all on CPU)
                for seq_idx in range(batch_size):
                    for feat_list_idx, feat_idx in enumerate(selected_features_list):
                        max_activation = float(max_acts_cpu[seq_idx, feat_list_idx])
                        max_token_idx = int(max_tok_cpu[seq_idx, feat_list_idx])
                        sum_activation = float(sum_acts_cpu[seq_idx, feat_list_idx])
                        
                        feature_all_sequences_max[feat_idx].append(
                            (max_activation, sum_activation, batch_idx, seq_idx, max_token_idx)
                        )
                        feature_all_sequences_sum[feat_idx].append(
                            (sum_activation, max_activation, batch_idx, seq_idx, max_token_idx)
                        )
                
                # Clean up intermediate tensors to reduce memory pressure
                del model_acts, model_acts_flat, sae_acts_flat, sae_acts
            
            if (batch_idx + 1) % 25 == 0:
                logger.info(f"    Processed {batch_idx + 1}/{N_BATCHES_DISCOVERY} batches...")
                sys_module.stdout.flush()
                # Periodic memory cleanup to prevent accumulation
                gc.collect()
                if torch.cuda.is_available():
                    try:
                        torch.cuda.empty_cache()
                    except:
                        pass
        
        logger.info(f"✓ Collected all sequences for {len(final_sampled_features)} features")
        sys_module.stdout.flush()
        
        # Step 2: Bin sequences and sample
        logger.info("  Step 2: Binning sequences and sampling...")
        sys_module.stdout.flush()
        
        collection_config = CollectionConfig(
            sentence_size=16,
            n_batches=0,
            use_absolute_value=use_absolute_value,
            n_high_for_explanation=20,
            n_high_for_evaluation=40,  # Increased to 40 to get 20 in bin_A and 20 in bin_B
            n_low_for_contrast=20,  # Reduced to 20 (only need 20 for bin_C)
        )
        
        collector = ActivationCollector(
            model=model,
            sae=sae,
            store=store,
            device=device,
            config=collection_config
        )
        
        sequence_references, collection_stats = collector.bin_and_sample_sequences(
            feature_indices=final_sampled_features,
            all_sequences_max=feature_all_sequences_max,
            all_sequences_sum=feature_all_sequences_sum,
            n_explanation=20,
            n_evaluation=40,  # Increased to 40 to get 20 in bin_A and 20 in bin_B
            n_low=20,  # Reduced to 20 (only need 20 for bin_C)
            random_seed=42
        )
        
        logger.info(f"✓ Binning and sampling complete")
        sys_module.stdout.flush()
        
        # Step 3: Decode only sampled sequences
        logger.info("  Step 3: Decoding sampled sequences...")
        sys_module.stdout.flush()
        
        collection_stats_result = collector.collect_optimized(
            feature_indices=final_sampled_features,
            sequence_references=sequence_references,
            batch_tokens_cache=batch_tokens_cache,
            use_absolute_value=use_absolute_value
        )
        
        # Clean up batch_tokens_cache immediately after use to free memory
        del batch_tokens_cache
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except:
                pass
        
        for feat_idx in collection_stats:
            if feat_idx in collection_stats_result:
                collection_stats[feat_idx].update(collection_stats_result[feat_idx])
        
        total_collected = sum(len(collector.high_explanation_sequences.get(f, [])) + 
                             len(collector.high_max_sequences.get(f, [])) + 
                             len(collector.low_max_sequences.get(f, [])) 
                             for f in final_sampled_features)
        logger.info(f"✓ Combined pass complete: Collected {total_collected} sequences total")
        
        # Clean up base model and SAE to free GPU memory before loading Mistral
        logger.info("\n[Memory Management] Freeing base model and SAE to make room for Mistral...")
        sys_module.stdout.flush()
        
        # Move base model to CPU (or delete if not needed)
        if model is not None:
            try:
                # Move model to CPU to free GPU memory
                model = model.cpu()
                if model_name in model_cache:
                    model_cache[model_name] = model  # Update cache with CPU model
                logger.info("  ✓ Moved base model to CPU")
            except Exception as e:
                logger.warning(f"  ⚠️  Could not move model to CPU: {e}")
        
        # Delete SAE from GPU (we don't need it anymore after collection)
        if sae is not None:
            try:
                del sae
                sae = None
                logger.info("  ✓ Deleted SAE from GPU")
            except Exception as e:
                logger.warning(f"  ⚠️  Could not delete SAE: {e}")
        
        # Clean up store
        if store is not None:
            try:
                del store
                store = None
                logger.info("  ✓ Deleted activation store")
            except Exception as e:
                logger.warning(f"  ⚠️  Could not delete store: {e}")
        
        # Aggressive GPU cleanup
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
                logger.info("  ✓ Cleared CUDA cache")
            except Exception as e:
                logger.warning(f"  ⚠️  CUDA cleanup warning: {e}")
        
        sys_module.stdout.flush()
        
        # Step 7: Explanation generation (Offline)
        logger.info("\n[7/8] Explanation generation (Offline)...")
        logger.info("  Note: Processing 80 features sequentially, 20 examples per feature")
        sys_module.stdout.flush()
        explanation_gen = ExplanationGenerator(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            n_examples=20,
            include_low_contrast=False,
            provider="offline",
            device=device,
        )
        
        explanations, _ = await explanation_gen.generate_explanations_batch(
            collector, final_sampled_features
        )
        logger.info(f"✓ Generated {len([e for e in explanations.values() if e.get('explanation')])} explanations")
        sys_module.stdout.flush()
        
        # Clean up explanation generator
        logger.info("Cleaning up explanation generator to free GPU memory...")
        explanation_gen.cleanup()
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except (RuntimeError, Exception):
                pass
        logger.info("✓ Memory freed")
        sys_module.stdout.flush()
        
        # Step 8: Multiple-choice evaluation (Offline)
        logger.info("\n[8/8] Multiple-choice evaluation (Offline)...")
        logger.info(f"  Token evaluator: offline, Model: mistralai/Mistral-7B-Instruct-v0.2")
        logger.info(f"  Sentence evaluator: offline, Model: mistralai/Mistral-7B-Instruct-v0.2")
        sys_module.stdout.flush()
        
        # Token evaluator
        logger.info("  Loading token-based evaluator...")
        sys_module.stdout.flush()
        token_evaluator = TokenBasedEvaluator(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            max_questions=20,
            n_examples=20,
            n_distractors=2,
            provider="offline",
            device=device,
        )
        
        token_results = {}
        logger.info(f"\n  Token-based evaluation: Evaluating {len(final_sampled_features)} features...")
        sys_module.stdout.flush()
        
        for feat_idx in tqdm(final_sampled_features, desc="Token MC"):
            if feat_idx not in explanations or explanations[feat_idx].get('error'):
                continue
            
            explanation = explanations[feat_idx].get('explanation')
            if not explanation:
                continue
            
            explanation_sequence_ids = explanations[feat_idx].get('explanation_sequence_ids', [])
            explanation_sequence_ids_set = set(explanation_sequence_ids) if explanation_sequence_ids else None
            
            # Get sequences from 3 bins (V2 ranking task)
            bin_A_seqs = collector.bin_A_sequences.get(feat_idx, [])
            bin_B_seqs = collector.bin_B_sequences.get(feat_idx, [])
            bin_C_seqs = collector.bin_C_sequences.get(feat_idx, [])
            
            token_result = await token_evaluator.evaluate_feature(
                feat_idx, explanation, bin_A_seqs, bin_B_seqs, bin_C_seqs, explanation_sequence_ids_set
            )
            token_results[feat_idx] = token_result
        
        logger.info("  Cleaning up token evaluator to free GPU memory...")
        token_evaluator.cleanup()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        logger.info("  ✓ Memory freed")
        sys_module.stdout.flush()
        
        # Sentence evaluator
        logger.info("  Loading sentence-based evaluator...")
        sys_module.stdout.flush()
        sentence_evaluator = SentenceBasedEvaluator(
            model_name="mistralai/Mistral-7B-Instruct-v0.2",
            n_distractors=2,
            max_questions=20,
            n_examples=20,
            provider="offline",
            device=device,
        )
        
        sentence_results = {}
        logger.info(f"\n  Sentence-based evaluation: Evaluating {len(final_sampled_features)} features...")
        sys_module.stdout.flush()
        
        for feat_idx in tqdm(final_sampled_features, desc="Sentence MC"):
            if feat_idx not in explanations or explanations[feat_idx].get('error'):
                continue
            
            explanation = explanations[feat_idx].get('explanation')
            if not explanation:
                continue
            
            explanation_sequence_ids = explanations[feat_idx].get('explanation_sequence_ids', [])
            explanation_sequence_ids_set = set(explanation_sequence_ids) if explanation_sequence_ids else None
            
            # Get sequences from 3 sum-based bins (V2 ranking task)
            bin_A_sum_seqs = collector.bin_A_sum_sequences.get(feat_idx, [])
            bin_B_sum_seqs = collector.bin_B_sum_sequences.get(feat_idx, [])
            bin_C_sum_seqs = collector.bin_C_sum_sequences.get(feat_idx, [])
            
            sentence_result = await sentence_evaluator.evaluate_feature(
                feat_idx, explanation, bin_A_sum_seqs, bin_B_sum_seqs, bin_C_sum_seqs, explanation_sequence_ids_set
            )
            sentence_results[feat_idx] = sentence_result
        
        logger.info(f"\n✓ Evaluation complete!")
        logger.info(f"  Token-based: {len(token_results)} features evaluated")
        logger.info(f"  Sentence-based: {len(sentence_results)} features evaluated")
        sys_module.stdout.flush()
        
        # Calculate summary statistics (V2 - 4 metrics per evaluation type)
        summary_stats = {
            'token': {'perfect_ranking': [], 'correlation': [], 'pairwise': [], 'rank_distance': []},
            'sentence': {'perfect_ranking': [], 'correlation': [], 'pairwise': [], 'rank_distance': []}
        }
        
        if token_results:
            for r in token_results.values():
                if not r.get('error'):
                    if 'perfect_ranking_acc' in r:
                        summary_stats['token']['perfect_ranking'].append(r['perfect_ranking_acc'])
                    if 'mean_rank_correlation' in r:
                        summary_stats['token']['correlation'].append(r['mean_rank_correlation'])
                    if 'mean_pairwise_acc' in r:
                        summary_stats['token']['pairwise'].append(r['mean_pairwise_acc'])
                    if 'mean_rank_distance' in r:
                        summary_stats['token']['rank_distance'].append(r['mean_rank_distance'])
        
        if sentence_results:
            for r in sentence_results.values():
                if not r.get('error'):
                    if 'perfect_ranking_acc' in r:
                        summary_stats['sentence']['perfect_ranking'].append(r['perfect_ranking_acc'])
                    if 'mean_rank_correlation' in r:
                        summary_stats['sentence']['correlation'].append(r['mean_rank_correlation'])
                    if 'mean_pairwise_acc' in r:
                        summary_stats['sentence']['pairwise'].append(r['mean_pairwise_acc'])
                    if 'mean_rank_distance' in r:
                        summary_stats['sentence']['rank_distance'].append(r['mean_rank_distance'])
        
        # Compute means
        token_means = {k: np.mean(v) if v else 0.0 for k, v in summary_stats['token'].items()}
        sentence_means = {k: np.mean(v) if v else 0.0 for k, v in summary_stats['sentence'].items()}
        
        logger.info(f"  Token metrics: Perfect={token_means['perfect_ranking']:.3f}, Corr={token_means['correlation']:.3f}, Pairwise={token_means['pairwise']:.3f}, Dist={token_means['rank_distance']:.3f}")
        logger.info(f"  Sentence metrics: Perfect={sentence_means['perfect_ranking']:.3f}, Corr={sentence_means['correlation']:.3f}, Pairwise={sentence_means['pairwise']:.3f}, Dist={sentence_means['rank_distance']:.3f}")
        sys_module.stdout.flush()
        
        # Create feature firing frequencies dict
        feature_firing_frequencies = {}
        for feat_idx in final_sampled_features:
            firing_count = int(sequence_firing_counts[feat_idx])
            firing_percentage = (firing_count / total_sequences) * 100 if total_sequences > 0 else 0.0
            feature_firing_frequencies[int(feat_idx)] = {
                'firing_count': firing_count,
                'firing_percentage': float(firing_percentage),
                'total_sequences': total_sequences
            }
        
        # Save results
        results = {
            'quantile_stats': convert_to_json_serializable(quantile_stats),
            'explanations': {int(k) if isinstance(k, np.integer) else k: 
                           {'explanation': v.get('explanation'), 'error': v.get('error')}
                           for k, v in explanations.items()},
            'collection_stats': convert_to_json_serializable(collection_stats),
            'token_results': convert_to_json_serializable(token_results),
            'sentence_results': convert_to_json_serializable(sentence_results),
            'feature_firing_frequencies': convert_to_json_serializable(feature_firing_frequencies),
            'summary': {
                'n_features_evaluated': len(token_results),
                'token_metrics': {k: float(v) for k, v in token_means.items()},
                'sentence_metrics': {k: float(v) for k, v in sentence_means.items()},
            },
            'metadata': {
                'combo_id': combo_id,
                'llm_name': BENCHMARK_CONFIG['llm_name'],
                'model_name': BENCHMARK_CONFIG['model_name'],
                'layer': BENCHMARK_CONFIG['layer'],
                'hook_name': BENCHMARK_CONFIG['hook_name'],
                'n_features_per_quantile': N_FEATURES_PER_QUANTILE,
                'n_batches_firing_count': N_BATCHES_DISCOVERY,  # Full batches for accurate firing count
                'n_batches_dominance': N_BATCHES_FAST_COUNT,  # Fewer batches for fast dominance estimate
                'n_batches_discovery': N_BATCHES_DISCOVERY,  # Full batches for sequence collection
                'total_sequences': total_sequences,
            }
        }
        
        results_file = output_dir / "results.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n✓ Saved results to: {results_file}")
        sys_module.stdout.flush()
        
        # Create and save 4 separate DAME metric plots
        logger.info("\n[Plotting] Creating 4 DAME metric plots...")
        sys_module.stdout.flush()
        try:
            create_dame_metric_plots(
                results=results,
                final_sampled_features=final_sampled_features,
                quantile_stats=quantile_stats,
                output_dir=output_dir
            )
            logger.info(f"✓ Saved 4 DAME metric plots to: {output_dir}")
        except Exception as e:
            logger.warning(f"⚠️  Failed to create DAME plots: {e}")
            import traceback
            logger.warning(traceback.format_exc())
        sys_module.stdout.flush()
        
        # Cleanup
        logger.info("\n[Cleanup] Cleaning up...")
        sys_module.stdout.flush()
        explanation_gen.cleanup()
        token_evaluator.cleanup()
        sentence_evaluator.cleanup()
        del sae, store, collector
        gc.collect()
        
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except (RuntimeError, Exception) as e:
                logger.debug(f"⚠️  Skipping CUDA cleanup (GPU may be unstable): {e}")
        
        elapsed = (datetime.now() - start_time).total_seconds()
        logger.info(f"\n✅ Successfully completed in {elapsed/60:.1f} minutes")
        sys_module.stdout.flush()
        
        return {
            'combo_id': combo_id,
            'status': 'success',
            'elapsed_seconds': elapsed,
            'n_features_evaluated': len(token_results),
            'token_metrics': {k: float(v) for k, v in token_means.items()},
            'sentence_metrics': {k: float(v) for k, v in sentence_means.items()},
        }
        
    except Exception as e:
        error_msg = str(e)[:500]
        logger.error(f"❌ Failed: {error_msg}", exc_info=True)
        sys_module.stdout.flush()
        
        # Aggressive cleanup after failure
        try:
            if 'explanation_gen' in locals():
                explanation_gen.cleanup()
            if 'token_evaluator' in locals():
                token_evaluator.cleanup()
            if 'sentence_evaluator' in locals():
                sentence_evaluator.cleanup()
            if 'collector' in locals():
                collector.marked_sequence_data.clear()
                collector.high_max_sequences.clear()
                collector.low_max_sequences.clear()
        except:
            pass
        
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except (RuntimeError, Exception):
                pass
        
        return {
            'combo_id': combo_id,
            'status': 'failed',
            'error': error_msg
        }


async def run_all_evaluations(
    device: str = None,
    skip_existing: bool = True,
):
    """Run dominance-aware evaluation on all Benchmark Training GPT-2 SAEs."""
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"
    
    combinations = generate_all_combinations()
    model_name = BENCHMARK_CONFIG['model_name']
    
    # Track progress
    total_analyses = len(combinations)
    completed = []
    skipped = []
    failed = []
    
    # Model cache (reuse model across SAEs)
    model_cache = {}
    
    start_time = datetime.now()
    
    logger.info(f"\n{'='*70}")
    logger.info(f"DOMINANCE-AWARE AUTO-INTERPRETABILITY EVALUATION (BENCHMARK OPT-125M)")
    logger.info(f"{'='*70}")
    logger.info(f"Architectures: {len(BENCHMARK_CONFIG['architectures'])}")
    logger.info(f"Widths: {len(BENCHMARK_CONFIG['widths'])}")
    logger.info(f"Total SAEs: {total_analyses}")
    logger.info(f"Features per SAE: {N_FEATURES_PER_QUANTILE * 4} ({N_FEATURES_PER_QUANTILE} per quantile)")
    logger.info(f"Device: {device}")
    logger.info(f"Output: {output_base_dir}")
    logger.info(f"Skip existing: {skip_existing}")
    logger.info(f"{'='*70}\n")
    sys_module.stdout.flush()
    
    # Process each combination
    for combo in combinations:
        combo_id = combo['combo_id']
        
        # Check if output exists
        if skip_existing and check_output_exists(combo_id, output_base_dir):
            logger.info(f"⏭️  Skipping {combo_id} (output exists)")
            skipped.append({'combo_id': combo_id})
            continue
        
        # Run evaluation
        result = await run_evaluation_for_sae(
            combo_id=combo_id,
            device=device,
            output_base_dir=output_base_dir,
            model_cache=model_cache,
        )
        
        if result['status'] == 'success':
            completed.append(result)
        else:
            failed.append(result)
    
    # Cleanup model cache
    if model_name in model_cache and model_cache[model_name] is not None:
        logger.info(f"\n🧹 Cleaning up model cache...")
        del model_cache[model_name]
        model_cache[model_name] = None
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except (RuntimeError, Exception):
                pass
        logger.info(f"✓ Cleaned up model cache")
    
    elapsed_time = (datetime.now() - start_time).total_seconds()
    
    # Print summary
    logger.info("\n" + "="*70)
    logger.info("BATCH EVALUATION COMPLETE (BENCHMARK GPT-2)")
    logger.info("="*70)
    logger.info(f"\nTotal SAEs: {total_analyses}")
    logger.info(f"✅ Completed: {len(completed)}")
    logger.info(f"⏭️  Skipped: {len(skipped)}")
    logger.info(f"❌ Failed: {len(failed)}")
    logger.info(f"⏱️  Time elapsed: {elapsed_time/3600:.2f} hours ({elapsed_time/60:.1f} minutes)")
    
    if failed:
        logger.info("\nFailed evaluations:")
        for f in failed:
            logger.info(f"  - {f['combo_id']}: {f.get('error', 'Unknown error')}")
    
    # Save summary
    summary = {
        'total_analyses': total_analyses,
        'completed': len(completed),
        'skipped': len(skipped),
        'failed': len(failed),
        'elapsed_seconds': elapsed_time,
        'completed_details': completed,
        'failed_details': failed,
        'skipped_details': skipped
    }
    
    summary_file = output_base_dir / "batch_summary_benchmark_gpt2.json"
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    logger.info(f"\n✓ Saved batch summary to: {summary_file}")


def main():
    parser = argparse.ArgumentParser(
        description="Run dominance-aware evaluation on all Benchmark Training OPT-125M SAEs using offline model",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--device", type=str, default=None, help="Device to use (default: cuda:0)")
    parser.add_argument("--skip-existing", action="store_true", default=True, help="Skip SAEs with existing results")
    parser.add_argument("--no-skip-existing", dest="skip_existing", action="store_false", help="Don't skip existing results")
    
    args = parser.parse_args()
    
    asyncio.run(run_all_evaluations(
        device=args.device,
        skip_existing=args.skip_existing,
    ))


if __name__ == "__main__":
    main()
