#!/usr/bin/env python3
"""
Sparsity and Dead Neuron Analysis Functions

Core helper functions for calculating sparsity and dead neuron metrics for SAEs:
1. Average active features (simple counting - features with activation > threshold)
2. Effective active features - count features that add up to X% (75%, 90%) of total activation
3. Dead neuron counting - features that never fire or fire very rarely
4. Common utilities for activation collection and store setup
"""

import numpy as np
import torch
from typing import Dict, List, Tuple, Optional
from tqdm.auto import tqdm
import time


def calculate_average_active_features(
    feature_activations: np.ndarray,
    activation_threshold: float = 1e-5
) -> float:
    """
    Calculate average number of active features per input.
    
    A feature is considered "active" if its absolute activation value > threshold.
    
    Args:
        feature_activations: (n_inputs, d_sae) array of feature activations
        activation_threshold: Threshold for considering a feature active (default: 1e-5)
        
    Returns:
        Average number of active features per input
    """
    # Process in chunks to avoid OOM for large arrays
    chunk_size = 10000  # Process 10k samples at a time
    n_inputs = feature_activations.shape[0]
    total_active = 0.0
    
    for i in range(0, n_inputs, chunk_size):
        chunk = feature_activations[i:i+chunk_size]
        # Count active features for each input (absolute value > threshold)
        active_mask = np.abs(chunk) > activation_threshold
        n_active_per_input = np.sum(active_mask, axis=1)  # (chunk_size,)
        total_active += np.sum(n_active_per_input)
    
    # Return average
    return float(total_active / n_inputs)


def calculate_effective_active_features(
    feature_activations: np.ndarray,
    percentile: float = 90.0,
    activation_threshold: float = 1e-5
) -> float:
    """
    Calculate average number of "effective active" features per input.
    
    For each input, we count how many features (when sorted by absolute activation)
    are needed to sum up to the specified percentile of total absolute activation.
    
    This avoids counting features with tiny activations (e.g., 0.00001) as "active"
    when they contribute negligibly to the total.
    
    Args:
        feature_activations: (n_inputs, d_sae) array of feature activations
        percentile: Percentile of total activation to consider (default: 90.0 for 90%)
        activation_threshold: Minimum threshold for a feature to be considered (default: 1e-5)
        
    Returns:
        Average number of effective active features per input
    """
    n_inputs = feature_activations.shape[0]
    n_effective_active_list = []
    
    for i in range(n_inputs):
        activations = feature_activations[i]  # (d_sae,)
        
        # Filter out features below threshold
        mask = np.abs(activations) > activation_threshold
        if not mask.any():
            n_effective_active_list.append(0)
            continue
        
        filtered_activations = activations[mask]
        
        # Calculate total absolute activation
        total_abs_activation = np.sum(np.abs(filtered_activations))
        
        if total_abs_activation == 0:
            n_effective_active_list.append(0)
            continue
        
        # Sort by absolute activation (descending)
        sorted_indices = np.argsort(np.abs(filtered_activations))[::-1]
        sorted_abs_activations = np.abs(filtered_activations[sorted_indices])
        
        # Calculate target sum (percentile of total)
        target_sum = total_abs_activation * (percentile / 100.0)
        
        # Count how many features are needed to reach target sum
        cumulative_sum = np.cumsum(sorted_abs_activations)
        # Find first index where cumulative_sum >= target_sum
        n_needed = np.searchsorted(cumulative_sum, target_sum, side='left') + 1
        
        # Ensure at least 1 if there are any activations, and at most the number of filtered features
        n_needed = max(1, min(n_needed, len(filtered_activations)))
        
        n_effective_active_list.append(n_needed)
    
    return float(np.mean(n_effective_active_list))


def calculate_dead_neuron_metrics(
    feature_activations: np.ndarray,
    activation_threshold: float = 1e-5,
    firing_frequency_thresholds: List[float] = [0.0, 0.01, 0.1]
) -> Dict:
    """
    Calculate dead neuron metrics for an SAE.
    
    Dead neurons are features that either:
    1. Never fire (0% firing frequency)
    2. Fire very rarely (below specified percentage thresholds)
    
    Args:
        feature_activations: (n_inputs, d_sae) array of feature activations
        activation_threshold: Threshold for considering a feature active (default: 1e-5)
        firing_frequency_thresholds: List of firing frequency thresholds as percentages 
                                     (default: [0.0, 0.01, 0.1] for 0%, 0.01%, 0.1%)
        
    Returns:
        Dictionary with dead neuron metrics:
        {
            'n_features': int,
            'n_total_tokens': int,
            'dead_neurons_never_fire': int,  # Features that never fire (0%)
            'dead_neurons_very_rare': int,   # Features firing < 0.01%
            'dead_neurons_rare': int,        # Features firing < 0.1%
            'dead_neuron_proportion_never': float,
            'dead_neuron_proportion_very_rare': float,
            'dead_neuron_proportion_rare': float,
            'firing_frequencies': list,  # (d_sae,) list of firing frequencies per feature (converted from np.ndarray for JSON serialization)
            'firing_frequency_stats': dict,    # Mean, median, quartiles of firing frequencies
        }
    """
    n_inputs, d_sae = feature_activations.shape
    n_total_tokens = n_inputs
    
    # Count firing occurrences per feature across all inputs
    # Process in chunks to avoid OOM
    chunk_size = 10000
    firing_counts = np.zeros(d_sae, dtype=np.int64)
    
    for i in range(0, n_inputs, chunk_size):
        chunk = feature_activations[i:i+chunk_size]  # (chunk_size, d_sae)
        # Count how many inputs have this feature active (absolute value > threshold)
        active_mask = np.abs(chunk) > activation_threshold  # (chunk_size, d_sae)
        firing_counts += np.sum(active_mask, axis=0)  # Sum across inputs for each feature
    
    # Calculate firing frequencies (percentage of tokens where feature fired)
    firing_frequencies = (firing_counts / n_total_tokens) * 100.0  # (d_sae,) in percentage
    
    # Count dead neurons at different thresholds
    # Sort thresholds in ascending order
    sorted_thresholds = sorted(firing_frequency_thresholds)
    
    dead_neuron_counts = {}
    dead_neuron_proportions = {}
    
    for threshold in sorted_thresholds:
        count = np.sum(firing_frequencies < threshold)
        dead_neuron_counts[f'dead_neurons_{threshold}'] = int(count)
        dead_neuron_proportions[f'dead_neuron_proportion_{threshold}'] = float(count / d_sae)
    
    # Calculate firing frequency statistics
    firing_frequency_stats = {
        'mean': float(np.mean(firing_frequencies)),
        'median': float(np.median(firing_frequencies)),
        'std': float(np.std(firing_frequencies)),
        'min': float(np.min(firing_frequencies)),
        'max': float(np.max(firing_frequencies)),
        'q25': float(np.percentile(firing_frequencies, 25)),
        'q75': float(np.percentile(firing_frequencies, 75)),
        'q90': float(np.percentile(firing_frequencies, 90)),
        'q95': float(np.percentile(firing_frequencies, 95)),
        'q99': float(np.percentile(firing_frequencies, 99)),
    }
    
    # Create results dictionary with readable names for common thresholds
    # Convert numpy array to list for JSON serialization
    results = {
        'n_features': int(d_sae),
        'n_total_tokens': int(n_total_tokens),
        'firing_frequencies': firing_frequencies.tolist(),  # Convert to list for JSON serialization
        'firing_frequency_stats': firing_frequency_stats,
    }
    
    # Add threshold-specific counts with readable names
    if 0.0 in sorted_thresholds:
        results['dead_neurons_never_fire'] = dead_neuron_counts['dead_neurons_0.0']
        results['dead_neuron_proportion_never'] = dead_neuron_proportions['dead_neuron_proportion_0.0']
    
    if 0.01 in sorted_thresholds:
        results['dead_neurons_very_rare'] = dead_neuron_counts['dead_neurons_0.01']
        results['dead_neuron_proportion_very_rare'] = dead_neuron_proportions['dead_neuron_proportion_0.01']
    
    if 0.1 in sorted_thresholds:
        results['dead_neurons_rare'] = dead_neuron_counts['dead_neurons_0.1']
        results['dead_neuron_proportion_rare'] = dead_neuron_proportions['dead_neuron_proportion_0.1']
    
    # Also include all thresholds for flexibility
    results['dead_neuron_counts_by_threshold'] = dead_neuron_counts
    results['dead_neuron_proportions_by_threshold'] = dead_neuron_proportions
    
    return results


def calculate_sparsity_metrics(
    sae,
    activations: np.ndarray,
    n_batches: int = 50,
    activation_threshold: float = 1e-5,
    percentiles: List[float] = [75.0, 90.0],
    include_dead_neuron_metrics: bool = True,
    firing_frequency_thresholds: List[float] = [0.0, 0.01, 0.1]
) -> Dict:
    """
    Calculate comprehensive sparsity and dead neuron metrics for an SAE.
    
    For each input activation:
    1. Encode through SAE to get feature activations
    2. Calculate average active features (simple counting)
    3. Calculate effective active features for each percentile (75%, 90%)
    4. Calculate dead neuron metrics (features that never fire or fire rarely)
    
    Args:
        sae: SAE object with encode() method
        activations: (n_samples, d_in) array of input activations
        n_batches: Number of batches to process (for memory efficiency)
        activation_threshold: Threshold for considering a feature active
        percentiles: List of percentiles for effective active calculation (default: [75.0, 90.0])
        include_dead_neuron_metrics: Whether to calculate dead neuron metrics (default: True)
        firing_frequency_thresholds: List of firing frequency thresholds as percentages 
                                     for dead neuron counting (default: [0.0, 0.01, 0.1])
        
    Returns:
        Dictionary with sparsity and dead neuron metrics:
        {
            'sparsity_metrics': {
                'average_active_features': float,
                'effective_active_features_75': float,
                'effective_active_features_90': float,
            },
            'dead_neuron_metrics': {
                'dead_neurons_never_fire': int,
                'dead_neurons_very_rare': int,
                'dead_neurons_rare': int,
                'dead_neuron_proportion_never': float,
                'dead_neuron_proportion_very_rare': float,
                'dead_neuron_proportion_rare': float,
                'firing_frequency_stats': dict,
                ...
            },
            'n_samples': int,
            'd_sae': int,
            'd_in': int
        }
    """
    print(f"  Calculating sparsity and dead neuron metrics...")
    print(f"  Processing {activations.shape[0]:,} activation vectors...")
    
    device = next(sae.parameters()).device if hasattr(sae, 'parameters') else 'cpu'
    
    # Process in batches to avoid OOM - DON'T concatenate everything!
    batch_size = max(1, activations.shape[0] // n_batches)
    
    # Accumulators for sparsity metrics
    total_average_active = 0.0
    total_effective_active_75 = 0.0
    total_effective_active_90 = 0.0
    n_processed = 0
    
    # Accumulator for dead neuron metrics (need all feature activations)
    all_feature_activations = []  # Will collect all batches for dead neuron analysis
    
    # Get d_sae from first batch
    first_batch = activations[:min(batch_size, activations.shape[0])]
    first_tensor = torch.from_numpy(first_batch).to(device)
    with torch.no_grad():
        first_feature_acts = sae.encode(first_tensor).cpu().numpy()
    d_sae = first_feature_acts.shape[1]
    
    # Process each batch and accumulate statistics
    for i in tqdm(range(0, activations.shape[0], batch_size), desc="Encoding activations", leave=False):
        batch_activations = activations[i:i+batch_size]
        batch_tensor = torch.from_numpy(batch_activations).to(device)
        
        with torch.no_grad():
            batch_feature_acts = sae.encode(batch_tensor).cpu().numpy()  # (batch_size, d_sae)
        
        batch_n = batch_feature_acts.shape[0]
        
        # Calculate sparsity metrics for this batch
        batch_avg_active = calculate_average_active_features(
            batch_feature_acts,
            activation_threshold=activation_threshold
        )
        
        batch_eff_active_75 = calculate_effective_active_features(
            batch_feature_acts,
            percentile=75.0,
            activation_threshold=activation_threshold
        )
        
        batch_eff_active_90 = calculate_effective_active_features(
            batch_feature_acts,
            percentile=90.0,
            activation_threshold=activation_threshold
        )
        
        # Accumulate sparsity metrics (weighted by batch size)
        total_average_active += batch_avg_active * batch_n
        total_effective_active_75 += batch_eff_active_75 * batch_n
        total_effective_active_90 += batch_eff_active_90 * batch_n
        n_processed += batch_n
        
        # Collect feature activations for dead neuron analysis
        if include_dead_neuron_metrics:
            all_feature_activations.append(batch_feature_acts)
        
        # Clear batch from memory (but keep reference in all_feature_activations)
        del batch_tensor
    
    print(f"  ✓ Encoded {n_processed:,} activations")
    
    # Calculate final sparsity averages
    sparsity_results = {
        'average_active_features': total_average_active / n_processed,
        'effective_active_features_75': total_effective_active_75 / n_processed,
        'effective_active_features_90': total_effective_active_90 / n_processed,
    }
    
    # Calculate dead neuron metrics if requested
    dead_neuron_results = {}
    if include_dead_neuron_metrics:
        print(f"  Calculating dead neuron metrics...")
        # Concatenate all feature activations for dead neuron analysis
        all_feature_acts = np.concatenate(all_feature_activations, axis=0)  # (n_samples, d_sae)
        dead_neuron_results = calculate_dead_neuron_metrics(
            all_feature_acts,
            activation_threshold=activation_threshold,
            firing_frequency_thresholds=firing_frequency_thresholds
        )
        print(f"  ✓ Dead neuron metrics calculated")
        # Clear large array
        del all_feature_acts
        del all_feature_activations
    
    # Combine results - include both nested structure and flat structure for backward compatibility
    results = {
        # Nested structure (new)
        'sparsity_metrics': sparsity_results,
        'n_samples': n_processed,
        'd_sae': d_sae,
        'd_in': int(activations.shape[1])
    }
    
    # Add dead neuron metrics if calculated
    if dead_neuron_results:
        results['dead_neuron_metrics'] = dead_neuron_results
    
    # Also include flat structure for backward compatibility with existing runners
    results.update({
        'average_active_features': sparsity_results['average_active_features'],
        'effective_active_features_75': sparsity_results['effective_active_features_75'],
        'effective_active_features_90': sparsity_results['effective_active_features_90'],
    })
    
    return results


def collect_activations(store, n_batches: int = 15):
    """
    Collect activations from the activation store.
    
    This is a common utility function used across all SAE family runners.
    
    Args:
        store: ActivationsStore object with next_batch() method
        n_batches: Number of batches to collect
        
    Returns:
        numpy array of shape (n_samples, d_in) containing all collected activations
    """
    print(f"\n📊 Collecting {n_batches} batches of activations...")
    all_activations = []
    
    for batch_idx in range(n_batches):
        batch_activations = store.next_batch()
        all_activations.append(batch_activations.cpu().numpy())
    
    all_activations = np.concatenate(all_activations, axis=0)
    print(f"✓ Collected {all_activations.shape[0]:,} activation vectors")
    
    return all_activations


def setup_activations_store_for_benchmark(
    model,
    sae,
    metadata: dict,
    device: str = None,
    width_name: str = None
):
    """
    Setup activations store for benchmark training SAEs.
    
    This is a common utility function used across benchmark training runners
    (opt125m, gpt2, tinystories).
    
    Args:
        model: HookedTransformer model
        sae: SAE object with cfg attribute
        metadata: Metadata dictionary containing model_name, hook_name, dataset_path, etc.
        device: Device to use (default: auto-detect cuda:0)
        width_name: Width name for adaptive buffer sizing (e.g., '16k', '65k')
    
    Returns:
        ActivationsStore ready for collecting activations
    """
    # Import here to avoid circular dependencies
    import sys
    from pathlib import Path
    
    # Add SAELens to path if needed
    parent_dir = Path(__file__).parent.parent.parent
    SAELENS_PATH = parent_dir / "SAELens-main"
    if str(SAELENS_PATH) not in sys.path:
        sys.path.insert(0, str(SAELENS_PATH))
    
    from sae_lens.training.activations_store import ActivationsStore
    from sae_lens.config import LanguageModelSAERunnerConfig
    from sae_lens.saes.standard_sae import StandardTrainingSAEConfig
    
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU is not available. GPU is required for activation collection.")
        device = "cuda:0"
    
    print("Setting up activations store...")
    
    dummy_sae_cfg = StandardTrainingSAEConfig(
        d_in=sae.cfg.d_in,
        d_sae=sae.cfg.d_sae,
        device=device,
        dtype="float32",
    )
    
    # Adaptive buffer sizing based on SAE width
    if width_name == '16k':
        n_batches_in_buffer = 8
        store_batch_size_prompts = 4
        print(f"  📉 Using reduced buffer for 16k: {n_batches_in_buffer} batches, {store_batch_size_prompts} prompts/batch")
    elif width_name == '65k':
        n_batches_in_buffer = 8
        store_batch_size_prompts = 4
        print(f"  📉 Using reduced buffer for 65k: {n_batches_in_buffer} batches, {store_batch_size_prompts} prompts/batch")
    else:
        n_batches_in_buffer = 16
        store_batch_size_prompts = 8
    
    # Get hook_name, using default if 'unknown' or missing
    hook_name = metadata.get('hook_name', 'blocks.6.hook_mlp_out')
    if hook_name == 'unknown':
        # Construct from LLM config if available
        try:
            from benchmark_config import LLM_CONFIGS
            llm_name = metadata.get('llm_name', metadata.get('model_name', 'gpt2'))
            if llm_name in LLM_CONFIGS:
                llm_config = LLM_CONFIGS[llm_name]
                layer = metadata.get('layer') or llm_config.get('middle_layer', 6)
                hook_template = llm_config.get('hook_name_template', 'blocks.{layer}.hook_mlp_out')
                hook_name = hook_template.format(layer=layer)
            else:
                hook_name = 'blocks.6.hook_mlp_out'  # Default fallback
        except ImportError:
            hook_name = 'blocks.6.hook_mlp_out'  # Default fallback
    
    cfg = LanguageModelSAERunnerConfig(
        sae=dummy_sae_cfg,
        model_name=metadata.get('model_name', 'gpt2'),
        hook_name=hook_name,
        dataset_path=metadata.get('dataset_path', 'apollo-research/Skylion007-openwebtext-tokenizer-gpt2'),
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
                print(f"     Retrying in 5 seconds...")
                time.sleep(5)
            else:
                print(f"  ❌ Failed after {max_retries} attempts")
                raise
