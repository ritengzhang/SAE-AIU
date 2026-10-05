#!/usr/bin/env python3
"""
Completeness Analysis Functions

Core helper functions for calculating completeness metrics for SAEs:
1. Mean Squared Error (MSE) - direct reconstruction fidelity
2. Cross-Entropy Loss Recovery - model performance preservation
3. KL Divergence - model behavior distribution preservation

Results are saved with normalized scores (ce_loss_score, kl_div_score).
"""

import numpy as np
import torch
from typing import Dict, Optional
from tqdm.auto import tqdm
import time
from functools import partial


def calculate_mse(
    original_activations: np.ndarray,
    reconstructed_activations: np.ndarray
) -> float:
    """
    Calculate Mean Squared Error between original and reconstructed activations.
    
    Args:
        original_activations: (n_samples, d_in) array of original activations
        reconstructed_activations: (n_samples, d_in) array of SAE reconstructions
        
    Returns:
        MSE value (lower is better)
    """
    mse = np.mean((original_activations - reconstructed_activations) ** 2)
    return float(mse)


def calculate_kl_divergence(
    original_logits: torch.Tensor,
    new_logits: torch.Tensor
) -> torch.Tensor:
    """
    Calculate KL divergence between two logit distributions.
    
    Args:
        original_logits: (batch, seq_len, vocab_size) original model logits
        new_logits: (batch, seq_len, vocab_size) new model logits
        
    Returns:
        KL divergence per token position: (batch, seq_len)
    """
    original_probs = torch.nn.functional.softmax(original_logits, dim=-1)
    log_original_probs = torch.log(original_probs + 1e-10)  # Add small epsilon for numerical stability
    new_probs = torch.nn.functional.softmax(new_logits, dim=-1)
    log_new_probs = torch.log(new_probs + 1e-10)
    
    kl_div = original_probs * (log_original_probs - log_new_probs)
    kl_div = kl_div.sum(dim=-1)  # Sum over vocabulary dimension
    return kl_div


def get_reconstruction_losses(
    sae,
    model,
    batch_tokens: torch.Tensor,
    hook_name: str,
    hook_head_index: Optional[int] = None,
    ignore_tokens: set = set(),
) -> Dict[str, torch.Tensor]:
    """
    Compute reconstruction losses for a batch of tokens.
    
    This function:
    1. Runs model with original activations (baseline)
    2. Runs model with SAE reconstructions replacing activations at hook
    3. Runs model with zero activations (ablation baseline)
    4. Computes CE loss and KL divergence for each
    
    Args:
        sae: SAE object with encode() and decode() methods
        model: HookedTransformer model
        batch_tokens: (batch_size, seq_len) token tensor
        hook_name: Hook name where SAE operates (e.g., 'blocks.6.hook_mlp_out')
        hook_head_index: Optional head index for attention hooks
        ignore_tokens: Set of token IDs to ignore in computation
        
    Returns:
        Dictionary with:
        - 'original_logits': baseline logits
        - 'original_ce_loss': baseline CE loss per token
        - 'recons_logits': logits with SAE reconstructions
        - 'recons_ce_loss': CE loss with SAE reconstructions per token
        - 'ablation_logits': logits with zero activations
        - 'ablation_ce_loss': CE loss with zero activations per token
        - 'kl_div_with_sae': KL divergence (original vs SAE)
        - 'kl_div_with_ablation': KL divergence (original vs ablation)
    """
    device = next(sae.parameters()).device if hasattr(sae, 'parameters') else batch_tokens.device
    
    # Get baseline (original model)
    original_logits, original_ce_loss = model(
        batch_tokens,
        return_type="both",
        loss_per_token=True
    )
    
    # Create mask for ignored tokens
    if len(ignore_tokens) > 0:
        mask = torch.logical_not(
            torch.any(
                torch.stack([batch_tokens == token for token in ignore_tokens], dim=0),
                dim=0,
            )
        )
    else:
        mask = torch.ones_like(batch_tokens, dtype=torch.bool)
    
    # Define hook functions for SAE reconstruction
    def standard_replacement_hook(activations: torch.Tensor, hook):
        original_device = activations.device
        activations = activations.to(device)
        
        # Encode and decode through SAE
        reconstructed = sae.decode(sae.encode(activations)).to(activations.dtype)
        
        # Apply mask to keep original activations for ignored tokens
        reconstructed = torch.where(
            mask[..., None], reconstructed, activations
        )
        
        return reconstructed.to(original_device)
    
    def all_head_replacement_hook(activations: torch.Tensor, hook):
        original_device = activations.device
        activations = activations.to(device)
        
        # Flatten head dimension, encode/decode, then reshape
        new_activations = sae.decode(sae.encode(activations.flatten(-2, -1))).to(activations.dtype)
        new_activations = new_activations.reshape(activations.shape)
        
        # Apply mask
        new_activations = torch.where(
            mask[..., None, None], new_activations, activations
        )
        
        return new_activations.to(original_device)
    
    def single_head_replacement_hook(activations: torch.Tensor, hook):
        original_device = activations.device
        activations = activations.to(device)
        
        new_activations = activations.clone()
        head_activations = sae.decode(sae.encode(activations[:, :, hook_head_index])).to(activations.dtype)
        
        masked_head = torch.where(
            mask[..., None], head_activations, activations[:, :, hook_head_index]
        )
        new_activations[:, :, hook_head_index] = masked_head
        
        return new_activations.to(original_device)
    
    # Define hook functions for zero ablation
    def standard_zero_ablate_hook(activations: torch.Tensor, hook):
        return torch.zeros_like(activations)
    
    def single_head_zero_ablate_hook(activations: torch.Tensor, hook):
        activations[:, :, hook_head_index] = torch.zeros_like(activations[:, :, hook_head_index])
        return activations
    
    # Choose appropriate hooks based on hook_name
    has_head_dim_key_substrings = ["hook_q", "hook_k", "hook_v", "hook_z"]
    if any(substring in hook_name for substring in has_head_dim_key_substrings):
        if hook_head_index is None:
            replacement_hook = all_head_replacement_hook
            zero_ablate_hook = standard_zero_ablate_hook
        else:
            replacement_hook = single_head_replacement_hook
            zero_ablate_hook = single_head_zero_ablate_hook
    else:
        replacement_hook = standard_replacement_hook
        zero_ablate_hook = standard_zero_ablate_hook
    
    # Run model with SAE reconstructions
    recons_logits, recons_ce_loss = model.run_with_hooks(
        batch_tokens,
        return_type="both",
        fwd_hooks=[(hook_name, replacement_hook)],
        loss_per_token=True,
    )
    
    # Run model with zero ablation
    ablation_logits, ablation_ce_loss = model.run_with_hooks(
        batch_tokens,
        return_type="both",
        fwd_hooks=[(hook_name, zero_ablate_hook)],
        loss_per_token=True,
    )
    
    # Compute KL divergences
    kl_div_with_sae = calculate_kl_divergence(original_logits, recons_logits)
    kl_div_with_ablation = calculate_kl_divergence(original_logits, ablation_logits)
    
    return {
        'original_logits': original_logits,
        'original_ce_loss': original_ce_loss,
        'recons_logits': recons_logits,
        'recons_ce_loss': recons_ce_loss,
        'ablation_logits': ablation_logits,
        'ablation_ce_loss': ablation_ce_loss,
        'kl_div_with_sae': kl_div_with_sae,
        'kl_div_with_ablation': kl_div_with_ablation,
    }


def calculate_downstream_reconstruction_metrics(
    sae,
    model,
    activation_store,
    hook_name: str,
    hook_head_index: Optional[int] = None,
    n_batches: int = 30,
    eval_batch_size_prompts: int = 8,
    ignore_tokens: set = set(),
) -> Dict[str, float]:
    """
    Calculate downstream reconstruction metrics (CE loss and KL divergence).
    
    Processes batches from activation store and computes:
    - ce_loss_without_sae: Baseline cross-entropy loss
    - ce_loss_with_sae: CE loss with SAE reconstructions
    - ce_loss_with_ablation: CE loss with zero activations
    - ce_loss_score: Normalized score (higher is better, 1.0 = perfect)
    - kl_div_with_sae: KL divergence with SAE reconstructions
    - kl_div_with_ablation: KL divergence with zero activations
    - kl_div_score: Normalized score (higher is better, 1.0 = perfect)
    
    Args:
        sae: SAE object
        model: HookedTransformer model
        activation_store: ActivationsStore object
        hook_name: Hook name where SAE operates
        hook_head_index: Optional head index for attention hooks
        n_batches: Number of batches to process
        eval_batch_size_prompts: Batch size for evaluation
        ignore_tokens: Set of token IDs to ignore
        
    Returns:
        Dictionary with all metrics
    """
    print(f"  Calculating downstream reconstruction metrics...")
    print(f"  Processing {n_batches} batches...")
    
    start_time = time.time()
    
    # Accumulators
    ce_loss_without_sae_list = []
    ce_loss_with_sae_list = []
    ce_loss_with_ablation_list = []
    kl_div_with_sae_list = []
    kl_div_with_ablation_list = []
    
    for batch_idx in tqdm(range(n_batches), desc="Reconstruction batches", leave=False):
        batch_tokens = activation_store.get_batch_tokens(eval_batch_size_prompts)
        
        # Get reconstruction losses for this batch
        batch_metrics = get_reconstruction_losses(
            sae=sae,
            model=model,
            batch_tokens=batch_tokens,
            hook_name=hook_name,
            hook_head_index=hook_head_index,
            ignore_tokens=ignore_tokens,
        )
        
        # Apply mask if needed
        if len(ignore_tokens) > 0:
            mask = torch.logical_not(
                torch.any(
                    torch.stack([batch_tokens == token for token in ignore_tokens], dim=0),
                    dim=0,
                )
            )
            # CE loss might be missing last token, adjust mask
            if batch_metrics['original_ce_loss'].shape[1] != mask.shape[1]:
                mask = mask[:, :-1]
            
            ce_loss_without_sae_list.append(batch_metrics['original_ce_loss'][mask])
            ce_loss_with_sae_list.append(batch_metrics['recons_ce_loss'][mask])
            ce_loss_with_ablation_list.append(batch_metrics['ablation_ce_loss'][mask])
            kl_div_with_sae_list.append(batch_metrics['kl_div_with_sae'][mask])
            kl_div_with_ablation_list.append(batch_metrics['kl_div_with_ablation'][mask])
        else:
            ce_loss_without_sae_list.append(batch_metrics['original_ce_loss'].flatten())
            ce_loss_with_sae_list.append(batch_metrics['recons_ce_loss'].flatten())
            ce_loss_with_ablation_list.append(batch_metrics['ablation_ce_loss'].flatten())
            kl_div_with_sae_list.append(batch_metrics['kl_div_with_sae'].flatten())
            kl_div_with_ablation_list.append(batch_metrics['kl_div_with_ablation'].flatten())
    
    # Concatenate and compute means
    ce_loss_without_sae = torch.cat(ce_loss_without_sae_list).mean().item()
    ce_loss_with_sae = torch.cat(ce_loss_with_sae_list).mean().item()
    ce_loss_with_ablation = torch.cat(ce_loss_with_ablation_list).mean().item()
    kl_div_with_sae = torch.cat(kl_div_with_sae_list).mean().item()
    kl_div_with_ablation = torch.cat(kl_div_with_ablation_list).mean().item()
    
    # Compute normalized scores
    ce_loss_score = (ce_loss_with_ablation - ce_loss_with_sae) / (ce_loss_with_ablation - ce_loss_without_sae)
    kl_div_score = (kl_div_with_ablation - kl_div_with_sae) / kl_div_with_ablation
    
    elapsed_time = time.time() - start_time
    print(f"  ⏱️  Downstream metrics (CE loss + KL div) took {elapsed_time:.1f}s ({elapsed_time/60:.1f} min)")
    
    return {
        'ce_loss_without_sae': ce_loss_without_sae,
        'ce_loss_with_sae': ce_loss_with_sae,
        'ce_loss_with_ablation': ce_loss_with_ablation,
        'ce_loss_score': ce_loss_score,
        'kl_div_with_sae': kl_div_with_sae,
        'kl_div_with_ablation': kl_div_with_ablation,
        'kl_div_score': kl_div_score,
    }


def calculate_completeness_metrics(
    sae,
    model,
    activations: np.ndarray,
    activation_store,
    hook_name: str,
    hook_head_index: Optional[int] = None,
    n_batches: int = 30,
    eval_batch_size_prompts: int = 8,
    ignore_tokens: set = set(),
) -> Dict:
    """
    Calculate all completeness metrics for an SAE.
    
    Computes:
    1. MSE - direct reconstruction fidelity
    2. CE Loss Recovery - model performance preservation
    3. KL Divergence - model behavior distribution preservation
    
    Args:
        sae: SAE object with encode() and decode() methods
        model: HookedTransformer model
        activations: (n_samples, d_in) array of input activations
        activation_store: ActivationsStore for downstream metrics
        hook_name: Hook name where SAE operates
        hook_head_index: Optional head index for attention hooks
        n_batches: Number of batches for downstream metrics
        eval_batch_size_prompts: Batch size for downstream evaluation
        ignore_tokens: Set of token IDs to ignore
        
    Returns:
        Dictionary with all completeness metrics
    """
    print(f"\n📊 Calculating completeness metrics...")
    print(f"  Processing {activations.shape[0]:,} activation vectors...")
    
    device = next(sae.parameters()).device if hasattr(sae, 'parameters') else 'cpu'
    
    # Step 1: Calculate MSE (direct reconstruction)
    print(f"\n[1/3] Calculating MSE...")
    mse_start_time = time.time()
    batch_size = 10000  # Process in chunks to avoid OOM
    mse_list = []
    
    for i in tqdm(range(0, activations.shape[0], batch_size), desc="MSE batches", leave=False):
        batch_acts = activations[i:i+batch_size]
        batch_tensor = torch.from_numpy(batch_acts).to(device)
        
        with torch.no_grad():
            # Encode and decode through SAE
            batch_recons = sae.decode(sae.encode(batch_tensor)).cpu().numpy()
        
        batch_mse = calculate_mse(batch_acts, batch_recons)
        mse_list.append(batch_mse)
    
    mse = np.mean(mse_list)
    mse_elapsed = time.time() - mse_start_time
    print(f"  ✓ MSE: {mse:.6e} (took {mse_elapsed:.1f}s)")
    
    # Step 2 & 3: Calculate downstream metrics (CE loss and KL divergence)
    # COMMENTED OUT - Only computing MSE for now
    # print(f"\n[2-3/3] Calculating downstream reconstruction metrics...")
    # downstream_metrics = calculate_downstream_reconstruction_metrics(
    #     sae=sae,
    #     model=model,
    #     activation_store=activation_store,
    #     hook_name=hook_name,
    #     hook_head_index=hook_head_index,
    #     n_batches=n_batches,
    #     eval_batch_size_prompts=eval_batch_size_prompts,
    #     ignore_tokens=ignore_tokens,
    # )
    # 
    # print(f"  ✓ CE Loss Score: {downstream_metrics['ce_loss_score']:.4f}")
    # print(f"  ✓ KL Div Score: {downstream_metrics['kl_div_score']:.4f}")
    
    # Combine all metrics (only MSE for now)
    results = {
        'mse': mse,
        # **downstream_metrics,  # COMMENTED OUT
        'n_samples': int(activations.shape[0]),
        'd_in': int(activations.shape[1]),
        'd_sae': int(sae.cfg.d_sae) if hasattr(sae, 'cfg') else None,
    }
    
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
    
    Reuses the implementation from sparsity_and_dead_neuron module.
    
    Args:
        model: HookedTransformer model
        sae: SAE object with cfg attribute
        metadata: Metadata dictionary containing model_name, hook_name, dataset_path, etc.
        device: Device to use (default: auto-detect cuda:0)
        width_name: Width name for adaptive buffer sizing (e.g., '16k', '65k')
    
    Returns:
        ActivationsStore ready for collecting activations
    """
    # Import from sparsity module to avoid duplication
    import sys
    from pathlib import Path
    
    parent_dir = Path(__file__).parent.parent.parent
    sys.path.insert(0, str(parent_dir))
    
    from aiu_bench.sparsity_dead_neuron.sparsity_and_deadneuron_analysis import (
        setup_activations_store_for_benchmark as _setup_store
    )
    
    return _setup_store(model, sae, metadata, device, width_name)
