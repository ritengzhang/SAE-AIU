#!/usr/bin/env python3
"""
Layer-by-Layer Evolution of Residual Stream Distribution

This script analyzes how the angular distribution of activations evolves
across ALL layers of a transformer model.

For each model/dataset pair:
1. Collect activations from ALL layers (residual stream after each block)
2. For each layer: unit normalize, compute mean direction, get angular distances
3. Visualize the distribution evolution across layers
4. Show min/max/median/mean angular distance statistics per layer

This reveals how the residual stream geometry changes as information flows
through the network.
"""

import sys
import os
import argparse
from pathlib import Path
# Import for model loading
# SAELens should be in a relative path or set via SAELENS_PATH environment variable
saelens_path = os.environ.get("SAELENS_PATH", str(Path(__file__).parent.parent.parent.parent / "SAELens-main"))
sys.path.append(saelens_path)
from transformer_lens import HookedTransformer
from transformer_lens.utils import tokenize_and_concatenate
from datasets import load_dataset

import json
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple
from tqdm import tqdm
from collections import defaultdict
from within_shell_uniformity import (
    compute_pairwise_uniformity,
    plot_within_shell_uniformity,
    plot_pairwise_angle_distributions
)
from pairwise_torus_detection import (
    compute_pairwise_angular_distances,
    plot_pairwise_torus_detection
)

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 150

# Experiment configurations: model/dataset pairs
# HIGH SAMPLE COUNT: 4-5x increase for definitive statistical conclusions
EXPERIMENTS = [
    # GPT-2 variants - all sizes, all datasets
    {"model": "distilgpt2", "dataset": "NeelNanda/pile-10k", "num_batches": 150, "batch_size": 16},
    {"model": "distilgpt2", "dataset": "NeelNanda/code-10k", "num_batches": 150, "batch_size": 16},
    {"model": "distilgpt2", "dataset": "NeelNanda/c4-10k", "num_batches": 150, "batch_size": 16},
    {"model": "distilgpt2", "dataset": "NeelNanda/wiki-10k", "num_batches": 150, "batch_size": 16},
    
    {"model": "gpt2-small", "dataset": "NeelNanda/pile-10k", "num_batches": 150, "batch_size": 16},
    {"model": "gpt2-small", "dataset": "NeelNanda/code-10k", "num_batches": 150, "batch_size": 16},
    {"model": "gpt2-small", "dataset": "NeelNanda/c4-10k", "num_batches": 150, "batch_size": 16},
    {"model": "gpt2-small", "dataset": "NeelNanda/wiki-10k", "num_batches": 150, "batch_size": 16},
    
    {"model": "gpt2-medium", "dataset": "NeelNanda/pile-10k", "num_batches": 120, "batch_size": 12},
    {"model": "gpt2-medium", "dataset": "NeelNanda/code-10k", "num_batches": 120, "batch_size": 12},
    {"model": "gpt2-medium", "dataset": "NeelNanda/c4-10k", "num_batches": 120, "batch_size": 12},
    {"model": "gpt2-medium", "dataset": "NeelNanda/wiki-10k", "num_batches": 120, "batch_size": 12},
    
    {"model": "gpt2-large", "dataset": "NeelNanda/pile-10k", "num_batches": 120, "batch_size": 8},
    {"model": "gpt2-large", "dataset": "NeelNanda/code-10k", "num_batches": 120, "batch_size": 8},
    {"model": "gpt2-large", "dataset": "NeelNanda/c4-10k", "num_batches": 120, "batch_size": 8},
    {"model": "gpt2-large", "dataset": "NeelNanda/wiki-10k", "num_batches": 120, "batch_size": 8},
    
    {"model": "gpt2-xl", "dataset": "NeelNanda/pile-10k", "num_batches": 150, "batch_size": 4},
    {"model": "gpt2-xl", "dataset": "NeelNanda/code-10k", "num_batches": 150, "batch_size": 4},
    {"model": "gpt2-xl", "dataset": "NeelNanda/c4-10k", "num_batches": 150, "batch_size": 4},
    {"model": "gpt2-xl", "dataset": "NeelNanda/wiki-10k", "num_batches": 150, "batch_size": 4},
    
    # Pythia models - all sizes, all datasets
    {"model": "pythia-70m", "dataset": "NeelNanda/pile-10k", "num_batches": 150, "batch_size": 16},
    {"model": "pythia-70m", "dataset": "NeelNanda/code-10k", "num_batches": 150, "batch_size": 16},
    {"model": "pythia-70m", "dataset": "NeelNanda/c4-10k", "num_batches": 150, "batch_size": 16},
    {"model": "pythia-70m", "dataset": "NeelNanda/wiki-10k", "num_batches": 150, "batch_size": 16},
    
    {"model": "pythia-160m", "dataset": "NeelNanda/pile-10k", "num_batches": 150, "batch_size": 16},
    {"model": "pythia-160m", "dataset": "NeelNanda/code-10k", "num_batches": 150, "batch_size": 16},
    {"model": "pythia-160m", "dataset": "NeelNanda/c4-10k", "num_batches": 150, "batch_size": 16},
    {"model": "pythia-160m", "dataset": "NeelNanda/wiki-10k", "num_batches": 150, "batch_size": 16},
    
    {"model": "pythia-410m", "dataset": "NeelNanda/pile-10k", "num_batches": 120, "batch_size": 12},
    {"model": "pythia-410m", "dataset": "NeelNanda/code-10k", "num_batches": 120, "batch_size": 12},
    {"model": "pythia-410m", "dataset": "NeelNanda/c4-10k", "num_batches": 120, "batch_size": 12},
    {"model": "pythia-410m", "dataset": "NeelNanda/wiki-10k", "num_batches": 120, "batch_size": 12},
    
    # GPT-Neo models - all datasets
    {"model": "EleutherAI/gpt-neo-125M", "dataset": "NeelNanda/pile-10k", "num_batches": 150, "batch_size": 16},
    {"model": "EleutherAI/gpt-neo-125M", "dataset": "NeelNanda/code-10k", "num_batches": 150, "batch_size": 16},
    {"model": "EleutherAI/gpt-neo-125M", "dataset": "NeelNanda/c4-10k", "num_batches": 150, "batch_size": 16},
    {"model": "EleutherAI/gpt-neo-125M", "dataset": "NeelNanda/wiki-10k", "num_batches": 150, "batch_size": 16},
    
    {"model": "EleutherAI/gpt-neo-1.3B", "dataset": "NeelNanda/pile-10k", "num_batches": 120, "batch_size": 8},
    {"model": "EleutherAI/gpt-neo-1.3B", "dataset": "NeelNanda/code-10k", "num_batches": 120, "batch_size": 8},
    {"model": "EleutherAI/gpt-neo-1.3B", "dataset": "NeelNanda/c4-10k", "num_batches": 120, "batch_size": 8},
    {"model": "EleutherAI/gpt-neo-1.3B", "dataset": "NeelNanda/wiki-10k", "num_batches": 120, "batch_size": 8},
    
    # Gemma 2B models - reduced batch sizes for memory efficiency
    {"model": "google/gemma-2-2b", "dataset": "NeelNanda/pile-10k", "num_batches": 60, "batch_size": 4},
    {"model": "google/gemma-2-2b", "dataset": "NeelNanda/code-10k", "num_batches": 60, "batch_size": 4},
    {"model": "google/gemma-2-2b", "dataset": "NeelNanda/c4-10k", "num_batches": 60, "batch_size": 4},
    {"model": "google/gemma-2-2b", "dataset": "NeelNanda/wiki-10k", "num_batches": 60, "batch_size": 4},
    
    # {"model": "google/gemma-2b", "dataset": "NeelNanda/pile-10k", "num_batches": 60, "batch_size": 4},
    # {"model": "google/gemma-2b", "dataset": "NeelNanda/code-10k", "num_batches": 60, "batch_size": 4},
    # {"model": "google/gemma-2b", "dataset": "NeelNanda/c4-10k", "num_batches": 60, "batch_size": 4},
    # {"model": "google/gemma-2b", "dataset": "NeelNanda/wiki-10k", "num_batches": 60, "batch_size": 4},
    
    # Large Gemma models - COMMENTED OUT: REQUIRE GPU LARGER THAN RTX 6000 (23GB)
    # gemma-2-9b needs ~30-40GB VRAM, gemma-7b needs ~20-24GB VRAM
    # These will fail on RTX 6000 - uncomment if you have A100 40GB, H100, or similar
    # {"model": "google/gemma-2-9b", "dataset": "NeelNanda/pile-10k", "num_batches": 40, "batch_size": 2},
    # {"model": "google/gemma-2-9b", "dataset": "NeelNanda/code-10k", "num_batches": 40, "batch_size": 2},
    # {"model": "google/gemma-2-9b", "dataset": "NeelNanda/c4-10k", "num_batches": 40, "batch_size": 2},
    # {"model": "google/gemma-2-9b", "dataset": "NeelNanda/wiki-10k", "num_batches": 40, "batch_size": 2},
    # 
    # {"model": "google/gemma-7b", "dataset": "NeelNanda/pile-10k", "num_batches": 40, "batch_size": 2},
    # {"model": "google/gemma-7b", "dataset": "NeelNanda/code-10k", "num_batches": 40, "batch_size": 2},
    # {"model": "google/gemma-7b", "dataset": "NeelNanda/c4-10k", "num_batches": 40, "batch_size": 2},
    # {"model": "google/gemma-7b", "dataset": "NeelNanda/wiki-10k", "num_batches": 40, "batch_size": 2},
    
    # Pythia-1B - final trained (checkpoint_index=None means final checkpoint)
    # {"model": "pythia-1b", "dataset": "NeelNanda/pile-10k", "num_batches": 120, "batch_size": 12},
    # {"model": "pythia-1b", "dataset": "NeelNanda/code-10k", "num_batches": 120, "batch_size": 12},
    # {"model": "pythia-1b", "dataset": "NeelNanda/c4-10k", "num_batches": 120, "batch_size": 12},
    # {"model": "pythia-1b", "dataset": "NeelNanda/wiki-10k", "num_batches": 120, "batch_size": 12},
    
    # Pythia-1B - random initialized (checkpoint_index=0)
    # {"model": "pythia-1b", "dataset": "NeelNanda/pile-10k", "num_batches": 120, "batch_size": 12, "checkpoint_index": 0},
    # {"model": "pythia-1b", "dataset": "NeelNanda/code-10k", "num_batches": 120, "batch_size": 12, "checkpoint_index": 0},
    # {"model": "pythia-1b", "dataset": "NeelNanda/c4-10k", "num_batches": 120, "batch_size": 12, "checkpoint_index": 0},
    # {"model": "pythia-1b", "dataset": "NeelNanda/wiki-10k", "num_batches": 120, "batch_size": 12, "checkpoint_index": 0},
    
    # OPT models - all datasets (selected sizes)
    # {"model": "opt-125m", "dataset": "NeelNanda/pile-10k", "num_batches": 150, "batch_size": 16},
    # {"model": "opt-125m", "dataset": "NeelNanda/code-10k", "num_batches": 150, "batch_size": 16},
    # {"model": "opt-125m", "dataset": "NeelNanda/c4-10k", "num_batches": 150, "batch_size": 16},
    # {"model": "opt-125m", "dataset": "NeelNanda/wiki-10k", "num_batches": 150, "batch_size": 16},
    
    # {"model": "opt-1.3b", "dataset": "NeelNanda/pile-10k", "num_batches": 120, "batch_size": 8},
    # {"model": "opt-1.3b", "dataset": "NeelNanda/code-10k", "num_batches": 120, "batch_size": 8},
    # {"model": "opt-1.3b", "dataset": "NeelNanda/c4-10k", "num_batches": 120, "batch_size": 8},
    # {"model": "opt-1.3b", "dataset": "NeelNanda/wiki-10k", "num_batches": 120, "batch_size": 8},
    
    # {"model": "opt-2.7b", "dataset": "NeelNanda/pile-10k", "num_batches": 100, "batch_size": 6},
    # {"model": "opt-2.7b", "dataset": "NeelNanda/code-10k", "num_batches": 100, "batch_size": 6},
    # {"model": "opt-2.7b", "dataset": "NeelNanda/c4-10k", "num_batches": 100, "batch_size": 6},
    # {"model": "opt-2.7b", "dataset": "NeelNanda/wiki-10k", "num_batches": 100, "batch_size": 6},
    
    # TinyStories-1L-21M - all datasets (from paper experiment details)
    {"model": "tiny-stories-1L-21M", "dataset": "NeelNanda/pile-10k", "num_batches": 150, "batch_size": 16},
    {"model": "tiny-stories-1L-21M", "dataset": "NeelNanda/code-10k", "num_batches": 150, "batch_size": 16},
    {"model": "tiny-stories-1L-21M", "dataset": "NeelNanda/c4-10k", "num_batches": 150, "batch_size": 16},
    {"model": "tiny-stories-1L-21M", "dataset": "NeelNanda/wiki-10k", "num_batches": 150, "batch_size": 16},
    
    # Qwen models - smallest variants (all datasets)
    # {"model": "Qwen/Qwen1.5-0.5B", "dataset": "NeelNanda/pile-10k", "num_batches": 150, "batch_size": 16},
    # {"model": "Qwen/Qwen1.5-0.5B", "dataset": "NeelNanda/code-10k", "num_batches": 150, "batch_size": 16},
    # {"model": "Qwen/Qwen1.5-0.5B", "dataset": "NeelNanda/c4-10k", "num_batches": 150, "batch_size": 16},
    # {"model": "Qwen/Qwen1.5-0.5B", "dataset": "NeelNanda/wiki-10k", "num_batches": 150, "batch_size": 16},
    
    # {"model": "Qwen/Qwen2-0.5B", "dataset": "NeelNanda/pile-10k", "num_batches": 150, "batch_size": 16},
    # {"model": "Qwen/Qwen2-0.5B", "dataset": "NeelNanda/code-10k", "num_batches": 150, "batch_size": 16},
    # {"model": "Qwen/Qwen2-0.5B", "dataset": "NeelNanda/c4-10k", "num_batches": 150, "batch_size": 16},
    # {"model": "Qwen/Qwen2-0.5B", "dataset": "NeelNanda/wiki-10k", "num_batches": 150, "batch_size": 16},
    
    # {"model": "Qwen/Qwen3-0.6B", "dataset": "NeelNanda/pile-10k", "num_batches": 150, "batch_size": 16},
    # {"model": "Qwen/Qwen3-0.6B", "dataset": "NeelNanda/code-10k", "num_batches": 150, "batch_size": 16},
    # {"model": "Qwen/Qwen3-0.6B", "dataset": "NeelNanda/c4-10k", "num_batches": 150, "batch_size": 16},
    # {"model": "Qwen/Qwen3-0.6B", "dataset": "NeelNanda/wiki-10k", "num_batches": 150, "batch_size": 16},
]

CUDA_DEVICE = 1  # Default CUDA device (can be overridden by command line arg)

# Angular bins for magnitude analysis (same as sphere coverage experiments)
ANGULAR_BIN_EDGES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120, 150, 180]


def get_layer_bin(layer_idx: int, n_layers: int) -> str:
    """
    Bin a layer index into categories based on position.
    
    Special layers (marked separately in aggregate analysis):
    - Layer 0: "embedding" (initial token embeddings)
    - Layer 1: "first_block" (first transformer block output)
    - Layer n_layers-1: "last_layer" (final layer output)
    
    Regular layers binned into 5 categories based on relative position:
    - "earlier_early", "early_mid", "mid", "later_mid", "later"
    
    Args:
        layer_idx: Current layer index (0 = embedding, 1+ = block outputs)
        n_layers: Total number of layers INCLUDING embedding (so n_layers = n_blocks + 1)
        
    Returns:
        Layer bin name
    """
    # Special layers
    if layer_idx == 0:
        return "embedding"
    if layer_idx == 1:
        return "first_block"
    if layer_idx == n_layers - 1:
        return "last_layer"
    
    # For binning regular layers, exclude the special ones
    # We bin layers 2 to n_layers-2 into 5 categories
    if n_layers <= 4:  # Too few layers for meaningful binning
        return "mid"
    
    # Calculate relative position among non-special layers (2 to n_layers-2)
    # Map layer_idx to position in range [0, 1]
    min_regular = 2
    max_regular = n_layers - 2
    relative_pos = (layer_idx - min_regular) / (max_regular - min_regular)
    
    if relative_pos < 0.2:
        return "earlier_early"
    elif relative_pos < 0.4:
        return "early_mid"
    elif relative_pos < 0.6:
        return "mid"
    elif relative_pos < 0.8:
        return "later_mid"
    else:
        return "later"


def setup_model_and_data(model_name, dataset_name, cuda_device=1, checkpoint_index=None):
    """Setup model and dataset for analysis. Requires CUDA GPU."""
    print(f"Loading model: {model_name}")
    if checkpoint_index is not None:
        print(f"  Using checkpoint_index: {checkpoint_index}")
    
    # Require CUDA GPU - no CPU fallback
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available. This script requires a CUDA-capable GPU with sufficient VRAM.")
    
    device = f"cuda:{cuda_device}"
    print(f"Using CUDA device: {cuda_device}")
    
    # Some models (like Qwen3) require trust_remote_code=True
    load_kwargs = {}
    if "Qwen3" in model_name or "qwen3" in model_name.lower():
        load_kwargs["trust_remote_code"] = True
    
    if checkpoint_index is not None:
        model = HookedTransformer.from_pretrained(model_name, device=device, checkpoint_index=checkpoint_index, **load_kwargs)
    else:
        model = HookedTransformer.from_pretrained(model_name, device=device, **load_kwargs)
    
    # Explicitly ensure model stays on GPU (never move to CPU)
    model.to(device)
    # Verify model is on GPU
    if hasattr(model, 'cfg') and hasattr(model.cfg, 'device'):
        if 'cpu' in str(model.cfg.device).lower():
            raise RuntimeError(f"Model was loaded to CPU! GPU device {device} is required but not available or not large enough.")
    print(f"✓ Model loaded and verified on {device}")
    
    print(f"Loading dataset: {dataset_name}")
    dataset = load_dataset(
        path=dataset_name,
        split="train",
        streaming=False,
    )
    
    token_dataset = tokenize_and_concatenate(
        dataset=dataset,
        tokenizer=model.tokenizer,
        streaming=True,
        max_length=128,
        add_bos_token=True,
    )
    
    return model, token_dataset, device


def collect_all_layers_activations(model, token_dataset, num_batches=50, batch_size=16):
    """
    Collect activations from ALL layers of the model, INCLUDING the initial embedding layer.
    
    Layer indexing:
    - Layer 0: Initial embeddings (token + positional embeddings)
    - Layer 1: After block 0 (first transformer block)
    - Layer 2: After block 1
    - ...
    - Layer N: After block N-1 (last transformer block)
    
    Returns:
        layer_activations: Dict[layer_idx, tensor] - activations for each layer (0 = embeddings)
        all_tokens: List of token IDs
        all_sequences: List of (sequence_idx, position) tuples
    """
    n_layers = model.cfg.n_layers
    
    # Create hook names for all layers INCLUDING embeddings
    # Layer 0 = embeddings, Layer 1+ = block outputs
    hook_names = ["hook_embed"] + [f"blocks.{layer_idx}.hook_resid_post" for layer_idx in range(n_layers)]
    
    print(f"Collecting activations from ALL {n_layers + 1} layers (including embeddings)...")
    print(f"  Layer 0: Initial embeddings")
    print(f"  Layers 1-{n_layers}: Transformer block outputs")
    print(f"Processing {num_batches} batches of {batch_size} sequences each...")
    
    # Storage for each layer (0 = embeddings, 1+ = block outputs)
    layer_activations = {i: [] for i in range(n_layers + 1)}
    all_tokens = []
    all_sequences = []
    
    with torch.no_grad():
        for batch_idx in tqdm(range(num_batches), desc="Collecting activations"):
            # Get batch of tokens
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size
            batch_tokens = token_dataset[start_idx:end_idx]["tokens"]
            
            # Run model and cache ALL layer activations (including embeddings)
            _, cache = model.run_with_cache(
                batch_tokens,
                names_filter=hook_names
            )
            
            # Extract activations from each layer
            for layer_idx in range(n_layers + 1):
                hook_name = hook_names[layer_idx]
                batch_activations = cache[hook_name]  # [batch, seq_len, d_model]
                batch_size_actual, seq_len, d_model = batch_activations.shape
                
                # Flatten to [batch * seq_len, d_model]
                flattened_acts = batch_activations.reshape(-1, d_model)
                layer_activations[layer_idx].append(flattened_acts.cpu())
            
            # Store token information (only once, same for all layers)
            if batch_idx == 0 or len(all_tokens) < (batch_idx + 1) * batch_size * seq_len:
                for seq_idx in range(batch_size_actual):
                    for pos in range(seq_len):
                        token_id = batch_tokens[seq_idx, pos].item()
                        all_tokens.append(token_id)
                        all_sequences.append((start_idx + seq_idx, pos))
    
    # Concatenate activations for each layer
    for layer_idx in range(n_layers + 1):
        layer_activations[layer_idx] = torch.cat(layer_activations[layer_idx], dim=0)
        if layer_idx == 0:
            print(f"Layer 0 (embeddings): {layer_activations[layer_idx].shape}")
        else:
            print(f"Layer {layer_idx} (after block {layer_idx-1}): {layer_activations[layer_idx].shape}")
    
    return layer_activations, all_tokens, all_sequences


def normalize_to_sphere(activations: np.ndarray) -> np.ndarray:
    """Normalize activation vectors to unit length."""
    norms = np.linalg.norm(activations, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    return activations / norms


def compute_uniform_baseline_R(n_samples: int, n_bootstrap: int = 1000, d_model: int = 768):
    """
    Compute expected R (mean resultant length) distribution for uniform sampling on sphere.
    
    IMPORTANT: This function accounts for the dataset size (n_samples) used to compute the observed R.
    The probability P(R >= observed_R | uniform) depends on n_samples because:
    - E[R] ≈ 1/sqrt(n_samples) (analytical mean)
    - Var[R] ≈ 1/n_samples (variance scales as 1/n)
    - Std[R] ≈ 1/sqrt(n_samples) (standard deviation scales as 1/sqrt(n))
    
    For uniform distribution on a sphere in high dimensions:
    - Mean and std both scale as 1/sqrt(n), so the distribution shape is preserved when scaled
    
    Args:
        n_samples: Number of samples used to compute the observed R (CRITICAL: must match actual data)
        n_bootstrap: Number of bootstrap iterations (1000 for accurate probability estimates)
        d_model: Dimensionality of the space
    
    Returns:
        dict with 'mean', 'std', 'percentiles' (1st, 5th, 25th, 50th, 75th, 95th, 99th),
        'R_samples_scaled' (for probability computation), 'probabilities' for reference R values,
        and 'n_samples' (stored for verification)
    """
    # Analytical mean: E[R] = 1/sqrt(n) for uniform distribution
    analytical_mean = 1.0 / np.sqrt(n_samples)
    
    # For percentiles, use bootstrap with a fixed reasonable sample size
    # The distribution shape scales predictably: both mean and std scale as 1/sqrt(n)
    bootstrap_sample_size = min(10000, n_samples)  # Use up to 10K for bootstrap
    
    R_samples = []
    for _ in range(n_bootstrap):
        # Sample random unit vectors (uniform on sphere)
        # Method: sample from multivariate normal, then normalize
        random_vectors = np.random.randn(bootstrap_sample_size, d_model)
        norms = np.linalg.norm(random_vectors, axis=1, keepdims=True)
        unit_vectors = random_vectors / norms
        
        # Compute mean resultant length
        mean_direction = np.mean(unit_vectors, axis=0)
        R = np.linalg.norm(mean_direction)
        R_samples.append(R)
    
    R_samples = np.array(R_samples)
    
    # Scale the bootstrap results to match the distribution for n_samples
    # CRITICAL: Both mean and std scale as 1/sqrt(n), so:
    # scale_factor = sqrt(bootstrap_sample_size) / sqrt(n_samples)
    # This ensures E[R_scaled] = 1/sqrt(n_samples) and Std[R_scaled] = Std[R_bootstrap] * scale_factor
    scale_factor = np.sqrt(bootstrap_sample_size) / np.sqrt(n_samples)
    R_samples_scaled = R_samples * scale_factor
    
    # Verify scaling: the mean should match analytical_mean
    scaled_mean = np.mean(R_samples_scaled)
    assert abs(scaled_mean - analytical_mean) < 0.01 * analytical_mean, \
        f"Scaling error: scaled_mean={scaled_mean:.6f}, analytical_mean={analytical_mean:.6f}"
    
    # Compute probabilities for reference R values
    reference_R_values = [0.1, 0.3, 0.5, 0.8]
    probabilities = {}
    for ref_R in reference_R_values:
        # P(R >= ref_R | uniform) = fraction of samples with R >= ref_R
        prob = np.mean(R_samples_scaled >= ref_R)
        probabilities[ref_R] = prob
    
    return {
        'mean': analytical_mean,  # Use analytical mean (more accurate)
        'std': np.std(R_samples_scaled),  # Scaled std from bootstrap (accounts for n_samples)
        'R_samples_scaled': R_samples_scaled,  # Store for probability computation (scaled to n_samples)
        'n_samples': n_samples,  # Store n_samples for verification
        'percentiles': {
            1: np.percentile(R_samples_scaled, 1),
            5: np.percentile(R_samples_scaled, 5),
            25: np.percentile(R_samples_scaled, 25),
            50: np.percentile(R_samples_scaled, 50),
            75: np.percentile(R_samples_scaled, 75),
            95: np.percentile(R_samples_scaled, 95),
            99: np.percentile(R_samples_scaled, 99)
        },
        'probabilities': probabilities  # P(R >= ref_R | uniform, n=n_samples) for reference values
    }


def compute_uniform_probability(observed_R: float, uniform_baseline: Dict) -> float:
    """
    Compute P(R >= observed_R | uniform distribution, n=n_samples).
    
    IMPORTANT: This probability accounts for the dataset size (n_samples) used to compute observed_R.
    The uniform_baseline must be computed with the same n_samples as the observed data.
    
    Args:
        observed_R: The observed mean resultant length (computed from n_samples data points)
        uniform_baseline: Result from compute_uniform_baseline_R (must use same n_samples)
    
    Returns:
        Probability that uniform distribution would produce R >= observed_R
        given n_samples data points. Lower probability = more unlikely under uniform.
    """
    R_samples_scaled = uniform_baseline['R_samples_scaled']
    # R_samples_scaled is already scaled to match the distribution for n_samples
    # So we can directly compute: P(R >= observed_R | uniform, n=n_samples)
    prob = np.mean(R_samples_scaled >= observed_R)
    return prob


def compute_angular_distance_stats(normalized_activations: np.ndarray, original_magnitudes: np.ndarray = None) -> Dict:
    """
    Compute angular distance statistics for one layer.
    
    Returns statistics about angular distances from mean direction.
    """
    n_samples, d_model = normalized_activations.shape
    
    # Compute mean direction
    mean_vector = np.mean(normalized_activations, axis=0)
    mean_vector_norm = np.linalg.norm(mean_vector)
    
    if mean_vector_norm > 0:
        mean_direction = mean_vector / mean_vector_norm
    else:
        mean_direction = mean_vector
    
    # Mean resultant length R (concentration measure)
    R = mean_vector_norm
    
    # Compute angular distances from mean direction
    dot_products = normalized_activations @ mean_direction
    dot_products = np.clip(dot_products, -1.0, 1.0)
    angular_distances_rad = np.arccos(dot_products)
    angular_distances_deg = np.degrees(angular_distances_rad)
    
    # Magnitude distribution by angular bins (if provided)
    magnitude_by_angle_bins = None
    if original_magnitudes is not None:
        bin_info = []
        for i in range(len(ANGULAR_BIN_EDGES) - 1):
            lower_angle_deg = ANGULAR_BIN_EDGES[i]
            upper_angle_deg = ANGULAR_BIN_EDGES[i + 1]
            lower_angle_rad = np.radians(lower_angle_deg)
            upper_angle_rad = np.radians(upper_angle_deg)
            
            mask = (angular_distances_rad >= lower_angle_rad) & (angular_distances_rad < upper_angle_rad)
            if i == len(ANGULAR_BIN_EDGES) - 2:
                mask = (angular_distances_rad >= lower_angle_rad) & (angular_distances_rad <= upper_angle_rad)
            
            n_vectors = np.sum(mask)
            
            if n_vectors > 0:
                avg_magnitude = float(np.mean(original_magnitudes[mask]))
                std_magnitude = float(np.std(original_magnitudes[mask]))
                median_magnitude = float(np.median(original_magnitudes[mask]))
            else:
                avg_magnitude = 0.0
                std_magnitude = 0.0
                median_magnitude = 0.0
            
            bin_info.append({
                "angle_range": f"{lower_angle_deg}-{upper_angle_deg}°",
                "angle_lower_deg": float(lower_angle_deg),
                "angle_upper_deg": float(upper_angle_deg),
                "angle_center_deg": float((lower_angle_deg + upper_angle_deg) / 2),
                "n_vectors": int(n_vectors),
                "pct_vectors": float(n_vectors / n_samples * 100) if n_samples > 0 else 0.0,
                "avg_magnitude": avg_magnitude,
                "std_magnitude": std_magnitude,
                "median_magnitude": median_magnitude,
                "min_magnitude": float(np.min(original_magnitudes[mask])) if n_vectors > 0 else 0.0,
                "max_magnitude": float(np.max(original_magnitudes[mask])) if n_vectors > 0 else 0.0,
            })
        magnitude_by_angle_bins = bin_info
    
    # Within-shell uniformity analysis (NEW!)
    uniformity_data = None
    if original_magnitudes is not None:
        uniformity_data = compute_pairwise_uniformity(
            normalized_vectors=normalized_activations,
            original_magnitudes=original_magnitudes,
            angular_distances_rad=angular_distances_rad,
            n_distance_bins=10,
            max_pairs_per_bin=5000,  # Reduced from 10000 for faster computation
            store_full_distribution=True  # Store full distributions for plotting
        )
    
    # Pairwise torus detection analysis (NEW!)
    pairwise_torus_data = None
    if original_magnitudes is not None:
        print(f"      Computing pairwise torus detection...")
        pairwise_torus_data = compute_pairwise_angular_distances(
            normalized_vectors=normalized_activations,
            sample_size=min(5000, n_samples)  # Reduced sample size for faster computation (was 10000)
        )
    
    # Store magnitude distribution if provided
    magnitude_distribution = original_magnitudes.tolist() if original_magnitudes is not None else []
    
    return {
        "n_samples": n_samples,
        "d_model": d_model,
        "mean_resultant_length": float(R),
        "mean_direction": mean_direction.tolist(),  # Convert to list for JSON
        "mean_vector": mean_vector.tolist(),  # Convert to list for JSON
        "angular_distances_deg": angular_distances_deg,  # Keep full array for plotting
        "min_deg": float(np.min(angular_distances_deg)),
        "max_deg": float(np.max(angular_distances_deg)),
        "mean_deg": float(np.mean(angular_distances_deg)),
        "median_deg": float(np.median(angular_distances_deg)),
        "std_deg": float(np.std(angular_distances_deg)),
        "percentile_25_deg": float(np.percentile(angular_distances_deg, 25)),
        "percentile_50_deg": float(np.percentile(angular_distances_deg, 50)),
        "percentile_75_deg": float(np.percentile(angular_distances_deg, 75)),
        "percentile_90_deg": float(np.percentile(angular_distances_deg, 90)),
        "percentile_95_deg": float(np.percentile(angular_distances_deg, 95)),
        "percentile_99_deg": float(np.percentile(angular_distances_deg, 99)),
        "percentile_100_deg": float(np.percentile(angular_distances_deg, 100)),
        "magnitude_distribution": magnitude_distribution,  # NEW! For magnitude plots
        "magnitude_by_angle_bins": magnitude_by_angle_bins,
        "within_shell_uniformity": uniformity_data,  # NEW!
        "pairwise_torus_analysis": pairwise_torus_data,  # NEW!
        # Sphere coverage metrics (for concentration analysis)
        "angular_variance": float(1 - R),
        "angular_std_rad": float(np.sqrt(-2 * np.log(max(R, 1e-10)))),
        "angular_std_deg": float(np.degrees(np.sqrt(-2 * np.log(max(R, 1e-10))))),
    }


def analyze_layer_evolution(layer_activations: Dict[int, torch.Tensor]) -> Dict:
    """
    Analyze angular distribution evolution across all layers.
    
    Args:
        layer_activations: Dict mapping layer_idx to activation tensor
        
    Returns:
        Dict with statistics for each layer
    """
    n_layers = len(layer_activations)
    results = {}
    
    print(f"\n📊 Computing angular distance statistics for {n_layers} layers...")
    
    for layer_idx in range(n_layers):
        print(f"  Layer {layer_idx}/{n_layers-1}...", end=" ")
        
        # Get activations for this layer
        activations = layer_activations[layer_idx].numpy()
        
        # Compute original magnitudes BEFORE normalization
        original_magnitudes = np.linalg.norm(activations, axis=1)
        
        # Normalize to unit sphere
        normalized = normalize_to_sphere(activations)
        
        # Compute angular distance statistics with magnitude info
        stats = compute_angular_distance_stats(normalized, original_magnitudes)
        
        results[layer_idx] = stats
        print(f"R={stats['mean_resultant_length']:.3f}, mean_angle={stats['mean_deg']:.1f}°")
    
    return results


def create_evolution_visualization(results: Dict, exp_config: Dict, output_dir: Path):
    """
    Create comprehensive visualization showing distribution evolution across layers.
    
    This creates a single figure with:
    1. All layer distributions overlaid (with same x/y axes)
    2. Summary statistics (min/max/median/mean) per layer
    
    Layer 0 = embeddings, Layer 1 = first block, Layer N-1 = last layer (special)
    """
    n_layers = len(results)
    
    # Create figure with multiple subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
    
    model_short = exp_config['model'].split('/')[-1]
    dataset_short = exp_config['dataset'].split('/')[-1]
    
    fig.suptitle(f'Layer-by-Layer Evolution of Angular Distribution\n' +
                 f'{model_short} on {dataset_short} ({n_layers} layers, L0=embeddings)',
                 fontsize=16, fontweight='bold')
    
    # Plot 1: All distributions overlaid
    ax1 = fig.add_subplot(gs[0:2, 0])
    
    # Use colormap to show layer progression
    colors = plt.cm.viridis(np.linspace(0, 1, n_layers))
    
    # Special layer colors
    special_colors = {0: 'red', 1: 'orange', n_layers-1: 'magenta'}
    
    for layer_idx in range(n_layers):
        angular_dists = results[layer_idx]['angular_distances_deg']
        
        # Determine color and label for special layers
        if layer_idx in special_colors:
            color = special_colors[layer_idx]
            alpha = 0.5
            linewidth = 2
            if layer_idx == 0:
                label = 'L0 (Embedding)'
            elif layer_idx == 1:
                label = 'L1 (First Block)'
            else:
                label = f'L{layer_idx} (Last Layer)'
        else:
            color = colors[layer_idx]
            alpha = 0.3
            linewidth = 1
            label = f'Layer {layer_idx}' if layer_idx % (max(1, n_layers // 5)) == 0 else ''
        
        # Plot histogram with transparency
        ax1.hist(angular_dists, bins=50, alpha=alpha, color=color,
                density=True, label=label, linewidth=linewidth, histtype='step' if layer_idx in special_colors else 'stepfilled')
    
    ax1.set_xlabel('Angular Distance from Mean Direction (degrees)', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Density', fontsize=12, fontweight='bold')
    ax1.set_title('All Layers\' Angular Distance Distributions Overlaid\n' +
                  '(Special: Red=Embedding, Orange=First Block, Magenta=Last Layer)', fontsize=13)
    ax1.set_xlim([0, 180])
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=8, loc='upper right')
    
    # Plot 2: Min/Max/Median/Mean progression
    ax2 = fig.add_subplot(gs[0, 1])
    
    layer_indices = list(range(n_layers))
    mins = [results[i]['min_deg'] for i in layer_indices]
    maxs = [results[i]['max_deg'] for i in layer_indices]
    means = [results[i]['mean_deg'] for i in layer_indices]
    medians = [results[i]['median_deg'] for i in layer_indices]
    p25s = [results[i]['percentile_25_deg'] for i in layer_indices]
    p75s = [results[i]['percentile_75_deg'] for i in layer_indices]
    
    ax2.plot(layer_indices, mins, 'o-', label='Min', color='blue', linewidth=2, markersize=4)
    ax2.plot(layer_indices, maxs, 's-', label='Max', color='red', linewidth=2, markersize=4)
    ax2.plot(layer_indices, means, '^-', label='Mean', color='green', linewidth=2, markersize=5)
    ax2.plot(layer_indices, medians, 'D-', label='Median', color='purple', linewidth=2, markersize=4)
    ax2.fill_between(layer_indices, p25s, p75s, alpha=0.2, color='gray', label='25th-75th percentile')
    
    # Mark special layers
    ax2.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Embedding')
    ax2.axvline(x=1, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='First Block')
    ax2.axvline(x=n_layers-1, color='magenta', linestyle='--', alpha=0.5, linewidth=1.5, label='Last Layer')
    
    ax2.set_xlabel('Layer Index (0=Embedding, 1=First Block)', fontsize=11, fontweight='bold')
    ax2.set_ylabel('Angular Distance (degrees)', fontsize=11, fontweight='bold')
    ax2.set_title('Angular Distance Statistics Across Layers', fontsize=12)
    ax2.legend(fontsize=7, ncol=2)
    ax2.grid(alpha=0.3)
    ax2.set_ylim([0, 180])
    
    # Plot 3: Concentration (Mean Resultant Length) progression
    ax3 = fig.add_subplot(gs[1, 1])
    
    Rs = [results[i]['mean_resultant_length'] for i in layer_indices]
    
    ax3.plot(layer_indices, Rs, 'o-', color='steelblue', linewidth=2, markersize=6)
    ax3.axhline(y=0.8, color='red', linestyle='--', alpha=0.3, linewidth=1, label='Highly concentrated (0.8)')
    ax3.axhline(y=0.5, color='orange', linestyle='--', alpha=0.3, linewidth=1, label='Moderate (0.5)')
    
    # Mark special layers
    ax3.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Embedding')
    ax3.axvline(x=1, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='First Block')
    ax3.axvline(x=n_layers-1, color='magenta', linestyle='--', alpha=0.5, linewidth=1.5, label='Last Layer')
    
    ax3.set_xlabel('Layer Index (0=Embedding)', fontsize=11, fontweight='bold')
    ax3.set_ylabel('Mean Resultant Length (R)', fontsize=11, fontweight='bold')
    ax3.set_title('Concentration (R) Across Layers\n(Higher = more concentrated)', fontsize=12)
    ax3.legend(fontsize=7, ncol=2)
    ax3.grid(alpha=0.3)
    ax3.set_ylim([0, 1])
    
    # Plot 4: Box plots for select layers (ALWAYS include special layers)
    ax4 = fig.add_subplot(gs[2, :])
    
    # Always include special layers: 0 (embedding), 1 (first block), n_layers-1 (last)
    special_layers = [0, 1, n_layers-1]
    
    # Add evenly spaced intermediate layers (max total 12 layers to keep readable)
    n_layers_to_show = min(12, n_layers)
    n_intermediate = n_layers_to_show - 3  # Reserve spots for special layers
    
    if n_intermediate > 0 and n_layers > 3:
        # Sample from layers 2 to n_layers-2
        intermediate_range = list(range(2, n_layers-1))
        if len(intermediate_range) > 0:
            layer_step = max(1, len(intermediate_range) // n_intermediate)
            intermediate_layers = intermediate_range[::layer_step][:n_intermediate]
        else:
            intermediate_layers = []
    else:
        intermediate_layers = []
    
    layers_to_show = sorted(set(special_layers + intermediate_layers))
    
    box_data = [results[i]['angular_distances_deg'] for i in layers_to_show]
    box_labels = []
    for i in layers_to_show:
        if i == 0:
            box_labels.append('L0\n(Emb)')
        elif i == 1:
            box_labels.append('L1\n(1st)')
        elif i == n_layers-1:
            box_labels.append(f'L{i}\n(Last)')
        else:
            box_labels.append(f'L{i}')
    
    bp = ax4.boxplot(box_data, tick_labels=box_labels, patch_artist=True,
                     showfliers=False)  # Hide outliers for clarity
    
    # Color boxes: special layers get special colors, others use colormap
    for patch, layer_idx in zip(bp['boxes'], layers_to_show):
        if layer_idx in special_colors:
            patch.set_facecolor(special_colors[layer_idx])
            patch.set_alpha(0.8)
            patch.set_linewidth(2)
        else:
            patch.set_facecolor(colors[layer_idx])
            patch.set_alpha(0.7)
    
    ax4.set_xlabel('Layer (Red=Embedding, Orange=First Block, Magenta=Last)', fontsize=11, fontweight='bold')
    ax4.set_ylabel('Angular Distance (degrees)', fontsize=11, fontweight='bold')
    ax4.set_title(f'Distribution Box Plots for Selected Layers (showing {len(layers_to_show)} of {n_layers} layers)', fontsize=12)
    ax4.grid(axis='y', alpha=0.3)
    ax4.set_ylim([0, 180])
    
    plt.savefig(output_dir / "layer_evolution_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved layer_evolution_analysis.png")


def create_individual_layer_plots(results: Dict, exp_config: Dict, output_dir: Path):
    """
    Create individual distribution plots for each layer in a grid.
    Layer 0 = embeddings, Layer 1 = first block, Layer N-1 = last layer
    """
    n_layers = len(results)
    
    # Determine grid size
    n_cols = 4
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten() if n_layers > 1 else [axes]
    
    model_short = exp_config['model'].split('/')[-1]
    dataset_short = exp_config['dataset'].split('/')[-1]
    
    fig.suptitle(f'Individual Layer Distributions: {model_short} on {dataset_short}\n(L0=Embedding, L1=First Block, L{n_layers-1}=Last Layer)',
                 fontsize=14, fontweight='bold')
    
    colors = plt.cm.viridis(np.linspace(0, 1, n_layers))
    special_colors = {0: 'red', 1: 'orange', n_layers-1: 'magenta'}
    
    # Compute uniform baseline R for reference
    n_samples = results[0]['n_samples']
    d_model = results[0].get('d_model', 768)
    uniform_baseline = compute_uniform_baseline_R(n_samples, d_model=d_model)
    uniform_R_mean = uniform_baseline['mean']
    
    # First pass: find max count for consistent y-axis
    max_count = 0
    for layer_idx in range(n_layers):
        angular_dists = results[layer_idx]['angular_distances_deg']
        counts, _ = np.histogram(angular_dists, bins=40, range=(0, 180))
        max_count = max(max_count, np.max(counts))
    
    # Second pass: plot with consistent axes
    for layer_idx in range(n_layers):
        ax = axes[layer_idx]
        angular_dists = results[layer_idx]['angular_distances_deg']
        
        # Use special colors for special layers
        if layer_idx in special_colors:
            color = special_colors[layer_idx]
            alpha = 0.8
            edgecolor = 'black'
            linewidth = 2
        else:
            color = colors[layer_idx]
            alpha = 0.7
            edgecolor = 'black'
            linewidth = 1
        
        ax.hist(angular_dists, bins=40, range=(0, 180), color=color, 
                alpha=alpha, edgecolor=edgecolor, linewidth=linewidth)
        
        # Mark statistics
        ax.axvline(results[layer_idx]['mean_deg'], color='darkred', linestyle='-',
                  linewidth=2, label=f"Mean: {results[layer_idx]['mean_deg']:.1f}°")
        ax.axvline(results[layer_idx]['median_deg'], color='darkorange', linestyle='--',
                  linewidth=2, label=f"Median: {results[layer_idx]['median_deg']:.1f}°")
        
        # Special layer titles with uniform baseline reference
        R_value = results[layer_idx]['mean_resultant_length']
        if layer_idx == 0:
            title = f"Layer 0 (EMBEDDING) | R={R_value:.3f} (Uniform: {uniform_R_mean:.3f})"
        elif layer_idx == 1:
            title = f"Layer 1 (FIRST BLOCK) | R={R_value:.3f} (Uniform: {uniform_R_mean:.3f})"
        elif layer_idx == n_layers-1:
            title = f"Layer {layer_idx} (LAST LAYER) | R={R_value:.3f} (Uniform: {uniform_R_mean:.3f})"
        else:
            title = f"Layer {layer_idx} | R={R_value:.3f} (Uniform: {uniform_R_mean:.3f})"
        
        ax.set_title(title, fontsize=10, fontweight='bold' if layer_idx in special_colors else 'normal')
        ax.set_xlabel('Angular Distance (deg)', fontsize=9)
        ax.set_ylabel('Count', fontsize=9)
        ax.legend(fontsize=7)
        ax.grid(alpha=0.3)
        ax.set_xlim([0, 180])
        ax.set_ylim([0, max_count * 1.1])  # Consistent y-axis with 10% padding
    
    # Hide unused subplots
    for idx in range(n_layers, len(axes)):
        axes[idx].axis('off')
    
    # Fix spacing - add more padding for title and first row
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    plt.subplots_adjust(top=0.95, hspace=0.4, wspace=0.3)  # Extra space for main title and subplot spacing
    plt.savefig(output_dir / "individual_layer_distributions.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved individual_layer_distributions.png")


def create_magnitude_by_layer_plots(results: Dict, layer_activations: Dict, exp_config: Dict, output_dir: Path):
    """
    Create plots showing angular distribution and magnitude for each layer.
    Left: Angular distance distribution (like individual_layer_distributions)
    Right: Magnitude per 10% data bin
    Layer 0 = embeddings, Layer 1 = first block, Layer N-1 = last layer
    """
    n_layers = len(results)
    
    model_short = exp_config['model'].split('/')[-1]
    dataset_short = exp_config['dataset'].split('/')[-1]
    
    # Create one big plot showing all layers
    fig, axes = plt.subplots(n_layers, 2, figsize=(16, 4*n_layers))
    if n_layers == 1:
        axes = axes.reshape(1, -1)
    
    fig.suptitle(f'Angular Distribution and Magnitude Analysis - All Layers\n{model_short} on {dataset_short}\n(L0=Embedding, L1=First Block, L{n_layers-1}=Last Layer)',
                 fontsize=14, fontweight='bold')
    
    colors_layers = plt.cm.viridis(np.linspace(0, 1, n_layers))
    special_colors = {0: 'red', 1: 'orange', n_layers-1: 'magenta'}
    
    # Find max count for consistent y-axis on left plots
    max_count = 0
    for layer_idx in range(n_layers):
        angular_dists = results[layer_idx]['angular_distances_deg']
        counts, _ = np.histogram(angular_dists, bins=40, range=(0, 180))
        max_count = max(max_count, np.max(counts))
    
    for layer_idx in range(n_layers):
        # Left plot: Angular distance distribution
        ax = axes[layer_idx, 0]
        angular_dists = results[layer_idx]['angular_distances_deg']
        
        # Use special colors for special layers
        if layer_idx in special_colors:
            color = special_colors[layer_idx]
            alpha = 0.8
            edgecolor = 'black'
            linewidth = 2
        else:
            color = colors_layers[layer_idx]
            alpha = 0.7
            edgecolor = 'black'
            linewidth = 1
        
        ax.hist(angular_dists, bins=40, range=(0, 180), color=color, 
                alpha=alpha, edgecolor=edgecolor, linewidth=linewidth)
        
        # Mark statistics
        ax.axvline(results[layer_idx]['mean_deg'], color='darkred', linestyle='-',
                  linewidth=2, label=f"Mean: {results[layer_idx]['mean_deg']:.1f}°")
        ax.axvline(results[layer_idx]['median_deg'], color='darkorange', linestyle='--',
                  linewidth=2, label=f"Median: {results[layer_idx]['median_deg']:.1f}°")
        
        # Special layer titles
        if layer_idx == 0:
            title = f'Layer 0 (EMBEDDING) - Angular Distance Distribution (R={results[layer_idx]["mean_resultant_length"]:.3f})'
        elif layer_idx == 1:
            title = f'Layer 1 (FIRST BLOCK) - Angular Distance Distribution (R={results[layer_idx]["mean_resultant_length"]:.3f})'
        elif layer_idx == n_layers-1:
            title = f'Layer {layer_idx} (LAST LAYER) - Angular Distance Distribution (R={results[layer_idx]["mean_resultant_length"]:.3f})'
        else:
            title = f'Layer {layer_idx} - Angular Distance Distribution (R={results[layer_idx]["mean_resultant_length"]:.3f})'
        
        ax.set_ylabel('Count', fontsize=9, fontweight='bold')
        ax.set_xlabel('Angular Distance from Mean (degrees)', fontsize=9, fontweight='bold')
        ax.set_title(title, fontsize=10, fontweight='bold' if layer_idx in special_colors else 'normal')
        ax.legend(fontsize=7)
        ax.grid(axis='y', alpha=0.3)
        ax.set_xlim([0, 180])
        ax.set_ylim([0, max_count * 1.1])
        
        # Right plot: Average magnitude per 10% data bin (percentile-based)
        ax = axes[layer_idx, 1]
        
        # Get angular distances and magnitudes for this layer
        angular_dists = results[layer_idx]['angular_distances_deg']
        layer_acts = layer_activations[layer_idx].numpy()
        layer_mags = np.linalg.norm(layer_acts, axis=1)
        
        # Create 10% percentile bins
        n_percentile_bins = 10
        percentile_edges = np.linspace(0, 100, n_percentile_bins + 1)
        distance_percentiles = np.percentile(angular_dists, percentile_edges)
        
        # Compute average magnitude for each percentile bin
        percentile_bin_mags = []
        percentile_bin_labels = []
        percentile_bin_ranges = []
        
        for i in range(n_percentile_bins):
            lower_pct = percentile_edges[i]
            upper_pct = percentile_edges[i + 1]
            lower_dist = distance_percentiles[i]
            upper_dist = distance_percentiles[i + 1]
            
            # Find vectors in this percentile range
            if i == n_percentile_bins - 1:
                mask = (angular_dists >= lower_dist) & (angular_dists <= upper_dist)
            else:
                mask = (angular_dists >= lower_dist) & (angular_dists < upper_dist)
            
            if np.sum(mask) > 0:
                avg_mag = np.mean(layer_mags[mask])
                percentile_bin_mags.append(avg_mag)
                percentile_bin_labels.append(f'{int(lower_pct)}-{int(upper_pct)}%')
                percentile_bin_ranges.append(f'{lower_dist:.1f}-{upper_dist:.1f}°')
            else:
                percentile_bin_mags.append(0)
                percentile_bin_labels.append(f'{int(lower_pct)}-{int(upper_pct)}%')
                percentile_bin_ranges.append('N/A')
        
        # Plot
        bin_colors_pct = plt.cm.Greens(np.linspace(0.3, 0.9, len(percentile_bin_labels)))
        bars = ax.bar(range(len(percentile_bin_labels)), percentile_bin_mags, 
                     color=bin_colors_pct, alpha=0.8, edgecolor='black')
        ax.set_xticks(range(len(percentile_bin_labels)))
        ax.set_xticklabels(percentile_bin_labels, rotation=45, ha='right', fontsize=7)
        ax.set_ylabel('Avg Magnitude', fontsize=9, fontweight='bold')
        
        # Special layer titles
        if layer_idx == 0:
            mag_title = f'Layer 0 (EMBEDDING) - Magnitude per 10% Data Bin\n(Percentile-based)'
        elif layer_idx == 1:
            mag_title = f'Layer 1 (FIRST BLOCK) - Magnitude per 10% Data Bin\n(Percentile-based)'
        elif layer_idx == n_layers-1:
            mag_title = f'Layer {layer_idx} (LAST LAYER) - Magnitude per 10% Data Bin\n(Percentile-based)'
        else:
            mag_title = f'Layer {layer_idx} - Magnitude per 10% Data Bin\n(Percentile-based)'
        
        ax.set_title(mag_title, fontsize=10, fontweight='bold' if layer_idx in special_colors else 'normal')
        ax.grid(axis='y', alpha=0.3)
        
        # Add distance range labels on bars
        for i, (bar, mag, range_str) in enumerate(zip(bars, percentile_bin_mags, percentile_bin_ranges)):
            if mag > 0:
                ax.text(bar.get_x() + bar.get_width()/2, mag * 0.5, range_str, 
                       ha='center', va='center', fontsize=5, rotation=90, color='white', fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_dir / "magnitude_by_angular_bins_all_layers.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved magnitude_by_angular_bins_all_layers.png")


def create_uniformity_plots_layers(results: Dict, exp_config: Dict, output_dir: Path):
    """Create within-shell uniformity analysis plots for selected layers."""
    
    n_layers = len(results)
    
    # Plot ALL layers (user requested this!)
    layers_to_plot = list(range(n_layers))
    
    model_short = exp_config['model'].split('/')[-1]
    dataset_short = exp_config['dataset'].split('/')[-1]
    
    # Create layer_specific subfolder
    layer_specific_dir = output_dir / "layer_specific"
    layer_specific_dir.mkdir(exist_ok=True)
    
    print(f"\n📈 Creating within-shell uniformity analysis plots for all layers...")
    print(f"   Plotting layers: {layers_to_plot}")
    
    for layer_idx in layers_to_plot:
        uniformity_data = results[layer_idx].get("within_shell_uniformity")
        
        if uniformity_data is None:
            print(f"   ⚠️  No uniformity data for layer {layer_idx}")
            continue
        
        print(f"   Plotting layer {layer_idx}...")
        
        # Plot 1: Within-shell uniformity summary (4 subplots)
        plot_within_shell_uniformity(
            uniformity_data=uniformity_data,
            output_path=layer_specific_dir / f"layer_{layer_idx}_within_shell_uniformity.png",
            title=f'Layer {layer_idx}\n{model_short} on {dataset_short}'
        )
        
        # Plot 2: Pairwise angle distributions (grid of histograms)
        plot_pairwise_angle_distributions(
            uniformity_data=uniformity_data,
            output_path=layer_specific_dir / f"layer_{layer_idx}_pairwise_distributions.png",
            title=f'Layer {layer_idx} - {model_short} on {dataset_short}'
        )
    
    print(f"   ✓ Completed uniformity plots for {len(layers_to_plot)} layers")


def create_pairwise_torus_plots_layers(results: Dict, exp_config: Dict, output_dir: Path):
    """Create pairwise torus detection plots for all layers."""
    
    n_layers = len(results)
    model_short = exp_config['model'].split('/')[-1]
    dataset_short = exp_config['dataset'].split('/')[-1]
    
    # Create layer_specific subfolder
    layer_specific_dir = output_dir / "layer_specific"
    layer_specific_dir.mkdir(exist_ok=True)
    
    print(f"\n🔍 Creating pairwise torus detection plots for all layers...")
    
    for layer_idx in range(n_layers):
        pairwise_data = results[layer_idx].get("pairwise_torus_analysis")
        
        if pairwise_data is None:
            print(f"   ⚠️  No pairwise torus data for layer {layer_idx}")
            continue
        
        print(f"   Plotting layer {layer_idx}...")
        
        # Create torus detection plot
        plot_pairwise_torus_detection(
            pairwise_data=pairwise_data,
            output_path=layer_specific_dir / f"layer_{layer_idx}_pairwise_torus_detection.png",
            title=f'Layer {layer_idx} - {model_short} on {dataset_short}'
        )
    
    print(f"   ✓ Completed pairwise torus detection plots for {n_layers} layers")


def create_sphere_coverage_plots_layers(results: Dict, exp_config: Dict, output_dir: Path):
    """Create sphere coverage analysis plots for all layers."""
    
    n_layers = len(results)
    model_short = exp_config['model'].split('/')[-1]
    dataset_short = exp_config['dataset'].split('/')[-1]
    
    # Create layer_specific subfolder
    layer_specific_dir = output_dir / "layer_specific"
    layer_specific_dir.mkdir(exist_ok=True)
    
    print(f"\n🌐 Creating sphere coverage plots for all layers...")
    
    for layer_idx in range(n_layers):
        stats = results[layer_idx]
        
        print(f"   Plotting layer {layer_idx}...")
        
        # Create sphere coverage plot (1x3 layout - removed sigma-based plot)
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Sphere Coverage Analysis - Layer {layer_idx}\n{model_short} on {dataset_short}',
                     fontsize=14, fontweight='bold')
        
        R = stats['mean_resultant_length']
        angular_std_deg = stats['angular_std_deg']
        
        # Plot 1: Mean Resultant Length (concentration)
        ax = axes[0]
        ax.bar(['R'], [R], color='steelblue', alpha=0.7, edgecolor='black', width=0.5)
        ax.axhline(y=0.8, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Highly conc. (0.8)')
        ax.axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Moderate (0.5)')
        ax.set_ylabel('Mean Resultant Length (R)', fontsize=11, fontweight='bold')
        ax.set_title(f'Concentration: R={R:.4f}\n(Higher = More Concentrated)', fontsize=12)
        ax.set_ylim([0, 1.1])
        ax.legend(fontsize=8)
        ax.grid(axis='y', alpha=0.3)
        ax.text(0, R + 0.05, f'{R:.4f}', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Plot 2: Angular standard deviation
        ax = axes[1]
        ax.bar(['Angular Std'], [angular_std_deg], color='coral', alpha=0.7, edgecolor='black', width=0.5)
        # Add baseline reference lines
        ax.axhline(y=87, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Uniform (~87°)')
        ax.axhline(y=30, color='green', linestyle='--', linewidth=1.5, alpha=0.5, label='Concentrated (30°)')
        ax.axhline(y=60, color='orange', linestyle='--', linewidth=1.5, alpha=0.5, label='Moderate (60°)')
        ax.set_ylabel('Degrees', fontsize=11, fontweight='bold')
        ax.set_title(f'Angular Standard Deviation\nσ={angular_std_deg:.1f}° (Lower = More Concentrated)', fontsize=12)
        ax.legend(fontsize=7, loc='upper right')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 100])  # Cap at 100° for visibility
        ax.text(0, angular_std_deg + 2, f'{angular_std_deg:.1f}°', ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        # Plot 3: Percentile-based cones
        ax = axes[2]
        percentiles = [50, 90, 95, 99, 100]
        # Get angular distances to compute percentiles
        angular_dists_deg = np.array([stats['min_deg'], stats['mean_deg'], stats['max_deg']])  # Simplified fallback
        # Better: use actual percentile fields if they exist, otherwise estimate
        angles = []
        for p in percentiles:
            if f'percentile_{p}_deg' in stats:
                angles.append(stats[f'percentile_{p}_deg'])
            else:
                # Estimate from available stats (linear interpolation)
                angles.append(stats['mean_deg'])  # Fallback
        # Compute coverage
        coverages = [(1 - np.cos(np.radians(a))) / 2 * 100 for a in angles]
        
        # Compute UNIFORM distribution baseline (theoretical)
        # For uniform on sphere: fraction p within angle θ → θ = arccos(1 - 2p)
        # For 100th percentile, use 180° (full hemisphere)
        uniform_angles = []
        for p in percentiles:
            if p == 100:
                uniform_angles.append(180.0)  # Full hemisphere coverage
            else:
                theta_rad = np.arccos(1 - 2 * (p / 100.0))
                uniform_angles.append(np.degrees(theta_rad))
        
        x = np.arange(len(percentiles))
        width = 0.25
        
        ax2 = ax.twinx()
        bars1 = ax.bar(x - width, angles, width, label='Your Cone Angle', color='skyblue', alpha=0.7, edgecolor='black')
        bars_uniform = ax.bar(x, uniform_angles, width, label='Uniform Baseline', color='orange', alpha=0.6, edgecolor='black')
        bars2 = ax2.bar(x + width, coverages, width, label='Surface Coverage (%)', color='lightcoral', alpha=0.7, edgecolor='black')
        
        # Add value labels on bars
        for i, (your_angle, uniform_angle, cov) in enumerate(zip(angles, uniform_angles, coverages)):
            ax.text(i - width, your_angle + 3, f'{your_angle:.0f}°', ha='center', fontsize=7, color='darkblue')
            ax.text(i, uniform_angle + 3, f'{uniform_angle:.0f}°', ha='center', fontsize=7, color='darkorange', fontweight='bold')
            ax2.text(i + width, cov + 2, f'{cov:.0f}%', ha='center', fontsize=7, color='darkred')
        
        ax.set_xlabel('Percentile', fontsize=10, fontweight='bold')
        ax.set_ylabel('Cone Angle (degrees)', fontsize=10, fontweight='bold', color='skyblue')
        ax2.set_ylabel('Surface Coverage (%)', fontsize=10, fontweight='bold', color='lightcoral')
        ax.set_title('Percentile-Based Cone Coverage\nBlue=Your Data | Orange=Uniform Baseline | Red=Surface %', fontsize=10)
        ax.set_xticks(x)
        ax.set_xticklabels([f'{p}%' for p in percentiles])
        ax.tick_params(axis='y', labelcolor='skyblue')
        ax2.tick_params(axis='y', labelcolor='lightcoral')
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim([0, 180])  # Max cone angle is 180°
        ax2.set_ylim([0, 100])  # Max surface coverage is 100%
        
        # Combine legends
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=7)
        
        plt.tight_layout()
        plt.savefig(layer_specific_dir / f"layer_{layer_idx}_sphere_coverage.png", dpi=150, bbox_inches='tight')
        plt.close()
    
    print(f"   ✓ Completed sphere coverage plots for {n_layers} layers")


def create_layer_aggregation_plots(results: Dict, exp_config: Dict, output_dir: Path):
    """
    Create aggregation plots showing how metrics evolve across layers.
    
    Plots:
    1. Concentration (R) across all layers
    2. Cone angle coverage (90%, 95%, 99%) across all layers  
    3. Surface coverage (90%, 95%, 99%) across all layers
    4. Torus detection metrics across all layers
    """
    n_layers = len(results)
    model_short = exp_config['model'].split('/')[-1]
    dataset_short = exp_config['dataset'].split('/')[-1]
    
    print(f"\n📈 Creating layer aggregation plots...")
    
    layers = list(range(n_layers))
    
    # Extract metrics for each layer
    R_values = [results[i]['mean_resultant_length'] for i in layers]
    sigma_values = [results[i]['angular_std_deg'] for i in layers]
    
    # Extract percentile data
    percentiles_50 = [results[i].get('percentile_50_deg', results[i]['median_deg']) for i in layers]
    percentiles_90 = [results[i].get('percentile_90_deg', 0) for i in layers]
    percentiles_95 = [results[i].get('percentile_95_deg', 0) for i in layers]
    percentiles_99 = [results[i].get('percentile_99_deg', 0) for i in layers]
    percentiles_100 = [results[i].get('percentile_100_deg', 0) for i in layers]
    
    # Extract torus detection metrics
    mean_min_pairwise = []
    mean_to_center = []
    min_to_center = []
    hole_indicator = []
    surface_coverage_95 = []
    
    for i in layers:
        pairwise_data = results[i].get('pairwise_torus_analysis', {})
        mean_min_pairwise.append(pairwise_data.get('mean_min_pairwise_deg', 0))
        mean_to_center.append(pairwise_data.get('mean_distance_to_mean_direction_deg', 0))
        min_to_center.append(pairwise_data.get('min_distance_to_mean_direction_deg', 0))
        hole_indicator.append(pairwise_data.get('hole_indicator_deg', 0))
        surface_coverage_95.append(pairwise_data.get('surface_coverage_95pct', 0))
    
    # Compute surface coverages from percentile angles
    surf_90 = [(1 - np.cos(np.radians(a))) / 2 * 100 for a in percentiles_90]
    surf_95 = [(1 - np.cos(np.radians(a))) / 2 * 100 for a in percentiles_95]
    surf_99 = [(1 - np.cos(np.radians(a))) / 2 * 100 for a in percentiles_99]
    surf_100 = [(1 - np.cos(np.radians(a))) / 2 * 100 for a in percentiles_100]
    
    # Plot 1: Concentration (R) evolution with uniform baseline and probability plot
    # Compute uniform baseline R distribution
    # IMPORTANT: Use n_samples from the actual data - this determines the probability distribution
    n_samples = results[0]['n_samples']  # Get sample size from first layer (all layers should have same n_samples)
    d_model = results[0].get('d_model', 768)  # Get d_model if available, default 768
    
    # Verify all layers have the same n_samples (they should, but check for safety)
    for layer_idx in layers:
        if results[layer_idx]['n_samples'] != n_samples:
            print(f"   ⚠️  Warning: Layer {layer_idx} has n_samples={results[layer_idx]['n_samples']}, "
                  f"expected {n_samples}. Using {n_samples} for uniform baseline.")
    
    # Compute uniform baseline distribution for this specific n_samples
    # This accounts for the dataset size in the probability calculations
    uniform_baseline = compute_uniform_baseline_R(n_samples, d_model=d_model)
    
    # Compute probabilities for each layer's R value
    # P(R >= observed_R | uniform, n=n_samples) - accounts for dataset size!
    layer_probs = []
    for R_val in R_values:
        prob = compute_uniform_probability(R_val, uniform_baseline)
        layer_probs.append(prob)
    
    # Create figure with two subplots side by side
    fig = plt.figure(figsize=(16, 8))
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], width_ratios=[2, 1], hspace=0.3, wspace=0.3)
    
    # Left plot: Concentration evolution
    ax = fig.add_subplot(gs[0, 0])
    
    # Plot actual R values
    ax.plot(layers, R_values, marker='o', linewidth=2, markersize=8, color='steelblue', label='Mean Resultant Length (R)', zorder=3)
    
    # Add uniform baseline (only show 1st and 5th percentiles since actual R is much higher)
    uniform_mean = uniform_baseline['mean']
    uniform_p1 = uniform_baseline['percentiles'][1]
    uniform_p5 = uniform_baseline['percentiles'][5]
    
    ax.axhline(y=uniform_mean, color='gray', linestyle='--', linewidth=2, alpha=0.7, 
               label=f'Uniform baseline (mean: {uniform_mean:.6f})', zorder=2)
    ax.fill_between(layers, [uniform_p1] * len(layers), [uniform_p5] * len(layers), 
                    color='gray', alpha=0.2, label=f'Uniform range (1st-5th percentile)', zorder=1)
    
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('R (Concentration)', fontsize=12, fontweight='bold')
    ax.set_title(f'Concentration Evolution Across Layers\n{model_short} on {dataset_short}', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10, loc='best')
    ax.set_ylim([0, 1.1])
    
    # Right plot: Probability bar plot
    ax_prob = fig.add_subplot(gs[0, 1])
    
    # Create bar plot for probabilities
    colors = ['red' if p < 0.01 else 'orange' if p < 0.05 else 'yellow' if p < 0.1 else 'lightblue' for p in layer_probs]
    bars = ax_prob.barh(layers, layer_probs, color=colors, alpha=0.7, edgecolor='black')
    
    # Add reference probability lines for R=0.1, 0.3, 0.5, 0.8
    ref_R_values = [0.1, 0.3, 0.5, 0.8]
    ref_colors = ['green', 'blue', 'purple', 'red']
    ref_probs = []
    ref_labels = []
    for ref_R, ref_color in zip(ref_R_values, ref_colors):
        ref_prob = uniform_baseline['probabilities'][ref_R]
        ref_probs.append(ref_prob)
        ref_labels.append(f'R={ref_R} (P={ref_prob:.2e})')
        ax_prob.axvline(x=ref_prob, color=ref_color, linestyle='--', linewidth=2, alpha=0.6, 
                       label=f'R={ref_R} (P={ref_prob:.2e})')
    
    ax_prob.set_xlabel(f'P(R ≥ observed_R | uniform, n={n_samples:,})', fontsize=12, fontweight='bold')
    ax_prob.set_ylabel('Layer', fontsize=12, fontweight='bold')
    ax_prob.set_title('Probability of Observing R\nunder Uniform Distribution', fontsize=14, fontweight='bold')
    ax_prob.set_xscale('log')  # Log scale since probabilities are very small
    ax_prob.grid(alpha=0.3, axis='x')
    ax_prob.legend(fontsize=9, loc='best')
    ax_prob.invert_yaxis()  # Match layer ordering
    
    # Add probability values as text on bars
    for i, (bar, prob) in enumerate(zip(bars, layer_probs)):
        if prob > 0:
            ax_prob.text(prob * 1.1, layers[i], f'{prob:.2e}', 
                        fontsize=8, verticalalignment='center', fontweight='bold')
    
    # Bottom left: Uniform baseline statistics table
    ax_stats = fig.add_subplot(gs[1, 0])
    ax_stats.axis('off')
    stats_text = f"Uniform Baseline Statistics (n={n_samples:,} samples, d={d_model}):\n"
    stats_text += f"  IMPORTANT: Probabilities account for dataset size (n={n_samples:,})!\n"
    stats_text += f"  Mean R: {uniform_mean:.6f} ± {uniform_baseline['std']:.6f}\n"
    stats_text += f"  Percentiles: 1st={uniform_baseline['percentiles'][1]:.6f}, "
    stats_text += f"5th={uniform_baseline['percentiles'][5]:.6f}, "
    stats_text += f"95th={uniform_baseline['percentiles'][95]:.6f}, "
    stats_text += f"99th={uniform_baseline['percentiles'][99]:.6f}\n"
    stats_text += f"  Reference Probabilities (for n={n_samples:,}):\n"
    for ref_R, ref_prob in zip(ref_R_values, ref_probs):
        stats_text += f"    P(R ≥ {ref_R} | uniform, n={n_samples:,}) = {ref_prob:.2e}\n"
    stats_text += f"  Note: For uniform distribution on sphere, R → 0 as n → ∞"
    
    ax_stats.text(0.5, 0.5, stats_text, transform=ax_stats.transAxes,
             fontsize=9, verticalalignment='center', horizontalalignment='center',
             family='monospace', bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    
    # Bottom right: Reference probability explanation
    ax_ref = fig.add_subplot(gs[1, 1])
    ax_ref.axis('off')
    ref_text = "Reference Probabilities:\n"
    ref_text += f"P(R ≥ X | uniform, n={n_samples:,})\n"
    ref_text += "means: Probability that\n"
    ref_text += "uniform distribution\n"
    ref_text += f"would produce R ≥ X\n"
    ref_text += f"with n={n_samples:,} samples.\n\n"
    ref_text += "Lower probability = more\n"
    ref_text += "unlikely under uniform,\n"
    ref_text += "indicating stronger\n"
    ref_text += "concentration/clustering.\n\n"
    ref_text += "NOTE: Probabilities\n"
    ref_text += "account for dataset size!"
    
    ax_ref.text(0.5, 0.5, ref_text, transform=ax_ref.transAxes,
             fontsize=10, verticalalignment='center', horizontalalignment='center',
             bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.3))
    
    plt.savefig(output_dir / "concentration_evolution_all_layers.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved concentration_evolution_all_layers.png")
    
    # Plot 2: Cone angle coverage evolution
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(layers, percentiles_90, marker='o', linewidth=2, markersize=6, label='90th percentile', alpha=0.7)
    ax.plot(layers, percentiles_95, marker='s', linewidth=2, markersize=6, label='95th percentile', alpha=0.7)
    ax.plot(layers, percentiles_99, marker='^', linewidth=2, markersize=6, label='99th percentile', alpha=0.7)
    ax.plot(layers, percentiles_100, marker='D', linewidth=2, markersize=6, label='100th percentile', alpha=0.7)
    ax.axhline(y=87, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Uniform baseline (87°)')
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Cone Angle (degrees)', fontsize=12, fontweight='bold')
    ax.set_title(f'Cone Angle Coverage Evolution Across Layers\n{model_short} on {dataset_short}', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 180])
    plt.tight_layout()
    plt.savefig(output_dir / "cone_angle_evolution_all_layers.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved cone_angle_evolution_all_layers.png")
    
    # Plot 3: Surface coverage evolution
    # Calculate uniform baseline for each percentile
    # For uniform distribution: 90th percentile ≈ 87°, 95th percentile ≈ 90°, 99th percentile ≈ 95°, 100th percentile = 180°
    uniform_surf_90 = (1 - np.cos(np.radians(87))) / 2 * 100
    uniform_surf_95 = (1 - np.cos(np.radians(90))) / 2 * 100
    uniform_surf_99 = (1 - np.cos(np.radians(95))) / 2 * 100
    uniform_surf_100 = (1 - np.cos(np.radians(180))) / 2 * 100  # 100% coverage at 180°
    
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    ax.plot(layers, surf_90, marker='o', linewidth=2, markersize=6, label='90th percentile', alpha=0.7)
    ax.plot(layers, surf_95, marker='s', linewidth=2, markersize=6, label='95th percentile', alpha=0.7)
    ax.plot(layers, surf_99, marker='^', linewidth=2, markersize=6, label='99th percentile', alpha=0.7)
    ax.plot(layers, surf_100, marker='D', linewidth=2, markersize=6, label='100th percentile', alpha=0.7)
    ax.axhline(y=uniform_surf_90, color='red', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Uniform baseline (90th: {uniform_surf_90:.1f}%)')
    ax.axhline(y=uniform_surf_95, color='orange', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Uniform baseline (95th: {uniform_surf_95:.1f}%)')
    ax.axhline(y=uniform_surf_99, color='purple', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Uniform baseline (99th: {uniform_surf_99:.1f}%)')
    ax.axhline(y=uniform_surf_100, color='green', linestyle='--', linewidth=1.5, alpha=0.6, label=f'Uniform baseline (100th: {uniform_surf_100:.1f}%)')
    ax.set_xlabel('Layer', fontsize=12, fontweight='bold')
    ax.set_ylabel('Surface Coverage (%)', fontsize=12, fontweight='bold')
    ax.set_title(f'Surface Coverage Evolution Across Layers\n{model_short} on {dataset_short}', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    ax.set_ylim([0, 100])
    plt.tight_layout()
    plt.savefig(output_dir / "surface_coverage_evolution_all_layers.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved surface_coverage_evolution_all_layers.png")
    
    # Plot 4: Torus detection metrics evolution
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Torus Detection Metrics Evolution\n{model_short} on {dataset_short}', fontsize=14, fontweight='bold')
    
    # Subplot 1: Mean nearest neighbor vs MIN distance to center
    ax = axes[0, 0]
    ax.plot(layers, mean_min_pairwise, marker='o', linewidth=2, markersize=6, label='Mean nearest neighbor', color='green')
    ax.plot(layers, min_to_center, marker='s', linewidth=2, markersize=6, label='MIN distance to mean', color='red')
    ax.set_xlabel('Layer', fontsize=10, fontweight='bold')
    ax.set_ylabel('Angular Distance (degrees)', fontsize=10, fontweight='bold')
    ax.set_title('Hole Detection: MIN to Mean vs Mean NN', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Subplot 2: Hole indicator (CORRECTED)
    ax = axes[0, 1]
    ax.plot(layers, hole_indicator, marker='o', linewidth=2, markersize=6, color='purple')
    ax.axhline(y=0, color='red', linestyle='--', linewidth=1, alpha=0.5, label='No hole (0)')
    ax.axhline(y=20, color='orange', linestyle='--', linewidth=1, alpha=0.7, label='Hole threshold (20°)')
    ax.set_xlabel('Layer', fontsize=10, fontweight='bold')
    ax.set_ylabel('Hole Indicator (degrees)', fontsize=10, fontweight='bold')
    ax.set_title('Hole Indicator\n(MIN to Mean - Mean NN)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    
    # Subplot 3: Surface coverage (95%)
    ax = axes[1, 0]
    ax.plot(layers, surface_coverage_95, marker='o', linewidth=2, markersize=6, color='coral')
    ax.set_xlabel('Layer', fontsize=10, fontweight='bold')
    ax.set_ylabel('Surface Coverage (%)', fontsize=10, fontweight='bold')
    ax.set_title('95% Surface Coverage\n(Lower = More Clustered)', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 100])
    
    # Subplot 4: Sigma vs uniform baseline
    ax = axes[1, 1]
    ax.plot(layers, sigma_values, marker='o', linewidth=2, markersize=6, label='σ (Angular Std Dev)', color='darkblue')
    ax.axhline(y=87, color='red', linestyle='--', linewidth=2, alpha=0.6, label='Uniform baseline (87°)')
    ax.set_xlabel('Layer', fontsize=10, fontweight='bold')
    ax.set_ylabel('σ (degrees)', fontsize=10, fontweight='bold')
    ax.set_title('Angular Std Dev vs Uniform', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 100])
    
    plt.tight_layout()
    plt.savefig(output_dir / "torus_detection_evolution_all_layers.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved torus_detection_evolution_all_layers.png")
    
    print(f"   ✓ Completed {n_layers} layer aggregation plots")


def create_magnitude_distribution_plots(results: Dict, exp_config: Dict, output_dir: Path):
    """
    Create magnitude distribution plots for all layers in one PNG.
    
    Plots:
    1. Non-normalized magnitudes (all layers, same scale)
    2. Normalized magnitudes (all layers, same scale)
    """
    n_layers = len(results)
    model_short = exp_config['model'].split('/')[-1]
    dataset_short = exp_config['dataset'].split('/')[-1]
    
    print(f"\n📊 Creating magnitude distribution plots...")
    
    # Determine number of rows and columns for subplots
    n_cols = 4
    n_rows = (n_layers + n_cols - 1) // n_cols
    
    # Plot 1: Non-normalized magnitudes
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    fig.suptitle(f'Magnitude Distribution (Non-Normalized) - All Layers\n{model_short} on {dataset_short}', 
                 fontsize=14, fontweight='bold')
    axes = axes.flatten() if n_layers > 1 else [axes]
    
    # Find global min and max magnitudes for consistent scaling
    all_magnitudes = []
    for i in range(n_layers):
        magnitudes = results[i].get('magnitude_distribution', [])
        if len(magnitudes) > 0:
            all_magnitudes.extend(magnitudes)
    
    if len(all_magnitudes) > 0:
        global_min = min(all_magnitudes)
        global_max = max(all_magnitudes)
        
        for layer_idx in range(n_layers):
            ax = axes[layer_idx]
            magnitudes = results[layer_idx].get('magnitude_distribution', [])
            
            if len(magnitudes) > 0:
                ax.hist(magnitudes, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
                ax.set_title(f'Layer {layer_idx}', fontsize=10, fontweight='bold')
                ax.set_xlabel('Magnitude', fontsize=9)
                ax.set_ylabel('Count', fontsize=9)
                ax.set_xlim([global_min, global_max])
                ax.grid(alpha=0.3)
                
                # Add mean line
                mean_mag = np.mean(magnitudes)
                ax.axvline(mean_mag, color='red', linestyle='--', linewidth=1, alpha=0.7, 
                          label=f'Mean: {mean_mag:.2f}')
                ax.legend(fontsize=7)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
                ax.set_title(f'Layer {layer_idx}', fontsize=10)
        
        # Hide empty subplots
        for idx in range(n_layers, len(axes)):
            axes[idx].axis('off')
        
        plt.tight_layout()
        plt.savefig(output_dir / "magnitude_distribution_all_layers.png", dpi=150, bbox_inches='tight')
        plt.close()
        print(f"   ✓ Saved magnitude_distribution_all_layers.png")
    else:
        print(f"   ⚠️  No magnitude data available for plotting")
    
    # Plot 2: Normalized magnitudes (each layer normalized to [0,1])
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    fig.suptitle(f'Magnitude Distribution (Normalized) - All Layers\n{model_short} on {dataset_short}', 
                 fontsize=14, fontweight='bold')
    axes = axes.flatten() if n_layers > 1 else [axes]
    
    for layer_idx in range(n_layers):
        ax = axes[layer_idx]
        magnitudes = results[layer_idx].get('magnitude_distribution', [])
        
        if len(magnitudes) > 0:
            # Normalize to [0, 1]
            mag_array = np.array(magnitudes)
            mag_min = mag_array.min()
            mag_max = mag_array.max()
            if mag_max > mag_min:
                normalized_magnitudes = (mag_array - mag_min) / (mag_max - mag_min)
            else:
                normalized_magnitudes = np.zeros_like(mag_array)
            
            ax.hist(normalized_magnitudes, bins=50, alpha=0.7, color='coral', edgecolor='black')
            ax.set_title(f'Layer {layer_idx}', fontsize=10, fontweight='bold')
            ax.set_xlabel('Normalized Magnitude', fontsize=9)
            ax.set_ylabel('Count', fontsize=9)
            ax.set_xlim([0, 1])
            ax.grid(alpha=0.3)
            
            # Add mean line
            mean_norm = np.mean(normalized_magnitudes)
            ax.axvline(mean_norm, color='red', linestyle='--', linewidth=1, alpha=0.7, 
                      label=f'Mean: {mean_norm:.2f}')
            ax.legend(fontsize=7)
        else:
            ax.text(0.5, 0.5, 'No data', ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f'Layer {layer_idx}', fontsize=10)
    
    # Hide empty subplots
    for idx in range(n_layers, len(axes)):
        axes[idx].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / "magnitude_distribution_normalized_all_layers.png", dpi=150, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved magnitude_distribution_normalized_all_layers.png")
    
    print(f"   ✓ Completed magnitude distribution plots for {n_layers} layers")


def create_layer_to_layer_distance_plots(results: Dict, exp_config: Dict, output_dir: Path):
    """
    Create plots showing angular and euclidean distance changes between consecutive layers.
    Shows distance from layer n to layer n+1 (both before and after normalization).
    """
    n_layers = len(results)
    
    if n_layers < 2:
        print("   ⚠️  Need at least 2 layers for layer-to-layer distances")
        return
    
    model_short = exp_config['model'].split('/')[-1]
    dataset_short = exp_config['dataset'].split('/')[-1]
    
    # Compute distances between consecutive layers
    angular_dists_before = []
    angular_dists_after = []
    euclidean_dists_before = []
    euclidean_dists_after = []
    layer_transitions = []
    
    for layer_idx in range(n_layers - 1):
        curr_layer = results[layer_idx]
        next_layer = results[layer_idx + 1]
        
        # Get mean vectors (before normalization) - convert from list to array
        curr_mean = np.array(curr_layer['mean_vector'])
        next_mean = np.array(next_layer['mean_vector'])
        
        # Get mean directions (after normalization) - convert from list to array
        curr_dir = np.array(curr_layer['mean_direction'])
        next_dir = np.array(next_layer['mean_direction'])
        
        # Euclidean distance BEFORE normalization
        euc_before = float(np.linalg.norm(curr_mean - next_mean))
        euclidean_dists_before.append(euc_before)
        
        # Euclidean distance AFTER normalization
        euc_after = float(np.linalg.norm(curr_dir - next_dir))
        euclidean_dists_after.append(euc_after)
        
        # Angular distance (need to normalize for angle calculation)
        # BEFORE normalization: normalize the mean vectors
        curr_mean_norm = curr_mean / np.linalg.norm(curr_mean)
        next_mean_norm = next_mean / np.linalg.norm(next_mean)
        dot_before = np.clip(np.dot(curr_mean_norm, next_mean_norm), -1.0, 1.0)
        ang_before = float(np.degrees(np.arccos(dot_before)))
        angular_dists_before.append(ang_before)
        
        # AFTER normalization: already normalized
        dot_after = np.clip(np.dot(curr_dir, next_dir), -1.0, 1.0)
        ang_after = float(np.degrees(np.arccos(dot_after)))
        angular_dists_after.append(ang_after)
        
        layer_transitions.append(f"L{layer_idx}→L{layer_idx+1}")
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle(f'Layer-to-Layer Mean Direction Distances\n{model_short} on {dataset_short}',
                 fontsize=14, fontweight='bold')
    
    x_pos = np.arange(len(layer_transitions))
    
    # Plot 1: Angular distances
    ax = axes[0, 0]
    ax.plot(x_pos, angular_dists_before, 'o-', label='Before Normalization', 
            color='lightblue', linewidth=2, markersize=8)
    ax.plot(x_pos, angular_dists_after, 's-', label='After Normalization', 
            color='darkblue', linewidth=2, markersize=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(layer_transitions, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Angular Distance (degrees)', fontsize=11, fontweight='bold')
    ax.set_title('Angular Distance Between Consecutive Layers', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Plot 2: Euclidean distances
    ax = axes[0, 1]
    ax.plot(x_pos, euclidean_dists_before, 'o-', label='Before Normalization', 
            color='lightcoral', linewidth=2, markersize=8)
    ax.plot(x_pos, euclidean_dists_after, 's-', label='After Normalization', 
            color='darkred', linewidth=2, markersize=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(layer_transitions, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Euclidean Distance', fontsize=11, fontweight='bold')
    ax.set_title('Euclidean Distance Between Consecutive Layers', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Plot 3: Combined comparison - Angular
    ax = axes[1, 0]
    width = 0.35
    ax.bar(x_pos - width/2, angular_dists_before, width, label='Before Norm', 
           color='lightblue', alpha=0.7, edgecolor='black')
    ax.bar(x_pos + width/2, angular_dists_after, width, label='After Norm', 
           color='darkblue', alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(layer_transitions, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Angular Distance (degrees)', fontsize=11, fontweight='bold')
    ax.set_title('Angular Distance Comparison', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Combined comparison - Euclidean
    ax = axes[1, 1]
    ax.bar(x_pos - width/2, euclidean_dists_before, width, label='Before Norm', 
           color='lightcoral', alpha=0.7, edgecolor='black')
    ax.bar(x_pos + width/2, euclidean_dists_after, width, label='After Norm', 
           color='darkred', alpha=0.7, edgecolor='black')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(layer_transitions, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel('Euclidean Distance', fontsize=11, fontweight='bold')
    ax.set_title('Euclidean Distance Comparison', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "layer_to_layer_distances.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved layer_to_layer_distances.png")


def create_anchor_distance_plots(results: Dict, exp_config: Dict, output_dir: Path):
    """
    Create plots showing distances from anchor layers to all other layers.
    Anchor layers:
    - Layer 0 (embedding)
    - Layer 1 (first block) 
    - Layer N-1 (last layer)
    
    Shows both angular and euclidean distances, before and after normalization.
    """
    n_layers = len(results)
    
    if n_layers < 2:
        print("   ⚠️  Need at least 2 layers for anchor distance plots")
        return
    
    model_short = exp_config['model'].split('/')[-1]
    dataset_short = exp_config['dataset'].split('/')[-1]
    
    # Compute distances from embedding layer (layer 0) to all others
    embedding_angular_before = []
    embedding_angular_after = []
    embedding_euclidean_before = []
    embedding_euclidean_after = []
    
    # Compute distances from first block (layer 1) to all others
    first_block_angular_before = []
    first_block_angular_after = []
    first_block_euclidean_before = []
    first_block_euclidean_after = []
    
    # Compute distances from last layer to all others
    last_layer_angular_before = []
    last_layer_angular_after = []
    last_layer_euclidean_before = []
    last_layer_euclidean_after = []
    
    embedding_mean = np.array(results[0]['mean_vector'])
    embedding_dir = np.array(results[0]['mean_direction'])
    first_block_mean = np.array(results[1]['mean_vector'])
    first_block_dir = np.array(results[1]['mean_direction'])
    last_mean = np.array(results[n_layers - 1]['mean_vector'])
    last_dir = np.array(results[n_layers - 1]['mean_direction'])
    
    for layer_idx in range(n_layers):
        curr_mean = np.array(results[layer_idx]['mean_vector'])
        curr_dir = np.array(results[layer_idx]['mean_direction'])
        curr_mean_norm = curr_mean / np.linalg.norm(curr_mean)
        
        # Distances from embedding layer (layer 0)
        euc_before_emb = float(np.linalg.norm(embedding_mean - curr_mean))
        euc_after_emb = float(np.linalg.norm(embedding_dir - curr_dir))
        embedding_euclidean_before.append(euc_before_emb)
        embedding_euclidean_after.append(euc_after_emb)
        
        embedding_mean_norm = embedding_mean / np.linalg.norm(embedding_mean)
        dot_before_emb = np.clip(np.dot(embedding_mean_norm, curr_mean_norm), -1.0, 1.0)
        ang_before_emb = float(np.degrees(np.arccos(dot_before_emb)))
        embedding_angular_before.append(ang_before_emb)
        
        dot_after_emb = np.clip(np.dot(embedding_dir, curr_dir), -1.0, 1.0)
        ang_after_emb = float(np.degrees(np.arccos(dot_after_emb)))
        embedding_angular_after.append(ang_after_emb)
        
        # Distances from first block (layer 1)
        euc_before_first = float(np.linalg.norm(first_block_mean - curr_mean))
        euc_after_first = float(np.linalg.norm(first_block_dir - curr_dir))
        first_block_euclidean_before.append(euc_before_first)
        first_block_euclidean_after.append(euc_after_first)
        
        first_block_mean_norm = first_block_mean / np.linalg.norm(first_block_mean)
        dot_before_first = np.clip(np.dot(first_block_mean_norm, curr_mean_norm), -1.0, 1.0)
        ang_before_first = float(np.degrees(np.arccos(dot_before_first)))
        first_block_angular_before.append(ang_before_first)
        
        dot_after_first = np.clip(np.dot(first_block_dir, curr_dir), -1.0, 1.0)
        ang_after_first = float(np.degrees(np.arccos(dot_after_first)))
        first_block_angular_after.append(ang_after_first)
        
        # Distances from last layer
        euc_before_last = float(np.linalg.norm(last_mean - curr_mean))
        euc_after_last = float(np.linalg.norm(last_dir - curr_dir))
        last_layer_euclidean_before.append(euc_before_last)
        last_layer_euclidean_after.append(euc_after_last)
        
        last_mean_norm = last_mean / np.linalg.norm(last_mean)
        dot_before_last = np.clip(np.dot(last_mean_norm, curr_mean_norm), -1.0, 1.0)
        ang_before_last = float(np.degrees(np.arccos(dot_before_last)))
        last_layer_angular_before.append(ang_before_last)
        
        dot_after_last = np.clip(np.dot(last_dir, curr_dir), -1.0, 1.0)
        ang_after_last = float(np.degrees(np.arccos(dot_after_last)))
        last_layer_angular_after.append(ang_after_last)
    
    # Create visualizations (3x2 grid for 3 anchor layers)
    fig, axes = plt.subplots(3, 2, figsize=(16, 16))
    fig.suptitle(f'Distance from Anchor Layers to All Layers\n{model_short} on {dataset_short}\n(L0=Embedding, L1=First Block, L{n_layers-1}=Last Layer)',
                 fontsize=14, fontweight='bold')
    
    layer_indices = list(range(n_layers))
    
    # Row 1: Distances from EMBEDDING (Layer 0)
    ax = axes[0, 0]
    ax.plot(layer_indices, embedding_angular_before, 'o-', label='Before Norm', 
            color='lightcoral', linewidth=2, markersize=8)
    ax.plot(layer_indices, embedding_angular_after, 's-', label='After Norm', 
            color='darkred', linewidth=2, markersize=8)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5, label='Embedding')
    ax.axvline(x=1, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='First Block')
    ax.axvline(x=n_layers-1, color='magenta', linestyle='--', alpha=0.5, linewidth=1.5, label='Last Layer')
    ax.set_xlabel('Layer Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Angular Distance from L0 (degrees)', fontsize=11, fontweight='bold')
    ax.set_title('Angular Distance from EMBEDDING (L0)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=8, ncol=2)
    ax.grid(alpha=0.3)
    
    ax = axes[0, 1]
    ax.plot(layer_indices, embedding_euclidean_before, 'o-', label='Before Norm', 
            color='lightcoral', linewidth=2, markersize=8)
    ax.plot(layer_indices, embedding_euclidean_after, 's-', label='After Norm', 
            color='darkred', linewidth=2, markersize=8)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axvline(x=1, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axvline(x=n_layers-1, color='magenta', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Layer Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Euclidean Distance from L0', fontsize=11, fontweight='bold')
    ax.set_title('Euclidean Distance from EMBEDDING (L0)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Row 2: Distances from FIRST BLOCK (Layer 1)
    ax = axes[1, 0]
    ax.plot(layer_indices, first_block_angular_before, 'o-', label='Before Norm', 
            color='lightyellow', linewidth=2, markersize=8, markeredgecolor='black')
    ax.plot(layer_indices, first_block_angular_after, 's-', label='After Norm', 
            color='orange', linewidth=2, markersize=8, markeredgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axvline(x=1, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, label='First Block')
    ax.axvline(x=n_layers-1, color='magenta', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Layer Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Angular Distance from L1 (degrees)', fontsize=11, fontweight='bold')
    ax.set_title('Angular Distance from FIRST BLOCK (L1)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    ax = axes[1, 1]
    ax.plot(layer_indices, first_block_euclidean_before, 'o-', label='Before Norm', 
            color='lightyellow', linewidth=2, markersize=8, markeredgecolor='black')
    ax.plot(layer_indices, first_block_euclidean_after, 's-', label='After Norm', 
            color='orange', linewidth=2, markersize=8, markeredgecolor='black')
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axvline(x=1, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axvline(x=n_layers-1, color='magenta', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Layer Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Euclidean Distance from L1', fontsize=11, fontweight='bold')
    ax.set_title('Euclidean Distance from FIRST BLOCK (L1)', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    # Row 3: Distances from LAST LAYER
    ax = axes[2, 0]
    ax.plot(layer_indices, last_layer_angular_before, 'o-', label='Before Norm', 
            color='plum', linewidth=2, markersize=8)
    ax.plot(layer_indices, last_layer_angular_after, 's-', label='After Norm', 
            color='darkmagenta', linewidth=2, markersize=8)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axvline(x=1, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axvline(x=n_layers-1, color='magenta', linestyle='--', alpha=0.5, linewidth=1.5, label='Last Layer')
    ax.set_xlabel('Layer Index', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'Angular Distance from L{n_layers-1} (degrees)', fontsize=11, fontweight='bold')
    ax.set_title(f'Angular Distance from LAST LAYER (L{n_layers-1})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    ax = axes[2, 1]
    ax.plot(layer_indices, last_layer_euclidean_before, 'o-', label='Before Norm', 
            color='plum', linewidth=2, markersize=8)
    ax.plot(layer_indices, last_layer_euclidean_after, 's-', label='After Norm', 
            color='darkmagenta', linewidth=2, markersize=8)
    ax.axvline(x=0, color='red', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axvline(x=1, color='orange', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.axvline(x=n_layers-1, color='magenta', linestyle='--', alpha=0.5, linewidth=1.5)
    ax.set_xlabel('Layer Index', fontsize=11, fontweight='bold')
    ax.set_ylabel(f'Euclidean Distance from L{n_layers-1}', fontsize=11, fontweight='bold')
    ax.set_title(f'Euclidean Distance from LAST LAYER (L{n_layers-1})', fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_dir / "anchor_layer_distances.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"   ✓ Saved anchor_layer_distances.png")


def run_single_experiment(exp_config, base_output_dir="/data/user/sae/activation_clustering_behavior1"):
    """Run layer evolution analysis for a single model/dataset pair."""
    
    print("\n" + "="*80)
    print(f"🔬 EXPERIMENT: {exp_config['model']} on {exp_config['dataset']}")
    print("="*80)
    
    model_short = exp_config['model'].split('/')[-1]
    dataset_short = exp_config['dataset'].split('/')[-1]
    checkpoint_index = exp_config.get('checkpoint_index', None)
    if checkpoint_index is not None:
        exp_name = f"{model_short}_ckpt{checkpoint_index}_{dataset_short}"
    else:
        exp_name = f"{model_short}_{dataset_short}"
    
    output_dir = Path(base_output_dir) / exp_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        start_time = time.time()
        
        # Step 1: Setup
        print("\n📦 Step 1: Loading model and dataset...")
        checkpoint_index = exp_config.get('checkpoint_index', None)
        model, token_dataset, device = setup_model_and_data(
            model_name=exp_config['model'],
            dataset_name=exp_config['dataset'],
            cuda_device=CUDA_DEVICE,
            checkpoint_index=checkpoint_index
        )
        
        n_layers = model.cfg.n_layers
        d_model = model.cfg.d_model
        print(f"Model has {n_layers} layers, d_model={d_model}")
        
        # Step 2: Collect activations from ALL layers
        print("\n📊 Step 2: Collecting activations from all layers...")
        layer_activations, all_tokens, all_sequences = collect_all_layers_activations(
            model=model,
            token_dataset=token_dataset,
            num_batches=exp_config['num_batches'],
            batch_size=exp_config['batch_size']
        )
        
        # Step 3: Analyze evolution
        print("\n🌐 Step 3: Analyzing angular distribution evolution...")
        evolution_results = analyze_layer_evolution(layer_activations)
        
        # Step 4: Create visualizations
        print("\n📈 Step 4: Creating visualizations...")
        create_evolution_visualization(evolution_results, exp_config, output_dir)
        create_individual_layer_plots(evolution_results, exp_config, output_dir)
        create_magnitude_by_layer_plots(evolution_results, layer_activations, exp_config, output_dir)
        create_layer_to_layer_distance_plots(evolution_results, exp_config, output_dir)
        create_anchor_distance_plots(evolution_results, exp_config, output_dir)
        create_uniformity_plots_layers(evolution_results, exp_config, output_dir)
        create_pairwise_torus_plots_layers(evolution_results, exp_config, output_dir)
        create_sphere_coverage_plots_layers(evolution_results, exp_config, output_dir)
        
        # Create layer aggregation plots
        create_layer_aggregation_plots(evolution_results, exp_config, output_dir)
        
        # Create magnitude distribution plots
        create_magnitude_distribution_plots(evolution_results, exp_config, output_dir)
        
        # Compile and save results
        results = {
            "experiment_name": exp_name,
            "model": exp_config['model'],
            "dataset": exp_config['dataset'],
            "n_layers": n_layers,
            "d_model": d_model,
            "n_samples_per_layer": evolution_results[0]['n_samples'],
            "layer_statistics": {
                layer_idx: {
                    k: v for k, v in stats.items() 
                    if k not in ['angular_distances_deg', 'within_shell_uniformity', 'pairwise_torus_analysis']  # Don't save full arrays or nested data with arrays
                }
                for layer_idx, stats in evolution_results.items()
            },
            "elapsed_time_seconds": time.time() - start_time,
            "success": True
        }
        
        with open(output_dir / "evolution_results.json", "w") as f:
            json.dump(results, f, indent=2)
        
        with open(output_dir / "experiment_config.json", "w") as f:
            json.dump(exp_config, f, indent=2)
        
        print(f"\n✅ Experiment completed in {results['elapsed_time_seconds']:.1f} seconds")
        print(f"📁 Results saved to: {output_dir}")
        
        # Print key findings
        actual_n_layers = len(evolution_results)
        print(f"\n📊 Key Findings:")
        print(f"   Layer 0 (embedding):  R={evolution_results[0]['mean_resultant_length']:.3f}, mean_angle={evolution_results[0]['mean_deg']:.1f}°")
        print(f"   Layer 1 (first block): R={evolution_results[1]['mean_resultant_length']:.3f}, mean_angle={evolution_results[1]['mean_deg']:.1f}°")
        mid_layer = actual_n_layers // 2
        print(f"   Layer {mid_layer} (middle):    R={evolution_results[mid_layer]['mean_resultant_length']:.3f}, mean_angle={evolution_results[mid_layer]['mean_deg']:.1f}°")
        print(f"   Layer {actual_n_layers-1} (last layer):   R={evolution_results[actual_n_layers-1]['mean_resultant_length']:.3f}, mean_angle={evolution_results[actual_n_layers-1]['mean_deg']:.1f}°")
        
        # Clean up - ensure everything is freed (models stay on GPU until deleted)
        # Explicitly delete model from GPU memory (model was never saved to disk)
        if torch.cuda.is_available():
            del model  # Delete model from GPU memory
            torch.cuda.empty_cache()
            torch.cuda.synchronize()  # Ensure all CUDA operations are complete
        
        # Delete other objects
        del token_dataset, layer_activations
        import gc
        gc.collect()  # Force garbage collection
        
        return results
        
    except Exception as e:
        print(f"\n❌ Experiment failed: {e}")
        import traceback
        traceback.print_exc()
        
        error_info = {
            "experiment_name": exp_name,
            "model": exp_config['model'],
            "dataset": exp_config['dataset'],
            "success": False,
            "error": str(e),
            "error_type": type(e).__name__
        }
        
        with open(output_dir / "error_log.json", "w") as f:
            json.dump(error_info, f, indent=2)
        
        # Clean up on error - ensure everything is freed (models stay on GPU until deleted)
        try:
            if 'model' in locals():
                if torch.cuda.is_available():
                    del model  # Delete model from GPU memory (never saved to disk)
                    torch.cuda.empty_cache()
                    torch.cuda.synchronize()
                else:
                    del model
            if 'token_dataset' in locals():
                del token_dataset
            if 'layer_activations' in locals():
                del layer_activations
            import gc
            gc.collect()  # Force garbage collection
        except:
            pass  # Ignore cleanup errors
        
        return error_info


def create_comparison_plots(all_results, base_output_dir="/data/user/sae/activation_clustering_behavior1"):
    """Create comparison plots across all experiments."""
    
    print("\n\n" + "="*80)
    print("📊 CREATING CROSS-EXPERIMENT COMPARISON PLOTS")
    print("="*80)
    
    output_dir = Path(base_output_dir) / "comparison_analysis"
    output_dir.mkdir(exist_ok=True)
    
    successful_results = [r for r in all_results if r.get("success", False)]
    
    if not successful_results:
        print("❌ No successful experiments to compare")
        return
    
    # Create comparison figure
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Comparison Across All Model/Dataset Pairs', fontsize=14, fontweight='bold')
    
    # Plot 1: Mean angle evolution
    ax = axes[0, 0]
    for result in successful_results:
        n_layers = result['n_layers']
        layer_indices = list(range(n_layers))
        mean_angles = [result['layer_statistics'][i]['mean_deg'] for i in layer_indices]
        
        label = f"{result['model'].split('/')[-1]} ({result['dataset'].split('/')[-1]})"
        ax.plot(layer_indices, mean_angles, 'o-', label=label, linewidth=2, markersize=4)
    
    ax.set_xlabel('Layer Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Angular Distance (degrees)', fontsize=11, fontweight='bold')
    ax.set_title('Mean Angular Distance Evolution', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    
    # Plot 2: Concentration (R) evolution
    ax = axes[0, 1]
    for result in successful_results:
        n_layers = result['n_layers']
        layer_indices = list(range(n_layers))
        Rs = [result['layer_statistics'][i]['mean_resultant_length'] for i in layer_indices]
        
        label = f"{result['model'].split('/')[-1]} ({result['dataset'].split('/')[-1]})"
        ax.plot(layer_indices, Rs, 'o-', label=label, linewidth=2, markersize=4)
    
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.3, label='Highly concentrated')
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.3, label='Moderate')
    ax.set_xlabel('Layer Index', fontsize=11, fontweight='bold')
    ax.set_ylabel('Mean Resultant Length (R)', fontsize=11, fontweight='bold')
    ax.set_title('Concentration Evolution', fontsize=12)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 3: First vs Last layer comparison
    ax = axes[1, 0]
    exp_names = []
    first_layer_R = []
    last_layer_R = []
    
    for result in successful_results:
        exp_names.append(f"{result['model'].split('/')[-1]}\n{result['dataset'].split('/')[-1][:8]}")
        first_layer_R.append(result['layer_statistics'][0]['mean_resultant_length'])
        last_layer_R.append(result['layer_statistics'][result['n_layers']-1]['mean_resultant_length'])
    
    x = np.arange(len(exp_names))
    width = 0.35
    
    ax.bar(x - width/2, first_layer_R, width, label='First Layer', color='skyblue', alpha=0.8)
    ax.bar(x + width/2, last_layer_R, width, label='Last Layer', color='coral', alpha=0.8)
    
    ax.set_ylabel('Mean Resultant Length (R)', fontsize=11, fontweight='bold')
    ax.set_title('First vs Last Layer Concentration', fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(exp_names, fontsize=8)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 4: Summary statistics table
    ax = axes[1, 1]
    ax.axis('off')
    
    summary_text = "SUMMARY STATISTICS\n\n"
    for result in successful_results:
        name = f"{result['model'].split('/')[-1]} on {result['dataset'].split('/')[-1]}"
        n_layers = result['n_layers']
        first_R = result['layer_statistics'][0]['mean_resultant_length']
        last_R = result['layer_statistics'][n_layers-1]['mean_resultant_length']
        delta_R = last_R - first_R
        
        summary_text += f"{name}:\n"
        summary_text += f"  Layers: {n_layers}\n"
        summary_text += f"  First R: {first_R:.3f}\n"
        summary_text += f"  Last R: {last_R:.3f}\n"
        summary_text += f"  ΔR: {delta_R:+.3f}\n\n"
    
    ax.text(0.05, 0.95, summary_text, transform=ax.transAxes,
            fontsize=9, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / "cross_experiment_comparison.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Comparison plots saved to: {output_dir}")


def create_aggregate_analysis(all_results, base_output_dir="/data/user/sae/activation_clustering_behavior1"):
    """
    Create aggregate analysis across all models using layer binning.
    
    Since different models have different numbers of layers, we bin layers into 5 categories:
    - earlier_early (0-20%)
    - early_mid (20-40%)
    - mid (40-60%)
    - later_mid (60-80%)
    - later (80-100%)
    """
    
    print("\n\n" + "="*80)
    print("📊 CREATING AGGREGATE ANALYSIS WITH LAYER BINNING")
    print("="*80)
    
    output_dir = Path(base_output_dir) / "aggregate_analysis"
    output_dir.mkdir(exist_ok=True)
    
    successful_results = [r for r in all_results if r.get("success", False)]
    
    if not successful_results:
        print("❌ No successful experiments to aggregate")
        return
    
    # Collect data by layer bin (including special layers)
    layer_bin_data = defaultdict(lambda: {
        'R_values': [],
        'mean_angles': [],
        'median_angles': [],
        'min_angles': [],
        'max_angles': [],
        'percentile_25_angles': [],
        'percentile_75_angles': [],
        'std_angles': [],
        'model_dataset': [],
        'magnitude_by_angle_bins': [],
        'within_shell_uniformity': []
    })
    
    # Track special vs regular bins
    special_bin_names = ['embedding', 'first_block', 'last_layer']
    regular_bin_names = ['earlier_early', 'early_mid', 'mid', 'later_mid', 'later']
    
    for result in successful_results:
        n_layers = result['n_layers']
        model_name = result['model'].split('/')[-1]
        dataset_name = result['dataset'].split('/')[-1]
        
        for layer_idx in range(n_layers):
            layer_bin = get_layer_bin(layer_idx, n_layers)
            stats = result['layer_statistics'][layer_idx]
            
            layer_bin_data[layer_bin]['R_values'].append(stats['mean_resultant_length'])
            layer_bin_data[layer_bin]['mean_angles'].append(stats['mean_deg'])
            layer_bin_data[layer_bin]['median_angles'].append(stats['median_deg'])
            layer_bin_data[layer_bin]['min_angles'].append(stats['min_deg'])
            layer_bin_data[layer_bin]['max_angles'].append(stats['max_deg'])
            layer_bin_data[layer_bin]['percentile_25_angles'].append(stats['percentile_25_deg'])
            layer_bin_data[layer_bin]['percentile_75_angles'].append(stats['percentile_75_deg'])
            layer_bin_data[layer_bin]['std_angles'].append(stats['std_deg'])
            layer_bin_data[layer_bin]['model_dataset'].append(f"{model_name}_{dataset_name}_L{layer_idx}")
            
            # Collect magnitude by angle bins if available
            if stats.get('magnitude_by_angle_bins') is not None:
                layer_bin_data[layer_bin]['magnitude_by_angle_bins'].append(stats['magnitude_by_angle_bins'])
            
            # Collect within-shell uniformity if available
            if stats.get('within_shell_uniformity') is not None:
                layer_bin_data[layer_bin]['within_shell_uniformity'].append(stats['within_shell_uniformity'])
    
    # Create aggregate visualizations
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle('Aggregate Analysis Across All Models and Layers\n(Special layers: Embedding, First Block, Last Layer | Regular layers binned by position)',
                 fontsize=14, fontweight='bold')
    
    # All bin names (special + regular)
    all_bin_names = ['embedding', 'first_block'] + regular_bin_names + ['last_layer']
    all_bin_labels = ['Embedding\n(L0)', 'First Block\n(L1)', 'Earlier Early\n(2+)', 'Early Mid', 'Mid', 
                      'Later Mid', 'Later', 'Last Layer\n(Ln)']
    # Special colors for special layers, gradient for regular layers
    all_bin_colors = ['red', 'orange', '#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', 'magenta']
    
    # Plot 1: R values by layer bin (box plot) - ALL BINS INCLUDING SPECIAL
    ax = axes[0, 0]
    r_data = [layer_bin_data[bin_name]['R_values'] for bin_name in all_bin_names if layer_bin_data[bin_name]['R_values']]
    r_labels = [all_bin_labels[i] for i, bn in enumerate(all_bin_names) if layer_bin_data[bn]['R_values']]
    r_colors = [all_bin_colors[i] for i, bn in enumerate(all_bin_names) if layer_bin_data[bn]['R_values']]
    bp = ax.boxplot(r_data, tick_labels=r_labels, patch_artist=True, showfliers=True)
    for patch, color in zip(bp['boxes'], r_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        # Make special layers stand out with thicker borders
        if color in ['red', 'orange', 'magenta']:
            patch.set_linewidth(2.5)
            patch.set_edgecolor('black')
    ax.axhline(y=0.8, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Highly conc.')
    ax.axhline(y=0.5, color='gray', linestyle='--', linewidth=1, alpha=0.5, label='Moderate')
    # Add vertical separators between special and regular layers
    ax.axvline(x=1.5, color='black', linestyle='-', linewidth=2, alpha=0.3)
    ax.axvline(x=6.5, color='black', linestyle='-', linewidth=2, alpha=0.3)
    ax.set_ylabel('Mean Resultant Length (R)', fontsize=10, fontweight='bold')
    ax.set_title('Concentration (R) Distribution by Layer\n(Thick borders = Special Layers)', fontsize=11)
    ax.legend(fontsize=8)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 2: Mean angular distance by layer bin (box plot)
    ax = axes[0, 1]
    angle_data = [layer_bin_data[bin_name]['mean_angles'] for bin_name in all_bin_names if layer_bin_data[bin_name]['mean_angles']]
    angle_labels = [all_bin_labels[i] for i, bn in enumerate(all_bin_names) if layer_bin_data[bn]['mean_angles']]
    angle_colors = [all_bin_colors[i] for i, bn in enumerate(all_bin_names) if layer_bin_data[bn]['mean_angles']]
    bp = ax.boxplot(angle_data, tick_labels=angle_labels, patch_artist=True, showfliers=True)
    for patch, color in zip(bp['boxes'], angle_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
        if color in ['red', 'orange', 'magenta']:
            patch.set_linewidth(2.5)
            patch.set_edgecolor('black')
    ax.axvline(x=1.5, color='black', linestyle='-', linewidth=2, alpha=0.3)
    ax.axvline(x=6.5, color='black', linestyle='-', linewidth=2, alpha=0.3)
    ax.set_ylabel('Mean Angular Distance (degrees)', fontsize=10, fontweight='bold')
    ax.set_title('Mean Angular Distance by Layer', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Plot 3: Average R by layer bin (bar chart)
    ax = axes[0, 2]
    avg_R = [np.mean(layer_bin_data[bin_name]['R_values']) if layer_bin_data[bin_name]['R_values'] else 0 
             for bin_name in all_bin_names]
    std_R = [np.std(layer_bin_data[bin_name]['R_values']) if layer_bin_data[bin_name]['R_values'] else 0 
             for bin_name in all_bin_names]
    bars = ax.bar(range(len(all_bin_labels)), avg_R, color=all_bin_colors, alpha=0.7, 
                  yerr=std_R, capsize=5, edgecolor='black')
    # Make special layer bars stand out
    for i, (bar, color) in enumerate(zip(bars, all_bin_colors)):
        if color in ['red', 'orange', 'magenta']:
            bar.set_linewidth(2.5)
            bar.set_alpha(0.9)
    ax.axvline(x=1.5, color='black', linestyle='-', linewidth=2, alpha=0.3)
    ax.axvline(x=6.5, color='black', linestyle='-', linewidth=2, alpha=0.3)
    ax.set_xticks(range(len(all_bin_labels)))
    ax.set_xticklabels(all_bin_labels, fontsize=7)
    ax.set_ylabel('Average R', fontsize=10, fontweight='bold')
    ax.set_title('Average Concentration by Layer', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 4: Sample count per bin
    ax = axes[1, 0]
    counts = [len(layer_bin_data[bin_name]['R_values']) for bin_name in all_bin_names]
    bars = ax.bar(range(len(all_bin_labels)), counts, color=all_bin_colors, alpha=0.7, edgecolor='black')
    for i, (bar, color) in enumerate(zip(bars, all_bin_colors)):
        if color in ['red', 'orange', 'magenta']:
            bar.set_linewidth(2.5)
            bar.set_alpha(0.9)
    ax.axvline(x=1.5, color='black', linestyle='-', linewidth=2, alpha=0.3)
    ax.axvline(x=6.5, color='black', linestyle='-', linewidth=2, alpha=0.3)
    ax.set_xticks(range(len(all_bin_labels)))
    ax.set_xticklabels(all_bin_labels, fontsize=7)
    ax.set_ylabel('Number of Layer Samples', fontsize=10, fontweight='bold')
    ax.set_title('Sample Count per Layer Bin', fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    for i, (bar, count) in enumerate(zip(bars, counts)):
        if count > 0:
            ax.text(bar.get_x() + bar.get_width()/2, count + max(counts)*0.02, str(count), 
                   ha='center', va='bottom', fontsize=8, fontweight='bold')
    
    # Plot 5: R trend across layer bins
    ax = axes[1, 1]
    x_pos = list(range(len(all_bin_labels)))
    ax.plot(x_pos, avg_R, 'o-', color='steelblue', linewidth=3, markersize=10)
    # Highlight special layers with different markers
    for i, (x, r, color) in enumerate(zip(x_pos, avg_R, all_bin_colors)):
        if color in ['red', 'orange', 'magenta']:
            ax.plot(x, r, 'D', color=color, markersize=15, markeredgecolor='black', markeredgewidth=2, zorder=10)
    ax.fill_between(x_pos, np.array(avg_R) - np.array(std_R), 
                     np.array(avg_R) + np.array(std_R), alpha=0.3, color='steelblue')
    ax.axvline(x=1.5, color='black', linestyle='-', linewidth=2, alpha=0.3)
    ax.axvline(x=6.5, color='black', linestyle='-', linewidth=2, alpha=0.3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_bin_labels, fontsize=7)
    ax.set_ylabel('Average R', fontsize=10, fontweight='bold')
    ax.set_title('Concentration Trend Across Layers\n(Diamonds = Special Layers)', fontsize=11)
    ax.grid(alpha=0.3)
    ax.set_ylim([0, 1])
    
    # Plot 6: Summary statistics
    ax = axes[1, 2]
    ax.axis('off')
    
    summary_text = "AGGREGATE SUMMARY\n\n"
    summary_text += f"Total experiments: {len(successful_results)}\n"
    summary_text += f"Total layer samples: {sum(counts)}\n\n"
    
    summary_text += "SPECIAL LAYERS:\n"
    for bin_name, bin_label in zip(special_bin_names, ['Embedding', 'First Block', 'Last Layer']):
        if layer_bin_data[bin_name]['R_values']:
            avg_r = np.mean(layer_bin_data[bin_name]['R_values'])
            n_samples = len(layer_bin_data[bin_name]['R_values'])
            summary_text += f"{bin_label}:\n"
            summary_text += f"  n={n_samples}, R={avg_r:.3f}\n"
    
    summary_text += "\nREGULAR LAYER BINS:\n"
    for bin_name, bin_label in zip(regular_bin_names, ['Earlier Early', 'Early Mid', 'Mid', 'Later Mid', 'Later']):
        if layer_bin_data[bin_name]['R_values']:
            avg_r = np.mean(layer_bin_data[bin_name]['R_values'])
            n_samples = len(layer_bin_data[bin_name]['R_values'])
            summary_text += f"{bin_label}:\n"
            summary_text += f"  n={n_samples}, R={avg_r:.3f}\n"
    
    ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
            fontsize=8, verticalalignment='top', family='monospace',
            bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))
    
    plt.tight_layout()
    plt.savefig(output_dir / "aggregate_layer_binning_analysis.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Save aggregate statistics (special + regular bins)
    aggregate_stats = {
        'special_layers': {},
        'regular_bins': {}
    }
    
    for bin_name in special_bin_names:
        if layer_bin_data[bin_name]['R_values']:
            aggregate_stats['special_layers'][bin_name] = {
                'n_samples': len(layer_bin_data[bin_name]['R_values']),
                'R_mean': float(np.mean(layer_bin_data[bin_name]['R_values'])),
                'R_std': float(np.std(layer_bin_data[bin_name]['R_values'])),
                'R_median': float(np.median(layer_bin_data[bin_name]['R_values'])),
                'angle_mean_mean': float(np.mean(layer_bin_data[bin_name]['mean_angles'])),
                'angle_mean_std': float(np.std(layer_bin_data[bin_name]['mean_angles'])),
                'angle_median_mean': float(np.mean(layer_bin_data[bin_name]['median_angles'])),
                'angle_min_mean': float(np.mean(layer_bin_data[bin_name]['min_angles'])),
                'angle_max_mean': float(np.mean(layer_bin_data[bin_name]['max_angles'])),
            }
    
    for bin_name in regular_bin_names:
        if layer_bin_data[bin_name]['R_values']:
            aggregate_stats['regular_bins'][bin_name] = {
                'n_samples': len(layer_bin_data[bin_name]['R_values']),
                'R_mean': float(np.mean(layer_bin_data[bin_name]['R_values'])),
                'R_std': float(np.std(layer_bin_data[bin_name]['R_values'])),
                'R_median': float(np.median(layer_bin_data[bin_name]['R_values'])),
                'angle_mean_mean': float(np.mean(layer_bin_data[bin_name]['mean_angles'])),
                'angle_mean_std': float(np.std(layer_bin_data[bin_name]['mean_angles'])),
                'angle_median_mean': float(np.mean(layer_bin_data[bin_name]['median_angles'])),
                'angle_min_mean': float(np.mean(layer_bin_data[bin_name]['min_angles'])),
                'angle_max_mean': float(np.mean(layer_bin_data[bin_name]['max_angles'])),
            }
    
    with open(output_dir / "aggregate_statistics.json", "w") as f:
        json.dump(aggregate_stats, f, indent=2)
    
    # Create additional comprehensive aggregate plots
    create_comprehensive_aggregate_plots(layer_bin_data, output_dir)
    
    print(f"✅ Aggregate analysis saved to: {output_dir}")
    print(f"   - aggregate_layer_binning_analysis.png")
    print(f"   - aggregate_statistics.json")
    print(f"   - aggregate_angular_distance_trends.png")
    print(f"   - aggregate_magnitude_by_angular_bins.png")
    print(f"   - aggregate_within_shell_uniformity.png")


def create_comprehensive_aggregate_plots(layer_bin_data, output_dir):
    """
    Create comprehensive aggregate plots:
    1. Angular distance trends (mean/median/min/max) across layer bins
    2. Magnitude by angular bins aggregation
    3. Within-shell uniformity aggregation
    
    Includes special layers (embedding, first_block, last_layer) highlighted.
    """
    # Special layers + regular bins
    special_bin_names = ['embedding', 'first_block', 'last_layer']
    regular_bin_names = ['earlier_early', 'early_mid', 'mid', 'later_mid', 'later']
    all_bin_names = ['embedding', 'first_block'] + regular_bin_names + ['last_layer']
    all_bin_labels = ['Embedding\n(L0)', 'First Block\n(L1)', 'Earlier Early', 'Early Mid', 'Mid', 
                      'Later Mid', 'Later', 'Last Layer\n(Ln)']
    all_bin_colors = ['red', 'orange', '#3498db', '#2ecc71', '#f39c12', '#e74c3c', '#9b59b6', 'magenta']
    
    # ============================================================
    # PLOT 1: Angular Distance Statistics Trends
    # ============================================================
    print("\n📈 Creating angular distance trends plot...")
    fig, axes = plt.subplots(2, 2, figsize=(18, 12))
    fig.suptitle('Aggregate Angular Distance Trends Across Layer Bins\n(All Models Combined | Special Layers Highlighted)',
                 fontsize=14, fontweight='bold')
    
    x_pos = list(range(len(all_bin_names)))
    
    # Mean angular distance
    ax = axes[0, 0]
    mean_vals = [np.mean(layer_bin_data[bn]['mean_angles']) if layer_bin_data[bn]['mean_angles'] else 0 for bn in all_bin_names]
    std_vals = [np.std(layer_bin_data[bn]['mean_angles']) if layer_bin_data[bn]['mean_angles'] else 0 for bn in all_bin_names]
    ax.plot(x_pos, mean_vals, 'o-', color='steelblue', linewidth=3, markersize=10, label='Mean')
    # Highlight special layers
    for i, (x, m, color) in enumerate(zip(x_pos, mean_vals, all_bin_colors)):
        if color in ['red', 'orange', 'magenta']:
            ax.plot(x, m, 'D', color=color, markersize=15, markeredgecolor='black', markeredgewidth=2, zorder=10)
    ax.fill_between(x_pos, np.array(mean_vals) - np.array(std_vals), 
                     np.array(mean_vals) + np.array(std_vals), alpha=0.3, color='steelblue')
    ax.axvline(x=1.5, color='black', linestyle='-', linewidth=1.5, alpha=0.3)
    ax.axvline(x=6.5, color='black', linestyle='-', linewidth=1.5, alpha=0.3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_bin_labels, fontsize=8)
    ax.set_ylabel('Mean Angular Distance (degrees)', fontsize=11, fontweight='bold')
    ax.set_title('Mean Angular Distance Trend\n(Diamonds = Special Layers)', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Median angular distance
    ax = axes[0, 1]
    median_vals = [np.mean(layer_bin_data[bn]['median_angles']) if layer_bin_data[bn]['median_angles'] else 0 for bn in all_bin_names]
    std_vals = [np.std(layer_bin_data[bn]['median_angles']) if layer_bin_data[bn]['median_angles'] else 0 for bn in all_bin_names]
    ax.plot(x_pos, median_vals, 's-', color='orange', linewidth=3, markersize=10, label='Median')
    for i, (x, m, color) in enumerate(zip(x_pos, median_vals, all_bin_colors)):
        if color in ['red', 'orange', 'magenta']:
            ax.plot(x, m, 'D', color=color, markersize=15, markeredgecolor='black', markeredgewidth=2, zorder=10)
    ax.fill_between(x_pos, np.array(median_vals) - np.array(std_vals), 
                     np.array(median_vals) + np.array(std_vals), alpha=0.3, color='orange')
    ax.axvline(x=1.5, color='black', linestyle='-', linewidth=1.5, alpha=0.3)
    ax.axvline(x=6.5, color='black', linestyle='-', linewidth=1.5, alpha=0.3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_bin_labels, fontsize=8)
    ax.set_ylabel('Median Angular Distance (degrees)', fontsize=11, fontweight='bold')
    ax.set_title('Median Angular Distance Trend', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Min/Max angular distance
    ax = axes[1, 0]
    min_vals = [np.mean(layer_bin_data[bn]['min_angles']) if layer_bin_data[bn]['min_angles'] else 0 for bn in all_bin_names]
    max_vals = [np.mean(layer_bin_data[bn]['max_angles']) if layer_bin_data[bn]['max_angles'] else 0 for bn in all_bin_names]
    ax.plot(x_pos, min_vals, 'v-', color='blue', linewidth=2.5, markersize=8, label='Min')
    ax.plot(x_pos, max_vals, '^-', color='red', linewidth=2.5, markersize=8, label='Max')
    ax.fill_between(x_pos, min_vals, max_vals, alpha=0.2, color='gray')
    ax.axvline(x=1.5, color='black', linestyle='-', linewidth=1.5, alpha=0.3)
    ax.axvline(x=6.5, color='black', linestyle='-', linewidth=1.5, alpha=0.3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_bin_labels, fontsize=8)
    ax.set_ylabel('Angular Distance (degrees)', fontsize=11, fontweight='bold')
    ax.set_title('Min/Max Angular Distance Trend', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend()
    
    # Combined view
    ax = axes[1, 1]
    ax.plot(x_pos, mean_vals, 'o-', color='steelblue', linewidth=2.5, markersize=8, label='Mean')
    ax.plot(x_pos, median_vals, 's-', color='orange', linewidth=2.5, markersize=8, label='Median')
    ax.plot(x_pos, min_vals, 'v-', color='blue', linewidth=2, markersize=6, label='Min', alpha=0.7)
    ax.plot(x_pos, max_vals, '^-', color='red', linewidth=2, markersize=6, label='Max', alpha=0.7)
    ax.axvline(x=1.5, color='black', linestyle='-', linewidth=1.5, alpha=0.3)
    ax.axvline(x=6.5, color='black', linestyle='-', linewidth=1.5, alpha=0.3)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(all_bin_labels, fontsize=8)
    ax.set_ylabel('Angular Distance (degrees)', fontsize=11, fontweight='bold')
    ax.set_title('All Statistics Combined', fontsize=12)
    ax.grid(alpha=0.3)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(output_dir / "aggregate_angular_distance_trends.png", dpi=150, bbox_inches='tight')
    plt.close()
    print("   ✓ Saved aggregate_angular_distance_trends.png")
    
    # ============================================================
    # PLOT 2: Magnitude by Angular Bins Aggregation
    # ============================================================
    print("\n📈 Creating magnitude by angular bins aggregation...")
    
    # Check if we have magnitude data
    has_magnitude_data = any(layer_bin_data[bn]['magnitude_by_angle_bins'] for bn in all_bin_names)
    
    if has_magnitude_data:
        # Show first 5 bins (3 special + 2 regular) in 2x3 grid
        bins_to_show = all_bin_names[:5]  # embedding, first_block, earlier_early, early_mid, mid
        labels_to_show = all_bin_labels[:5]
        colors_to_show = all_bin_colors[:5]
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Aggregate Magnitude by Angular Bins\n(All Models Combined | First 5 Bins)',
                     fontsize=14, fontweight='bold')
        
        # For each layer bin, aggregate magnitude stats across angular bins
        for bin_idx, (bin_name, bin_label, bin_color) in enumerate(zip(bins_to_show, labels_to_show, colors_to_show)):
            ax_idx = bin_idx if bin_idx < 5 else 5
            ax = axes.flatten()[ax_idx]
            
            if not layer_bin_data[bin_name]['magnitude_by_angle_bins']:
                ax.text(0.5, 0.5, f'No data for\n{bin_label}', ha='center', va='center', 
                       transform=ax.transAxes, fontsize=12)
                title = f'{bin_label}'
                if bin_color in ['red', 'orange', 'magenta']:
                    title += ' (SPECIAL)'
                ax.set_title(title, fontsize=11, fontweight='bold' if bin_color in ['red', 'orange', 'magenta'] else 'normal')
                continue
            
            # Aggregate magnitude by angular bins
            mag_data = layer_bin_data[bin_name]['magnitude_by_angle_bins']
            n_angular_bins = len(mag_data[0])
            
            # Average across all samples in this layer bin
            avg_mags = []
            std_mags = []
            pct_vecs = []
            angle_labels = []
            
            for ang_bin_idx in range(n_angular_bins):
                mags_for_bin = [sample[ang_bin_idx]['avg_magnitude'] for sample in mag_data]
                pcts_for_bin = [sample[ang_bin_idx]['pct_vectors'] for sample in mag_data]
                
                avg_mags.append(np.mean(mags_for_bin))
                std_mags.append(np.std(mags_for_bin))
                pct_vecs.append(np.mean(pcts_for_bin))
                angle_labels.append(mag_data[0][ang_bin_idx]['angle_range'])
            
            # Plot magnitude
            x_ang = list(range(len(angle_labels)))
            ax.bar(x_ang, avg_mags, color=bin_color, alpha=0.7 if bin_color not in ['red', 'orange', 'magenta'] else 0.9, 
                  edgecolor='black', linewidth=2 if bin_color in ['red', 'orange', 'magenta'] else 1)
            ax.errorbar(x_ang, avg_mags, yerr=std_mags, fmt='none', ecolor='black', capsize=3, alpha=0.6)
            ax.set_xticks(x_ang)
            ax.set_xticklabels(angle_labels, rotation=45, ha='right', fontsize=7)
            ax.set_ylabel('Avg Magnitude', fontsize=9, fontweight='bold')
            title = f'{bin_label}\n(n={len(mag_data)} samples)'
            if bin_color in ['red', 'orange', 'magenta']:
                title += ' [SPECIAL]'
            ax.set_title(title, fontsize=10, fontweight='bold' if bin_color in ['red', 'orange', 'magenta'] else 'normal')
            ax.grid(axis='y', alpha=0.3)
        
        # Last subplot: comparison across layer bins (ALL bins)
        ax = axes.flatten()[5]
        for bin_idx, (bin_name, bin_label, bin_color) in enumerate(zip(all_bin_names, all_bin_labels, all_bin_colors)):
            if not layer_bin_data[bin_name]['magnitude_by_angle_bins']:
                continue
            
            mag_data = layer_bin_data[bin_name]['magnitude_by_angle_bins']
            n_angular_bins = len(mag_data[0])
            
            angle_centers = [mag_data[0][i]['angle_center_deg'] for i in range(n_angular_bins)]
            avg_mags = [np.mean([sample[i]['avg_magnitude'] for sample in mag_data]) for i in range(n_angular_bins)]
            
            # Use thicker lines and larger markers for special layers
            linewidth = 3 if bin_color in ['red', 'orange', 'magenta'] else 1.5
            markersize = 8 if bin_color in ['red', 'orange', 'magenta'] else 5
            alpha = 1.0 if bin_color in ['red', 'orange', 'magenta'] else 0.6
            
            ax.plot(angle_centers, avg_mags, 'o-', color=bin_color, linewidth=linewidth, 
                   markersize=markersize, label=bin_label.replace('\n', ' '), alpha=alpha,
                   markeredgecolor='black' if bin_color in ['red', 'orange', 'magenta'] else None,
                   markeredgewidth=1.5 if bin_color in ['red', 'orange', 'magenta'] else 0)
        
        ax.set_xlabel('Angular Distance (degrees)', fontsize=10, fontweight='bold')
        ax.set_ylabel('Avg Magnitude', fontsize=10, fontweight='bold')
        ax.set_title('Comparison Across All Layer Bins\n(Bold = Special Layers)', fontsize=11)
        ax.legend(fontsize=6, ncol=2)
        ax.grid(alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(output_dir / "aggregate_magnitude_by_angular_bins.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved aggregate_magnitude_by_angular_bins.png")
    else:
        print("   ⚠️  No magnitude by angular bins data available")
    
    # ============================================================
    # PLOT 3: Within-Shell Uniformity Aggregation
    # ============================================================
    print("\n📈 Creating within-shell uniformity aggregation...")
    
    # Check if we have uniformity data
    has_uniformity_data = any(layer_bin_data[bn]['within_shell_uniformity'] for bn in all_bin_names)
    
    if has_uniformity_data:
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle('Aggregate Within-Shell Uniformity Across Layer Bins\n(All Models Combined | Special Layers Highlighted)',
                     fontsize=14, fontweight='bold')
        
        # Extract uniformity metrics across layer bins
        uniformity_metrics = defaultdict(lambda: defaultdict(list))
        
        for bin_name in all_bin_names:
            if not layer_bin_data[bin_name]['within_shell_uniformity']:
                continue
            
            for unif_data in layer_bin_data[bin_name]['within_shell_uniformity']:
                # Extract mean pairwise distance per magnitude bin
                if 'bin_uniformity' in unif_data:
                    for dist_bin in unif_data['bin_uniformity']:
                        bin_center = dist_bin['distance_actual_mean_deg']
                        mean_pw_dist = dist_bin['mean_pairwise_angle_deg']
                        uniformity_metrics[bin_name]['bin_centers'].append(bin_center)
                        uniformity_metrics[bin_name]['mean_pw_dists'].append(mean_pw_dist)
        
        # Plot 1: Mean pairwise distance trend across layer bins
        ax = axes[0, 0]
        for bin_idx, (bin_name, bin_color) in enumerate(zip(all_bin_names, all_bin_colors)):
            if bin_name not in uniformity_metrics or not uniformity_metrics[bin_name]['mean_pw_dists']:
                continue
            
            # Average over all samples
            pw_dists = uniformity_metrics[bin_name]['mean_pw_dists']
            avg_pw = np.mean(pw_dists)
            std_pw = np.std(pw_dists)
            
            alpha = 0.9 if bin_color in ['red', 'orange', 'magenta'] else 0.7
            linewidth = 2.5 if bin_color in ['red', 'orange', 'magenta'] else 1
            ax.bar(bin_idx, avg_pw, color=bin_color, alpha=alpha, edgecolor='black', linewidth=linewidth)
            ax.errorbar(bin_idx, avg_pw, yerr=std_pw, fmt='none', ecolor='black', capsize=5, alpha=0.8)
        
        ax.axvline(x=1.5, color='black', linestyle='-', linewidth=1.5, alpha=0.3)
        ax.axvline(x=6.5, color='black', linestyle='-', linewidth=1.5, alpha=0.3)
        ax.set_xticks(range(len(all_bin_names)))
        ax.set_xticklabels(all_bin_labels, fontsize=8)
        ax.set_ylabel('Mean Pairwise Angular Distance', fontsize=11, fontweight='bold')
        ax.set_title('Within-Shell Uniformity by Layer Bin\n(Thick borders = Special Layers)', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 2: Trend line
        ax = axes[0, 1]
        avg_pws = []
        std_pws = []
        for bin_name in all_bin_names:
            if bin_name in uniformity_metrics and uniformity_metrics[bin_name]['mean_pw_dists']:
                avg_pws.append(np.mean(uniformity_metrics[bin_name]['mean_pw_dists']))
                std_pws.append(np.std(uniformity_metrics[bin_name]['mean_pw_dists']))
            else:
                avg_pws.append(np.nan)
                std_pws.append(0)
        
        # Filter out NaN values
        valid_indices = [i for i, v in enumerate(avg_pws) if not np.isnan(v)]
        if valid_indices:
            valid_x = [i for i in valid_indices]
            valid_avg = [avg_pws[i] for i in valid_indices]
            valid_std = [std_pws[i] for i in valid_indices]
            valid_colors = [all_bin_colors[i] for i in valid_indices]
            
            ax.plot(valid_x, valid_avg, 'o-', color='purple', linewidth=3, markersize=10)
            # Highlight special layers with colored diamonds
            for x, avg, color in zip(valid_x, valid_avg, valid_colors):
                if color in ['red', 'orange', 'magenta']:
                    ax.plot(x, avg, 'D', color=color, markersize=15, markeredgecolor='black', markeredgewidth=2, zorder=10)
            ax.fill_between(valid_x, 
                           np.array(valid_avg) - np.array(valid_std),
                           np.array(valid_avg) + np.array(valid_std),
                           alpha=0.3, color='purple')
        
        ax.axvline(x=1.5, color='black', linestyle='-', linewidth=1.5, alpha=0.3)
        ax.axvline(x=6.5, color='black', linestyle='-', linewidth=1.5, alpha=0.3)
        ax.set_xticks(range(len(all_bin_names)))
        ax.set_xticklabels(all_bin_labels, fontsize=8)
        ax.set_ylabel('Mean Pairwise Angular Distance', fontsize=11, fontweight='bold')
        ax.set_title('Uniformity Trend Across Layers\n(Diamonds = Special Layers)', fontsize=12)
        ax.grid(alpha=0.3)
        
        # Plot 3: Box plot comparison
        ax = axes[1, 0]
        box_data = []
        box_labels_actual = []
        box_colors_actual = []
        for bin_idx, (bin_name, bin_label, bin_color) in enumerate(zip(all_bin_names, all_bin_labels, all_bin_colors)):
            if bin_name in uniformity_metrics and uniformity_metrics[bin_name]['mean_pw_dists']:
                box_data.append(uniformity_metrics[bin_name]['mean_pw_dists'])
                box_labels_actual.append(bin_label)
                box_colors_actual.append(bin_color)
        
        if box_data:
            bp = ax.boxplot(box_data, tick_labels=box_labels_actual, patch_artist=True, showfliers=True)
            for patch, color in zip(bp['boxes'], box_colors_actual):
                patch.set_facecolor(color)
                alpha = 0.9 if color in ['red', 'orange', 'magenta'] else 0.7
                linewidth = 2.5 if color in ['red', 'orange', 'magenta'] else 1
                patch.set_alpha(alpha)
                patch.set_linewidth(linewidth)
                if color in ['red', 'orange', 'magenta']:
                    patch.set_edgecolor('black')
        
        ax.set_ylabel('Mean Pairwise Angular Distance', fontsize=11, fontweight='bold')
        ax.set_title('Distribution Comparison\n(Thick borders = Special Layers)', fontsize=12)
        ax.grid(axis='y', alpha=0.3)
        
        # Plot 4: Summary statistics
        ax = axes[1, 1]
        ax.axis('off')
        
        summary_text = "UNIFORMITY SUMMARY\n\n"
        summary_text += "SPECIAL LAYERS:\n"
        for bin_name, bin_label in zip(['embedding', 'first_block', 'last_layer'], ['Embedding (L0)', 'First Block (L1)', 'Last Layer']):
            if bin_name in uniformity_metrics and uniformity_metrics[bin_name]['mean_pw_dists']:
                pw_dists = uniformity_metrics[bin_name]['mean_pw_dists']
                summary_text += f"{bin_label}:\n"
                summary_text += f"  n={len(pw_dists)}, PW={np.mean(pw_dists):.2f}°\n"
        
        summary_text += "\nREGULAR BINS:\n"
        for bin_name, bin_label in zip(regular_bin_names, ['Earlier Early', 'Early Mid', 'Mid', 'Later Mid', 'Later']):
            if bin_name in uniformity_metrics and uniformity_metrics[bin_name]['mean_pw_dists']:
                pw_dists = uniformity_metrics[bin_name]['mean_pw_dists']
                summary_text += f"{bin_label}:\n"
                summary_text += f"  n={len(pw_dists)}, PW={np.mean(pw_dists):.2f}°\n"
        
        ax.text(0.1, 0.95, summary_text, transform=ax.transAxes,
                fontsize=8, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round', facecolor='lavender', alpha=0.3))
        
        plt.tight_layout()
        plt.savefig(output_dir / "aggregate_within_shell_uniformity.png", dpi=150, bbox_inches='tight')
        plt.close()
        print("   ✓ Saved aggregate_within_shell_uniformity.png")
    else:
        print("   ⚠️  No within-shell uniformity data available")


def run_all_experiments(base_output_dir="/data/user/sae/activation_clustering_behavior1", cuda_device=None):
    """Run all layer evolution experiments."""
    
    # Override global CUDA_DEVICE if specified
    global CUDA_DEVICE
    if cuda_device is not None:
        CUDA_DEVICE = cuda_device
    
    print("="*80)
    print("🚀 STARTING LAYER EVOLUTION ANALYSIS")
    print("="*80)
    print(f"Total experiments to run: {len(EXPERIMENTS)}")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available. This script requires a CUDA-capable GPU with sufficient VRAM.")
    print(f"Device: CUDA:{CUDA_DEVICE}")
    print(f"Output directory: {base_output_dir}")
    print("="*80)
    
    Path(base_output_dir).mkdir(exist_ok=True)
    
    all_results = []
    
    for i, exp_config in enumerate(EXPERIMENTS, 1):
        print(f"\n\n{'='*80}")
        print(f"📋 PROGRESS: Experiment {i}/{len(EXPERIMENTS)}")
        print(f"{'='*80}")
        
        result = run_single_experiment(exp_config, base_output_dir)
        all_results.append(result)
    
    # Create comparison plots
    create_comparison_plots(all_results, base_output_dir)
    
    # Create aggregate analysis with layer binning
    create_aggregate_analysis(all_results, base_output_dir)
    
    # Save summary
    summary_path = Path(base_output_dir) / "summary_report.json"
    with open(summary_path, "w") as f:
        json.dump({
            "total_experiments": len(EXPERIMENTS),
            "successful": sum(1 for r in all_results if r.get("success", False)),
            "failed": sum(1 for r in all_results if not r.get("success", False)),
            "all_results": all_results
        }, f, indent=2)
    
    print(f"\n📄 Summary saved to: {summary_path}")
    print("\n" + "="*80)
    print("🎉 ALL EXPERIMENTS COMPLETE!")
    print("="*80)
    
    return all_results


if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Layer Evolution Analysis')
    parser.add_argument('--cuda', type=int, default=1, 
                       help='CUDA device to use (default: 1)')
    args = parser.parse_args()
    
    # Update global settings based on args
    cuda_device_to_use = args.cuda
    
    # Verify CUDA is available before starting
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is not available. This script requires a CUDA-capable GPU with sufficient VRAM.")
    
    print("🌐 Layer-by-Layer Evolution of Residual Stream Distribution")
    print("Tracking how angular distribution changes across all layers")
    print()
    
    try:
        results = run_all_experiments(cuda_device=cuda_device_to_use)
        
        print("\n✨ Analysis complete! Check the results:")
        print("   📂 /data/user/sae/activation_clustering_behavior1/ - Individual experiment results")
        print("   📂 /data/user/sae/activation_clustering_behavior1/comparison_analysis/ - Cross-experiment comparisons")
        
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted by user. Partial results may be available.")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        import traceback
        traceback.print_exc()

