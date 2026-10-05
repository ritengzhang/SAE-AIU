#!/usr/bin/env python3
"""
Activation Collection Helpers

Shared utilities for collecting activations from models for SAE analysis.
Used across all SAE families (SAE Bench, JBloom OpenAI v5, CanRager, etc.).
"""

import torch
from typing import Optional
from transformer_lens import HookedTransformer

# Add parent directories to path
import sys
from pathlib import Path
parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

sae_lens_dir = parent_dir / "SAELens-main"
sys.path.insert(0, str(sae_lens_dir))

from sae_lens.training.activations_store import ActivationsStore
from sae_lens.config import LanguageModelSAERunnerConfig
from sae_lens.saes.standard_sae import StandardTrainingSAEConfig


def setup_activations_store(
    model: HookedTransformer,
    sae,
    metadata: dict,
    device: str = None,
    dataset_path: Optional[str] = None,
    width_name: Optional[str] = None  # For adaptive buffer sizing
) -> ActivationsStore:
    """
    Setup activations store for collecting activations.
    
    Args:
        model: HookedTransformer model
        sae: SAE object (for getting d_in, d_sae)
        metadata: Metadata dictionary with 'hook_name', 'model_name', 'dataset_path'
        device: Device to use (default: auto-detect cuda:0, requires GPU)
            Can be overridden to use cuda:1 or other devices.
        dataset_path: Override dataset path from metadata
        
    Returns:
        ActivationsStore ready for collecting activations
    """
    # Auto-detect device if not specified - require GPU
    if device is None:
        import torch
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
    
    # Get dataset path
    if dataset_path is None:
        dataset_path = metadata.get('dataset_path', 'NeelNanda/pile-10k')
    print(f"  Using dataset: {dataset_path}")
    
    # Get hook_name from metadata (this is the SPECIFIC layer the SAE was trained on!)
    # For Pythia-160M SAEs: blocks.8.hook_resid_post (layer 8)
    # For Gemma-2-2B SAEs: blocks.12.hook_resid_post (layer 12)
    hook_name = metadata.get('hook_name', 'blocks.8.hook_resid_post')
    print(f"  ✓ Using hook: {hook_name} (SAE was trained on this specific layer)")
    
    # Reduce context_size for Gemma models to save memory (512 → 128)
    model_name = metadata.get('model_name', 'pythia-160m-deduped')
    if 'gemma' in model_name.lower():
        context_size = 128  # Reduced from 512 for memory efficiency
        print(f"  📉 Using reduced context_size={context_size} for Gemma model")
        # For Gemma with small context_size, ensure buffer is large enough for collection
        # We collect 10 batches, so buffer needs to be >= 10
        if width_name == '16k':
            n_batches_in_buffer = 12  # >= 10 batches we collect
            store_batch_size_prompts = 2  # Reduced for memory
            print(f"  📉 Using reduced buffer for Gemma 16k: {n_batches_in_buffer} batches, {store_batch_size_prompts} prompts/batch")
        elif width_name == '65k':
            n_batches_in_buffer = 12  # >= 10 batches we collect
            store_batch_size_prompts = 2  # Reduced for memory
            print(f"  📉 Using reduced buffer for Gemma 65k: {n_batches_in_buffer} batches, {store_batch_size_prompts} prompts/batch")
        else:
            # For 4k Gemma
            n_batches_in_buffer = 12  # >= 10 batches we collect
            store_batch_size_prompts = 2  # Reduced for memory
            print(f"  📉 Using reduced buffer for Gemma 4k: {n_batches_in_buffer} batches, {store_batch_size_prompts} prompts/batch")
    else:
        context_size = 512  # Default for smaller models
        # Adaptive buffer sizing based on SAE width to reduce memory usage
        # Default: n_batches_in_buffer=16, store_batch_size_prompts=8 → ~600 MB buffer
        if width_name == '16k':
            # Reduce buffer for 16k: 8 batches, 4 prompts → ~150 MB buffer
            n_batches_in_buffer = 8
            store_batch_size_prompts = 4
            print(f"  📉 Using reduced buffer for 16k: {n_batches_in_buffer} batches, {store_batch_size_prompts} prompts/batch")
        elif width_name == '65k':
            # Reduce buffer for 65k: 8 batches, 4 prompts → ~150 MB buffer (needs to be >= n_batches we collect)
            n_batches_in_buffer = 8
            store_batch_size_prompts = 4
            print(f"  📉 Using reduced buffer for 65k: {n_batches_in_buffer} batches, {store_batch_size_prompts} prompts/batch")
        else:
            # Default for 4k and others
            n_batches_in_buffer = 16
            store_batch_size_prompts = 8
    
    # Calculate train_batch_size_tokens to be <= buffer size
    # Buffer size = n_batches_in_buffer * store_batch_size_prompts * context_size
    buffer_size_tokens = n_batches_in_buffer * store_batch_size_prompts * context_size
    # Use a safe batch size that's <= buffer size
    train_batch_size_tokens = min(256, buffer_size_tokens // 2)  # Use half of buffer size for safety
    
    cfg = LanguageModelSAERunnerConfig(
        sae=dummy_sae_cfg,
        model_name=model_name,
        hook_name=hook_name,
        dataset_path=dataset_path,
        is_dataset_tokenized=True,
        streaming=True,
        train_batch_size_tokens=train_batch_size_tokens,
        context_size=context_size,
        n_batches_in_buffer=n_batches_in_buffer,
        training_tokens=4096 * 100,
        store_batch_size_prompts=store_batch_size_prompts,
        device=device,
        seed=42,
        dtype="float32",
    )
    
    # Retry logic for network issues
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
                import time
                time.sleep(5)
            else:
                print(f"  ❌ Failed after {max_retries} attempts")
                raise


def load_base_model(model_name: str, device: str = None, use_auth_token: bool = True) -> HookedTransformer:
    """
    Load base model for activation collection.
    
    Args:
        model_name: Model name (e.g., 'pythia-160m-deduped', 'google/gemma-2-2b')
        device: Device to load model on (default: auto-detect cuda:0, requires GPU)
            Can be overridden to use cuda:1 or other devices.
            Note: LLM models use transformers library caching (automatic, separate from SAE cache).
        use_auth_token: Whether to use HuggingFace authentication token for gated repos.
            If True, will use token from environment variable HF_TOKEN or HuggingFace CLI login.
            Set to False to disable authentication (for public repos only).
        
    Returns:
        HookedTransformer model
    """
    # Auto-detect device if not specified - require GPU
    if device is None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU is not available. GPU is required for model loading.")
        device = "cuda:0"
    
    # Handle HuggingFace authentication for gated repos
    # HookedTransformer.from_pretrained automatically uses HuggingFace CLI login token if available
    # OR reads from HF_TOKEN environment variable automatically
    # We should NOT pass token explicitly to avoid "multiple values" error
    from_pretrained_kwargs = {}
    if use_auth_token:
        import os
        
        # Check if token is available (either in env or via CLI login)
        # But DON'T pass it explicitly - let transformers/huggingface_hub handle it automatically
        hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
        
        if hf_token:
            print(f"Loading base model: {model_name}...")
            print(f"  Using HuggingFace token from environment variable (auto-detected)")
        else:
            # Don't pass token parameter - let HuggingFace automatically use CLI login token if available
            print(f"Loading base model: {model_name}...")
            print(f"  Using HuggingFace CLI authentication (if logged in)")
    else:
        print(f"Loading base model: {model_name}...")
    
    # Add retry logic for model loading (handles incomplete downloads, network issues, etc.)
    import time
    max_retries = 3
    retry_delay = 5  # seconds
    
    for attempt in range(max_retries):
        try:
            model = HookedTransformer.from_pretrained(model_name, device=device, **from_pretrained_kwargs)
            print(f"✓ Loaded model: {model_name}")
            return model
        except Exception as e:
            error_msg = str(e)
            # Check if it's a download/corruption issue
            if "incomplete" in error_msg.lower() or "deserializing" in error_msg.lower() or "metadata" in error_msg.lower():
                if attempt < max_retries - 1:
                    print(f"  ⚠️  Model download incomplete or corrupted (attempt {attempt + 1}/{max_retries})")
                    print(f"     Clearing cache and retrying in {retry_delay} seconds...")
                    # Clear the model cache to force fresh download
                    import shutil
                    cache_path = Path.home() / ".cache" / "huggingface" / "hub" / f"models--{model_name.replace('/', '--')}"
                    if cache_path.exists():
                        shutil.rmtree(cache_path)
                        print(f"     ✓ Cleared cache: {cache_path}")
                    time.sleep(retry_delay)
                    retry_delay *= 2  # Exponential backoff
                else:
                    print(f"  ❌ Failed to load model after {max_retries} attempts")
                    raise
            else:
                # For other errors, don't retry
                raise
