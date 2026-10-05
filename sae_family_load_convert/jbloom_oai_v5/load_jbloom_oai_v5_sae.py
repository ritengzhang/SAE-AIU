#!/usr/bin/env python3
"""
jbloom OAI-v5 SAE Loader

Functions for loading jbloom OAI-v5 SAEs from HuggingFace.
These are SAELens Standard Format SAEs trained on GPT-2 Small.

Repositories:
- GPT2-Small-OAI-v5-32k-resid-post-SAEs
- GPT2-Small-OAI-v5-128k-resid-post-SAEs
- GPT2-Small-OAI-v5-32k-resid-mid-SAEs
- GPT2-Small-OAI-v5-128k-resid-mid-SAEs
- GPT2-Small-OAI-v5-32k-mlp-out-SAEs
- GPT2-Small-OAI-v5-128k-mlp-out-SAEs
- GPT2-Small-OAI-v5-32k-attn-out-SAEs
- GPT2-Small-OAI-v5-128k-attn-out-SAEs

Each repo contains 12 SAEs (layers 0-11).
"""

import shutil
from pathlib import Path
from typing import Optional, Dict, Tuple

# Add parent directories to path for imports
import sys
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

sae_lens_dir = parent_dir / "SAELens-main"
sys.path.insert(0, str(sae_lens_dir))

from sae_lens import SAE
from sae_lens.loading.pretrained_sae_loaders import sae_lens_huggingface_loader


# Map hook types to hook names
HOOK_TYPE_MAP = {
    'resid_post': 'hook_resid_post',
    'resid_mid': 'hook_resid_mid',
    'mlp_out': 'hook_mlp_out',
    'attn_out': 'hook_attn_out',
}


def get_folder_name(width: str, layer: int) -> str:
    """
    Get folder name for SAE in HuggingFace repo.
    
    Args:
        width: '32k' or '128k'
        layer: Layer number (0-11)
        
    Returns:
        Folder name (e.g., 'v5_32k_layer_0.pt' for 32k, 'v5_128k_layer_0' for 128k)
    """
    # Note: 32k repos use .pt extension, 128k repos don't
    if width == '32k':
        return f"v5_32k_layer_{layer}.pt"
    else:
        return f"v5_128k_layer_{layer}"


def parse_repo_id(repo_id: str) -> Tuple[str, str, str]:
    """
    Parse repo_id to extract width and hook type.
    
    Args:
        repo_id: e.g., 'jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs'
        
    Returns:
        Tuple of (width, hook_type, base_repo_id)
    """
    # Extract width (32k or 128k)
    if '32k' in repo_id:
        width = '32k'
    elif '128k' in repo_id:
        width = '128k'
    else:
        raise ValueError(f"Could not determine width from repo_id: {repo_id}")
    
    # Extract hook type
    if 'resid-post' in repo_id:
        hook_type = 'resid_post'
    elif 'resid-mid' in repo_id:
        hook_type = 'resid_mid'
    elif 'mlp-out' in repo_id:
        hook_type = 'mlp_out'
    elif 'attn-out' in repo_id:
        hook_type = 'attn_out'
    else:
        raise ValueError(f"Could not determine hook type from repo_id: {repo_id}")
    
    return width, hook_type, repo_id


def load_jbloom_oai_v5_sae(
    repo_id: str,
    layer: int,
    device: str = None,
    cache_dir: Optional[str] = None
) -> Tuple[SAE, Dict]:
    """
    Load jbloom OAI-v5 SAE from HuggingFace.
    
    Args:
        repo_id: HuggingFace repo ID (e.g., 'jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs')
        layer: Layer number (0-11)
        device: Device to load SAE on (default: auto-detect cuda:0, requires GPU)
        cache_dir: Optional cache directory (default: None, uses HuggingFace default cache)
            Note: SAEs are cached by HuggingFace Hub. Use cleanup_sae_cache() after analysis.
        
    Returns:
        Tuple of (SAE object, metadata dictionary)
    """
    # Auto-detect device if not specified - require GPU
    if device is None:
        import torch
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU is not available. GPU is required for SAE operations.")
        device = "cuda:0"
    
    print(f"Loading jbloom OAI-v5 SAE from HuggingFace...")
    print(f"  Repo: {repo_id}")
    print(f"  Layer: {layer}")
    
    # Parse repo_id to get width and hook type
    width, hook_type, base_repo_id = parse_repo_id(repo_id)
    
    # Get folder name
    folder_name = get_folder_name(width, layer)
    
    # Get hook name
    hook_name_base = HOOK_TYPE_MAP[hook_type]
    hook_name = f"blocks.{layer}.{hook_name_base}"
    
    print(f"  Width: {width}")
    print(f"  Hook type: {hook_type}")
    print(f"  Folder: {folder_name}")
    print(f"  Hook name: {hook_name}")
    
    # Load using SAELens loader (already compatible!)
    cfg_overrides = {
        'dataset_path': 'Skylion007/openwebtext',
    }
    
    try:
        cfg_dict, state_dict, log_sparsity = sae_lens_huggingface_loader(
            repo_id=base_repo_id,
            folder_name=folder_name,
            device=device,
            force_download=False,
            cfg_overrides=cfg_overrides
        )
    except Exception as e:
        # Try without .pt extension if it failed (for 32k repos that might not have .pt)
        if width == '32k' and '.pt' in folder_name:
            folder_name_no_pt = folder_name.replace('.pt', '')
            print(f"  ⚠️  Retrying without .pt extension: {folder_name_no_pt}")
            try:
                cfg_dict, state_dict, log_sparsity = sae_lens_huggingface_loader(
                    repo_id=base_repo_id,
                    folder_name=folder_name_no_pt,
                    device=device,
                    force_download=False,
                    cfg_overrides=cfg_overrides
                )
            except Exception as e2:
                print(f"  ❌ Failed with both .pt and without .pt extension")
                raise e  # Raise original error
        else:
            raise
    
    # Create SAE from config and state dict
    sae = SAE.from_dict(cfg_dict)
    sae.load_state_dict(state_dict, strict=False)
    sae = sae.to(device)
    
    print(f"✓ Loaded SAE: d_in={sae.cfg.d_in}, d_sae={sae.cfg.d_sae}")
    
    # Extract metadata
    metadata = {
        'model_name': 'gpt2-small',
        'hook_name': hook_name,
        'dataset_path': 'Skylion007/openwebtext',
        'width': width,
        'hook_type': hook_type,
        'layer': layer,
        'repo_id': base_repo_id,
        'folder_name': folder_name,
    }
    
    # Store cache path for cleanup (similar to saebench approach)
    # Use hf_hub_download to get the actual cache path
    # This is the same file that sae_lens_huggingface_loader uses, so it will use cached version
    from huggingface_hub import hf_hub_download
    import os
    
    # Download cfg.json to get the actual cache path
    # This gives us the actual folder containing cfg.json and sae_weights.safetensors
    try:
        cfg_path = hf_hub_download(
            repo_id=base_repo_id,
            filename=f"{folder_name}/cfg.json",
            force_download=False,
            cache_dir=cache_dir
        )
        # Extract the cache directory (parent of cfg.json)
        # This is the actual folder containing cfg.json and sae_weights.safetensors
        cache_path = Path(cfg_path).parent
        metadata['_cache_path'] = str(cache_path)
        print(f"  Cache path: {cache_path}")
    except Exception as e:
        # Fallback: track repo_id and folder_name for manual cleanup
        print(f"  ⚠️  Could not determine cache path: {e}")
        metadata['_cache_repo_id'] = base_repo_id
        metadata['_cache_folder_name'] = folder_name
        metadata['_cache_path'] = None  # Will use fallback cleanup
    
    return sae, metadata


def cleanup_sae_cache(cache_path: str = None, metadata: Dict = None) -> None:
    """
    Delete cached SAE files after analysis is complete.
    
    This should be called after completing analysis on a particular SAE to free disk space.
    Only SAEs need cache cleanup; LLM models use transformers library caching (separate, automatic).
    
    Args:
        cache_path: Direct path to cached SAE directory (preferred, from metadata['_cache_path'])
        metadata: Metadata dictionary from load_jbloom_oai_v5_sae() (fallback if cache_path not provided)
    """
    # Prefer direct cache path (like saebench approach)
    if cache_path:
        cache_dir = Path(cache_path)
        if cache_dir.exists():
            try:
                shutil.rmtree(cache_dir)
                print(f"✓ Cleaned up SAE cache: {cache_dir}")
                return
            except Exception as e:
                print(f"⚠️  Failed to clean up cache {cache_dir}: {e}")
        else:
            print(f"⚠️  Cache path does not exist: {cache_dir}")
            return
    
    # Fallback: use metadata to find cache (for compatibility)
    if metadata:
        repo_id = metadata.get('_cache_repo_id') or metadata.get('repo_id')
        folder_name = metadata.get('_cache_folder_name') or metadata.get('folder_name')
        
        if not repo_id or not folder_name:
            print("⚠️  No cache information in metadata, skipping cleanup")
            return
        
        # HuggingFace Hub cache structure:
        # ~/.cache/huggingface/hub/models--{org}--{repo}/snapshots/{hash}/{folder_name}/
        import os
        cache_base = os.path.expanduser("~/.cache/huggingface/hub")
        
        # Find cache directory for this repo
        repo_cache_name = f"models--{repo_id.replace('/', '--')}"
        repo_cache_path = Path(cache_base) / repo_cache_name
        
        if not repo_cache_path.exists():
            print(f"⚠️  Cache path does not exist: {repo_cache_path}")
            return
        
        # Find the snapshot directory (HuggingFace uses hash-based snapshots)
        snapshots = list(repo_cache_path.glob("snapshots/*"))
        if not snapshots:
            print(f"⚠️  No snapshots found in {repo_cache_path}")
            return
        
        # Clean up the specific folder in each snapshot
        cleaned = False
        for snapshot_dir in snapshots:
            sae_folder = snapshot_dir / folder_name
            if sae_folder.exists():
                try:
                    shutil.rmtree(sae_folder)
                    print(f"✓ Cleaned up SAE cache: {sae_folder}")
                    cleaned = True
                except Exception as e:
                    print(f"⚠️  Failed to clean up {sae_folder}: {e}")
        
        if not cleaned:
            print(f"⚠️  No SAE cache found to clean up for {repo_id}/{folder_name}")
    else:
        print("⚠️  No cache_path or metadata provided, skipping cleanup")


def cleanup_repo_cache(repo_id: str) -> None:
    """
    Clean up entire HuggingFace repo cache including blobs after all SAEs from a repo are done.
    
    This should be called after completing analysis on all SAEs from a particular repo to free disk space.
    This deletes both snapshots and blobs, which is safe once all SAEs from the repo are processed.
    
    Args:
        repo_id: HuggingFace repo ID (e.g., 'jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs')
    """
    import os
    cache_base = Path(os.path.expanduser("~/.cache/huggingface/hub"))
    
    # Convert repo_id to cache directory name format
    # e.g., 'jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs' -> 'models--jbloom--GPT2-Small-OAI-v5-32k-resid-post-SAEs'
    repo_cache_name = f"models--{repo_id.replace('/', '--')}"
    repo_cache_dir = cache_base / repo_cache_name
    
    if repo_cache_dir.exists():
        try:
            # Delete entire repo cache (including blobs and snapshots)
            shutil.rmtree(repo_cache_dir)
            print(f"✓ Cleaned up repo cache (including blobs): {repo_cache_dir}")
        except Exception as e:
            print(f"⚠️  Failed to clean up repo cache {repo_cache_dir}: {e}")
    else:
        print(f"⚠️  Repo cache does not exist: {repo_cache_dir}")

