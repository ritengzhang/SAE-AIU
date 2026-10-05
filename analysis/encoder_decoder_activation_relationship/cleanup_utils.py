#!/usr/bin/env python3
"""
Utility functions for cleaning up cached models and SAEs after analysis.
"""

import shutil
from pathlib import Path
from typing import Optional


def cleanup_model_cache(model_name: str) -> None:
    """
    Delete cached transformers model files after analysis is complete.
    
    This should be called after completing analysis on a particular model to free disk space.
    Transformers library caches models in ~/.cache/huggingface/hub/
    
    The cache directory format is: models--{org}--{model_name} where '/' becomes '--'
    For example:
    - 'pythia-160m-deduped' -> 'models--EleutherAI--pythia-160m-deduped'
    - 'gemma-2-2b-it' -> 'models--google--gemma-2-2b-it'
    
    Args:
        model_name: Model name (e.g., 'pythia-160m-deduped', 'gemma-2-2b-it')
    """
    cache_dir = Path.home() / ".cache" / "huggingface" / "hub"
    
    if not cache_dir.exists():
        print(f"⚠️  Cache directory does not exist: {cache_dir}")
        return
    
    # Common organization prefixes for different model families
    org_prefixes = {
        'pythia': 'EleutherAI',
        'gpt-neo': 'EleutherAI',
        'gemma': 'google',
        'gpt2': '',  # GPT-2 models don't have org prefix
        'distilgpt2': '',  # DistilGPT-2 doesn't have org prefix
        'opt': 'facebook',
    }
    
    # Determine org prefix based on model name
    org_prefix = None
    for key, org in org_prefixes.items():
        if key in model_name.lower():
            org_prefix = org
            break
    
    # Try different cache directory name patterns
    possible_names = []
    
    # Pattern 1: With org prefix (if we found one)
    if org_prefix:
        cache_name = f"models--{org_prefix}--{model_name.replace('/', '--')}"
        possible_names.append(cache_name)
    
    # Pattern 2: Direct model name (for models without org prefix like gpt2)
    if not org_prefix or 'gpt2' in model_name.lower() or 'distil' in model_name.lower():
        cache_name = f"models--{model_name.replace('/', '--')}"
        possible_names.append(cache_name)
    
    # Pattern 3: Search for any directory containing the model name
    # This is a fallback for edge cases
    model_name_clean = model_name.replace('/', '--').replace('-', '_')
    for item in cache_dir.iterdir():
        if item.is_dir() and item.name.startswith('models--') and model_name_clean in item.name.replace('-', '_'):
            if item.name not in possible_names:
                possible_names.append(item.name)
    
    # Try to delete each possible cache directory
    deleted_any = False
    for cache_name in possible_names:
        model_cache_path = cache_dir / cache_name
        if model_cache_path.exists():
            try:
                shutil.rmtree(model_cache_path)
                print(f"✓ Cleaned up model cache: {model_cache_path.name}")
                deleted_any = True
            except Exception as e:
                print(f"⚠️  Failed to clean up {model_cache_path.name}: {e}")
    
    if not deleted_any:
        print(f"⚠️  Could not find model cache for: {model_name}")
        print(f"   Searched for: {possible_names}")
        print(f"   Cache directory: {cache_dir}")

