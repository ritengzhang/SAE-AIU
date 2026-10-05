#!/usr/bin/env python3
"""
SAE Bench Gemma SAE Loader

Functions for loading SAE Bench Gemma models from HuggingFace and converting them
to SAELens-compatible format.

Note: Gemma repos may have different folder structure than Pythia repos.
"""

import json
import shutil
import torch
from pathlib import Path
from typing import Optional, Dict, Tuple
from huggingface_hub import snapshot_download

# Add parent directories to path for imports
import sys
parent_dir = Path(__file__).parent.parent.parent
sys.path.insert(0, str(parent_dir))

sae_lens_dir = parent_dir / "SAELens-main"
sys.path.insert(0, str(sae_lens_dir))

from sae_lens import SAE
from sae_lens.saes.standard_sae import StandardSAEConfig, StandardSAE
from sae_lens.saes.topk_sae import TopKSAEConfig, TopKSAE
from sae_lens.saes.sae import SAEMetadata


def map_sae_type_to_repo_folder(sae_type_folder: str, repo_id: str) -> str:
    """
    Map expected SAE type folder name to actual folder name in repo.
    
    The 4k repo (adamkarvonen) uses names like: Standard_gemma-2-2b__0108
    The 16k/65k repos (canrager) use names like: gemma-2-2b_standard_new_width-2pow14_date-0107
    
    Args:
        sae_type_folder: Expected SAE type folder name
        repo_id: HuggingFace repo ID to determine which naming convention to use
        
    Returns:
        Actual folder name in the repo
    """
    # Check if this is a canrager repo (16k or 65k) which uses different naming
    if 'canrager' in repo_id.lower():
        # Map expected names to actual folder names in canrager repos
        mapping = {
            'Standard_gemma-2-2b__0108': 'gemma-2-2b_standard_new_width-2pow14_date-0107',
            'TopK_gemma-2-2b__0108': 'gemma-2-2b_top_k_width-2pow14_date-0107',
            'BatchTopK_gemma-2-2b__0108': 'gemma-2-2b_batch_top_k_width-2pow14_date-0107',
            'JumpRelu_gemma-2-2b__0108': 'gemma-2-2b_jump_relu_width-2pow14_date-0107',
            'GatedSAE_gemma-2-2b__0108': 'gemma-2-2b_gated_width-2pow14_date-0107',
            'PAnneal_gemma-2-2b__0108': 'gemma-2-2b_p_anneal_width-2pow14_date-0107',
            'MatryoshkaBatchTopK_gemma-2-2b__0108': 'gemma-2-2b_matryoshka_batch_top_k_width-2pow14_date-0107',
        }
        
        # Handle 65k width (2pow16 instead of 2pow14)
        if '2pow16' in repo_id or '65k' in repo_id.lower():
            # Replace 2pow14 with 2pow16 for 65k repos
            mapped = mapping.get(sae_type_folder, sae_type_folder)
            if '2pow14' in mapped:
                mapped = mapped.replace('2pow14', '2pow16')
            return mapped
        
        return mapping.get(sae_type_folder, sae_type_folder)
    
    # For adamkarvonen repos (4k), use the name as-is
    return sae_type_folder


def download_saebench_gemma_sae(
    repo_id: str,
    sae_type_folder: str,
    layer_folder: str = "resid_post_layer_8",
    trainer_folder: str = "trainer_0",
    cache_dir: Optional[str] = None
) -> Path:
    """
    Download SAE Bench SAE from HuggingFace.
    
    Args:
        repo_id: HuggingFace repo ID (e.g., "adamkarvonen/saebench_pythia-160m-deduped_width-2pow16_date-0108")
        sae_type_folder: SAE type folder (e.g., "TopK_pythia-160m-deduped__0108")
        layer_folder: Layer folder (e.g., "resid_post_layer_8")
        trainer_folder: Trainer/seed folder (e.g., "trainer_0")
        cache_dir: Optional cache directory (default: None, uses HuggingFace default cache at ~/.cache/huggingface/hub/)
            Note: SAEs are cached by HuggingFace Hub. Use cleanup_sae_cache() after analysis to free disk space.
        
    Returns:
        Path to downloaded SAE directory
    """
    print(f"Downloading SAE Bench SAE from HuggingFace...")
    print(f"  Repo: {repo_id}")
    print(f"  SAE Type: {sae_type_folder}")
    print(f"  Layer: {layer_folder}")
    print(f"  Trainer: {trainer_folder}")
    
    # Map expected folder name to actual folder name in repo
    actual_folder_name = map_sae_type_to_repo_folder(sae_type_folder, repo_id)
    if actual_folder_name != sae_type_folder:
        print(f"  Mapped to actual folder: {actual_folder_name}")
    
    # Download the specific folder structure: {actual_folder_name}/{layer}/{trainer}/
    pattern = f"{actual_folder_name}/{layer_folder}/{trainer_folder}/**"
    local_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=pattern,
        cache_dir=cache_dir,
        local_files_only=False
    )
    
    # The downloaded folder structure: local_dir/{actual_folder_name}/{layer}/{trainer}/
    sae_path = Path(local_dir) / actual_folder_name / layer_folder / trainer_folder
    
    if not sae_path.exists():
        # Try alternative path structure
        alt_path = Path(local_dir) / layer_folder / trainer_folder
        if alt_path.exists():
            sae_path = alt_path
        else:
            # Last resort: check if files are directly in local_dir
            if (Path(local_dir) / "ae.pt").exists() or (Path(local_dir) / "config.json").exists():
                sae_path = Path(local_dir)
            else:
                # List available folders for debugging
                available_folders = [d.name for d in Path(local_dir).iterdir() if d.is_dir()] if Path(local_dir).exists() else []
                raise FileNotFoundError(
                    f"Could not find SAE files at expected path: {sae_path}\n"
                    f"Downloaded to: {local_dir}\n"
                    f"Pattern used: {pattern}\n"
                    f"Available folders: {available_folders[:10]}"  # Show first 10 folders
                )
    
    print(f"✓ Downloaded SAE to: {sae_path}")
    return sae_path


def load_saebench_config(config_path: Path) -> Dict:
    """
    Load SAE Bench config.json file.
    
    Args:
        config_path: Path to config.json
        
    Returns:
        Dictionary with config data
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def convert_saebench_to_saelens(
    ae_pt_path: Path,
    config_json_path: Path,
    device: str = None
) -> SAE:
    """
    Convert SAE Bench format to SAELens format.
    
    SAE Bench format:
    - encoder.weight: [d_sae, d_in]
    - decoder.weight: [d_in, d_sae]
    - encoder.bias: [d_sae]
    - b_dec: [d_in]
    
    SAELens format:
    - W_enc: [d_in, d_sae] (transpose)
    - W_dec: [d_sae, d_in] (transpose)
    - b_enc: [d_sae]
    - b_dec: [d_in]
    
    Args:
        ae_pt_path: Path to ae.pt checkpoint file
        config_json_path: Path to config.json file
        device: Device to load SAE on (default: auto-detect cuda:0, requires GPU)
        
    Returns:
        SAELens SAE object
    """
    # Auto-detect device if not specified - require GPU
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU is not available. GPU is required for SAE operations.")
        device = "cuda:0"
    
    # Load config
    saebench_config = load_saebench_config(config_json_path)
    
    trainer_config = saebench_config.get('trainer', {})
    dict_class = trainer_config.get('dict_class', 'AutoEncoder')
    activation_dim = trainer_config.get('activation_dim', 768)
    dict_size = trainer_config.get('dict_size', 65536)
    k = trainer_config.get('k', 20)
    lm_name = trainer_config.get('lm_name', 'EleutherAI/pythia-160m-deduped')
    layer = trainer_config.get('layer', 8)
    
    # Load checkpoint to CPU first (safer for large 65k SAEs - prevents CUDA errors during transpose)
    # For 65k SAEs, loading directly to GPU can cause OOM/CUDA errors during transpose operations
    checkpoint = torch.load(ae_pt_path, map_location='cpu')
    
    # Determine architecture
    is_topk = 'TopK' in dict_class or 'topk' in dict_class.lower()
    
    # Create SAELens config
    if is_topk:
        sae_cfg = TopKSAEConfig(
            d_in=activation_dim,
            d_sae=dict_size,
            k=k,
            device=device,
            dtype="float32",
            metadata=SAEMetadata(
                model_name=lm_name,
                hook_name=f"blocks.{layer}.hook_resid_post",
                dataset_path="NeelNanda/pile-10k",
            )
        )
        sae = TopKSAE(sae_cfg)
    else:
        # Standard (ReLU) SAE
        sae_cfg = StandardSAEConfig(
            d_in=activation_dim,
            d_sae=dict_size,
            device=device,
            dtype="float32",
            metadata=SAEMetadata(
                model_name=lm_name,
                hook_name=f"blocks.{layer}.hook_resid_post",
                dataset_path="NeelNanda/pile-10k",
            )
        )
        sae = StandardSAE(sae_cfg)
    
    # Map weights from SAE Bench format to SAELens format (transpose on CPU, safer for large tensors)
    state_dict = {}
    
    if 'encoder.weight' in checkpoint:
        # SAE Bench: [d_sae, d_in] -> SAELens: [d_in, d_sae] (transpose on CPU)
        state_dict['W_enc'] = checkpoint['encoder.weight'].T
    elif 'W_enc' in checkpoint:
        # Some checkpoints (e.g., JumpReLU, Matryoshka) have W_enc directly
        # Already in SAELens format: [d_in, d_sae]
        state_dict['W_enc'] = checkpoint['W_enc']
    
    if 'decoder.weight' in checkpoint:
        # SAE Bench: [d_in, d_sae] -> SAELens: [d_sae, d_in] (transpose on CPU)
        state_dict['W_dec'] = checkpoint['decoder.weight'].T
    elif 'W_dec' in checkpoint:
        # Some checkpoints (e.g., JumpReLU, Matryoshka) have W_dec directly
        # Already in SAELens format: [d_sae, d_in]
        state_dict['W_dec'] = checkpoint['W_dec']
    
    if 'encoder.bias' in checkpoint:
        state_dict['b_enc'] = checkpoint['encoder.bias']
    elif 'b_enc' in checkpoint:
        state_dict['b_enc'] = checkpoint['b_enc']
    if 'b_dec' in checkpoint:
        state_dict['b_dec'] = checkpoint['b_dec']
    
    # Load state dict (will move tensors to device during load_state_dict)
    sae.load_state_dict(state_dict, strict=False)
    sae = sae.to(device)  # Ensure everything is on the correct device
    
    return sae


def load_saebench_gemma_sae(
    repo_id: str,
    sae_type_folder: str,
    layer_folder: str = "resid_post_layer_12",
    trainer_folder: str = "trainer_0",
    device: str = None,
    cache_dir: Optional[str] = None
) -> Tuple[SAE, Dict]:
    """
    Load SAE Bench SAE from HuggingFace and convert to SAELens format.
    
    Args:
        repo_id: HuggingFace repo ID
        sae_type_folder: SAE type folder name
        layer_folder: Layer folder name
        trainer_folder: Trainer folder name
        device: Device to load SAE on (default: auto-detect cuda:0, requires GPU)
            Can be overridden to use cuda:1 or other devices.
        cache_dir: Optional cache directory (default: None, uses HuggingFace default cache)
            Note: SAEs are cached by HuggingFace Hub. Use cleanup_sae_cache() after analysis.
        
    Returns:
        Tuple of (SAE object, metadata dictionary)
    """
    # Auto-detect device if not specified - require GPU
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU is not available. GPU is required for SAE operations.")
        device = "cuda:0"
    
    # Download SAE
    sae_path = download_saebench_gemma_sae(
        repo_id=repo_id,
        sae_type_folder=sae_type_folder,
        layer_folder=layer_folder,
        trainer_folder=trainer_folder,
        cache_dir=cache_dir
    )
    
    # SAE Bench format uses ae.pt and config.json (not SAELens format)
    # Convert directly from SAE Bench format to SAELens-compatible format
    ae_pt_path = sae_path / "ae.pt"
    config_json_path = sae_path / "config.json"
    
    if not ae_pt_path.exists() or not config_json_path.exists():
        raise FileNotFoundError(
            f"SAE Bench files not found at {sae_path}. "
            f"Expected ae.pt and config.json"
        )
    
    # Convert SAE Bench format to SAELens format
    print(f"Converting SAE Bench format to SAELens format...")
    sae = convert_saebench_to_saelens(ae_pt_path, config_json_path, device=device)
    print(f"✓ Loaded SAE from SAE Bench format")
    
    # Extract metadata
    if hasattr(sae.cfg, 'metadata') and sae.cfg.metadata:
        metadata = sae.cfg.metadata
        metadata_dict = {
            'model_name': getattr(metadata, 'model_name', 'pythia-160m-deduped'),
            'hook_name': getattr(metadata, 'hook_name', None),
            'dataset_path': 'NeelNanda/pile-10k',
        }
    else:
        metadata_dict = {
            'model_name': 'pythia-160m-deduped',
            'hook_name': None,
            'dataset_path': 'NeelNanda/pile-10k',
        }
    
    # Try to get hook_name from config or infer from layer name
    if hasattr(sae.cfg, 'hook_name') and sae.cfg.hook_name:
        metadata_dict['hook_name'] = sae.cfg.hook_name
    elif not metadata_dict.get('hook_name'):
        # Infer hook from layer folder name (e.g., "resid_post_layer_8" -> layer 8)
        # Extract layer number from layer_folder string
        import re
        layer_match = re.search(r'layer_(\d+)', layer_folder)
        if layer_match:
            layer_num = int(layer_match.group(1))
            metadata_dict['hook_name'] = f'blocks.{layer_num}.hook_resid_post'
        else:
            # Fallback: try common patterns
            if 'layer_8' in layer_folder:
                metadata_dict['hook_name'] = 'blocks.8.hook_resid_post'
            elif 'layer_12' in layer_folder:
                metadata_dict['hook_name'] = 'blocks.12.hook_resid_post'
            else:
                metadata_dict['hook_name'] = 'blocks.8.hook_resid_post'  # Default fallback
        print(f"⚠️  No hook_name found, inferred from layer_folder: {metadata_dict['hook_name']}")
    
    print(f"  d_in: {sae.cfg.d_in}, d_sae: {sae.cfg.d_sae}")
    
    # Store cache path in metadata for cleanup later
    metadata_dict['_cache_path'] = str(sae_path)
    
    return sae, metadata_dict


def cleanup_sae_cache(cache_path: str) -> None:
    """
    Delete cached SAE files after analysis is complete.
    
    This should be called after completing analysis on a particular SAE to free disk space.
    Only SAEs need cache cleanup; LLM models use transformers library caching (separate, automatic).
    
    Note: This deletes the snapshot directory. Blobs are cleaned up separately when all SAEs
    from a repo are done (see cleanup_repo_cache()).
    
    Args:
        cache_path: Path to the cached SAE directory (from metadata['_cache_path'])
    """
    cache_dir = Path(cache_path)
    if cache_dir.exists():
        try:
            shutil.rmtree(cache_dir)
            print(f"✓ Cleaned up SAE cache: {cache_dir}")
        except Exception as e:
            print(f"⚠️  Failed to clean up cache {cache_dir}: {e}")
    else:
        print(f"⚠️  Cache path does not exist: {cache_dir}")


def cleanup_repo_cache(repo_id: str) -> None:
    """
    Clean up entire HuggingFace repo cache including blobs after all SAEs from a repo are done.
    
    This should be called after completing analysis on all SAEs from a particular repo to free disk space.
    This deletes both snapshots and blobs, which is safe once all SAEs from the repo are processed.
    
    Args:
        repo_id: HuggingFace repo ID (e.g., 'canrager/saebench_gemma-2-2b_width-2pow16_date-0107')
    """
    cache_base = Path.home() / ".cache" / "huggingface" / "hub"
    
    # Convert repo_id to cache directory name format
    # e.g., 'canrager/saebench_gemma-2-2b_width-2pow16_date-0107' -> 'models--canrager--saebench_gemma-2-2b_width-2pow16_date-0107'
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

