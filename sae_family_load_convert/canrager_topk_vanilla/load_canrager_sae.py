#!/usr/bin/env python3
"""
canrager TopK+Vanilla SAE Loader

Functions for loading canrager TopK and Vanilla SAEs from HuggingFace and converting them
to SAELens-compatible format.

Structure:
- Repos: canrager/saebench_gemma-2-2b_width-2pow{12|14|16}_date-1109
- Architectures: TopK or Vanilla
- Layers: 5, 12, 19
- Trainers: trainer_0 through trainer_5
- Path: gemma-2-2b_{topk|vanilla}_width-2pow{12|14|16}_date-1109/resid_post_layer_{5|12|19}/trainer_{0-5}/
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


def get_repo_id(width_name: str) -> str:
    """
    Get HuggingFace repo ID for given width.
    
    Args:
        width_name: '4k', '16k', or '65k'
        
    Returns:
        Repo ID string
    """
    width_to_repo = {
        '4k': 'canrager/saebench_gemma-2-2b_width-2pow12_date-1109',
        '16k': 'canrager/saebench_gemma-2-2b_width-2pow14_date-1109',
        '65k': 'canrager/saebench_gemma-2-2b_width-2pow16_date-1109',
    }
    return width_to_repo.get(width_name.lower(), width_to_repo['4k'])


def get_architecture_folder(architecture: str, width_name: str) -> str:
    """
    Get folder name for architecture in repo.
    
    Args:
        architecture: 'topk' or 'vanilla'
        width_name: '4k', '16k', or '65k'
        
    Returns:
        Folder name in repo (e.g., 'gemma-2-2b_topk_width-2pow12_date-1109')
    """
    width_to_pow = {
        '4k': '2pow12',
        '16k': '2pow14',
        '65k': '2pow16',
    }
    pow_str = width_to_pow.get(width_name.lower(), '2pow12')
    
    arch_name = 'topk' if architecture.lower() == 'topk' else 'vanilla'
    return f'gemma-2-2b_{arch_name}_width-{pow_str}_date-1109'


def download_canrager_sae(
    repo_id: str,
    architecture: str,
    layer: int,
    trainer_folder: str = "trainer_0",
    cache_dir: Optional[str] = None
) -> Path:
    """
    Download canrager SAE from HuggingFace.
    
    Args:
        repo_id: HuggingFace repo ID (e.g., "canrager/saebench_gemma-2-2b_width-2pow12_date-1109")
        architecture: 'topk' or 'vanilla'
        layer: Layer number (5, 12, or 19)
        trainer_folder: Trainer folder (e.g., "trainer_0")
        cache_dir: Optional cache directory (default: None, uses HuggingFace default cache)
        
    Returns:
        Path to downloaded SAE directory
    """
    print(f"Downloading canrager SAE from HuggingFace...")
    print(f"  Repo: {repo_id}")
    print(f"  Architecture: {architecture}")
    print(f"  Layer: {layer}")
    print(f"  Trainer: {trainer_folder}")
    
    # Infer width from repo_id
    if '2pow12' in repo_id:
        width_name = '4k'
    elif '2pow14' in repo_id:
        width_name = '16k'
    elif '2pow16' in repo_id:
        width_name = '65k'
    else:
        width_name = '4k'  # Default
    
    # Get architecture folder name
    arch_folder = get_architecture_folder(architecture, width_name)
    layer_folder = f"resid_post_layer_{layer}"
    
    # Download the specific folder structure: {arch_folder}/{layer_folder}/{trainer_folder}/
    pattern = f"{arch_folder}/{layer_folder}/{trainer_folder}/**"
    local_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=pattern,
        cache_dir=cache_dir,
        local_files_only=False
    )
    
    # The downloaded folder structure: local_dir/{arch_folder}/{layer_folder}/{trainer_folder}/
    sae_path = Path(local_dir) / arch_folder / layer_folder / trainer_folder
    
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
                    f"Available folders: {available_folders[:10]}"
                )
    
    print(f"✓ Downloaded SAE to: {sae_path}")
    return sae_path


def load_canrager_config(config_path: Path) -> Dict:
    """
    Load canrager config.json file.
    
    Args:
        config_path: Path to config.json
        
    Returns:
        Dictionary with config data
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def convert_canrager_to_saelens(
    ae_pt_path: Path,
    config_json_path: Path,
    device: str = None
) -> SAE:
    """
    Convert canrager format to SAELens format.
    
    Same format as SAE Bench: ae.pt + config.json
    Reuses conversion logic from saebench_gemma.
    
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
    config = load_canrager_config(config_json_path)
    
    trainer_config = config.get('trainer', {})
    dict_class = trainer_config.get('dict_class', 'AutoEncoder')
    activation_dim = trainer_config.get('activation_dim', 768)
    dict_size = trainer_config.get('dict_size', 65536)
    k = trainer_config.get('k', 20)
    lm_name = trainer_config.get('lm_name', 'google/gemma-2-2b')
    layer = trainer_config.get('layer', 12)
    
    # Load checkpoint to CPU first (safer for large 65k SAEs)
    checkpoint = torch.load(ae_pt_path, map_location='cpu')
    
    # Determine architecture
    is_topk = 'TopK' in dict_class or 'topk' in dict_class.lower() or 'top_k' in dict_class.lower()
    
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
    
    # Map weights from canrager format to SAELens format (transpose on CPU)
    state_dict = {}
    
    if 'encoder.weight' in checkpoint:
        # canrager: [d_sae, d_in] -> SAELens: [d_in, d_sae] (transpose)
        state_dict['W_enc'] = checkpoint['encoder.weight'].T
    if 'decoder.weight' in checkpoint:
        # canrager: [d_in, d_sae] -> SAELens: [d_sae, d_in] (transpose)
        state_dict['W_dec'] = checkpoint['decoder.weight'].T
    if 'encoder.bias' in checkpoint:
        state_dict['b_enc'] = checkpoint['encoder.bias']
    elif 'b_enc' in checkpoint:
        state_dict['b_enc'] = checkpoint['b_enc']
    if 'b_dec' in checkpoint:
        state_dict['b_dec'] = checkpoint['b_dec']
    
    # Load state dict (will move tensors to device during load_state_dict)
    sae.load_state_dict(state_dict, strict=False)
    sae = sae.to(device)
    
    return sae


def load_canrager_sae(
    architecture: str,
    width_name: str,
    layer: int,
    trainer_folder: str = "trainer_0",
    device: str = None,
    cache_dir: Optional[str] = None
) -> Tuple[SAE, Dict]:
    """
    Load canrager SAE from HuggingFace and convert to SAELens format.
    
    Args:
        architecture: 'topk' or 'vanilla'
        width_name: '4k', '16k', or '65k'
        layer: Layer number (5, 12, or 19)
        trainer_folder: Trainer folder (e.g., "trainer_0")
        device: Device to load SAE on (default: auto-detect cuda:0, requires GPU)
        cache_dir: Optional cache directory (default: None, uses HuggingFace default cache)
        
    Returns:
        Tuple of (SAE object, metadata dictionary)
    """
    # Auto-detect device if not specified - require GPU
    if device is None:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA/GPU is not available. GPU is required for SAE operations.")
        device = "cuda:0"
    
    # Get repo ID
    repo_id = get_repo_id(width_name)
    
    # Download SAE
    sae_path = download_canrager_sae(
        repo_id=repo_id,
        architecture=architecture,
        layer=layer,
        trainer_folder=trainer_folder,
        cache_dir=cache_dir
    )
    
    # canrager format uses ae.pt and config.json
    ae_pt_path = sae_path / "ae.pt"
    config_json_path = sae_path / "config.json"
    
    if not ae_pt_path.exists() or not config_json_path.exists():
        raise FileNotFoundError(
            f"canrager SAE files not found at {sae_path}. "
            f"Expected ae.pt and config.json"
        )
    
    # Convert canrager format to SAELens format
    print(f"Converting canrager format to SAELens format...")
    sae = convert_canrager_to_saelens(ae_pt_path, config_json_path, device=device)
    print(f"✓ Loaded SAE from canrager format")
    
    # Extract metadata
    if hasattr(sae.cfg, 'metadata') and sae.cfg.metadata:
        metadata = sae.cfg.metadata
        metadata_dict = {
            'model_name': getattr(metadata, 'model_name', 'google/gemma-2-2b'),
            'hook_name': getattr(metadata, 'hook_name', None),
            'dataset_path': 'NeelNanda/pile-10k',
        }
    else:
        metadata_dict = {
            'model_name': 'google/gemma-2-2b',
            'hook_name': None,
            'dataset_path': 'NeelNanda/pile-10k',
        }
    
    # Set hook_name from layer
    if not metadata_dict.get('hook_name'):
        metadata_dict['hook_name'] = f'blocks.{layer}.hook_resid_post'
    
    print(f"  Architecture: {architecture}")
    print(f"  Layer: {layer}")
    print(f"  d_in: {sae.cfg.d_in}, d_sae: {sae.cfg.d_sae}")
    
    # Store cache path in metadata for cleanup later
    metadata_dict['_cache_path'] = str(sae_path)
    metadata_dict['architecture'] = architecture
    metadata_dict['width_name'] = width_name
    metadata_dict['layer'] = layer
    metadata_dict['trainer'] = trainer_folder
    
    return sae, metadata_dict


def cleanup_sae_cache(cache_path: str) -> None:
    """
    Delete cached SAE files after analysis is complete.
    
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
    
    Args:
        repo_id: HuggingFace repo ID (e.g., 'canrager/saebench_gemma-2-2b_width-2pow12_date-1109')
    """
    cache_base = Path.home() / ".cache" / "huggingface" / "hub"
    
    # Convert repo_id to cache directory name format
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

