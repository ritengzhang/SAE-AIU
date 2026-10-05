#!/usr/bin/env python3
"""
SAE Bench SAE Loader

Functions for loading SAE Bench models from HuggingFace and converting them
to SAELens-compatible format.
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


def download_saebench_sae(
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
    
    # Download the specific folder structure: {sae_type_folder}/{layer}/{trainer}/
    pattern = f"{sae_type_folder}/{layer_folder}/{trainer_folder}/**"
    local_dir = snapshot_download(
        repo_id=repo_id,
        allow_patterns=pattern,
        cache_dir=cache_dir,
        local_files_only=False
    )
    
    # The downloaded folder structure: local_dir/{sae_type_folder}/{layer}/{trainer}/
    sae_path = Path(local_dir) / sae_type_folder / layer_folder / trainer_folder
    
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
                raise FileNotFoundError(
                    f"Could not find SAE files at expected path: {sae_path}\n"
                    f"Downloaded to: {local_dir}\n"
                    f"Pattern used: {pattern}"
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
    
    # Load checkpoint
    checkpoint = torch.load(ae_pt_path, map_location=device)
    
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
    
    # Map weights from SAE Bench format to SAELens format
    state_dict = {}
    
    if 'encoder.weight' in checkpoint:
        # SAE Bench: [d_sae, d_in] -> SAELens: [d_in, d_sae] (transpose)
        state_dict['W_enc'] = checkpoint['encoder.weight'].T
    elif 'W_enc' in checkpoint:
        # Some checkpoints (e.g., JumpReLU, Matryoshka) have W_enc directly
        # Already in SAELens format: [d_in, d_sae]
        state_dict['W_enc'] = checkpoint['W_enc']
    
    if 'decoder.weight' in checkpoint:
        # SAE Bench: [d_in, d_sae] -> SAELens: [d_sae, d_in] (transpose)
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
    
    # Load state dict
    sae.load_state_dict(state_dict, strict=False)
    sae = sae.to(device)
    
    return sae


def load_saebench_sae(
    repo_id: str,
    sae_type_folder: str,
    layer_folder: str = "resid_post_layer_8",
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
    sae_path = download_saebench_sae(
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


def convert_saebench_to_sparsify_format(
    saebench_dir: Path,
    output_dir: Path,
    hookpoint: str,
    overwrite: bool = False
) -> Tuple[Path, Path]:
    """
    Convert SAE Bench format to Sparsify/Delphi format.
    
    This creates the files that Delphi/Sparsify expects:
    - {output_dir}/{hookpoint}/cfg.json (Sparsify config format)
    - {output_dir}/{hookpoint}/sae.safetensors (SafeTensors weights)
    
    SAE Bench format:
    - {sae_type}/{layer}/{trainer}/config.json
    - {sae_type}/{layer}/{trainer}/ae.pt
    
    Sparsify/Delphi format:
    - {path}/{hookpoint}/cfg.json
    - {path}/{hookpoint}/sae.safetensors
    
    Args:
        saebench_dir: Path to SAE Bench directory containing ae.pt and config.json
        output_dir: Base output directory (will create {output_dir}/{hookpoint}/)
        hookpoint: Hookpoint name (e.g., "layers.8" or "resid_post_layer_8")
        overwrite: Whether to overwrite existing files (default: False)
        
    Returns:
        Tuple of (cfg_json_path, sae_safetensors_path)
    """
    # Create hookpoint directory
    hookpoint_dir = output_dir / hookpoint
    hookpoint_dir.mkdir(parents=True, exist_ok=True)
    
    cfg_json_path = hookpoint_dir / "cfg.json"
    sae_safetensors_path = hookpoint_dir / "sae.safetensors"
    
    # Load SAE Bench config.json
    config_json_src = saebench_dir / "config.json"
    if not config_json_src.exists():
        raise FileNotFoundError(f"SAE Bench config.json not found at {config_json_src}")
    
    with open(config_json_src, 'r') as f:
        saebench_config = json.load(f)
    
    trainer_cfg = saebench_config.get('trainer', {})
    
    # Convert to sparsify format (flat structure, not nested)
    # Sparsify expects: d_in, num_latents (or expansion_factor), k (for TopK)
    activation_dim = trainer_cfg.get('activation_dim', 768)
    dict_size = trainer_cfg.get('dict_size', 4096)
    
    sparsify_cfg = {
        'd_in': activation_dim,
        'num_latents': dict_size,
        'expansion_factor': dict_size / activation_dim,
    }
    
    # Add k for TopK/BatchTopK SAEs
    if 'k' in trainer_cfg:
        sparsify_cfg['k'] = trainer_cfg['k']
        sparsify_cfg['signed'] = trainer_cfg.get('signed', False)
    
    # Add normalize_decoder if present
    if 'normalize_decoder' in trainer_cfg:
        sparsify_cfg['normalize_decoder'] = trainer_cfg['normalize_decoder']
    
    # Write sparsify-compatible cfg.json
    if cfg_json_path.exists() and not overwrite:
        print(f"⚠️  cfg.json already exists at {cfg_json_path}, skipping (use overwrite=True to replace)")
    else:
        with open(cfg_json_path, 'w') as f:
            json.dump(sparsify_cfg, f, indent=2)
        print(f"✓ Created cfg.json (sparsify format) with d_in={activation_dim}, num_latents={dict_size}")
    
    # Convert ae.pt to sae.safetensors
    ae_pt_src = saebench_dir / "ae.pt"
    if not ae_pt_src.exists():
        raise FileNotFoundError(f"SAE Bench ae.pt not found at {ae_pt_src}")
    
    if sae_safetensors_path.exists() and not overwrite:
        # Check if it's actually a SafeTensors file with correct keys
        try:
            from safetensors import safe_open
            with safe_open(sae_safetensors_path, framework="pt", device="cpu") as f:
                keys = set(f.keys())
            # Sparsify expects: encoder.weight, encoder.bias, W_dec, b_dec (mix format!)
            required_keys = {"encoder.weight", "encoder.bias", "W_dec", "b_dec"}
            if required_keys.issubset(keys):
                print(f"✓ SafeTensors file already exists with correct keys, skipping conversion")
                return cfg_json_path, sae_safetensors_path
            else:
                print(f"⚠️  SafeTensors file exists but has wrong keys {keys}, recreating...")
        except:
            print(f"⚠️  SafeTensors file exists but is corrupted, recreating...")
    
    # Convert PyTorch checkpoint to SafeTensors format
    try:
        from safetensors.torch import save_file
        
        # Load PyTorch checkpoint
        checkpoint = torch.load(ae_pt_src, map_location="cpu", weights_only=True)
        
        # Convert to SafeTensors format for sparsify
        # Sparsify's SparseCoder.load_from_disk expects: encoder.weight, encoder.bias, W_dec, b_dec
        # (Mix format: encoder.weight/encoder.bias but W_dec/b_dec, not decoder.weight!)
        state_dict = {}
        
        # Handle encoder.weight (from SAE Bench format or convert from W_enc)
        if "encoder.weight" in checkpoint:
            state_dict["encoder.weight"] = checkpoint["encoder.weight"].contiguous()
        elif "W_enc" in checkpoint:
            # W_enc is [d_in, d_sae], encoder.weight should be [d_sae, d_in] (transpose)
            state_dict["encoder.weight"] = checkpoint["W_enc"].T.contiguous()
        
        # Handle encoder.bias (from SAE Bench format or convert from b_enc)
        if "encoder.bias" in checkpoint:
            state_dict["encoder.bias"] = checkpoint["encoder.bias"].contiguous()
        elif "b_enc" in checkpoint:
            state_dict["encoder.bias"] = checkpoint["b_enc"].contiguous()
        
        # Handle W_dec (convert from decoder.weight or use W_dec directly)
        if "W_dec" in checkpoint:
            state_dict["W_dec"] = checkpoint["W_dec"].contiguous()
        elif "decoder.weight" in checkpoint:
            # decoder.weight is [d_in, d_sae], W_dec should be [d_sae, d_in] (transpose)
            state_dict["W_dec"] = checkpoint["decoder.weight"].T.contiguous()
        
        # Handle b_dec (from SAE Bench - can be "b_dec", "decoder.bias", or "bias")
        if "b_dec" in checkpoint:
            state_dict["b_dec"] = checkpoint["b_dec"].contiguous()
        elif "decoder.bias" in checkpoint:
            state_dict["b_dec"] = checkpoint["decoder.bias"].contiguous()
        elif "bias" in checkpoint:
            # Some SAE Bench files use "bias" instead of "b_dec"
            state_dict["b_dec"] = checkpoint["bias"].contiguous()
        
        # Save as SafeTensors
        save_file(state_dict, sae_safetensors_path)
        print(f"✓ Converted ae.pt -> sae.safetensors (SafeTensors format)")
    except ImportError:
        raise ImportError(
            "safetensors library not available. Install with: pip install safetensors"
        )
    except Exception as e:
        raise RuntimeError(f"Failed to convert weights: {e}") from e
    
    return cfg_json_path, sae_safetensors_path


def cleanup_sae_cache(cache_path: str) -> None:
    """
    Delete cached SAE files after analysis is complete.
    
    This should be called after completing analysis on a particular SAE to free disk space.
    Only SAEs need cache cleanup; LLM models use transformers library caching (separate, automatic).
    
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

