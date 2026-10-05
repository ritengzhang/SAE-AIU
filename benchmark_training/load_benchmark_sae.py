"""
Helper functions for loading SAEs from benchmark training results.
"""

import sys
from pathlib import Path
from typing import Dict, Tuple
import json

# Add SAELens to path
SAELENS_PATH = Path(__file__).parent.parent / "SAELens-main"
if str(SAELENS_PATH) not in sys.path:
    sys.path.insert(0, str(SAELENS_PATH))

# Add benchmark_training to path for imports
BENCHMARK_TRAINING_PATH = Path(__file__).parent
if str(BENCHMARK_TRAINING_PATH) not in sys.path:
    sys.path.insert(0, str(BENCHMARK_TRAINING_PATH))

from sae_lens import SAE
from transformer_lens import HookedTransformer

try:
    from benchmark_config import LLM_CONFIGS
except ImportError:
    # Fallback if running from different directory
    import importlib.util
    config_path = Path(__file__).parent / "benchmark_config.py"
    spec = importlib.util.spec_from_file_location("benchmark_config", config_path)
    benchmark_config = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(benchmark_config)
    LLM_CONFIGS = benchmark_config.LLM_CONFIGS


def load_benchmark_sae(
    combo_id: str,
    llm_name: str = "gpt2",
    device: str = "cuda",
    base_dir: Path = None
) -> Tuple[SAE, Dict]:
    """
    Load a benchmark training SAE from disk.
    
    Args:
        combo_id: Combination identifier (e.g., "standard_L6_4k_l1_coefficient1.0")
        llm_name: LLM name (default: "gpt2")
        device: Device to load on (default: "cuda")
        base_dir: Base directory (default: {DATA_DIR}/sae/training, where DATA_DIR defaults to /data/user)
    
    Returns:
        Tuple of (SAE, metadata_dict)
    """
    if base_dir is None:
        # Use absolute path (same as training runners)
        data_dir = os.environ.get("DATA_DIR", "/data/user")
        base_dir = Path(data_dir) / "sae" / "training"
    
    sae_path = base_dir / llm_name / "checkpoints" / combo_id
    
    if not sae_path.exists():
        raise FileNotFoundError(f"SAE checkpoint directory not found: {sae_path}")
    
    # Check for required files
    weights_file = sae_path / "sae_weights.safetensors"
    cfg_file = sae_path / "cfg.json"
    
    if not weights_file.exists() or not cfg_file.exists():
        raise FileNotFoundError(
            f"SAE files not found at {sae_path}. "
            f"Expected sae_weights.safetensors and cfg.json"
        )
    
    # Load SAE using SAELens
    sae = SAE.load_from_disk(str(sae_path), device=device)
    
    # Load metadata from cfg.json
    with open(cfg_file, 'r') as f:
        cfg_dict = json.load(f)
    
    # Extract metadata - get hook_name from cfg.json or construct from LLM config
    hook_name = cfg_dict.get('hook_name', None)
    if hook_name is None or hook_name == 'unknown':
        # Construct hook_name from LLM config if available
        if llm_name in LLM_CONFIGS:
            llm_config = LLM_CONFIGS[llm_name]
            layer = cfg_dict.get('layer', llm_config.get('middle_layer', 6))
            hook_template = llm_config.get('hook_name_template', 'blocks.{layer}.hook_mlp_out')
            hook_name = hook_template.format(layer=layer)
        else:
            hook_name = 'blocks.6.hook_mlp_out'  # Default fallback
    
    # Extract metadata
    metadata = {
        'model_name': cfg_dict.get('model_name', llm_name),
        'hook_name': hook_name,
        'd_in': cfg_dict.get('d_in', sae.cfg.d_in),
        'd_sae': cfg_dict.get('d_sae', sae.cfg.d_sae),
        'architecture': cfg_dict.get('architecture', 'unknown'),
        'combo_id': combo_id,
        'llm_name': llm_name,
        'sae_path': str(sae_path),
    }
    
    # Add LLM config info if available
    if llm_name in LLM_CONFIGS:
        llm_config = LLM_CONFIGS[llm_name]
        # Also add layer info if not already present
        if 'layer' not in metadata:
            layer = cfg_dict.get('layer', llm_config.get('middle_layer', 6))
            metadata['layer'] = layer
        metadata.update({
            'model_class_name': llm_config['model_class_name'],
            'dataset_path': llm_config['dataset_path'],
            'is_dataset_tokenized': llm_config['is_dataset_tokenized'],
            'streaming': llm_config['streaming'],
        })
    
    return sae, metadata


def load_base_model_for_llm(
    llm_name: str = "gpt2",
    device: str = "cuda"
) -> HookedTransformer:
    """
    Load the base model for a given LLM.
    
    Args:
        llm_name: LLM name (default: "gpt2")
        device: Device to load on (default: "cuda")
    
    Returns:
        HookedTransformer model
    """
    if llm_name not in LLM_CONFIGS:
        raise ValueError(f"Unknown LLM: {llm_name}")
    
    llm_config = LLM_CONFIGS[llm_name]
    model_name = llm_config['model_name']
    
    model = HookedTransformer.from_pretrained(model_name, device=device)
    return model


def parse_combo_id(combo_id: str) -> Dict:
    """
    Parse a combo_id string to extract components.
    
    Format: {architecture}_L{layer}_{width}_{sparsity_param}{sparsity_value}
    Examples: 
        "standard_L6_4k_l1_coefficient1.0"
        "topk_L6_16k_k50"
        "jumprelu_L6_65k_l0_coefficient3.0"
        "matching_pursuit_L6_4k_residual_threshold0.01"
    
    Returns:
        Dict with keys: architecture, layer, width, sparsity_param, sparsity_value
    """
    import re
    
    # Find the layer pattern "L{digit}" - this is a reliable anchor
    layer_match = re.search(r'_L(\d+)_', combo_id)
    if not layer_match:
        raise ValueError(f"Cannot find layer pattern in combo_id: {combo_id}")
    
    layer = int(layer_match.group(1))
    layer_start = layer_match.start()
    layer_end = layer_match.end()
    
    # Everything before "_L{digit}_" is the architecture (may contain underscores)
    architecture = combo_id[:layer_start].rstrip('_')
    
    # Everything after "_L{digit}_" needs to be split
    after_layer = combo_id[layer_end:]
    parts = after_layer.split('_')
    
    if len(parts) < 2:
        raise ValueError(f"Cannot parse combo_id (not enough parts after layer): {combo_id}")
    
    # First part after layer is width
    width = parts[0]  # "4k", "16k", "65k"
    
    # Remaining parts form the sparsity parameter
    sparsity_parts = parts[1:]
    
    # Join all remaining parts and try to extract numeric value from the end
    combined = '_'.join(sparsity_parts)
    
    # Pattern to match: parameter name followed by numeric value at the end
    # Examples: "l1_coefficient1.0" -> param="l1_coefficient", value=1.0
    #           "k50" -> param="k", value=50
    #           "residual_threshold0.01" -> param="residual_threshold", value=0.01
    
    # Try to match a float or int at the end
    match = re.search(r'^(.+?)(\d+\.?\d*)$', combined)
    if match:
        sparsity_param = match.group(1).rstrip('_')
        sparsity_value = float(match.group(2))
    else:
        # Fallback: try to extract numeric from last part
        if len(sparsity_parts) > 0:
            last_part = sparsity_parts[-1]
            # Check if last part is purely numeric
            if re.match(r'^\d+\.?\d*$', last_part):
                sparsity_value = float(last_part)
                sparsity_param = '_'.join(sparsity_parts[:-1]) if len(sparsity_parts) > 1 else 'unknown'
            else:
                # Try to extract number embedded in last part
                num_match = re.search(r'(\d+\.?\d*)$', last_part)
                if num_match:
                    sparsity_value = float(num_match.group(1))
                    sparsity_param = last_part[:num_match.start()] if num_match.start() > 0 else '_'.join(sparsity_parts[:-1])
                else:
                    # No numeric value found
                    sparsity_param = combined
                    sparsity_value = None
        else:
            sparsity_param = combined
            sparsity_value = None
    
    return {
        'architecture': architecture,
        'layer': layer,
        'width': width,
        'sparsity_param': sparsity_param,
        'sparsity_value': sparsity_value,
        'combo_id': combo_id,
    }
