#!/usr/bin/env python3
"""
Utility functions for cross-architecture aggregation analysis.

Provides helper functions to:
1. Load sparsity metrics from analysis 0
2. Extract architecture type from folder names or configs
3. Match analysis results across different analyses
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from collections import defaultdict


def extract_architecture_from_folder_name(sae_type_folder: str) -> str:
    """
    Extract architecture type from SAE type folder name.
    
    Examples:
    - "Standard_pythia-160m-deduped__0108" -> "ReLU"
    - "TopK_pythia-160m-deduped__0108" -> "TopK"
    - "BatchTopK_pythia-160m-deduped__0108" -> "BatchTopK"
    - "JumpRelu_pythia-160m-deduped__0108" -> "JumpReLU"
    - "GatedSAE_pythia-160m-deduped__0108" -> "GatedSAE"
    - "PAnneal_pythia-160m-deduped__0108" -> "PAnneal"
    - "MatryoshkaBatchTopK_pythia-160m-deduped__0108" -> "MatryoshkaBatchTopK"
    
    Args:
        sae_type_folder: SAE type folder name
        
    Returns:
        Architecture type string
    """
    # Remove model-specific suffixes
    base_name = sae_type_folder.split('_pythia')[0].split('_gemma')[0]
    
    # Map to architecture names
    architecture_map = {
        'Standard': 'ReLU',
        'TopK': 'TopK',
        'BatchTopK': 'BatchTopK',
        'JumpRelu': 'JumpReLU',
        'GatedSAE': 'GatedSAE',
        'PAnneal': 'PAnneal',
        'MatryoshkaBatchTopK': 'MatryoshkaBatchTopK',
    }
    
    return architecture_map.get(base_name, base_name)


def extract_architecture_from_config(config_path: Path) -> Optional[str]:
    """
    Extract architecture type from config.json file.
    
    Args:
        config_path: Path to config.json
        
    Returns:
        Architecture type string or None if not found
    """
    try:
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Check dict_class field
        dict_class = config.get('dict_class', '')
        if 'TopK' in dict_class:
            if 'Matryoshka' in dict_class:
                return 'MatryoshkaBatchTopK'
            elif 'Batch' in dict_class:
                return 'BatchTopK'
            else:
                return 'TopK'
        elif 'JumpReLU' in dict_class or 'JumpRelu' in dict_class:
            return 'JumpReLU'
        elif 'Gated' in dict_class:
            return 'GatedSAE'
        elif 'PAnneal' in dict_class or 'P_anneal' in dict_class:
            return 'PAnneal'
        elif 'Standard' in dict_class or 'ReLU' in dict_class:
            return 'ReLU'
            
    except Exception as e:
        pass
    
    return None


def get_architecture_type(sae_type_folder: str, config_path: Optional[Path] = None) -> str:
    """
    Get architecture type from folder name or config.json.
    
    Args:
        sae_type_folder: SAE type folder name
        config_path: Optional path to config.json for fallback
        
    Returns:
        Architecture type string
    """
    # Try folder name first
    arch = extract_architecture_from_folder_name(sae_type_folder)
    
    # If that didn't work and we have config, try config
    if arch == sae_type_folder.split('_')[0] and config_path and config_path.exists():
        arch_from_config = extract_architecture_from_config(config_path)
        if arch_from_config:
            return arch_from_config
    
    return arch


def extract_base_trainer_name(trainer_folder: str) -> str:
    """
    Extract base trainer name from trainer folder.
    
    Examples:
    - "trainer_0_k20" -> "trainer_0"
    - "trainer_3" -> "trainer_3"
    - "trainer_1_l1_penalty_0_012" -> "trainer_1"
    """
    # Match pattern: trainer_X (where X is a number)
    import re
    match = re.match(r'(trainer_\d+)', trainer_folder)
    if match:
        return match.group(1)
    return trainer_folder


def load_sparsity_metrics(
    model_family: str,
    sparsity_base_dir: Path = Path("/data/user/sae/analysis0_sparsity")
) -> Dict[str, Dict[str, Dict[str, Dict]]]:
    """
    Load all sparsity metrics from analysis 0.
    
    Returns:
        {
            'width': {
                'sae_type': {
                    'trainer': {
                        'average_active_features': float,
                        'effective_active_features_75': float,
                        'effective_active_features_90': float,
                        'd_sae': int
                    }
                }
            }
        }
    """
    results = defaultdict(lambda: defaultdict(dict))
    
    base_dir = sparsity_base_dir / model_family
    if not base_dir.exists():
        print(f"  ⚠️  Sparsity base directory not found: {base_dir}")
        return results
    
    results_files = list(base_dir.rglob("results.json"))
    print(f"Found {len(results_files)} sparsity results.json files")
    
    for results_file in results_files:
        # Parse path: {model_family}/{width}/{sae_type}/{trainer}/results.json
        parts = results_file.parts
        try:
            width_idx = parts.index(model_family) + 1
            width = parts[width_idx]
            sae_type = parts[width_idx + 1]
            trainer = parts[width_idx + 2]
        except (ValueError, IndexError):
            continue
        
        # Load results.json
        try:
            with open(results_file, 'r') as f:
                data = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            continue
        
        # Extract sparsity metrics
        sparsity_metrics = data.get('sparsity_metrics', {})
        if not sparsity_metrics:
            continue
        
        results[width][sae_type][trainer] = {
            'average_active_features': sparsity_metrics.get('average_active_features'),
            'effective_active_features_75': sparsity_metrics.get('effective_active_features_75'),
            'effective_active_features_90': sparsity_metrics.get('effective_active_features_90'),
            'd_sae': data.get('config', {}).get('d_sae'),
        }
    
    return results


def match_analysis_with_sparsity(
    analysis_results_path: Path,
    sparsity_metrics: Dict[str, Dict[str, Dict[str, Dict]]],
    model_family: str
) -> Optional[Dict]:
    """
    Match an analysis result file or directory with its corresponding sparsity metrics.
    
    Args:
        analysis_results_path: Path to analysis results.json or directory containing it
        sparsity_metrics: Sparsity metrics dict from load_sparsity_metrics
        model_family: Model family name (e.g., 'saebench_pythia')
        
    Returns:
        Dictionary with matched data or None if not found
    """
    # Handle both file and directory paths
    if analysis_results_path.is_dir():
        # Look for results.json in the directory
        results_file = analysis_results_path / "results.json"
        if not results_file.exists():
            return None
    else:
        results_file = analysis_results_path
    
    # Parse path: {model_family}/{width}/{sae_type}/{trainer}/results.json
    parts = results_file.parts
    try:
        width_idx = parts.index(model_family) + 1
        width = parts[width_idx]
        sae_type = parts[width_idx + 1]
        trainer = parts[width_idx + 2]
    except (ValueError, IndexError):
        return None
    
    # Extract base trainer name (e.g., "trainer_0" from "trainer_0_k20")
    base_trainer = extract_base_trainer_name(trainer)
    
    # Try exact match first
    sparsity_data = sparsity_metrics.get(width, {}).get(sae_type, {}).get(trainer)
    
    # If not found, try base trainer name
    if not sparsity_data:
        sparsity_data = sparsity_metrics.get(width, {}).get(sae_type, {}).get(base_trainer)
    
    # If still not found, search for any key that starts with base_trainer
    # (handles cases like trainer_0_k_20 vs trainer_0_k20)
    if not sparsity_data:
        sae_type_dict = sparsity_metrics.get(width, {}).get(sae_type, {})
        for key in sae_type_dict.keys():
            if key.startswith(base_trainer + '_') or key == base_trainer:
                sparsity_data = sae_type_dict[key]
                break
    
    if not sparsity_data:
        return None
    
    # Load analysis results
    try:
        with open(results_file, 'r') as f:
            analysis_data = json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return None
    
    return {
        'width': width,
        'sae_type': sae_type,
        'trainer': trainer,
        'sparsity': sparsity_data,
        'analysis': analysis_data,
    }

