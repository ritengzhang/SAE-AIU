"""
Configuration file for SAE benchmark training.

Defines all parameters for the comprehensive benchmark:
- LLMs (3 models from different families)
- Architectures (7 types)
- Widths (3 sizes: 4k, 16k, 65k)
- Sparsity levels (3 per architecture)
- Layers (1 middle layer per LLM)
"""

import os
from dataclasses import dataclass
from typing import Any

# LLM Configuration
LLM_CONFIGS = {
    "gpt2": {
        "model_name": "gpt2",
        "model_class_name": "HookedTransformer",
        "d_in": 768,  # GPT-2 small MLP output dimension
        "middle_layer": 6,  # 12 layers total, layer 6 is middle
        "hook_name_template": "blocks.{layer}.hook_mlp_out",
        "dataset_path": "apollo-research/Skylion007-openwebtext-tokenizer-gpt2",
        "is_dataset_tokenized": True,
        "streaming": True,
    },
    "tiny-stories": {
        "model_name": "tiny-stories-1L-21M",
        "model_class_name": "HookedTransformer",
        "d_in": 1024,  # TinyStories MLP output dimension
        "middle_layer": 0,  # Only 1 layer, so layer 0
        "hook_name_template": "blocks.{layer}.hook_mlp_out",
        "dataset_path": "apollo-research/roneneldan-TinyStories-tokenizer-gpt2",
        "is_dataset_tokenized": True,
        "streaming": True,
    },
    "opt-125m": {
        "model_name": "facebook/opt-125m",
        "model_class_name": "HookedTransformer",
        "d_in": 768,  # OPT-125M MLP output dimension (assuming similar to GPT-2)
        "middle_layer": 6,  # 12 layers total, layer 6 is middle
        "hook_name_template": "blocks.{layer}.hook_mlp_out",
        "dataset_path": "apollo-research/Skylion007-openwebtext-tokenizer-gpt2",  # Use GPT-2 tokenizer compatible dataset
        "is_dataset_tokenized": True,
        "streaming": True,
    },
}

# Architecture configurations
ARCHITECTURES = {
    "standard": {
        "sparsity_param": "l1_coefficient",
        "sparsity_values": [1.0, 2.0, 5.0],  # 3 values
        "config_class": "StandardTrainingSAEConfig",
    },
    "gated": {
        "sparsity_param": "l1_coefficient",
        "sparsity_values": [1.0, 2.0, 5.0],  # 3 values
        "config_class": "GatedTrainingSAEConfig",
    },
    "jumprelu": {
        "sparsity_param": "l0_coefficient",
        "sparsity_values": [1.0, 3.0, 5.0],  # 3 values
        "config_class": "JumpReLUTrainingSAEConfig",
    },
    "topk": {
        "sparsity_param": "k",
        "sparsity_values": [50, 100, 200],  # 3 values
        "config_class": "TopKTrainingSAEConfig",
    },
    "batchtopk": {
        "sparsity_param": "k",
        "sparsity_values": [50, 100, 200],  # 3 values
        "config_class": "BatchTopKTrainingSAEConfig",
    },
    "matryoshka_batchtopk": {
        "sparsity_param": "k",
        "sparsity_values": [50, 100, 200],  # 3 values
        "config_class": "MatryoshkaBatchTopKTrainingSAEConfig",
        # Note: Also needs matryoshka_widths, will be set based on d_sae
    },
    # Note: matching_pursuit excluded (too slow, 10-100x slower than other architectures)
    # Note: Transcoder excluded for now as it lacks training support
    # matching_pursuit is NOT included in ARCHITECTURES - do not add it
}

# Width configurations (d_sae values)
WIDTHS = {
    "4k": 4096,
    "16k": 16384,
    "65k": 65536,
}

# Standard training parameters
TRAINING_CONFIG = {
    "total_training_steps": 18000,
    "train_batch_size_tokens": 4096,
    "lr": 5e-5,
    "lr_warm_up_steps": 0,
    "lr_decay_steps": 3600,  # 20% of 18000
    "lr_scheduler_name": "constant",
    "context_size": 512,
    "n_batches_in_buffer": 32,  # Reduced from 64 to save memory (matching old code)
    "store_batch_size_prompts": 8,  # Reduced from 16 to save memory (matching old code)
    "normalize_activations": "expected_average_only_in",
    "apply_b_dec_to_input": False,
    "adam_beta1": 0.9,
    "adam_beta2": 0.999,
    "feature_sampling_window": 1000,
    "dead_feature_window": 1000,
    "dead_feature_threshold": 1e-4,
    "n_eval_batches": 10,
    "eval_batch_size_prompts": 2,
    "n_checkpoints": 0,
    "checkpoint_path": None,
    "save_final_checkpoint": False,
    "dtype": "float32",
    "autocast": True,
    "compile_llm": True,
    "compile_sae": True,
}

# Output directory structure
# Can be overridden via DATA_DIR environment variable
OUTPUT_BASE_DIR = os.environ.get("DATA_DIR", "data/user/sae/training")

# WandB configuration
WANDB_CONFIG = {
    "log_to_wandb": True,
    "wandb_project": "sae_benchmark_training",
    "wandb_log_frequency": 50,
    "eval_every_n_wandb_logs": 10,
}
