"""
Utility functions for SAE benchmark training.
"""

import json
import sys
from pathlib import Path
from typing import Any

# Add SAELens to path if needed
SAELENS_PATH = Path(__file__).parent.parent / "SAELens-main"
if str(SAELENS_PATH) not in sys.path:
    sys.path.insert(0, str(SAELENS_PATH))

import json

from benchmark_config import LLM_CONFIGS, TRAINING_CONFIG, WANDB_CONFIG
from sae_lens import (
    BatchTopKTrainingSAEConfig,
    GatedTrainingSAEConfig,
    JumpReLUTrainingSAEConfig,
    LanguageModelSAERunnerConfig,
    LoggingConfig,
    MatchingPursuitTrainingSAEConfig,
    MatryoshkaBatchTopKTrainingSAEConfig,
    StandardTrainingSAEConfig,
    TopKTrainingSAEConfig,
)
from sae_lens.load_model import load_model
from transformer_lens import HookedTransformer


# Import OUTPUT_BASE_DIR lazily to allow override in runners
def get_output_base_dir():
    """Get OUTPUT_BASE_DIR, allowing for runtime overrides."""
    from benchmark_config import OUTPUT_BASE_DIR

    return OUTPUT_BASE_DIR


def get_sae_config_class(arch_name: str):
    """Get the SAE config class for a given architecture."""
    config_classes = {
        "standard": StandardTrainingSAEConfig,
        "gated": GatedTrainingSAEConfig,
        "jumprelu": JumpReLUTrainingSAEConfig,
        "topk": TopKTrainingSAEConfig,
        "batchtopk": BatchTopKTrainingSAEConfig,
        "matryoshka_batchtopk": MatryoshkaBatchTopKTrainingSAEConfig,
        # Note: matching_pursuit excluded (too slow, 10-100x slower)
    }
    return config_classes[arch_name]


def create_sae_config(combo: dict) -> Any:
    """Create an SAE config from a combination dictionary."""
    config_class = get_sae_config_class(combo["architecture"])

    # Base parameters
    config_kwargs = {
        "d_in": combo["d_in"],
        "d_sae": combo["d_sae"],
        "normalize_activations": TRAINING_CONFIG["normalize_activations"],
        "apply_b_dec_to_input": TRAINING_CONFIG["apply_b_dec_to_input"],
    }

    # Architecture-specific sparsity parameter
    sparsity_param = combo["sparsity_param"]
    config_kwargs[sparsity_param] = combo["sparsity_value"]

    # Special handling for different architectures
    if combo["architecture"] == "standard" or combo["architecture"] == "gated":
        config_kwargs["l1_warm_up_steps"] = (
            TRAINING_CONFIG["total_training_steps"] // 20
        )

    elif combo["architecture"] == "jumprelu":
        config_kwargs["l0_warm_up_steps"] = (
            TRAINING_CONFIG["total_training_steps"] // 20
        )
        config_kwargs["jumprelu_sparsity_loss_mode"] = "tanh"
        config_kwargs["jumprelu_tanh_scale"] = 4.0
        config_kwargs["jumprelu_bandwidth"] = 2.0
        config_kwargs["jumprelu_init_threshold"] = 0.1
        config_kwargs["pre_act_loss_coefficient"] = 3e-6

    elif combo["architecture"] == "matryoshka_batchtopk":
        config_kwargs["matryoshka_widths"] = combo["matryoshka_widths"]

    return config_class(**config_kwargs)


def create_training_config(
    combo: dict, device: str = "cuda"
) -> LanguageModelSAERunnerConfig:
    """Create a complete LanguageModelSAERunnerConfig from a combination."""
    sae_config = create_sae_config(combo)

    # Create run name for WandB
    run_name = (
        f"{combo['llm']}_{combo['architecture']}_"
        f"L{combo['layer']}_{combo['width']}_"
        f"{combo['sparsity_param']}{combo['sparsity_value']}"
    )

    # Output path for this run
    output_base_dir = get_output_base_dir()
    output_dir = (
        Path(output_base_dir)
        / combo["llm"]
        / "checkpoints"
        / (
            f"{combo['architecture']}_L{combo['layer']}_{combo['width']}_{combo['sparsity_param']}{combo['sparsity_value']}"
        )
    )

    config = LanguageModelSAERunnerConfig(
        # SAE config
        sae=sae_config,
        # Model config
        model_name=combo["model_name"],
        model_class_name=combo["model_class_name"],
        hook_name=combo["hook_name"],
        # Dataset config
        dataset_path=combo["dataset_path"],
        is_dataset_tokenized=combo["is_dataset_tokenized"],
        streaming=combo["streaming"],
        # Training parameters
        train_batch_size_tokens=TRAINING_CONFIG["train_batch_size_tokens"],
        lr=TRAINING_CONFIG["lr"],
        lr_warm_up_steps=TRAINING_CONFIG["lr_warm_up_steps"],
        lr_decay_steps=TRAINING_CONFIG["lr_decay_steps"],
        lr_scheduler_name=TRAINING_CONFIG["lr_scheduler_name"],
        adam_beta1=TRAINING_CONFIG["adam_beta1"],
        adam_beta2=TRAINING_CONFIG["adam_beta2"],
        # Activation store parameters (streaming, no caching)
        context_size=TRAINING_CONFIG["context_size"],
        n_batches_in_buffer=TRAINING_CONFIG["n_batches_in_buffer"],
        store_batch_size_prompts=TRAINING_CONFIG["store_batch_size_prompts"],
        training_tokens=TRAINING_CONFIG["total_training_steps"]
        * TRAINING_CONFIG["train_batch_size_tokens"],
        use_cached_activations=False,  # Stream activations, don't cache
        cached_activations_path=None,  # No caching
        # Feature resampling
        feature_sampling_window=TRAINING_CONFIG["feature_sampling_window"],
        dead_feature_window=TRAINING_CONFIG["dead_feature_window"],
        dead_feature_threshold=TRAINING_CONFIG["dead_feature_threshold"],
        # Evaluation
        n_eval_batches=TRAINING_CONFIG["n_eval_batches"],
        eval_batch_size_prompts=TRAINING_CONFIG["eval_batch_size_prompts"],
        # Checkpointing
        n_checkpoints=TRAINING_CONFIG["n_checkpoints"],
        checkpoint_path=str(output_dir) if TRAINING_CONFIG["checkpoint_path"] else None,
        save_final_checkpoint=TRAINING_CONFIG["save_final_checkpoint"],
        output_path=str(output_dir),
        # Logging - use separate WandB project for each LLM
        logger=LoggingConfig(
            log_to_wandb=WANDB_CONFIG["log_to_wandb"],
            wandb_project=f"sae_benchmark_{combo['llm']}",
            wandb_log_frequency=WANDB_CONFIG["wandb_log_frequency"],
            eval_every_n_wandb_logs=WANDB_CONFIG["eval_every_n_wandb_logs"],
            run_name=run_name,
        ),
        # Device and compilation
        device=device,
        dtype=TRAINING_CONFIG["dtype"],
        autocast=TRAINING_CONFIG["autocast"],
        compile_llm=TRAINING_CONFIG["compile_llm"],
        compile_sae=TRAINING_CONFIG["compile_sae"],
        # Misc
        seed=42,
        prepend_bos=True,
    )

    return config


def get_cached_activations_path(llm_name: str, layer: int) -> Path:
    """Get the path for cached activations for a given LLM and layer."""
    output_base_dir = get_output_base_dir()
    return Path(output_base_dir) / llm_name / "activations" / f"layer_{layer}"


def check_if_cached(llm_name: str, layer: int) -> bool:
    """Check if activations are already cached for a given LLM and layer."""
    cache_path = get_cached_activations_path(llm_name, layer)
    # Check if the directory exists and has actual dataset files (not just .tmp_shards)
    if not cache_path.exists():
        return False
    # Check for actual dataset files (arrow, parquet, or dataset_info.json)
    dataset_files = list(cache_path.glob("*.arrow")) + list(
        cache_path.glob("*.parquet")
    )
    has_info = (cache_path / "dataset_info.json").exists()
    # If only .tmp_shards exists, caching was interrupted - not valid
    tmp_only = (
        len(list(cache_path.iterdir())) == 1 and (cache_path / ".tmp_shards").exists()
    )
    return (len(dataset_files) > 0 or has_info) and not tmp_only


def load_combinations() -> list[dict[str, Any]]:
    """Load all combinations from JSON file."""
    output_base_dir = get_output_base_dir()
    combinations_path = Path(output_base_dir) / "combinations.json"
    # If relative path doesn't exist, try absolute path
    if not combinations_path.exists() and not combinations_path.is_absolute():
        data_dir = os.environ.get("DATA_DIR", "/data/user")
        combinations_path = Path(data_dir) / "sae" / "training" / "combinations.json"
    if not combinations_path.exists():
        raise FileNotFoundError(
            f"Combinations file not found: {combinations_path}. "
            "Run generate_combinations.py first."
        )

    with open(combinations_path, "r") as f:
        return json.load(f)


def filter_combinations_for_llm(
    combinations: list[dict[str, Any]], llm_name: str
) -> list[dict[str, Any]]:
    """Filter combinations for a specific LLM."""
    return [c for c in combinations if c["llm"] == llm_name]


def save_results(llm_name: str, results: dict[str, Any]) -> None:
    """Save results to JSON file."""
    output_base_dir = get_output_base_dir()
    results_path = Path(output_base_dir) / llm_name / "results.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)

    with open(results_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Saved results to {results_path}")


def is_combo_complete(llm_name: str, combo_id: str) -> bool:
    """
    Check if a combination has already been completed.

    Checks for:
    1. Checkpoint directory with required files (sae_weights.safetensors, cfg.json)
    2. Results.json file listing this combo as completed

    Args:
        llm_name: Name of the LLM (e.g., 'gpt2', 'opt-125m')
        combo_id: Combination identifier string

    Returns:
        True if combination is already complete, False otherwise
    """
    # Get OUTPUT_BASE_DIR dynamically to allow runtime overrides
    output_base_dir = get_output_base_dir()

    # Check for checkpoint directory with required files
    output_dir = Path(output_base_dir) / llm_name / "checkpoints" / combo_id
    if output_dir.exists():
        # Check for required checkpoint files (SAELens saves directly in checkpoint dir)
        weights_file = output_dir / "sae_weights.safetensors"
        cfg_file = output_dir / "cfg.json"
        if weights_file.exists() and cfg_file.exists():
            return True

        # Also check for final checkpoint directory (if SAELens uses that format)
        final_checkpoints = list(output_dir.glob("final_*"))
        if final_checkpoints:
            return True

    # Also check results.json if it exists
    results_path = Path(output_base_dir) / llm_name / "results.json"
    if results_path.exists():
        try:
            with open(results_path, "r") as f:
                results = json.load(f)
                if combo_id in results.get("completed", []):
                    return True
        except (json.JSONDecodeError, IOError):
            # If results.json is corrupted or unreadable, don't skip
            pass

    return False


def print_summary(results: dict[str, Any]) -> None:
    """Print summary of results."""
    print(f"\n{'=' * 60}")
    print("Results Summary")
    print(f"{'=' * 60}")
    print(f"Completed: {len(results['completed'])}")
    print(f"Failed: {len(results['failed'])}")
    print(f"Skipped: {len(results['skipped'])}")
    print(f"{'=' * 60}\n")

    if results["failed"]:
        print("Failed combinations:")
        for failure in results["failed"]:
            print(f"  - {failure['combo']}: {failure['error']}")
