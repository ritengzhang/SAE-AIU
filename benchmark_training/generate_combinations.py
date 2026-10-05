"""
Generate all parameter combinations for the SAE benchmark.

Total combinations per LLM: 7 architectures × 1 layer × 3 widths × 3 sparsity = 63
Total for 3 LLMs: 63 × 3 = 189 combinations
"""

import json
from pathlib import Path
from typing import Any

from benchmark_config import (
    ARCHITECTURES,
    LLM_CONFIGS,
    OUTPUT_BASE_DIR,
    WIDTHS,
)


def generate_all_combinations() -> list[dict[str, Any]]:
    """
    Generate all parameter combinations for the benchmark.
    
    Returns:
        List of dictionaries, each containing a complete configuration for one training run.
    """
    combinations = []
    
    for llm_name, llm_config in LLM_CONFIGS.items():
        layer = llm_config["middle_layer"]
        hook_name = llm_config["hook_name_template"].format(layer=layer)
        d_in = llm_config["d_in"]
        
        for arch_name, arch_config in ARCHITECTURES.items():
            for width_name, d_sae in WIDTHS.items():
                for sparsity_value in arch_config["sparsity_values"]:
                    # Create combination dictionary
                    combo = {
                        "llm": llm_name,
                        "architecture": arch_name,
                        "layer": layer,
                        "width": width_name,
                        "d_sae": d_sae,
                        "sparsity_value": sparsity_value,
                        "sparsity_param": arch_config["sparsity_param"],
                        "config_class": arch_config["config_class"],
                        "hook_name": hook_name,
                        "d_in": d_in,
                        # LLM-specific config
                        "model_name": llm_config["model_name"],
                        "model_class_name": llm_config["model_class_name"],
                        "dataset_path": llm_config["dataset_path"],
                        "is_dataset_tokenized": llm_config["is_dataset_tokenized"],
                        "streaming": llm_config["streaming"],
                    }
                    
                    # Special handling for MatryoshkaBatchTopK
                    if arch_name == "matryoshka_batchtopk":
                        # Set matryoshka_widths based on d_sae
                        if d_sae == 4096:
                            combo["matryoshka_widths"] = [1024, 2048, 4096]
                        elif d_sae == 16384:
                            combo["matryoshka_widths"] = [4096, 8192, 16384]
                        else:  # 65536
                            combo["matryoshka_widths"] = [16384, 32768, 65536]
                    
                    combinations.append(combo)
    
    return combinations


def save_combinations(combinations: list[dict[str, Any]], output_path: Path) -> None:
    """Save combinations to JSON file."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        json.dump(combinations, f, indent=2)


def print_summary(combinations: list[dict[str, Any]]) -> None:
    """Print summary statistics."""
    print(f"\n{'='*60}")
    print("SAE Benchmark Combinations Summary")
    print(f"{'='*60}")
    
    # Count by LLM
    llm_counts = {}
    for combo in combinations:
        llm = combo["llm"]
        llm_counts[llm] = llm_counts.get(llm, 0) + 1
    
    print(f"\nTotal combinations: {len(combinations)}")
    print(f"\nPer LLM:")
    for llm, count in llm_counts.items():
        print(f"  {llm}: {count} combinations")
    
    # Count by architecture
    arch_counts = {}
    for combo in combinations:
        arch = combo["architecture"]
        arch_counts[arch] = arch_counts.get(arch, 0) + 1
    
    print(f"\nPer Architecture:")
    for arch, count in arch_counts.items():
        print(f"  {arch}: {count} combinations")
    
    # Count by width
    width_counts = {}
    for combo in combinations:
        width = combo["width"]
        width_counts[width] = width_counts.get(width, 0) + 1
    
    print(f"\nPer Width:")
    for width, count in width_counts.items():
        print(f"  {width}: {count} combinations")
    
    print(f"\n{'='*60}\n")


if __name__ == "__main__":
    # Generate all combinations
    combinations = generate_all_combinations()
    
    # Print summary
    print_summary(combinations)
    
    # Save to file
    output_path = Path(OUTPUT_BASE_DIR) / "combinations.json"
    save_combinations(combinations, output_path)
    print(f"Saved {len(combinations)} combinations to {output_path}")

