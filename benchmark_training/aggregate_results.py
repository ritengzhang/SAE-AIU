"""
Aggregate and analyze results from SAE benchmark training.

Parses WandB runs or local checkpoints to create comparison tables.
"""

import json
import sys
from pathlib import Path
from typing import Any

# Add SAELens to path
SAELENS_PATH = Path(__file__).parent.parent / "SAELens-main"
if str(SAELENS_PATH) not in sys.path:
    sys.path.insert(0, str(SAELENS_PATH))

import pandas as pd
import wandb

from benchmark_config import LLM_CONFIGS, OUTPUT_BASE_DIR


def load_local_results() -> dict[str, Any]:
    """Load results from local JSON files."""
    results = {}
    
    for llm_name in LLM_CONFIGS.keys():
        results_path = Path(OUTPUT_BASE_DIR) / llm_name / "results.json"
        if results_path.exists():
            with open(results_path, "r") as f:
                results[llm_name] = json.load(f)
    
    return results


def fetch_wandb_runs(project_name: str = "sae_benchmark_training") -> list[dict[str, Any]]:
    """Fetch runs from WandB."""
    api = wandb.Api()
    runs = api.runs(project_name)
    
    run_data = []
    for run in runs:
        try:
            run_data.append({
                "name": run.name,
                "config": run.config,
                "summary": run.summary,
                "state": run.state,
                "url": run.url,
            })
        except Exception as e:
            print(f"Error fetching run {run.name}: {e}")
            continue
    
    return run_data


def parse_run_name(run_name: str) -> dict[str, str] | None:
    """
    Parse run name format: {llm}_{arch}_L{layer}_{width}_{sparsity_param}{sparsity_value}
    
    Example: gpt2_standard_L6_16k_l1_coefficient2.0
    """
    try:
        parts = run_name.split("_")
        if len(parts) < 5:
            return None
        
        llm = parts[0]
        arch = parts[1]
        layer = parts[2].replace("L", "")
        
        # Find width (4k, 16k, 65k)
        width = None
        for part in parts:
            if part in ["4k", "16k", "65k"]:
                width = part
                break
        
        if not width:
            return None
        
        # Find sparsity param and value
        sparsity_parts = "_".join(parts[parts.index(width) + 1:])
        # Try to extract sparsity param and value
        # This is a bit heuristic, but should work for most cases
        if "l1_coefficient" in sparsity_parts:
            sparsity_param = "l1_coefficient"
            sparsity_value = float(sparsity_parts.replace("l1_coefficient", ""))
        elif "l0_coefficient" in sparsity_parts:
            sparsity_param = "l0_coefficient"
            sparsity_value = float(sparsity_parts.replace("l0_coefficient", ""))
        elif sparsity_parts.startswith("k"):
            sparsity_param = "k"
            sparsity_value = int(sparsity_parts.replace("k", ""))
        elif sparsity_parts.startswith("residual_threshold"):
            sparsity_param = "residual_threshold"
            sparsity_value = float(sparsity_parts.replace("residual_threshold", ""))
        else:
            return None
        
        return {
            "llm": llm,
            "architecture": arch,
            "layer": layer,
            "width": width,
            "sparsity_param": sparsity_param,
            "sparsity_value": sparsity_value,
        }
    except Exception:
        return None


def create_results_dataframe(wandb_runs: list[dict[str, Any]]) -> pd.DataFrame:
    """Create a pandas DataFrame from WandB runs."""
    rows = []
    
    for run in wandb_runs:
        parsed = parse_run_name(run["name"])
        if not parsed:
            continue
        
        summary = run.get("summary", {})
        config = run.get("config", {})
        
        row = {
            **parsed,
            "state": run.get("state", "unknown"),
            "url": run.get("url", ""),
            # Extract key metrics from summary
            "final_sparsity": summary.get("metrics/sparsity/mean", None),
            "final_mse": summary.get("metrics/mse_loss/mean", None),
            "final_l0": summary.get("metrics/l0/mean", None),
            "dead_features": summary.get("metrics/dead_feature_count/mean", None),
            "explained_variance": summary.get("metrics/explained_variance/mean", None),
        }
        
        rows.append(row)
    
    return pd.DataFrame(rows)


def create_comparison_tables(df: pd.DataFrame, output_dir: Path) -> None:
    """Create comparison tables grouped by different dimensions."""
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Table 1: By architecture
    arch_table = df.groupby(["architecture", "width", "sparsity_value"]).agg({
        "final_sparsity": "mean",
        "final_mse": "mean",
        "final_l0": "mean",
        "dead_features": "mean",
        "explained_variance": "mean",
    }).round(4)
    
    arch_table.to_csv(output_dir / "comparison_by_architecture.csv")
    print(f"✓ Saved architecture comparison to {output_dir / 'comparison_by_architecture.csv'}")
    
    # Table 2: By LLM
    llm_table = df.groupby(["llm", "architecture"]).agg({
        "final_sparsity": "mean",
        "final_mse": "mean",
        "final_l0": "mean",
        "dead_features": "mean",
        "explained_variance": "mean",
    }).round(4)
    
    llm_table.to_csv(output_dir / "comparison_by_llm.csv")
    print(f"✓ Saved LLM comparison to {output_dir / 'comparison_by_llm.csv'}")
    
    # Table 3: By width
    width_table = df.groupby(["width", "architecture"]).agg({
        "final_sparsity": "mean",
        "final_mse": "mean",
        "final_l0": "mean",
        "dead_features": "mean",
        "explained_variance": "mean",
    }).round(4)
    
    width_table.to_csv(output_dir / "comparison_by_width.csv")
    print(f"✓ Saved width comparison to {output_dir / 'comparison_by_width.csv'}")


def print_summary_stats(df: pd.DataFrame) -> None:
    """Print summary statistics."""
    print("\n" + "="*60)
    print("Results Summary")
    print("="*60)
    
    print(f"\nTotal runs: {len(df)}")
    print(f"Completed runs: {len(df[df['state'] == 'finished'])}")
    print(f"Failed runs: {len(df[df['state'] == 'failed'])}")
    
    if len(df[df['state'] == 'finished']) > 0:
        completed = df[df['state'] == 'finished']
        print(f"\nAverage Metrics (completed runs only):")
        print(f"  Sparsity: {completed['final_sparsity'].mean():.4f}")
        print(f"  MSE: {completed['final_mse'].mean():.6f}")
        print(f"  L0: {completed['final_l0'].mean():.2f}")
        print(f"  Dead Features: {completed['dead_features'].mean():.1f}")
        print(f"  Explained Variance: {completed['explained_variance'].mean():.4f}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Aggregate SAE benchmark results")
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="sae_benchmark_training",
        help="WandB project name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(Path(OUTPUT_BASE_DIR) / "summaries"),
        help="Output directory for summary tables",
    )
    
    args = parser.parse_args()
    
    print("Fetching runs from WandB...")
    wandb_runs = fetch_wandb_runs(project_name=args.wandb_project)
    print(f"Found {len(wandb_runs)} runs")
    
    if len(wandb_runs) == 0:
        print("No runs found. Make sure WandB project name is correct.")
        sys.exit(1)
    
    # Create DataFrame
    df = create_results_dataframe(wandb_runs)
    print(f"Parsed {len(df)} runs")
    
    # Print summary
    print_summary_stats(df)
    
    # Create comparison tables
    output_dir = Path(args.output_dir)
    create_comparison_tables(df, output_dir)
    
    # Save full DataFrame
    df.to_csv(output_dir / "all_results.csv", index=False)
    print(f"\n✓ Saved full results to {output_dir / 'all_results.csv'}")

