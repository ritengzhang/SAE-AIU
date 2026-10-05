"""
Plot results from toy validation experiment.
Only generates the Q1 deficit summary — the core plot for the rebuttal.
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ARCH_COLORS = {
    "standard": "tab:blue",
    "topk": "tab:orange",
    "gated": "tab:green",
    "jumprelu": "tab:red",
    "batchtopk": "tab:purple",
    "matryoshka": "tab:brown",
    "panneal": "tab:cyan",
}
ARCH_ORDER = [
    "standard",
    "topk",
    "gated",
    "jumprelu",
    "batchtopk",
    "matryoshka",
    "panneal",
]
ARCH_LABELS = {
    "standard": "Standard",
    "topk": "TopK",
    "gated": "Gated",
    "jumprelu": "JumpReLU",
    "batchtopk": "BatchTopK",
    "matryoshka": "Matryoshka",
    "panneal": "PAnneal",
}


def load_results(
    results_dir: str = "<DATA_ROOT>/toy_validation/results",
) -> dict:
    results_dir = Path(results_dir)
    with open(results_dir / "all_results.json") as f:
        return json.load(f)


def group_by_arch(results: dict) -> dict:
    grouped = {}
    for name, res in results.items():
        arch = res["config"]["arch"]
        if arch not in grouped:
            grouped[arch] = {}
        grouped[arch][name] = res
    return grouped


def plot_q1_deficit_summary(results: dict, output_dir: str):
    """Bar chart: Q1 deficit across SAEs, grouped by architecture.
    No y-axis labels. X-axis in percentage."""
    output_dir = Path(output_dir)
    grouped = group_by_arch(results)

    archs = [a for a in ARCH_ORDER if a in grouped]
    n_archs = len(archs)
    n_cols = min(4, n_archs)
    n_rows = (n_archs + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows), sharey=False
    )
    if n_rows == 1 and n_cols == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    for i, arch in enumerate(archs):
        ax = axes[i]
        entries = []
        for name, res in sorted(grouped[arch].items()):
            dm = res.get("dominance_monosemanticity", {})
            if "error" in dm:
                continue
            entries.append(
                {
                    "deficit": dm["q1_deficit"],
                    "p_value": dm["p_value"],
                }
            )

        entries.sort(key=lambda e: e["deficit"])
        deficits_pct = [e["deficit"] * 100 for e in entries]
        pvals = [e["p_value"] for e in entries]

        n_sig = sum(1 for e in entries if e["deficit"] < 0 and e["p_value"] < 0.05)

        bars = ax.barh(
            range(len(deficits_pct)), deficits_pct, color=ARCH_COLORS[arch], alpha=0.7
        )
        for j, (d, p) in enumerate(zip(deficits_pct, pvals)):
            if d < 0 and p < 0.05:
                ax.text(d - 0.3, j, "*", va="center", fontsize=11, fontweight="bold")

        ax.set_yticks([])
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_xlabel("Q1 Deficit (%)")
        ax.set_title(f"{ARCH_LABELS[arch]} ({n_sig}/{len(entries)} sig.)")
        ax.grid(True, alpha=0.3, axis="x")

    for i in range(len(archs), len(axes)):
        axes[i].set_visible(False)

    plt.suptitle("Ground-Truth Q1 Monosemanticity Deficit: mean(Q1) - mean(Q2,Q3,Q4)\n(* = p<0.05, Mann-Whitney U test)", fontsize=12)
    plt.tight_layout()
    plt.savefig(output_dir / "q1_deficit_summary.png", dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {output_dir / 'q1_deficit_summary.png'}")


def print_summary_table(results: dict):
    print("\n" + "=" * 80)
    print("SUMMARY FOR REBUTTAL")
    print("=" * 80)

    grouped = group_by_arch(results)

    print(f"\nTotal SAEs evaluated: {len(results)}")
    for arch in ARCH_ORDER:
        if arch in grouped:
            print(f"  {ARCH_LABELS[arch]}: {len(grouped[arch])}")

    n_total = 0
    n_significant = 0
    q1_deficits = []
    for name, res in results.items():
        dm = res.get("dominance_monosemanticity", {})
        if "error" in dm:
            continue
        n_total += 1
        if dm["p_value"] < 0.05 and dm["q1_deficit"] < 0:
            n_significant += 1
        q1_deficits.append(dm["q1_deficit"])

    print(f"\nQ1 Monosemanticity Deficit:")
    print(f"  {n_significant}/{n_total} SAEs show significant Q1 deficit (p<0.05)")
    print(f"  Mean Q1 deficit: {np.mean(q1_deficits) * 100:.2f}%")

    for arch in ARCH_ORDER:
        if arch not in grouped:
            continue
        n_sig = 0
        n_tot = 0
        for res in grouped[arch].values():
            dm = res.get("dominance_monosemanticity", {})
            if "error" in dm:
                continue
            n_tot += 1
            if dm["p_value"] < 0.05 and dm["q1_deficit"] < 0:
                n_sig += 1
        print(f"  {ARCH_LABELS[arch]}: {n_sig}/{n_tot} significant")


def main():
    results_dir = "<DATA_ROOT>/toy_validation/results"
    output_dir = "<REPO_ROOT>/new/toy_validation/final_plots_for_response"
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    results = load_results(results_dir)
    print(f"Loaded {len(results)} SAE results")

    plot_q1_deficit_summary(results, output_dir)
    print_summary_table(results)


if __name__ == "__main__":
    main()
