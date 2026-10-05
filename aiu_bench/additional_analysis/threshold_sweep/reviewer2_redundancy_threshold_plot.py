#!/usr/bin/env python3
"""
Reviewer 2 W4: Redundancy threshold sweep plot.

Reads analysis8 results and produces two plots:
  1. Unfiltered: % SAEs where Q1 has most redundant pairs, per architecture, across thresholds
  2. Dead-neuron filtered: same but with dead-neuron pairs excluded

X-axis: cosine similarity threshold (0.1 - 0.9)
Y-axis: % of SAEs where Q1 has the most redundant pairs
Lines: one per architecture + overall (dashed black) + 25% chance line

Usage:
    python reviewer2_redundancy_threshold_plot.py \
        --results-dir <DATA_ROOT>/analysis8_redundancy_threshold_sweep \
        --output-dir <REPO_ROOT>/rebuttal/final_code/plots
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ARCHITECTURE_ORDER = [
    "Standard",
    "Gated",
    "TopK",
    "BatchTopK",
    "JumpReLU",
    "Matryoshka",
    "PAnneal",
]

ARCHITECTURE_COLORS = {
    "Standard": "#1f77b4",
    "Gated": "#ff7f0e",
    "TopK": "#2ca02c",
    "BatchTopK": "#d62728",
    "JumpReLU": "#9467bd",
    "Matryoshka": "#8c564b",
    "PAnneal": "#e377c2",
}


ARCH_NORMALIZE = {
    "standard": "Standard",
    "gated": "Gated",
    "topk": "TopK",
    "batchtopk": "BatchTopK",
    "jumprelu": "JumpReLU",
    "matryoshka": "Matryoshka",
    "matryoshka_batchtopk": "Matryoshka",
    "panneal": "PAnneal",
    "vanilla": "Standard",
    "Standard": "Standard",
    "Gated": "Gated",
    "TopK": "TopK",
    "BatchTopK": "BatchTopK",
    "JumpReLU": "JumpReLU",
    "Matryoshka": "Matryoshka",
    "MatryoshkaBatchTopK": "Matryoshka",
    "PAnneal": "PAnneal",
    "Vanilla": "Standard",
}


def load_all_results(results_dir: Path):
    """Load all analysis8 JSON results, returning list of (architecture, result_dict)."""
    entries = []
    for json_path in sorted(results_dir.rglob("results.json")):
        with open(json_path) as f:
            data = json.load(f)
        arch = data.get("architecture") or data.get("sae_type") or "unknown"
        # jbloom has no architecture field — all are Standard
        if arch == "unknown":
            arch = "Standard"
        arch = ARCH_NORMALIZE.get(arch, arch)
        entries.append((arch, data))
    return entries


def compute_q1_wins(entries, filtered=False):
    """
    For each architecture and threshold, compute % of SAEs where Q1 has the most
    redundant pairs (among SAEs with at least one pair).
    """
    key = "quantile_pairs_filtered" if filtered else "quantile_pairs"

    arch_results = defaultdict(lambda: defaultdict(lambda: {"wins": 0, "total": 0}))

    for arch, data in entries:
        pairs_data = data.get(key)
        if pairs_data is None:
            continue

        for t in data["thresholds"]:
            counts = {q: pairs_data[str(q)].get(t, 0) for q in [1, 2, 3, 4]}
            total_pairs = sum(counts.values())
            if total_pairs == 0:
                continue

            arch_results[arch][t]["total"] += 1
            q1_count = counts[1]
            others_max = max(counts[2], counts[3], counts[4])
            if q1_count > 0 and q1_count >= others_max:
                arch_results[arch][t]["wins"] += 1

    return arch_results


def compute_overall_q1_wins(entries, filtered=False):
    """Same as compute_q1_wins but pooled across all architectures."""
    key = "quantile_pairs_filtered" if filtered else "quantile_pairs"

    results = defaultdict(lambda: {"wins": 0, "total": 0})

    for _arch, data in entries:
        pairs_data = data.get(key)
        if pairs_data is None:
            continue

        for t in data["thresholds"]:
            counts = {q: pairs_data[str(q)].get(t, 0) for q in [1, 2, 3, 4]}
            total_pairs = sum(counts.values())
            if total_pairs == 0:
                continue

            results[t]["total"] += 1
            q1_count = counts[1]
            others_max = max(counts[2], counts[3], counts[4])
            if q1_count > 0 and q1_count >= others_max:
                results[t]["wins"] += 1

    return results


def _plot_single_panel(ax, entries, filtered=False):
    """Plot a single threshold sweep panel on the given axes."""
    arch_results = compute_q1_wins(entries, filtered=filtered)
    overall = compute_overall_q1_wins(entries, filtered=filtered)

    all_thresholds = sorted(overall.keys(), key=float)
    xs = [float(t) for t in all_thresholds]

    for arch in ARCHITECTURE_ORDER:
        if arch not in arch_results:
            continue
        ar = arch_results[arch]
        ys = []
        for t in all_thresholds:
            info = ar.get(t, {"wins": 0, "total": 0})
            if info["total"] > 0:
                ys.append(100.0 * info["wins"] / info["total"])
            else:
                ys.append(np.nan)
        ax.plot(
            xs, ys,
            marker="o", markersize=4,
            label=arch,
            color=ARCHITECTURE_COLORS.get(arch, "gray"),
            linewidth=1.5, alpha=0.8,
        )

    # Overall (thick black dashed)
    ys_overall = []
    for t in all_thresholds:
        info = overall[t]
        if info["total"] > 0:
            ys_overall.append(100.0 * info["wins"] / info["total"])
        else:
            ys_overall.append(np.nan)
    ax.plot(xs, ys_overall, "k--", linewidth=2.5, label="Overall", alpha=0.9)

    # 25% chance line
    ax.axhline(25, color="gray", linestyle=":", linewidth=1, alpha=0.5, label="Chance (25%)")

    ax.set_xlabel("Decoder Cosine Similarity Threshold", fontsize=12)
    ax.set_ylabel("% SAEs where Q1 has most redundant pairs", fontsize=11)
    suffix = " (dead neurons filtered)" if filtered else " (unfiltered)"
    ax.set_title(f"Q1 Redundancy Concentration{suffix}", fontsize=13)
    ax.set_xlim(0.05, 0.95)
    ax.set_ylim(0, 105)
    ax.set_xticks(xs)
    ax.grid(True, alpha=0.3)


def plot_sweep(entries, output_dir: Path, filtered=False):
    """Create a single threshold sweep plot."""
    fig, ax = plt.subplots(figsize=(8, 5))
    _plot_single_panel(ax, entries, filtered=filtered)
    ax.legend(loc="best", fontsize=9, ncol=2)
    plt.tight_layout()
    tag = "filtered" if filtered else "unfiltered"
    out_path = output_dir / f"redundancy_threshold_sweep_{tag}.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def plot_sweep_combined(entries, output_dir: Path):
    """Create a combined plot with unfiltered (left) and filtered (right)."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))

    _plot_single_panel(ax1, entries, filtered=False)
    ax1.set_title("(a) Unfiltered", fontsize=13)

    _plot_single_panel(ax2, entries, filtered=True)
    ax2.set_title("(b) Dead neurons filtered", fontsize=13)

    # Single shared legend below
    handles, labels = ax2.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=5, fontsize=10,
               bbox_to_anchor=(0.5, -0.06))

    fig.suptitle("Q1 Redundancy Concentration Across Thresholds (610 SAEs, 7 architectures)",
                 fontsize=14, y=1.02)
    plt.tight_layout(rect=[0, 0.06, 1, 0.98])
    out_path = output_dir / "redundancy_threshold_sweep_combined.png"
    plt.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir", type=str,
        default="<DATA_ROOT>/analysis8_redundancy_threshold_sweep",
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="<REPO_ROOT>/rebuttal/final_code/plots",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = load_all_results(results_dir)
    print(f"Loaded {len(entries)} SAE results")

    archs = defaultdict(int)
    for arch, _ in entries:
        archs[arch] += 1
    for arch, count in sorted(archs.items()):
        print(f"  {arch}: {count}")

    plot_sweep(entries, output_dir, filtered=False)
    has_filtered = any("quantile_pairs_filtered" in d for _, d in entries)
    if has_filtered:
        plot_sweep(entries, output_dir, filtered=True)
        plot_sweep_combined(entries, output_dir)
    else:
        print("No dead-neuron filtered data found, skipping filtered plot")


if __name__ == "__main__":
    main()
