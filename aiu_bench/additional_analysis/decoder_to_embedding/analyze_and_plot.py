"""
Full analysis + visualization for decoder-similarity → semantic-alignment experiment.
Pools results across all LLM families (Gemma, Pythia, GPT-2).

Outputs:
  1. Per-family bar plot: mean embed sim per decoder-sim bin (with error bars)
  2. Combined scatter plot: continuous decoder sim vs embed sim
  3. Window-size sweep: Spearman rho vs window size per family
  4. Stats summary JSON
"""

import argparse
import glob
import json
import os
from collections import defaultdict

import matplotlib
import numpy as np
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── config ──────────────────────────────────────────────────────────────
RESULT_ROOT = "<DATA_ROOT>/rebuttal_redundancy_to_semantic_alignment"
PLOT_DIR = os.path.join(os.path.dirname(__file__), "plots")

FAMILIES = {
    "Gemma-2 (2B)": os.path.join(RESULT_ROOT, "saebench_gemma", "4k"),
    "Pythia (160M)": os.path.join(RESULT_ROOT, "saebench_pythia", "4k"),
    "GPT-2 (124M)": os.path.join(RESULT_ROOT, "benchmark_training_gpt2", "4k"),
    "OPT (125M)": os.path.join(RESULT_ROOT, "benchmark_training_opt125m", "4k"),
    "TinyStories (21M)": os.path.join(
        RESULT_ROOT, "benchmark_training_tinystories", "4k"
    ),
}

BINS_ORDER = ["0.0-0.3", "0.3-0.5", "0.5-0.7", "0.7-0.9", "0.9-1.0"]
BIN_LABELS = ["[0, 0.3)", "[0.3, 0.5)", "[0.5, 0.7)", "[0.7, 0.9)", "[0.9, 1.0]"]
WINDOW_SIZES = ["8", "16", "32", "64"]
PRIMARY_W = "8"  # best signal


def load_family(base_path):
    """Load all results.json files under a family directory."""
    files = sorted(
        glob.glob(os.path.join(base_path, "**", "results.json"), recursive=True)
    )
    results = []
    for f in files:
        with open(f) as fh:
            r = json.load(fh)
            r["_path"] = f
            results.append(r)
    return results


def extract_pairs(results, window_sizes=WINDOW_SIZES, bins_order=BINS_ORDER):
    """Extract all pair-level data from a list of result dicts.

    Returns:
        all_pairs: {w: {"decoder_sim": [], "embed_sim": []}}
        bin_level: {w: {bin: [embed_sims]}}
        per_sae: {w: [list of (ds_list, es_list) per SAE]}
    """
    all_pairs = {w: {"decoder_sim": [], "embed_sim": []} for w in window_sizes}
    bin_level = {w: {b: [] for b in bins_order} for w in window_sizes}
    per_sae = {w: [] for w in window_sizes}

    for r in results:
        sae_pairs = {w: ([], []) for w in window_sizes}
        for bin_name in bins_order:
            if bin_name not in r.get("pair_details", {}):
                continue
            for w in window_sizes:
                if w not in r["pair_details"][bin_name]:
                    continue
                for p in r["pair_details"][bin_name][w]:
                    ds = p["decoder_sim"]
                    es = p["embed_sim"]
                    all_pairs[w]["decoder_sim"].append(ds)
                    all_pairs[w]["embed_sim"].append(es)
                    bin_level[w][bin_name].append(es)
                    sae_pairs[w][0].append(ds)
                    sae_pairs[w][1].append(es)
        for w in window_sizes:
            if len(sae_pairs[w][0]) > 10:
                per_sae[w].append(sae_pairs[w])

    return all_pairs, bin_level, per_sae


def compute_stats(all_pairs, bin_level, per_sae, bins_order=BINS_ORDER):
    """Compute summary statistics for a family."""
    stats_out = {}
    for w in WINDOW_SIZES:
        ds = np.array(all_pairs[w]["decoder_sim"])
        es = np.array(all_pairs[w]["embed_sim"])

        # Pooled Spearman
        if len(ds) > 2:
            rho, pval = stats.spearmanr(ds, es)
        else:
            rho, pval = float("nan"), float("nan")

        # Per-SAE Spearman distribution
        per_sae_rhos = []
        for ds_s, es_s in per_sae[w]:
            r, _ = stats.spearmanr(ds_s, es_s)
            per_sae_rhos.append(r)
        rhos = np.array(per_sae_rhos)

        # One-sample t-test: rho > 0
        if len(rhos) > 1:
            t_stat, t_pval = stats.ttest_1samp(rhos, 0)
        else:
            t_stat, t_pval = float("nan"), float("nan")

        # Per-bin means
        bin_means = {}
        for b in bins_order:
            vals = bin_level[w][b]
            if vals:
                bin_means[b] = {
                    "mean": float(np.mean(vals)),
                    "std": float(np.std(vals)),
                    "sem": float(np.std(vals) / np.sqrt(len(vals))),
                    "n": len(vals),
                }
            else:
                bin_means[b] = {"mean": None, "std": None, "sem": None, "n": 0}

        stats_out[f"w{w}"] = {
            "n_pairs": int(len(ds)),
            "pooled_spearman_rho": float(rho),
            "pooled_spearman_p": float(pval),
            "n_saes": int(len(rhos)),
            "per_sae_mean_rho": float(np.mean(rhos)) if len(rhos) > 0 else None,
            "per_sae_median_rho": float(np.median(rhos)) if len(rhos) > 0 else None,
            "per_sae_std_rho": float(np.std(rhos)) if len(rhos) > 0 else None,
            "ttest_t": float(t_stat),
            "ttest_p": float(t_pval),
            "bins": bin_means,
        }

    return stats_out


# ── plotting ────────────────────────────────────────────────────────────


def plot_bar_per_family(family_data, save_path):
    """Bar chart: per-bin mean embedding similarity, one subplot per family, using W=8."""
    n_families = len(family_data)
    fig, axes = plt.subplots(1, n_families, figsize=(5 * n_families, 4.5), sharey=True)
    if n_families == 1:
        axes = [axes]

    w = PRIMARY_W
    colors = plt.cm.Blues(np.linspace(0.3, 0.9, len(BINS_ORDER)))

    for ax, (family_name, (all_pairs, bin_level, per_sae)) in zip(
        axes, family_data.items()
    ):
        means = []
        sems = []
        ns = []
        for b in BINS_ORDER:
            vals = bin_level[w][b]
            if vals:
                means.append(np.mean(vals))
                sems.append(np.std(vals) / np.sqrt(len(vals)))
                ns.append(len(vals))
            else:
                means.append(0)
                sems.append(0)
                ns.append(0)

        x = np.arange(len(BINS_ORDER))
        bars = ax.bar(
            x,
            means,
            yerr=sems,
            capsize=4,
            color=colors,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(BIN_LABELS, rotation=30, ha="right", fontsize=8)
        ax.set_title(family_name, fontsize=11, fontweight="bold")
        ax.set_xlabel("Decoder cosine similarity bin", fontsize=9)

        # Annotate n on each bar
        for i, (bar, n) in enumerate(zip(bars, ns)):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + sems[i] + 0.002,
                f"n={n}",
                ha="center",
                va="bottom",
                fontsize=7,
                color="gray",
            )

        # Spearman annotation
        ds = np.array(all_pairs[w]["decoder_sim"])
        es = np.array(all_pairs[w]["embed_sim"])
        if len(ds) > 2:
            rho, pval = stats.spearmanr(ds, es)
            p_str = f"p<1e-10" if pval < 1e-10 else f"p={pval:.1e}"
            ax.text(
                0.95,
                0.95,
                f"Spearman rho={rho:.3f}\n({p_str})",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
            )

    axes[0].set_ylabel("Mean cross-pair embedding similarity", fontsize=9)
    fig.suptitle(
        "Decoder Cosine Similarity Predicts Semantic Similarity (W=8 tokens)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_combined_bar(family_data, save_path, use_median=False):
    """Single grouped bar chart: all families side by side per bin, W=8."""
    w = PRIMARY_W
    agg_func = np.median if use_median else np.mean
    agg_label = "Median" if use_median else "Mean"
    n_families = len(family_data)
    n_bins = len(BINS_ORDER)
    bar_width = 0.8 / n_families

    fig, ax = plt.subplots(figsize=(8, 5))
    family_colors = [
        "#4C72B0",
        "#DD8452",
        "#55A868",
        "#C44E52",
        "#8172B3",
    ]  # blue, orange, green, red, purple

    for i, (family_name, (all_pairs, bin_level, per_sae)) in enumerate(
        family_data.items()
    ):
        centers = []
        sems = []
        for b in BINS_ORDER:
            vals = bin_level[w][b]
            if vals:
                centers.append(agg_func(vals))
                sems.append(np.std(vals) / np.sqrt(len(vals)))
            else:
                centers.append(0)
                sems.append(0)

        x = np.arange(n_bins) + i * bar_width
        ax.bar(
            x,
            centers,
            bar_width,
            yerr=sems,
            capsize=3,
            label=family_name,
            color=family_colors[i % len(family_colors)],
            edgecolor="black",
            linewidth=0.5,
            alpha=0.85,
        )

    ax.set_xticks(np.arange(n_bins) + bar_width * (n_families - 1) / 2)
    ax.set_xticklabels(BIN_LABELS, rotation=25, ha="right", fontsize=9)
    ax.set_xlabel("Decoder cosine similarity bin", fontsize=10)
    ax.set_ylabel(f"{agg_label} cross-pair embedding similarity", fontsize=10)
    ax.set_title(
        f"Higher Decoder Similarity → Higher Semantic Similarity ({agg_label})",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_scatter(family_data, save_path):
    """Scatter: continuous decoder sim vs embed sim per family, W=8."""
    w = PRIMARY_W
    n_families = len(family_data)
    fig, axes = plt.subplots(1, n_families, figsize=(5 * n_families, 4.5), sharey=True)
    if n_families == 1:
        axes = [axes]

    for ax, (family_name, (all_pairs, bin_level, per_sae)) in zip(
        axes, family_data.items()
    ):
        ds = np.array(all_pairs[w]["decoder_sim"])
        es = np.array(all_pairs[w]["embed_sim"])

        ax.scatter(ds, es, alpha=0.08, s=6, color="#4C72B0", rasterized=True)

        # Binned means for trend line
        n_trend_bins = 20
        bin_edges = np.linspace(0, 1, n_trend_bins + 1)
        bin_centers = []
        bin_means = []
        for j in range(n_trend_bins):
            mask = (ds >= bin_edges[j]) & (ds < bin_edges[j + 1])
            if mask.sum() >= 5:
                bin_centers.append((bin_edges[j] + bin_edges[j + 1]) / 2)
                bin_means.append(np.mean(es[mask]))
        ax.plot(
            bin_centers,
            bin_means,
            "r-o",
            markersize=4,
            linewidth=2,
            label="Binned mean",
        )

        rho, pval = stats.spearmanr(ds, es)
        p_str = "p<1e-10" if pval < 1e-10 else f"p={pval:.1e}"
        ax.set_title(f"{family_name}\nSpearman rho={rho:.3f} ({p_str})", fontsize=10)
        ax.set_xlabel("Decoder cosine similarity", fontsize=9)
        ax.legend(fontsize=8)

    axes[0].set_ylabel("Cross-pair embedding similarity", fontsize=9)
    fig.suptitle(
        "Feature Pair Decoder Similarity vs Semantic Similarity (W=8)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_window_sweep(family_stats, save_path):
    """Line plot: Spearman rho vs window size for each family."""
    fig, ax = plt.subplots(figsize=(6, 4))
    markers = ["o", "s", "^", "D", "v"]
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    for i, (family_name, st) in enumerate(family_stats.items()):
        ws = [int(w) for w in WINDOW_SIZES]
        rhos = [st[f"w{w}"]["pooled_spearman_rho"] for w in WINDOW_SIZES]
        ax.plot(
            ws,
            rhos,
            f"-{markers[i % len(markers)]}",
            color=colors[i % len(colors)],
            label=family_name,
            linewidth=2,
            markersize=7,
        )

    ax.set_xlabel("Window size W (tokens)", fontsize=10)
    ax.set_ylabel("Spearman rho (decoder sim vs embed sim)", fontsize=10)
    ax.set_title(
        "Effect of Context Window Size on Correlation", fontsize=12, fontweight="bold"
    )
    ax.set_xticks([int(w) for w in WINDOW_SIZES])
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_per_sae_rho_distribution(family_data, save_path):
    """Histogram/violin of per-SAE Spearman rho across families, W=8."""
    w = PRIMARY_W
    fig, ax = plt.subplots(figsize=(9, 4))
    colors = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3"]

    data_for_violin = []
    labels = []
    for i, (family_name, (all_pairs, bin_level, per_sae)) in enumerate(
        family_data.items()
    ):
        rhos = []
        for ds_s, es_s in per_sae[w]:
            r, _ = stats.spearmanr(ds_s, es_s)
            rhos.append(r)
        data_for_violin.append(rhos)
        labels.append(family_name)

    parts = ax.violinplot(data_for_violin, showmeans=True, showmedians=True)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors[i % len(colors)])
        pc.set_alpha(0.6)

    ax.set_xticks(range(1, len(labels) + 1))
    ax.set_xticklabels(labels, fontsize=9)
    ax.axhline(0, color="gray", linestyle="--", linewidth=0.8)
    ax.set_ylabel("Per-SAE Spearman rho", fontsize=10)
    ax.set_title(
        "Distribution of Per-SAE Correlation (W=8)", fontsize=12, fontweight="bold"
    )
    ax.grid(axis="y", alpha=0.3)

    # Annotate means
    for i, rhos in enumerate(data_for_violin):
        m = np.mean(rhos)
        ax.text(
            i + 1, m + 0.02, f"mean={m:.3f}", ha="center", fontsize=8, color=colors[i]
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_overall_violin(family_data, save_path):
    """Violin plot: distribution of embedding similarities per decoder-sim bin, pooled, W=8."""
    w = PRIMARY_W
    bin_all = {b: [] for b in BINS_ORDER}
    all_ds, all_es = [], []

    for family_name, (all_pairs, bin_level, per_sae) in family_data.items():
        for b in BINS_ORDER:
            bin_all[b].extend(bin_level[w][b])
        all_ds.extend(all_pairs[w]["decoder_sim"])
        all_es.extend(all_pairs[w]["embed_sim"])

    total_pairs = len(all_ds)
    rho, _ = stats.spearmanr(all_ds, all_es)

    fig, ax = plt.subplots(figsize=(7, 5))
    data = [bin_all[b] for b in BINS_ORDER]
    colors_v = plt.cm.Blues(np.linspace(0.35, 0.85, len(BINS_ORDER)))

    parts = ax.violinplot(data, showmeans=True, showmedians=True, showextrema=False)
    for i, pc in enumerate(parts["bodies"]):
        pc.set_facecolor(colors_v[i])
        pc.set_edgecolor("black")
        pc.set_linewidth(0.5)
        pc.set_alpha(0.7)
    parts["cmeans"].set_color("red")
    parts["cmedians"].set_color("black")

    ax.set_xticks(range(1, len(BINS_ORDER) + 1))
    ax.set_xticklabels(BIN_LABELS, rotation=25, ha="right", fontsize=10)
    ax.set_xlabel("Decoder cosine similarity bin", fontsize=11)
    ax.set_ylabel("Cross-pair embedding similarity", fontsize=11)
    ax.set_title(
        "Distribution of Semantic Similarity by Decoder Similarity\n"
        f"({total_pairs:,} pairs, Spearman rho={rho:.3f}, p<1e-10)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)

    # Annotate mean & median per bin
    for i, vals in enumerate(data):
        if vals:
            m = np.mean(vals)
            md = np.median(vals)
            ax.text(
                i + 1, m + 0.005, f"mean={m:.3f}", ha="center", fontsize=7, color="red"
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_overall_bar(family_data, save_path):
    """Single bar plot: all pairs pooled across all families and architectures, W=8."""
    w = PRIMARY_W
    bin_all = {b: [] for b in BINS_ORDER}
    total_pairs = 0
    total_saes = 0
    all_ds, all_es = [], []

    for family_name, (all_pairs, bin_level, per_sae) in family_data.items():
        total_saes += len(per_sae[w])
        for b in BINS_ORDER:
            bin_all[b].extend(bin_level[w][b])
        all_ds.extend(all_pairs[w]["decoder_sim"])
        all_es.extend(all_pairs[w]["embed_sim"])

    total_pairs = len(all_ds)
    rho, pval = stats.spearmanr(all_ds, all_es)

    fig, ax = plt.subplots(figsize=(6, 5))
    colors = plt.cm.Blues(np.linspace(0.35, 0.85, len(BINS_ORDER)))

    means, sems, ns = [], [], []
    for b in BINS_ORDER:
        vals = bin_all[b]
        if vals:
            means.append(np.mean(vals))
            sems.append(np.std(vals) / np.sqrt(len(vals)))
            ns.append(len(vals))
        else:
            means.append(0)
            sems.append(0)
            ns.append(0)

    x = np.arange(len(BINS_ORDER))
    bars = ax.bar(
        x,
        means,
        yerr=sems,
        capsize=5,
        color=colors,
        edgecolor="black",
        linewidth=0.7,
        width=0.65,
    )
    ax.set_xticks(x)
    ax.set_xticklabels(BIN_LABELS, rotation=25, ha="right", fontsize=10)
    ax.set_xlabel("Decoder cosine similarity bin", fontsize=11)
    ax.set_ylabel("Mean cross-pair embedding similarity", fontsize=11)
    ax.set_title(
        "Decoder Similarity Predicts Semantic Similarity\n"
        f"({total_saes} SAEs, {total_pairs:,} pairs, Spearman rho={rho:.3f}, p<1e-10)",
        fontsize=12,
        fontweight="bold",
    )
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


# ── per-architecture analysis ───────────────────────────────────────────

# Normalize architecture names across SAE Bench (capitalized) and benchmark training (lowercase)
ARCH_NORMALIZE = {
    "Standard": "Standard",
    "standard": "Standard",
    "Gated": "Gated",
    "gated": "Gated",
    "GatedSAE": "Gated",
    "JumpReLU": "JumpReLU",
    "jumprelu": "JumpReLU",
    "TopK": "TopK",
    "topk": "TopK",
    "BatchTopK": "BatchTopK",
    "batchtopk": "BatchTopK",
    "Matryoshka": "Matryoshka",
    "MatryoshkaBatchTopK": "Matryoshka",
    "matryoshka_batchtopk": "Matryoshka",
    "PAnneal": "PAnneal",
}


def load_all_results_with_arch():
    """Load all results across all families, tagged with normalized architecture."""
    all_results_by_arch = defaultdict(list)
    for name, path in FAMILIES.items():
        for f in sorted(
            glob.glob(os.path.join(path, "**", "results.json"), recursive=True)
        ):
            with open(f) as fh:
                r = json.load(fh)
            raw_arch = r["summary"]["metadata"].get("architecture", "unknown")
            arch = ARCH_NORMALIZE.get(raw_arch, raw_arch)
            r["_path"] = f
            r["_family"] = name
            all_results_by_arch[arch].append(r)
    return all_results_by_arch


def plot_bar_per_architecture(save_path):
    """Bar chart: per-bin mean embed sim, one subplot per architecture, W=8."""
    arch_results = load_all_results_with_arch()
    # Sort architectures alphabetically
    arch_names = sorted(arch_results.keys())
    n_archs = len(arch_names)
    if n_archs == 0:
        return

    fig, axes = plt.subplots(1, n_archs, figsize=(4 * n_archs, 4.5), sharey=True)
    if n_archs == 1:
        axes = [axes]

    w = PRIMARY_W
    bar_colors = plt.cm.Oranges(np.linspace(0.3, 0.9, len(BINS_ORDER)))

    for ax, arch_name in zip(axes, arch_names):
        results = arch_results[arch_name]
        all_pairs, bin_level, per_sae = extract_pairs(results)

        means, sems, ns = [], [], []
        for b in BINS_ORDER:
            vals = bin_level[w][b]
            if vals:
                means.append(np.mean(vals))
                sems.append(np.std(vals) / np.sqrt(len(vals)))
                ns.append(len(vals))
            else:
                means.append(0)
                sems.append(0)
                ns.append(0)

        x = np.arange(len(BINS_ORDER))
        bars = ax.bar(
            x,
            means,
            yerr=sems,
            capsize=4,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.5,
        )
        ax.set_xticks(x)
        ax.set_xticklabels(BIN_LABELS, rotation=30, ha="right", fontsize=7)
        ax.set_title(
            f"{arch_name}\n({len(results)} SAEs)", fontsize=10, fontweight="bold"
        )
        ax.set_xlabel("Decoder cos sim bin", fontsize=8)

        # Spearman annotation
        ds = np.array(all_pairs[w]["decoder_sim"])
        es = np.array(all_pairs[w]["embed_sim"])
        if len(ds) > 2:
            rho, pval = stats.spearmanr(ds, es)
            p_str = "p<1e-10" if pval < 1e-10 else f"p={pval:.1e}"
            ax.text(
                0.95,
                0.95,
                f"rho={rho:.3f}\n({p_str})",
                transform=ax.transAxes,
                ha="right",
                va="top",
                fontsize=7,
                bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow", alpha=0.8),
            )

    axes[0].set_ylabel("Mean cross-pair embedding similarity", fontsize=9)
    fig.suptitle(
        "Per-Architecture: Decoder Similarity Predicts Semantic Similarity (W=8)",
        fontsize=12,
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def plot_combined_bar_per_architecture(save_path, use_median=False):
    """Grouped bar chart: all architectures side by side per bin, W=8."""
    arch_results = load_all_results_with_arch()
    arch_names = sorted(arch_results.keys())
    n_archs = len(arch_names)
    if n_archs == 0:
        return

    agg_func = np.median if use_median else np.mean
    agg_label = "Median" if use_median else "Mean"
    w = PRIMARY_W
    n_bins = len(BINS_ORDER)
    bar_width = 0.8 / n_archs

    fig, ax = plt.subplots(figsize=(10, 5))
    arch_colors = plt.cm.tab10(np.linspace(0, 1, n_archs))

    for i, arch_name in enumerate(arch_names):
        results = arch_results[arch_name]
        _, bin_level, _ = extract_pairs(results)

        centers, sems = [], []
        for b in BINS_ORDER:
            vals = bin_level[w][b]
            if vals:
                centers.append(agg_func(vals))
                sems.append(np.std(vals) / np.sqrt(len(vals)))
            else:
                centers.append(0)
                sems.append(0)

        x = np.arange(n_bins) + i * bar_width
        ax.bar(
            x,
            centers,
            bar_width,
            yerr=sems,
            capsize=2,
            label=f"{arch_name} ({len(results)})",
            color=arch_colors[i],
            edgecolor="black",
            linewidth=0.3,
            alpha=0.85,
        )

    ax.set_xticks(np.arange(n_bins) + bar_width * (n_archs - 1) / 2)
    ax.set_xticklabels(BIN_LABELS, rotation=25, ha="right", fontsize=9)
    ax.set_xlabel("Decoder cosine similarity bin", fontsize=10)
    ax.set_ylabel(f"{agg_label} cross-pair embedding similarity", fontsize=10)
    ax.set_title(
        f"Per-Architecture: Higher Decoder Sim → Higher Semantic Sim ({agg_label}, W=8)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(fontsize=7, loc="upper left", ncol=2)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.close(fig)


def print_per_architecture_stats():
    """Print per-architecture Spearman stats table."""
    arch_results = load_all_results_with_arch()
    arch_names = sorted(arch_results.keys())
    w = PRIMARY_W

    print("\n" + "=" * 80)
    print("PER-ARCHITECTURE SUMMARY (W=8)")
    print("=" * 80)
    print(
        f"{'Architecture':<18} {'N SAEs':<8} {'N Pairs':<10} {'Pooled rho':<12} {'Per-SAE rho':<14} {'t-test p':<12}"
    )
    print("-" * 80)

    arch_stats = {}
    for arch_name in arch_names:
        results = arch_results[arch_name]
        all_pairs, bin_level, per_sae = extract_pairs(results)

        ds = np.array(all_pairs[w]["decoder_sim"])
        es = np.array(all_pairs[w]["embed_sim"])
        rho, pval = (
            stats.spearmanr(ds, es) if len(ds) > 2 else (float("nan"), float("nan"))
        )

        per_sae_rhos = [stats.spearmanr(d, e)[0] for d, e in per_sae[w]]
        rhos_arr = np.array(per_sae_rhos)
        mean_rho = float(np.mean(rhos_arr)) if len(rhos_arr) > 0 else float("nan")
        if len(rhos_arr) > 1:
            _, t_pval = stats.ttest_1samp(rhos_arr, 0)
        else:
            t_pval = float("nan")

        print(
            f"{arch_name:<18} {len(results):<8} {len(ds):<10} {rho:<12.4f} {mean_rho:<14.4f} {t_pval:<12.2e}"
        )

        arch_stats[arch_name] = {
            "n_saes": len(results),
            "n_pairs": len(ds),
            "pooled_rho": float(rho),
            "per_sae_mean_rho": mean_rho,
            "ttest_p": float(t_pval),
        }

    return arch_stats


# ── main ────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--families",
        nargs="*",
        default=None,
        help="Which families to include (default: all available)",
    )
    args = parser.parse_args()

    os.makedirs(PLOT_DIR, exist_ok=True)

    # Load all available families
    family_data = {}  # name -> (all_pairs, bin_level, per_sae)
    family_stats = {}

    for name, path in FAMILIES.items():
        if args.families and name not in args.families:
            continue
        results = load_family(path)
        if not results:
            print(f"  {name}: no results found at {path}, skipping")
            continue
        print(f"  {name}: {len(results)} SAEs loaded from {path}")
        all_pairs, bin_level, per_sae = extract_pairs(results)
        family_data[name] = (all_pairs, bin_level, per_sae)
        family_stats[name] = compute_stats(all_pairs, bin_level, per_sae)

    if not family_data:
        print("No data found!")
        return

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY (W=8)")
    print("=" * 80)
    print(
        f"{'Family':<20} {'N SAEs':<8} {'N Pairs':<10} {'Pooled rho':<12} {'Per-SAE rho':<14} {'t-test p':<12}"
    )
    print("-" * 80)
    for name, st in family_stats.items():
        s = st["w8"]
        print(
            f"{name:<20} {s['n_saes']:<8} {s['n_pairs']:<10} {s['pooled_spearman_rho']:<12.4f} "
            f"{s['per_sae_mean_rho']:<14.4f} {s['ttest_p']:<12.2e}"
        )

    print("\nPer-bin means (W=8):")
    print(f"{'Bin':<12}", end="")
    for name in family_data:
        print(f"  {name:<18}", end="")
    print()
    for b, bl in zip(BINS_ORDER, BIN_LABELS):
        print(f"{bl:<12}", end="")
        for name in family_stats:
            bm = family_stats[name]["w8"]["bins"][b]
            if bm["mean"] is not None:
                print(f"  {bm['mean']:.4f} (n={bm['n']:<5})", end="")
            else:
                print(f"  {'--':<18}", end="")
        print()

    # Save stats
    stats_path = os.path.join(PLOT_DIR, "stats_summary.json")
    with open(stats_path, "w") as f:
        json.dump(family_stats, f, indent=2)
    print(f"\n  Saved stats: {stats_path}")

    # Generate plots
    print("\nGenerating plots...")
    plot_bar_per_family(family_data, os.path.join(PLOT_DIR, "bar_per_family.png"))
    plot_combined_bar(
        family_data, os.path.join(PLOT_DIR, "bar_combined.png"), use_median=False
    )
    plot_combined_bar(
        family_data, os.path.join(PLOT_DIR, "bar_combined_median.png"), use_median=True
    )
    plot_overall_bar(family_data, os.path.join(PLOT_DIR, "bar_overall.png"))
    plot_overall_violin(family_data, os.path.join(PLOT_DIR, "violin_overall.png"))
    plot_scatter(family_data, os.path.join(PLOT_DIR, "scatter.png"))
    plot_window_sweep(family_stats, os.path.join(PLOT_DIR, "window_sweep.png"))
    plot_per_sae_rho_distribution(
        family_data, os.path.join(PLOT_DIR, "per_sae_rho_violin.png")
    )

    # Per-architecture plots
    print("\nPer-architecture analysis...")
    arch_stats = print_per_architecture_stats()
    plot_bar_per_architecture(os.path.join(PLOT_DIR, "bar_per_architecture.png"))
    plot_combined_bar_per_architecture(
        os.path.join(PLOT_DIR, "bar_combined_architecture.png"), use_median=False
    )
    plot_combined_bar_per_architecture(
        os.path.join(PLOT_DIR, "bar_combined_architecture_median.png"), use_median=True
    )

    # Save arch stats
    arch_stats_path = os.path.join(PLOT_DIR, "stats_per_architecture.json")
    with open(arch_stats_path, "w") as f:
        json.dump(arch_stats, f, indent=2)
    print(f"  Saved: {arch_stats_path}")

    print("\nDone!")


if __name__ == "__main__":
    main()
