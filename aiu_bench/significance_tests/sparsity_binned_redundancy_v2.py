#!/usr/bin/env python3
"""
Sparsity-binned redundancy analysis using analysis8 data (threshold sweep).

Uses per-SAE counting (% of SAEs where Q1 has the most within-quantile
redundant pairs) instead of q1_frac, and includes both filtered and
unfiltered results.

Data sources:
  - L0: <DATA_ROOT>/analysis0_sparsity/**/results.json
  - Redundancy: <DATA_ROOT>/analysis8_redundancy_threshold_sweep/**/results.json
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SPARSITY_ROOT = Path("<DATA_ROOT>/analysis0_sparsity")
ANALYSIS8_ROOT = Path("<DATA_ROOT>/analysis8_redundancy_threshold_sweep")
THRESHOLD = "0.7"
L0_BINS = [(0, 100), (100, 200), (200, 300), (300, 500), (500, 1000), (1000, 5000)]

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


def detect_source(path_str):
    pl = path_str.lower()
    if "saebench_pythia" in pl:
        return "pythia"
    if "saebench_gemma" in pl:
        return "gemma"
    if "benchmark_training_gpt2" in pl or "benchmark_gpt2" in pl:
        return "gpt2"
    if "benchmark_training_opt" in pl or "benchmark_opt" in pl:
        return "opt125m"
    if "benchmark_training_tiny" in pl or "benchmark_tiny" in pl:
        return "tinystories"
    if "canrager" in pl:
        return "canrager"
    if "jbloom" in pl:
        return "jbloom"
    return "unknown"


def make_key(data, path_str):
    """Create matching key between sparsity and analysis8 results."""
    source = detect_source(path_str)
    cfg = data.get("config", data)
    combo_id = cfg.get("combo_id", "")
    arch = cfg.get("architecture", cfg.get("sae_type", ""))
    trainer = cfg.get("trainer", "")
    width = str(cfg.get("width", cfg.get("width_name", cfg.get("d_sae", ""))))
    layer = str(cfg.get("layer", ""))

    if source == "jbloom":
        repo = cfg.get("repo_id", "")
        if "/" in repo:
            repo = repo.split("/", 1)[1]
        return (source, repo, layer)

    if combo_id:
        return (source, combo_id)

    if trainer:
        return (source, arch, width, trainer)

    # Fallback: use path components
    parts = Path(path_str).parts
    return (source, *[p for p in parts[-4:] if p != "results.json"])


def load_sparsity():
    """Load L0 from analysis0_sparsity."""
    data = {}
    for f in sorted(SPARSITY_ROOT.rglob("results.json")):
        try:
            d = json.load(open(f))
            sm = d.get("sparsity_metrics", {})
            avg_l0 = sm.get("average_active_features")
            if avg_l0 is None:
                continue
            cfg = d.get("config", {})
            arch = cfg.get("architecture", cfg.get("sae_type", ""))
            arch = ARCH_NORMALIZE.get(arch, ARCH_NORMALIZE.get(arch.lower(), arch))
            key = make_key(d, str(f))
            data[key] = {"avg_l0": avg_l0, "architecture": arch}
        except Exception:
            continue
    return data


def load_analysis8():
    """Load analysis8 threshold sweep results."""
    data = {}
    for f in sorted(ANALYSIS8_ROOT.rglob("results.json")):
        try:
            d = json.load(open(f))
            arch = d.get("architecture", d.get("sae_type", "Standard"))
            if arch == "unknown":
                arch = "Standard"
            arch = ARCH_NORMALIZE.get(arch, ARCH_NORMALIZE.get(arch.lower(), arch))

            key = make_key(d, str(f))
            data[key] = {"data": d, "architecture": arch, "path": str(f)}
        except Exception:
            continue
    return data


def q1_wins(pairs_data, threshold):
    """Check if Q1 has the most within-quantile redundant pairs."""
    counts = {}
    for q in [1, 2, 3, 4]:
        q_data = pairs_data.get(str(q), pairs_data.get(q, {}))
        counts[q] = q_data.get(threshold, 0)
    total = sum(counts.values())
    if total == 0:
        return None  # no pairs at this threshold
    q1_count = counts[1]
    others_max = max(counts[2], counts[3], counts[4])
    return q1_count > 0 and q1_count >= others_max


def analyze_bin(entries, threshold, tag):
    """Analyze one L0 bin."""
    wins_unf = 0
    wins_filt = 0
    total_unf = 0
    total_filt = 0
    arch_wins_unf = defaultdict(lambda: [0, 0])  # [wins, total]
    arch_wins_filt = defaultdict(lambda: [0, 0])

    for entry in entries:
        d = entry["data"]
        arch = entry["architecture"]

        # Unfiltered
        qp = d.get("quantile_pairs")
        if qp:
            result = q1_wins(qp, threshold)
            if result is not None:
                total_unf += 1
                arch_wins_unf[arch][1] += 1
                if result:
                    wins_unf += 1
                    arch_wins_unf[arch][0] += 1

        # Filtered
        qpf = d.get("quantile_pairs_filtered")
        if qpf:
            result = q1_wins(qpf, threshold)
            if result is not None:
                total_filt += 1
                arch_wins_filt[arch][1] += 1
                if result:
                    wins_filt += 1
                    arch_wins_filt[arch][0] += 1

    return {
        "unfiltered": {
            "wins": wins_unf,
            "total": total_unf,
            "pct": 100 * wins_unf / total_unf if total_unf > 0 else 0,
            "per_arch": {
                a: {
                    "wins": v[0],
                    "total": v[1],
                    "pct": 100 * v[0] / v[1] if v[1] > 0 else 0,
                }
                for a, v in sorted(arch_wins_unf.items())
            },
        },
        "filtered": {
            "wins": wins_filt,
            "total": total_filt,
            "pct": 100 * wins_filt / total_filt if total_filt > 0 else 0,
            "per_arch": {
                a: {
                    "wins": v[0],
                    "total": v[1],
                    "pct": 100 * v[0] / v[1] if v[1] > 0 else 0,
                }
                for a, v in sorted(arch_wins_filt.items())
            },
        },
    }


def main():
    print("=" * 90)
    print("SPARSITY-BINNED REDUNDANCY (per-SAE Q1 wins, analysis8 data)")
    print("=" * 90)

    sparsity = load_sparsity()
    analysis8 = load_analysis8()
    print(f"Loaded: Sparsity={len(sparsity)}, Analysis8={len(analysis8)}")

    # Match by key
    matched = []
    for key in analysis8:
        if key in sparsity:
            entry = analysis8[key]
            entry["avg_l0"] = sparsity[key]["avg_l0"]
            matched.append(entry)

    print(f"Matched: {len(matched)}")

    # Architecture distribution
    arch_counts = defaultdict(int)
    for e in matched:
        arch_counts[e["architecture"]] += 1
    print("Architecture distribution:")
    for a in sorted(arch_counts):
        print(f"  {a}: {arch_counts[a]}")

    # Bin by L0
    print(f"\nThreshold: {THRESHOLD}")
    print(f"\n{'Bin':<15} | {'--- Unfiltered ---':^30} | {'--- Filtered ---':^30}")
    print(
        f"{'':15} | {'Q1win%':>8} {'wins/total':>12} {'':>8} | {'Q1win%':>8} {'wins/total':>12}"
    )
    print("-" * 85)

    for lo, hi in L0_BINS:
        bin_entries = [e for e in matched if lo <= e["avg_l0"] < hi]
        if not bin_entries:
            continue
        label = f"L0 {lo}-{hi}"
        result = analyze_bin(bin_entries, THRESHOLD, label)
        u = result["unfiltered"]
        f_ = result["filtered"]
        print(
            f"{label:<15} | {u['pct']:>7.1f}% {u['wins']:>5d}/{u['total']:<5d}        | {f_['pct']:>7.1f}% {f_['wins']:>5d}/{f_['total']:<5d}"
        )

    # Per-architecture detail
    print(f"\n{'=' * 90}")
    print("PER-ARCHITECTURE BREAKDOWN (filtered, threshold=0.7)")
    print(f"{'=' * 90}")

    all_archs = sorted(set(e["architecture"] for e in matched))
    header = f"{'Bin':<15}"
    for a in all_archs:
        header += f"  {a:>10}"
    print(header)
    print("-" * (15 + 12 * len(all_archs)))

    for lo, hi in L0_BINS:
        bin_entries = [e for e in matched if lo <= e["avg_l0"] < hi]
        if not bin_entries:
            continue
        label = f"L0 {lo}-{hi}"
        result = analyze_bin(bin_entries, THRESHOLD, label)
        row = f"{label:<15}"
        for a in all_archs:
            ad = result["filtered"]["per_arch"].get(a, {})
            n = ad.get("total", 0)
            if n > 0:
                row += f"  {ad['pct']:>5.0f}%({n:>2d})"
            else:
                row += f"  {'--':>10}"
        print(row)

    # JumpReLU detail
    print(f"\n{'=' * 90}")
    print("JUMPRELU DETAIL (filtered, threshold=0.7)")
    print(f"{'=' * 90}")
    jr_entries = [e for e in matched if e["architecture"] == "JumpReLU"]
    jr_entries.sort(key=lambda x: x["avg_l0"])

    print(
        f"{'avg_L0':>8} {'Q1win':>6} {'Q1':>6} {'Q2':>6} {'Q3':>6} {'Q4':>6} {'total':>6}"
    )
    for e in jr_entries:
        d = e["data"]
        qpf = d.get("quantile_pairs_filtered", {})
        counts = {}
        for q in [1, 2, 3, 4]:
            q_data = qpf.get(str(q), qpf.get(q, {}))
            counts[q] = q_data.get(THRESHOLD, 0)
        total = sum(counts.values())
        if total == 0:
            continue
        win = q1_wins(qpf, THRESHOLD)
        print(
            f"{e['avg_l0']:>8.0f} {'YES' if win else 'no':>6} {counts[1]:>6} {counts[2]:>6} {counts[3]:>6} {counts[4]:>6} {total:>6}"
        )


if __name__ == "__main__":
    main()
