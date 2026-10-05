#!/usr/bin/env python3
"""Aggregate per-SAE sparsity / dead-neuron results into per-architecture summaries.

Walks --root for **/results.json (per-SAE C4/C5 output). Produces:
  - l0_per_architecture.json         (median L0 in absolute count, %dict, %d_model)
  - dead_neuron_per_architecture.json (per-arch dead-neuron rates at three thresholds)

Mirrors paper Table tab:per_arch_l0.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, median


def detect_arch(p, fallback):
    s = (str(p) + " " + (fallback or "")).lower()
    for needle, name in [
        ("matryoshka", "Matryoshka"),
        ("batchtopk", "BatchTopK"),
        ("jumprelu", "JumpReLU"),
        ("panneal", "PAnneal"),
        ("gated", "Gated"),
        ("topk", "TopK"),
        ("standard", "Standard ReLU"),
    ]:
        if needle in s:
            return name
    return "Unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    l0_count = defaultdict(list)
    l0_pct_dict = defaultdict(list)
    l0_pct_dmodel = defaultdict(list)
    dead_rare = defaultdict(list)
    dead_very_rare = defaultdict(list)
    dead_never = defaultdict(list)

    for rf in root.rglob("results.json"):
        try:
            with open(rf) as f:
                d = json.load(f)
        except Exception:
            continue
        sm = d.get("sparsity_metrics", {})
        dnm = d.get("dead_neuron_metrics", {})
        cfg = d.get("config", {})
        if not sm or not dnm:
            continue
        arch = detect_arch(rf, cfg.get("architecture", ""))
        l0 = sm.get("average_active_features")
        n_total = dnm.get("n_features")
        d_in = cfg.get("d_in")
        if l0 is None or not n_total:
            continue
        l0_count[arch].append(l0)
        l0_pct_dict[arch].append(100 * l0 / n_total)
        if d_in:
            l0_pct_dmodel[arch].append(100 * l0 / d_in)
        for key, dst in [
            ("dead_neuron_proportion_rare", dead_rare),
            ("dead_neuron_proportion_very_rare", dead_very_rare),
            ("dead_neuron_proportion_never", dead_never),
        ]:
            v = dnm.get(key)
            if v is not None:
                dst[arch].append(v * 100 if v <= 1 else v)

    # L0 summary
    l0_summary = {}
    for arch in l0_count:
        n = len(l0_count[arch])
        l0_summary[arch] = {
            "n_saes": n,
            "median_l0_count": round(median(l0_count[arch]), 1),
            "median_l0_pct_dict": round(median(l0_pct_dict[arch]), 2),
            "median_l0_pct_dmodel": round(median(l0_pct_dmodel[arch]), 2)
            if l0_pct_dmodel[arch]
            else None,
        }
    (out / "l0_per_architecture.json").write_text(json.dumps(l0_summary, indent=2))

    # Dead-neuron summary
    dn_summary = {}
    for arch in l0_count:
        dn_summary[arch] = {
            "n_saes": len(l0_count[arch]),
            "rare_pct_mean": round(mean(dead_rare[arch]), 2)
            if dead_rare.get(arch)
            else None,
            "rare_pct_median": round(median(dead_rare[arch]), 2)
            if dead_rare.get(arch)
            else None,
            "very_rare_pct_mean": round(mean(dead_very_rare[arch]), 2)
            if dead_very_rare.get(arch)
            else None,
            "never_pct_mean": round(mean(dead_never[arch]), 2)
            if dead_never.get(arch)
            else None,
        }
    (out / "dead_neuron_per_architecture.json").write_text(
        json.dumps(dn_summary, indent=2)
    )

    # Print summary
    print(f"=== L0 per architecture ===")
    print(
        f"{'Architecture':18s} {'#SAE':>5}  {'Median L0':>11}  {'L0/dict %':>11}  {'L0/d_model %':>13}"
    )
    for arch, s in sorted(l0_summary.items()):
        dm = s.get("median_l0_pct_dmodel")
        dm_str = f"{dm:>13.2f}" if dm is not None else "          —"
        print(
            f"{arch:18s} {s['n_saes']:>5}  {s['median_l0_count']:>10.1f}  {s['median_l0_pct_dict']:>10.2f}  {dm_str}"
        )
    print(f"\nWrote {out}/l0_per_architecture.json + dead_neuron_per_architecture.json")


if __name__ == "__main__":
    main()
