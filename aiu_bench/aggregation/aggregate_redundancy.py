#!/usr/bin/env python3
"""Aggregate per-SAE redundancy results into architecture-level summaries.

Walks --root for **/results.json (per-SAE redundancy output, e.g. from
analysis8 threshold-sweep or per-family runners). Produces:
  - redundancy_per_architecture_filtered.json   (Q1..Q4 share + Cohen's d)
  - redundancy_per_architecture_unfiltered.json (same + pair filter rate)
  - redundancy_significance.json                (per-arch + overall Wilcoxon-style)

Per-SAE share at threshold theta:
    share(q) = quantile_pairs[q][theta] / sum_q quantile_pairs[q][theta]
where 'filtered' uses quantile_pairs_filtered (dead-neuron pairs removed).

Numbers are averaged across SAEs of the same architecture (each SAE weighted
equally). This matches the per-SAE Wilcoxon weighting used in Section 5.2.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev


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


def cohens_d(diffs):
    if len(diffs) < 2:
        return 0.0
    s = stdev(diffs)
    return mean(diffs) / s if s > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--threshold", default="0.7")
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    th = args.threshold

    # Per-SAE percentages (filtered + unfiltered) and per-SAE pair counts (for Wilcoxon)
    per_sae_filt = defaultdict(list)
    per_sae_unfilt = defaultdict(list)
    per_sae_pair_count_filt = defaultdict(list)  # (q1_count, q234_count) per SAE
    filter_rate = defaultdict(list)

    for rf in root.rglob("results.json"):
        try:
            with open(rf) as f:
                d = json.load(f)
        except Exception:
            continue
        if "quantile_pairs" not in d or "quantile_pairs_filtered" not in d:
            continue
        arch = detect_arch(rf, d.get("architecture", ""))

        u = {q: d["quantile_pairs"].get(q, {}).get(th, 0) for q in "1234"}
        f = {q: d["quantile_pairs_filtered"].get(q, {}).get(th, 0) for q in "1234"}
        u_total, f_total = sum(u.values()), sum(f.values())
        if u_total > 0:
            per_sae_unfilt[arch].append([100 * u[q] / u_total for q in "1234"])
            if f_total > 0:
                filter_rate[arch].append(100 * (1 - f_total / u_total))
        if f_total > 0:
            per_sae_filt[arch].append([100 * f[q] / f_total for q in "1234"])
            q1, q234 = f["1"], (f["2"] + f["3"] + f["4"])
            per_sae_pair_count_filt[arch].append((q1, q234))

    # Build outputs
    filt_summary = {}
    unfilt_summary = {}
    sig_summary = {}
    for arch in per_sae_filt:
        n = len(per_sae_filt[arch])
        f_avg = [mean(x[i] for x in per_sae_filt[arch]) for i in range(4)]
        filt_summary[arch] = {
            "n_saes": n,
            "Q1_pct": round(f_avg[0], 2),
            "Q2_pct": round(f_avg[1], 2),
            "Q3_pct": round(f_avg[2], 2),
            "Q4_pct": round(f_avg[3], 2),
        }
        if arch in per_sae_unfilt:
            u_avg = [mean(x[i] for x in per_sae_unfilt[arch]) for i in range(4)]
            fr = mean(filter_rate[arch]) if filter_rate.get(arch) else 0.0
            unfilt_summary[arch] = {
                "n_saes": len(per_sae_unfilt[arch]),
                "Q1_pct": round(u_avg[0], 2),
                "Q2_pct": round(u_avg[1], 2),
                "Q3_pct": round(u_avg[2], 2),
                "Q4_pct": round(u_avg[3], 2),
                "pair_filter_rate_pct": round(fr, 2),
            }

        # Significance (filtered): paired Wilcoxon on Q1 - Q234 pair counts per SAE
        diffs = [q1 - q234 for q1, q234 in per_sae_pair_count_filt[arch]]
        sig_summary[arch] = {
            "n_saes": n,
            "cohens_d_filtered": round(cohens_d(diffs), 3),
            "mean_q1_minus_q234_filtered": round(mean(diffs), 2) if diffs else 0.0,
        }

    (out / "redundancy_per_architecture_filtered.json").write_text(
        json.dumps(filt_summary, indent=2)
    )
    (out / "redundancy_per_architecture_unfiltered.json").write_text(
        json.dumps(unfilt_summary, indent=2)
    )
    (out / "redundancy_significance.json").write_text(json.dumps(sig_summary, indent=2))

    # Print
    print(f"=== Filtered per-architecture quantile share (threshold={th}) ===")
    print(
        f"{'Architecture':18s} {'#SAE':>5}  {'Q1%':>7} {'Q2%':>7} {'Q3%':>7} {'Q4%':>7}"
    )
    for a, s in sorted(filt_summary.items()):
        print(
            f"{a:18s} {s['n_saes']:>5}  {s['Q1_pct']:>6.2f}  {s['Q2_pct']:>6.2f}  "
            f"{s['Q3_pct']:>6.2f}  {s['Q4_pct']:>6.2f}"
        )
    print(
        f"\nWrote {out}/redundancy_per_architecture_filtered.json + unfiltered + significance"
    )


if __name__ == "__main__":
    main()
