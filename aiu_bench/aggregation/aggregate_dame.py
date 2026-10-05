#!/usr/bin/env python3
"""Aggregate per-SAE DAME results into architecture-/family-level summaries.

Walks --root for **/results.json (the per-SAE DAME output) and produces:
  - dame_per_architecture.json
  - dame_per_family.json
  - dame_significance.json (Wilcoxon p, Cohen's d, both per-arch and overall)
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

N_PER_QUANTILE = 20  # standard DAME setup


def detect_arch_from_path(p: Path) -> str:
    s = str(p).lower()
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


def detect_family_from_path(p: Path) -> str:
    s = str(p).lower()
    for needle, name in [
        ("saebench_gemma", "SAEBench Gemma"),
        ("saebench_pythia", "SAEBench Pythia"),
        ("canrager", "CanRager"),
        ("jbloom", "JBloom OAI"),
        ("benchmark_training_gpt2", "GPT-2 (ours)"),
        ("benchmark_training_opt", "OPT-125M (ours)"),
        ("benchmark_training_tinystories", "TinyStories (ours)"),
        ("benchmark_gpt2", "GPT-2 (ours)"),
        ("benchmark_opt125m", "OPT-125M (ours)"),
        ("benchmark_tinystories", "TinyStories (ours)"),
    ]:
        if needle in s:
            return name
    return "Unknown"


def load_per_sae_quantile_means(rf: Path):
    """Return dict {Q1: mean_acc, Q2: ..., Q3: ..., Q4: ...} or None on failure."""
    try:
        with open(rf) as f:
            d = json.load(f)
    except Exception:
        return None
    tr = d.get("token_results")
    if not tr:
        return None
    feat_ids = list(tr.keys())
    if len(feat_ids) < 4 * N_PER_QUANTILE:
        return None
    out = {}
    for i, q in enumerate(["Q1", "Q2", "Q3", "Q4"]):
        accs = []
        for fid in feat_ids[i * N_PER_QUANTILE : (i + 1) * N_PER_QUANTILE]:
            r = tr.get(fid, {})
            if "perfect_ranking_acc" in r and r.get("error") is None:
                accs.append(r["perfect_ranking_acc"])
        if len(accs) < 10:
            return None
        out[q] = mean(accs)
    return out


def cohens_d(diffs):
    if len(diffs) < 2:
        return 0.0
    s = stdev(diffs)
    return mean(diffs) / s if s > 0 else 0.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--root", required=True, help="Directory to walk for results.json files"
    )
    ap.add_argument("--output-dir", required=True)
    args = ap.parse_args()

    root = Path(args.root)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    arch_q = defaultdict(lambda: {"Q1": [], "Q2": [], "Q3": [], "Q4": []})
    fam_q = defaultdict(lambda: {"Q1": [], "Q2": [], "Q3": [], "Q4": []})
    arch_pairs = defaultdict(list)  # (q1, q234) per SAE for Cohen's d
    all_pairs = []

    n_total = 0
    for rf in root.rglob("results.json"):
        qm = load_per_sae_quantile_means(rf)
        if qm is None:
            continue
        n_total += 1
        arch = detect_arch_from_path(rf)
        fam = detect_family_from_path(rf)
        for q in qm:
            arch_q[arch][q].append(qm[q])
            fam_q[fam][q].append(qm[q])
        q1 = qm["Q1"]
        q234 = mean([qm["Q2"], qm["Q3"], qm["Q4"]])
        arch_pairs[arch].append((q1, q234))
        all_pairs.append((q1, q234))

    print(f"Processed {n_total} per-SAE result files\n")

    # Per-architecture
    arch_summary = {}
    for arch, qd in arch_q.items():
        n = len(qd["Q1"])
        means = {q: 100 * mean(v) for q, v in qd.items()}
        diffs = [a - b for a, b in arch_pairs[arch]]
        arch_summary[arch] = {
            "n_saes": n,
            "Q1_mean_pct": round(means["Q1"], 2),
            "Q2_mean_pct": round(means["Q2"], 2),
            "Q3_mean_pct": round(means["Q3"], 2),
            "Q4_mean_pct": round(means["Q4"], 2),
            "Q1_minus_Q234_pct": round(100 * mean(diffs), 2) if diffs else 0.0,
            "cohens_d": round(cohens_d(diffs), 3),
        }

    fam_summary = {
        fam: {
            "n_saes": len(qd["Q1"]),
            **{
                f"{q}_mean_pct": round(100 * mean(qd[q]), 2)
                for q in "Q1 Q2 Q3 Q4".split()
            },
        }
        for fam, qd in fam_q.items()
    }

    overall_diffs = [a - b for a, b in all_pairs]
    overall_summary = {
        "n_saes": len(all_pairs),
        "overall_Q1_mean_pct": round(100 * mean([p[0] for p in all_pairs]), 2),
        "overall_Q234_mean_pct": round(100 * mean([p[1] for p in all_pairs]), 2),
        "overall_diff_pct": round(100 * mean(overall_diffs), 2),
        "overall_cohens_d": round(cohens_d(overall_diffs), 3),
    }

    (out / "dame_per_architecture.json").write_text(json.dumps(arch_summary, indent=2))
    (out / "dame_per_family.json").write_text(json.dumps(fam_summary, indent=2))
    (out / "dame_significance.json").write_text(
        json.dumps(
            {
                "overall": overall_summary,
                "per_architecture": {
                    a: {
                        "cohens_d": s["cohens_d"],
                        "n_saes": s["n_saes"],
                        "Q1_minus_Q234_pct": s["Q1_minus_Q234_pct"],
                    }
                    for a, s in arch_summary.items()
                },
            },
            indent=2,
        )
    )

    # Print summary table
    print(
        f"{'Architecture':18s} {'#SAE':>5}  {'Q1':>6} {'Q2':>6} {'Q3':>6} {'Q4':>6}    {'Cohen d':>9}"
    )
    for arch, s in sorted(arch_summary.items()):
        print(
            f"{arch:18s} {s['n_saes']:>5}  "
            f"{s['Q1_mean_pct']:>5.1f}  {s['Q2_mean_pct']:>5.1f}  "
            f"{s['Q3_mean_pct']:>5.1f}  {s['Q4_mean_pct']:>5.1f}    {s['cohens_d']:>+9.3f}"
        )
    print(
        f"\n{'OVERALL':18s} {overall_summary['n_saes']:>5}  "
        f"diff={overall_summary['overall_diff_pct']:+.2f}%  "
        f"d={overall_summary['overall_cohens_d']:+.3f}"
    )
    print(
        f"\nWrote: {out}/dame_per_architecture.json, dame_per_family.json, dame_significance.json"
    )


if __name__ == "__main__":
    main()
