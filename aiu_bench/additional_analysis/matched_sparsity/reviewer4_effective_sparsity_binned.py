#!/usr/bin/env python3
"""
Reviewer 4 W2/Q1: Fair cross-architecture comparison under fixed effective sparsity.

Bins SAEs by effective L0 (from analysis0_sparsity) into ranges (0-100, 100-200, ...),
then within each bin tests whether the Q1 monosemanticity deficit and Q1 redundancy
concentration are significant — showing the pattern holds even when sparsity is controlled.

Data sources:
  - L0: <DATA_ROOT>/analysis0_sparsity/**/results.json  (effective_active_features_90)
  - Q1 deficit: <DATA_ROOT>/benchmark/mono/dame/**/results.json
  - Redundancy: <DATA_ROOT>/analysis6*/**/dominance_quantile_redundancy/results.json

Output:
  - rebuttal/final_code/results_reviewer4_sparsity_binned.json
"""

import glob
import json
import re
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy import stats

warnings.filterwarnings("ignore")

SPARSITY_ROOT = Path("<DATA_ROOT>/analysis0_sparsity")
DAME_ROOT = Path("<DATA_ROOT>/benchmark/mono/dame")
REDUNDANCY_PATTERNS = [
    "<DATA_ROOT>/analysis6_frequency_quantile_redundancy/**/dominance_quantile_redundancy/results.json",
    "<DATA_ROOT>/analysis6_frequency_quantile_redundancy_no_dec_sim/**/dominance_quantile_redundancy/results.json",
    "<DATA_ROOT>/old/analysis6_frequency_quantile_redundancy/**/dominance_quantile_redundancy/results.json",
]
OUTPUT_FILE = Path(__file__).parent / "results_reviewer4_sparsity_binned.json"

N_PER_QUANTILE = 20
THRESHOLD = "0.7"
L0_BINS = [(0, 100), (100, 200), (200, 300), (300, 500), (500, 1000), (1000, 5000)]


def make_match_key(config_or_meta, path_str):
    """Create a canonical matching key from config/metadata."""
    c = config_or_meta
    pl = path_str.lower()

    if "jbloom" in pl:
        repo = c.get("repo_id", "")
        # Normalize repo: might be full "jbloom/GPT2-..." or just "GPT2-..."
        if "/" in repo:
            repo = repo.split("/", 1)[1]
        layer = str(c.get("layer", ""))
        return ("jbloom", repo, layer)

    if "canrager" in pl:
        arch = c.get("architecture", "")
        layer = str(c.get("layer", ""))
        trainer = c.get("trainer", "")
        tn = re.match(r"trainer_(\d+)", trainer)
        trainer_num = tn.group(1) if tn else trainer
        width = str(c.get("width", c.get("width_name", "")))
        return ("canrager", arch, width, layer, trainer_num)

    if "saebench" in pl:
        sae_type = c.get("sae_type", "")
        width = str(c.get("width", c.get("width_name", "")))
        trainer = c.get("trainer", "")
        tn = re.match(r"trainer_(\d+)", trainer)
        trainer_num = tn.group(1) if tn else trainer
        model = "gemma" if "gemma" in pl else "pythia"
        return ("saebench", model, sae_type, width, trainer_num)

    if "benchmark" in pl:
        combo_id = c.get("combo_id", "")
        if "gpt2" in pl:
            llm = "gpt2"
        elif "opt" in pl:
            llm = "opt125m"
        elif "tiny" in pl or "stories" in pl:
            llm = "tinystories"
        else:
            llm = c.get("llm_name", c.get("llm", "unknown"))
        return ("benchmark", llm, combo_id)

    return None


def detect_architecture(config_or_meta, path_str):
    """Detect SAE architecture from config or path."""
    arch = config_or_meta.get("architecture", "")
    arch_map = {
        "standard": "Standard",
        "gated": "Gated",
        "topk": "TopK",
        "batchtopk": "BatchTopK",
        "jumprelu": "JumpReLU",
        "matryoshka_batchtopk": "Matryoshka",
        "panneal": "PAnneal",
    }
    if arch.lower() in arch_map:
        return arch_map[arch.lower()]

    sae_type = config_or_meta.get("sae_type", "")
    p = (path_str + " " + sae_type).lower()
    if "matryoshka" in p:
        return "Matryoshka"
    if "batchtopk" in p:
        return "BatchTopK"
    if "panneal" in p:
        return "PAnneal"
    if "jumprelu" in p:
        return "JumpReLU"
    if "gatedsae" in p:
        return "Gated"
    if "gated" in p:
        return "Gated"
    if "topk" in p and "batch" not in p:
        return "TopK"
    if "standard" in p or "vanilla" in p:
        return "Standard"
    if "jbloom" in p:
        return "Standard"
    return "unknown"


def extract_meta_from_path(path_str):
    """Extract matching metadata from file path when no config is available."""
    parts = Path(path_str).parts
    meta = {}
    for p in parts:
        if p.startswith("trainer_"):
            meta["trainer"] = p
        if p.startswith("layer_"):
            try:
                meta["layer"] = int(p.replace("layer_", ""))
            except ValueError:
                meta["layer"] = p.replace("layer_", "")
        if p in ("4k", "16k", "32k", "65k", "128k"):
            meta["width"] = p
        if "GPT2-Small-OAI" in p:
            meta["repo_id"] = p
        # SAE type
        for prefix in [
            "TopK_",
            "Standard_",
            "JumpRelu_",
            "JumpReLU_",
            "BatchTopK_",
            "MatryoshkaBatchTopK_",
            "PAnneal_",
            "Gated_",
            "GatedSAE_",
        ]:
            if p.startswith(prefix):
                meta["sae_type"] = p
        if p in ("topk", "vanilla", "standard", "gated", "jumprelu", "batchtopk"):
            meta["architecture"] = p
        # Combo ID
        if re.match(
            r"(standard|gated|topk|batchtopk|jumprelu|matryoshka|panneal)_L\d+", p
        ):
            meta["combo_id"] = p
    return meta


def load_sparsity():
    """Load effective L0 for each SAE."""
    data = {}
    for f in sorted(SPARSITY_ROOT.rglob("results.json")):
        try:
            d = json.load(open(f))
            sm = d.get("sparsity_metrics", {})
            l0 = sm.get("effective_active_features_90")
            if l0 is None:
                continue
            cfg = d.get("config", {})
            key = make_match_key(cfg, str(f))
            if key:
                data[key] = {
                    "l0": l0,
                    "architecture": detect_architecture(cfg, str(f)),
                }
        except Exception:
            continue
    return data


def load_dame():
    """Load Q1 deficit from DAME results."""
    data = {}
    for f in sorted(DAME_ROOT.rglob("results.json")):
        if "aggregation" in str(f):
            continue
        try:
            d = json.load(open(f))
            if "token_results" not in d:
                continue
            tr = d["token_results"]
            feat_ids = list(tr.keys())
            if len(feat_ids) != 4 * N_PER_QUANTILE:
                continue

            q_accs = {}
            for qi, qn in enumerate(["Q1", "Q2", "Q3", "Q4"]):
                s, e = qi * N_PER_QUANTILE, (qi + 1) * N_PER_QUANTILE
                accs = [
                    tr[fid]["perfect_ranking_acc"]
                    for fid in feat_ids[s:e]
                    if tr[fid].get("error") is None
                ]
                q_accs[qn] = accs
            if any(len(v) < 10 for v in q_accs.values()):
                continue

            q1_mean = np.mean(q_accs["Q1"])
            q234_mean = np.mean(q_accs["Q2"] + q_accs["Q3"] + q_accs["Q4"])

            meta = d.get("metadata", {})
            key = make_match_key(meta, str(f))
            if key:
                data[key] = {
                    "q1_mean": q1_mean,
                    "q234_mean": q234_mean,
                    "deficit": q234_mean - q1_mean,
                }
        except Exception:
            continue
    return data


def load_redundancy():
    """Load Q1 redundancy concentration."""
    data = {}
    for pattern in REDUNDANCY_PATTERNS:
        for f in sorted(glob.glob(pattern, recursive=True)):
            try:
                d = json.load(open(f))
                qr = d.get("quantile_results", {})
                if not all(q in qr for q in ["1", "2", "3", "4"]):
                    continue
                q_pairs = {
                    q: qr[q]["redundant_pairs"].get(THRESHOLD, 0)
                    for q in ["1", "2", "3", "4"]
                }
                total = sum(q_pairs.values())
                if total == 0:
                    continue

                meta = extract_meta_from_path(str(f))
                key = make_match_key(meta, str(f))
                if key:
                    data[key] = {
                        "q1_pairs": q_pairs["1"],
                        "q234_mean_pairs": np.mean(
                            [q_pairs["2"], q_pairs["3"], q_pairs["4"]]
                        ),
                        "q1_frac": q_pairs["1"] / total,
                        "total_pairs": total,
                    }
            except Exception:
                continue
    return data


def wilcoxon_greater(values):
    """One-sample Wilcoxon signed-rank test: is median > 0?"""
    values = np.array(values)
    values = values[~np.isnan(values)]
    if len(values) < 5:
        return None, None, len(values)
    try:
        stat, p = stats.wilcoxon(values, alternative="greater")
        return stat, p, len(values)
    except Exception:
        return None, None, len(values)


def main():
    print("=" * 90)
    print("REVIEWER 4 W2/Q1: SPARSITY-BINNED Q1 DEFICIT & REDUNDANCY ANALYSIS")
    print("=" * 90)

    sparsity = load_sparsity()
    dame = load_dame()
    redundancy = load_redundancy()
    print(
        f"\nLoaded: Sparsity={len(sparsity)}, DAME={len(dame)}, Redundancy={len(redundancy)}"
    )

    # Match
    dame_matched = []
    for key in dame:
        if key in sparsity:
            dame_matched.append(
                {
                    **dame[key],
                    "l0": sparsity[key]["l0"],
                    "architecture": sparsity[key]["architecture"],
                    "key": str(key),
                }
            )
    red_matched = []
    for key in redundancy:
        if key in sparsity:
            red_matched.append(
                {
                    **redundancy[key],
                    "l0": sparsity[key]["l0"],
                    "architecture": sparsity[key]["architecture"],
                    "key": str(key),
                }
            )

    print(
        f"Matched: DAME+Sparsity={len(dame_matched)}, Redundancy+Sparsity={len(red_matched)}"
    )

    if len(dame_matched) < 100:
        # Debug: show sample keys
        sp_keys = list(sparsity.keys())[:3]
        dame_keys = list(dame.keys())[:3]
        print(f"\n  Sample sparsity keys: {sp_keys}")
        print(f"  Sample DAME keys:     {dame_keys}")

    # Architecture distribution
    arch_counts = defaultdict(int)
    for s in dame_matched:
        arch_counts[s["architecture"]] += 1
    print(f"\nArchitecture distribution (DAME matched):")
    for a in sorted(arch_counts):
        print(f"  {a}: {arch_counts[a]}")

    # ===== PART 1: Q1 Deficit by L0 Bin =====
    print(f"\n{'=' * 90}")
    print("PART 1: Q1 MONOSEMANTICITY DEFICIT BY EFFECTIVE L0 BIN")
    print(f"{'=' * 90}")
    print(f"  Deficit = mean(Q234 acc) - mean(Q1 acc); positive = Q1 worse")

    print(
        f"\n{'Bin':<15} {'N':>5} {'Mean deficit':>14} {'Med deficit':>13} {'% deficit>0':>13} {'Wilcoxon p':>12}"
    )
    print("-" * 75)

    bin_results_dame = {}
    for lo, hi in L0_BINS:
        bin_saes = [s for s in dame_matched if lo <= s["l0"] < hi]
        if not bin_saes:
            continue
        deficits = [s["deficit"] for s in bin_saes]
        n = len(deficits)
        _, wp, _ = wilcoxon_greater(deficits)
        wp_str = f"{wp:.2e}" if wp is not None else "N/A"
        label = f"L0 {lo}-{hi}"

        arch_bd = defaultdict(list)
        for s in bin_saes:
            arch_bd[s["architecture"]].append(s["deficit"])

        print(
            f"{label:<15} {n:>5} {np.mean(deficits):>14.4f} {np.median(deficits):>13.4f} "
            f"{100 * np.mean([d > 0 for d in deficits]):>12.1f}% {wp_str:>12}"
        )

        bin_results_dame[label] = {
            "n": n,
            "mean_deficit": float(np.mean(deficits)),
            "median_deficit": float(np.median(deficits)),
            "pct_with_deficit": float(100 * np.mean([d > 0 for d in deficits])),
            "wilcoxon_p": float(wp) if wp is not None else None,
            "per_architecture": {
                arch: {
                    "n": len(ds),
                    "mean_deficit": float(np.mean(ds)),
                    "pct_deficit": float(100 * np.mean([d > 0 for d in ds])),
                }
                for arch, ds in arch_bd.items()
            },
        }

    # Per-architecture table
    print(f"\n--- Per-architecture breakdown (mean deficit) ---")
    all_archs = sorted(set(s["architecture"] for s in dame_matched))
    header = f"{'Bin':<15}"
    for a in all_archs:
        header += f" {a:>12}"
    print(header)
    print("-" * (15 + 13 * len(all_archs)))
    for lo, hi in L0_BINS:
        label = f"L0 {lo}-{hi}"
        if label not in bin_results_dame:
            continue
        row = f"{label:<15}"
        for a in all_archs:
            ad = bin_results_dame[label]["per_architecture"].get(a, {})
            n = ad.get("n", 0)
            if n > 0:
                row += f" {ad['mean_deficit']:>+.3f}({n})"
            else:
                row += f" {'--':>12}"
        print(row)

    # ===== PART 2: Q1 Redundancy by L0 Bin =====
    print(f"\n{'=' * 90}")
    print("PART 2: Q1 REDUNDANCY CONCENTRATION BY EFFECTIVE L0 BIN")
    print(f"{'=' * 90}")
    print(f"  Q1 frac = Q1 redundant pairs / total; > 0.25 = concentrated in Q1")

    print(
        f"\n{'Bin':<15} {'N':>5} {'Mean Q1frac':>13} {'Med Q1frac':>12} {'% > 0.25':>10} {'Wilcoxon p':>12}"
    )
    print("-" * 70)

    bin_results_red = {}
    for lo, hi in L0_BINS:
        bin_saes = [s for s in red_matched if lo <= s["l0"] < hi]
        if not bin_saes:
            continue
        fracs = [s["q1_frac"] for s in bin_saes]
        n = len(fracs)
        shifted = [f - 0.25 for f in fracs]
        _, wp, _ = wilcoxon_greater(shifted)
        wp_str = f"{wp:.2e}" if wp is not None else "N/A"
        label = f"L0 {lo}-{hi}"
        print(
            f"{label:<15} {n:>5} {np.mean(fracs):>13.4f} {np.median(fracs):>12.4f} "
            f"{100 * np.mean([f > 0.25 for f in fracs]):>9.1f}% {wp_str:>12}"
        )
        bin_results_red[label] = {
            "n": n,
            "mean_q1_frac": float(np.mean(fracs)),
            "median_q1_frac": float(np.median(fracs)),
            "pct_concentrated": float(100 * np.mean([f > 0.25 for f in fracs])),
            "wilcoxon_p": float(wp) if wp is not None else None,
        }

    # ===== SUMMARY =====
    print(f"\n{'=' * 90}")
    print("SUMMARY")
    print(f"{'=' * 90}")
    sig_dame = [
        b
        for b, v in bin_results_dame.items()
        if v.get("wilcoxon_p") is not None and v["wilcoxon_p"] < 0.05
    ]
    sig_red = [
        b
        for b, v in bin_results_red.items()
        if v.get("wilcoxon_p") is not None and v["wilcoxon_p"] < 0.05
    ]
    print(f"Q1 deficit significant (p<0.05) in bins: {sig_dame}")
    print(f"Q1 redundancy significant (p<0.05) in bins: {sig_red}")

    output = {
        "dame_matched": len(dame_matched),
        "red_matched": len(red_matched),
        "q1_deficit_by_bin": bin_results_dame,
        "q1_redundancy_by_bin": bin_results_red,
        "significant_deficit_bins": sig_dame,
        "significant_redundancy_bins": sig_red,
    }
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
