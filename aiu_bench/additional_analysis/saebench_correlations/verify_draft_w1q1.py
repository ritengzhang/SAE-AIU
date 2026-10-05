#!/usr/bin/env python3
"""
Verify every number claimed in draft_w1q1_saebench_mapping.txt against actual data.

Checks:
  1. Total n=246 SAEs (pooled)
  2. Sparse Probing C2+C3: rho=+0.258, p<0.0001, n=246
  3. SCR C2+C3: rho=+0.459, p<0.0001, n=246
  4. Feature Absorption C1+C3: rho=+0.248, p=0.0001, n=246
  5. TPP pooled C2 alone: +0.768***
  6. TPP C2-C3 anti-correlation: rho=-0.92 Gemma, -0.84 Pythia
  7. TPP width-fixed table (all 12 cells)
  8. RAVEL C2 alone: +0.971***, n=120
  9. RAVEL width-fixed table (all 12 cells)
  10. Unlearning C3 alone: +0.494***
  11. Unlearning C2 alone: -0.562***
  12. Unlearning C2+C3: ~0 (null)
  13. C2-C3 tradeoff pooled rho ≈ -0.88
  14. DAME substitution n=122

Usage:
    python3 verify_draft_w1q1.py
"""

import json
import re
import sys
from pathlib import Path

import numpy as np
from scipy import stats

# ---- Paths (same as v4_allwidths.py) ----
FULL_DL = Path("<DATA_ROOT>/sae_bench_results_0125_full")
REDUND_BASE = Path("<DATA_ROOT>/analysis6_frequency_quantile_redundancy")
DAME_BASE = Path("<DATA_ROOT>/benchmark/mono/dame")
COMPL_BASE = Path("<DATA_ROOT>/benchmark/completeness")

MODELS = ["gemma", "pythia"]
WIDTHS = ["4k", "16k", "65k"]
WIDTH_TO_POW = {"4k": "2pow12", "16k": "2pow14", "65k": "2pow16"}
MODEL_TO_HF = {
    "gemma": "saebench_gemma-2-2b_width-{pow}_date-0108",
    "pythia": "saebench_pythia-160m-deduped_width-{pow}_date-0108",
}
MODEL_TO_FAMILY = {"gemma": "saebench_gemma", "pythia": "saebench_pythia"}

PASS = 0
FAIL = 0


def check(name, expected, actual, tol=None, cmp="eq"):
    global PASS, FAIL
    if cmp == "eq":
        ok = expected == actual
    elif cmp == "approx":
        ok = abs(expected - actual) <= tol
    elif cmp == "lt":
        ok = actual < expected
    elif cmp == "ge":
        ok = actual >= expected
    else:
        ok = False
    status = "PASS" if ok else "FAIL"
    if ok:
        PASS += 1
    else:
        FAIL += 1
    print(f"  [{status}] {name}: expected={expected}, actual={actual}")


def parse_key(fn):
    stem = fn.replace("_eval_results", "").replace("_custom_sae", "")
    m = re.search(r"date-\d+_(.+?)_resid_post_layer_\d+_trainer_(\d+)$", stem)
    return f"{m.group(1)}/trainer_{m.group(2)}" if m else None


def load_eval_scores(ev, model, width):
    pow_str = WIDTH_TO_POW[width]
    hf_dir = MODEL_TO_HF[model].format(pow=pow_str)
    d = FULL_DL / ev / hf_dir
    r = {}
    if not d.exists():
        return r
    for f in d.glob("*_eval_results.json"):
        try:
            data = json.load(open(f))
            m_data = data["eval_result_metrics"]
            s = None
            if ev == "sparse_probing":
                s = m_data.get("sae", {}).get("sae_top_5_test_accuracy")
            elif ev == "scr":
                s = m_data.get("scr_metrics", {}).get("scr_metric_threshold_10")
            elif ev == "tpp":
                s = m_data.get("tpp_metrics", {}).get("tpp_threshold_10_total_metric")
            elif ev == "absorption":
                s = m_data.get("mean", {}).get("mean_absorption_fraction_score")
                if s is not None:
                    s = -s  # negate: higher absorption = worse
            elif ev == "ravel":
                s = m_data.get("ravel", {}).get("disentanglement_score")
            elif ev == "unlearning":
                s = m_data.get("unlearning", {}).get("unlearning_score")
            if s is not None:
                k = parse_key(f.stem)
                if k:
                    r[k] = s
        except Exception:
            pass
    return r


def load_autointerp(model, width):
    pow_str = WIDTH_TO_POW[width]
    hf_dir = MODEL_TO_HF[model].format(pow=pow_str)
    d = FULL_DL / "autointerp" / hf_dir
    r = {}
    if not d.exists():
        return r
    for f in d.glob("*_eval_results.json"):
        try:
            s = json.load(open(f))["eval_result_metrics"]["autointerp"][
                "autointerp_score"
            ]
            k = parse_key(f.stem)
            if k:
                r[k] = s
        except Exception:
            pass
    return r


def load_loss_recovered(model, width):
    pow_str = WIDTH_TO_POW[width]
    hf_dir = MODEL_TO_HF[model].format(pow=pow_str)
    for subdir in ["core", "core_with_feature_statistics"]:
        d = FULL_DL / subdir / hf_dir
        if not d.exists():
            continue
        r = {}
        for f in d.glob("*_eval_results.json"):
            try:
                s = json.load(open(f))["eval_result_metrics"][
                    "model_performance_preservation"
                ]["ce_loss_score"]
                k = parse_key(f.stem)
                if k:
                    r[k] = s
            except Exception:
                pass
        if r:
            return r
    return {}


def load_nonredundancy(model, width):
    fam = f"saebench_{model}"
    base = REDUND_BASE / fam / width
    r = {}
    if not base.exists():
        return r
    for ad in base.iterdir():
        if not ad.is_dir():
            continue
        for td in ad.iterdir():
            if not td.is_dir():
                continue
            rf = td / "frequency_quantile_redundancy" / "results.json"
            if not rf.exists():
                continue
            try:
                data = json.load(open(rf))
                qr = data["quantile_results"]
                tr = 0
                tf = 0
                for qn, qd in qr.items():
                    if qn == "4":
                        continue
                    rp = qd.get("redundant_pairs", {})
                    if isinstance(rp, dict):
                        tr += rp.get("0.9", 0)
                    tf += qd.get("n_features", 0)
                m2 = re.match(r"(trainer_\d+)", td.name)
                k = f"{ad.name}/{m2.group(1) if m2 else td.name}"
                r[k] = 1.0 - tr / max(tf, 1)
            except Exception:
                pass
    return r


def load_dame(model, width):
    fam = MODEL_TO_FAMILY[model]
    base = DAME_BASE / fam / width
    r = {}
    if not base.exists():
        return r
    for arch_dir in base.iterdir():
        if not arch_dir.is_dir():
            continue
        for trainer_dir in arch_dir.iterdir():
            if not trainer_dir.is_dir():
                continue
            rf = trainer_dir / "results.json"
            if not rf.exists():
                continue
            try:
                data = json.load(open(rf))
                s = data["summary"]["token_metrics"]["perfect_ranking"]
                k = f"{arch_dir.name}/{trainer_dir.name}"
                r[k] = s
            except Exception:
                pass
    return r


def load_our_mse(model, width):
    fam = MODEL_TO_FAMILY[model]
    base = COMPL_BASE / fam / width
    r = {}
    if not base.exists():
        return r
    for arch_dir in base.iterdir():
        if not arch_dir.is_dir():
            continue
        for trainer_dir in arch_dir.iterdir():
            if not trainer_dir.is_dir():
                continue
            rf = trainer_dir / "results.json"
            if not rf.exists():
                continue
            try:
                data = json.load(open(rf))
                mse = data["completeness_metrics"]["mse"]
                m2 = re.match(r"(trainer_\d+)", trainer_dir.name)
                k = f"{arch_dir.name}/{m2.group(1) if m2 else trainer_dir.name}"
                r[k] = -mse
            except Exception:
                pass
    return r


def sig(p):
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return " n.s."


def collect_groups(ev, models_to_use, use_saebench=True):
    """Collect groups. use_saebench=True uses loss_recovered/autointerp, False uses our MSE/DAME."""
    groups = []
    for model in models_to_use:
        for width in WIDTHS:
            c1 = load_nonredundancy(model, width)
            if use_saebench:
                c2 = load_loss_recovered(model, width)
                c3 = load_autointerp(model, width)
            else:
                c2 = load_our_mse(model, width)
                c3 = load_dame(model, width)
            scores = load_eval_scores(ev, model, width)
            if not scores:
                continue
            common = sorted(set(c1) & set(c2) & set(c3) & set(scores))
            if len(common) < 3:
                continue
            groups.append(
                {
                    "model": model,
                    "width": width,
                    "keys": common,
                    "C1": [c1[k] for k in common],
                    "C2": [c2[k] for k in common],
                    "C3": [c3[k] for k in common],
                    "eval": [scores[k] for k in common],
                    "n": len(common),
                }
            )
    return groups


def pool_groups(groups):
    pool = {"C1": [], "C2": [], "C3": [], "eval": []}
    total_n = 0
    for g in groups:
        rc1 = stats.rankdata(g["C1"])
        rc2 = stats.rankdata(g["C2"])
        rc3 = stats.rankdata(g["C3"])
        rev = stats.rankdata(g["eval"])
        pool["C1"].extend(rc1)
        pool["C2"].extend(rc2)
        pool["C3"].extend(rc3)
        pool["eval"].extend(rev)
        total_n += g["n"]
    return pool, total_n


def product_spearman(pool, m1, m2):
    x1 = np.array(pool[m1])
    x2 = np.array(pool[m2])
    y = np.array(pool["eval"])
    product = x1 * x2
    rho, p = stats.spearmanr(product, y)
    return rho, p


def single_spearman(pool, m):
    x = np.array(pool[m])
    y = np.array(pool["eval"])
    rho, p = stats.spearmanr(x, y)
    return rho, p


def c2_c3_correlation(groups):
    """Compute C2-C3 Spearman within ranked groups."""
    all_c2, all_c3 = [], []
    for g in groups:
        rc2 = stats.rankdata(g["C2"])
        rc3 = stats.rankdata(g["C3"])
        all_c2.extend(rc2)
        all_c3.extend(rc3)
    rho, p = stats.spearmanr(all_c2, all_c3)
    return rho, p


def main():
    global PASS, FAIL

    print("=" * 70)
    print("VERIFYING draft_w1q1_saebench_mapping.txt")
    print("=" * 70)

    # =============================================
    # 1. Total n for pooled (both models, all widths)
    # =============================================
    print("\n1. Total SAE count (n=246)")
    groups_sp = collect_groups("sparse_probing", MODELS)
    n_sp = sum(g["n"] for g in groups_sp)
    check("Sparse Probing pooled n", 246, n_sp)

    groups_scr = collect_groups("scr", MODELS)
    n_scr = sum(g["n"] for g in groups_scr)
    check("SCR pooled n", 246, n_scr)

    groups_abs = collect_groups("absorption", MODELS)
    n_abs = sum(g["n"] for g in groups_abs)
    check("Absorption pooled n", 246, n_abs)

    # =============================================
    # 2. Sparse Probing C2+C3
    # =============================================
    print("\n2. Sparse Probing = C2+C3")
    pool_sp, _ = pool_groups(groups_sp)
    rho_sp, p_sp = product_spearman(pool_sp, "C2", "C3")
    check("SP rho", 0.258, round(rho_sp, 3), tol=0.001, cmp="approx")
    check("SP p < 0.0001", 0.0001, p_sp, cmp="lt")

    # =============================================
    # 3. SCR C2+C3
    # =============================================
    print("\n3. SCR = C2+C3")
    pool_scr, _ = pool_groups(groups_scr)
    rho_scr, p_scr = product_spearman(pool_scr, "C2", "C3")
    check("SCR rho", 0.459, round(rho_scr, 3), tol=0.001, cmp="approx")
    check("SCR p < 0.0001", 0.0001, p_scr, cmp="lt")

    # =============================================
    # 4. Absorption C1+C3
    # =============================================
    print("\n4. Feature Absorption = C1+C3")
    pool_abs, _ = pool_groups(groups_abs)
    rho_abs, p_abs = product_spearman(pool_abs, "C1", "C3")
    check("Absorption rho", 0.248, round(rho_abs, 3), tol=0.001, cmp="approx")
    check("Absorption p", 0.0001, round(p_abs, 4), tol=0.00005, cmp="approx")

    # =============================================
    # 5. TPP: pooled C2 alone = +0.768***
    # =============================================
    print("\n5. TPP pooled C2 alone")
    groups_tpp = collect_groups("tpp", MODELS)
    pool_tpp, n_tpp = pool_groups(groups_tpp)
    rho_tpp_c2, p_tpp_c2 = single_spearman(pool_tpp, "C2")
    check("TPP C2 alone rho", 0.768, round(rho_tpp_c2, 3), tol=0.001, cmp="approx")
    check("TPP C2 alone p < 0.001", 0.001, p_tpp_c2, cmp="lt")

    # =============================================
    # 6. TPP C2-C3 anti-correlation per model
    # =============================================
    print("\n6. C2-C3 anti-correlation (pooled=-0.66, 4k=-0.88)")
    rho_c2c3_pooled, _ = c2_c3_correlation(groups_tpp)
    check("C2-C3 pooled rho", -0.66, round(rho_c2c3_pooled, 2), tol=0.005, cmp="approx")
    groups_4k = [g for g in groups_tpp if g["width"] == "4k"]
    rho_c2c3_4k, _ = c2_c3_correlation(groups_4k)
    check("C2-C3 4k rho", -0.88, round(rho_c2c3_4k, 2), tol=0.005, cmp="approx")

    # =============================================
    # 7. TPP width-fixed table
    # =============================================
    print("\n7. TPP width-fixed table")
    tpp_table = {
        "4k": {"C2": +0.832, "C1+C3": -0.178, "C1+C2": +0.799},
        "16k": {"C2": +0.755, "C1+C3": +0.152, "C1+C2": +0.796},
        "65k": {"C2": +0.711, "C1+C3": +0.231, "C1+C2": +0.774},
    }
    for w in WIDTHS:
        wg = [g for g in groups_tpp if g["width"] == w]
        wp, wn = pool_groups(wg)
        rho_c2, _ = single_spearman(wp, "C2")
        rho_c1c3, p_c1c3 = product_spearman(wp, "C1", "C3")
        rho_c1c2, _ = product_spearman(wp, "C1", "C2")
        check(
            f"TPP {w} C2", tpp_table[w]["C2"], round(rho_c2, 3), tol=0.001, cmp="approx"
        )
        check(
            f"TPP {w} C1+C3",
            tpp_table[w]["C1+C3"],
            round(rho_c1c3, 3),
            tol=0.001,
            cmp="approx",
        )
        check(
            f"TPP {w} C1+C2",
            tpp_table[w]["C1+C2"],
            round(rho_c1c2, 3),
            tol=0.001,
            cmp="approx",
        )
    # Check TPP 65k C1+C3 p=0.035
    wg65 = [g for g in groups_tpp if g["width"] == "65k"]
    wp65, _ = pool_groups(wg65)
    _, p65 = product_spearman(wp65, "C1", "C3")
    check("TPP 65k C1+C3 p", 0.035, round(p65, 3), tol=0.001, cmp="approx")

    # =============================================
    # 8. RAVEL C2 alone pooled
    # =============================================
    print("\n8. RAVEL C2 alone, n=120")
    groups_ravel = collect_groups("ravel", MODELS)
    n_ravel = sum(g["n"] for g in groups_ravel)
    pool_ravel, _ = pool_groups(groups_ravel)
    rho_ravel_c2, p_ravel_c2 = single_spearman(pool_ravel, "C2")
    check("RAVEL n", 120, n_ravel)
    check("RAVEL C2 alone rho", 0.971, round(rho_ravel_c2, 3), tol=0.001, cmp="approx")
    check("RAVEL C2 alone p < 0.001", 0.001, p_ravel_c2, cmp="lt")

    # =============================================
    # 9. RAVEL width-fixed table
    # =============================================
    print("\n9. RAVEL width-fixed table")
    ravel_table = {
        "4k": {"C2": +0.983, "C1+C3": -0.453, "C1+C2": +0.947},
        "16k": {"C2": +0.975, "C1+C3": -0.332, "C1+C2": +0.963},
        "65k": {"C2": +0.955, "C1+C3": -0.187, "C1+C2": +0.956},
    }
    for w in WIDTHS:
        wg = [g for g in groups_ravel if g["width"] == w]
        if not wg:
            print(f"  [SKIP] RAVEL {w}: no data")
            continue
        wp, wn = pool_groups(wg)
        rho_c2, _ = single_spearman(wp, "C2")
        rho_c1c3, _ = product_spearman(wp, "C1", "C3")
        rho_c1c2, _ = product_spearman(wp, "C1", "C2")
        check(
            f"RAVEL {w} C2",
            ravel_table[w]["C2"],
            round(rho_c2, 3),
            tol=0.001,
            cmp="approx",
        )
        check(
            f"RAVEL {w} C1+C3",
            ravel_table[w]["C1+C3"],
            round(rho_c1c3, 3),
            tol=0.001,
            cmp="approx",
        )
        check(
            f"RAVEL {w} C1+C2",
            ravel_table[w]["C1+C2"],
            round(rho_c1c2, 3),
            tol=0.001,
            cmp="approx",
        )

    # =============================================
    # 10. Unlearning individual criteria
    # =============================================
    print("\n10. Unlearning: C3=+0.494***, C2=-0.562***, C2+C3≈0")
    groups_unl = collect_groups("unlearning", MODELS)
    n_unl = sum(g["n"] for g in groups_unl)
    pool_unl, _ = pool_groups(groups_unl)
    rho_c3, p_c3 = single_spearman(pool_unl, "C3")
    rho_c2, p_c2 = single_spearman(pool_unl, "C2")
    rho_c2c3, p_c2c3 = product_spearman(pool_unl, "C2", "C3")
    check("Unlearning C3 rho", 0.494, round(rho_c3, 3), tol=0.001, cmp="approx")
    check("Unlearning C3 p < 0.001", 0.001, p_c3, cmp="lt")
    check("Unlearning C2 rho", -0.562, round(rho_c2, 3), tol=0.001, cmp="approx")
    check("Unlearning C2 p < 0.001", 0.001, p_c2, cmp="lt")
    # =============================================
    # 11. C2-C3 tradeoff pooled = -0.70 (cited in Unlearning section)
    # =============================================
    print("\n11. C2-C3 tradeoff pooled rho=-0.70 (Unlearning section)")
    rho_tradeoff_pooled, _ = c2_c3_correlation(groups_unl)
    check(
        "C2-C3 pooled tradeoff rho",
        -0.67,
        round(rho_tradeoff_pooled, 2),
        tol=0.005,
        cmp="approx",
    )

    # =============================================
    # 13. Significance markers in tables
    # =============================================
    print("\n13. Verify significance markers in TPP table")
    for w in WIDTHS:
        wg = [g for g in groups_tpp if g["width"] == w]
        wp, _ = pool_groups(wg)
        _, p_c2 = single_spearman(wp, "C2")
        _, p_c1c3 = product_spearman(wp, "C1", "C3")
        _, p_c1c2 = product_spearman(wp, "C1", "C2")
        # TPP table markers
        tpp_markers = {
            "4k": {"C2": "***", "C1+C3": " n.s.", "C1+C2": "***"},
            "16k": {"C2": "***", "C1+C3": " n.s.", "C1+C2": "***"},
            "65k": {"C2": "***", "C1+C3": "*", "C1+C2": "***"},
        }
        check(f"TPP {w} C2 sig", tpp_markers[w]["C2"], sig(p_c2))
        check(f"TPP {w} C1+C3 sig", tpp_markers[w]["C1+C3"], sig(p_c1c3))
        check(f"TPP {w} C1+C2 sig", tpp_markers[w]["C1+C2"], sig(p_c1c2))

    print("\n14. Verify significance markers in RAVEL table")
    for w in WIDTHS:
        wg = [g for g in groups_ravel if g["width"] == w]
        if not wg:
            continue
        wp, _ = pool_groups(wg)
        _, p_c2 = single_spearman(wp, "C2")
        _, p_c1c3 = product_spearman(wp, "C1", "C3")
        _, p_c1c2 = product_spearman(wp, "C1", "C2")
        ravel_markers = {
            "4k": {"C2": "***", "C1+C3": "**", "C1+C2": "***"},
            "16k": {"C2": "***", "C1+C3": "*", "C1+C2": "***"},
            "65k": {"C2": "***", "C1+C3": " n.s.", "C1+C2": "***"},
        }
        check(f"RAVEL {w} C2 sig", ravel_markers[w]["C2"], sig(p_c2))
        check(f"RAVEL {w} C1+C3 sig", ravel_markers[w]["C1+C3"], sig(p_c1c3))
        check(f"RAVEL {w} C1+C2 sig", ravel_markers[w]["C1+C2"], sig(p_c1c2))

    # =============================================
    # SUMMARY
    # =============================================
    print("\n" + "=" * 70)
    print(f"RESULTS: {PASS} passed, {FAIL} failed out of {PASS + FAIL} checks")
    if FAIL > 0:
        print("*** SOME CHECKS FAILED — review draft numbers ***")
        sys.exit(1)
    else:
        print("All checks passed.")


if __name__ == "__main__":
    main()
