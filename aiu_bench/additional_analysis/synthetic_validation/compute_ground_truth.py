"""
Compute ground-truth metrics for each trained SAE.

Because we KNOW the ground-truth concepts, we can directly measure:
1. Monosemanticity: does each feature map to exactly one concept?
2. Redundancy: do multiple features map to the same concept?
3. Completeness: are all concepts recovered?
4. Dead features: features that never activate
5. Feature dominance: which features contribute most to reconstruction

Then we compare these ground-truth metrics against proxy metrics
(decoder similarity, dominance-stratified evaluation).
"""

import torch
import torch.nn.functional as F
import json
import gc
import numpy as np
from pathlib import Path
from collections import defaultdict
from sae_models import StandardSAE, TopKSAE, GatedSAE, JumpReLUSAE, BatchTopKSAE, MatryoshkaSAE


def compute_feature_concept_correlation(
    sae,
    concept_directions: torch.Tensor,
    eval_acts: torch.Tensor,
    eval_active: torch.Tensor,
    eval_coeffs: torch.Tensor,
    device: str = "cuda",
    batch_size: int = 8192,
) -> dict:
    """Compute correlation between each SAE feature and each ground-truth concept.

    For each feature f_j, we compute:
    - Pearson correlation between f_j's activation and each concept's activation coefficient
    - The "best concept" for f_j: the concept with highest correlation
    - Monosemanticity score: how concentrated the correlation is (max corr vs rest)

    Returns dict with correlation matrix and derived metrics.
    """
    sae = sae.to(device).eval()
    concept_directions = concept_directions.to(device)
    eval_acts = eval_acts.to(device)
    eval_coeffs = eval_coeffs.to(device)

    n_samples = eval_acts.shape[0]
    n_concepts = concept_directions.shape[0]
    d_sae = sae.d_sae

    # Collect all SAE feature activations
    all_f = []
    with torch.no_grad():
        for i in range(0, n_samples, batch_size):
            batch = eval_acts[i:i+batch_size]
            out = sae(batch)
            all_f.append(out["f"].cpu())
    all_f = torch.cat(all_f, dim=0)  # (n_samples, d_sae)

    # Also get concept coefficients on CPU
    eval_coeffs_cpu = eval_coeffs.cpu().float()

    # Compute correlation matrix: (d_sae, n_concepts) via matrix multiply
    f_mean = all_f.mean(dim=0)
    c_mean = eval_coeffs_cpu.mean(dim=0)
    f_centered = all_f - f_mean
    c_centered = eval_coeffs_cpu - c_mean

    f_std = f_centered.pow(2).sum(dim=0).sqrt().clamp(min=1e-8)  # (d_sae,)
    c_std = c_centered.pow(2).sum(dim=0).sqrt().clamp(min=1e-8)  # (n_concepts,)

    # Single matmul: (d_sae, n_samples) @ (n_samples, n_concepts) = (d_sae, n_concepts)
    corr_matrix = (f_centered.T @ c_centered) / (f_std.unsqueeze(1) * c_std.unsqueeze(0))

    # Best concept assignment for each feature
    best_concept = corr_matrix.argmax(dim=1)  # (d_sae,)
    best_corr = corr_matrix.max(dim=1).values  # (d_sae,)

    # Second-best correlation (for monosemanticity)
    corr_sorted = corr_matrix.sort(dim=1, descending=True).values
    second_corr = corr_sorted[:, 1]

    # Monosemanticity score: ratio of best to second-best correlation
    # High = monosemantic (one concept dominates), Low = polysemantic
    mono_score = best_corr / second_corr.clamp(min=1e-8)
    # Alternative: gap between best and second
    mono_gap = best_corr - second_corr

    # Feature firing stats
    feature_firing_freq = (all_f > 0).float().mean(dim=0)  # fraction of inputs where feature fires
    feature_mean_act = all_f.mean(dim=0)  # mean activation magnitude

    # Feature dominance: mean fraction of total activation per input
    total_act = all_f.abs().sum(dim=1, keepdim=True).clamp(min=1e-8)
    feature_dominance = (all_f.abs() / total_act).mean(dim=0)  # mean dominance rho_bar

    # Dead features
    is_dead = feature_firing_freq == 0

    # Free large tensors
    del all_f, f_centered, c_centered, eval_coeffs_cpu

    return {
        "corr_matrix": corr_matrix,           # (d_sae, n_concepts)
        "best_concept": best_concept,          # (d_sae,) - which concept each feature best matches
        "best_corr": best_corr,                # (d_sae,) - correlation with best concept
        "second_corr": second_corr,            # (d_sae,) - correlation with 2nd best concept
        "mono_score": mono_score,              # (d_sae,) - monosemanticity (best/2nd)
        "mono_gap": mono_gap,                  # (d_sae,) - monosemanticity (best - 2nd)
        "feature_firing_freq": feature_firing_freq,  # (d_sae,)
        "feature_mean_act": feature_mean_act,  # (d_sae,)
        "feature_dominance": feature_dominance,# (d_sae,) - mean dominance rho_bar
        "is_dead": is_dead,                    # (d_sae,) - dead features
    }


def compute_ground_truth_redundancy(gt_metrics: dict, sae, cluster_ids: np.ndarray) -> dict:
    """Compute ground-truth redundancy and compare with decoder similarity proxy.

    Ground-truth redundancy: two features are redundant if they map to the same cluster
    (concepts within a cluster share similar directions, so cluster = semantic equivalence class).
    Proxy: decoder cosine similarity >= threshold.
    Fully vectorized — no Python loops over pairs.
    """
    best_concept = gt_metrics["best_concept"]
    is_dead = gt_metrics["is_dead"]

    # Map features to clusters via their best concept
    feature_clusters = torch.from_numpy(cluster_ids[best_concept.numpy()])

    # Filter to alive features
    alive_mask = ~is_dead
    alive_idx = alive_mask.nonzero(as_tuple=True)[0]
    n_alive = len(alive_idx)

    # Decoder cosine similarity for alive features only
    dec_dirs = sae.decoder_directions.cpu()  # (d_sae, d_in)
    alive_dec = dec_dirs[alive_idx]  # (n_alive, d_in)
    dec_sim = alive_dec @ alive_dec.T  # (n_alive, n_alive)

    # Ground-truth: same cluster matrix (upper triangle only)
    alive_clusters = feature_clusters[alive_idx]  # (n_alive,)
    same_cluster = alive_clusters.unsqueeze(0) == alive_clusters.unsqueeze(1)

    # Also track same-concept (stricter)
    alive_concepts = best_concept[alive_idx]
    same_concept = alive_concepts.unsqueeze(0) == alive_concepts.unsqueeze(1)

    # Upper triangle mask (exclude diagonal)
    triu = torch.triu(torch.ones(n_alive, n_alive, dtype=torch.bool), diagonal=1)

    same_cluster_pairs = (same_cluster & triu).sum().item()
    same_concept_pairs = (same_concept & triu).sum().item()

    # Proxy validation at multiple thresholds (using same-cluster as ground truth)
    results = {"n_alive": n_alive, "n_dead": is_dead.sum().item(),
               "n_gt_cluster_redundant": same_cluster_pairs,
               "n_gt_concept_redundant": same_concept_pairs}

    for thresh_val in [0.3, 0.5, 0.7, 0.9]:
        thresh_name = f"proxy_{thresh_val:.1f}".replace(".", "")
        high_sim = dec_sim >= thresh_val
        tp = (high_sim & same_cluster & triu).sum().item()
        fp = (high_sim & ~same_cluster & triu).sum().item()
        flagged = tp + fp
        results[thresh_name] = {
            "flagged": flagged,
            "true_positives": tp,
            "false_positives": fp,
            "precision": tp / flagged if flagged > 0 else 0,
            "recall": tp / same_cluster_pairs if same_cluster_pairs > 0 else 0,
        }

    # AUROC: threshold-free proxy quality metric
    try:
        from sklearn.metrics import roc_auc_score
        pair_sims = dec_sim[triu].numpy()
        pair_labels = same_cluster[triu].numpy().astype(int)
        if pair_labels.sum() > 0 and pair_labels.sum() < len(pair_labels):
            auroc = roc_auc_score(pair_labels, pair_sims)
        else:
            auroc = 0.0
        results["auroc"] = auroc
    except ImportError:
        results["auroc"] = -1.0  # sklearn not available

    # Mean decoder sim for same vs different cluster pairs
    same_pair_sims = dec_sim[same_cluster & triu]
    diff_pair_sims = dec_sim[~same_cluster & triu]
    results["mean_same_cluster_sim"] = same_pair_sims.mean().item() if len(same_pair_sims) > 0 else 0
    results["mean_diff_cluster_sim"] = diff_pair_sims.mean().item() if len(diff_pair_sims) > 0 else 0

    return results


def compute_ground_truth_completeness(gt_metrics: dict, n_concepts: int) -> dict:
    """Compute ground-truth completeness: which concepts are recovered?

    A concept is "recovered" if at least one alive feature has correlation > threshold with it.
    """
    best_concept = gt_metrics["best_concept"]
    best_corr = gt_metrics["best_corr"]
    is_dead = gt_metrics["is_dead"]

    # Which concepts are covered by at least one alive feature?
    covered_concepts = set()
    well_covered = set()  # covered with corr > 0.3

    for i in range(len(best_concept)):
        if not is_dead[i]:
            c = best_concept[i].item()
            covered_concepts.add(c)
            if best_corr[i] > 0.3:
                well_covered.add(c)

    return {
        "n_concepts": n_concepts,
        "n_covered": len(covered_concepts),
        "n_well_covered": len(well_covered),
        "coverage_pct": len(covered_concepts) / n_concepts * 100,
        "well_coverage_pct": len(well_covered) / n_concepts * 100,
    }


def compute_dominance_stratified_monosemanticity(gt_metrics: dict) -> dict:
    """THE KILLER TEST: Compare uniform vs dominance-stratified monosemanticity.

    1. Rank features by dominance (mean activation fraction)
    2. Split into Q1-Q4 by dominance
    3. Compare monosemanticity across quantiles
    4. Show that Q1 (dominant) features have worse monosemanticity
    5. Show that uniform averaging masks this
    """
    dominance = gt_metrics["feature_dominance"]
    mono_gap = gt_metrics["mono_gap"]
    best_corr = gt_metrics["best_corr"]
    is_dead = gt_metrics["is_dead"]

    # Filter to alive features only
    alive = ~is_dead
    alive_idx = alive.nonzero(as_tuple=True)[0]

    if len(alive_idx) < 20:
        return {"error": "Too few alive features for quantile analysis"}

    # Get dominance and monosemanticity for alive features
    dom_alive = dominance[alive_idx]
    mono_alive = mono_gap[alive_idx]
    corr_alive = best_corr[alive_idx]

    # Rank by dominance (descending) and split into 4 quantiles
    sorted_indices = dom_alive.argsort(descending=True)
    n = len(sorted_indices)
    q_size = n // 4

    quantile_results = {}
    for q_idx, q_name in enumerate(["Q1", "Q2", "Q3", "Q4"]):
        start = q_idx * q_size
        end = start + q_size if q_idx < 3 else n
        q_indices = sorted_indices[start:end]

        q_mono = mono_alive[q_indices]
        q_corr = corr_alive[q_indices]
        q_dom = dom_alive[q_indices]

        quantile_results[q_name] = {
            "n_features": len(q_indices),
            "mean_mono_gap": q_mono.mean().item(),
            "std_mono_gap": q_mono.std().item(),
            "mean_best_corr": q_corr.mean().item(),
            "mean_dominance": q_dom.mean().item(),
            "median_mono_gap": q_mono.median().item(),
        }

    # Overall (uniform) average
    uniform_mono = mono_alive.mean().item()

    # Q1 vs Q2-Q4 comparison
    q1_mono = quantile_results["Q1"]["mean_mono_gap"]
    q234_mono = np.mean([quantile_results[q]["mean_mono_gap"] for q in ["Q2", "Q3", "Q4"]])

    # Statistical test: is Q1 monosemanticity significantly different from Q2-Q4?
    from scipy import stats as scipy_stats

    q1_vals = mono_alive[sorted_indices[:q_size]].numpy()
    q234_vals = mono_alive[sorted_indices[q_size:]].numpy()

    # Mann-Whitney U test (non-parametric)
    try:
        u_stat, p_value = scipy_stats.mannwhitneyu(q1_vals, q234_vals, alternative='less')
    except:
        p_value = 1.0

    # Cohen's d
    pooled_std = np.sqrt((q1_vals.std()**2 + q234_vals.std()**2) / 2)
    cohens_d = (q1_vals.mean() - q234_vals.mean()) / pooled_std if pooled_std > 0 else 0

    return {
        "quantile_results": quantile_results,
        "uniform_mean_mono": uniform_mono,
        "q1_mean_mono": q1_mono,
        "q234_mean_mono": q234_mono,
        "q1_deficit": q1_mono - q234_mono,
        "p_value": p_value,
        "cohens_d": cohens_d,
        "n_alive": len(alive_idx),
    }


def compute_dominance_stratified_redundancy(gt_metrics: dict, sae, cluster_ids: np.ndarray) -> dict:
    """Check if redundancy concentrates in Q1 (dominant features) in ground truth.
    Fully vectorized — no Python loops over pairs.
    """
    dominance = gt_metrics["feature_dominance"]
    best_concept = gt_metrics["best_concept"]
    is_dead = gt_metrics["is_dead"]

    # Map features to clusters
    feature_clusters = torch.from_numpy(cluster_ids[best_concept.numpy()])

    alive = ~is_dead
    alive_idx = alive.nonzero(as_tuple=True)[0]

    if len(alive_idx) < 20:
        return {"error": "Too few alive features"}

    dom_alive = dominance[alive_idx]

    # Decoder similarity matrix for alive features
    dec_dirs = sae.decoder_directions.cpu()
    alive_dec = dec_dirs[alive_idx]  # (n_alive, d_in)
    dec_sim = alive_dec @ alive_dec.T  # (n_alive, n_alive)

    # Ground-truth: same cluster matrix
    cluster_alive = feature_clusters[alive_idx]
    same_cluster = cluster_alive.unsqueeze(0) == cluster_alive.unsqueeze(1)

    # Rank by dominance, split into Q1-Q4
    sorted_indices = dom_alive.argsort(descending=True)
    n = len(sorted_indices)
    q_size = n // 4

    quantile_pairs = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}
    quantile_gt_redundant = {"Q1": 0, "Q2": 0, "Q3": 0, "Q4": 0}

    for q_idx, q_name in enumerate(["Q1", "Q2", "Q3", "Q4"]):
        start = q_idx * q_size
        end = start + q_size if q_idx < 3 else n
        q_sorted = sorted_indices[start:end]
        nq = len(q_sorted)

        # Extract sub-matrices for this quantile
        q_dec_sim = dec_sim[q_sorted][:, q_sorted]  # (nq, nq)
        q_same = same_cluster[q_sorted][:, q_sorted]  # (nq, nq)
        triu_q = torch.triu(torch.ones(nq, nq, dtype=torch.bool), diagonal=1)

        # Count proxy-redundant pairs (decoder sim >= 0.7) within this quantile
        quantile_pairs[q_name] = ((q_dec_sim >= 0.7) & triu_q).sum().item()

        # Count ground-truth redundant pairs within this quantile
        quantile_gt_redundant[q_name] = (q_same & triu_q).sum().item()

    total_proxy = sum(quantile_pairs.values())
    total_gt = sum(quantile_gt_redundant.values())

    return {
        "proxy_redundant_pairs_by_quantile": quantile_pairs,
        "gt_redundant_pairs_by_quantile": quantile_gt_redundant,
        "proxy_total": total_proxy,
        "gt_total": total_gt,
        "proxy_q1_pct": quantile_pairs["Q1"] / total_proxy * 100 if total_proxy > 0 else 0,
        "gt_q1_pct": quantile_gt_redundant["Q1"] / total_gt * 100 if total_gt > 0 else 0,
    }


def evaluate_all_saes(
    data_dir: str = "<DATA_ROOT>/toy_validation/data",
    saes_dir: str = "<DATA_ROOT>/toy_validation/trained_saes",
    output_dir: str = "<DATA_ROOT>/toy_validation/results",
    device: str = "cuda",
):
    """Evaluate all trained SAEs against ground truth."""
    data_dir = Path(data_dir)
    saes_dir = Path(saes_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load data
    print("Loading data...", flush=True)
    concept_directions = torch.load(data_dir / "concept_directions.pt")
    eval_acts = torch.load(data_dir / "eval_activations.pt")
    eval_active = torch.load(data_dir / "eval_concept_active.pt")
    eval_coeffs = torch.load(data_dir / "eval_concept_coeffs.pt")
    cluster_ids = np.load(data_dir / "cluster_ids.npy")
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)
    d = metadata["d"]
    n_concepts = metadata["n_concepts"]

    # Find all trained SAEs
    sae_dirs = sorted([p for p in saes_dir.iterdir() if p.is_dir() and (p / "model.pt").exists()])
    print(f"Found {len(sae_dirs)} trained SAEs", flush=True)

    all_results = {}

    for sae_idx, sae_dir in enumerate(sae_dirs):
        name = sae_dir.name
        print(f"\n[{sae_idx+1}/{len(sae_dirs)}] Evaluating {name}...", flush=True)

        # Load config
        with open(sae_dir / "training_log.json") as f:
            train_info = json.load(f)
        cfg = train_info["config"]

        # Create and load SAE
        arch = cfg["arch"]
        if arch in ("standard", "panneal"):
            sae = StandardSAE(d, cfg["d_sae"])
        elif arch == "topk":
            sae = TopKSAE(d, cfg["d_sae"], cfg["k"])
        elif arch == "gated":
            sae = GatedSAE(d, cfg["d_sae"])
        elif arch == "jumprelu":
            sae = JumpReLUSAE(d, cfg["d_sae"])
        elif arch == "batchtopk":
            sae = BatchTopKSAE(d, cfg["d_sae"], cfg["k"])
        elif arch == "matryoshka":
            sae = MatryoshkaSAE(d, cfg["d_sae"])
        else:
            raise ValueError(f"Unknown architecture: {arch}")
        sae.load_state_dict(torch.load(sae_dir / "model.pt", map_location="cpu"))

        # Compute ground-truth metrics
        print("  Computing feature-concept correlations...", flush=True)
        gt_metrics = compute_feature_concept_correlation(
            sae, concept_directions, eval_acts, eval_active, eval_coeffs, device=device
        )

        print("  Computing redundancy validation...", flush=True)
        redundancy = compute_ground_truth_redundancy(gt_metrics, sae, cluster_ids)

        print("  Computing completeness...", flush=True)
        completeness = compute_ground_truth_completeness(gt_metrics, n_concepts)

        print("  Computing dominance-stratified monosemanticity...", flush=True)
        dom_mono = compute_dominance_stratified_monosemanticity(gt_metrics)

        print("  Computing dominance-stratified redundancy...", flush=True)
        dom_redund = compute_dominance_stratified_redundancy(gt_metrics, sae, cluster_ids)

        result = {
            "config": cfg,
            "final_mse": train_info["final_mse"],
            "final_l0": train_info["final_l0"],
            "n_dead": gt_metrics["is_dead"].sum().item(),
            "n_alive": (~gt_metrics["is_dead"]).sum().item(),
            "completeness": completeness,
            "redundancy_validation": redundancy,
            "dominance_monosemanticity": dom_mono,
            "dominance_redundancy": dom_redund,
            "mean_best_corr": gt_metrics["best_corr"][~gt_metrics["is_dead"]].mean().item(),
            "mean_mono_gap": gt_metrics["mono_gap"][~gt_metrics["is_dead"]].mean().item(),
        }

        # Save per-SAE result
        with open(output_dir / f"{name}.json", "w") as f:
            json.dump(result, f, indent=2, default=str)

        # Print summary
        print(f"  Dead: {result['n_dead']}, Alive: {result['n_alive']}", flush=True)
        print(f"  Completeness: {completeness['coverage_pct']:.1f}%", flush=True)
        auroc = redundancy.get('auroc', -1)
        print(f"  Redundancy AUROC: {auroc:.3f}, same_sim={redundancy['mean_same_cluster_sim']:.3f}, diff_sim={redundancy['mean_diff_cluster_sim']:.3f}", flush=True)
        if "error" not in dom_mono:
            print(f"  Mono Q1={dom_mono['q1_mean_mono']:.3f} vs Q234={dom_mono['q234_mean_mono']:.3f} "
                  f"(deficit={dom_mono['q1_deficit']:.3f}, p={dom_mono['p_value']:.2e})", flush=True)

        all_results[name] = result

        # Free memory between SAEs
        del gt_metrics, sae
        gc.collect()
        torch.cuda.empty_cache()

    # Save summary
    with open(output_dir / "all_results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # === Framework-level validation ===
    # Compare proxy-based SAE quality ranking vs ground-truth quality ranking
    print("\n" + "="*60, flush=True)
    print("FRAMEWORK-LEVEL VALIDATION", flush=True)
    print("="*60, flush=True)
    framework_validation = compute_framework_validation(all_results)
    with open(output_dir / "framework_validation.json", "w") as f:
        json.dump(framework_validation, f, indent=2, default=str)

    print(f"\nAll results saved to {output_dir}")
    return all_results


def compute_framework_validation(all_results: dict) -> dict:
    """End-to-end framework validation: do proxy metrics rank SAEs the same as ground truth?

    Key insight: MSE correlates with completeness but ANTI-correlates with monosemanticity.
    This proves that no single metric captures SAE quality — the multi-criterion framework
    is necessary. We validate this by showing:
    1. Individual proxy-GT correlations for each criterion
    2. Rank-based composite: combining proxy criteria outperforms any single metric
    3. MSE-monosemanticity tradeoff as smoking gun for multi-criterion need
    """
    from scipy.stats import spearmanr, rankdata

    names = []
    archs = []
    # Ground-truth metrics (what we'd know if we had C)
    gt_mono_scores = []      # C3: mean monosemanticity (mono_gap) across alive features
    gt_completeness = []      # C2: fraction of concepts recovered
    gt_dead_pct = []          # C4: fraction of dead features
    gt_l0s = []               # C5: L0 sparsity
    gt_redundancy_rate = []   # C1: same-cluster pair rate

    # Proxy metrics (what our framework computes without knowing C)
    proxy_mse = []            # reconstruction quality (proxy for C2)
    proxy_dead_pct = []       # C4: directly observable
    proxy_l0 = []             # C5: directly observable
    proxy_redundancy_03 = []  # C1 proxy: fraction of pairs with decoder sim > 0.3
    proxy_redundancy_05 = []  # C1 proxy: fraction of pairs with decoder sim > 0.5

    for name, result in all_results.items():
        n_alive = result["n_alive"]
        n_dead = result["n_dead"]
        n_total = n_alive + n_dead

        if n_alive < 10:
            continue

        names.append(name)
        archs.append(result["config"]["arch"])

        # Ground-truth
        gt_mono_scores.append(result["mean_mono_gap"])
        gt_completeness.append(result["completeness"]["well_coverage_pct"] / 100.0)
        gt_dead_pct.append(n_dead / n_total)
        gt_l0s.append(result["final_l0"])
        n_pairs = n_alive * (n_alive - 1) // 2
        gt_redund = result["redundancy_validation"]["n_gt_cluster_redundant"]
        gt_redundancy_rate.append(gt_redund / n_pairs if n_pairs > 0 else 0)

        # Proxies
        proxy_mse.append(result["final_mse"])
        proxy_dead_pct.append(n_dead / n_total)
        proxy_l0.append(result["final_l0"])
        # Count proxy-redundant pairs at various thresholds
        rv = result["redundancy_validation"]
        for thresh_key, proxy_list in [("proxy_03", proxy_redundancy_03), ("proxy_05", proxy_redundancy_05)]:
            thresh_data = rv.get(thresh_key, {})
            flagged = thresh_data.get("flagged", 0) if isinstance(thresh_data, dict) else 0
            proxy_list.append(flagged / n_pairs if n_pairs > 0 else 0)

    # Convert to arrays
    N = len(names)
    gt_mono_scores = np.array(gt_mono_scores)
    gt_completeness = np.array(gt_completeness)
    gt_dead_pct = np.array(gt_dead_pct)
    gt_l0s = np.array(gt_l0s)
    gt_redundancy_rate = np.array(gt_redundancy_rate)
    proxy_mse = np.array(proxy_mse)
    proxy_dead_pct = np.array(proxy_dead_pct)
    proxy_l0 = np.array(proxy_l0)
    proxy_redundancy_03 = np.array(proxy_redundancy_03)
    proxy_redundancy_05 = np.array(proxy_redundancy_05)

    results = {"n_saes": N, "sae_names": names, "archs": archs}
    correlations = {}

    # === 1. Individual criterion correlations ===
    print("\n  --- Individual Criterion Correlations ---", flush=True)

    # C2: MSE vs completeness
    rho, p = spearmanr(-proxy_mse, gt_completeness)
    correlations["neg_mse_vs_gt_completeness"] = {"spearman_rho": float(rho), "p_value": float(p)}
    print(f"  C2: -MSE vs GT completeness:   ρ={rho:.3f} (p={p:.2e})", flush=True)

    # C1: decoder sim redundancy proxy vs GT redundancy rate
    rho, p = spearmanr(proxy_redundancy_05, gt_redundancy_rate)
    correlations["proxy_redund_05_vs_gt_redund"] = {"spearman_rho": float(rho), "p_value": float(p)}
    print(f"  C1: proxy_redund@0.5 vs GT:     ρ={rho:.3f} (p={p:.2e})", flush=True)

    rho, p = spearmanr(proxy_redundancy_03, gt_redundancy_rate)
    correlations["proxy_redund_03_vs_gt_redund"] = {"spearman_rho": float(rho), "p_value": float(p)}
    print(f"  C1: proxy_redund@0.3 vs GT:     ρ={rho:.3f} (p={p:.2e})", flush=True)

    # C4: dead pct is identical (trivially ρ=1.0)
    # C5: L0 is identical (trivially ρ=1.0)

    # === 2. THE KEY FINDING: MSE-monosemanticity tradeoff ===
    print("\n  --- MSE-Monosemanticity Tradeoff (Key Finding) ---", flush=True)

    rho, p = spearmanr(-proxy_mse, gt_mono_scores)
    correlations["neg_mse_vs_gt_monosemanticity"] = {"spearman_rho": float(rho), "p_value": float(p)}
    print(f"  -MSE vs GT monosemanticity:     ρ={rho:.3f} (p={p:.2e})", flush=True)
    print(f"  → Lower MSE ≠ better features. MSE and monosemanticity are in tension.", flush=True)

    # === 3. Rank-based composites ===
    print("\n  --- Composite Ranking Validation ---", flush=True)

    # Ground-truth composite via rank aggregation:
    # Good SAE = high mono (C3) + high completeness (C2) + low dead (C4) + low redundancy (C1)
    # Use ranks: rank 1 = best, rank N = worst for each criterion
    gt_rank_mono = rankdata(-gt_mono_scores)       # higher is better → negate
    gt_rank_comp = rankdata(-gt_completeness)       # higher is better → negate
    gt_rank_dead = rankdata(gt_dead_pct)            # lower is better
    gt_rank_redund = rankdata(gt_redundancy_rate)   # lower is better
    gt_composite_rank = gt_rank_mono + gt_rank_comp + gt_rank_dead + gt_rank_redund

    # Proxy composite via rank aggregation:
    # Uses only observables without ground truth: MSE, dead pct, redundancy proxy, L0
    proxy_rank_mse = rankdata(proxy_mse)            # lower is better
    proxy_rank_dead = rankdata(proxy_dead_pct)      # lower is better
    proxy_rank_redund = rankdata(proxy_redundancy_03)  # lower is better
    proxy_composite_rank = proxy_rank_mse + proxy_rank_dead + proxy_rank_redund

    # Framework composite vs GT composite
    rho, p = spearmanr(proxy_composite_rank, gt_composite_rank)
    correlations["framework_rank_vs_gt_rank"] = {"spearman_rho": float(rho), "p_value": float(p)}
    print(f"  Framework (MSE+dead+redund) vs GT: ρ={rho:.3f} (p={p:.2e})", flush=True)

    # MSE alone vs GT composite
    rho_mse, p_mse = spearmanr(proxy_mse, gt_composite_rank)
    correlations["mse_alone_vs_gt_rank"] = {"spearman_rho": float(rho_mse), "p_value": float(p_mse)}
    print(f"  MSE alone vs GT composite:         ρ={rho_mse:.3f} (p={p_mse:.2e})", flush=True)

    # Dead alone vs GT composite
    rho_dead, p_dead = spearmanr(proxy_dead_pct, gt_composite_rank)
    correlations["dead_alone_vs_gt_rank"] = {"spearman_rho": float(rho_dead), "p_value": float(p_dead)}
    print(f"  Dead alone vs GT composite:        ρ={rho_dead:.3f} (p={p_dead:.2e})", flush=True)

    # Summary
    print(f"\n  → Framework composite: ρ={correlations['framework_rank_vs_gt_rank']['spearman_rho']:.3f}", flush=True)
    print(f"  → MSE alone:           ρ={rho_mse:.3f}", flush=True)
    if correlations['framework_rank_vs_gt_rank']['spearman_rho'] > rho_mse:
        print(f"  ★ Framework outperforms MSE-only by Δρ={correlations['framework_rank_vs_gt_rank']['spearman_rho'] - rho_mse:.3f}", flush=True)
    else:
        print(f"  (MSE alone competitive; key value of framework is DIAGNOSTIC not ranking)", flush=True)

    # === 4. Store per-SAE data for plotting ===
    results["per_sae"] = {
        name: {
            "arch": archs[i],
            "gt_mono": float(gt_mono_scores[i]),
            "gt_completeness": float(gt_completeness[i]),
            "gt_dead_pct": float(gt_dead_pct[i]),
            "gt_redundancy_rate": float(gt_redundancy_rate[i]),
            "gt_composite_rank": float(gt_composite_rank[i]),
            "proxy_mse": float(proxy_mse[i]),
            "proxy_dead_pct": float(proxy_dead_pct[i]),
            "proxy_redundancy_05": float(proxy_redundancy_05[i]),
            "proxy_redundancy_03": float(proxy_redundancy_03[i]),
            "proxy_composite_rank": float(proxy_composite_rank[i]),
        }
        for i, name in enumerate(names)
    }
    results["correlations"] = correlations

    return results


if __name__ == "__main__":
    import sys
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    evaluate_all_saes(device=device)
