#!/usr/bin/env python3
"""
Aggregate all completeness v2 results and compute correlations between metrics.

Loads results_v2_completeness.json from all 7 SAE families and computes
Spearman/Pearson correlations between MSE and other metrics (R², cosine similarity,
CE loss score, KL div score).

Output:
  - rebuttal_code/results_completeness_v2_correlations.json
  - Console table of correlations
"""

import json
import glob
import numpy as np
from pathlib import Path
from scipy import stats

BASE_DIR = Path("<DATA_ROOT>/sae/benchmark/completeness")
OUTPUT_FILE = Path("<REPO_ROOT>/rebuttal_code/results_completeness_v2_correlations.json")

# SAE family configurations
FAMILIES = {
    'saebench_gemma': 'SAEBench Gemma 2B',
    'saebench_pythia': 'SAEBench Pythia 160M',
    'canrager': 'canrager TopK+Vanilla (Gemma 2B)',
    'jbloom_oai_v5': 'jbloom OAI-v5 (GPT-2 Small)',
    'benchmark_training_gpt2': 'Benchmark Training GPT-2',
    'benchmark_training_opt125m': 'Benchmark Training OPT-125M',
    'benchmark_training_tinystories': 'Benchmark Training TinyStories',
}

METRICS = ['mse', 'explained_variance', 'cossim', 'ce_loss_score', 'kl_div_score']


def load_all_results():
    """Load all results_v2_completeness.json files."""
    all_results = []
    family_counts = {}

    for family_key, family_name in FAMILIES.items():
        pattern = str(BASE_DIR / family_key / "**" / "results_v2_completeness.json")
        files = glob.glob(pattern, recursive=True)
        family_counts[family_key] = len(files)

        for f in files:
            try:
                data = json.load(open(f))
                metrics = data.get('completeness_metrics', {})
                config = data.get('config', {})

                entry = {
                    'family': family_key,
                    'family_name': family_name,
                    'path': f,
                    'model': config.get('model_name', 'unknown'),
                }
                for m in METRICS:
                    entry[m] = metrics.get(m)

                all_results.append(entry)
            except (json.JSONDecodeError, Exception) as e:
                print(f"  Warning: Failed to load {f}: {e}")

    return all_results, family_counts


def compute_correlations(results, metric_pairs=None):
    """Compute Spearman and Pearson correlations between metric pairs."""
    if metric_pairs is None:
        # Default: MSE vs all others
        metric_pairs = [
            ('mse', 'explained_variance'),
            ('mse', 'cossim'),
            ('mse', 'ce_loss_score'),
            ('mse', 'kl_div_score'),
            ('explained_variance', 'cossim'),
            ('explained_variance', 'ce_loss_score'),
            ('explained_variance', 'kl_div_score'),
            ('cossim', 'ce_loss_score'),
            ('cossim', 'kl_div_score'),
            ('ce_loss_score', 'kl_div_score'),
        ]

    correlations = {}
    for m1, m2 in metric_pairs:
        # Filter to results that have both metrics
        valid = [(r[m1], r[m2]) for r in results
                 if r.get(m1) is not None and r.get(m2) is not None]

        if len(valid) < 3:
            correlations[f'{m1}_vs_{m2}'] = {
                'n': len(valid),
                'spearman_r': None,
                'spearman_p': None,
                'pearson_r': None,
                'pearson_p': None,
            }
            continue

        x = np.array([v[0] for v in valid])
        y = np.array([v[1] for v in valid])

        spearman_r, spearman_p = stats.spearmanr(x, y)
        pearson_r, pearson_p = stats.pearsonr(x, y)

        correlations[f'{m1}_vs_{m2}'] = {
            'n': len(valid),
            'spearman_r': float(spearman_r),
            'spearman_p': float(spearman_p),
            'pearson_r': float(pearson_r),
            'pearson_p': float(pearson_p),
        }

    return correlations


def main():
    print("=" * 70)
    print("COMPLETENESS V2 METRICS CORRELATION ANALYSIS")
    print("=" * 70)

    # Load all results
    all_results, family_counts = load_all_results()

    print(f"\nTotal results loaded: {len(all_results)}")
    print("\nPer-family counts:")
    for family_key, count in sorted(family_counts.items()):
        print(f"  {FAMILIES[family_key]}: {count}")

    # Count non-null per metric
    print("\nMetric coverage:")
    for m in METRICS:
        n = sum(1 for r in all_results if r.get(m) is not None)
        print(f"  {m}: {n}/{len(all_results)}")

    # Compute overall correlations
    print("\n" + "=" * 70)
    print("OVERALL CORRELATIONS (all families)")
    print("=" * 70)

    overall_corr = compute_correlations(all_results)

    # Print table
    print(f"\n{'Pair':<35} {'N':>5} {'Spearman r':>12} {'p-value':>12} {'Pearson r':>12} {'p-value':>12}")
    print("-" * 90)
    for pair, vals in overall_corr.items():
        if vals['spearman_r'] is not None:
            print(f"{pair:<35} {vals['n']:>5} {vals['spearman_r']:>12.4f} {vals['spearman_p']:>12.2e} {vals['pearson_r']:>12.4f} {vals['pearson_p']:>12.2e}")
        else:
            print(f"{pair:<35} {vals['n']:>5} {'N/A':>12} {'N/A':>12} {'N/A':>12} {'N/A':>12}")

    # Compute per-family correlations
    per_family_corr = {}
    for family_key in FAMILIES:
        family_results = [r for r in all_results if r['family'] == family_key]
        if len(family_results) >= 3:
            per_family_corr[family_key] = compute_correlations(family_results)

    # Print per-family correlations for key pairs
    key_pairs = ['mse_vs_explained_variance', 'mse_vs_cossim', 'mse_vs_ce_loss_score', 'mse_vs_kl_div_score']

    print("\n" + "=" * 70)
    print("PER-FAMILY CORRELATIONS (Spearman r)")
    print("=" * 70)

    print(f"\n{'Family':<35}", end="")
    for pair in key_pairs:
        label = pair.replace('mse_vs_', '')
        print(f" {label:>15}", end="")
    print()
    print("-" * 100)

    for family_key, family_name in FAMILIES.items():
        if family_key in per_family_corr:
            print(f"{family_name:<35}", end="")
            for pair in key_pairs:
                vals = per_family_corr[family_key].get(pair, {})
                r = vals.get('spearman_r')
                if r is not None:
                    print(f" {r:>15.4f}", end="")
                else:
                    print(f" {'N/A':>15}", end="")
            print()

    # Save results
    output = {
        'summary': {
            'total_results': len(all_results),
            'family_counts': family_counts,
            'metric_coverage': {m: sum(1 for r in all_results if r.get(m) is not None) for m in METRICS},
        },
        'overall_correlations': overall_corr,
        'per_family_correlations': per_family_corr,
        'all_results': [{k: v for k, v in r.items() if k != 'path'} for r in all_results],
    }

    with open(OUTPUT_FILE, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✅ Results saved to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
