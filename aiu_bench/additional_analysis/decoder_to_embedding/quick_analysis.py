#!/usr/bin/env python3
"""Quick analysis of completed results across all families."""

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

base = Path("<DATA_ROOT>/rebuttal_redundancy_to_semantic_alignment")
families = {
    "Gemma 4k": base / "saebench_gemma" / "4k",
    "Pythia 4k": base / "saebench_pythia" / "4k",
    "GPT-2 4k": base / "benchmark_training_gpt2" / "4k",
}

all_files = []
for name, d in families.items():
    files = list(d.rglob("results.json")) if d.exists() else []
    print(f"  {name}: {len(files)} SAEs")
    all_files.extend(files)

print(f"\nTotal: {len(all_files)} SAEs\n")

pooled = {}
for f in all_files:
    data = json.load(open(f))
    for bin_label, windows in data["pair_details"].items():
        if bin_label not in pooled:
            pooled[bin_label] = {}
        for w, pairs in windows.items():
            if w not in pooled[bin_label]:
                pooled[bin_label][w] = []
            pooled[bin_label][w].extend(
                [p["embed_sim"] for p in pairs if not np.isnan(p["embed_sim"])]
            )

bins_sorted = sorted(pooled.keys())
windows = ["8", "16", "32", "64"]

print(f"{'Bin':<12} {'n_pairs':<10} {'W=8':<10} {'W=16':<10} {'W=32':<10} {'W=64':<10}")
print("-" * 62)
for b in bins_sorted:
    n = len(pooled[b].get("8", []))
    row = f"{b:<12} {n:<10}"
    for w in windows:
        vals = pooled[b].get(w, [])
        if vals:
            row += f" {np.mean(vals):.4f}    "
        else:
            row += " N/A       "
    print(row)

print("\nSpearman correlation (per-pair decoder_sim vs embed_sim):")
for w in windows:
    all_dec, all_emb = [], []
    for f in all_files:
        data = json.load(open(f))
        for bin_label, windows_data in data["pair_details"].items():
            for p in windows_data.get(w, []):
                if not np.isnan(p["embed_sim"]):
                    all_dec.append(p["decoder_sim"])
                    all_emb.append(p["embed_sim"])
    if len(all_dec) > 10:
        rho, pval = spearmanr(all_dec, all_emb)
        print(f"  W={w}: rho={rho:.4f}, p={pval:.2e}, n={len(all_dec)} pairs")
