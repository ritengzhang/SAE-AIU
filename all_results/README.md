# Pre-computed Result Files

This folder contains a curated subset of the per-SAE result data needed to
verify and extend the paper's analyses without re-running expensive
evaluations. See `../README.md` for the full picture of what is and isn't
shipped with this code release.

## Contents

```
all_results/
├── benchmark/
│   ├── completeness/             # FULL: 775 per-SAE completeness JSONs (~0.6 MB)
│   ├── mono/dame/                # DAME aggregates + samples
│   ├── redundancy/               # Sample of detailed redundancy results
│   └── sparsity_dead_neuron/     # Sample of detailed sparsity / dead-unit results
├── analysis_result/              # Sample analysis configs (clustering,
│                                 #   encoder-decoder, mean-direction divergence)
└── training/                     # 162 training configurations for the SAEs we
                                  #   trained (combinations.json + README)
```

## What's full vs. sampled

| Subfolder | Coverage |
|---|---|
| `benchmark/completeness/` | **FULL** — every SAE we evaluated for C2 (775 files) |
| `benchmark/mono/dame/` | Aggregates + small sample of per-SAE detail |
| `benchmark/redundancy/` | Representative sample (full set is ~10 GB) |
| `benchmark/sparsity_dead_neuron/` | Representative sample |
| `analysis_result/` | Sample of per-SAE configs / metadata |
| `training/` | Full 162 training-hyperparameter spec |

## What's NOT included

- **Detailed per-SAE result JSONs** for redundancy / sparsity / DAME beyond
  the included sample (full set is ~10 GB; representative sample is shipped
  here so readers can inspect the file format and verify the aggregation
  pipeline against a small subset).
- **Pre-computed aggregated statistics** (Wilcoxon p-values, Cohen's d,
  Spearman correlations) — these can be regenerated directly using the
  scripts in `../aiu_bench/aggregation/` and
  `../aiu_bench/significance_tests/` on the included or full result data.
- **SAE checkpoints** for our newly trained SAEs (~29 GB); they are hosted on
  HuggingFace (https://huggingface.co/ritengzhang/SAE-AIU). All training
  hyperparameters are documented in `training/combinations.json`.

To regenerate any of the missing artifacts from code, see:
- `../aiu_bench/<criterion>/runners/` for per-family SAE evaluation runners
- `../aiu_bench/generic_runners/` for generic runners that work on any SAE
- `../aiu_bench/aggregation/` for per-SAE → per-architecture aggregation
- `../aiu_bench/significance_tests/` for the Wilcoxon and Cohen's d scripts
- `../benchmark_training/` for the SAE training pipeline
