# Additional Analysis

Extended analyses produced during paper revision, supporting the main-paper
claims with additional robustness checks, validations, and cross-evaluations.
All scripts are anonymized; absolute paths use `<DATA_ROOT>` (path to the data
storage root) and `<REPO_ROOT>` (path to this code's repository root)
placeholders --- substitute with your local paths.

## Contents

```
additional_analysis/
├── decoder_to_embedding/         (A2/B5/B6) Decoder cosine -> semantic alignment
├── synthetic_validation/         (A3) AIU framework validation on ground-truth concepts
├── threshold_sweep/              (B4) Redundancy threshold robustness
├── completeness_multi_metric/    (B9) MSE vs R^2 / cosine / CE / KL completeness metrics
├── matched_sparsity/             (B3) Q1 effects controlled by effective L0
└── saebench_correlations/        (B7/C1) Empirical mapping of AIU criteria to SAEBench evals
```

Each subfolder contains code only; pre-computed result JSONs are not
shipped. Reviewers can regenerate the per-analysis outputs by running the
scripts on the appropriate input data (see per-folder summaries below).

## Per-folder summary

### `decoder_to_embedding/`
Validates decoder cosine similarity as a proxy for semantic alignment:
- 5 runners (`run_saebench_*.py`, `run_benchmark_training_*.py`) compute, for
  each SAE, per-feature explanation embeddings (sentence-BERT) and pair-level
  decoder cosine vs.\ embedding cosine.
- `analyze_and_plot.py` aggregates across all 136 SAEs (95k feature pairs)
  and produces the bar / violin / scatter / window-sweep plots.
- `quick_analysis.py` is a small inspection utility.

### `synthetic_validation/`
End-to-end synthetic benchmark with 128 known ground-truth concepts and
105 SAEs (7 architectures x 3 widths x 5 sparsity levels):
- `generate_synthetic_data.py`: synthesize activations from ground-truth concepts.
- `compute_ground_truth.py`: per-feature monosemanticity gap from ground truth.
- `sae_models.py`: SAE architectures (Standard, TopK, BatchTopK, Matryoshka,
  JumpReLU, Gated, PAnneal).
- `train_saes.py` / `train_missing.py`: train all combinations.
- `plot_results.py`: produce the Q1 deficit summary plot.

### `threshold_sweep/`
Robustness of Q1 redundancy concentration across decoder cosine thresholds
theta in {0.1, 0.2, ..., 0.9}:
- `reviewer2_redundancy_threshold_plot.py`: reads per-threshold redundancy
  results and produces the combined sweep plot.

### `completeness_multi_metric/`
Validates MSE as completeness proxy by comparing to alternative metrics:
- `aggregate_completeness_v2.py`: aggregates across 600+ SAEs and computes
  Spearman / Pearson correlations between mse, explained_variance (R^2),
  cossim, ce_loss_score, kl_div_score.
- `reviewer4_completeness_correlations.py`: scatter / matrix plots of the
  pairwise correlations.

### `matched_sparsity/`
Controls for sparsity confound:
- `reviewer4_effective_sparsity_binned.py`: bins SAEs by effective L0 and
  reports DAME deficit + redundancy concentration per bin. Used to show the
  Q1 effect persists across all sparsity levels.

### `saebench_correlations/`
Empirical validation of the AIU framework against SAEBench evaluations:
- `verify_draft_w1q1.py`: computes per-criterion-pair Spearman correlations
  between AIU criteria (C1--C5) and 6 SAEBench evaluations (Sparse Probing,
  SCR, TPP, Feature Absorption, RAVEL, Unlearning) over 246 SAEs
  (Gemma + Pythia, 3 widths).

## How results map to the paper

| Folder | Paper section |
|---|---|
| decoder_to_embedding | Section 4.2 (main figure) + appendix |
| synthetic_validation | Section 5.1 forward-ref + appendix subsection |
| threshold_sweep | Section 5.2 main-text mention + appendix figure |
| completeness_multi_metric | Section D.1 (appendix; updates D.1.1 to mention 5 metrics) |
| matched_sparsity | Section 5 main-text mention + appendix |
| saebench_correlations | Appendix subsection within "Proxy Metrics" + Discussion (Section 6) |
