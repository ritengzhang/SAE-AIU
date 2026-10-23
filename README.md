# SAE-AIU: Atomic Interpretable Unit Framework for SAE Evaluation

This is the code release accompanying the paper "Rethinking SAE Evaluation: An Atomic Interpretable Unit Framework Reveals Hidden Polysemanticity and Redundancy" by Riteng Zhang and Xiaoqian Wang (NeurIPS 2026, Evaluations & Datasets Track). It contains the evaluation framework (DAME, AIU criteria proxies, redundancy / completeness / sparsity / dead-unit metrics), training pipeline for newly trained SAEs, and analysis scripts that produce the per-SAE numbers reported in the paper.

Paper: [arXiv:2610.30474](https://arxiv.org/abs/2610.30474)

Checkpoints of our newly trained SAEs are hosted on HuggingFace: https://huggingface.co/ritengzhang/SAE-AIU

## Installation

Tested with Python 3.10–3.11.

```bash
# 1. Create a fresh environment
python -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. (Optional) For the DAME LLM judge, set OPENAI_API_KEY
export OPENAI_API_KEY=<your-key>
```

GPU (CUDA-enabled PyTorch) strongly recommended for SAE forward passes; CPU-only will work for the aggregation / significance scripts using the bundled `all_results/`.

## Quick sanity check

To verify the install works without running any SAE forward passes, you can re-aggregate the bundled completeness results (775 SAEs, ~0.6 MB):

```bash
python aiu_bench/aggregation/aggregate_dame.py \
    --root all_results/benchmark/mono/dame/ --output-dir /tmp/dame_summary/
```

This should print a per-architecture Q1/Q2/Q3/Q4 monosemanticity table matching Section 4.1 of the paper.

## Package contents

```
.
├── aiu_bench/                        # AIU criterion evaluation code
│   ├── completeness/                 # C2: reconstruction quality (MSE, R^2, cossim, CE/KL)
│   ├── monosemanticity/              # C3: DAME (dominance-aware monosemanticity)
│   ├── redundancy/                   # C1: decoder-cosine redundancy
│   ├── sparsity_dead_neuron/         # C4 / C5: dead-unit + sparsity metrics
│   ├── significance_tests/           # Statistical tests (Wilcoxon, Cohen's d) for paper Section 4
│   │                                 #   - DAME Q1 deficit per architecture
│   │                                 #   - Redundancy Q1 concentration filtered + unfiltered
│   └── additional_analysis/          # Appendix-supporting analyses (see its own README)
│       ├── decoder_to_embedding/        # Validates decoder cosine vs. semantic alignment
│       ├── synthetic_validation/        # DAME on ground-truth concepts
│       ├── threshold_sweep/             # Redundancy threshold robustness
│       ├── completeness_multi_metric/   # MSE vs R^2 / cossim / CE / KL correlations
│       ├── matched_sparsity/            # Effective L0 confound control
│       └── saebench_correlations/       # AIU criteria <-> SAEBench mapping
│
├── analysis/                         # Auxiliary analysis (activation clustering,
│                                     #   encoder-decoder relationship, mean direction divergence)
├── aiu_bench/generic_runners/        # Reusable runners for ANY SAE (saelens or local)
│                                     #   - run_dame.py, run_redundancy.py, run_completeness.py,
│                                     #     run_sparsity_dead_neuron.py, _loader.py
├── aiu_bench/aggregation/            # Standalone aggregation scripts:
│                                     #   - aggregate_dame.py, aggregate_redundancy.py,
│                                     #     aggregate_sparsity_dead_neuron.py
├── benchmark_training/               # Training pipeline for our newly trained SAEs
│                                     #   (GPT-2, OPT-125M, TinyStories)
├── sae_family_load_convert/          # Loaders for external SAE families
│                                     #   (SAEBench, JBloom OAI, CanRager)
│
└── all_results/                      # Pre-computed result files
    ├── benchmark/
    │   ├── completeness/             # FULL completeness dataset (all 775 evaluated SAEs)
    │   ├── mono/dame/                # DAME summary aggregates
    │   ├── redundancy/               # Sample of detailed redundancy results
    │   └── sparsity_dead_neuron/     # Sample of detailed sparsity / dead-unit results
    ├── analysis_result/              # Sample analysis configs and metadata
    └── training/                     # 162 training configs for our SAEs (combinations.json)
```

## What's included vs. excluded

### Included
- **Complete evaluation code** for all 5 AIU criteria.
- **Complete training code** for our newly trained SAE variants (GPT-2, OPT-125M, TinyStories), with all 162 hyperparameter configurations in `all_results/training/combinations.json`.
- **All appendix-supporting analysis code**: decoder-to-embedding alignment, synthetic-validation pipeline, threshold sweep, matched-sparsity binning, completeness multi-metric correlations, SAEBench correlations.
- **Complete completeness benchmark dataset** (775 per-SAE result files, ~0.6 MB) plus representative samples for redundancy / sparsity-dead-neuron / DAME.

### Excluded (regenerable from code)
- **Detailed per-SAE result JSONs** for redundancy, sparsity, and analysis subfolders (a representative sample is included). Total full dataset is ~10 GB.
- **Pre-computed aggregated statistics** (Wilcoxon p-values, Cohen's d, Spearman correlations, etc.) from significance tests and appendix analyses --- these can be regenerated directly by running the provided scripts on the included or full result data.
- **SAE checkpoints** for our newly trained SAEs (~29 GB) are hosted separately on HuggingFace: https://huggingface.co/ritengzhang/SAE-AIU. All training hyperparameters are documented in `all_results/training/combinations.json`; checkpoints can also be regenerated from the provided code.
- **Plot files (PNG / PDF)**. The paper's figures are embedded in the paper PDF; here we ship only the data and code that produce them.

## Path conventions

Some scripts contain absolute paths to a data root. To run them, substitute:
- `<DATA_ROOT>` -> your local path that holds the per-SAE result trees (the `benchmark/` subfolder structure under `all_results/` mirrors what scripts expect at this root).
- `<REPO_ROOT>` -> the top-level path of this repository.

Where present, these placeholders mark the substitution points clearly.

Several scripts also respect the environment variable `DATA_DIR` (defaulting to `/data/user`) for the data root. Setting this once is usually sufficient.

## Usage: running the framework on your own SAE

Two entry points depending on how your SAE is stored:

**SAELens / HuggingFace style** (e.g., a public SAE released on HuggingFace):
```bash
cd aiu_bench/generic_runners
python run_dame.py \
    --sae-style saelens \
    --sae-source jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs \
    --activation-source gpt2-small:blocks.6.hook_resid_post \
    --output-dir ./out/dame_jbloom_layer6/
```

**Local-checkpoint style** (e.g., an SAE you trained yourself, with `config.json` + `weights.pt`):
```bash
python run_dame.py \
    --sae-style local \
    --sae-source ./training_runs/my_sae/checkpoint_final/ \
    --activation-source ./activations/gpt2_layer6.pt \
    --output-dir ./out/dame_my_sae/
```

The same flags work for `run_redundancy.py`, `run_completeness.py`, `run_sparsity_dead_neuron.py`. See `aiu_bench/generic_runners/README.md` for full CLI documentation and how to add a new SAE-loading style.

After you've run the per-SAE evaluations, the standalone scripts in `aiu_bench/aggregation/` walk a directory of `results.json` files and produce architecture-level / family-level summary tables (matching the LaTeX tables in the paper):

```bash
python aiu_bench/aggregation/aggregate_dame.py            --root ./out/ --output-dir ./summary/
python aiu_bench/aggregation/aggregate_redundancy.py      --root ./out/ --output-dir ./summary/
python aiu_bench/aggregation/aggregate_sparsity_dead_neuron.py --root ./out/ --output-dir ./summary/
```

For the family-specific runners that reproduce the exact paper numbers (per family / architecture), see `aiu_bench/<criterion>/runners/run_all_*.py`.

## Reproduction pointers

| Paper claim | Code |
|---|---|
| DAME Q1 monosemanticity deficit (Section 4.1) | `aiu_bench/monosemanticity/` + `aiu_bench/significance_tests/review1.py` |
| Redundancy Q1 concentration (Section 4.2, Appendix F.4) | `aiu_bench/redundancy/` + `aiu_bench/significance_tests/sparsity_binned_redundancy_v2.py` |
| Decoder cosine vs. semantic alignment (Appendix D.1.1) | `aiu_bench/additional_analysis/decoder_to_embedding/` |
| Synthetic validation of DAME (Appendix D.3.2) | `aiu_bench/additional_analysis/synthetic_validation/` |
| Threshold sweep robustness (Appendix F.4) | `aiu_bench/additional_analysis/threshold_sweep/` |
| Completeness multi-metric correlations (Appendix D.2) | `aiu_bench/additional_analysis/completeness_multi_metric/` |
| Matched-sparsity confound control (Appendix F.4) | `aiu_bench/additional_analysis/matched_sparsity/` |
| Empirical SAEBench mapping (Appendix D.6) | `aiu_bench/additional_analysis/saebench_correlations/` |

For each SAEBench-loaded family we also ship the loader: see `sae_family_load_convert/` for the conversion utilities used to read SAEBench, JBloom OAI, and CanRager checkpoints into a uniform interface.

## License

The code in this repository is released under the MIT License (see `LICENSE`). External models, SAE
checkpoints, and datasets used by the scripts are not redistributed here and remain under their own
licenses.

## Citation

```bibtex
@inproceedings{zhang2026rethinking,
  title     = {Rethinking SAE Evaluation: An Atomic Interpretable Unit Framework Reveals Hidden Polysemanticity and Redundancy},
  author    = {Zhang, Riteng and Wang, Xiaoqian},
  booktitle = {Advances in Neural Information Processing Systems (Evaluations and Datasets Track)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2610.30474}
}
```
