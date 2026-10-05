# Training Configurations

This folder documents the hyperparameter configurations used to train our
benchmark SAEs (Section 5 / Appendix experiment details).

## Contents

- `combinations.json`: 162 training combinations (LLM, architecture, layer,
  width, sparsity hyperparameters) covering GPT-2, OPT-125M, and TinyStories
  across all evaluated SAE architectures.

## Note

Checkpoint files are not included (~28 GB total). See `../../README.md` for
why large artifacts were excluded.

## Reproducing the SAEs

To re-train any combination:
- The training code is in `../../benchmark_training/`
- The hyperparameter spec is in `combinations.json`
- The runners are in `../../benchmark_training/runners/`

See `../../benchmark_training/README.md` for detailed instructions.
