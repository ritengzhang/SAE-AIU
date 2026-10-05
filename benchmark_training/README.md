# SAE Benchmark Training

Comprehensive benchmark for training Sparse Autoencoders (SAEs) across multiple architectures, LLMs, widths, and sparsity levels.

## Overview

- **6 Architectures trained here**: Standard, Gated, JumpReLU, TopK, BatchTopK, MatryoshkaBatchTopK. (PAnneal, the 7th architecture in the paper, comes from external SAEBench releases — see `../sae_family_load_convert/`.)
- **3 LLMs**: GPT-2, TinyStories, OPT-125M
- **3 Widths**: 4k (4096), 16k (16384), 65k (65536)
- **3 Sparsity levels** per architecture
- **1 Middle layer** per LLM
- **Total**: 54 combinations per LLM = **162 total combinations** (see `../all_results/training/combinations.json`)

## Setup

1. Ensure SAELens is installed and accessible:
   ```bash
   # SAELens should be in ../SAELens-main/
   ```

2. Install dependencies:
   ```bash
   pip install torch transformer-lens sae-lens wandb datasets
   ```

## Usage

### 1. Generate Combinations

First, generate all parameter combinations:

```bash
cd benchmark_training
python generate_combinations.py
```

This creates the training combinations spec (162 combinations).

### 2. Cache Activations (Optional but Recommended)

Pre-cache activations for efficient training sweeps:

```bash
# Cache for all LLMs
python cache_activations.py

# Cache for specific LLM
python cache_activations.py --llm gpt2
```

Activations are saved to `data/user/sae/training/{llm_name}/activations/layer_{layer}/`

### 3. Run Benchmark

Run benchmarks using individual runners for each LLM:

```bash
# Run GPT-2 benchmark (54 combinations)
python runners/run_gpt2.py

# Run TinyStories benchmark (54 combinations)
python runners/run_tiny_stories.py

# Run OPT-125M benchmark (54 combinations)
python runners/run_opt_125m.py
```

Each runner processes all 54 combinations for its specific LLM.

## Output Structure

```
data/user/sae/training/
├── combinations.json              # All combinations
├── {llm_name}/
│   ├── activations/              # Cached activations
│   │   └── layer_{layer}/
│   ├── checkpoints/              # Training checkpoints
│   │   └── {arch}_L{layer}_{width}_{sparsity}/
│   ├── results.json              # Results summary
│   └── wandb/                    # WandB logs
```

## Configuration

Edit `benchmark_config.py` to modify:
- LLM choices and configurations
- Architecture sparsity parameter ranges
- Training hyperparameters
- Output directories

## Training Parameters

- **Training steps**: 18,000 per run
- **Batch size**: 4,096 tokens
- **Learning rate**: 5e-5
- **Context size**: 512
- **Total tokens per run**: ~73.7M

## Notes

- Transcoder architectures are excluded (no training support in SAELens)
- Each LLM is processed completely before moving to the next
- Failed runs are logged but don't stop the benchmark
- WandB logging is enabled by default (project: `sae_benchmark_training`)

