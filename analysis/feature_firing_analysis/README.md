# Feature Firing Analysis (No Interpretability)

This folder contains helper functions for analyzing feature firing patterns in SAEs.

## Purpose

Extracted from `feature_firing_analysis_no_inter_sae_bench.py` to enable:
- Reusable firing frequency analysis across different SAE sources
- Consistent analysis methodology regardless of SAE origin
- Easy integration with different SAE loading mechanisms

## Key Analyses

1. **Firing Frequency Distribution**: How often features fire across the dataset
2. **Cumulative Magnitude Distribution**: Top N% features vs magnitude contribution
3. **Per-Input Magnitude Analysis**: How features rank within each input
4. **Firing Frequency vs Magnitude**: Relationship between firing frequency and activation magnitude

## Workflow

```
1. Load SAE (from any source)
   - SAEs are downloaded and cached locally (required for HuggingFace Hub)
2. Collect activations (from model or pre-collected)
3. Compute firing statistics:
   - Firing counts per feature
   - Firing frequencies
   - Magnitude statistics
   - Feature rankings
4. Create visualizations
5. Save results
6. **Clean up**: Delete cached SAE files after analysis completes
```

## Cache Management

**Note**: After completing analysis on a particular SAE, delete the cached SAE files to free disk space. Only SAEs need cache cleanup; LLM models use transformers library caching (separate, automatic).

## Output

Results are saved to: `{DATA_DIR}/sae/feature_firing_analysis/{sae_source}/{sae_name}/` (where DATA_DIR defaults to /data/user)

Output includes:
- Firing frequency distribution plots
- Cumulative magnitude distribution plots
- Per-input magnitude analysis plots
- Firing frequency vs magnitude plots

## Usage

See runner scripts for example usage with different SAE sources.

