# Encoder-Decoder-Activation Relationship Analysis

This folder contains helper functions for analyzing the angular relationships between encoder vectors, decoder vectors, and original activations when features are fired.

## Purpose

This module extracts and modularizes the core analysis functions from `mean_activation_pattern_analysis.py` to enable:
- Reusable analysis across different SAE sources (HuggingFace, local, etc.)
- Consistent analysis methodology regardless of SAE origin
- Easy integration with different SAE loading mechanisms

## Key Analysis

The main analysis focuses on **activated features only** (not random pairs):
- For each input activation, run through SAE encoder to get feature activations
- Identify which features fired (activation > threshold)
- For fired features, compute:
  - **Encoder vector → Input activation** (angular distance)
  - **Decoder vector → Input activation** (angular distance)
  - **Encoder vector → Decoder vector** (same feature, angular distance)

## Workflow

```
1. Load SAE (from any source)
   - SAEs are downloaded and cached locally (required for HuggingFace Hub)
2. Collect activations (from model or pre-collected)
3. For each activation:
   a. Encode through SAE → get feature activations
   b. Find fired features (activation > threshold)
   c. For each fired feature:
      - Compute encoder_vector → input_activation angle
      - Compute decoder_vector → input_activation angle  
      - Compute encoder_vector → decoder_vector angle
4. Aggregate results by percentile (top 25%, 50%, 75%, all)
5. Create visualization (2_activated_features_analysis.png)
6. Save results
7. **Clean up**: Delete cached SAE files after analysis completes
```

## Cache Management

**Note**: After completing analysis on a particular SAE, delete the cached SAE files to free disk space. Only SAEs need cache cleanup; LLM models use transformers library caching (separate, automatic).

## Output

Results are saved to: `{DATA_DIR}/sae/encoder_decoder_activation_relationship/{sae_source}/{sae_name}/` (where DATA_DIR defaults to /data/user)

Main output file: `2_activated_features_analysis.png` containing:
- Top-left: Encoder → Input activation angular distances
- Top-right: Decoder → Input activation angular distances
- Bottom-left: Encoder → Decoder (same feature) angular distances
- Bottom-right: Box plots summary across percentiles

## Usage

See `run_analysis.py` for example usage with different SAE sources.

