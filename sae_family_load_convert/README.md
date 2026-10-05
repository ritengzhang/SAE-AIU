# SAE Family Load and Convert

This folder contains SAE loading, conversion, and activation collection functions organized by SAE source/family.

## Structure

Each SAE source has its own subfolder:
- `saebench/` - SAE Bench models from HuggingFace
- (Future: other sources like Anthropic, etc.)

## Purpose

- **Load SAEs** from different sources (HuggingFace, local, etc.)
- **Convert** SAE formats to a standardized format compatible with our analysis code
- **Collect activations** from models for analysis

## Workflow

```
1. Load SAE from source (e.g., HuggingFace)
   - SAEs are downloaded and cached locally (required for HuggingFace Hub)
2. Convert to standardized format (SAELens-compatible)
3. Load base model (LLM models use transformers library caching, separate from SAE cache)
4. Setup activation collection
5. Return SAE, model, and activation store ready for analysis
6. **After analysis completes**: Clean up SAE cache to free disk space
```

## Cache Management

**Important**: SAEs must be downloaded and cached (HuggingFace Hub requirement). However, **after completing analysis on a particular SAE, the cached files should be deleted** to free disk space.

- **SAE caching**: Managed by this module (HuggingFace Hub cache)
- **LLM caching**: Handled automatically by transformers library (no manual cleanup needed)

## Usage

Each subfolder contains:
- `load_sae.py` - SAE loading functions
- `convert_sae.py` - Format conversion functions  
- `collect_activations.py` - Activation collection helpers
- `test_load.py` - Test script to verify loading works

## Output

Loaded SAEs are compatible with:
- `../analysis/encoder_decoder_activation_relationship/` — Angular analysis
- `../analysis/feature_firing_analysis/` — Firing analysis
- Other analysis modules under `../analysis/` and `../aiu_bench/`

