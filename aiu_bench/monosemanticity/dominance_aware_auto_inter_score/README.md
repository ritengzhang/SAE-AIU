# DAME: Dominance-Aware Multiple-choice Evaluation

**Self-contained DAME workflow** - This module implements the complete DAME (Dominance-Aware Multiple-choice Evaluation) workflow for evaluating SAE feature interpretability using a 3-way ranking task.

## Overview

This module implements the DAME evaluation workflow with the following key features:
- **Mean dominance ranking**: Features ranked by average contribution percentage across all input tokens (not firing frequency)
- **3-bin strategy**: Sequences binned into HIGH-high (BIN A), HIGH-low (BIN B), and LOW (BIN C)
- **3-way ranking task**: LLM ranks 3 examples (one from each bin) by activation strength
- **4 evaluation metrics**: Perfect ranking accuracy, rank correlation, pairwise accuracy, and rank distance
- **Brevity constraint**: Explanations limited to 15 words maximum

## Architecture

### Components

- **Core modules** (`core/`): Self-contained implementation
  - `collection_stage.py`: Dominance calculation, 3-bin strategy, sequence collection
  - `explanation_stage.py`: Explanation generation with token/word limits
  - `evaluation_stage.py`: 3-way ranking task and 4-metric evaluation
  - `visualization.py`: 4 separate metric plots

- **Runners** (`runners/`): Batch runners for different SAE benchmarks
  - `run_all_saebench_gemma.py`: SAE Bench Gemma SAEs
  - `run_all_saebench_pythia.py`: SAE Bench Pythia SAEs
  - `run_all_benchmark_gpt2.py`: GPT-2 benchmark SAEs
  - `run_all_benchmark_tinystories.py`: TinyStories benchmark SAEs
  - `run_all_benchmark_opt125m.py`: OPT-125M benchmark SAEs
  - `run_all_jbloom_oai_v5.py`: JBloom OAI v5 SAEs
  - `run_all_canrager.py`: Canrager SAEs

### Core Implementation

All core logic is in `core/`:
- `collection_stage.py`: `ActivationCollector`, `CollectionConfig` - dominance calculation, 3-bin strategy
- `explanation_stage.py`: `ExplanationGenerator` - generates concise feature explanations
- `evaluation_stage.py`: `TokenBasedEvaluator`, `SentenceBasedEvaluator` - 3-way ranking evaluation
- `visualization.py`: `create_dame_metric_plots` - generates 4 separate metric plots

## Usage

### Run All SAE Bench Gemma SAEs

```bash
cd aiu_bench/monosemanticity/dominance_aware_auto_inter_score/runners
python run_all_saebench_gemma.py --device cuda:0
```

### Parameters

- `--device`: Device to use (default: cuda:0)
- `--skip-existing`: Skip SAEs that already have results (default: True)
- `--no-skip-existing`: Don't skip existing results (force re-run)

### Configuration

- **Feature ranking**: Mean dominance (not firing frequency)
- **Minimum firing threshold**: 80 sequences (ensures enough data for 3-bin strategy)
- **Features per quantile**: 20 (80 total per SAE: 20 × 4 quantiles)
- **Batches for discovery**: 50 batches
- **Binning strategy**: 3 bins (BIN A: HIGH-high with 20 sequences, BIN B: HIGH-low with 20 sequences, BIN C: LOW with 20 sequences)
- **Explanation examples**: Top 20 sequences from HIGH bin (excluded from questions)
- **Evaluation sequences**: 40 sequences for evaluation (20 in bin_A, 20 in bin_B) - bin_A has higher activations than bin_B
- **Evaluation questions**: 20 questions per feature (3-way ranking task)
- **Evaluation task**: 3-way ranking (not binary MC)
- **Evaluation metrics**: 4 metrics (perfect ranking, correlation, pairwise, rank distance)
- **LLM Provider**: Offline (Mistral 7B) - all runners use offline mode
- **Explanation limits**: max_new_tokens=40, maximum 15 words

## Workflow

The DAME workflow implements the following steps:

1. **Pass 0**: Fast dominance counting (50 batches) → compute mean dominance per feature
2. **Filtering & Quantile Sampling**: 
   - Filter features with ≥80 firing examples
   - Rank by mean dominance (not frequency)
   - Sample 20 features per quantile (80 total)
3. **Collection & Binning**: 
   - Collect all sequences for sampled features
   - Sort by activation magnitude (descending)
   - Top 20 sequences → explanation (excluded from questions)
   - Next 40 sequences → split into bin_A (20 highest) and bin_B (20 lower) - bin_A has higher activations than bin_B
   - Bottom 20 sequences → bin_C (LOW)
4. **Explanation**: Generate explanations using top 20 sequences (max 40 tokens, 15 words)
5. **Evaluation**: 
   - 3-way ranking task: Sample 1 from each bin (A, B, C) - 20 questions per feature
   - Ask LLM to rank by activation strength (1=highest, 2=middle, 3=lowest)
   - True ranking: A=1 (highest), B=2 (middle), C=3 (lowest)
   - Compute 4 metrics per question: perfect ranking, correlation, pairwise, rank distance
   - Both token-based and sentence-based evaluation

See [`WORKFLOW_EXPLANATION.md`](./WORKFLOW_EXPLANATION.md) for detailed workflow documentation.

## Output

Results are saved to:
```
{DATA_DIR}/sae/benchmark/mono/dame/ (where DATA_DIR defaults to /data/user)
├── {benchmark_name}/{width}/{sae_type}/{trainer}/
│   ├── results.json          # Complete results with 4 metrics per question
│   ├── dame_perfect_ranking_plot.png
│   ├── dame_correlation_plot.png
│   ├── dame_pairwise_plot.png
│   └── dame_rank_distance_plot.png
```

### Results JSON Structure

```json
{
  "quantile_stats": {
    "Q1 (0-25%)": { "n_features": 350, "n_sampled": 20, "mean_dominance": 0.000836, ... },
    "Q2 (25-50%)": { ... },
    "Q3 (50-75%)": { ... },
    "Q4 (75-100%)": { ... }
  },
  "explanations": {
    "feature_idx": { 
      "explanation": "...", 
      "explanation_sequence_ids": [...],
      "error": null 
    }
  },
  "collection_stats": { ... },
  "token_results": {
    "feature_idx": { 
      "perfect_ranking_acc": 0.65,
      "mean_rank_correlation": 0.72,
      "mean_pairwise_acc": 0.80,
      "mean_rank_distance": 0.35,
      "n_questions": 20,
      "questions": [
        {
          "question": "...",
          "true_ranking": {"A": 1, "B": 2, "C": 3},
          "predicted_ranking": {"A": 1, "B": 2, "C": 3},
          "perfect_ranking": true,
          "rank_correlation": 1.0,
          "pairwise_acc": 1.0,
          "rank_distance": 0.0
        },
        ...
      ]
    }
  },
  "sentence_results": {
    "feature_idx": { ... }  // Same structure as token_results
  },
  "summary": {
    "n_features_evaluated": 80,
    "token_metrics": {
      "perfect_ranking": 0.650,
      "correlation": 0.720,
      "pairwise": 0.800,
      "rank_distance": 0.350
    },
    "sentence_metrics": {
      "perfect_ranking": 0.645,
      "correlation": 0.715,
      "pairwise": 0.795,
      "rank_distance": 0.360
    }
  },
  "metadata": {
    "repo_id": "...",
    "sae_type": "...",
    "n_features_per_quantile": 20,
    ...
  }
}
```

## Key Differences from Previous Workflow

### Ranking Method
- **OLD**: Rank by firing frequency C_i
- **NEW**: Rank by mean dominance ρ̄_i = E_x[|f_i(φ(x))| / Σ_j |f_j(φ(x))|]

### Filtering Threshold
- **OLD**: ≥40 firing examples
- **NEW**: ≥80 firing examples (ensures enough data for 3-bin strategy)

### Binning Strategy
- **OLD**: 2 bins (high/low, median split)
- **NEW**: 3 bins (BIN A: HIGH-high with 20 sequences, BIN B: HIGH-low with 20 sequences, BIN C: LOW with 20 sequences)
- **NEW**: bin_A has higher activations than bin_B (ensured by sorting before splitting)

### Evaluation Task
- **OLD**: Binary multiple-choice ("Which token has highest activation?")
- **NEW**: 3-way ranking task ("Rank these 3 examples: 1=highest, 2=middle, 3=lowest")

### Evaluation Metrics
- **OLD**: Single accuracy metric
- **NEW**: 4 metrics (perfect ranking, correlation, pairwise, rank distance)

### Explanation Generation
- **OLD**: 200 tokens, no word limit
- **NEW**: 40 tokens, maximum 15 words, explicit brevity constraint

### Output Directory
- **OLD**: `{DATA_DIR}/sae/benchmark/monosemanticity/dominance_aware_auto_inter_score/`
- **NEW**: `{DATA_DIR}/sae/benchmark/mono/dame/` (where DATA_DIR defaults to /data/user)

## Backends

The evaluation stage supports two LLM-judge backends (set via the `provider`
argument; default `"openai"`):

- `"openai"`: OpenAI API. Approximate cost ~\$0.24-0.30 per SAE (depends on
  width and feature count). Prompt caching enabled via `prompt_cache_key`.
- `"offline"`: local Mistral-7B inference (no API cost, requires GPU).

The paper's reported DAME numbers were produced with the OpenAI backend.

### Time Estimate

- **Per SAE**: ~60-90 minutes for 80 features (depends on SAE width and backend)
- **Bottlenecks**: Collection & binning (~30-40 min), explanation generation (~5-10 min), evaluation (~20-40 min)

## References

- **Workflow documentation**: [`WORKFLOW_EXPLANATION.md`](./WORKFLOW_EXPLANATION.md)
- **Change summary**: [`../v2change.md`](../v2change.md)
- **Test runner**: [`runners/test_single_sae_dame.py`](./runners/test_single_sae_dame.py) (single SAE test)

## Notes

- **Self-contained**: All core logic in `core/` (not dependent on `interpretability_mc_evaluation`)
- **Dominance-based ranking**: Features ranked by mean dominance, not firing frequency
- **3-bin strategy**: 20 sequences per bin (bin_A: HIGH-high, bin_B: HIGH-low, bin_C: LOW)
- **Activation ordering**: bin_A has higher activations than bin_B (ensured by sorting before splitting)
- **20 questions per feature**: One question from each bin (A, B, C) for fair evaluation
- **No overlap**: Top 20 sequences used for explanation are excluded from evaluation questions
- **4 separate plots**: Each metric plotted separately for detailed analysis
- **Detailed results**: `results.json` contains per-question metrics for full transparency
- **Backend**: OpenAI by default (used for paper numbers); offline Mistral-7B available via `provider="offline"`