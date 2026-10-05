# Aggregation Scripts

Standalone scripts that walk a directory of per-SAE result JSON files (produced by the runners under `aiu_bench/*/runners/` or `aiu_bench/generic_runners/`) and produce architecture-level / family-level summary statistics.

Each script takes a `--root` directory (where it walks `**/results.json`) and a `--output-dir` for the summaries. They print a markdown-style summary table to stdout and write a JSON aggregation to disk.

## Files

| Script | Aggregates |
|---|---|
| `aggregate_dame.py` | Per-SAE DAME metrics → per-architecture, per-family Q1/Q2/Q3/Q4 means + Wilcoxon |
| `aggregate_redundancy.py` | Per-SAE quantile pair counts → per-architecture filtered & unfiltered share |
| `aggregate_sparsity_dead_neuron.py` | Per-SAE sparsity / dead-neuron stats → per-architecture median L₀ + dead-neuron rates |

## Usage

```bash
# After running the per-SAE evaluations into ./out/
python aggregate_dame.py --root ./out/ --output-dir ./summary/
python aggregate_redundancy.py --root ./out/ --output-dir ./summary/
python aggregate_sparsity_dead_neuron.py --root ./out/ --output-dir ./summary/
```

Each script auto-detects architecture from the SAE config / path. If your per-SAE configs don't include an `architecture` field, set `--default-arch standard` (or use `--arch-from-path` to extract from directory name).

## What the scripts produce

**aggregate_dame.py** writes:
- `dame_per_architecture.json` — per-arch Q1/Q2/Q3/Q4 means + per-SAE counts
- `dame_per_family.json` — per-family breakdown
- `dame_significance.json` — per-arch Wilcoxon p, Cohen's d, overall pooled stats

**aggregate_redundancy.py** writes:
- `redundancy_per_architecture_filtered.json` — Q1/Q2/Q3/Q4 quantile share, dead-neuron-filtered, per-SAE averaged
- `redundancy_per_architecture_unfiltered.json` — same, unfiltered + pair filter rate
- `redundancy_significance.json` — per-arch Wilcoxon p, Cohen's d (filtered)

**aggregate_sparsity_dead_neuron.py** writes:
- `l0_per_architecture.json` — median L₀ in absolute count, fraction of dictionary, fraction of d_model
- `dead_neuron_per_architecture.json` — per-arch dead-neuron rates at three thresholds

These JSON files mirror the per-architecture tables in the paper (Section 5 + Appendix C).
