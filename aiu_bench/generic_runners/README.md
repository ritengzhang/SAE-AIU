# Generic Runners

Reusable entry points for running each AIU criterion analysis on **your own SAEs**, without writing a family-specific loader. Each runner accepts an SAE in one of two styles:

- **`saelens`**: an SAE loaded via the [SAELens](https://github.com/jbloomAus/SAELens) library or any compatible HuggingFace repository (e.g., `release_id` + `sae_id` lookup, or direct `from_pretrained` style).
- **`local`**: an SAE checkpoint stored on disk as `(config.json, weights.pt)` produced by your own training pipeline.

If your SAE family does not match either style, you can still adapt one of the family-specific runners under each analysis subfolder (e.g., `aiu_bench/redundancy/runners/run_all_saebench_pythia.py`) as a template.

## Folder layout

```
generic_runners/
├── README.md                       (this file)
├── _loader.py                      Shared SAE loading helpers (both styles)
├── run_dame.py                     C3: Dominance-Aware Monosemanticity Evaluation
├── run_redundancy.py               C1: decoder-cosine redundancy + quantile breakdown
├── run_completeness.py             C2: MSE / R^2 / cossim / CE / KL completeness metrics
└── run_sparsity_dead_neuron.py     C4 + C5: dead-neuron rate + average L_0
```

## Usage

All runners share the same CLI flags:

| Flag | Description | Example |
|---|---|---|
| `--sae-style` | Either `saelens` or `local` | `saelens` |
| `--sae-source` | For `saelens`: HuggingFace repo ID or release identifier. For `local`: directory containing `config.json` + `weights.pt`. | `jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs` (saelens) or `./my_sae_run/checkpoint_final/` (local) |
| `--activation-source` | Either path to a pre-cached activation tensor (`.pt`) or `model_name:hook_name` for on-the-fly extraction. | `gpt2-small:blocks.6.hook_resid_post` |
| `--output-dir` | Where to write `results.json` and any intermediate artefacts. | `./out/my_sae_dame/` |
| `--n-features` | Number of features to evaluate (DAME only). Defaults to 80 (20 per quantile). | `80` |

### Examples

#### DAME on a SAELens SAE
```bash
python run_dame.py \
    --sae-style saelens \
    --sae-source jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs \
    --activation-source gpt2-small:blocks.6.hook_resid_post \
    --output-dir ./out/dame_jbloom_gpt2_layer6/
```

#### Redundancy on a locally-trained SAE
```bash
python run_redundancy.py \
    --sae-style local \
    --sae-source ./training_runs/my_topk_sae/checkpoint_final/ \
    --activation-source ./activations/gpt2_layer6.pt \
    --output-dir ./out/redundancy_my_topk/
```

#### Completeness (5 metrics) on multiple SAEs in a sweep
```bash
for ckpt in ./training_runs/sweep_*; do
    python run_completeness.py \
        --sae-style local \
        --sae-source $ckpt \
        --activation-source ./activations/gpt2_layer6.pt \
        --output-dir ./out/completeness_sweep/$(basename $ckpt)
done
```

## After running

Each runner writes:
- `results.json` — primary output (per-feature / per-pair statistics)
- `config.json` — record of the run config (SAE source, activation source, hyperparameters)

To turn many per-SAE result files into the architecture-level / family-level summary statistics reported in the paper, use the scripts in `../aggregation/`:

```bash
python ../aggregation/aggregate_dame.py --root ./out/ --output-dir ./summary/
python ../aggregation/aggregate_redundancy.py --root ./out/ --output-dir ./summary/
```

## Adapting to a new SAE family

If your SAE doesn't match the `saelens` or `local` styles, the cleanest extension point is `_loader.py`. Add a new loader function that returns:
```python
{
    'encoder_weight': torch.Tensor,    # (d_sae, d_in)
    'decoder_weight': torch.Tensor,    # (d_sae, d_in)
    'encoder_bias': torch.Tensor,      # (d_sae,)
    'decoder_bias': torch.Tensor,      # (d_in,)
    'd_in': int,
    'd_sae': int,
    'architecture': str,               # one of: 'standard', 'topk', 'batchtopk', 'matryoshka', 'jumprelu', 'gated', 'panneal'
    'sparsity_param': dict,            # architecture-specific (e.g., {'k': 50} for topk)
}
```
and register it in the `LOADERS` dict at the top of `_loader.py`. Then pass `--sae-style your_new_style` to any runner.
