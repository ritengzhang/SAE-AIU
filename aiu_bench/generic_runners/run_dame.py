#!/usr/bin/env python3
"""Generic DAME (C3 monosemanticity) runner template.

Computes per-feature DAME metrics (perfect_ranking, correlation, pairwise,
rank_distance) plus per-quantile statistics for Q1..Q4 (top-decile dominance
to bottom-decile).

Usage:
    python run_dame.py \\
        --sae-style saelens \\
        --sae-source jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs \\
        --activation-source gpt2-small:blocks.6.hook_resid_post \\
        --output-dir ./out/

This is a generic wrapper around the DAME core that lives in
aiu_bench/monosemanticity/dominance_aware_auto_inter_score/. For the
family-specific scripts that reproduce the paper numbers, see
aiu_bench/monosemanticity/dominance_aware_auto_inter_score/runners/.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from aiu_bench.generic_runners._loader import load_sae


def parse_args():
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    p.add_argument("--sae-style", required=True, choices=["saelens", "local"])
    p.add_argument(
        "--sae-source",
        required=True,
        help="HuggingFace repo / SAELens id (saelens) or local checkpoint dir (local)",
    )
    p.add_argument(
        "--activation-source",
        required=True,
        help="Either path to cached activations (.pt) or 'model:hook' for on-the-fly",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--n-features",
        type=int,
        default=80,
        help="Total features to evaluate (default 80, i.e. 20 per quantile)",
    )
    p.add_argument("--n-per-quantile", type=int, default=20)
    p.add_argument("--n-validation-questions", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[run_dame] Loading SAE ({args.sae_style}: {args.sae_source})")
    sae_dict = load_sae(args.sae_style, args.sae_source)
    print(
        f"[run_dame] {sae_dict['architecture']} SAE: d_in={sae_dict['d_in']}, d_sae={sae_dict['d_sae']}"
    )

    # The DAME core entry point is in
    # aiu_bench/monosemanticity/dominance_aware_auto_inter_score/.
    # Wire your loaded SAE + activation source into the core pipeline below.
    # Family-specific examples (run_all_*.py) under that folder show the full
    # end-to-end pattern including the LLM-as-judge call.
    from aiu_bench.monosemanticity.dominance_aware_auto_inter_score.core import (
        dame_pipeline,
    )  # noqa

    # NOTE: dame_pipeline expects activations already cached. For on-the-fly
    # extraction, see analysis/feature_firing_analysis/runners/ as a reference.
    results = dame_pipeline.run(
        sae_weights={
            "encoder_weight": sae_dict["encoder_weight"],
            "decoder_weight": sae_dict["decoder_weight"],
            "encoder_bias": sae_dict["encoder_bias"],
            "decoder_bias": sae_dict["decoder_bias"],
        },
        activation_source=args.activation_source,
        n_features=args.n_features,
        n_per_quantile=args.n_per_quantile,
        n_validation_questions=args.n_validation_questions,
    )

    (out / "results.json").write_text(json.dumps(results, indent=2))
    config = {
        "analysis": "dame",
        "sae_style": args.sae_style,
        "sae_source": args.sae_source,
        "activation_source": args.activation_source,
        "n_features": args.n_features,
        "architecture": sae_dict["architecture"],
        "d_in": sae_dict["d_in"],
        "d_sae": sae_dict["d_sae"],
    }
    (out / "config.json").write_text(json.dumps(config, indent=2))
    print(f"[run_dame] Wrote {out}/results.json + config.json")
    if "summary" in results:
        tm = results["summary"].get("token_metrics", {})
        print(
            f"[run_dame] token-based perfect_ranking: {tm.get('perfect_ranking', float('nan')):.3f}"
        )


if __name__ == "__main__":
    main()
