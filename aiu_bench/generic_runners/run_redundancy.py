#!/usr/bin/env python3
"""Generic redundancy (C1) runner template.

Computes per-quantile redundant-pair counts (decoder cosine similarity >=
threshold), with and without dead-neuron filtering, plus a threshold sweep.

Usage:
    python run_redundancy.py \\
        --sae-style local \\
        --sae-source ./training_runs/my_topk_sae/checkpoint_final/ \\
        --activation-source ./activations/gpt2_layer6.pt \\
        --output-dir ./out/redundancy_my_topk/

For the family-specific scripts that reproduce paper numbers, see
aiu_bench/redundancy/runners/run_all_*.py.
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
    p.add_argument("--sae-source", required=True)
    p.add_argument(
        "--activation-source",
        required=True,
        help="Either path to cached activations (.pt) or 'model:hook' for on-the-fly",
    )
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--threshold",
        type=float,
        default=0.7,
        help="Decoder cosine similarity threshold for redundant pairs (default 0.7)",
    )
    p.add_argument(
        "--threshold-sweep",
        action="store_true",
        help="Sweep threshold from 0.1 to 0.9 in steps of 0.1",
    )
    p.add_argument("--n-quantiles", type=int, default=4)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[run_redundancy] Loading SAE ({args.sae_style}: {args.sae_source})")
    sae_dict = load_sae(args.sae_style, args.sae_source)
    print(
        f"[run_redundancy] {sae_dict['architecture']}: d_in={sae_dict['d_in']}, d_sae={sae_dict['d_sae']}"
    )

    # The redundancy core lives in aiu_bench/redundancy/quantile_redundancy_analysis.py.
    # It needs (a) decoder weights for cosine similarity, (b) per-feature
    # mean-dominance scores for quantile ranking, (c) firing counts for
    # dead-neuron filtering. The first comes from sae_dict; the latter two
    # require a forward pass over activations, which the core handles.
    from aiu_bench.redundancy import quantile_redundancy_analysis as qra  # noqa

    thresholds = (
        [round(0.1 * i, 1) for i in range(1, 10)]
        if args.threshold_sweep
        else [args.threshold]
    )

    results = qra.analyze_redundancy(
        decoder_weight=sae_dict["decoder_weight"],
        encoder_weight=sae_dict["encoder_weight"],
        encoder_bias=sae_dict["encoder_bias"],
        activation_source=args.activation_source,
        thresholds=thresholds,
        n_quantiles=args.n_quantiles,
        compute_filtered=True,
    )

    (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "config.json").write_text(
        json.dumps(
            {
                "analysis": "redundancy",
                "sae_style": args.sae_style,
                "sae_source": args.sae_source,
                "activation_source": args.activation_source,
                "thresholds": thresholds,
                "n_quantiles": args.n_quantiles,
                "architecture": sae_dict["architecture"],
                "d_in": sae_dict["d_in"],
                "d_sae": sae_dict["d_sae"],
            },
            indent=2,
        )
    )
    print(f"[run_redundancy] Wrote {out}/results.json + config.json")


if __name__ == "__main__":
    main()
