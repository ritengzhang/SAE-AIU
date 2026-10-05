#!/usr/bin/env python3
"""Generic completeness (C2) runner template.

Computes the five completeness metrics across a single SAE:
  - mse, explained_variance (R^2), cossim, ce_loss_score, kl_div_score.

Usage:
    python run_completeness.py \\
        --sae-style local \\
        --sae-source ./training_runs/my_sae/checkpoint_final/ \\
        --activation-source ./activations/gpt2_layer6.pt \\
        --model-name gpt2 \\
        --output-dir ./out/completeness_my_sae/

The CE-loss and KL-divergence scores require running the model with SAE
reconstructions substituted at the hook point, so a model handle is needed
when computing those.
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
    p.add_argument("--activation-source", required=True)
    p.add_argument("--output-dir", required=True)
    p.add_argument(
        "--model-name",
        default=None,
        help="Required for CE/KL scores. Matches HuggingFace transformer name.",
    )
    p.add_argument(
        "--hook-name", default=None, help="Required with --model-name for CE/KL scores"
    )
    p.add_argument(
        "--metrics",
        default="mse,explained_variance,cossim,ce_loss_score,kl_div_score",
        help="Comma-separated subset of completeness metrics to compute",
    )
    p.add_argument("--max-batches", type=int, default=20)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[run_completeness] Loading SAE ({args.sae_style}: {args.sae_source})")
    sae_dict = load_sae(args.sae_style, args.sae_source)
    print(
        f"[run_completeness] {sae_dict['architecture']}: d_in={sae_dict['d_in']}, d_sae={sae_dict['d_sae']}"
    )

    from aiu_bench.completeness import completeness_analysis as ca  # noqa

    metrics = [m.strip() for m in args.metrics.split(",")]
    needs_model = ("ce_loss_score" in metrics) or ("kl_div_score" in metrics)
    if needs_model and (not args.model_name or not args.hook_name):
        print(
            "[run_completeness] WARN: --model-name + --hook-name not given; "
            "skipping ce_loss_score / kl_div_score"
        )
        metrics = [m for m in metrics if m not in ("ce_loss_score", "kl_div_score")]

    results = ca.evaluate_completeness(
        sae=sae_dict,
        activation_source=args.activation_source,
        model_name=args.model_name,
        hook_name=args.hook_name,
        metrics=metrics,
        max_batches=args.max_batches,
    )

    (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "config.json").write_text(
        json.dumps(
            {
                "analysis": "completeness",
                "sae_style": args.sae_style,
                "sae_source": args.sae_source,
                "activation_source": args.activation_source,
                "model_name": args.model_name,
                "hook_name": args.hook_name,
                "metrics": metrics,
                "architecture": sae_dict["architecture"],
                "d_in": sae_dict["d_in"],
                "d_sae": sae_dict["d_sae"],
            },
            indent=2,
        )
    )
    print(f"[run_completeness] Wrote {out}/results.json + config.json")
    sm = results.get("metrics", {})
    summary = " | ".join(
        f"{k}={v:.4f}" for k, v in sm.items() if isinstance(v, (int, float))
    )
    if summary:
        print(f"[run_completeness] {summary}")


if __name__ == "__main__":
    main()
