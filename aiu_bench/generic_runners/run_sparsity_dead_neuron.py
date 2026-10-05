#!/usr/bin/env python3
"""Generic sparsity (C5) + dead-neuron (C4) runner template.

Computes:
  - average_active_features (L_0)
  - effective_active_features at 75% / 90% magnitude coverage
  - dead-neuron proportions at three thresholds:
      * never (firing count = 0)
      * very_rare (firing < 1e-7)
      * rare      (firing < 1e-5, the paper's standard "dead" definition)

Usage:
    python run_sparsity_dead_neuron.py \\
        --sae-style saelens \\
        --sae-source jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs \\
        --activation-source gpt2-small:blocks.6.hook_resid_post \\
        --output-dir ./out/sparsity_jbloom/
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
    p.add_argument("--max-batches", type=int, default=50)
    return p.parse_args()


def main():
    args = parse_args()
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[run_sparsity_dead] Loading SAE ({args.sae_style}: {args.sae_source})")
    sae_dict = load_sae(args.sae_style, args.sae_source)
    print(
        f"[run_sparsity_dead] {sae_dict['architecture']}: d_in={sae_dict['d_in']}, d_sae={sae_dict['d_sae']}"
    )

    from aiu_bench.sparsity_dead_neuron import sparsity_and_deadneuron_analysis as sda  # noqa

    results = sda.analyze(
        sae=sae_dict,
        activation_source=args.activation_source,
        max_batches=args.max_batches,
    )

    (out / "results.json").write_text(json.dumps(results, indent=2))
    (out / "config.json").write_text(
        json.dumps(
            {
                "analysis": "sparsity_dead_neuron",
                "sae_style": args.sae_style,
                "sae_source": args.sae_source,
                "activation_source": args.activation_source,
                "architecture": sae_dict["architecture"],
                "d_in": sae_dict["d_in"],
                "d_sae": sae_dict["d_sae"],
            },
            indent=2,
        )
    )
    print(f"[run_sparsity_dead] Wrote {out}/results.json + config.json")
    sm = results.get("sparsity_metrics", {})
    dnm = results.get("dead_neuron_metrics", {})
    if sm:
        print(
            f"[run_sparsity_dead] avg_active = {sm.get('average_active_features', float('nan')):.1f}"
        )
    if dnm:
        print(
            f"[run_sparsity_dead] dead (rare): {dnm.get('dead_neuron_proportion_rare', float('nan')) * 100:.2f}%"
        )


if __name__ == "__main__":
    main()
