"""Shared SAE loading helpers for the generic runners.

Two styles are supported out-of-the-box:
  - 'saelens': load via the SAELens library (or HuggingFace direct), using a
    release_id / sae_id pair or a HuggingFace repo + filename.
  - 'local':   load from a local directory containing 'config.json' and
    'weights.pt' (produced by our benchmark_training/ pipeline).

To add a new style for a custom SAE family, write a function that returns the
unified dict described in the README and register it in LOADERS at the bottom
of this file.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None  # type: ignore


def _check_torch():
    if torch is None:
        raise ImportError("torch is required for SAE loading. Install with `pip install torch`.")


# ----------------------------------------------------------------------------
# Style 1: SAELens / HuggingFace
# ----------------------------------------------------------------------------

def load_saelens(source: str) -> Dict[str, Any]:
    """Load an SAE via the SAELens library.

    `source` is interpreted as either:
      - a HuggingFace repo ID (e.g., 'jbloom/GPT2-Small-OAI-v5-32k-resid-post-SAEs')
      - a SAELens release_id:sae_id pair (e.g., 'gpt2-small-res-jb:blocks.6.hook_resid_pre')

    The SAELens library handles the actual I/O. We extract the underlying
    weight tensors and return them in the unified dict.
    """
    _check_torch()
    try:
        from sae_lens import SAE  # type: ignore
    except ImportError as e:
        raise ImportError(
            "SAELens is not installed. Install with `pip install sae-lens` or "
            "use --sae-style local with a checkpoint directory."
        ) from e

    if ':' in source:
        release_id, sae_id = source.split(':', 1)
        sae, _, _ = SAE.from_pretrained(release=release_id, sae_id=sae_id)
    else:
        # treat as HuggingFace repo id; SAELens supports this
        sae, _, _ = SAE.from_pretrained(release=source, sae_id='default')

    arch_str = _detect_architecture_from_saelens(sae)
    return {
        'encoder_weight': sae.W_enc.detach().cpu(),
        'decoder_weight': sae.W_dec.detach().cpu(),
        'encoder_bias': sae.b_enc.detach().cpu() if hasattr(sae, 'b_enc') else None,
        'decoder_bias': sae.b_dec.detach().cpu() if hasattr(sae, 'b_dec') else None,
        'd_in': int(sae.cfg.d_in),
        'd_sae': int(sae.cfg.d_sae),
        'architecture': arch_str,
        'sparsity_param': {},
        'source': source,
    }


def _detect_architecture_from_saelens(sae) -> str:
    """Best-effort architecture detection from a SAELens SAE config."""
    cfg = getattr(sae, 'cfg', None)
    if cfg is None:
        return 'standard'
    arch = getattr(cfg, 'architecture', None) or getattr(cfg, 'sae_class', None) or ''
    arch = str(arch).lower()
    for needle in ['matryoshka', 'batchtopk', 'jumprelu', 'panneal', 'gated', 'topk', 'standard']:
        if needle in arch:
            return {'standard': 'standard', 'topk': 'topk', 'batchtopk': 'batchtopk',
                    'matryoshka': 'matryoshka', 'jumprelu': 'jumprelu',
                    'gated': 'gated', 'panneal': 'panneal'}[needle]
    return 'standard'


# ----------------------------------------------------------------------------
# Style 2: Local directory
# ----------------------------------------------------------------------------

def load_local(source: str) -> Dict[str, Any]:
    """Load an SAE from a local directory containing config.json + weights.pt.

    Expected directory layout (matches our benchmark_training/ pipeline):
        source/
            config.json     # has fields: architecture, d_in, d_sae, ...
            weights.pt      # state dict with keys: encoder.weight, decoder.weight, ...
    """
    _check_torch()
    src = Path(source)
    cfg_path = src / 'config.json'
    weights_path = src / 'weights.pt'
    if not cfg_path.exists() or not weights_path.exists():
        raise FileNotFoundError(
            f"Expected {cfg_path} and {weights_path}. Got source={src}. "
            "If your local SAE has a different layout, adapt load_local() in _loader.py."
        )

    with open(cfg_path) as f:
        cfg = json.load(f)
    state = torch.load(weights_path, map_location='cpu')

    # Heuristic key resolution (handles both nested and flat state dicts)
    def _find(key_substrings):
        for k, v in state.items():
            kl = k.lower()
            if all(s in kl for s in key_substrings):
                return v
        return None

    enc_w = _find(['encoder', 'weight'])
    dec_w = _find(['decoder', 'weight'])
    enc_b = _find(['encoder', 'bias'])
    dec_b = _find(['decoder', 'bias'])

    if enc_w is None or dec_w is None:
        raise KeyError(
            "Could not find encoder/decoder weights in state dict. "
            f"Available keys: {list(state.keys())[:8]}..."
        )

    return {
        'encoder_weight': enc_w,
        'decoder_weight': dec_w,
        'encoder_bias': enc_b,
        'decoder_bias': dec_b,
        'd_in': int(cfg.get('d_in') or enc_w.shape[1]),
        'd_sae': int(cfg.get('d_sae') or enc_w.shape[0]),
        'architecture': cfg.get('architecture', 'standard'),
        'sparsity_param': cfg.get('sparsity_param') or {},
        'source': source,
    }


# ----------------------------------------------------------------------------
# Registry
# ----------------------------------------------------------------------------

LOADERS = {
    'saelens': load_saelens,
    'local':   load_local,
}


def load_sae(style: str, source: str) -> Dict[str, Any]:
    if style not in LOADERS:
        raise ValueError(f"Unknown SAE style '{style}'. Choices: {list(LOADERS)}")
    return LOADERS[style](source)
