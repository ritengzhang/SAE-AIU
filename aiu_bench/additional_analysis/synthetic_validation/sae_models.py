"""
Simple SAE implementations for toy validation.
Self-contained, no SAELens dependency needed.

Architectures:
1. Standard ReLU SAE with L1 sparsity penalty
2. TopK SAE with hard top-k selection
3. Gated SAE with learned gating mechanism
4. JumpReLU SAE with learnable threshold
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class StandardSAE(nn.Module):
    """Standard ReLU SAE with L1 sparsity penalty."""

    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae

        # Encoder: W_enc (d_in -> d_sae) + bias
        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))

        # Decoder: W_dec (d_sae -> d_in), columns are unit vectors
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_in) * 0.01)
        self._normalize_decoder()

        # Decoder bias (subtracted before encoding, added after decoding)
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def _normalize_decoder(self):
        """Normalize decoder columns to unit vectors."""
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Encode: subtract bias, linear transform, ReLU."""
        x_centered = x - self.b_dec
        pre_acts = x_centered @ self.W_enc + self.b_enc
        return F.relu(pre_acts)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        """Decode: linear transform + add bias."""
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> dict:
        f = self.encode(x)
        x_hat = self.decode(f)
        return {"x_hat": x_hat, "f": f}

    @property
    def decoder_directions(self) -> torch.Tensor:
        """Return normalized decoder weight vectors (d_sae, d_in)."""
        return F.normalize(self.W_dec.data, dim=1)


class TopKSAE(nn.Module):
    """TopK SAE with hard top-k activation selection."""

    def __init__(self, d_in: int, d_sae: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k

        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_in) * 0.01)
        self._normalize_decoder()
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def _normalize_decoder(self):
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_dec
        pre_acts = x_centered @ self.W_enc + self.b_enc
        # Apply ReLU first, then keep only top-k
        acts = F.relu(pre_acts)
        # Top-k selection
        topk_vals, topk_idx = acts.topk(self.k, dim=-1)
        result = torch.zeros_like(acts)
        result.scatter_(-1, topk_idx, topk_vals)
        return result

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> dict:
        f = self.encode(x)
        x_hat = self.decode(f)
        return {"x_hat": x_hat, "f": f}

    @property
    def decoder_directions(self) -> torch.Tensor:
        return F.normalize(self.W_dec.data, dim=1)


class BatchTopKSAE(nn.Module):
    """BatchTopK SAE: top-k selection across the entire batch (Bussmann et al., 2024)."""

    def __init__(self, d_in: int, d_sae: int, k: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae
        self.k = k

        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_in) * 0.01)
        self._normalize_decoder()
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def _normalize_decoder(self):
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_dec
        pre_acts = F.relu(x_centered @ self.W_enc + self.b_enc)
        # Batch-level top-k: keep k * batch_size activations total
        batch_size = x.shape[0]
        total_k = self.k * batch_size
        flat = pre_acts.reshape(-1)
        if total_k < flat.numel():
            threshold = flat.topk(total_k).values[-1]
            return pre_acts * (pre_acts >= threshold).float()
        return pre_acts

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> dict:
        f = self.encode(x)
        x_hat = self.decode(f)
        return {"x_hat": x_hat, "f": f}

    @property
    def decoder_directions(self) -> torch.Tensor:
        return F.normalize(self.W_dec.data, dim=1)


class MatryoshkaSAE(nn.Module):
    """Matryoshka SAE: nested feature groups trained with random truncation (Bricken et al., 2024-style)."""

    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae

        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_in) * 0.01)
        self._normalize_decoder()
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def _normalize_decoder(self):
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_dec
        return F.relu(x_centered @ self.W_enc + self.b_enc)

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor, truncate_to: int = None) -> dict:
        f = self.encode(x)
        if truncate_to is not None and truncate_to < self.d_sae:
            f_trunc = f.clone()
            f_trunc[:, truncate_to:] = 0
            x_hat = self.decode(f_trunc)
            return {"x_hat": x_hat, "f": f_trunc}
        x_hat = self.decode(f)
        return {"x_hat": x_hat, "f": f}

    @property
    def decoder_directions(self) -> torch.Tensor:
        return F.normalize(self.W_dec.data, dim=1)


class GatedSAE(nn.Module):
    """Gated SAE: separate gating and magnitude pathways (Rajamanoharan et al., 2024)."""

    def __init__(self, d_in: int, d_sae: int):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae

        # Magnitude pathway
        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        # Gate pathway (separate weights)
        self.W_gate = nn.Parameter(torch.randn(d_in, d_sae) * 0.01)
        self.b_gate = nn.Parameter(torch.zeros(d_sae))
        # Decoder
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_in) * 0.01)
        self._normalize_decoder()
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def _normalize_decoder(self):
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_dec
        magnitude = F.relu(x_centered @ self.W_enc + self.b_enc)
        gate = torch.sigmoid(x_centered @ self.W_gate + self.b_gate)
        return magnitude * gate

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> dict:
        f = self.encode(x)
        x_hat = self.decode(f)
        # For L1 penalty, use gate pre-activations
        x_centered = x - self.b_dec
        gate_pre = x_centered @ self.W_gate + self.b_gate
        return {"x_hat": x_hat, "f": f, "gate_pre": gate_pre}

    @property
    def decoder_directions(self) -> torch.Tensor:
        return F.normalize(self.W_dec.data, dim=1)


class JumpReLUSAE(nn.Module):
    """JumpReLU SAE: ReLU with learnable per-feature threshold (Rajamanoharan et al., 2024)."""

    def __init__(self, d_in: int, d_sae: int, init_threshold: float = 0.1):
        super().__init__()
        self.d_in = d_in
        self.d_sae = d_sae

        self.W_enc = nn.Parameter(torch.randn(d_in, d_sae) * 0.01)
        self.b_enc = nn.Parameter(torch.zeros(d_sae))
        # Learnable threshold (log-space for positivity)
        self.log_threshold = nn.Parameter(torch.full((d_sae,), float(torch.tensor(init_threshold).log())))
        self.W_dec = nn.Parameter(torch.randn(d_sae, d_in) * 0.01)
        self._normalize_decoder()
        self.b_dec = nn.Parameter(torch.zeros(d_in))

    def _normalize_decoder(self):
        with torch.no_grad():
            self.W_dec.data = F.normalize(self.W_dec.data, dim=1)

    @property
    def threshold(self):
        return self.log_threshold.exp()

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x_centered = x - self.b_dec
        pre_acts = x_centered @ self.W_enc + self.b_enc
        # JumpReLU: zero out if below threshold, keep value if above
        mask = (pre_acts > self.threshold).float()
        # Straight-through estimator for gradient
        return F.relu(pre_acts) * mask

    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    def forward(self, x: torch.Tensor) -> dict:
        f = self.encode(x)
        x_hat = self.decode(f)
        return {"x_hat": x_hat, "f": f}

    @property
    def decoder_directions(self) -> torch.Tensor:
        return F.normalize(self.W_dec.data, dim=1)
