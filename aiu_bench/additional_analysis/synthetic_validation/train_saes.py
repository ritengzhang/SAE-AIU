"""
Train SAEs on synthetic activations with various configurations.

Trains Standard ReLU and TopK SAEs at different sparsity levels and dictionary sizes.
"""

import torch
import torch.nn.functional as F
import json
import time
from pathlib import Path
from sae_models import StandardSAE, TopKSAE, GatedSAE, JumpReLUSAE, BatchTopKSAE, MatryoshkaSAE


def train_sae(
    sae,
    train_data: torch.Tensor,
    l1_coeff: float = 0.0,
    lr: float = 3e-4,
    n_epochs: int = 5,
    batch_size: int = 4096,
    device: str = "cuda",
    normalize_decoder_freq: int = 100,
    arch: str = "standard",
) -> dict:
    """Train an SAE on synthetic activations.

    Args:
        sae: SAE model
        train_data: (n_samples, d) training activations
        l1_coeff: L1 sparsity penalty
        lr: learning rate
        n_epochs: number of passes through data
        batch_size: training batch size
        device: cuda or cpu
        normalize_decoder_freq: normalize decoder every N steps
        arch: architecture name (for special training logic)

    Returns:
        training log dict
    """
    import random as py_random
    sae = sae.to(device)
    train_data = train_data.to(device)
    optimizer = torch.optim.Adam(sae.parameters(), lr=lr)

    n_samples = train_data.shape[0]
    n_batches = n_samples // batch_size
    total_steps = n_epochs * n_batches
    step = 0
    log = {"losses": [], "mse_losses": [], "l1_losses": [], "l0s": []}

    # Matryoshka truncation sizes (powers of 2 up to d_sae)
    if arch == "matryoshka":
        trunc_sizes = []
        s = sae.d_sae
        while s >= max(16, sae.d_sae // 16):
            trunc_sizes.append(s)
            s //= 2
        trunc_sizes.append(sae.d_sae)  # also train full

    for epoch in range(n_epochs):
        perm = torch.randperm(n_samples, device=device)
        train_data_shuffled = train_data[perm]

        epoch_mse = 0.0
        epoch_l1 = 0.0
        epoch_l0 = 0.0

        for batch_idx in range(n_batches):
            x = train_data_shuffled[batch_idx * batch_size : (batch_idx + 1) * batch_size]

            # Architecture-specific forward pass
            if arch == "matryoshka":
                trunc = py_random.choice(trunc_sizes)
                out = sae(x, truncate_to=trunc)
            else:
                out = sae(x)

            x_hat = out["x_hat"]
            f = out["f"]

            # MSE loss
            mse_loss = F.mse_loss(x_hat, x)

            # L1 loss with optional progressive annealing
            if arch == "panneal":
                # Ramp L1 from 0 to l1_coeff over first 50% of training
                progress = step / total_steps
                effective_l1 = l1_coeff * min(1.0, progress / 0.5)
            else:
                effective_l1 = l1_coeff

            # For Gated SAE, apply L1 to gate pre-activations (pi_gate)
            if arch == "gated" and "gate_pre" in out:
                l1_loss = torch.sigmoid(out["gate_pre"]).mean() * effective_l1
            else:
                l1_loss = f.abs().mean() * effective_l1

            loss = mse_loss + l1_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if step % normalize_decoder_freq == 0:
                sae._normalize_decoder()

            l0 = (f > 0).float().sum(dim=-1).mean().item()

            epoch_mse += mse_loss.item()
            epoch_l1 += l1_loss.item()
            epoch_l0 += l0
            step += 1

        epoch_mse /= n_batches
        epoch_l1 /= n_batches
        epoch_l0 /= n_batches

        log["losses"].append(epoch_mse + epoch_l1)
        log["mse_losses"].append(epoch_mse)
        log["l1_losses"].append(epoch_l1)
        log["l0s"].append(epoch_l0)

        print(f"  Epoch {epoch+1}/{n_epochs}: MSE={epoch_mse:.6f}, L1={epoch_l1:.6f}, L0={epoch_l0:.1f}",
              flush=True)

    sae._normalize_decoder()
    return log


def train_all_configurations(
    data_dir: str = "<DATA_ROOT>/toy_validation/data",
    output_dir: str = "<DATA_ROOT>/toy_validation/trained_saes",
    device: str = "cuda",
):
    """Train all SAE configurations."""
    data_dir = Path(data_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load training data
    print("Loading training data...")
    train_acts = torch.load(data_dir / "train_activations.pt")
    with open(data_dir / "metadata.json") as f:
        metadata = json.load(f)
    d = metadata["d"]  # 256
    print(f"  Shape: {train_acts.shape}, d={d}")

    # Define configurations
    configs = []

    # Standard ReLU at different sparsity levels and widths
    # With 128 concepts in d=256, SAEs of 256-1024 are realistic
    for d_sae in [256, 512, 1024]:
        for l1_coeff in [0.005, 0.01, 0.03, 0.05, 0.1]:
            configs.append({
                "arch": "standard",
                "d_sae": d_sae,
                "l1_coeff": l1_coeff,
                "name": f"standard_d{d_sae}_l1_{l1_coeff}",
            })

    # TopK at different k values and widths
    for d_sae in [256, 512, 1024]:
        for k in [5, 10, 20, 50, 100]:
            configs.append({
                "arch": "topk",
                "d_sae": d_sae,
                "k": k,
                "name": f"topk_d{d_sae}_k{k}",
            })

    # Gated SAE at different sparsity levels
    for d_sae in [256, 512, 1024]:
        for l1_coeff in [0.005, 0.01, 0.03, 0.05, 0.1]:
            configs.append({
                "arch": "gated",
                "d_sae": d_sae,
                "l1_coeff": l1_coeff,
                "name": f"gated_d{d_sae}_l1_{l1_coeff}",
            })

    # JumpReLU SAE at different sparsity levels
    for d_sae in [256, 512, 1024]:
        for l1_coeff in [0.005, 0.01, 0.03, 0.05, 0.1]:
            configs.append({
                "arch": "jumprelu",
                "d_sae": d_sae,
                "l1_coeff": l1_coeff,
                "name": f"jumprelu_d{d_sae}_l1_{l1_coeff}",
            })

    # BatchTopK at different k values
    for d_sae in [256, 512, 1024]:
        for k in [5, 10, 20, 50, 100]:
            configs.append({
                "arch": "batchtopk",
                "d_sae": d_sae,
                "k": k,
                "name": f"batchtopk_d{d_sae}_k{k}",
            })

    # Matryoshka SAE (Standard-like but trained with random truncation)
    for d_sae in [256, 512, 1024]:
        for l1_coeff in [0.005, 0.01, 0.03, 0.05, 0.1]:
            configs.append({
                "arch": "matryoshka",
                "d_sae": d_sae,
                "l1_coeff": l1_coeff,
                "name": f"matryoshka_d{d_sae}_l1_{l1_coeff}",
            })

    # PAnneal (Standard architecture, progressive L1 annealing)
    for d_sae in [256, 512, 1024]:
        for l1_coeff in [0.005, 0.01, 0.03, 0.05, 0.1]:
            configs.append({
                "arch": "panneal",
                "d_sae": d_sae,
                "l1_coeff": l1_coeff,
                "name": f"panneal_d{d_sae}_l1_{l1_coeff}",
            })

    print(f"\nTotal configurations to train: {len(configs)}")

    all_results = {}

    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] Training {cfg['name']}...")
        t0 = time.time()

        # Create SAE
        if cfg["arch"] == "standard":
            sae = StandardSAE(d, cfg["d_sae"])
            l1_coeff = cfg["l1_coeff"]
        elif cfg["arch"] == "topk":
            sae = TopKSAE(d, cfg["d_sae"], cfg["k"])
            l1_coeff = 0.0  # TopK doesn't need L1
        elif cfg["arch"] == "gated":
            sae = GatedSAE(d, cfg["d_sae"])
            l1_coeff = cfg["l1_coeff"]
        elif cfg["arch"] == "jumprelu":
            sae = JumpReLUSAE(d, cfg["d_sae"])
            l1_coeff = cfg["l1_coeff"]
        elif cfg["arch"] == "batchtopk":
            sae = BatchTopKSAE(d, cfg["d_sae"], cfg["k"])
            l1_coeff = 0.0  # BatchTopK doesn't need L1
        elif cfg["arch"] == "matryoshka":
            sae = MatryoshkaSAE(d, cfg["d_sae"])
            l1_coeff = cfg["l1_coeff"]
        elif cfg["arch"] == "panneal":
            sae = StandardSAE(d, cfg["d_sae"])  # Same architecture, different training
            l1_coeff = cfg["l1_coeff"]
        else:
            raise ValueError(f"Unknown architecture: {cfg['arch']}")

        # Train
        log = train_sae(
            sae, train_acts,
            l1_coeff=l1_coeff,
            lr=3e-4,
            n_epochs=20,
            batch_size=4096,
            device=device,
            arch=cfg["arch"],
        )

        elapsed = time.time() - t0
        print(f"  Done in {elapsed:.1f}s")

        # Save model and config
        save_path = output_dir / cfg["name"]
        save_path.mkdir(parents=True, exist_ok=True)
        torch.save(sae.state_dict(), save_path / "model.pt")

        result = {
            "config": cfg,
            "final_mse": log["mse_losses"][-1],
            "final_l0": log["l0s"][-1],
            "training_time": elapsed,
            "log": log,
        }
        with open(save_path / "training_log.json", "w") as f:
            json.dump(result, f, indent=2)

        all_results[cfg["name"]] = result

    # Save summary
    summary_path = output_dir / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\nAll models saved to {output_dir}")
    print(f"Summary: {summary_path}")
    return all_results


if __name__ == "__main__":
    import sys
    device = sys.argv[1] if len(sys.argv) > 1 else "cuda"
    train_all_configurations(device=device)
