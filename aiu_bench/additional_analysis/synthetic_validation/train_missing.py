"""Train only the missing SAE configs (9 new ones to bring all architectures to 15)."""
import torch
import json
import time
from pathlib import Path
from sae_models import StandardSAE, TopKSAE, BatchTopKSAE
from train_saes import train_sae

data_dir = Path("<DATA_ROOT>/toy_validation/data")
output_dir = Path("<DATA_ROOT>/toy_validation/trained_saes")

train_acts = torch.load(data_dir / "train_activations.pt")
with open(data_dir / "metadata.json") as f:
    metadata = json.load(f)
d = metadata["d"]

# New configs: TopK k=100, BatchTopK k=100, PAnneal l1=0.005
configs = []
for d_sae in [256, 512, 1024]:
    configs.append({"arch": "topk", "d_sae": d_sae, "k": 100, "name": f"topk_d{d_sae}_k100"})
    configs.append({"arch": "batchtopk", "d_sae": d_sae, "k": 100, "name": f"batchtopk_d{d_sae}_k100"})
    configs.append({"arch": "panneal", "d_sae": d_sae, "l1_coeff": 0.005, "name": f"panneal_d{d_sae}_l1_0.005"})

for i, cfg in enumerate(configs):
    save_path = output_dir / cfg["name"]
    if (save_path / "model.pt").exists():
        print(f"[{i+1}/{len(configs)}] Skipping {cfg['name']} (already exists)")
        continue
    print(f"\n[{i+1}/{len(configs)}] Training {cfg['name']}...")
    t0 = time.time()
    if cfg["arch"] == "topk":
        sae = TopKSAE(d, cfg["d_sae"], cfg["k"])
        l1_coeff = 0.0
    elif cfg["arch"] == "batchtopk":
        sae = BatchTopKSAE(d, cfg["d_sae"], cfg["k"])
        l1_coeff = 0.0
    elif cfg["arch"] == "panneal":
        sae = StandardSAE(d, cfg["d_sae"])
        l1_coeff = cfg["l1_coeff"]

    log = train_sae(sae, train_acts, l1_coeff=l1_coeff, lr=3e-4, n_epochs=20,
                    batch_size=4096, device="cuda", arch=cfg["arch"])
    elapsed = time.time() - t0
    print(f"  Done in {elapsed:.1f}s")

    save_path.mkdir(parents=True, exist_ok=True)
    torch.save(sae.state_dict(), save_path / "model.pt")
    result = {"config": cfg, "final_mse": log["mse_losses"][-1],
              "final_l0": log["l0s"][-1], "training_time": elapsed, "log": log}
    with open(save_path / "training_log.json", "w") as f:
        json.dump(result, f, indent=2)

print("\nDone training 9 new configs!")
