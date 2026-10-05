"""
Generate synthetic activations with known ground-truth concepts.

Design:
- d=256 dimensional activation space
- n_clusters=64 concept clusters, each with ~8 sub-concepts = 512 total concepts
- Sub-concepts within a cluster share similar directions (cosine sim ~0.85-0.95)
- Power-law frequency distribution: some concepts are much more frequent
- Per input: sample k active concepts, combine linearly with random coefficients
- This creates natural dominance imbalance for testing DAME insights

The cluster structure is critical for proxy validation:
- Features encoding sub-concepts from the same cluster have similar decoder directions
- This lets us validate decoder cosine similarity as a redundancy proxy
"""

import torch
import numpy as np
import json
from pathlib import Path


def generate_concept_directions(
    n_clusters: int, concepts_per_cluster: int, d: int,
    within_cluster_noise: float = 0.3, seed: int = 42,
) -> tuple:
    """Generate concept directions organized in clusters.

    Each cluster has a random base direction. Sub-concepts are perturbations
    of the base direction by adding a random vector with controlled NORM
    (not per-component std). This ensures within-cluster cosine similarity
    of ~1/(1+noise^2) ≈ 0.92 for noise=0.3, regardless of d.

    Returns:
        directions: (n_concepts, d) tensor of unit vectors
        cluster_ids: (n_concepts,) array of cluster assignments
    """
    rng = torch.Generator().manual_seed(seed)
    n_concepts = n_clusters * concepts_per_cluster

    # Generate cluster centers
    centers = torch.randn(n_clusters, d, generator=rng)
    centers = centers / centers.norm(dim=1, keepdim=True)

    directions = torch.zeros(n_concepts, d)
    cluster_ids = np.zeros(n_concepts, dtype=np.int32)

    for c in range(n_clusters):
        for j in range(concepts_per_cluster):
            idx = c * concepts_per_cluster + j
            # Perturb center direction with controlled-norm noise
            noise = torch.randn(d, generator=rng)
            noise = noise / noise.norm() * within_cluster_noise  # ||noise|| = within_cluster_noise
            direction = centers[c] + noise
            directions[idx] = direction / direction.norm()
            cluster_ids[idx] = c

    return directions, cluster_ids


def generate_concept_frequencies(n_concepts: int, alpha: float = 0.8, seed: int = 42) -> np.ndarray:
    """Generate power-law activation probabilities for each concept.

    Concept i has probability proportional to (i+1)^(-alpha).
    This means concept 0 fires very often (dominant), concept n-1 fires rarely.

    Returns: (n_concepts,) array of activation probabilities
    """
    np.random.seed(seed)
    ranks = np.arange(1, n_concepts + 1, dtype=np.float64)
    probs = ranks ** (-alpha)
    # Normalize to reasonable range: most frequent ~0.3, least frequent ~0.02
    probs = probs / probs.max() * 0.3
    probs = np.clip(probs, 0.02, 0.5)
    return probs


def generate_activations(
    concept_directions: torch.Tensor,
    concept_probs: np.ndarray,
    n_samples: int = 200000,
    noise_std: float = 0.05,
    seed: int = 42,
) -> tuple:
    """Generate synthetic activations as sparse combinations of concepts.

    For each sample:
    1. Each concept is active with probability concept_probs[i]
    2. Active concepts get random positive coefficients ~ Exponential(1)
    3. Activation = sum(coeff_i * direction_i) + noise

    Returns:
        activations: (n_samples, d) tensor
        concept_active: (n_samples, n_concepts) bool tensor - which concepts are active
        concept_coeffs: (n_samples, n_concepts) tensor - activation coefficients (0 if inactive)
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    n_concepts, d = concept_directions.shape

    # Sample which concepts are active per input
    concept_active = torch.from_numpy(
        np.random.random((n_samples, n_concepts)) < concept_probs[None, :]
    )

    # Ensure at least 1 concept active per input
    no_active = concept_active.sum(dim=1) == 0
    if no_active.any():
        for idx in no_active.nonzero(as_tuple=True)[0]:
            concept_active[idx, np.random.randint(n_concepts)] = True

    # Sample activation coefficients (positive, exponential distribution)
    concept_coeffs = torch.from_numpy(
        np.random.exponential(1.0, (n_samples, n_concepts))
    ).float()
    concept_coeffs[~concept_active] = 0.0

    # Compute activations: x = sum(coeff_i * direction_i) + noise
    activations = concept_coeffs @ concept_directions  # (n_samples, d)

    if noise_std > 0:
        noise = torch.randn_like(activations) * noise_std
        activations = activations + noise

    return activations, concept_active, concept_coeffs


def compute_concept_stats(
    concept_active: torch.Tensor, concept_coeffs: torch.Tensor,
    concept_directions: torch.Tensor, cluster_ids: np.ndarray,
) -> dict:
    """Compute statistics about the generated data."""
    n_samples, n_concepts = concept_active.shape

    firing_freq = concept_active.float().mean(dim=0)
    active_per_input = concept_active.float().sum(dim=1)

    # Dominance
    total_coeffs = concept_coeffs.sum(dim=1, keepdim=True).clamp(min=1e-8)
    dominance = (concept_coeffs / total_coeffs).mean(dim=0)

    # Within-cluster cosine similarity
    n_clusters = cluster_ids.max() + 1
    within_sims = []
    for c in range(n_clusters):
        idx = np.where(cluster_ids == c)[0]
        if len(idx) < 2:
            continue
        dirs = concept_directions[idx]
        sim = dirs @ dirs.T
        mask = torch.triu(torch.ones(len(idx), len(idx), dtype=torch.bool), diagonal=1)
        within_sims.extend(sim[mask].tolist())

    return {
        "n_samples": n_samples,
        "n_concepts": n_concepts,
        "n_clusters": int(n_clusters),
        "mean_active_per_input": active_per_input.mean().item(),
        "std_active_per_input": active_per_input.std().item(),
        "min_firing_freq": firing_freq.min().item(),
        "max_firing_freq": firing_freq.max().item(),
        "median_firing_freq": firing_freq.median().item(),
        "top10_dominance": dominance.topk(10).values.tolist(),
        "bottom10_dominance": dominance.sort().values[:10].tolist(),
        "within_cluster_sim_mean": float(np.mean(within_sims)),
        "within_cluster_sim_min": float(np.min(within_sims)),
        "within_cluster_sim_max": float(np.max(within_sims)),
    }


def generate_and_save(output_dir: str = "<DATA_ROOT>/toy_validation/data"):
    """Generate and save all synthetic data."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Parameters
    d = 256
    n_clusters = 32
    concepts_per_cluster = 4
    n_concepts = n_clusters * concepts_per_cluster  # 128
    n_train = 200000
    n_eval = 50000
    alpha = 0.8
    noise_std = 0.05
    within_cluster_noise = 0.15

    print(f"Generating synthetic data: d={d}, n_concepts={n_concepts}")
    print(f"  {n_clusters} clusters x {concepts_per_cluster} sub-concepts each")
    print(f"  Within-cluster perturbation noise: {within_cluster_noise}")
    print(f"  Training samples: {n_train}, Eval samples: {n_eval}")
    print(f"  Power-law alpha: {alpha}, Noise std: {noise_std}")

    # Generate concept directions and frequencies
    concept_directions, cluster_ids = generate_concept_directions(
        n_clusters, concepts_per_cluster, d,
        within_cluster_noise=within_cluster_noise,
    )
    concept_probs = generate_concept_frequencies(n_concepts, alpha=alpha)

    print(f"\nConcept frequency range: [{concept_probs.min():.3f}, {concept_probs.max():.3f}]")
    print(f"Top 5 concept probs: {concept_probs[:5]}")
    print(f"Bottom 5 concept probs: {concept_probs[-5:]}")

    # Generate training data
    print("\nGenerating training data...")
    train_acts, train_active, train_coeffs = generate_activations(
        concept_directions, concept_probs, n_samples=n_train, noise_std=noise_std, seed=42
    )
    train_stats = compute_concept_stats(train_active, train_coeffs, concept_directions, cluster_ids)
    print(f"  Mean concepts active per input: {train_stats['mean_active_per_input']:.1f}")
    print(f"  Firing freq range: [{train_stats['min_firing_freq']:.3f}, {train_stats['max_firing_freq']:.3f}]")
    print(f"  Within-cluster concept cosine sim: {train_stats['within_cluster_sim_mean']:.3f} "
          f"(range: {train_stats['within_cluster_sim_min']:.3f}-{train_stats['within_cluster_sim_max']:.3f})")

    # Generate eval data (different seed)
    print("Generating eval data...")
    eval_acts, eval_active, eval_coeffs = generate_activations(
        concept_directions, concept_probs, n_samples=n_eval, noise_std=noise_std, seed=123
    )

    # Save everything
    print(f"\nSaving to {output_dir}...")
    torch.save(concept_directions, output_dir / "concept_directions.pt")
    np.save(output_dir / "concept_probs.npy", concept_probs)
    np.save(output_dir / "cluster_ids.npy", cluster_ids)
    torch.save(train_acts, output_dir / "train_activations.pt")
    torch.save(train_active, output_dir / "train_concept_active.pt")
    torch.save(train_coeffs, output_dir / "train_concept_coeffs.pt")
    torch.save(eval_acts, output_dir / "eval_activations.pt")
    torch.save(eval_active, output_dir / "eval_concept_active.pt")
    torch.save(eval_coeffs, output_dir / "eval_concept_coeffs.pt")

    # Save metadata
    metadata = {
        "d": d,
        "n_concepts": n_concepts,
        "n_clusters": n_clusters,
        "concepts_per_cluster": concepts_per_cluster,
        "within_cluster_noise": within_cluster_noise,
        "n_train": n_train,
        "n_eval": n_eval,
        "alpha": alpha,
        "noise_std": noise_std,
        "train_stats": train_stats,
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print("Done!")
    return metadata


if __name__ == "__main__":
    generate_and_save()
