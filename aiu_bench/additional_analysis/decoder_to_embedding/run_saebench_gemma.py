#!/usr/bin/env python3
"""
Decoder Similarity → Semantic Alignment validation for SAE Bench Gemma.

For each SAE:
1. Load SAE + base model
2. Collect top-K activating sequences per feature (tracking token IDs + max-act position)
3. Compute decoder cosine similarity matrix
4. Sample pairs per decoder similarity bin
5. Embed top-activation windows with SentenceTransformer (sweep over window sizes)
6. Compute cross-pair embedding similarity
7. Save per-SAE results

Usage:
    python -m rebuttal_redundancy_to_semantic_alignment.run_saebench_gemma --device cuda:0
"""

import argparse
import gc
import heapq
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

sae_lens_dir = parent_dir / "SAELens-main"
sys.path.insert(0, str(sae_lens_dir))

from sae_family_load_convert.saebench import (
    load_base_model,
    setup_activations_store,
)
from sae_family_load_convert.saebench_gemma import (
    cleanup_sae_cache,
    load_saebench_gemma_sae,
)

# ── Config ──────────────────────────────────────────────────────────────────

SAE_BENCH_GEMMA_CONFIG = {
    "repos": {
        "4k": "adamkarvonen/saebench_gemma-2-2b_width-2pow12_date-0108",
        "16k": "canrager/saebench_gemma-2-2b_width-2pow14_date-0107",
        "65k": "canrager/saebench_gemma-2-2b_width-2pow16_date-0107",
    },
    "layer": "resid_post_layer_12",
    "sae_types": [
        "Standard_gemma-2-2b__0108",
        "TopK_gemma-2-2b__0108",
        "BatchTopK_gemma-2-2b__0108",
        "JumpRelu_gemma-2-2b__0108",
        "GatedSAE_gemma-2-2b__0108",
        "PAnneal_gemma-2-2b__0108",
        "MatryoshkaBatchTopK_gemma-2-2b__0108",
    ],
}

TRAINERS = [
    "trainer_0",
    "trainer_1",
    "trainer_2",
    "trainer_3",
    "trainer_4",
    "trainer_5",
]

ARCH_MAP = {
    "Standard": "Standard",
    "TopK": "TopK",
    "BatchTopK": "BatchTopK",
    "JumpRelu": "JumpReLU",
    "GatedSAE": "Gated",
    "PAnneal": "PAnneal",
    "MatryoshkaBatchTopK": "Matryoshka",
}

# Top-K sequences to keep per feature
TOP_K_SEQS = 20

# Number of batches to collect activations from
N_COLLECTION_BATCHES = 50

# Decoder similarity bins: [low, high)
SIMILARITY_BINS = [
    (0.0, 0.3),
    (0.3, 0.5),
    (0.5, 0.7),
    (0.7, 0.9),
    (0.9, 1.01),  # 1.01 to include 1.0
]

# Number of pairs to sample per bin
PAIRS_PER_BIN = 300

# Embedding window sizes (tokens centered on max-activation token)
WINDOW_SIZES = [8, 16, 32, 64]

# Sentence transformer model
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"


# ── Top activation collection ──────────────────────────────────────────────


class TopActivationTracker:
    """
    Track top-K activating sequences per feature using min-heaps.

    For each feature, maintains a heap of (max_activation, seq_index, max_token_pos).
    We keep the top-K sequences by highest max_activation.
    """

    def __init__(self, d_sae: int, top_k: int = TOP_K_SEQS):
        self.d_sae = d_sae
        self.top_k = top_k
        # Per-feature min-heap: (activation_value, global_seq_idx, max_token_pos)
        self.heaps: List[list] = [[] for _ in range(d_sae)]
        # Store all sequences' token IDs: global_seq_idx -> token_ids
        self.all_tokens: Dict[int, np.ndarray] = {}
        self.global_seq_count = 0

    def process_batch(
        self,
        feature_acts: np.ndarray,  # [batch_size, seq_len, d_sae]
        token_ids: np.ndarray,  # [batch_size, seq_len]
    ):
        """Process one batch: update heaps for all features."""
        batch_size, seq_len, d_sae = feature_acts.shape

        # For each sequence in batch, compute max activation per feature
        # max_acts: [batch_size, d_sae], max_positions: [batch_size, d_sae]
        max_acts = feature_acts.max(axis=1)  # [batch_size, d_sae]
        max_positions = feature_acts.argmax(axis=1)  # [batch_size, d_sae]

        for seq_i in range(batch_size):
            global_idx = self.global_seq_count + seq_i
            self.all_tokens[global_idx] = token_ids[seq_i]

            for feat_j in range(d_sae):
                act_val = float(max_acts[seq_i, feat_j])
                if act_val <= 0:
                    continue  # feature didn't fire

                token_pos = int(max_positions[seq_i, feat_j])
                entry = (act_val, global_idx, token_pos)

                heap = self.heaps[feat_j]
                if len(heap) < self.top_k:
                    heapq.heappush(heap, entry)
                elif act_val > heap[0][0]:
                    heapq.heapreplace(heap, entry)

        self.global_seq_count += batch_size

    def get_top_sequences(
        self, feature_idx: int
    ) -> List[Tuple[float, np.ndarray, int]]:
        """
        Get top-K sequences for a feature, sorted descending by activation.

        Returns: List of (activation_value, token_ids, max_token_pos)
        """
        heap = self.heaps[feature_idx]
        results = []
        for act_val, global_idx, token_pos in sorted(heap, reverse=True):
            if global_idx in self.all_tokens:
                results.append((act_val, self.all_tokens[global_idx], token_pos))
        return results

    def prune_unused_tokens(self):
        """Remove stored tokens that aren't referenced by any feature's top-K."""
        used_indices = set()
        for heap in self.heaps:
            for _, global_idx, _ in heap:
                used_indices.add(global_idx)

        to_remove = set(self.all_tokens.keys()) - used_indices
        for idx in to_remove:
            del self.all_tokens[idx]
        print(
            f"  Pruned {len(to_remove)} unused sequences, kept {len(self.all_tokens)}"
        )


def collect_top_activations(
    model,
    sae,
    metadata: dict,
    device: str,
    width_name: str,
    n_batches: int = N_COLLECTION_BATCHES,
) -> TopActivationTracker:
    """
    Collect top-K activating sequences per feature.

    Uses the ActivationsStore to get batches. Since ActivationsStore returns
    flattened token activations [batch_tokens, d_in], we need to also get
    the raw token IDs from the store to track sequences.
    """
    d_sae = sae.cfg.d_sae
    tracker = TopActivationTracker(d_sae=d_sae)

    # We need access to the underlying dataset to get token IDs.
    # The ActivationsStore flattens sequences, so we work with the dataset directly.
    from datasets import load_dataset

    dataset_path = metadata.get("dataset_path", "NeelNanda/pile-10k")
    hook_name = metadata["hook_name"]

    # Context size matches what ActivationsStore uses for Gemma
    context_size = 128

    print(f"  Loading dataset: {dataset_path}")
    ds = load_dataset(dataset_path, split="train", streaming=True)
    ds_iter = iter(ds)

    # Get tokenizer for tokenizing raw text
    tokenizer = model.tokenizer

    batch_size = 4  # small for Gemma memory
    seqs_collected = 0

    # Pre-tokenize: buffer raw text, tokenize, chunk into context_size segments
    token_buffer = []

    def refill_token_buffer(min_seqs: int = 20):
        """Tokenize raw text into context_size chunks."""
        while len(token_buffer) < min_seqs:
            try:
                item = next(ds_iter)
            except StopIteration:
                break
            text = item["text"]
            token_ids = tokenizer.encode(text, add_special_tokens=False)
            # Chunk into context_size segments
            for start in range(0, len(token_ids) - context_size + 1, context_size):
                token_buffer.append(token_ids[start : start + context_size])

    for batch_idx in range(n_batches):
        # Collect a batch of token sequences
        refill_token_buffer(min_seqs=batch_size * 2)

        token_batch = []
        for _ in range(batch_size):
            if not token_buffer:
                break
            token_batch.append(token_buffer.pop(0))

        if not token_batch:
            break

        token_ids = np.array(token_batch, dtype=np.int64)  # [batch_size, context_size]
        token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)

        # Run model to get hook activations
        with torch.no_grad():
            _, cache = model.run_with_cache(token_tensor, names_filter=[hook_name])
            hook_acts = cache[hook_name]  # [batch_size, context_size, d_in]

            # Encode through SAE
            flat_acts = hook_acts.reshape(-1, sae.cfg.d_in)  # [batch*ctx, d_in]
            feature_acts = sae.encode(flat_acts)  # [batch*ctx, d_sae]
            feature_acts = feature_acts.reshape(
                len(token_batch), context_size, d_sae
            )  # [batch, ctx, d_sae]

        # Track top activations
        tracker.process_batch(
            feature_acts=feature_acts.cpu().numpy(),
            token_ids=token_ids,
        )
        seqs_collected += len(token_batch)

        # Clean up
        del cache, hook_acts, flat_acts, feature_acts, token_tensor
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
            # Prune periodically to save memory
            tracker.prune_unused_tokens()

        if batch_idx % 10 == 0:
            print(
                f"    Batch {batch_idx + 1}/{n_batches}, seqs collected: {seqs_collected}"
            )

    # Final prune
    tracker.prune_unused_tokens()
    print(f"  Collection done: {seqs_collected} sequences processed")
    return tracker


# ── Decoder similarity + pair sampling ──────────────────────────────────────


def compute_decoder_similarity_matrix(decoder_weights: np.ndarray) -> np.ndarray:
    """Compute full pairwise cosine similarity. decoder_weights: [d_sae, d_in]"""
    norms = np.linalg.norm(decoder_weights, axis=1, keepdims=True) + 1e-8
    normed = decoder_weights / norms
    return normed @ normed.T


def sample_pairs_per_bin(
    sim_matrix: np.ndarray,
    bins: List[Tuple[float, float]],
    n_per_bin: int,
    alive_mask: Optional[np.ndarray] = None,
    rng: np.random.Generator = None,
) -> Dict[str, List[Tuple[int, int, float]]]:
    """
    Sample feature pairs from each decoder similarity bin.

    Args:
        sim_matrix: [d_sae, d_sae] cosine similarity
        bins: list of (low, high) similarity ranges
        n_per_bin: max pairs to sample per bin
        alive_mask: [d_sae] boolean, True if feature is alive (fires at all)
        rng: random generator

    Returns:
        Dict mapping bin label -> list of (feat_i, feat_j, similarity)
    """
    if rng is None:
        rng = np.random.default_rng(42)

    d_sae = sim_matrix.shape[0]

    # Get upper triangle indices
    triu_i, triu_j = np.triu_indices(d_sae, k=1)
    triu_sims = sim_matrix[triu_i, triu_j]

    # Filter to alive pairs if mask provided
    if alive_mask is not None:
        valid = alive_mask[triu_i] & alive_mask[triu_j]
        triu_i = triu_i[valid]
        triu_j = triu_j[valid]
        triu_sims = triu_sims[valid]

    result = {}
    for low, high in bins:
        label = f"{low:.1f}-{high:.1f}"
        mask = (triu_sims >= low) & (triu_sims < high)
        indices = np.where(mask)[0]

        if len(indices) == 0:
            result[label] = []
            continue

        # Sample
        if len(indices) > n_per_bin:
            chosen = rng.choice(indices, size=n_per_bin, replace=False)
        else:
            chosen = indices

        pairs = [
            (int(triu_i[idx]), int(triu_j[idx]), float(triu_sims[idx]))
            for idx in chosen
        ]
        result[label] = pairs
        print(f"    Bin {label}: {len(pairs)} pairs (from {len(indices)} available)")

    return result


# ── Embedding + similarity computation ──────────────────────────────────────


def extract_window_text(
    token_ids: np.ndarray,
    max_token_pos: int,
    window_size: int,
    tokenizer,
) -> str:
    """Extract a window of tokens centered on max_token_pos and decode to text."""
    seq_len = len(token_ids)
    half_w = window_size // 2
    start = max(0, max_token_pos - half_w)
    end = min(seq_len, max_token_pos + half_w)

    # If window would be too small at edges, expand the other direction
    if end - start < window_size:
        if start == 0:
            end = min(seq_len, start + window_size)
        elif end == seq_len:
            start = max(0, end - window_size)

    window_tokens = token_ids[start:end].tolist()
    return tokenizer.decode(window_tokens, skip_special_tokens=True)


def compute_cross_embedding_similarity(
    texts_a: List[str],
    texts_b: List[str],
    embed_model: SentenceTransformer,
) -> float:
    """
    Compute mean cosine similarity between two sets of texts.

    For each text in A, compute cosine similarity with each text in B,
    then return the overall mean.
    """
    if not texts_a or not texts_b:
        return float("nan")

    emb_a = embed_model.encode(texts_a, convert_to_numpy=True)  # [K, dim]
    emb_b = embed_model.encode(texts_b, convert_to_numpy=True)  # [K, dim]

    # Normalize
    emb_a = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-8)
    emb_b = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-8)

    # Cross similarity: [K_a, K_b]
    cross_sim = emb_a @ emb_b.T
    return float(np.mean(cross_sim))


def evaluate_pairs(
    pairs: List[Tuple[int, int, float]],
    tracker: TopActivationTracker,
    tokenizer,
    embed_model: SentenceTransformer,
    window_sizes: List[int] = WINDOW_SIZES,
) -> Dict[str, List[dict]]:
    """
    For each pair, compute cross-embedding similarity at each window size.

    Returns: {window_size_str: [{"feat_i": ..., "feat_j": ..., "decoder_sim": ..., "embed_sim": ...}, ...]}
    """
    results_by_window = {str(w): [] for w in window_sizes}

    for pair_idx, (feat_i, feat_j, dec_sim) in enumerate(pairs):
        top_seqs_i = tracker.get_top_sequences(feat_i)
        top_seqs_j = tracker.get_top_sequences(feat_j)

        if not top_seqs_i or not top_seqs_j:
            continue

        for w in window_sizes:
            texts_i = [
                extract_window_text(tok_ids, pos, w, tokenizer)
                for _, tok_ids, pos in top_seqs_i
            ]
            texts_j = [
                extract_window_text(tok_ids, pos, w, tokenizer)
                for _, tok_ids, pos in top_seqs_j
            ]

            # Filter empty strings
            texts_i = [t for t in texts_i if t.strip()]
            texts_j = [t for t in texts_j if t.strip()]

            embed_sim = compute_cross_embedding_similarity(
                texts_i, texts_j, embed_model
            )

            results_by_window[str(w)].append(
                {
                    "feat_i": feat_i,
                    "feat_j": feat_j,
                    "decoder_sim": dec_sim,
                    "embed_sim": embed_sim,
                }
            )

        if (pair_idx + 1) % 50 == 0:
            print(f"      Evaluated {pair_idx + 1}/{len(pairs)} pairs")

    return results_by_window


# ── Main runner ─────────────────────────────────────────────────────────────


def detect_architecture(sae_type_folder: str) -> str:
    for prefix, arch in ARCH_MAP.items():
        if sae_type_folder.startswith(prefix):
            return arch
    return "unknown"


def run_one_sae(
    repo_id: str,
    sae_type_folder: str,
    trainer: str,
    width_name: str,
    model,
    embed_model: SentenceTransformer,
    device: str,
    output_dir: Path,
):
    """Load one SAE, collect top activations, compute pair similarities, save."""
    arch = detect_architecture(sae_type_folder)

    out_path = output_dir / "saebench_gemma" / width_name / sae_type_folder / trainer
    results_file = out_path / "results.json"
    if results_file.exists():
        print(f"  SKIP (exists): {results_file}")
        return

    out_path.mkdir(parents=True, exist_ok=True)

    # Load SAE
    sae, metadata = load_saebench_gemma_sae(
        repo_id=repo_id,
        sae_type_folder=sae_type_folder,
        layer_folder=SAE_BENCH_GEMMA_CONFIG["layer"],
        trainer_folder=trainer,
        device=device,
    )

    d_sae = sae.cfg.d_sae
    print(f"  SAE loaded: d_sae={d_sae}, arch={arch}")

    # Step 1: Collect top activations
    print(f"  Step 1: Collecting top-{TOP_K_SEQS} activating sequences...")
    t0 = time.time()
    tracker = collect_top_activations(
        model=model,
        sae=sae,
        metadata=metadata,
        device=device,
        width_name=width_name,
        n_batches=N_COLLECTION_BATCHES,
    )
    print(f"  Collection took {time.time() - t0:.1f}s")

    # Step 2: Compute decoder similarity matrix
    print(f"  Step 2: Computing decoder similarity matrix...")
    decoder_weights = sae.W_dec.detach().cpu().numpy()
    sim_matrix = compute_decoder_similarity_matrix(decoder_weights)

    # Build alive mask: feature has at least 1 entry in its heap
    alive_mask = np.array([len(tracker.heaps[i]) > 0 for i in range(d_sae)])
    n_alive = alive_mask.sum()
    print(f"  Alive features: {n_alive}/{d_sae}")

    # Step 3: Sample pairs per bin
    print(f"  Step 3: Sampling pairs per similarity bin...")
    binned_pairs = sample_pairs_per_bin(
        sim_matrix=sim_matrix,
        bins=SIMILARITY_BINS,
        n_per_bin=PAIRS_PER_BIN,
        alive_mask=alive_mask,
    )

    # Free similarity matrix
    del sim_matrix, decoder_weights
    gc.collect()

    # Step 4: Evaluate pairs with embeddings at each window size
    print(f"  Step 4: Computing embedding similarities (sweep over window sizes)...")
    tokenizer = model.tokenizer

    all_results = {}
    for bin_label, pairs in binned_pairs.items():
        if not pairs:
            all_results[bin_label] = {str(w): [] for w in WINDOW_SIZES}
            continue
        print(f"    Bin {bin_label}: evaluating {len(pairs)} pairs...")
        all_results[bin_label] = evaluate_pairs(
            pairs=pairs,
            tracker=tracker,
            tokenizer=tokenizer,
            embed_model=embed_model,
            window_sizes=WINDOW_SIZES,
        )

    # Step 5: Compute summary statistics
    summary = {
        "metadata": {
            "architecture": arch,
            "sae_type": sae_type_folder,
            "width": width_name,
            "trainer": trainer,
            "repo_id": repo_id,
            "d_sae": d_sae,
            "n_alive": int(n_alive),
            "top_k_seqs": TOP_K_SEQS,
            "n_collection_batches": N_COLLECTION_BATCHES,
            "pairs_per_bin": PAIRS_PER_BIN,
            "window_sizes": WINDOW_SIZES,
            "embed_model": EMBED_MODEL_NAME,
        },
        "bins": {},
    }

    for bin_label in all_results:
        summary["bins"][bin_label] = {}
        for w_str, pair_results in all_results[bin_label].items():
            if not pair_results:
                summary["bins"][bin_label][w_str] = {
                    "n_pairs": 0,
                    "mean_embed_sim": None,
                    "std_embed_sim": None,
                    "mean_decoder_sim": None,
                }
                continue

            embed_sims = [
                r["embed_sim"] for r in pair_results if not np.isnan(r["embed_sim"])
            ]
            decoder_sims = [r["decoder_sim"] for r in pair_results]

            summary["bins"][bin_label][w_str] = {
                "n_pairs": len(pair_results),
                "mean_embed_sim": float(np.mean(embed_sims)) if embed_sims else None,
                "std_embed_sim": float(np.std(embed_sims)) if embed_sims else None,
                "median_embed_sim": float(np.median(embed_sims))
                if embed_sims
                else None,
                "mean_decoder_sim": float(np.mean(decoder_sims))
                if decoder_sims
                else None,
            }

    # Save full results (pair-level detail)
    full_output = {
        "summary": summary,
        "pair_details": {},
    }
    for bin_label in all_results:
        full_output["pair_details"][bin_label] = all_results[bin_label]

    with open(results_file, "w") as f:
        json.dump(full_output, f, indent=2)
    print(f"  SAVED: {results_file}")

    # Cleanup
    cleanup_sae_cache(metadata["_cache_path"])
    del tracker, sae, binned_pairs, all_results
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Decoder similarity → semantic alignment validation (SAE Bench Gemma)"
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="<DATA_ROOT>/rebuttal_redundancy_to_semantic_alignment",
    )
    parser.add_argument(
        "--widths",
        nargs="+",
        default=["4k"],
        help="Which widths to run (default: 4k). Options: 4k, 16k, 65k",
    )
    parser.add_argument(
        "--sae-types",
        nargs="+",
        default=None,
        help="Specific SAE types to run (default: all). E.g. Standard_gemma-2-2b__0108",
    )
    parser.add_argument(
        "--trainers",
        nargs="+",
        default=None,
        help="Specific trainers to run (default: all). E.g. trainer_0 trainer_1",
    )
    args = parser.parse_args()

    device = args.device
    if device is None:
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

    output_dir = Path(args.output_dir)

    sae_types = args.sae_types or SAE_BENCH_GEMMA_CONFIG["sae_types"]
    trainers = args.trainers or TRAINERS
    widths = args.widths

    total = len(sae_types) * len(widths) * len(trainers)
    print(f"Decoder→Semantic alignment: SAEBench Gemma ({total} SAEs)")
    print(f"  Widths: {widths}")
    print(f"  Window sizes: {WINDOW_SIZES}")

    # Load sentence transformer once
    print(f"Loading sentence transformer: {EMBED_MODEL_NAME}")
    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    print(f"  Loaded.")

    done = 0
    failed = []

    for width_name in widths:
        repo_id = SAE_BENCH_GEMMA_CONFIG["repos"][width_name]
        print(f"\n{'=' * 60}")
        print(f"Width: {width_name} ({repo_id})")
        print(f"{'=' * 60}")

        # Load base model once per width
        first_sae, first_meta = load_saebench_gemma_sae(
            repo_id=repo_id,
            sae_type_folder=sae_types[0],
            layer_folder=SAE_BENCH_GEMMA_CONFIG["layer"],
            trainer_folder="trainer_0",
            device=device,
        )
        model = load_base_model(first_meta["model_name"], device=device)
        cleanup_sae_cache(first_meta["_cache_path"])
        del first_sae
        gc.collect()
        torch.cuda.empty_cache()

        for sae_type in sae_types:
            for trainer in trainers:
                done += 1
                print(f"\n[{done}/{total}] {width_name}/{sae_type}/{trainer}")
                try:
                    run_one_sae(
                        repo_id=repo_id,
                        sae_type_folder=sae_type,
                        trainer=trainer,
                        width_name=width_name,
                        model=model,
                        embed_model=embed_model,
                        device=device,
                        output_dir=output_dir,
                    )
                except Exception as e:
                    import traceback

                    print(f"  FAILED: {e}")
                    traceback.print_exc()
                    failed.append(f"{width_name}/{sae_type}/{trainer}: {e}")

        # Cleanup model
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}")
    print(f"Done: {done - len(failed)} succeeded, {len(failed)} failed")
    if failed:
        for f in failed:
            print(f"  FAIL: {f}")


if __name__ == "__main__":
    main()
