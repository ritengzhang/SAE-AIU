#!/usr/bin/env python3
"""
Decoder Similarity → Semantic Alignment validation for Benchmark Training OPT-125M.

Same pipeline as GPT-2 runner but for OPT-125M SAEs.
Dataset is pre-tokenized (input_ids key).

Usage:
    python -m rebuttal_redundancy_to_semantic_alignment.run_benchmark_training_opt125m --device cuda:0
"""

import argparse
import gc
import heapq
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer

parent_dir = Path(__file__).parent.parent
sys.path.insert(0, str(parent_dir))

sae_lens_dir = parent_dir / "SAELens-main"
sys.path.insert(0, str(sae_lens_dir))

from benchmark_training.load_benchmark_sae import (
    load_base_model_for_llm,
    load_benchmark_sae,
    parse_combo_id,
)

# ── Config ──────────────────────────────────────────────────────────────────

LLM_NAME = "opt-125m"
OUTPUT_SUBDIR = "benchmark_training_opt125m"

TOP_K_SEQS = 20
N_COLLECTION_BATCHES = 50
SIMILARITY_BINS = [(0.0, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
PAIRS_PER_BIN = 300
WINDOW_SIZES = [8, 16, 32, 64]
EMBED_MODEL_NAME = "all-MiniLM-L6-v2"

BATCH_SIZE = 16
CONTEXT_SIZE = 256

DATASET_PATH = "apollo-research/Skylion007-openwebtext-tokenizer-gpt2"
TOKEN_KEY = "input_ids"


# ── Shared classes/functions (same as GPT-2 runner) ────────────────────────


class TopActivationTracker:
    def __init__(self, d_sae: int, top_k: int = TOP_K_SEQS):
        self.d_sae = d_sae
        self.top_k = top_k
        self.heaps: List[list] = [[] for _ in range(d_sae)]
        self.all_tokens: Dict[int, np.ndarray] = {}
        self.global_seq_count = 0

    def process_batch(self, feature_acts: np.ndarray, token_ids: np.ndarray):
        batch_size, seq_len, d_sae = feature_acts.shape
        max_acts = feature_acts.max(axis=1)
        max_positions = feature_acts.argmax(axis=1)
        for seq_i in range(batch_size):
            global_idx = self.global_seq_count + seq_i
            self.all_tokens[global_idx] = token_ids[seq_i]
            for feat_j in range(d_sae):
                act_val = float(max_acts[seq_i, feat_j])
                if act_val <= 0:
                    continue
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
        heap = self.heaps[feature_idx]
        results = []
        for act_val, global_idx, token_pos in sorted(heap, reverse=True):
            if global_idx in self.all_tokens:
                results.append((act_val, self.all_tokens[global_idx], token_pos))
        return results

    def prune_unused_tokens(self):
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
    model, sae, metadata, device, n_batches=N_COLLECTION_BATCHES
):
    d_sae = sae.cfg.d_sae
    tracker = TopActivationTracker(d_sae=d_sae)

    from datasets import load_dataset

    hook_name = metadata["hook_name"]

    print(f"  Loading dataset: {DATASET_PATH} (pre-tokenized)")
    ds = load_dataset(DATASET_PATH, split="train", streaming=True)
    ds_iter = iter(ds)
    seqs_collected = 0

    token_buffer = []

    def refill_token_buffer(min_seqs=40):
        while len(token_buffer) < min_seqs:
            try:
                item = next(ds_iter)
            except StopIteration:
                break
            tokens = item[TOKEN_KEY]
            for start in range(0, len(tokens) - CONTEXT_SIZE + 1, CONTEXT_SIZE):
                token_buffer.append(tokens[start : start + CONTEXT_SIZE])

    for batch_idx in range(n_batches):
        refill_token_buffer(min_seqs=BATCH_SIZE * 2)
        token_batch = []
        for _ in range(BATCH_SIZE):
            if not token_buffer:
                break
            token_batch.append(token_buffer.pop(0))
        if not token_batch:
            break

        token_ids = np.array(token_batch, dtype=np.int64)
        token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)

        with torch.no_grad():
            _, cache = model.run_with_cache(token_tensor, names_filter=[hook_name])
            hook_acts = cache[hook_name]
            flat_acts = hook_acts.reshape(-1, sae.cfg.d_in)
            feature_acts = sae.encode(flat_acts)
            feature_acts = feature_acts.reshape(len(token_batch), CONTEXT_SIZE, d_sae)

        tracker.process_batch(feature_acts.cpu().numpy(), token_ids)
        seqs_collected += len(token_batch)

        del cache, hook_acts, flat_acts, feature_acts, token_tensor
        if batch_idx % 10 == 0:
            torch.cuda.empty_cache()
            tracker.prune_unused_tokens()
            print(
                f"    Batch {batch_idx + 1}/{n_batches}, seqs collected: {seqs_collected}"
            )

    tracker.prune_unused_tokens()
    print(f"  Collection done: {seqs_collected} sequences processed")
    return tracker


def compute_decoder_similarity_matrix(decoder_weights):
    norms = np.linalg.norm(decoder_weights, axis=1, keepdims=True) + 1e-8
    normed = decoder_weights / norms
    return normed @ normed.T


def sample_pairs_per_bin(sim_matrix, bins, n_per_bin, alive_mask=None, rng=None):
    if rng is None:
        rng = np.random.default_rng(42)
    d_sae = sim_matrix.shape[0]
    triu_i, triu_j = np.triu_indices(d_sae, k=1)
    triu_sims = sim_matrix[triu_i, triu_j]
    if alive_mask is not None:
        valid = alive_mask[triu_i] & alive_mask[triu_j]
        triu_i, triu_j, triu_sims = triu_i[valid], triu_j[valid], triu_sims[valid]
    result = {}
    for low, high in bins:
        label = f"{low:.1f}-{high:.1f}"
        mask = (triu_sims >= low) & (triu_sims < high)
        indices = np.where(mask)[0]
        if len(indices) == 0:
            result[label] = []
            continue
        chosen = rng.choice(indices, size=min(n_per_bin, len(indices)), replace=False)
        pairs = [
            (int(triu_i[idx]), int(triu_j[idx]), float(triu_sims[idx]))
            for idx in chosen
        ]
        result[label] = pairs
        print(f"    Bin {label}: {len(pairs)} pairs (from {len(indices)} available)")
    return result


def extract_window_text(token_ids, max_token_pos, window_size, tokenizer):
    seq_len = len(token_ids)
    half_w = window_size // 2
    start = max(0, max_token_pos - half_w)
    end = min(seq_len, max_token_pos + half_w)
    if end - start < window_size:
        if start == 0:
            end = min(seq_len, start + window_size)
        elif end == seq_len:
            start = max(0, end - window_size)
    return tokenizer.decode(token_ids[start:end].tolist(), skip_special_tokens=True)


def compute_cross_embedding_similarity(texts_a, texts_b, embed_model):
    if not texts_a or not texts_b:
        return float("nan")
    emb_a = embed_model.encode(texts_a, convert_to_numpy=True)
    emb_b = embed_model.encode(texts_b, convert_to_numpy=True)
    emb_a = emb_a / (np.linalg.norm(emb_a, axis=1, keepdims=True) + 1e-8)
    emb_b = emb_b / (np.linalg.norm(emb_b, axis=1, keepdims=True) + 1e-8)
    return float(np.mean(emb_a @ emb_b.T))


def evaluate_pairs(pairs, tracker, tokenizer, embed_model, window_sizes=WINDOW_SIZES):
    results_by_window = {str(w): [] for w in window_sizes}
    for pair_idx, (feat_i, feat_j, dec_sim) in enumerate(pairs):
        top_seqs_i = tracker.get_top_sequences(feat_i)
        top_seqs_j = tracker.get_top_sequences(feat_j)
        if not top_seqs_i or not top_seqs_j:
            continue
        for w in window_sizes:
            texts_i = [
                t
                for t in (
                    extract_window_text(tok, pos, w, tokenizer)
                    for _, tok, pos in top_seqs_i
                )
                if t.strip()
            ]
            texts_j = [
                t
                for t in (
                    extract_window_text(tok, pos, w, tokenizer)
                    for _, tok, pos in top_seqs_j
                )
                if t.strip()
            ]
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


def run_one_sae(combo, model, embed_model, device, output_dir):
    combo_id = f"{combo['architecture']}_L{combo['layer']}_{combo['width']}_{combo['sparsity_param']}{combo['sparsity_value']}"
    parsed = parse_combo_id(combo_id)
    width_name = parsed["width"]
    architecture = parsed["architecture"]

    out_path = output_dir / OUTPUT_SUBDIR / width_name / architecture / combo_id
    results_file = out_path / "results.json"
    if results_file.exists():
        print(f"  SKIP (exists): {results_file}")
        return
    out_path.mkdir(parents=True, exist_ok=True)

    sae, metadata = load_benchmark_sae(
        combo_id=combo_id, llm_name=LLM_NAME, device=device
    )
    d_sae = sae.cfg.d_sae
    print(f"  SAE loaded: d_sae={d_sae}, arch={architecture}")

    print(f"  Step 1: Collecting top-{TOP_K_SEQS} activating sequences...")
    t0 = time.time()
    tracker = collect_top_activations(
        model=model, sae=sae, metadata=metadata, device=device
    )
    print(f"  Collection took {time.time() - t0:.1f}s")

    print(f"  Step 2: Computing decoder similarity matrix...")
    decoder_weights = sae.W_dec.detach().cpu().numpy()
    sim_matrix = compute_decoder_similarity_matrix(decoder_weights)
    alive_mask = np.array([len(tracker.heaps[i]) > 0 for i in range(d_sae)])
    n_alive = alive_mask.sum()
    print(f"  Alive features: {n_alive}/{d_sae}")

    print(f"  Step 3: Sampling pairs per similarity bin...")
    binned_pairs = sample_pairs_per_bin(
        sim_matrix, SIMILARITY_BINS, PAIRS_PER_BIN, alive_mask
    )
    del sim_matrix, decoder_weights
    gc.collect()

    print(f"  Step 4: Computing embedding similarities...")
    tokenizer = model.tokenizer
    all_results = {}
    for bin_label, pairs in binned_pairs.items():
        if not pairs:
            all_results[bin_label] = {str(w): [] for w in WINDOW_SIZES}
            continue
        print(f"    Bin {bin_label}: evaluating {len(pairs)} pairs...")
        all_results[bin_label] = evaluate_pairs(pairs, tracker, tokenizer, embed_model)

    summary = {
        "metadata": {
            "architecture": architecture,
            "combo_id": combo_id,
            "width": width_name,
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

    with open(results_file, "w") as f:
        json.dump({"summary": summary, "pair_details": all_results}, f, indent=2)
    print(f"  SAVED: {results_file}")

    del tracker, sae, binned_pairs, all_results
    gc.collect()
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description=f"Decoder→semantic alignment (Benchmark Training {LLM_NAME})"
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--output-dir",
        type=str,
        default="<DATA_ROOT>/rebuttal_redundancy_to_semantic_alignment",
    )
    parser.add_argument("--widths", nargs="+", default=None)
    args = parser.parse_args()

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)

    with open("<DATA_ROOT>/training/combinations.json") as f:
        combinations = json.load(f)
    combos = [c for c in combinations if c["llm"] == LLM_NAME]

    if args.widths:
        combos = [c for c in combos if c["width"] in args.widths]

    total = len(combos)
    print(f"Decoder→Semantic alignment: Benchmark Training {LLM_NAME} ({total} SAEs)")
    print(f"  Context: {CONTEXT_SIZE}, Batch: {BATCH_SIZE}")

    embed_model = SentenceTransformer(EMBED_MODEL_NAME, device="cpu")
    model = load_base_model_for_llm(LLM_NAME, device=device)

    done, failed = 0, []
    try:
        for combo in combos:
            done += 1
            combo_id = f"{combo['architecture']}_L{combo['layer']}_{combo['width']}_{combo['sparsity_param']}{combo['sparsity_value']}"
            print(f"\n[{done}/{total}] {combo_id}")
            try:
                run_one_sae(combo, model, embed_model, device, output_dir)
            except Exception as e:
                import traceback

                print(f"  FAILED: {e}")
                traceback.print_exc()
                failed.append(f"{combo_id}: {e}")
                gc.collect()
                torch.cuda.empty_cache()
    finally:
        del model
        gc.collect()
        torch.cuda.empty_cache()

    print(f"\n{'=' * 60}\nDone: {done - len(failed)} succeeded, {len(failed)} failed")
    for f_entry in failed:
        print(f"  FAIL: {f_entry}")


if __name__ == "__main__":
    main()
