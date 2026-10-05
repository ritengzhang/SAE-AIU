"""
Stage 1: Collection with Two-Pass Approach

Pass 1: Compute global max per feature
Pass 2: Mark high/low firing sentences/tokens (sparse storage)
"""

import torch
import numpy as np
from typing import Dict, List, Set, Tuple, Optional
from dataclasses import dataclass
from tqdm import tqdm
from collections import defaultdict
import hashlib


@dataclass
class CollectionConfig:
    """Configuration for activation collection."""
    sentence_size: int = 16  # Sentence size in tokens (16 or 32)
    n_batches: int = 1000  # Batches for collecting sequences
    use_absolute_value: bool = True  # For signed activations (TopK SAEs)
    n_high_for_explanation: int = 20  # Top sequences for explanation
    n_high_for_evaluation: int = 20  # Next sequences for evaluation
    n_low_for_contrast: int = 60  # Lowest sequences for contrast (need 60 for evaluation: each question uses 2-3 distractors)


@dataclass
class MarkedSequence:
    """A marked sequence with its activation data."""
    tokens: List[str]
    activations: List[float]  # Raw activations (no normalization)
    sentence_boundaries: List[int]  # Indices where sentences start
    max_activation: float  # Max activation in sequence (raw)
    sentence_sums: List[float]  # Sum per sentence (raw)
    sequence_id: str  # Unique identifier
    high_act_token_idx: int = -1  # Index of highest activating token in the sequence


class ActivationCollector:
    """
    Collects and marks activations using two-pass approach.
    
    Pass 1: Compute global max per feature
    Pass 2: Mark high/low firing sequences and sentences
    """
    
    def __init__(
        self,
        model,
        sae,
        store,
        device: str = "cpu",
        config: Optional[CollectionConfig] = None
    ):
        self.model = model
        self.sae = sae
        self.store = store
        self.device = device
        self.config = config or CollectionConfig()
        self.tokenizer = model.tokenizer
        
        # Storage for sequences (V2 - 3-bin strategy):
        # - high_explanation_sequences: Top 20 sequences for explanation (excluded from questions)
        # - bin_A_sequences: HIGH-high (~top 25% of firing, excluding top 20 for explanation)
        # - bin_B_sequences: HIGH-low (~25-50% of firing)
        # - bin_C_sequences: LOW (bottom 50%, may include non-firing)
        # Legacy storage (for backward compatibility during transition):
        self.high_explanation_sequences: Dict[int, List[MarkedSequence]] = defaultdict(list)
        self.bin_A_sequences: Dict[int, List[MarkedSequence]] = defaultdict(list)
        self.bin_B_sequences: Dict[int, List[MarkedSequence]] = defaultdict(list)
        self.bin_C_sequences: Dict[int, List[MarkedSequence]] = defaultdict(list)
        
        # Sum-based bins for sentence-based evaluation (same 3-bin strategy, sorted by sentence_sum)
        self.bin_A_sum_sequences: Dict[int, List[MarkedSequence]] = defaultdict(list)
        self.bin_B_sum_sequences: Dict[int, List[MarkedSequence]] = defaultdict(list)
        self.bin_C_sum_sequences: Dict[int, List[MarkedSequence]] = defaultdict(list)
        # Keep old bins for now (will be removed later):
        self.high_max_sequences: Dict[int, List[MarkedSequence]] = defaultdict(list)
        self.low_max_sequences: Dict[int, List[MarkedSequence]] = defaultdict(list)
        self.high_sum_sequences: Dict[int, List[MarkedSequence]] = defaultdict(list)
        self.low_sum_sequences: Dict[int, List[MarkedSequence]] = defaultdict(list)
        
        # Sentence IDs for tracking (legacy, may be used elsewhere)
        self.high_sum_sentences: Dict[int, Set[str]] = defaultdict(set)
        self.low_sum_sentences: Dict[int, Set[str]] = defaultdict(set)
        
        # Store full sequence data for marked sequences (needed for explanation/evaluation)
        self.marked_sequence_data: Dict[int, List[MarkedSequence]] = defaultdict(list)
    
    def compute_sentence_id(self, sentence_tokens: List[int]) -> str:
        """Create unique ID for a sentence (hash of tokens)."""
        token_str = "".join([self.tokenizer.decode([t]) for t in sentence_tokens])
        return hashlib.md5(token_str.encode()).hexdigest()
    
    def collect_sequences(
        self, 
        feature_indices: List[int],
        sentence_size: Optional[int] = None
    ) -> Dict[int, List[MarkedSequence]]:
        """
        Collect full sequences for each feature.
        Since sequences are already the right size (sentence_size), we use them directly.
        
        Returns:
            Dict mapping feature_idx -> list of MarkedSequence objects
        """
        sentence_size = sentence_size or self.config.sentence_size
        print(f"\n{'='*70}")
        print(f"Collecting sequences (using {sentence_size}-token sequences directly)")
        print(f"{'='*70}")
        
        # Storage for all collected sequences
        all_sequences = defaultdict(list)
        
        for batch_idx in tqdm(range(self.config.n_batches), desc="Collection"):
            # Get tokens
            batch_tokens = self.store.get_batch_tokens()  # [batch_size, seq_len]
            
            # Get activations
            with torch.no_grad():
                model_acts = self.store.get_activations(batch_tokens)  # [batch, seq, d_in]
                
                # Flatten for SAE
                batch_size, seq_len = batch_tokens.shape
                model_acts_flat = model_acts.reshape(-1, model_acts.shape[-1])
                model_acts_flat = model_acts_flat.to(self.device)
                
                # Encode with SAE
                sae_acts_flat = self.sae.encode(model_acts_flat)  # [batch*seq, d_sae]
                sae_acts = sae_acts_flat.reshape(batch_size, seq_len, self.sae.cfg.d_sae)
            
            # Process each sequence directly (no centering needed - sequences are already sentence_size)
            for seq_idx in range(batch_size):
                tokens = batch_tokens[seq_idx].cpu().tolist()
                
                # Use sequence directly (should already be sentence_size tokens)
                if len(tokens) != sentence_size:
                    # If sequence is longer, truncate or pad (shouldn't happen if context_size matches)
                    if len(tokens) > sentence_size:
                        tokens = tokens[:sentence_size]
                    else:
                        # Pad if needed (shouldn't happen)
                        tokens = tokens + [0] * (sentence_size - len(tokens))
                
                for feat_idx in feature_indices:
                    feature_acts = sae_acts[seq_idx, :len(tokens), feat_idx].cpu().numpy()  # [sentence_size]
                    
                    # Find max activation position and value
                    if self.config.use_absolute_value:
                        abs_acts = np.abs(feature_acts)
                        max_act_idx = int(np.argmax(abs_acts))
                        max_act_raw = float(abs_acts[max_act_idx])
                        act_sum = float(abs_acts.sum())
                    else:
                        max_act_idx = int(np.argmax(feature_acts))
                        max_act_raw = float(feature_acts[max_act_idx])
                        act_sum = float(np.maximum(feature_acts, 0).sum())  # Sum of positive activations
                    
                    # Decode tokens to strings
                    token_strs = [self.tokenizer.decode([t]) for t in tokens]
                    
                    # Create marked sequence
                    marked_seq = MarkedSequence(
                        tokens=token_strs,
                        activations=feature_acts.tolist(),  # Raw activations
                        sentence_boundaries=[0],  # Only one sentence (the whole sequence)
                        max_activation=max_act_raw,  # Raw max activation
                        sentence_sums=[act_sum],  # Raw sum
                        sequence_id=f"pass1_batch{batch_idx}_seq{seq_idx}_feat{feat_idx}",
                        high_act_token_idx=max_act_idx  # Index of highest activating token
                    )
                    
                    all_sequences[feat_idx].append(marked_seq)
                    self.marked_sequence_data[feat_idx].append(marked_seq)
            
            # Periodic cleanup
            if (batch_idx + 1) % 10 == 0:
                torch.cuda.empty_cache()
        
        print(f"\nCollection complete: Collected sequences per feature")
        for feat_idx in feature_indices[:5]:  # Show first 5
            n_seqs = len(all_sequences.get(feat_idx, []))
            print(f"  Feature {feat_idx}: {n_seqs} sequences")
        if len(feature_indices) > 5:
            print(f"  ... ({len(feature_indices) - 5} more features)")
        
        return dict(all_sequences)
    
    
    def collect(
        self,
        feature_indices: List[int],
        sentence_size: Optional[int] = None
    ) -> Dict[int, Dict]:
        """
        Collect sequences and bin them into high/low for explanation and evaluation.
        
        IMPORTANT: Resets the store to ensure we use the same batches as feature discovery.
        
        Returns:
            Dict with collection statistics per feature
        """
        sentence_size = sentence_size or self.config.sentence_size
        
        # CRITICAL: Reset store to ensure we use the same batches as feature discovery
        if hasattr(self.store, 'reset_input_dataset'):
            self.store.reset_input_dataset()
            if hasattr(self.store, 'iterable_sequences') and hasattr(self.store, '_iterate_tokenized_sequences'):
                self.store.iterable_sequences = self.store._iterate_tokenized_sequences()
            print(f"✓ Reset store for collection (ensuring same batches as feature discovery)")
        
        # Collect all sequences
        all_sequences = self.collect_sequences(feature_indices, sentence_size)
        
        # Initialize stats
        stats = {feat_idx: {
            'n_high_explanation': 0,
            'n_high_evaluation': 0,
            'n_low_contrast': 0,
            'n_total_sequences': 0,
            'n_firing_sequences': 0,
            'n_non_firing_sequences': 0
        } for feat_idx in feature_indices}
        
        # Bin sequences
        self._bin_sequences(feature_indices, all_sequences, stats)
        
        return stats
    
    def _bin_sequences(
        self,
        feature_indices: List[int],
        all_sequences: Dict[int, List[MarkedSequence]],
        stats: Dict[int, Dict]
    ):
        """
        Bin sequences into 3 bins using V2 strategy:
        - BIN A (HIGH-high): Top ~25% of firing activations (excluding top 20 for explanation)
        - BIN B (HIGH-low): ~25-50% of firing activations
        - BIN C (LOW): Bottom 50% (may include non-firing)
        
        Steps:
        1. Sort all sequences by max_activation (descending)
        2. Split at 50% → HIGH half (top 50%) and LOW half (bottom 50%)
        3. Remove non-firing from HIGH half → high_half_firing
        4. Take top 20 from high_half_firing → explanation (marked as USED)
        5. Split remaining high_half_firing in middle → BIN A and BIN B
        6. BIN C = LOW half
        """
        for feat_idx in feature_indices:
            all_seqs = all_sequences.get(feat_idx, [])
            
            if len(all_seqs) == 0:
                continue
            
            # Track total sequences
            stats[feat_idx]['n_total_sequences'] = len(all_seqs)
            
            # Step 1: Sort all sequences by max_activation (descending)
            all_seqs_sorted = sorted(all_seqs, key=lambda seq: seq.max_activation, reverse=True)
            
            # Step 2: Split at 50% mark → HIGH half (top 50%) and LOW half (bottom 50%)
            total = len(all_seqs_sorted)
            split_point = total // 2
            high_half = all_seqs_sorted[:split_point]
            low_half = all_seqs_sorted[split_point:]
            
            # Step 3: Remove non-firing from HIGH half
            high_half_firing = [seq for seq in high_half if seq.max_activation > 0]
            stats[feat_idx]['n_firing_sequences'] = len(high_half_firing) + len([seq for seq in low_half if seq.max_activation > 0])
            stats[feat_idx]['n_non_firing_sequences'] = len([seq for seq in all_seqs if seq.max_activation == 0])
            
            # Step 4: Take top 20 from high_half_firing for explanation (marked as USED)
            n_explanation = self.config.n_high_for_explanation  # 20
            if len(high_half_firing) >= n_explanation:
                high_explanation = high_half_firing[:n_explanation]
                remaining_high = high_half_firing[n_explanation:]
            else:
                # Edge case: Not enough sequences in high_half_firing
                high_explanation = high_half_firing
                remaining_high = []
            
            # Step 5: Split remaining high_half_firing in the middle → BIN A and BIN B
            if len(remaining_high) >= 2:
                mid_point = len(remaining_high) // 2
                bin_A = remaining_high[:mid_point]  # HIGH-high (top ~25% of firing, excluding top 20)
                bin_B = remaining_high[mid_point:]  # HIGH-low (~25-50% of firing)
            else:
                # Edge case: Not enough sequences to split
                bin_A = remaining_high if len(remaining_high) > 0 else []
                bin_B = []
            
            # Step 6: BIN C = LOW half (bottom 50%, may include non-firing)
            bin_C = low_half
            
            # Store bins (V2 structure)
            self.high_explanation_sequences[feat_idx] = high_explanation
            self.bin_A_sequences[feat_idx] = bin_A
            self.bin_B_sequences[feat_idx] = bin_B
            self.bin_C_sequences[feat_idx] = bin_C
            
            # Also populate legacy bins for backward compatibility (for now)
            # These will be removed later after evaluation is updated
            self.high_max_sequences[feat_idx] = bin_A + bin_B  # Combined HIGH for legacy use
            self.low_max_sequences[feat_idx] = bin_C
            
            # For sentence-based evaluation: repeat process using sentence_sum
            all_seqs_by_sum = sorted(all_seqs, key=lambda seq: seq.sentence_sums[0] if seq.sentence_sums else 0.0, reverse=True)
            total_sum = len(all_seqs_by_sum)
            split_point_sum = total_sum // 2
            high_half_sum = all_seqs_by_sum[:split_point_sum]
            low_half_sum = all_seqs_by_sum[split_point_sum:]
            
            high_half_firing_sum = [seq for seq in high_half_sum if (seq.sentence_sums[0] if seq.sentence_sums else 0.0) > 0]
            
            if len(high_half_firing_sum) >= n_explanation:
                high_sum_explanation = high_half_firing_sum[:n_explanation]
                remaining_high_sum = high_half_firing_sum[n_explanation:]
            else:
                high_sum_explanation = high_half_firing_sum
                remaining_high_sum = []
            
            if len(remaining_high_sum) >= 2:
                mid_point_sum = len(remaining_high_sum) // 2
                bin_A_sum = remaining_high_sum[:mid_point_sum]
                bin_B_sum = remaining_high_sum[mid_point_sum:]
            else:
                bin_A_sum = remaining_high_sum if len(remaining_high_sum) > 0 else []
                bin_B_sum = []
            
            bin_C_sum = low_half_sum
            
            # Store sum-based bins separately (V2)
            self.bin_A_sum_sequences[feat_idx] = bin_A_sum
            self.bin_B_sum_sequences[feat_idx] = bin_B_sum
            self.bin_C_sum_sequences[feat_idx] = bin_C_sum
            
            # Also populate legacy bins for backward compatibility
            self.high_sum_sequences[feat_idx] = bin_A_sum + bin_B_sum  # Combined for legacy
            self.low_sum_sequences[feat_idx] = bin_C_sum
            
            # Update stats
            stats[feat_idx]['n_high_explanation'] = len(high_explanation)
            stats[feat_idx]['n_bin_A'] = len(bin_A)
            stats[feat_idx]['n_bin_B'] = len(bin_B)
            stats[feat_idx]['n_bin_C'] = len(bin_C)
            
            # Legacy stats
            stats[feat_idx]['n_high_evaluation'] = len(bin_A) + len(bin_B)
            stats[feat_idx]['n_low_contrast'] = len(bin_C)
        
        print(f"\nBinning complete (V2 - 3-bin strategy):")
        for feat_idx in feature_indices[:5]:
            s = stats[feat_idx]
            print(f"  Feature {feat_idx}:")
            print(f"    High explanation: {s['n_high_explanation']} sequences (top 20, excluded from questions)")
            print(f"    BIN A (HIGH-high): {s['n_bin_A']} sequences")
            print(f"    BIN B (HIGH-low):  {s['n_bin_B']} sequences")
            print(f"    BIN C (LOW):       {s['n_bin_C']} sequences")
    
    def collect_optimized(
        self,
        feature_indices: List[int],
        sequence_references: Dict[int, Dict[str, List]],
        batch_tokens_cache: Dict[int, torch.Tensor],
        use_absolute_value: bool = False
    ) -> Dict[int, Dict]:
        """
        Optimized collection: Only collect pre-selected sequences from cached batches.
        BATCHED VERSION: Groups sequences by batch_idx to avoid re-encoding the same batch multiple times.
        
        Args:
            feature_indices: List of feature indices to collect
            sequence_references: Dict mapping feat_idx -> {
                'high_explanation': [(max_act, batch_idx, seq_idx, sum_act, max_tok_idx), ...],
                'high_evaluation': [...],
                'low_contrast': [...]
            }
            batch_tokens_cache: Dict mapping batch_idx -> batch_tokens tensor
            use_absolute_value: Whether to use absolute values for activations
        
        Returns:
            Dict with collection statistics per feature
        """
        from collections import defaultdict
        from tqdm import tqdm
        import torch
        import numpy as np
        
        print(f"\n{'='*70}")
        print(f"Optimized Collection (BATCHED): Collecting pre-selected sequences only")
        print(f"{'='*70}")
        
        sentence_size = self.config.sentence_size
        
        # Initialize stats
        stats = {feat_idx: {
            'n_high_explanation': 0,
            'n_high_evaluation': 0,
            'n_low_contrast': 0,
            'n_total_sequences': 0,
        } for feat_idx in feature_indices}
        
        # Initialize storage
        high_explanation_seqs_all = defaultdict(list)
        high_evaluation_seqs_all = defaultdict(list)
        high_evaluation_by_sum_seqs_all = defaultdict(list)
        low_contrast_seqs_all = defaultdict(list)
        low_contrast_by_sum_seqs_all = defaultdict(list)
        
        # Step 1: Group all sequence references by (batch_idx, feat_idx)
        # Structure: batch_sequences[batch_idx][feat_idx] = {
        #   'high_explanation': [(seq_idx, sum_act, max_tok_idx), ...],
        #   'high_evaluation': [...],
        #   ...
        # }
        batch_sequences = defaultdict(lambda: defaultdict(lambda: {
            'high_explanation': [],
            'high_evaluation': [],
            'high_evaluation_by_sum': [],
            'low_contrast': [],
            'low_contrast_by_sum': []
        }))
        
        for feat_idx in feature_indices:
            refs = sequence_references.get(feat_idx, {})
            
            for ref in refs.get('high_explanation', []):
                _, batch_idx, seq_idx, sum_act, max_tok_idx = ref
                batch_sequences[batch_idx][feat_idx]['high_explanation'].append((seq_idx, sum_act, max_tok_idx))
            
            for ref in refs.get('high_evaluation', []):
                _, batch_idx, seq_idx, sum_act, max_tok_idx = ref
                batch_sequences[batch_idx][feat_idx]['high_evaluation'].append((seq_idx, sum_act, max_tok_idx))
            
            for ref in refs.get('high_evaluation_by_sum', []):
                _, batch_idx, seq_idx, max_act, max_tok_idx = ref
                batch_sequences[batch_idx][feat_idx]['high_evaluation_by_sum'].append((seq_idx, max_act, max_tok_idx))
            
            for ref in refs.get('low_contrast', []):
                _, batch_idx, seq_idx, sum_act, max_tok_idx = ref
                batch_sequences[batch_idx][feat_idx]['low_contrast'].append((seq_idx, sum_act, max_tok_idx))
            
            for ref in refs.get('low_contrast_by_sum', []):
                _, batch_idx, seq_idx, sum_act, max_tok_idx = ref
                batch_sequences[batch_idx][feat_idx]['low_contrast_by_sum'].append((seq_idx, sum_act, max_tok_idx))
        
        # Step 2: Process each batch once, extract all sequences from that batch
        for batch_idx in tqdm(sorted(batch_sequences.keys()), desc="Processing batches"):
            batch_tokens = batch_tokens_cache[batch_idx]
            batch_size, seq_len = batch_tokens.shape
            
            # Encode entire batch ONCE
            with torch.no_grad():
                model_acts = self.store.get_activations(batch_tokens)
                model_acts_flat = model_acts.reshape(-1, model_acts.shape[-1])
                model_acts_flat = model_acts_flat.to(self.device)
                
                sae_acts_flat = self.sae.encode(model_acts_flat)
                sae_acts = sae_acts_flat.reshape(batch_size, seq_len, self.sae.cfg.d_sae)
                
                if use_absolute_value:
                    sae_acts = torch.abs(sae_acts)
                
                sae_acts_cpu = sae_acts.cpu().numpy()  # Move to CPU once
            
            # Batch decode all tokens for this batch
            all_tokens_list = batch_tokens.cpu().tolist()
            all_token_strs = {}  # Cache decoded tokens: seq_idx -> [token_strs]
            
            # Extract all sequences for all features from this batch
            for feat_idx, seq_refs in batch_sequences[batch_idx].items():
                # Process each sequence type
                for seq_type in ['high_explanation', 'high_evaluation', 'high_evaluation_by_sum', 'low_contrast', 'low_contrast_by_sum']:
                    for seq_idx, sum_act, max_tok_idx in seq_refs[seq_type]:
                        try:
                            # Get tokens (with caching)
                            if seq_idx not in all_token_strs:
                                tokens = all_tokens_list[seq_idx]
                                if len(tokens) != sentence_size:
                                    if len(tokens) > sentence_size:
                                        tokens = tokens[:sentence_size]
                                    else:
                                        tokens = tokens + [0] * (sentence_size - len(tokens))
                                # Batch decode tokens
                                all_token_strs[seq_idx] = [self.tokenizer.decode([t]) for t in tokens]
                            
                            token_strs = all_token_strs[seq_idx]
                            
                            # Extract activations for this sequence and feature
                            feature_acts = sae_acts_cpu[seq_idx, :len(token_strs), feat_idx]
                            
                            # Compute max activation and sum
                            if use_absolute_value:
                                abs_acts = np.abs(feature_acts)
                                max_act_idx = int(np.argmax(abs_acts))
                                max_act_raw = float(abs_acts[max_act_idx])
                                act_sum = float(abs_acts.sum())
                            else:
                                max_act_idx = int(np.argmax(feature_acts))
                                max_act_raw = float(feature_acts[max_act_idx])
                                act_sum = float(np.maximum(feature_acts, 0).sum())
                            
                            # Create marked sequence
                            marked_seq = MarkedSequence(
                                tokens=token_strs,
                                activations=feature_acts.tolist(),
                                sentence_boundaries=[0],
                                max_activation=max_act_raw,
                                sentence_sums=[act_sum],
                                sequence_id=f"optimized_batch{batch_idx}_seq{seq_idx}_feat{feat_idx}",
                                high_act_token_idx=max_act_idx
                            )
                            
                            # Store in appropriate list
                            if seq_type == 'high_explanation':
                                high_explanation_seqs_all[feat_idx].append(marked_seq)
                            elif seq_type == 'high_evaluation':
                                high_evaluation_seqs_all[feat_idx].append(marked_seq)
                            elif seq_type == 'high_evaluation_by_sum':
                                high_evaluation_by_sum_seqs_all[feat_idx].append(marked_seq)
                            elif seq_type == 'low_contrast':
                                low_contrast_seqs_all[feat_idx].append(marked_seq)
                            elif seq_type == 'low_contrast_by_sum':
                                low_contrast_by_sum_seqs_all[feat_idx].append(marked_seq)
                                
                        except Exception as e:
                            print(f"Warning: Failed to create sequence for batch {batch_idx}, seq {seq_idx}, feat {feat_idx}: {e}")
        
        # Step 3: Store all sequences in collector
        for feat_idx in feature_indices:
            self.high_explanation_sequences[feat_idx] = high_explanation_seqs_all[feat_idx]
            self.high_max_sequences[feat_idx] = high_evaluation_seqs_all[feat_idx]
            self.high_sum_sequences[feat_idx] = high_evaluation_by_sum_seqs_all[feat_idx]
            self.low_max_sequences[feat_idx] = low_contrast_seqs_all[feat_idx]
            self.low_sum_sequences[feat_idx] = low_contrast_by_sum_seqs_all[feat_idx]
            
            # CRITICAL: Populate bin_A, bin_B, bin_C for evaluation stage
            # Split high_evaluation sequences into bin_A (HIGH-high) and bin_B (HIGH-low)
            # IMPORTANT: bin_A must have HIGHER activations than bin_B
            high_eval_seqs = high_evaluation_seqs_all[feat_idx]
            if len(high_eval_seqs) >= 2:
                # Sort by max_activation (descending) to ensure bin_A > bin_B
                high_eval_sorted = sorted(high_eval_seqs, key=lambda seq: seq.max_activation, reverse=True)
                mid_point = len(high_eval_sorted) // 2
                bin_A = high_eval_sorted[:mid_point]  # HIGH-high (top 50% - HIGHER activations)
                bin_B = high_eval_sorted[mid_point:]   # HIGH-low (bottom 50% - LOWER activations)
                # Verify: All bin_A activations should be >= all bin_B activations
                if len(bin_A) > 0 and len(bin_B) > 0:
                    min_bin_A = min(seq.max_activation for seq in bin_A)
                    max_bin_B = max(seq.max_activation for seq in bin_B)
                    if min_bin_A < max_bin_B:
                        # This shouldn't happen if sorting worked, but log a warning if it does
                        print(f"Warning: Feature {feat_idx} bin_A min ({min_bin_A:.6f}) < bin_B max ({max_bin_B:.6f}) - sorting issue!")
            else:
                # Edge case: Not enough sequences to split
                bin_A = high_eval_seqs if len(high_eval_seqs) > 0 else []
                bin_B = []
            
            # bin_C = low_contrast sequences
            bin_C = low_contrast_seqs_all[feat_idx]
            
            # Store in bin dictionaries (V2 structure for evaluation)
            self.bin_A_sequences[feat_idx] = bin_A
            self.bin_B_sequences[feat_idx] = bin_B
            self.bin_C_sequences[feat_idx] = bin_C
            
            # For sentence-based evaluation: split by sum
            # IMPORTANT: bin_A_sum must have HIGHER sums than bin_B_sum
            high_eval_sum_seqs = high_evaluation_by_sum_seqs_all[feat_idx]
            if len(high_eval_sum_seqs) >= 2:
                # Sort by sentence_sum (descending) to ensure bin_A_sum > bin_B_sum
                high_eval_sum_sorted = sorted(high_eval_sum_seqs, key=lambda seq: seq.sentence_sums[0] if seq.sentence_sums else 0.0, reverse=True)
                mid_point_sum = len(high_eval_sum_sorted) // 2
                bin_A_sum = high_eval_sum_sorted[:mid_point_sum]  # HIGH-high (HIGHER sums)
                bin_B_sum = high_eval_sum_sorted[mid_point_sum:]  # HIGH-low (LOWER sums)
                # Verify: All bin_A_sum should be >= all bin_B_sum
                if len(bin_A_sum) > 0 and len(bin_B_sum) > 0:
                    min_bin_A_sum = min(seq.sentence_sums[0] if seq.sentence_sums else 0.0 for seq in bin_A_sum)
                    max_bin_B_sum = max(seq.sentence_sums[0] if seq.sentence_sums else 0.0 for seq in bin_B_sum)
                    if min_bin_A_sum < max_bin_B_sum:
                        print(f"Warning: Feature {feat_idx} bin_A_sum min ({min_bin_A_sum:.6f}) < bin_B_sum max ({max_bin_B_sum:.6f}) - sorting issue!")
            else:
                bin_A_sum = high_eval_sum_seqs if len(high_eval_sum_seqs) > 0 else []
                bin_B_sum = []
            
            bin_C_sum = low_contrast_by_sum_seqs_all[feat_idx]
            
            # Store in sum-based bin dictionaries
            self.bin_A_sum_sequences[feat_idx] = bin_A_sum
            self.bin_B_sum_sequences[feat_idx] = bin_B_sum
            self.bin_C_sum_sequences[feat_idx] = bin_C_sum
            
            # Update stats
            stats[feat_idx]['n_high_explanation'] = len(high_explanation_seqs_all[feat_idx])
            stats[feat_idx]['n_high_evaluation'] = len(high_evaluation_seqs_all[feat_idx])
            stats[feat_idx]['n_low_contrast'] = len(low_contrast_seqs_all[feat_idx])
            stats[feat_idx]['n_bin_A'] = len(bin_A)
            stats[feat_idx]['n_bin_B'] = len(bin_B)
            stats[feat_idx]['n_bin_C'] = len(bin_C)
            stats[feat_idx]['n_total_sequences'] = (
                len(high_explanation_seqs_all[feat_idx]) + 
                len(high_evaluation_seqs_all[feat_idx]) + 
                len(low_contrast_seqs_all[feat_idx])
            )
        
        print(f"\nOptimized collection complete:")
        for feat_idx in feature_indices[:5]:
            s = stats[feat_idx]
            print(f"  Feature {feat_idx}: {s['n_total_sequences']} sequences (exp: {s['n_high_explanation']}, eval: {s['n_high_evaluation']}, low: {s['n_low_contrast']})")
            print(f"    Bins: A={s.get('n_bin_A', 0)}, B={s.get('n_bin_B', 0)}, C={s.get('n_bin_C', 0)}")
        
        return stats
    
    def _create_marked_sequence_from_cache(
        self,
        batch_tokens: torch.Tensor,
        seq_idx: int,
        feat_idx: int,
        batch_idx: int,
        use_absolute_value: bool
    ) -> Optional[MarkedSequence]:
        """Create a MarkedSequence from cached batch tokens."""
        try:
            # Get tokens for this sequence
            tokens = batch_tokens[seq_idx].cpu().tolist()
            if len(tokens) != self.config.sentence_size:
                if len(tokens) > self.config.sentence_size:
                    tokens = tokens[:self.config.sentence_size]
                else:
                    tokens = tokens + [0] * (self.config.sentence_size - len(tokens))
            
            # Get activations for the full batch (more efficient)
            with torch.no_grad():
                model_acts = self.store.get_activations(batch_tokens)
                batch_size, seq_len = batch_tokens.shape
                model_acts_flat = model_acts.reshape(-1, model_acts.shape[-1])
                model_acts_flat = model_acts_flat.to(self.device)
                
                sae_acts_flat = self.sae.encode(model_acts_flat)
                sae_acts = sae_acts_flat.reshape(batch_size, seq_len, self.sae.cfg.d_sae)
                
                if use_absolute_value:
                    sae_acts = torch.abs(sae_acts)
                
                # Extract activations for this specific sequence and feature
                feature_acts = sae_acts[seq_idx, :len(tokens), feat_idx].cpu().numpy()
            
            # Compute max activation and sum
            if use_absolute_value:
                abs_acts = np.abs(feature_acts)
                max_act_idx = int(np.argmax(abs_acts))
                max_act_raw = float(abs_acts[max_act_idx])
                act_sum = float(abs_acts.sum())
            else:
                max_act_idx = int(np.argmax(feature_acts))
                max_act_raw = float(feature_acts[max_act_idx])
                act_sum = float(np.maximum(feature_acts, 0).sum())
            
            # Decode tokens
            token_strs = [self.tokenizer.decode([t]) for t in tokens]
            
            # Create marked sequence
            marked_seq = MarkedSequence(
                tokens=token_strs,
                activations=feature_acts.tolist(),
                sentence_boundaries=[0],
                max_activation=max_act_raw,
                sentence_sums=[act_sum],
                sequence_id=f"optimized_batch{batch_idx}_seq{seq_idx}_feat{feat_idx}",
                high_act_token_idx=max_act_idx
            )
            
            return marked_seq
            
        except Exception as e:
            print(f"Warning: Failed to create sequence for batch {batch_idx}, seq {seq_idx}, feat {feat_idx}: {e}")
            return None
    
    def bin_and_sample_sequences(
        self,
        feature_indices: List[int],
        all_sequences_max: Dict[int, List[Tuple[float, float, int, int, int]]],  # feat_idx -> [(max_act, sum_act, batch_idx, seq_idx, max_tok_idx), ...]
        all_sequences_sum: Dict[int, List[Tuple[float, float, int, int, int]]],  # feat_idx -> [(sum_act, max_act, batch_idx, seq_idx, max_tok_idx), ...]
        n_explanation: int = 20,
        n_evaluation: int = 20,
        n_low: int = 40,
        random_seed: int = 42
    ) -> Tuple[Dict[int, Dict[str, List]], Dict[int, Dict]]:
        """
        Bin all sequences into 3 bins (high/middle/low) and sample from bins.
        
        Logic:
        1. Bin ALL sequences (including non-firing) into 3 bins
        2. Remove non-firing sequences from high bin
        3. If high bin < 1/3 after removal, use all fired sequences
        4. Sample explanation: top n_explanation from high bin
        5. Sample evaluation: n_evaluation from remaining high bin (without replacement)
        6. Sample low: n_low from low bin (without replacement)
        
        Args:
            feature_indices: List of feature indices
            all_sequences_max: Dict mapping feat_idx -> list of (max_act, sum_act, batch_idx, seq_idx, max_tok_idx)
            all_sequences_sum: Dict mapping feat_idx -> list of (sum_act, max_act, batch_idx, seq_idx, max_tok_idx)
            n_explanation: Number of sequences for explanation
            n_evaluation: Number of sequences for evaluation
            n_low: Number of low sequences for contrast
            random_seed: Random seed for reproducibility
        
        Returns:
            Tuple of (sequence_references, collection_stats)
            sequence_references: Dict mapping feat_idx -> {
                'high_explanation': [(max_act, batch_idx, seq_idx, sum_act, max_tok_idx), ...],
                'high_evaluation': [...],
                'high_evaluation_by_sum': [...],
                'low_contrast': [...],
                'low_contrast_by_sum': [...]
            }
            collection_stats: Dict mapping feat_idx -> stats dict
        """
        import random
        random.seed(random_seed)
        
        sequence_references = {}
        collection_stats = {}
        
        for feat_idx in feature_indices:
            # Token-based: Bin by max_activation
            all_max = all_sequences_max.get(feat_idx, [])
            if len(all_max) == 0:
                continue
            
            # Sort by max_activation (descending) - ALL sequences including non-firing
            all_max_sorted = sorted(all_max, key=lambda x: x[0], reverse=True)
            
            # Bin ALL sequences into 3 bins (high/middle/low)
            n_total = len(all_max_sorted)
            bin_size = max(1, n_total // 3)
            
            high_bin_max = all_max_sorted[:bin_size]
            middle_bin_max = all_max_sorted[bin_size:2*bin_size]
            low_bin_max = all_max_sorted[2*bin_size:]
            
            # Remove non-firing sequences from high bin (activation > 0 check)
            high_bin_max_fired = [seq for seq in high_bin_max if seq[0] > 0]
            
            # If high bin < 1/3 after removal, use all fired sequences from entire dataset
            if len(high_bin_max_fired) < bin_size:
                # Get all fired sequences (from entire dataset, not just high bin)
                all_fired_max = [seq for seq in all_max_sorted if seq[0] > 0]
                high_bin_max_fired = all_fired_max
            
            # Sample explanation sequences: top n_explanation from high bin (without replacement)
            n_explanation_actual = min(n_explanation, len(high_bin_max_fired))
            if n_explanation_actual > 0:
                high_explanation_max = high_bin_max_fired[:n_explanation_actual]  # Top n, no sampling needed
            else:
                high_explanation_max = []
            
            # Sample evaluation sequences: remaining high bin (after removing explanation ones, without replacement)
            remaining_high_max = high_bin_max_fired[n_explanation_actual:]
            n_evaluation_actual = min(n_evaluation, len(remaining_high_max))
            if n_evaluation_actual > 0:
                if len(remaining_high_max) >= n_evaluation_actual:
                    high_evaluation_max = random.sample(remaining_high_max, n_evaluation_actual)  # Sample without replacement
                else:
                    high_evaluation_max = remaining_high_max
            else:
                high_evaluation_max = []
            
            # Sample low sequences: from low bin (without replacement)
            n_low_actual = min(n_low, len(low_bin_max))
            if n_low_actual > 0:
                if len(low_bin_max) >= n_low_actual:
                    low_contrast_max = random.sample(low_bin_max, n_low_actual)  # Sample without replacement
                else:
                    low_contrast_max = low_bin_max
            else:
                low_contrast_max = []
            
            # Sentence-based: Bin by sum_activation
            all_sum = all_sequences_sum.get(feat_idx, [])
            if len(all_sum) > 0:
                # Sort by sum_activation (descending) - ALL sequences including non-firing
                all_sum_sorted = sorted(all_sum, key=lambda x: x[0], reverse=True)
                
                # Bin ALL sequences into 3 bins (high/middle/low)
                n_total_sum = len(all_sum_sorted)
                bin_size_sum = max(1, n_total_sum // 3)
                high_bin_sum = all_sum_sorted[:bin_size_sum]
                low_bin_sum = all_sum_sorted[2*bin_size_sum:]
                
                # Remove non-firing sequences from high bin (sum_activation > 0)
                high_bin_sum_fired = [seq for seq in high_bin_sum if seq[0] > 0]
                
                # If high bin < 1/3 after removal, use all fired sequences from entire dataset
                if len(high_bin_sum_fired) < bin_size_sum:
                    # Get all fired sequences (from entire dataset, not just high bin)
                    all_fired_sum = [seq for seq in all_sum_sorted if seq[0] > 0]
                    high_bin_sum_fired = all_fired_sum
                
                # Create set of explanation sequence IDs (batch_idx, seq_idx) to exclude from evaluation
                explanation_ids = {(batch_idx, seq_idx) for _, _, batch_idx, seq_idx, _ in high_explanation_max}
                
                # Remove explanation sequences from high bin for sentence-based evaluation
                remaining_high_sum = [seq for seq in high_bin_sum_fired 
                                     if (seq[2], seq[3]) not in explanation_ids]  # seq[2]=batch_idx, seq[3]=seq_idx
                
                # Sample evaluation sequences: from remaining high bin (without replacement)
                n_evaluation_sum_actual = min(n_evaluation, len(remaining_high_sum))
                if n_evaluation_sum_actual > 0:
                    if len(remaining_high_sum) >= n_evaluation_sum_actual:
                        high_evaluation_by_sum = random.sample(remaining_high_sum, n_evaluation_sum_actual)  # Sample without replacement
                    else:
                        high_evaluation_by_sum = remaining_high_sum
                else:
                    # Fallback: use high bin even if it includes explanation sequences
                    if len(high_bin_sum_fired) >= n_evaluation:
                        high_evaluation_by_sum = random.sample(high_bin_sum_fired, n_evaluation)
                    else:
                        high_evaluation_by_sum = high_bin_sum_fired
                
                # Sample low sequences: from low bin (without replacement)
                n_low_sum_actual = min(n_low, len(low_bin_sum))
                if n_low_sum_actual > 0:
                    if len(low_bin_sum) >= n_low_sum_actual:
                        low_contrast_by_sum = random.sample(low_bin_sum, n_low_sum_actual)  # Sample without replacement
                    else:
                        low_contrast_by_sum = low_bin_sum
                else:
                    low_contrast_by_sum = []
            else:
                high_evaluation_by_sum = []
                low_contrast_by_sum = []
            
            # Format sequence references (convert to format expected by collect_optimized)
            sequence_references[feat_idx] = {
                'high_explanation': [(max_act, batch_idx, seq_idx, sum_act, max_tok_idx) 
                                     for max_act, sum_act, batch_idx, seq_idx, max_tok_idx in high_explanation_max],
                'high_evaluation': [(max_act, batch_idx, seq_idx, sum_act, max_tok_idx) 
                                   for max_act, sum_act, batch_idx, seq_idx, max_tok_idx in high_evaluation_max],
                'high_evaluation_by_sum': [(sum_act, batch_idx, seq_idx, max_act, max_tok_idx) 
                                           for sum_act, max_act, batch_idx, seq_idx, max_tok_idx in high_evaluation_by_sum],
                'low_contrast': [(max_act, batch_idx, seq_idx, sum_act, max_tok_idx) 
                                for max_act, sum_act, batch_idx, seq_idx, max_tok_idx in low_contrast_max],
                'low_contrast_by_sum': [(sum_act, batch_idx, seq_idx, max_act, max_tok_idx) 
                                       for sum_act, max_act, batch_idx, seq_idx, max_tok_idx in low_contrast_by_sum],
            }
            
            collection_stats[feat_idx] = {
                'n_total_max': len(all_max),
                'n_total_sum': len(all_sum),
                'n_high_bin_max': len(high_bin_max),
                'n_high_bin_max_fired': len(high_bin_max_fired),
                'n_low_bin_max': len(low_bin_max),
                'n_high_explanation': len(high_explanation_max),
                'n_high_evaluation': len(high_evaluation_max),
                'n_low_contrast': len(low_contrast_max),
            }
        
        return sequence_references, collection_stats
    
    def fast_count_features(
        self,
        n_batches: int,
        use_absolute_value: bool = False,
        min_firing_count: int = 80,
        progress_callback=None
    ) -> Tuple[List[int], np.ndarray, int, Dict[int, int]]:
        """
        Pass 0: Fast counting for feature discovery and quantile sampling.
        
        Scans batches and counts how many sequences each feature fired in.
        Does NOT decode tokens or store full sequences - just counts.
        
        Args:
            n_batches: Number of batches to scan
            use_absolute_value: Whether to use absolute values for activations
            min_firing_count: Minimum firing count to consider feature active
            progress_callback: Optional callback function(batch_idx, n_batches) for progress updates
        
        Returns:
            Tuple of:
            - active_features: List of feature indices that fired at least min_firing_count times
            - sequence_firing_counts: Array of firing counts per feature
            - total_sequences: Total number of sequences processed
            - feature_firing_map: Dict mapping feat_idx -> firing_count (for convenience)
        """
        active_features = set()
        sequence_firing_counts = np.zeros(self.sae.cfg.d_sae, dtype=np.int64)
        total_sequences = 0
        
        for batch_idx in range(n_batches):
            batch_tokens = self.store.get_batch_tokens()
            if batch_tokens is None:
                break
            
            batch_size = batch_tokens.shape[0]
            total_sequences += batch_size
            
            with torch.no_grad():
                model_acts = self.store.get_activations(batch_tokens)
                seq_len = batch_tokens.shape[1]
                model_acts_flat = model_acts.reshape(-1, model_acts.shape[-1])
                model_acts_flat = model_acts_flat.to(self.device)
                
                sae_acts_flat = self.sae.encode(model_acts_flat)
                sae_acts = sae_acts_flat.reshape(batch_size, seq_len, self.sae.cfg.d_sae)
                
                if use_absolute_value:
                    sae_acts = torch.abs(sae_acts)
                
                # Count sequences where feature fired (fast - batch processing on GPU)
                # Process entire batch on GPU first, then transfer once
                batch_fired = (sae_acts > 0).any(dim=1)  # [batch_size, d_sae] - all on GPU
                batch_fired_cpu = batch_fired.cpu().numpy()  # ONE transfer for entire batch
                
                # Update counts (all on CPU numpy arrays)
                sequence_firing_counts += batch_fired_cpu.sum(axis=0).astype(np.int64)
                
                # Update active features set
                for seq_idx in range(batch_size):
                    active_features.update(np.where(batch_fired_cpu[seq_idx])[0])
            
            if progress_callback:
                progress_callback(batch_idx + 1, n_batches)
        
        # Filter to features with minimum firing count
        active_features_list = sorted([f for f in active_features if sequence_firing_counts[f] >= min_firing_count])
        feature_firing_map = {f: int(sequence_firing_counts[f]) for f in active_features_list}
        
        return active_features_list, sequence_firing_counts, total_sequences, feature_firing_map
    
    @staticmethod
    def sample_features_by_quantiles(
        active_features: List[int],
        sequence_firing_counts: np.ndarray,
        total_sequences: int,
        n_features_per_quantile: int = 10,
        min_firing_count: int = 40
    ) -> Tuple[List[int], Dict[str, Dict]]:
        """
        Sample features by quantiles based on firing frequency.
        
        Args:
            active_features: List of active feature indices
            sequence_firing_counts: Array of firing counts per feature
            total_sequences: Total number of sequences processed
            n_features_per_quantile: Number of features to sample per quantile
            min_firing_count: Minimum firing count (already filtered, but kept for consistency)
        
        Returns:
            Tuple of:
            - sampled_features: List of sampled feature indices
            - quantile_stats: Dict mapping quantile name -> stats dict
        """
        # Filter and sort by firing frequency
        filtered_features = [f for f in active_features if sequence_firing_counts[f] >= min_firing_count]
        filtered_sorted = sorted(filtered_features, key=lambda f: sequence_firing_counts[f], reverse=True)
        
        n_filtered = len(filtered_sorted)
        quartile_size = n_filtered // 4
        quartile_boundaries = [
            (0, quartile_size),
            (quartile_size, 2 * quartile_size),
            (2 * quartile_size, 3 * quartile_size),
            (3 * quartile_size, n_filtered)
        ]
        
        sampled_features = []
        quantile_stats = {}
        quantile_names = ["Q1 (0-25%)", "Q2 (25-50%)", "Q3 (50-75%)", "Q4 (75-100%)"]
        
        for quartile_idx, (start, end) in enumerate(quartile_boundaries):
            quartile_features = filtered_sorted[start:end]
            n_to_take = min(n_features_per_quantile, len(quartile_features))
            sampled = quartile_features[:n_to_take]
            sampled_features.extend(sampled)
            
            if len(quartile_features) > 0:
                quartile_firing_counts = [sequence_firing_counts[f] for f in quartile_features]
                max_firing = max(quartile_firing_counts)
                min_firing = min(quartile_firing_counts)
                mean_firing = sum(quartile_firing_counts) / len(quartile_firing_counts)
                firing_percentage = (mean_firing / total_sequences) * 100 if total_sequences > 0 else 0.0
                
                quantile_stats[quantile_names[quartile_idx]] = {
                    'n_features': len(quartile_features),
                    'n_sampled': n_to_take,
                    'max_firing': max_firing,
                    'min_firing': min_firing,
                    'mean_firing': mean_firing,
                    'firing_percentage': firing_percentage,
                }
        
        return sampled_features, quantile_stats
    
    def fast_count_dominance(
        self,
        n_batches: int,
        use_absolute_value: bool = False,
        min_firing_count: int = 80,
        progress_callback=None
    ) -> Tuple[List[int], np.ndarray, np.ndarray, int, Dict[int, float]]:
        """
        Pass 0: Fast counting for dominance-based feature discovery.
        
        Computes dominance = average (feature_activation / total_activation) across ALL input token positions.
        This includes positions where the feature didn't fire (dominance = 0), giving the true average across all data.
        
        Args:
            n_batches: Number of batches to scan
            use_absolute_value: Whether to use absolute values for activations
            min_firing_count: Minimum firing count to consider feature active
            progress_callback: Optional callback function(batch_idx, n_batches) for progress updates
        
        Returns:
            Tuple of:
            - active_features: List of feature indices that fired at least min_firing_count times
            - sequence_firing_counts: Array of firing counts per feature (same as frequency-based)
            - dominance_scores: Array of average dominance per feature
            - total_sequences: Total number of sequences processed
            - feature_dominance_map: Dict mapping feat_idx -> dominance_score
        """
        active_features = set()
        sequence_firing_counts = np.zeros(self.sae.cfg.d_sae, dtype=np.int64)
        
        # For dominance computation:
        # - dominance_sum: sum of (feature_act / total_act) for each token position
        # - dominance_count: number of ALL token positions processed (not just firings)
        dominance_sum = np.zeros(self.sae.cfg.d_sae, dtype=np.float64)
        dominance_count = np.zeros(self.sae.cfg.d_sae, dtype=np.int64)
        
        total_sequences = 0
        
        for batch_idx in range(n_batches):
            batch_tokens = self.store.get_batch_tokens()
            if batch_tokens is None:
                break
            
            batch_size = batch_tokens.shape[0]
            total_sequences += batch_size
            
            with torch.no_grad():
                model_acts = self.store.get_activations(batch_tokens)
                seq_len = batch_tokens.shape[1]
                model_acts_flat = model_acts.reshape(-1, model_acts.shape[-1])
                model_acts_flat = model_acts_flat.to(self.device)
                
                sae_acts_flat = self.sae.encode(model_acts_flat)
                sae_acts = sae_acts_flat.reshape(batch_size, seq_len, self.sae.cfg.d_sae)
                
                if use_absolute_value:
                    sae_acts = torch.abs(sae_acts)
                
                # Convert to numpy for dominance computation
                sae_acts_np = sae_acts.cpu().numpy()  # [batch_size, seq_len, d_sae]
                
                # For each token position, compute dominance
                for seq_idx in range(batch_size):
                    seq_acts = sae_acts_np[seq_idx]  # [seq_len, d_sae]
                    
                    # Count sequences where feature fired
                    seq_fired = (seq_acts > 0).any(axis=0)  # [d_sae]
                    sequence_firing_counts += seq_fired.astype(np.int64)
                    active_features.update(np.where(seq_fired)[0])
                    
                    # Compute dominance for each token position
                    # IMPORTANT: Compute across ALL inputs, not just when feature fires
                    for tok_idx in range(seq_len):
                        tok_acts = seq_acts[tok_idx]  # [d_sae]
                        
                        # Total activation at this position (sum of all features)
                        total_act = np.sum(tok_acts)
                        
                        if total_act > 0:
                            # For ALL features (even if activation is 0) - compute dominance across all inputs
                            for feat_idx in range(self.sae.cfg.d_sae):
                                # Dominance = this feature's activation / total activation (can be 0 if feature didn't fire)
                                dominance = tok_acts[feat_idx] / total_act
                                dominance_sum[feat_idx] += dominance
                                dominance_count[feat_idx] += 1  # Count ALL token positions
                        else:
                            # If total_act == 0, all features have 0 activation, dominance is undefined
                            # But we still count the position for consistency (dominance = 0)
                            for feat_idx in range(self.sae.cfg.d_sae):
                                dominance_count[feat_idx] += 1  # Count position, dominance remains 0 (sum stays 0)
            
            if progress_callback:
                progress_callback(batch_idx + 1, n_batches)
        
        # Compute average dominance: sum / count
        # dominance_count now represents ALL token positions processed (not just firings)
        dominance_scores = np.zeros(self.sae.cfg.d_sae, dtype=np.float64)
        for feat_idx in range(self.sae.cfg.d_sae):
            if dominance_count[feat_idx] > 0:
                # Average across ALL input token positions (includes positions where feature didn't fire)
                dominance_scores[feat_idx] = dominance_sum[feat_idx] / dominance_count[feat_idx]
        
        # Filter to features with minimum firing count
        active_features_list = sorted([f for f in active_features if sequence_firing_counts[f] >= min_firing_count])
        feature_dominance_map = {f: float(dominance_scores[f]) for f in active_features_list}
        
        return active_features_list, sequence_firing_counts, dominance_scores, total_sequences, feature_dominance_map
    
    @staticmethod
    def sample_features_by_dominance_quantiles(
        active_features: List[int],
        dominance_scores: np.ndarray,
        sequence_firing_counts: np.ndarray,
        total_sequences: int,
        n_features_per_quantile: int = 10,
        min_firing_count: int = 80
    ) -> Tuple[List[int], Dict[str, Dict]]:
        """
        Sample features by quantiles based on DOMINANCE (not frequency).
        
        Args:
            active_features: List of active feature indices
            dominance_scores: Array of dominance scores per feature
            sequence_firing_counts: Array of firing counts per feature (for stats)
            total_sequences: Total number of sequences processed
            n_features_per_quantile: Number of features to sample per quantile
            min_firing_count: Minimum firing count (for filtering)
        
        Returns:
            Tuple of:
            - sampled_features: List of sampled feature indices
            - quantile_stats: Dict mapping quantile name -> stats dict
        """
        # Filter by minimum firing count
        filtered_features = [f for f in active_features if sequence_firing_counts[f] >= min_firing_count]
        
        # Sort by DOMINANCE (descending - highest dominance first)
        filtered_sorted = sorted(filtered_features, key=lambda f: dominance_scores[f], reverse=True)
        
        n_filtered = len(filtered_sorted)
        quartile_size = n_filtered // 4
        quartile_boundaries = [
            (0, quartile_size),
            (quartile_size, 2 * quartile_size),
            (2 * quartile_size, 3 * quartile_size),
            (3 * quartile_size, n_filtered)
        ]
        
        sampled_features = []
        quantile_stats = {}
        quantile_names = ["Q1 (0-25%)", "Q2 (25-50%)", "Q3 (50-75%)", "Q4 (75-100%)"]
        
        for quartile_idx, (start, end) in enumerate(quartile_boundaries):
            quartile_features = filtered_sorted[start:end]
            n_to_take = min(n_features_per_quantile, len(quartile_features))
            sampled = quartile_features[:n_to_take]
            sampled_features.extend(sampled)
            
            if len(quartile_features) > 0:
                quartile_dominance_scores = [dominance_scores[f] for f in quartile_features]
                quartile_firing_counts = [sequence_firing_counts[f] for f in quartile_features]
                max_dominance = max(quartile_dominance_scores)
                min_dominance = min(quartile_dominance_scores)
                mean_dominance = sum(quartile_dominance_scores) / len(quartile_dominance_scores)
                max_firing = max(quartile_firing_counts)
                min_firing = min(quartile_firing_counts)
                mean_firing = sum(quartile_firing_counts) / len(quartile_firing_counts)
                firing_percentage = (mean_firing / total_sequences) * 100 if total_sequences > 0 else 0.0
                
                quantile_stats[quantile_names[quartile_idx]] = {
                    'n_features': len(quartile_features),
                    'n_sampled': n_to_take,
                    'min_dominance': float(min_dominance),
                    'max_dominance': float(max_dominance),
                    'mean_dominance': float(mean_dominance),
                    'min_firing': min_firing,
                    'max_firing': max_firing,
                    'mean_firing': mean_firing,
                    'firing_percentage': firing_percentage,
                }
        
        return sampled_features, quantile_stats

