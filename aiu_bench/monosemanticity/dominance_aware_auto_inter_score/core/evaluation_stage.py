"""
Stage 3: Multiple-Choice Evaluation

Evaluates explanations using token-based and sentence-based multiple choice questions.
"""

import asyncio
import os
import random
import torch
from typing import List, Dict, Set, Optional
from openai import AsyncOpenAI
import torch
from dotenv import load_dotenv
import re
from tqdm import tqdm
import numpy as np
from scipy.stats import spearmanr

from .collection_stage import MarkedSequence, ActivationCollector

load_dotenv()


def parse_multiple_choice_answer(response_text: str, n_options: int) -> Optional[int]:
    """Parse multiple choice answer from model response."""
    response_text = response_text.strip().upper()
    
    # Try letter (A, B, C, ...)
    letter_match = re.search(r'\b([A-Z])\b', response_text)
    if letter_match:
        letter = letter_match.group(1)
        idx = ord(letter) - ord('A')
        if 0 <= idx < n_options:
            return idx
    
    # Try number (1, 2, 3, ...)
    number_match = re.search(r'\b(\d+)\b', response_text)
    if number_match:
        num = int(number_match.group(1))
        if 1 <= num <= n_options:
            return num - 1
        elif 0 <= num < n_options:
            return num
    
    return None


def parse_ranking_response(response_text: str) -> Optional[Dict[str, int]]:
    """
    Parse ranking response in format "Example 1=?, Example 2=?, Example 3=?".
    
    Returns dict mapping "Example 1" -> rank (1, 2, or 3), or None if parsing fails.
    """
    response_text = response_text.strip()
    
    # Pattern: "Example 1=1, Example 2=2, Example 3=3" or "Example 1=1 Example 2=2 Example 3=3"
    ranking = {}
    
    # Try to find all "Example X=Y" patterns
    pattern = r'Example\s+(\d+)\s*=\s*(\d+)'
    matches = re.findall(pattern, response_text, re.IGNORECASE)
    
    if len(matches) == 3:
        for example_num_str, rank_str in matches:
            example_num = int(example_num_str)
            rank = int(rank_str)
            if 1 <= example_num <= 3 and 1 <= rank <= 3:
                ranking[f"Example {example_num}"] = rank
        
        if len(ranking) == 3:
            return ranking
    
    # Fallback: try to extract numbers in order
    numbers = re.findall(r'\b([123])\b', response_text)
    if len(numbers) == 3:
        # Assume order is Example 1, Example 2, Example 3
        ranking = {
            "Example 1": int(numbers[0]),
            "Example 2": int(numbers[1]),
            "Example 3": int(numbers[2])
        }
        return ranking
    
    return None


def compute_ranking_metrics(
    predicted_ranking: Dict[str, int],
    true_ranking: Dict[str, int]
) -> Dict[str, float]:
    """
    Compute 4 metrics for ranking evaluation.
    
    Args:
        predicted_ranking: Dict mapping "Example 1" -> rank (1-3)
        true_ranking: Dict mapping "Example 1" -> rank (1-3)
    
    Returns:
        Dict with:
        - perfect_ranking: 1.0 if exact match, 0.0 otherwise
        - rank_correlation: Spearman correlation (-1 to 1)
        - pairwise_accuracy: Dict with A_vs_B, B_vs_C, A_vs_C (each 1.0 or 0.0)
        - rank_distance: Average absolute difference in positions
    """
    
    # Extract ranks in order: Example 1, Example 2, Example 3
    pred_ranks = [predicted_ranking.get(f"Example {i}", 0) for i in [1, 2, 3]]
    true_ranks = [true_ranking.get(f"Example {i}", 0) for i in [1, 2, 3]]
    
    # 1. Perfect ranking accuracy
    perfect_ranking = 1.0 if pred_ranks == true_ranks else 0.0
    
    # 2. Rank correlation (Spearman)
    try:
        if len(set(pred_ranks)) > 1 and len(set(true_ranks)) > 1:  # Need variation for correlation
            corr, _ = spearmanr(pred_ranks, true_ranks)
            rank_correlation = float(corr) if not np.isnan(corr) else 0.0
        else:
            rank_correlation = 1.0 if pred_ranks == true_ranks else 0.0
    except:
        rank_correlation = 0.0
    
    # 3. Pairwise accuracy
    # True ranking: Example 1 (BIN A) = 1, Example 2 (BIN B) = 2, Example 3 (BIN C) = 3
    # So: A > B, B > C, A > C
    pred_A = pred_ranks[0]  # Example 1 (BIN A)
    pred_B = pred_ranks[1]  # Example 2 (BIN B)
    pred_C = pred_ranks[2]  # Example 3 (BIN C)
    
    true_A = true_ranks[0]
    true_B = true_ranks[1]
    true_C = true_ranks[2]
    
    pairwise_accuracy = {
        "A_vs_B": 1.0 if (pred_A < pred_B) == (true_A < true_B) else 0.0,  # Lower rank = higher activation
        "B_vs_C": 1.0 if (pred_B < pred_C) == (true_B < true_C) else 0.0,
        "A_vs_C": 1.0 if (pred_A < pred_C) == (true_A < true_C) else 0.0,
    }
    
    # 4. Rank distance (average absolute difference)
    rank_distance = np.mean([abs(p - t) for p, t in zip(pred_ranks, true_ranks)])
    
    return {
        "perfect_ranking": perfect_ranking,
        "rank_correlation": rank_correlation,
        "pairwise_accuracy": pairwise_accuracy,
        "rank_distance": rank_distance
    }


async def ask_model_question(
    client: Optional[AsyncOpenAI] = None,
    model_name: str = None,
    prompt: str = None,
    max_tokens: int = 10,
    provider: str = "openai",
    offline_model = None,
    offline_tokenizer = None,
    device: str = None,
    prompt_cache_key: Optional[str] = None,  # For OpenAI caching optimization
) -> str:
    """Ask model a multiple choice question."""
    if provider == "openai":
        try:
            create_kwargs = {
                "model": model_name,
                "messages": [
                    {"role": "system", "content": "Answer multiple choice questions concisely with just the letter or number."},
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": max_tokens,
                "temperature": 0.0
            }
            # Enable caching if prompt is long enough (> 1024 tokens)
            # Use provided cache key, or infer from prompt
            if len(prompt) > 1024:
                if prompt_cache_key:
                    create_kwargs["prompt_cache_key"] = prompt_cache_key
                else:
                    # Fallback: infer from prompt content
                    if "token" in prompt.lower() or "Token" in prompt:
                        create_kwargs["prompt_cache_key"] = "sae_eval_token"
                    elif "sentence" in prompt.lower() or "Sentence" in prompt:
                        create_kwargs["prompt_cache_key"] = "sae_eval_sentence"
                    else:
                        create_kwargs["prompt_cache_key"] = "sae_eval_generic"
            
            response = await client.chat.completions.create(**create_kwargs)
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"Error calling API: {e}")
            return ""
    else:  # offline
        try:
            # Format prompt for chat model
            messages = [
                {"role": "system", "content": "Answer multiple choice questions concisely with just the letter or number."},
                {"role": "user", "content": prompt}
            ]
            
            # Tokenize
            formatted_prompt = offline_tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = offline_tokenizer(formatted_prompt, return_tensors="pt")
            
            # Move inputs to the same device as the model
            model_device = next(offline_model.parameters()).device
            inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = offline_model.generate(
                    **inputs,
                    max_new_tokens=max_tokens,
                    temperature=0.0,
                    do_sample=False,
                    pad_token_id=offline_tokenizer.eos_token_id,
                )
            
            # Decode (only the new tokens)
            generated_text = offline_tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:], 
                skip_special_tokens=True
            )
            return generated_text.strip()
        except Exception as e:
            print(f"Error generating with offline model: {e}")
            return ""


async def ask_model_questions_batch(
    prompts: List[str],
    model_name: str = None,
    max_tokens: int = 10,
    provider: str = "offline",
    offline_model = None,
    offline_tokenizer = None,
    device: str = None,
    batch_size: int = 4,  # Optimal batch size for 20 questions (5 batches)
) -> List[str]:
    """
    Ask model multiple choice questions in batch (offline mode only).
    
    Args:
        prompts: List of question prompts
        model_name: Model name (unused for offline, kept for API compatibility)
        max_tokens: Maximum tokens to generate
        provider: Must be "offline" for batch processing
        offline_model: Loaded model
        offline_tokenizer: Loaded tokenizer
        device: Device to use
        batch_size: Batch size for processing (default 4, optimal for 20 questions)
    
    Returns:
        List of model responses
    """
    if provider != "offline":
        raise ValueError(f"Batch processing only supported for offline provider, got {provider}")
    
    if offline_model is None or offline_tokenizer is None:
        raise ValueError("offline_model and offline_tokenizer must be provided for batch processing")
    
    # Ensure pad_token is set
    if offline_tokenizer.pad_token is None:
        offline_tokenizer.pad_token = offline_tokenizer.eos_token
    
    all_responses = []
    
    # Process in batches
    for i in range(0, len(prompts), batch_size):
        batch_prompts = prompts[i:i+batch_size]
        
        # Format prompts for chat model
        messages_list = [
            [
                {"role": "system", "content": "Answer multiple choice questions concisely with just the letter or number."},
                {"role": "user", "content": prompt}
            ]
            for prompt in batch_prompts
        ]
        
        # Apply chat template and tokenize
        formatted_prompts = [
            offline_tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
            for msgs in messages_list
        ]
        
        # Tokenize with padding
        inputs = offline_tokenizer(
            formatted_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=4096,  # Mistral context window
        )
        
        # Move inputs to the same device as the model
        model_device = next(offline_model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
        
        # Generate
        with torch.no_grad():
            outputs = offline_model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=0.0,
                do_sample=False,
                pad_token_id=offline_tokenizer.eos_token_id,
            )
        
        # Decode (remove prompt tokens)
        batch_responses = []
        for j, output in enumerate(outputs):
            prompt_length = inputs['input_ids'][j].shape[0]
            generated = offline_tokenizer.decode(
                output[prompt_length:],
                skip_special_tokens=True
            )
            batch_responses.append(generated.strip())
        
        all_responses.extend(batch_responses)
    
    return all_responses


class TokenBasedEvaluator:
    """Token-based multiple choice evaluator using cross-sequence evaluation."""
    
    def __init__(
        self, 
        model_name: str = "gpt-4.1-nano", 
        max_questions: int = 20, 
        n_examples: int = 10, 
        n_distractors: int = 2,
        provider: str = "openai",  # "openai" or "offline"
        device: str = None,
    ):
        self.model_name = model_name
        self.max_questions = max_questions
        self.n_examples = n_examples  # Number of examples used in explanation (for pool sizing)
        self.n_distractors = n_distractors  # Number of low sequences to use as distractors (1-3)
        self.provider = provider
        self.device = device or "cuda:0" if torch.cuda.is_available() else "cpu"
        
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found")
            self.client = AsyncOpenAI(api_key=api_key)
            self.offline_model = None
            self.offline_tokenizer = None
        elif provider == "offline":
            # Load model directly with transformers (no vLLM)
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"Loading offline model: {model_name}")
            self.offline_tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Use specific device instead of "auto" to prevent splitting across GPUs
            self.use_device_map = False
            try:
                self.offline_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map=self.device,  # Use specific device, not "auto"
                    low_cpu_mem_usage=True,
                )
                self.use_device_map = True
                # Get the device of the first parameter (for input placement)
                self.model_device = next(self.offline_model.parameters()).device
            except Exception as e:
                # Fallback: try without device_map and move manually
                print(f"Warning: device_map='{self.device}' failed, trying manual device placement: {e}")
                try:
                    self.offline_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                    )
                    self.offline_model = self.offline_model.to(self.device)
                    self.model_device = self.device
                except Exception as e2:
                    print(f"Error: All loading methods failed: {e2}")
                    raise
            
            self.offline_model.eval()  # Set to evaluation mode
            self.client = None
            print(f"✓ Loaded offline model: {model_name}")
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'offline'")
    
    def cleanup(self):
        """Clean up offline model and tokenizer to free memory."""
        if self.provider == "offline" and self.offline_model is not None:
            print(f"[Cleanup] Cleaning up offline sentence evaluator model...")
            del self.offline_model
            del self.offline_tokenizer
            self.offline_model = None
            self.offline_tokenizer = None
            if torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print(f"✓ Cleaned up offline sentence evaluator model")
    
    async def evaluate_feature(
        self,
        feature_idx: int,
        explanation: str,
        bin_A_sequences: List[MarkedSequence],
        bin_B_sequences: List[MarkedSequence],
        bin_C_sequences: List[MarkedSequence],
        explanation_sequence_ids: Optional[Set[str]] = None
    ) -> Dict:
        """
        Evaluate feature using 3-way ranking task (V2).
        
        For each question:
        - Sample 1 from BIN A (HIGH-high)
        - Sample 1 from BIN B (HIGH-low)
        - Sample 1 from BIN C (LOW)
        - Ask LLM to rank by activation strength (1=highest, 2=middle, 3=lowest)
        - True ranking: Example 1 (BIN A) = 1, Example 2 (BIN B) = 2, Example 3 (BIN C) = 3
        
        Args:
            feature_idx: Feature index
            explanation: Generated explanation for this feature
            bin_A_sequences: HIGH-high sequences (~top 25% of firing, excluding top 20 for explanation)
            bin_B_sequences: HIGH-low sequences (~25-50% of firing)
            bin_C_sequences: LOW sequences (bottom 50%, may include non-firing)
            explanation_sequence_ids: Set of sequence IDs used in explanation (to exclude for fairness)
        """
        import random
        random.seed(42)
        
        # Exclude explanation sequences from all bins
        if explanation_sequence_ids:
            explanation_sequence_ids_set = set(explanation_sequence_ids)
            bin_A_sequences = [seq for seq in bin_A_sequences if seq.sequence_id not in explanation_sequence_ids_set]
            bin_B_sequences = [seq for seq in bin_B_sequences if seq.sequence_id not in explanation_sequence_ids_set]
            bin_C_sequences = [seq for seq in bin_C_sequences if seq.sequence_id not in explanation_sequence_ids_set]
        
        # Need at least 1 sequence from each bin per question
        n_questions = min(self.max_questions, len(bin_A_sequences), len(bin_B_sequences), len(bin_C_sequences))
        
        if n_questions == 0:
            print(f"    ⚠️  Feature {feature_idx}: Not enough sequences in bins (A: {len(bin_A_sequences)}, B: {len(bin_B_sequences)}, C: {len(bin_C_sequences)})")
            return {
                "feature_idx": feature_idx,
                "perfect_ranking_acc": 0.0,
                "mean_rank_correlation": 0.0,
                "mean_pairwise_acc": 0.0,
                "mean_rank_distance": 0.0,
                "n_questions": 0,
                "error": "Not enough sequences in bins",
                "error_detail": f"BIN A: {len(bin_A_sequences)}, BIN B: {len(bin_B_sequences)}, BIN C: {len(bin_C_sequences)}",
                "questions": []
            }
        
        print(f"    Feature {feature_idx}: Creating {n_questions} 3-way ranking questions...")
        
        questions = []
        question_details = []
        true_rankings = []  # List of true ranking dicts
        
        for q_idx in range(n_questions):
            # Sample 1 from each bin
            seq_A = bin_A_sequences[q_idx % len(bin_A_sequences)] if len(bin_A_sequences) > 0 else None
            seq_B = bin_B_sequences[q_idx % len(bin_B_sequences)] if len(bin_B_sequences) > 0 else None
            seq_C = bin_C_sequences[q_idx % len(bin_C_sequences)] if len(bin_C_sequences) > 0 else None
            
            if seq_A is None or seq_B is None or seq_C is None:
                continue
            
            if len(seq_A.tokens) == 0 or len(seq_B.tokens) == 0 or len(seq_C.tokens) == 0:
                continue
            
            # Find token with highest activation in each sequence
            token_A_idx = max(range(len(seq_A.activations)), key=lambda i: seq_A.activations[i])
            token_B_idx = max(range(len(seq_B.activations)), key=lambda i: seq_B.activations[i])
            token_C_idx = max(range(len(seq_C.activations)), key=lambda i: seq_C.activations[i])
            
            token_A = seq_A.tokens[token_A_idx]
            token_B = seq_B.tokens[token_B_idx]
            token_C = seq_C.tokens[token_C_idx]
            
            activation_A = seq_A.activations[token_A_idx]
            activation_B = seq_B.activations[token_B_idx]
            activation_C = seq_C.activations[token_C_idx]
            
            # True ranking: A=1 (highest), B=2 (middle), C=3 (lowest)
            true_ranking = {
                "Example 1": 1,  # BIN A (HIGH-high)
                "Example 2": 2,  # BIN B (HIGH-low)
                "Example 3": 3   # BIN C (LOW)
            }
            true_rankings.append(true_ranking)
            
            # Build ranking question
            question_parts = [
                f"Given this explanation:",
                f"\"{explanation}\"",
                "",
                f"Rank these 3 examples by activation strength (1=highest, 2=middle, 3=lowest):",
                "",
                f"Example 1: \"{' '.join(seq_A.tokens)}\"",
                f"  Token: {token_A}",
                "",
                f"Example 2: \"{' '.join(seq_B.tokens)}\"",
                f"  Token: {token_B}",
                "",
                f"Example 3: \"{' '.join(seq_C.tokens)}\"",
                f"  Token: {token_C}",
                "",
                f"Output only: Example 1=?, Example 2=?, Example 3=?"
            ]
            
            question_text = "\n".join(question_parts)
            questions.append(question_text)
            
            question_details.append({
                "question": question_text,
                "explanation": explanation,
                "examples": [
                    {
                        "label": "Example 1",
                        "bin": "A",
                        "sequence_text": " ".join(seq_A.tokens),
                        "token": token_A,
                        "activation": float(activation_A),
                        "true_rank": 1
                    },
                    {
                        "label": "Example 2",
                        "bin": "B",
                        "sequence_text": " ".join(seq_B.tokens),
                        "token": token_B,
                        "activation": float(activation_B),
                        "true_rank": 2
                    },
                    {
                        "label": "Example 3",
                        "bin": "C",
                        "sequence_text": " ".join(seq_C.tokens),
                        "token": token_C,
                        "activation": float(activation_C),
                        "true_rank": 3
                    }
                ],
                "true_ranking": true_ranking
            })
        
        if len(questions) == 0:
            print(f"    Feature {feature_idx}: ⚠️  No valid ranking questions")
            return {
                "feature_idx": feature_idx,
                "perfect_ranking_acc": 0.0,
                "mean_rank_correlation": 0.0,
                "mean_pairwise_acc": 0.0,
                "mean_rank_distance": 0.0,
                "n_questions": 0,
                "error": "Too few sequences",
                "error_detail": f"BIN A: {len(bin_A_sequences)}, BIN B: {len(bin_B_sequences)}, BIN C: {len(bin_C_sequences)}",
                "questions": []
            }
        
        # Evaluate questions
        print(f"    Feature {feature_idx}: Evaluating {len(questions)} ranking questions...")
        responses = []
        
        if self.provider == "offline":
            # Use batch processing for offline mode (faster)
            batch_size = 4  # Optimal batch size for 20 questions (5 batches)
            responses = await ask_model_questions_batch(
                prompts=questions,
                model_name=self.model_name,
                max_tokens=20,  # Need more tokens for ranking format
                provider=self.provider,
                offline_model=self.offline_model,
                offline_tokenizer=self.offline_tokenizer,
                device=self.device,
                batch_size=batch_size,
            )
        else:
            # Sequential processing for OpenAI API
            for i, question in enumerate(tqdm(questions, desc=f"      Feature {feature_idx} ranking", leave=False)):
                response = await ask_model_question(
                    self.client, 
                    self.model_name, 
                    question,
                    max_tokens=20,  # Need more tokens for ranking format
                    provider=self.provider,
                    offline_model=self.offline_model,
                    offline_tokenizer=self.offline_tokenizer,
                    device=self.device,
                )
                responses.append(response)
                if self.provider == "openai":
                    await asyncio.sleep(0.5)  # Rate limiting for API
        
        # Parse ranking responses and compute metrics
        all_metrics = []
        for i, (response, true_ranking) in enumerate(zip(responses, true_rankings)):
            predicted_ranking = parse_ranking_response(response)
            
            if predicted_ranking is None:
                # Failed to parse - use worst case ranking
                predicted_ranking = {"Example 1": 3, "Example 2": 3, "Example 3": 3}
            
            # Compute metrics for this question
            metrics = compute_ranking_metrics(predicted_ranking, true_ranking)
            all_metrics.append(metrics)
            
            # Add to question details
            question_details[i]["predicted_ranking"] = predicted_ranking
            question_details[i]["model_response"] = response
            question_details[i]["metrics"] = metrics
        
        # Aggregate metrics across all questions
        perfect_ranking_acc = np.mean([m["perfect_ranking"] for m in all_metrics])
        mean_rank_correlation = np.mean([m["rank_correlation"] for m in all_metrics])
        mean_pairwise_acc = np.mean([
            (m["pairwise_accuracy"]["A_vs_B"] + m["pairwise_accuracy"]["B_vs_C"] + m["pairwise_accuracy"]["A_vs_C"]) / 3.0
            for m in all_metrics
        ])
        mean_rank_distance = np.mean([m["rank_distance"] for m in all_metrics])
        
        return {
            "feature_idx": feature_idx,
            "perfect_ranking_acc": float(perfect_ranking_acc),
            "mean_rank_correlation": float(mean_rank_correlation),
            "mean_pairwise_acc": float(mean_pairwise_acc),
            "mean_rank_distance": float(mean_rank_distance),
            "n_questions": len(questions),
            "error": None,
            "questions": question_details
        }


class SentenceBasedEvaluator:
    """Sentence-based multiple choice evaluator with proper binning."""
    
    def __init__(
        self,
        model_name: str = "gpt-4.1-nano",
        n_distractors: int = 3,
        max_questions: int = 20,
        n_examples: int = 10,
        provider: str = "openai",  # "openai" or "offline"
        device: str = None,
    ):
        self.model_name = model_name
        self.n_distractors = n_distractors
        self.max_questions = max_questions
        self.n_examples = n_examples  # Number of examples used in explanation (for pool sizing)
        self.provider = provider
        self.device = device or "cuda:0" if torch.cuda.is_available() else "cpu" or "cuda:0" if torch.cuda.is_available() else "cpu"
        
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found")
            self.client = AsyncOpenAI(api_key=api_key)
            self.offline_model = None
            self.offline_tokenizer = None
        elif provider == "offline":
            # Load model directly with transformers (no vLLM)
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"Loading offline model: {model_name}")
            self.offline_tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Use specific device instead of "auto" to prevent splitting across GPUs
            self.use_device_map = False
            try:
                self.offline_model = AutoModelForCausalLM.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16,
                    device_map=self.device,  # Use specific device, not "auto"
                    low_cpu_mem_usage=True,
                )
                self.use_device_map = True
                # Get the device of the first parameter (for input placement)
                self.model_device = next(self.offline_model.parameters()).device
            except Exception as e:
                # Fallback: try without device_map and move manually
                print(f"Warning: device_map='{self.device}' failed, trying manual device placement: {e}")
                try:
                    self.offline_model = AutoModelForCausalLM.from_pretrained(
                        model_name,
                        torch_dtype=torch.float16,
                        low_cpu_mem_usage=True,
                    )
                    self.offline_model = self.offline_model.to(self.device)
                    self.model_device = self.device
                except Exception as e2:
                    print(f"Error: All loading methods failed: {e2}")
                    raise
            
            self.offline_model.eval()  # Set to evaluation mode
            self.client = None
            print(f"✓ Loaded offline model: {model_name}")
        else:
            raise ValueError(f"Unknown provider: {provider}. Use 'openai' or 'offline'")
    
    def cleanup(self):
        """Clean up offline model and tokenizer to free memory."""
        if self.provider == "offline" and self.offline_model is not None:
            print(f"[Cleanup] Cleaning up offline sentence evaluator model...")
            del self.offline_model
            del self.offline_tokenizer
            self.offline_model = None
            self.offline_tokenizer = None
            if torch.cuda.is_available():
                import gc
                gc.collect()
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            print(f"✓ Cleaned up offline sentence evaluator model")
    
    def extract_sentences_from_sequence(
        self,
        marked_seq: MarkedSequence
    ) -> List[tuple]:
        """
        Extract sentences from a marked sequence.
        
        Returns:
            List of (sentence_text, sentence_sum, is_high) tuples
        """
        sentences = []
        sentence_starts = marked_seq.sentence_boundaries + [len(marked_seq.tokens)]
        
        for i in range(len(sentence_starts) - 1):
            start = sentence_starts[i]
            end = sentence_starts[i + 1]
            
            sentence_tokens = marked_seq.tokens[start:end]
            sentence_acts = marked_seq.activations[start:end]
            sentence_sum = sum(sentence_acts)
            sentence_text = "".join(sentence_tokens)
            
            # Determine if high (using threshold from collector config)
            # We'll pass threshold separately
            sentences.append((sentence_text, sentence_sum, None))  # is_high determined later
        
        return sentences
    
    async def evaluate_feature(
        self,
        feature_idx: int,
        explanation: str,
        bin_A_sum_sequences: List[MarkedSequence],
        bin_B_sum_sequences: List[MarkedSequence],
        bin_C_sum_sequences: List[MarkedSequence],
        explanation_sequence_ids: Optional[Set[str]] = None
    ) -> Dict:
        """
        Evaluate feature using 3-way ranking task (V2) - sentence-based.
        
        For each question:
        - Sample 1 from BIN A (HIGH-high, sorted by sentence_sum)
        - Sample 1 from BIN B (HIGH-low, sorted by sentence_sum)
        - Sample 1 from BIN C (LOW, sorted by sentence_sum)
        - Ask LLM to rank by activation strength (1=highest, 2=middle, 3=lowest)
        - True ranking: Example 1 (BIN A) = 1, Example 2 (BIN B) = 2, Example 3 (BIN C) = 3
        
        Args:
            feature_idx: Feature index
            explanation: Generated explanation for this feature
            bin_A_sum_sequences: HIGH-high sequences by sentence_sum
            bin_B_sum_sequences: HIGH-low sequences by sentence_sum
            bin_C_sum_sequences: LOW sequences by sentence_sum
            explanation_sequence_ids: Set of sequence IDs used in explanation (to exclude for fairness)
        """
        import random
        random.seed(42)
        
        # Exclude explanation sequences from all bins
        if explanation_sequence_ids:
            explanation_sequence_ids_set = set(explanation_sequence_ids)
            bin_A_sum_sequences = [seq for seq in bin_A_sum_sequences if seq.sequence_id not in explanation_sequence_ids_set]
            bin_B_sum_sequences = [seq for seq in bin_B_sum_sequences if seq.sequence_id not in explanation_sequence_ids_set]
            bin_C_sum_sequences = [seq for seq in bin_C_sum_sequences if seq.sequence_id not in explanation_sequence_ids_set]
        
        # Need at least 1 sequence from each bin per question
        n_questions = min(self.max_questions, len(bin_A_sum_sequences), len(bin_B_sum_sequences), len(bin_C_sum_sequences))
        
        if n_questions == 0:
            print(f"    ⚠️  Feature {feature_idx}: Not enough sequences in bins (A: {len(bin_A_sum_sequences)}, B: {len(bin_B_sum_sequences)}, C: {len(bin_C_sum_sequences)})")
            return {
                "feature_idx": feature_idx,
                "perfect_ranking_acc": 0.0,
                "mean_rank_correlation": 0.0,
                "mean_pairwise_acc": 0.0,
                "mean_rank_distance": 0.0,
                "n_questions": 0,
                "error": "Not enough sequences in bins",
                "error_detail": f"BIN A: {len(bin_A_sum_sequences)}, BIN B: {len(bin_B_sum_sequences)}, BIN C: {len(bin_C_sum_sequences)}",
                "questions": []
            }
        
        print(f"    Feature {feature_idx}: Creating {n_questions} sentence-based 3-way ranking questions...")
        
        questions = []
        question_details = []
        true_rankings = []  # List of true ranking dicts
        
        for q_idx in range(n_questions):
            # Sample 1 from each bin
            seq_A = bin_A_sum_sequences[q_idx % len(bin_A_sum_sequences)] if len(bin_A_sum_sequences) > 0 else None
            seq_B = bin_B_sum_sequences[q_idx % len(bin_B_sum_sequences)] if len(bin_B_sum_sequences) > 0 else None
            seq_C = bin_C_sum_sequences[q_idx % len(bin_C_sum_sequences)] if len(bin_C_sum_sequences) > 0 else None
            
            if seq_A is None or seq_B is None or seq_C is None:
                continue
            
            if len(seq_A.tokens) == 0 or len(seq_B.tokens) == 0 or len(seq_C.tokens) == 0:
                continue
            
            # Get sentence text and sum for each sequence
            sentence_A = "".join(seq_A.tokens)
            sentence_B = "".join(seq_B.tokens)
            sentence_C = "".join(seq_C.tokens)
            
            sum_A = seq_A.sentence_sums[0] if seq_A.sentence_sums else sum(seq_A.activations)
            sum_B = seq_B.sentence_sums[0] if seq_B.sentence_sums else sum(seq_B.activations)
            sum_C = seq_C.sentence_sums[0] if seq_C.sentence_sums else sum(seq_C.activations)
            
            # True ranking: A=1 (highest), B=2 (middle), C=3 (lowest)
            true_ranking = {
                "Example 1": 1,  # BIN A (HIGH-high)
                "Example 2": 2,  # BIN B (HIGH-low)
                "Example 3": 3   # BIN C (LOW)
            }
            true_rankings.append(true_ranking)
            
            # Build ranking question
            question_parts = [
                f"Given this explanation:",
                f"\"{explanation}\"",
                "",
                f"Rank these 3 examples by activation strength (1=highest, 2=middle, 3=lowest):",
                "",
                f"Example 1: \"{sentence_A}\"",
                "",
                f"Example 2: \"{sentence_B}\"",
                "",
                f"Example 3: \"{sentence_C}\"",
                "",
                f"Output only: Example 1=?, Example 2=?, Example 3=?"
            ]
            
            question_text = "\n".join(question_parts)
            questions.append(question_text)
            
            question_details.append({
                "question": question_text,
                "explanation": explanation,
                "examples": [
                    {
                        "label": "Example 1",
                        "bin": "A",
                        "sentence_text": sentence_A,
                        "sum": float(sum_A),
                        "true_rank": 1
                    },
                    {
                        "label": "Example 2",
                        "bin": "B",
                        "sentence_text": sentence_B,
                        "sum": float(sum_B),
                        "true_rank": 2
                    },
                    {
                        "label": "Example 3",
                        "bin": "C",
                        "sentence_text": sentence_C,
                        "sum": float(sum_C),
                        "true_rank": 3
                    }
                ],
                "true_ranking": true_ranking
            })
        
        if len(questions) == 0:
            print(f"    Feature {feature_idx}: ⚠️  No valid sentence-based ranking questions")
            return {
                "feature_idx": feature_idx,
                "perfect_ranking_acc": 0.0,
                "mean_rank_correlation": 0.0,
                "mean_pairwise_acc": 0.0,
                "mean_rank_distance": 0.0,
                "n_questions": 0,
                "error": "Too few sequences",
                "error_detail": f"BIN A: {len(bin_A_sum_sequences)}, BIN B: {len(bin_B_sum_sequences)}, BIN C: {len(bin_C_sum_sequences)}",
                "questions": []
            }
        
        # Evaluate questions
        print(f"    Feature {feature_idx}: Evaluating {len(questions)} sentence-based ranking questions...")
        responses = []
        
        if self.provider == "offline":
            # Use batch processing for offline mode (faster)
            batch_size = 4  # Optimal batch size for 20 questions (5 batches)
            responses = await ask_model_questions_batch(
                prompts=questions,
                model_name=self.model_name,
                max_tokens=20,  # Need more tokens for ranking format
                provider=self.provider,
                offline_model=self.offline_model,
                offline_tokenizer=self.offline_tokenizer,
                device=self.device,
                batch_size=batch_size,
            )
        else:
            # Sequential processing for OpenAI API
            for i, question in enumerate(tqdm(questions, desc=f"      Feature {feature_idx} sentence ranking", leave=False)):
                cache_key = f"sae_eval_sentence_feat_{feature_idx}" if self.provider == "openai" and len(question) > 1024 else None
                response = await ask_model_question(
                    self.client, 
                    self.model_name, 
                    question,
                    max_tokens=20,  # Need more tokens for ranking format
                    provider=self.provider,
                    offline_model=self.offline_model,
                    offline_tokenizer=self.offline_tokenizer,
                    device=self.device,
                    prompt_cache_key=cache_key,
                )
                responses.append(response)
                if self.provider == "openai":
                    await asyncio.sleep(0.5)  # Rate limiting for API
        
        # Parse ranking responses and compute metrics
        all_metrics = []
        for i, (response, true_ranking) in enumerate(zip(responses, true_rankings)):
            predicted_ranking = parse_ranking_response(response)
            
            if predicted_ranking is None:
                # Failed to parse - use worst case ranking
                predicted_ranking = {"Example 1": 3, "Example 2": 3, "Example 3": 3}
            
            # Compute metrics for this question
            metrics = compute_ranking_metrics(predicted_ranking, true_ranking)
            all_metrics.append(metrics)
            
            # Add to question details
            question_details[i]["predicted_ranking"] = predicted_ranking
            question_details[i]["model_response"] = response
            question_details[i]["metrics"] = metrics
        
        # Aggregate metrics across all questions
        perfect_ranking_acc = np.mean([m["perfect_ranking"] for m in all_metrics])
        mean_rank_correlation = np.mean([m["rank_correlation"] for m in all_metrics])
        mean_pairwise_acc = np.mean([
            (m["pairwise_accuracy"]["A_vs_B"] + m["pairwise_accuracy"]["B_vs_C"] + m["pairwise_accuracy"]["A_vs_C"]) / 3.0
            for m in all_metrics
        ])
        mean_rank_distance = np.mean([m["rank_distance"] for m in all_metrics])
        
        print(f"    Feature {feature_idx}: ✓ Sentence-based ranking - Perfect: {perfect_ranking_acc:.3f}, Corr: {mean_rank_correlation:.3f}, Pairwise: {mean_pairwise_acc:.3f}")
        
        return {
            "feature_idx": feature_idx,
            "perfect_ranking_acc": float(perfect_ranking_acc),
            "mean_rank_correlation": float(mean_rank_correlation),
            "mean_pairwise_acc": float(mean_pairwise_acc),
            "mean_rank_distance": float(mean_rank_distance),
            "n_questions": len(questions),
            "error": None,
            "questions": question_details
        }

