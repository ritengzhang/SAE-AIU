"""
Stage 2: Explanation Generation

Generates explanations using high-activating sequences with optional low-firing contrast.
"""

import asyncio
import os
import logging
import torch
from typing import List, Dict, Optional
from openai import AsyncOpenAI
from dotenv import load_dotenv
from tqdm import tqdm

from .collection_stage import MarkedSequence, ActivationCollector

load_dotenv()

logger = logging.getLogger(__name__)


class ExplanationGenerator:
    """
    Generates natural language explanations for features.
    
    Uses high-activating sequences with optional low-firing contrast examples.
    """
    
    def __init__(
        self,
        model_name: str = "gpt-4.1-nano",
        n_examples: int = 20,
        include_low_contrast: bool = True,
        provider: str = "openai",  # "openai" or "offline"
        device: str = None,
    ):
        self.model_name = model_name
        self.n_examples = n_examples
        self.include_low_contrast = include_low_contrast
        self.provider = provider
        self.device = device or "cuda:0" if torch.cuda.is_available() else "cpu"
        
        if provider == "openai":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY not found in environment")
            self.client = AsyncOpenAI(api_key=api_key)
            self.offline_model = None
            self.offline_tokenizer = None
        elif provider == "offline":
            # Load model directly with transformers (no vLLM)
            from transformers import AutoModelForCausalLM, AutoTokenizer
            
            print(f"Loading offline model: {model_name}")
            self.offline_tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Use specific device instead of "auto" to prevent splitting across GPUs
            # For Mistral and similar models, we'll load on the specified device only
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
                print(f"Warning: device_map='auto' failed, trying manual device placement: {e}")
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
    
    def format_activation_record(
        self,
        marked_seq: MarkedSequence,
        label: str = ""
    ) -> str:
        """
        Format a marked sequence for the explanation prompt (FAIR FORMAT).
        
        Shows FULL sentence + TOP activation token only (no activation values).
        This ensures fairness: all features get the same information.
        
        Args:
            marked_seq: MarkedSequence with tokens and activations
            label: Not used (kept for compatibility, no label shown to save tokens)
        
        Returns:
            Formatted string showing full sentence + top activation token
        """
        lines = []
        
        # Show full sentence (all tokens concatenated)
        full_sentence = "".join(marked_seq.tokens)
        lines.append(f"Sentence: {full_sentence}")
        
        # Find top activation token (highest activation in sequence)
        token_act_pairs = list(zip(marked_seq.tokens, marked_seq.activations))
        token_act_pairs_sorted = sorted(token_act_pairs, key=lambda x: x[1], reverse=True)
        top_token, top_activation = token_act_pairs_sorted[0]
        
        # Show only the top activation token (no activation value shown)
        lines.append(f"Top activation token: \"{top_token}\"")
        
        return "\n".join(lines)
    
    async def generate_explanation(
        self,
        feature_idx: int,
        high_sequences: List[MarkedSequence],
        low_sequences: Optional[List[MarkedSequence]] = None
    ) -> Dict:
        """
        Generate explanation for a feature.
        
        Args:
            feature_idx: Feature index
            high_sequences: List of high-activating sequences
            low_sequences: Optional list of low-activating sequences (for contrast)
        
        Returns:
            Dict with explanation and metadata
        """
        # Use top sequences directly (already sorted by max_activation descending)
        # These are the top 20 sequences from binning
        import random
        random.seed(42)  # Reproducibility
        n_sample = min(self.n_examples, len(high_sequences))
        sampled_high = random.sample(high_sequences, n_sample)
        
        # Build prompt - OPTIMIZED FOR CACHING:
        # Static instructions first (cacheable), then dynamic examples
        
        # Static instruction part (same for all features - cacheable)
        static_instructions = (
            "You are analyzing a feature in a sparse autoencoder (SAE) that detects patterns in text.\n"
            "\n"
            "Below are examples of text sequences where this feature has TOP FIRING (highest activation).\n"
            "For each example, you will see:\n"
            "- The full sentence (all tokens concatenated)\n"
            "- The top activation token (the single token with highest activation in that sequence)\n"
            "\n"
            "Based on these examples, provide a clear, concise explanation of what this feature detects.\n"
            "Focus on the patterns that cause high activation.\n"
            "\n"
            "IMPORTANT CONSTRAINTS:\n"
            "- Explain in ONE complete sentence, maximum 15 words.\n"
            "- Do NOT mention activation frequency, how often this feature fires, or how common/rare it is.\n"
            "- Focus on concrete content: what lexical, syntactic, or semantic patterns cause activation.\n"
            "- Keep the explanation minimal and to the point.\n"
            "\n"
        )
        
        # Dynamic part (feature-specific examples)
        dynamic_parts = [
            f"The feature index is {feature_idx}.\n",
            ""
        ]
        
        # Add high-activating examples (no [HIGH] label - saves tokens, already clear from instructions)
        for i, seq in enumerate(sampled_high, 1):
            dynamic_parts.append(f"Example {i}:")
            dynamic_parts.append(self.format_activation_record(seq, ""))
            dynamic_parts.append("")
        
        dynamic_parts.append("Explanation:")
        
        dynamic_content = "\n".join(dynamic_parts)
        
        # Combine: static (cacheable) + dynamic
        prompt = static_instructions + dynamic_content
        
        try:
            if self.provider == "openai":
                # Add small delay to avoid rate limits
                await asyncio.sleep(0.1)
                
                # Use prompt_cache_key to enable caching for the static instruction part
                # Same key for all explanation requests (they share the static prefix)
                # Caching automatically activates for prompts > 1024 tokens
                create_kwargs = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "system", "content": "Explain in ONE complete sentence, maximum 15 words."},
                        {"role": "user", "content": prompt}
                    ],
                    "max_tokens": 40,
                    "temperature": 0.0,
                }
                # Add prompt_cache_key if prompt is long enough (caching works for > 1024 tokens)
                # Using same key for all explanations since they share static prefix
                if len(prompt) > 1024:
                    create_kwargs["prompt_cache_key"] = "sae_explanation_static"
                
                response = await self.client.chat.completions.create(**create_kwargs)
                explanation = response.choices[0].message.content.strip()
            else:  # offline
                # Format prompt for chat model
                messages = [
                    {"role": "system", "content": "Explain in ONE complete sentence, maximum 15 words."},
                    {"role": "user", "content": prompt}
                ]
                
                # Tokenize
                formatted_prompt = self.offline_tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                inputs = self.offline_tokenizer(formatted_prompt, return_tensors="pt")
                # Move inputs to the same device as the model
                if self.use_device_map:
                    inputs = {k: v.to(self.model_device) for k, v in inputs.items()}
                else:
                    inputs = {k: v.to(self.device) for k, v in inputs.items()}
                
                # Generate
                with torch.no_grad():
                    outputs = self.offline_model.generate(
                        **inputs,
                        max_new_tokens=40,
                        temperature=0.0,
                        do_sample=False,
                        pad_token_id=self.offline_tokenizer.eos_token_id,
                    )
                
                # Decode (only the new tokens)
                generated_text = self.offline_tokenizer.decode(
                    outputs[0][inputs['input_ids'].shape[1]:], 
                    skip_special_tokens=True
                )
                explanation = generated_text.strip()
            
            # Track which sequences were used for explanation (to exclude from evaluation)
            explanation_sequence_ids = {seq.sequence_id for seq in sampled_high}
            
            return {
                "explanation": explanation,
                "explanation_sequence_ids": list(explanation_sequence_ids),
                "error": None
            }
            
        except Exception as e:
            error_str = str(e)
            
            # Check for quota errors - these should fail fast, not retry
            if "429" in error_str or "insufficient_quota" in error_str.lower() or "quota" in error_str.lower():
                logger.error(f"QUOTA ERROR for feature {feature_idx}: {error_str}")
                logger.error("API quota exceeded. Please check your OpenAI billing and quota limits.")
                return {
                    "explanation": None,
                    "explanation_sequence_ids": [],
                    "error": f"QUOTA_EXCEEDED: {error_str}"
                }
            
            logger.error(f"Error generating explanation for feature {feature_idx}: {e}")
            return {
                "explanation": None,
                "explanation_sequence_ids": [],
                "error": str(e)
            }
    
    async def generate_explanations_batch(
        self,
        collector: ActivationCollector,
        feature_indices: List[int]
    ) -> tuple[Dict[int, Dict], Dict[int, bool]]:
        """
        Generate explanations for multiple features in batch.
        
        Args:
            collector: ActivationCollector with collected sequences
            feature_indices: List of feature indices to generate explanations for
        
        Returns:
            Tuple of (explanations_dict, missing_low_contrast_dict)
            - explanations_dict: Dict mapping feature_idx to explanation result
            - missing_low_contrast_dict: Dict mapping feature_idx to bool (whether low contrast was missing)
        """
        explanations = {}
        missing_low_contrast = {}
        
        for feat_idx in tqdm(feature_indices, desc="Generating explanations"):
            # Get high sequences for explanation
            high_sequences = collector.high_explanation_sequences.get(feat_idx, [])
            
            if not high_sequences:
                explanations[feat_idx] = {
                    "explanation": None,
                    "explanation_sequence_ids": [],
                    "error": "No high sequences available"
                }
                missing_low_contrast[feat_idx] = True
                continue
            
            # Get low sequences if include_low_contrast is True
            low_sequences = None
            if self.include_low_contrast:
                low_sequences = collector.low_max_sequences.get(feat_idx, [])
                if not low_sequences:
                    missing_low_contrast[feat_idx] = True
                else:
                    missing_low_contrast[feat_idx] = False
            else:
                missing_low_contrast[feat_idx] = False
            
            # Generate explanation
            result = await self.generate_explanation(
                feature_idx=feat_idx,
                high_sequences=high_sequences,
                low_sequences=low_sequences
            )
            explanations[feat_idx] = result
            
            # Check for quota error and stop early
            if result.get('error') and 'QUOTA_EXCEEDED' in result.get('error', ''):
                logger.error(f"\n{'='*70}")
                logger.error("QUOTA EXCEEDED - Stopping explanation generation")
                logger.error("Please check your OpenAI API quota and billing.")
                logger.error(f"{'='*70}\n")
                break
        
        return explanations, missing_low_contrast
    
    def cleanup(self):
        """Clean up offline model and tokenizer to free memory."""
        if self.provider == "offline" and self.offline_model is not None:
            del self.offline_model
            del self.offline_tokenizer
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            self.offline_model = None
            self.offline_tokenizer = None