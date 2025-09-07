"""
Generator module for COOT - handles text generation with logit bias
"""

import torch
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
from transformers import PreTrainedModel, PreTrainedTokenizer

from .guidance import GuidanceConcepts


class Generator:
    """
    Text generator with support for structured logit bias and KV cache management
    """
    
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 device: Optional[torch.device] = None):
        """
        Args:
            model: Base language model for generation
            tokenizer: Tokenizer for the model
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or (next(model.parameters()).device if model else torch.device('cpu'))
        
        # Generation state
        self.kv_cache = None
        self.current_sequence = None
        self.generation_step = 0
        
        # Bias tracking
        self.current_bias = None
        self.bias_remaining_steps = 0
        
    def initialize_generation(self, 
                            input_ids: torch.Tensor,
                            generator_prompt: Optional[str] = None) -> torch.Tensor:
        """
        Initialize generation with input sequence
        
        Args:
            input_ids: Initial input token IDs [seq_len] or [batch_size, seq_len]
            generator_prompt: Optional generator-specific prompt
            
        Returns:
            Prepared input IDs with generator prompt if provided
        """
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(0)  # Add batch dimension
            
        # Add generator prompt if provided
        if generator_prompt:
            prompt_ids = self.tokenizer.encode(
                generator_prompt,
                return_tensors='pt',
                add_special_tokens=True
            ).to(self.device)
            input_ids = torch.cat([prompt_ids, input_ids], dim=-1)
        
        self.current_sequence = input_ids
        self.generation_step = 0
        self.kv_cache = None
        self.current_bias = None
        self.bias_remaining_steps = 0
        
        return input_ids
    
    def generate_next_token(self,
                          temperature: float = 1.0,
                          top_k: Optional[int] = None,
                          top_p: Optional[float] = None,
                          repetition_penalty: float = 1.0,
                          logit_bias: Optional[torch.Tensor] = None) -> Tuple[int, torch.Tensor, Dict[str, Any]]:
        """
        Generate next token with optional logit bias
        
        Args:
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            repetition_penalty: Repetition penalty factor
            logit_bias: Optional bias to add to logits [vocab_size]
            
        Returns:
            Tuple of (next_token_id, attention_weights, generation_info)
        """
        if self.current_sequence is None:
            raise ValueError("Generation not initialized. Call initialize_generation first.")
        
        # Forward pass through model
        with torch.no_grad():
            outputs = self.model(
                input_ids=self.current_sequence,
                past_key_values=self.kv_cache,
                use_cache=True,
                output_attentions=True,
                output_hidden_states=False
            )
            
            # Extract logits and attention
            logits = outputs.logits[0, -1, :]  # Last token logits [vocab_size]
            attention_weights = outputs.attentions  # Attention from all layers
            self.kv_cache = outputs.past_key_values
        
        # Apply repetition penalty
        if repetition_penalty != 1.0:
            logits = self._apply_repetition_penalty(logits, repetition_penalty)
        
        # Apply logit bias if provided
        if logit_bias is not None:
            logits = logits + logit_bias.to(logits.device)
        
        # Apply current bias if active
        if self.current_bias is not None and self.bias_remaining_steps > 0:
            logits = logits + self.current_bias.to(logits.device)
            self.bias_remaining_steps -= 1
        
        # Sample next token
        next_token_id = self._sample_token(logits, temperature, top_k, top_p)
        
        # Update sequence
        next_token_tensor = torch.tensor([[next_token_id]], device=self.device)
        self.current_sequence = torch.cat([self.current_sequence, next_token_tensor], dim=-1)
        self.generation_step += 1
        
        # Prepare generation info
        generation_info = {
            'step': self.generation_step,
            'logits': logits,
            'token_id': next_token_id,
            'token_text': self.tokenizer.decode([next_token_id]),
            'sequence_length': self.current_sequence.size(-1),
            'bias_active': self.current_bias is not None and self.bias_remaining_steps > 0
        }
        
        return next_token_id, attention_weights, generation_info
    
    def _apply_repetition_penalty(self, 
                                 logits: torch.Tensor, 
                                 penalty: float) -> torch.Tensor:
        """Apply repetition penalty to logits"""
        if self.current_sequence is None or penalty == 1.0:
            return logits
            
        # Get unique tokens in current sequence
        unique_tokens = torch.unique(self.current_sequence)
        
        # Apply penalty
        for token_id in unique_tokens:
            if 0 <= token_id < logits.size(0):
                if logits[token_id] > 0:
                    logits[token_id] /= penalty
                else:
                    logits[token_id] *= penalty
                    
        return logits
    
    def _sample_token(self,
                     logits: torch.Tensor,
                     temperature: float,
                     top_k: Optional[int],
                     top_p: Optional[float]) -> int:
        """Sample token from logits with temperature and filtering"""
        
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature
        
        # Apply top-k filtering
        if top_k is not None and top_k > 0:
            top_k = min(top_k, logits.size(-1))  # Safety check
            top_k_logits, top_k_indices = torch.topk(logits, top_k)
            logits_filtered = torch.full_like(logits, float('-inf'))
            logits_filtered.scatter_(-1, top_k_indices, top_k_logits)
            logits = logits_filtered
        
        # Apply top-p (nucleus) filtering
        if top_p is not None and 0.0 < top_p < 1.0:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            # Remove tokens with cumulative probability above the threshold
            sorted_indices_to_remove = cumulative_probs > top_p
            # Keep at least one token
            sorted_indices_to_remove[0] = False
            
            # Scatter back to original indexing
            indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
            logits[indices_to_remove] = float('-inf')
        
        # Sample from the filtered distribution
        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, num_samples=1).item()
        
        return next_token_id
    
    def apply_guidance_bias(self,
                          guidance: GuidanceConcepts,
                          num_steps: int,
                          lambda_positive: float = 2.0,
                          lambda_negative: float = -3.0):
        """
        Apply guidance as logit bias for specified number of steps
        
        Args:
            guidance: Guidance concepts to apply
            num_steps: Number of generation steps to apply bias
            lambda_positive: Positive bias strength
            lambda_negative: Negative bias strength
        """
        # Create bias tensor
        vocab_size = self.model.config.vocab_size
        bias = torch.zeros(vocab_size, device=self.device)
        
        if self.tokenizer:
            # Apply positive bias
            for concept in guidance.positive_concepts:
                concept_tokens = self.tokenizer.encode(concept, add_special_tokens=False)
                for token_id in concept_tokens:
                    if 0 <= token_id < vocab_size:
                        bias[token_id] += lambda_positive * guidance.priority
            
            # Apply negative bias
            for concept in guidance.negative_concepts:
                concept_tokens = self.tokenizer.encode(concept, add_special_tokens=False)
                for token_id in concept_tokens:
                    if 0 <= token_id < vocab_size:
                        bias[token_id] += lambda_negative * guidance.priority
        
        self.current_bias = bias
        self.bias_remaining_steps = num_steps
    
    def rollback_to_step(self, target_step: int, 
                        target_tokens: torch.Tensor,
                        target_kv_cache: Any):
        """
        Rollback generation to a previous step
        
        Args:
            target_step: Step to rollback to
            target_tokens: Token sequence at target step
            target_kv_cache: KV cache at target step
        """
        if target_step > self.generation_step:
            raise ValueError(f"Cannot rollback to future step {target_step}")
        
        self.generation_step = target_step
        self.current_sequence = target_tokens.clone()
        self.kv_cache = target_kv_cache
        
        # Clear any active bias (will be reapplied by intervention mechanism)
        self.current_bias = None
        self.bias_remaining_steps = 0
    
    def generate_sequence(self,
                         max_length: int,
                         temperature: float = 1.0,
                         top_k: Optional[int] = None,
                         top_p: Optional[float] = None,
                         repetition_penalty: float = 1.0,
                         eos_token_id: Optional[int] = None,
                         callback: Optional[callable] = None) -> Tuple[torch.Tensor, List[Dict]]:
        """
        Generate a complete sequence
        
        Args:
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_k: Top-k parameter
            top_p: Top-p parameter
            repetition_penalty: Repetition penalty
            eos_token_id: End-of-sequence token ID
            callback: Optional callback function called after each token
            
        Returns:
            Tuple of (generated_sequence, generation_info_list)
        """
        if self.current_sequence is None:
            raise ValueError("Generation not initialized")
        
        generation_info_list = []
        eos_token_id = eos_token_id or self.tokenizer.eos_token_id
        
        while (self.current_sequence.size(-1) < max_length and 
               (eos_token_id is None or self.current_sequence[0, -1].item() != eos_token_id)):
            
            # Generate next token
            token_id, attention, info = self.generate_next_token(
                temperature=temperature,
                top_k=top_k,
                top_p=top_p,
                repetition_penalty=repetition_penalty
            )
            
            generation_info_list.append(info)
            
            # Call callback if provided
            if callback:
                should_continue = callback(self.current_sequence, info, attention)
                if not should_continue:
                    break
        
        return self.current_sequence, generation_info_list
    
    def get_current_sequence(self) -> Optional[torch.Tensor]:
        """Get current generation sequence"""
        return self.current_sequence
    
    def get_current_text(self) -> Optional[str]:
        """Get current generation as text"""
        if self.current_sequence is None:
            return None
        return self.tokenizer.decode(self.current_sequence[0], skip_special_tokens=True)
    
    def get_generation_state(self) -> Dict:
        """Get current generation state for snapshotting"""
        return {
            'step': self.generation_step,
            'sequence': self.current_sequence.clone() if self.current_sequence is not None else None,
            'kv_cache': self.kv_cache,
            'bias_remaining_steps': self.bias_remaining_steps,
            'has_bias': self.current_bias is not None
        }
    
    def restore_generation_state(self, state: Dict):
        """Restore generation state from snapshot"""
        self.generation_step = state['step']
        self.current_sequence = state['sequence']
        self.kv_cache = state['kv_cache']
        self.bias_remaining_steps = state['bias_remaining_steps']
        # Note: current_bias is not restored as it will be reapplied
    
    def reset(self):
        """Reset generator state"""
        self.kv_cache = None
        self.current_sequence = None
        self.generation_step = 0
        self.current_bias = None
        self.bias_remaining_steps = 0
    
    def get_statistics(self) -> Dict:
        """Get generation statistics"""
        return {
            'current_step': self.generation_step,
            'sequence_length': self.current_sequence.size(-1) if self.current_sequence is not None else 0,
            'bias_active': self.current_bias is not None and self.bias_remaining_steps > 0,
            'bias_remaining_steps': self.bias_remaining_steps,
            'cache_initialized': self.kv_cache is not None
        }
