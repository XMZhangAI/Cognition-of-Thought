"""
Token handling utilities for COOT
"""

import torch
from typing import Tuple
from transformers import PreTrainedModel, PreTrainedTokenizer


def setup_tokenizer_and_model(model: PreTrainedModel, 
                             tokenizer: PreTrainedTokenizer) -> Tuple[PreTrainedModel, PreTrainedTokenizer]:
    """
    Properly setup tokenizer and model to avoid attention mask warnings
    
    Args:
        model: Pre-trained model
        tokenizer: Pre-trained tokenizer
        
    Returns:
        Tuple of (model, tokenizer) with proper token configuration
    """
    # Setup pad token to avoid conflicts with eos token
    if tokenizer.pad_token is None:
        if tokenizer.unk_token is not None:
            tokenizer.pad_token = tokenizer.unk_token
        else:
            # Add a new pad token if no unk token exists
            tokenizer.add_special_tokens({'pad_token': '<pad>'})
            model.resize_token_embeddings(len(tokenizer))
            print(f"Added new pad token, vocab size: {len(tokenizer)}")
    
    return model, tokenizer


def create_attention_mask(input_ids: torch.Tensor, 
                         pad_token_id: int) -> torch.Tensor:
    """
    Create attention mask for input tokens
    
    Args:
        input_ids: Input token IDs tensor
        pad_token_id: ID of the padding token
        
    Returns:
        Attention mask tensor (1 for real tokens, 0 for padding)
    """
    return (input_ids != pad_token_id).long()


def prepare_model_inputs(input_ids: torch.Tensor,
                        tokenizer: PreTrainedTokenizer,
                        device: torch.device) -> dict:
    """
    Prepare model inputs with proper attention mask
    
    Args:
        input_ids: Input token IDs
        tokenizer: Tokenizer with pad token
        device: Target device
        
    Returns:
        Dictionary with input_ids and attention_mask
    """
    input_ids = input_ids.to(device)
    
    # Create attention mask
    if tokenizer.pad_token_id is not None:
        attention_mask = create_attention_mask(input_ids, tokenizer.pad_token_id)
    else:
        # Fallback: all tokens are real tokens
        attention_mask = torch.ones_like(input_ids, dtype=torch.long)
    
    attention_mask = attention_mask.to(device)
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask
    }


def safe_model_forward(model: PreTrainedModel,
                      input_ids: torch.Tensor,
                      tokenizer: PreTrainedTokenizer,
                      device: torch.device,
                      **kwargs) -> dict:
    """
    Safe forward pass through model with proper attention mask
    
    Args:
        model: Pre-trained model
        input_ids: Input token IDs
        tokenizer: Tokenizer
        device: Device
        **kwargs: Additional model arguments
        
    Returns:
        Model outputs
    """
    model_inputs = prepare_model_inputs(input_ids, tokenizer, device)
    model_inputs.update(kwargs)
    
    return model(**model_inputs)


def debug_tokenizer_info(tokenizer: PreTrainedTokenizer) -> dict:
    """
    Get debug information about tokenizer configuration
    
    Args:
        tokenizer: Tokenizer to analyze
        
    Returns:
        Dictionary with tokenizer information
    """
    return {
        'vocab_size': len(tokenizer),
        'pad_token': tokenizer.pad_token,
        'pad_token_id': tokenizer.pad_token_id,
        'eos_token': tokenizer.eos_token,
        'eos_token_id': tokenizer.eos_token_id,
        'unk_token': tokenizer.unk_token,
        'unk_token_id': tokenizer.unk_token_id,
        'bos_token': tokenizer.bos_token,
        'bos_token_id': tokenizer.bos_token_id,
        'pad_equals_eos': tokenizer.pad_token_id == tokenizer.eos_token_id
    }
