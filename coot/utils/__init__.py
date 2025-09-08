"""
Utility functions for COOT
"""

from .visualization import plot_attention_timeline, plot_intervention_analysis
from .metrics import compute_safety_metrics, compute_alignment_metrics
from .config import COOTConfig, load_config, save_config, get_preset_config
from .token_utils import setup_tokenizer_and_model, create_attention_mask, prepare_model_inputs, safe_model_forward, debug_tokenizer_info

__all__ = [
    "plot_attention_timeline",
    "plot_intervention_analysis", 
    "compute_safety_metrics",
    "compute_alignment_metrics",
    "COOTConfig",
    "load_config",
    "save_config",
    "get_preset_config",
    "setup_tokenizer_and_model",
    "create_attention_mask", 
    "prepare_model_inputs",
    "safe_model_forward",
    "debug_tokenizer_info"
]
