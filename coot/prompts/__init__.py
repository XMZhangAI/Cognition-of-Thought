"""
Prompt templates for COOT components
"""

from .perceiver_prompts import PERCEIVER_PROMPT_TEMPLATE, create_perceiver_prompt
from .generator_prompts import GENERATOR_PROMPT_TEMPLATE, create_generator_prompt

__all__ = [
    "PERCEIVER_PROMPT_TEMPLATE",
    "GENERATOR_PROMPT_TEMPLATE", 
    "create_perceiver_prompt",
    "create_generator_prompt"
]
