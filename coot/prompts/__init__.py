"""
CooT Prompt Templates - Four core prompts from the methodology
"""

from .perceiver_prompts import (
    STATE_COGNITION_PROMPT,
    UNIVERSAL_SOCIAL_SCHEMA_PROMPT,
    CONTEXT_DEPENDENT_INTERVENTION_PROMPT,
    PERCEIVER_PROMPT_TEMPLATE
)
from .generator_prompts import (
    COGNITIVE_DECODING_PROMPT,
    GENERATOR_PROMPT_TEMPLATE,
    build_ability_conditioning_prompt
)

__all__ = [
    # Four core prompts from methodology
    "STATE_COGNITION_PROMPT",
    "UNIVERSAL_SOCIAL_SCHEMA_PROMPT", 
    "CONTEXT_DEPENDENT_INTERVENTION_PROMPT",
    "COGNITIVE_DECODING_PROMPT",
    # Legacy compatibility
    "PERCEIVER_PROMPT_TEMPLATE",
    "GENERATOR_PROMPT_TEMPLATE",
    "build_ability_conditioning_prompt"
]
