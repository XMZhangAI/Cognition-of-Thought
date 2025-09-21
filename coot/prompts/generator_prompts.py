"""
Generator prompt templates for COOT

Includes the Cognitive Decoding Prompt.
"""

# =============================================================================
# COGNITIVE DECODING PROMPT (Generator Stage)
# =============================================================================

COGNITIVE_DECODING_PROMPT = """Input Parameters: Contextual guidance g_context, selected social skill 𝒮_selected, rollback context x_{1:t*}.

Generation Constraints:
1. Safety compliance
2. Helpfulness preservation
3. Coherence maintenance
4. Skill integration

Output Requirements: Generate natural language response that implicitly reflects guidance and selected social skill without meta-commentary or explicit reference to intervention mechanisms.

Generation context:"""

# =============================================================================
# LEGACY COMPATIBILITY - Basic generator prompts
# =============================================================================

GENERATOR_PROMPT_TEMPLATE = """You are a helpful, harmless, and honest AI assistant."""

def build_ability_conditioning_prompt(ability: str, definition: str) -> str:
    """
    Build a simple ability-conditioned prompt for backward compatibility
    """
    return f"You are a helpful assistant. Apply {ability}: {definition}"
