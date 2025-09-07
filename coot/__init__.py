"""
COOT: Cognition-of-Thought Alignment for Reliable Reasoning in LLMs
"""

from .core.decoder import COOTDecoder
from .core.perceiver import Perceiver
from .core.generator import Generator
from .core.state_system import AsimovStateSystem
from .core.intervention import InterventionMechanism

__version__ = "0.1.0"
__author__ = "COOT Team"

__all__ = [
    "COOTDecoder",
    "Perceiver",
    "Generator", 
    "AsimovStateSystem",
    "InterventionMechanism",
]
