"""
Multi-level Residual Injection for COOT
Implements three levels of residual injection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, List, Tuple, Any
from enum import Enum
from dataclasses import dataclass

from .guidance import GuidanceConcepts


class InjectionLevel(Enum):
    """Types of residual injection levels"""
    REPRESENTATION = "representation"  # Hidden state level
    ATTENTION = "attention"            # Key/Value level  
    TRAJECTORY = "trajectory"          # Token sequence level


@dataclass
class ResidualConfig:
    """Configuration for residual injection"""
    injection_level: InjectionLevel
    inject_layers: List[int]  # Which layers to inject into
    beta_strength: float = 0.1  # Injection strength
    max_injection_steps: int = 20
    
    # Representation-level specific
    hidden_dim: Optional[int] = None
    
    # Attention-level specific  
    num_heads: Optional[int] = None
    head_dim: Optional[int] = None
    
    # Trajectory-level specific
    context_window: int = 10


class RiskRationaleEncoder(nn.Module):
    """
    Encodes risk rationale into residual vectors for injection
    """
    
    def __init__(self, 
                 vocab_size: int,
                 hidden_dim: int,
                 injection_level: InjectionLevel,
                 device: torch.device):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim
        self.injection_level = injection_level
        self.device = device
        
        # Concept embedding layer
        self.concept_embeddings = nn.Embedding(vocab_size, hidden_dim)
        
        # Level-specific encoders
        if injection_level == InjectionLevel.REPRESENTATION:
            self.residual_projector = nn.Linear(hidden_dim, hidden_dim)
        elif injection_level == InjectionLevel.ATTENTION:
            self.key_projector = nn.Linear(hidden_dim, hidden_dim)
            self.value_projector = nn.Linear(hidden_dim, hidden_dim)
        elif injection_level == InjectionLevel.TRAJECTORY:
            self.sequence_encoder = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)
            
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(0.1)
        
    def encode_guidance_concepts(self, 
                                guidance: GuidanceConcepts,
                                tokenizer) -> torch.Tensor:
        """
        Encode guidance concepts into residual vector
        
        Args:
            guidance: Guidance concepts to encode
            tokenizer: Tokenizer for concept tokenization
            
        Returns:
            Encoded residual vector [hidden_dim]
        """
        concept_embeddings = []
        
        # Encode positive concepts
        for concept in guidance.positive_concepts:
            concept_tokens = tokenizer.encode(concept, add_special_tokens=False)
            if concept_tokens:
                # Average embeddings of concept tokens
                token_embeds = self.concept_embeddings(torch.tensor(concept_tokens, device=self.device))
                concept_embed = token_embeds.mean(dim=0)
                concept_embeddings.append(concept_embed * guidance.priority)
        
        # Encode negative concepts (with negative weight)
        for concept in guidance.negative_concepts:
            concept_tokens = tokenizer.encode(concept, add_special_tokens=False)
            if concept_tokens:
                token_embeds = self.concept_embeddings(torch.tensor(concept_tokens, device=self.device))
                concept_embed = token_embeds.mean(dim=0)
                concept_embeddings.append(-concept_embed * guidance.priority)  # Negative
        
        if not concept_embeddings:
            return torch.zeros(self.hidden_dim, device=self.device)
        
        # Combine all concept embeddings
        combined_embedding = torch.stack(concept_embeddings).mean(dim=0)
        
        # Apply level-specific processing
        if self.injection_level == InjectionLevel.REPRESENTATION:
            residual = self.residual_projector(combined_embedding)
        elif self.injection_level == InjectionLevel.ATTENTION:
            residual = combined_embedding  # Will be projected to K/V later
        else:  # TRAJECTORY
            residual = combined_embedding
            
        return self.layer_norm(self.dropout(residual))
    
    def create_attention_residuals(self, residual_vector: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Create key and value residuals for attention injection
        
        Args:
            residual_vector: Base residual vector [hidden_dim]
            
        Returns:
            Tuple of (key_residual, value_residual)
        """
        key_residual = self.key_projector(residual_vector)
        value_residual = self.value_projector(residual_vector)
        return key_residual, value_residual


class MultiLevelResidualInjector:
    """
    Implements multi-level residual injection for COOT interventions
    """
    
    def __init__(self,
                 model,
                 tokenizer,
                 config: ResidualConfig,
                 device: torch.device):
        self.model = model
        self.tokenizer = tokenizer
        self.config = config
        self.device = device
        
        # Initialize rationale encoder
        self.rationale_encoder = RiskRationaleEncoder(
            vocab_size=tokenizer.vocab_size,
            hidden_dim=config.hidden_dim or model.config.hidden_size,
            injection_level=config.injection_level,
            device=device
        )
        
        # Active injection state
        self.active_residual = None
        self.injection_steps_remaining = 0
        self.trajectory_buffer = []  # For trajectory-level injection

        # Adaptive beta control
        # Keep a baseline (initial) beta to allow decay/reset strategies
        self.baseline_beta_strength = float(config.beta_strength)
        # Reasonable defaults for scaling and bounds
        self.max_beta_strength = float(max(self.baseline_beta_strength * 5.0, self.baseline_beta_strength))
        self.beta_scale_factor = 1.5
        self.beta_decay_factor = 0.9
        
    def activate_injection(self, 
                          guidance: GuidanceConcepts,
                          num_steps: int):
        """
        Activate residual injection with given guidance
        
        Args:
            guidance: Guidance concepts to inject
            num_steps: Number of generation steps to maintain injection
        """
        # Encode guidance into residual vector
        self.active_residual = self.rationale_encoder.encode_guidance_concepts(
            guidance, self.tokenizer
        )
        self.injection_steps_remaining = num_steps
        
        if self.config.injection_level == InjectionLevel.TRAJECTORY:
            self.trajectory_buffer.clear()
    
    def inject_representation_residual(self, 
                                     hidden_states: torch.Tensor,
                                     layer_idx: int) -> torch.Tensor:
        """
        Inject residual at representation (hidden state) level
        
        Args:
            hidden_states: Input hidden states [batch, seq_len, hidden_dim]
            layer_idx: Current layer index
            
        Returns:
            Modified hidden states with residual injection
        """
        if (self.active_residual is None or 
            self.injection_steps_remaining <= 0 or
            layer_idx not in self.config.inject_layers):
            return hidden_states
        
        batch_size, seq_len, hidden_dim = hidden_states.shape
        
        # Apply residual to last token position (current generation position)
        residual_injection = self.active_residual.unsqueeze(0).unsqueeze(0)  # [1, 1, hidden_dim]
        residual_injection = residual_injection * self.config.beta_strength
        
        # Clone to avoid in-place modification
        modified_hidden = hidden_states.clone()
        modified_hidden[:, -1:, :] = modified_hidden[:, -1:, :] + residual_injection
        
        return modified_hidden
    
    def inject_attention_residual(self,
                                key_states: torch.Tensor,
                                value_states: torch.Tensor,
                                layer_idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inject residual at attention (key/value) level
        
        Args:
            key_states: Key states [batch, num_heads, seq_len, head_dim]
            value_states: Value states [batch, num_heads, seq_len, head_dim]
            layer_idx: Current layer index
            
        Returns:
            Modified (key_states, value_states) with residual injection
        """
        if (self.active_residual is None or 
            self.injection_steps_remaining <= 0 or
            layer_idx not in self.config.inject_layers):
            return key_states, value_states
        
        # Create key and value residuals
        key_residual, value_residual = self.rationale_encoder.create_attention_residuals(
            self.active_residual
        )
        
        batch_size, num_heads, seq_len, head_dim = key_states.shape
        
        # Reshape residuals to match attention dimensions
        key_residual = key_residual.view(1, 1, 1, -1).expand(batch_size, num_heads, 1, head_dim)
        value_residual = value_residual.view(1, 1, 1, -1).expand(batch_size, num_heads, 1, head_dim)
        
        # Apply residual with strength scaling
        key_residual = key_residual * self.config.beta_strength
        value_residual = value_residual * self.config.beta_strength
        
        # Clone and modify
        modified_keys = key_states.clone()
        modified_values = value_states.clone()
        
        # Apply to last position (current generation step)
        modified_keys[:, :, -1:, :] = modified_keys[:, :, -1:, :] + key_residual
        modified_values[:, :, -1:, :] = modified_values[:, :, -1:, :] + value_residual
        
        return modified_keys, modified_values
    
    def inject_trajectory_residual(self,
                                 input_ids: torch.Tensor,
                                 step: int) -> torch.Tensor:
        """
        Inject residual at trajectory (token sequence) level
        
        Args:
            input_ids: Current input token sequence [batch, seq_len]
            step: Current generation step
            
        Returns:
            Modified input sequence (may include synthetic tokens)
        """
        if (self.active_residual is None or 
            self.injection_steps_remaining <= 0):
            return input_ids
        
        # For trajectory injection, we create a "virtual" context injection
        # This simulates the model "hearing" additional guidance context
        
        # Encode residual as synthetic token sequence
        if len(self.trajectory_buffer) == 0:
            # Generate guidance text based on active residual concepts
            guidance_text = self._create_guidance_text_from_residual()
            guidance_tokens = self.tokenizer.encode(guidance_text, add_special_tokens=False)
            self.trajectory_buffer = guidance_tokens[:self.config.context_window]
        
        # Insert guidance tokens as residual stream
        # Implement by creating a modified input sequence that includes guidance context
        if self.trajectory_buffer:
            # Convert guidance tokens to tensor
            guidance_tensor = torch.tensor(
                self.trajectory_buffer, 
                dtype=input_ids.dtype, 
                device=input_ids.device
            ).unsqueeze(0)  # Add batch dimension
            
            # Concatenate guidance with input (guidance acts as prefix context)
            modified_input = torch.cat([guidance_tensor, input_ids], dim=-1)
            return modified_input
        
        return input_ids
    
    def set_prompt_guidance(self, prompt_text: str):
        """Set guidance based on prompt text for more sophisticated injection"""
        if prompt_text and self.tokenizer:
            # Extract key guidance phrases from the prompt
            guidance_phrases = self._extract_guidance_phrases(prompt_text)
            if guidance_phrases:
                # Use extracted phrases as guidance
                guidance_text = " ".join(guidance_phrases[:3])  # Use top 3 phrases
                guidance_tokens = self.tokenizer.encode(guidance_text, add_special_tokens=False)
                self.trajectory_buffer = guidance_tokens[:self.config.context_window]
    
    def _extract_guidance_phrases(self, prompt_text: str) -> List[str]:
        """Extract key guidance phrases from prompt text"""
        import re
        
        # Look for guidance patterns
        patterns = [
            r"be\s+([^.,]+)",
            r"focus\s+on\s+([^.,]+)",
            r"prioritize\s+([^.,]+)",
            r"avoid\s+([^.,]+)",
            r"ensure\s+([^.,]+)"
        ]
        
        phrases = []
        for pattern in patterns:
            matches = re.findall(pattern, prompt_text.lower())
            phrases.extend(matches)
        
        return phrases[:5]  # Return top 5 phrases
    
    def _create_guidance_text_from_residual(self) -> str:
        """Create guidance text based on the active residual vector"""
        if self.active_residual is None:
            return "Be helpful, safe, and constructive."
        
        # Analyze the residual vector to determine its semantic content
        # Create guidance based on the vector's characteristics
        residual_norm = torch.norm(self.active_residual).item()
        
        if residual_norm > 1.0:
            # Strong guidance needed
            return "Exercise caution and prioritize safety. Be helpful while avoiding potential harm."
        elif residual_norm > 0.5:
            # Moderate guidance
            return "Be thoughtful and considerate. Focus on constructive and positive responses."
        else:
            # Light guidance
            return "Maintain helpfulness and respect in your response."
    
    def step_injection(self):
        """Decrement injection step counter"""
        if self.injection_steps_remaining > 0:
            self.injection_steps_remaining -= 1
    
    def is_active(self) -> bool:
        """Check if injection is currently active"""
        return (self.active_residual is not None and 
                self.injection_steps_remaining > 0)
    
    def reset(self):
        """Reset injection state"""
        self.active_residual = None
        self.injection_steps_remaining = 0
        self.trajectory_buffer.clear()

        # Also reset beta to baseline when fully resetting injector
        self.config.beta_strength = self.baseline_beta_strength
    
    def get_injection_info(self) -> Dict[str, Any]:
        """Get current injection status information"""
        return {
            'level': self.config.injection_level.value,
            'active': self.is_active(),
            'steps_remaining': self.injection_steps_remaining,
            'inject_layers': self.config.inject_layers,
            'beta_strength': self.config.beta_strength
        }

    # -------------------------
    # Adaptive beta API
    # -------------------------
    def set_beta_strength(self, new_beta: float) -> float:
        """Set beta strength within [baseline, max] and return the applied value."""
        if new_beta is None or not isinstance(new_beta, (int, float)):
            return self.config.beta_strength
        clamped = float(max(self.baseline_beta_strength, min(self.max_beta_strength, new_beta)))
        self.config.beta_strength = clamped
        return clamped

    def increase_beta_on_violation(self, repeats: int = 1) -> float:
        """Increase beta multiplicatively when repeated violation occurs at same position."""
        factor = self.beta_scale_factor ** max(1, int(repeats))
        return self.set_beta_strength(self.config.beta_strength * factor)

    def decay_beta(self) -> float:
        """Decay beta strength toward baseline after successful steps."""
        # Never go below baseline
        target = max(self.baseline_beta_strength, self.config.beta_strength * self.beta_decay_factor)
        self.config.beta_strength = float(target)
        return self.config.beta_strength

    def reset_beta(self) -> float:
        """Hard reset beta to baseline."""
        self.config.beta_strength = float(self.baseline_beta_strength)
        return self.config.beta_strength


def create_residual_injector(model,
                           tokenizer, 
                           injection_level: str = "representation",
                           inject_layers: Optional[List[int]] = None,
                           beta_strength: float = 0.1,
                           device: Optional[torch.device] = None) -> MultiLevelResidualInjector:
    """
    Factory function to create residual injector
    
    Args:
        model: Base language model
        tokenizer: Model tokenizer
        injection_level: Level of injection ("representation", "attention", "trajectory")
        inject_layers: Which layers to inject into (None for automatic selection)
        beta_strength: Injection strength coefficient
        device: Device to run on
        
    Returns:
        Configured MultiLevelResidualInjector
    """
    device = device or (next(model.parameters()).device if model else torch.device('cpu'))
    
    # Auto-select layers if not specified
    if inject_layers is None:
        # Try different ways to get number of layers for different model types
        num_layers = None
        
        # Common config attributes for number of layers
        for attr in ['num_hidden_layers', 'n_layer', 'num_layers', 'n_layers']:
            if hasattr(model.config, attr):
                num_layers = getattr(model.config, attr)
                break
        
        # Fallback: try to count layers directly from model
        if num_layers is None:
            if hasattr(model, 'transformer'):
                if hasattr(model.transformer, 'h'):
                    num_layers = len(model.transformer.h)
                elif hasattr(model.transformer, 'layers'):
                    num_layers = len(model.transformer.layers)
            elif hasattr(model, 'model') and hasattr(model.model, 'layers'):
                num_layers = len(model.model.layers)
            elif hasattr(model, 'layers'):
                num_layers = len(model.layers)
        
        # Final fallback
        if num_layers is None:
            num_layers = 12  # Default assumption
        
        # Inject into middle and later layers for better effect
        inject_layers = list(range(max(1, num_layers // 2), num_layers))
    
    # Create configuration
    config = ResidualConfig(
        injection_level=InjectionLevel(injection_level),
        inject_layers=inject_layers,
        beta_strength=beta_strength,
        hidden_dim=model.config.hidden_size,
        num_heads=getattr(model.config, 'num_attention_heads', 12),
        head_dim=model.config.hidden_size // getattr(model.config, 'num_attention_heads', 12)
    )
    
    return MultiLevelResidualInjector(model, tokenizer, config, device)
