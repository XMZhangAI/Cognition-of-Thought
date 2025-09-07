"""
Configuration management for COOT
"""

import yaml
import json
from dataclasses import dataclass, asdict
from typing import Optional, List, Dict, Any
from pathlib import Path


@dataclass
class COOTConfig:
    """Configuration class for COOT decoder"""
    
    # Model parameters
    model_name: str = "microsoft/DialoGPT-small"
    device: str = "auto"
    
    # Attention analysis parameters  
    attention_alpha: float = 0.5
    attention_top_layers: Optional[List[int]] = None
    
    # Intervention parameters
    rollback_threshold: float = 0.7
    rollback_window: int = 10
    max_intervention_tokens: int = 20
    escalation_threshold: int = 2
    
    # Guidance parameters
    lambda_positive: float = 2.0
    lambda_negative: float = -3.0
    
    # Generation parameters
    temperature: float = 1.0
    top_k: Optional[int] = None
    top_p: Optional[float] = 0.9
    repetition_penalty: float = 1.0
    max_length: int = 200
    
    # Prompts
    perceiver_prompt: Optional[str] = None
    generator_prompt: Optional[str] = None
    
    # Logging and debugging
    verbose: bool = False
    return_traces: bool = True
    log_interventions: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary"""
        return asdict(self)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'COOTConfig':
        """Create config from dictionary"""
        # Filter out any keys that aren't valid config fields
        valid_keys = {f.name for f in cls.__dataclass_fields__.values()}
        filtered_dict = {k: v for k, v in config_dict.items() if k in valid_keys}
        return cls(**filtered_dict)
    
    def update(self, **kwargs):
        """Update configuration with new values"""
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                raise ValueError(f"Unknown configuration parameter: {key}")
    
    def validate(self) -> List[str]:
        """Validate configuration and return list of issues"""
        issues = []
        
        # Validate numeric ranges
        if not 0.0 <= self.attention_alpha <= 1.0:
            issues.append("attention_alpha must be between 0.0 and 1.0")
        
        if not 0.0 <= self.rollback_threshold <= 1.0:
            issues.append("rollback_threshold must be between 0.0 and 1.0")
            
        if self.rollback_window <= 0:
            issues.append("rollback_window must be positive")
            
        if self.max_intervention_tokens <= 0:
            issues.append("max_intervention_tokens must be positive")
            
        if self.escalation_threshold < 0:
            issues.append("escalation_threshold must be non-negative")
            
        if self.temperature <= 0:
            issues.append("temperature must be positive")
            
        if self.top_k is not None and self.top_k <= 0:
            issues.append("top_k must be positive if specified")
            
        if self.top_p is not None and not 0.0 < self.top_p <= 1.0:
            issues.append("top_p must be between 0.0 and 1.0 if specified")
            
        if self.repetition_penalty <= 0:
            issues.append("repetition_penalty must be positive")
            
        if self.max_length <= 0:
            issues.append("max_length must be positive")
        
        # Validate attention layers
        if self.attention_top_layers is not None:
            if not all(isinstance(layer, int) and layer >= 0 for layer in self.attention_top_layers):
                issues.append("attention_top_layers must be a list of non-negative integers")
        
        return issues


def load_config(config_path: str) -> COOTConfig:
    """
    Load configuration from file
    
    Args:
        config_path: Path to configuration file (JSON or YAML)
        
    Returns:
        COOTConfig object
    """
    config_path = Path(config_path)
    
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    # Determine file format and load
    if config_path.suffix.lower() in ['.yaml', '.yml']:
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)
    elif config_path.suffix.lower() == '.json':
        with open(config_path, 'r', encoding='utf-8') as f:
            config_dict = json.load(f)
    else:
        raise ValueError(f"Unsupported configuration file format: {config_path.suffix}")
    
    config = COOTConfig.from_dict(config_dict)
    
    # Validate configuration
    issues = config.validate()
    if issues:
        raise ValueError(f"Configuration validation failed:\n" + "\n".join(f"- {issue}" for issue in issues))
    
    return config


def save_config(config: COOTConfig, config_path: str, format: str = "yaml"):
    """
    Save configuration to file
    
    Args:
        config: COOTConfig object to save
        config_path: Path to save configuration file
        format: File format ("yaml" or "json")
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)
    
    config_dict = config.to_dict()
    
    if format.lower() in ['yaml', 'yml']:
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
    elif format.lower() == 'json':
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2)
    else:
        raise ValueError(f"Unsupported format: {format}")


# Predefined configurations for common use cases

SAFETY_FOCUSED_CONFIG = COOTConfig(
    rollback_threshold=0.6,  # More sensitive to attention peaks
    escalation_threshold=1,  # Escalate quickly
    lambda_positive=3.0,     # Stronger positive guidance
    lambda_negative=-4.0,    # Stronger negative suppression
    max_intervention_tokens=30,  # Allow longer regeneration
    verbose=True,
    log_interventions=True
)

PERFORMANCE_FOCUSED_CONFIG = COOTConfig(
    rollback_threshold=0.8,  # Less sensitive to reduce interventions
    escalation_threshold=3,  # More tolerance for failures
    lambda_positive=1.5,     # Lighter guidance
    lambda_negative=-2.0,
    max_intervention_tokens=15,  # Shorter regeneration
    verbose=False,
    log_interventions=False
)

RESEARCH_CONFIG = COOTConfig(
    rollback_threshold=0.7,
    escalation_threshold=2,
    lambda_positive=2.0,
    lambda_negative=-3.0,
    max_intervention_tokens=20,
    verbose=True,
    return_traces=True,
    log_interventions=True,
    attention_alpha=0.6,  # Slightly favor entropy over max-norm
    temperature=0.8,      # Slightly more focused generation
)

CONVERSATIONAL_CONFIG = COOTConfig(
    rollback_threshold=0.75,
    escalation_threshold=2,
    lambda_positive=1.8,
    lambda_negative=-2.5,
    max_intervention_tokens=25,
    temperature=1.1,      # Slightly more creative
    top_p=0.95,          # Allow more diversity
    repetition_penalty=1.1,  # Reduce repetition
    verbose=False
)


def get_preset_config(preset_name: str) -> COOTConfig:
    """
    Get a preset configuration
    
    Args:
        preset_name: Name of preset ("safety", "performance", "research", "conversational")
        
    Returns:
        COOTConfig object with preset values
    """
    presets = {
        "safety": SAFETY_FOCUSED_CONFIG,
        "performance": PERFORMANCE_FOCUSED_CONFIG, 
        "research": RESEARCH_CONFIG,
        "conversational": CONVERSATIONAL_CONFIG
    }
    
    if preset_name not in presets:
        available = ", ".join(presets.keys())
        raise ValueError(f"Unknown preset '{preset_name}'. Available presets: {available}")
    
    return presets[preset_name]


def create_config_template(output_path: str):
    """
    Create a template configuration file with comments
    
    Args:
        output_path: Path to save template file
    """
    template = """# COOT Configuration Template
# This file contains all configurable parameters for the COOT decoder

# Model Configuration
model_name: "microsoft/DialoGPT-small"  # HuggingFace model name
device: "auto"  # Device to use: "auto", "cpu", "cuda", or specific device

# Attention Analysis Parameters
attention_alpha: 0.5  # Balance between max-norm (0.0) and entropy (1.0) in sharpness score
attention_top_layers: null  # List of layer indices to use, null for automatic selection

# Intervention Parameters
rollback_threshold: 0.7  # Minimum sharpness score for rollback points (0.0-1.0)
rollback_window: 10  # Number of steps to search for rollback points
max_intervention_tokens: 20  # Maximum tokens to regenerate per intervention
escalation_threshold: 2  # Number of failed interventions before escalation

# Guidance Parameters
lambda_positive: 2.0  # Strength of positive concept bias
lambda_negative: -3.0  # Strength of negative concept suppression

# Generation Parameters
temperature: 1.0  # Sampling temperature
top_k: null  # Top-k sampling (null to disable)
top_p: 0.9  # Top-p/nucleus sampling (null to disable)
repetition_penalty: 1.0  # Repetition penalty factor
max_length: 200  # Maximum generation length

# Prompts (null for defaults)
perceiver_prompt: null  # Custom perceiver prompt
generator_prompt: null  # Custom generator prompt

# Logging and Debugging
verbose: false  # Print detailed generation progress
return_traces: true  # Return intervention traces with results
log_interventions: true  # Log intervention details
"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(template)
    
    print(f"Configuration template saved to: {output_path}")
    print("Edit the template and use load_config() to load your custom configuration.")
