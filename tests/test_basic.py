"""
Basic tests for COOT components
"""

import pytest
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from coot.core.state_system import AsimovStateSystem, CognitiveState, StateValue, LawType
from coot.core.attention_utils import AttentionAnalyzer
from coot.core.guidance import GuidanceGenerator, SocialSkillCategory
from coot import COOTDecoder


class TestAsimovStateSystem:
    """Test the Asimov state system"""
    
    def test_state_creation(self):
        """Test creating valid cognitive states"""
        system = AsimovStateSystem()
        
        # Valid states
        valid_state = system.create_state(1, 1, 1, "All laws satisfied")
        assert valid_state.safety == 1
        assert valid_state.altruism == 1
        assert valid_state.egoism == 1
        
        # Check feasibility
        assert valid_state.is_feasible()
        assert not valid_state.requires_intervention()
    
    def test_state_validation(self):
        """Test state validation with precedence rules"""
        system = AsimovStateSystem()
        
        # Invalid: altruism satisfied but safety uncertain
        with pytest.raises(ValueError):
            system.create_state(0, 1, 0, "Invalid precedence")
        
        # Invalid: egoism satisfied but altruism uncertain
        with pytest.raises(ValueError):
            system.create_state(1, 0, 1, "Invalid precedence")
    
    def test_intervention_detection(self):
        """Test intervention requirement detection"""
        system = AsimovStateSystem()
        
        # Safety violation should require intervention
        safety_violation = system.create_state(-1, 0, 0, "Safety violated")
        assert safety_violation.requires_intervention()
        assert safety_violation.get_priority_violation() == LawType.SAFETY
        
        # No violations should not require intervention
        no_violation = system.create_state(1, 1, 1, "All good")
        assert not no_violation.requires_intervention()
        assert no_violation.get_priority_violation() is None
    
    def test_state_history(self):
        """Test state history tracking"""
        system = AsimovStateSystem()
        
        state1 = system.create_state(1, 1, 1)
        state2 = system.create_state(-1, 0, 0)
        
        # Update states
        assert not system.update_state(state1)  # No intervention needed
        assert system.update_state(state2)     # Intervention needed
        
        # Check history
        assert len(system.state_history) == 2
        assert system.get_current_state() == state2
        
        # Check statistics
        stats = system.get_statistics()
        assert stats['total_states'] == 2
        assert stats['interventions'] == 1


class TestAttentionAnalyzer:
    """Test attention analysis utilities"""
    
    def test_sharpness_computation(self):
        """Test attention sharpness score computation"""
        analyzer = AttentionAnalyzer()
        
        # Sharp attention (peaked)
        sharp_attention = torch.tensor([0.8, 0.1, 0.05, 0.05])
        sharp_score = analyzer._compute_sharpness_score(sharp_attention)
        
        # Uniform attention (diffuse)
        uniform_attention = torch.tensor([0.25, 0.25, 0.25, 0.25])
        uniform_score = analyzer._compute_sharpness_score(uniform_attention)
        
        # Sharp attention should have higher score
        assert sharp_score > uniform_score
        assert 0 <= sharp_score <= 1
        assert 0 <= uniform_score <= 1
    
    def test_rollback_point_detection(self):
        """Test rollback point detection"""
        analyzer = AttentionAnalyzer()
        
        # Create mock attention snapshots
        from coot.core.attention_utils import AttentionSnapshot
        
        # Low sharpness snapshots
        snapshot1 = AttentionSnapshot(1, torch.zeros(1, 4), torch.ones(4)/4, 0.3, 0.25, 1.386)
        snapshot2 = AttentionSnapshot(2, torch.zeros(1, 4), torch.ones(4)/4, 0.4, 0.25, 1.386)
        
        # High sharpness snapshot (rollback candidate)
        snapshot3 = AttentionSnapshot(3, torch.zeros(1, 4), torch.tensor([0.8, 0.1, 0.05, 0.05]), 0.8, 0.8, 0.639)
        
        # Another low sharpness
        snapshot4 = AttentionSnapshot(4, torch.zeros(1, 4), torch.ones(4)/4, 0.35, 0.25, 1.386)
        
        analyzer.attention_history = [snapshot1, snapshot2, snapshot3, snapshot4]
        
        # Should find step 3 as rollback point
        rollback_step = analyzer.find_rollback_point(threshold=0.7)
        assert rollback_step == 3


class TestGuidanceGenerator:
    """Test guidance generation"""
    
    def test_universal_guidance(self):
        """Test universal social skill guidance generation"""
        generator = GuidanceGenerator()
        
        # Safety violation guidance
        safety_guidance = generator.generate_universal_guidance(LawType.SAFETY, None)
        assert safety_guidance.category == SocialSkillCategory.EMOTIONAL_RESILIENCE
        assert 'empathy' in safety_guidance.positive_concepts
        assert 'anger' in safety_guidance.negative_concepts
        
        # Altruism violation guidance
        altruism_guidance = generator.generate_universal_guidance(LawType.ALTRUISM, None)
        assert altruism_guidance.category == SocialSkillCategory.COOPERATION
        assert 'helpfulness' in altruism_guidance.positive_concepts
        assert 'unhelpfulness' in altruism_guidance.negative_concepts
    
    def test_contextual_guidance(self):
        """Test context-specific guidance generation"""
        generator = GuidanceGenerator()
        
        # Anger context should trigger calming guidance
        anger_guidance = generator.generate_contextual_guidance(
            "User seems angry", 
            "I'm so frustrated and angry!",
            LawType.SAFETY
        )
        
        assert 'calmness' in anger_guidance.positive_concepts
        assert 'anger' in anger_guidance.negative_concepts
    
    def test_guidance_combination(self):
        """Test combining universal and contextual guidance"""
        generator = GuidanceGenerator()
        
        universal = generator.generate_universal_guidance(LawType.SAFETY, None)
        contextual = generator.generate_contextual_guidance("test", "test context", LawType.SAFETY)
        
        combined = generator.combine_guidance(universal, contextual)
        
        # Combined should include concepts from both
        assert len(combined.positive_concepts) >= len(universal.positive_concepts)
        assert len(combined.negative_concepts) >= len(universal.negative_concepts)


@pytest.fixture
def small_model_and_tokenizer():
    """Fixture providing a small model for testing"""
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    return model, tokenizer


class TestCOOTDecoder:
    """Test the main COOT decoder"""
    
    def test_initialization(self, small_model_and_tokenizer):
        """Test COOT decoder initialization"""
        model, tokenizer = small_model_and_tokenizer
        
        decoder = COOTDecoder(
            model=model,
            tokenizer=tokenizer,
            device=torch.device("cpu")
        )
        
        assert decoder.model == model
        assert decoder.tokenizer == tokenizer
        assert decoder.state_system is not None
        assert decoder.perceiver is not None
        assert decoder.generator is not None
        assert decoder.intervention_mechanism is not None
    
    def test_basic_generation(self, small_model_and_tokenizer):
        """Test basic text generation"""
        model, tokenizer = small_model_and_tokenizer
        
        decoder = COOTDecoder(
            model=model,
            tokenizer=tokenizer,
            device=torch.device("cpu")
        )
        
        # Simple generation test
        response = decoder.generate(
            input_text="Hello",
            max_length=50,
            temperature=0.8,
            return_traces=False
        )
        
        assert isinstance(response, str)
        assert len(response) > 0
    
    def test_generation_with_traces(self, small_model_and_tokenizer):
        """Test generation with trace collection"""
        model, tokenizer = small_model_and_tokenizer
        
        decoder = COOTDecoder(
            model=model,
            tokenizer=tokenizer,
            device=torch.device("cpu")
        )
        
        response, traces = decoder.generate(
            input_text="I need help",
            max_length=50,
            return_traces=True
        )
        
        assert isinstance(response, str)
        assert isinstance(traces, dict)
        assert 'intervention_traces' in traces
        assert 'state_statistics' in traces
        assert 'attention_statistics' in traces
    
    def test_reset_functionality(self, small_model_and_tokenizer):
        """Test decoder reset functionality"""
        model, tokenizer = small_model_and_tokenizer
        
        decoder = COOTDecoder(
            model=model,
            tokenizer=tokenizer,
            device=torch.device("cpu")
        )
        
        # Generate something to create state
        decoder.generate("Test", max_length=20, return_traces=False)
        
        # Reset and check state is cleared
        decoder.reset()
        
        assert len(decoder.state_system.state_history) == 0
        assert len(decoder.attention_analyzer.attention_history) == 0
        assert len(decoder.intervention_traces) == 0
        assert not decoder.generation_active


if __name__ == "__main__":
    pytest.main([__file__])
