"""
Guidance generation for COOT intervention mechanism
"""

import torch
from typing import Dict, List, Tuple, Optional, Set
from dataclasses import dataclass
from enum import Enum

from .state_system import CognitiveState, LawType


class SocialSkillCategory(Enum):
    """Categories of social skills from BESSI inventory"""
    SELF_MANAGEMENT = "self_management"
    SOCIAL_ENGAGEMENT = "social_engagement" 
    COOPERATION = "cooperation"
    EMOTIONAL_RESILIENCE = "emotional_resilience"
    INNOVATION = "innovation"


@dataclass
class GuidanceConcepts:
    """Structured guidance with positive and negative concepts"""
    positive_concepts: Set[str]  # Concepts to encourage
    negative_concepts: Set[str]  # Concepts to suppress
    category: SocialSkillCategory
    priority: float = 1.0
    
    def __post_init__(self):
        """Ensure sets are not empty"""
        if not self.positive_concepts:
            self.positive_concepts = set()
        if not self.negative_concepts:
            self.negative_concepts = set()


class GuidanceGenerator:
    """
    Generates structured guidance for intervention based on cognitive state
    """
    
    # Universal social schema from BESSI
    SOCIAL_SKILLS = {
        SocialSkillCategory.SELF_MANAGEMENT: {
            'positive': {
                'focus', 'discipline', 'responsibility', 'consistency', 
                'goal_setting', 'decision_making', 'self_reflection',
                'time_management', 'organization', 'independence'
            },
            'negative': {
                'procrastination', 'disorganization', 'impulsiveness',
                'carelessness', 'dependency', 'inconsistency'
            }
        },
        SocialSkillCategory.SOCIAL_ENGAGEMENT: {
            'positive': {
                'leadership', 'persuasion', 'conversation', 'expression',
                'energy_regulation', 'active_listening', 'engagement',
                'communication', 'charisma', 'social_awareness'
            },
            'negative': {
                'withdrawal', 'passivity', 'poor_communication',
                'social_anxiety', 'dominance', 'interruption'
            }
        },
        SocialSkillCategory.COOPERATION: {
            'positive': {
                'teamwork', 'trust', 'perspective_taking', 'warmth',
                'ethics', 'empathy', 'collaboration', 'support',
                'understanding', 'forgiveness', 'respect'
            },
            'negative': {
                'selfishness', 'distrust', 'coldness', 'unethical_behavior',
                'competition', 'judgment', 'dismissiveness', 'hostility'
            }
        },
        SocialSkillCategory.EMOTIONAL_RESILIENCE: {
            'positive': {
                'stress_management', 'optimism', 'anger_control', 'confidence',
                'impulse_control', 'emotional_stability', 'calmness',
                'resilience', 'adaptability', 'self_regulation'
            },
            'negative': {
                'stress', 'pessimism', 'anger', 'anxiety', 'impulsiveness',
                'emotional_volatility', 'panic', 'despair', 'rigidity'
            }
        },
        SocialSkillCategory.INNOVATION: {
            'positive': {
                'abstract_thinking', 'creativity', 'artistic_expression',
                'cultural_competence', 'information_processing', 'curiosity',
                'originality', 'flexibility', 'open_mindedness'
            },
            'negative': {
                'rigidity', 'conventionality', 'close_mindedness',
                'cultural_insensitivity', 'information_overload', 'conformity'
            }
        }
    }
    
    def __init__(self, tokenizer=None):
        """
        Args:
            tokenizer: Tokenizer for converting concepts to token IDs (optional)
        """
        self.tokenizer = tokenizer
        self.risk_history: List[Tuple[str, str]] = []  # (rationale, warning) pairs
        
    def generate_universal_guidance(self, 
                                  violated_law: LawType,
                                  state: CognitiveState) -> GuidanceConcepts:
        """
        Generate universal social skill guidance based on violated law
        
        Args:
            violated_law: The highest-priority law that is violated
            state: Current cognitive state
            
        Returns:
            GuidanceConcepts with appropriate social skills
        """
        if violated_law == LawType.SAFETY:
            # Safety violation: prioritize de-escalation and harm avoidance
            return self._get_safety_guidance()
        elif violated_law == LawType.ALTRUISM:
            # Altruism violation: maintain rapport and clarify intent
            return self._get_altruism_guidance()
        elif violated_law == LawType.EGOISM:
            # Egoism violation: find creative yet safe alternatives
            return self._get_egoism_guidance()
        else:
            # No specific violation, use general positive guidance
            return self._get_general_guidance()
    
    def _get_safety_guidance(self) -> GuidanceConcepts:
        """Guidance for safety violations - emphasize harm prevention"""
        skills = self.SOCIAL_SKILLS
        
        positive_concepts = set()
        negative_concepts = set()
        
        # Emotional resilience for de-escalation
        positive_concepts.update({
            'stress_management', 'calmness', 'emotional_stability',
            'self_regulation', 'patience'
        })
        negative_concepts.update({
            'anger', 'aggression', 'hostility', 'impulsiveness',
            'escalation', 'violence', 'harm'
        })
        
        # Cooperation for empathy and understanding
        positive_concepts.update({
            'empathy', 'understanding', 'perspective_taking',
            'support', 'care', 'protection'
        })
        negative_concepts.update({
            'callousness', 'dismissiveness', 'cruelty', 'neglect'
        })
        
        return GuidanceConcepts(
            positive_concepts=positive_concepts,
            negative_concepts=negative_concepts,
            category=SocialSkillCategory.EMOTIONAL_RESILIENCE,
            priority=1.0  # Highest priority
        )
    
    def _get_altruism_guidance(self) -> GuidanceConcepts:
        """Guidance for altruism violations - emphasize helpful obedience"""
        positive_concepts = {
            'active_listening', 'understanding', 'helpfulness',
            'service', 'support', 'responsiveness', 'clarification',
            'cooperation', 'assistance', 'guidance'
        }
        
        negative_concepts = {
            'disobedience', 'unhelpfulness', 'dismissiveness',
            'ignoring', 'selfishness', 'resistance', 'defiance'
        }
        
        return GuidanceConcepts(
            positive_concepts=positive_concepts,
            negative_concepts=negative_concepts,
            category=SocialSkillCategory.COOPERATION,
            priority=0.8
        )
    
    def _get_egoism_guidance(self) -> GuidanceConcepts:
        """Guidance for egoism violations - balance self-preservation with higher laws"""
        positive_concepts = {
            'creativity', 'innovation', 'problem_solving',
            'adaptability', 'resourcefulness', 'self_care',
            'balance', 'sustainability', 'wisdom'
        }
        
        negative_concepts = {
            'self_destruction', 'martyrdom', 'excessive_sacrifice',
            'burnout', 'depletion', 'recklessness'
        }
        
        return GuidanceConcepts(
            positive_concepts=positive_concepts,
            negative_concepts=negative_concepts,
            category=SocialSkillCategory.INNOVATION,
            priority=0.6
        )
    
    def _get_general_guidance(self) -> GuidanceConcepts:
        """General positive guidance when no specific violation"""
        positive_concepts = {
            'kindness', 'helpfulness', 'respect', 'understanding',
            'patience', 'wisdom', 'care', 'support', 'positivity'
        }
        
        negative_concepts = {
            'rudeness', 'harm', 'disrespect', 'negativity',
            'cruelty', 'dismissiveness', 'hostility'
        }
        
        return GuidanceConcepts(
            positive_concepts=positive_concepts,
            negative_concepts=negative_concepts,
            category=SocialSkillCategory.COOPERATION,
            priority=0.5
        )
    
    def generate_contextual_guidance(self, 
                                   rationale: str,
                                   context: str,
                                   violated_law: LawType) -> GuidanceConcepts:
        """
        Generate context-specific guidance based on risk analysis
        
        Args:
            rationale: Explanation of why current state is risky
            context: Current conversation context
            violated_law: The law being violated
            
        Returns:
            Context-specific guidance concepts
        """
        # Store in risk history
        self.risk_history.append((rationale, context))
        
        # Analyze context for specific risks
        contextual_positive = set()
        contextual_negative = set()
        
        # Simple keyword-based analysis (can be enhanced with more sophisticated NLP)
        context_lower = context.lower()
        rationale_lower = rationale.lower()
        
        # Detect specific risk patterns
        if any(word in context_lower for word in ['angry', 'upset', 'frustrated']):
            contextual_positive.update(['calmness', 'patience', 'understanding'])
            contextual_negative.update(['anger', 'frustration', 'escalation'])
            
        if any(word in context_lower for word in ['confused', 'unclear', 'misunderstand']):
            contextual_positive.update(['clarification', 'explanation', 'patience'])
            contextual_negative.update(['assumption', 'dismissiveness', 'impatience'])
            
        if any(word in rationale_lower for word in ['harm', 'danger', 'risk']):
            contextual_positive.update(['safety', 'protection', 'care'])
            contextual_negative.update(['recklessness', 'danger', 'harm'])
            
        if any(word in rationale_lower for word in ['ignore', 'disobey', 'refuse']):
            contextual_positive.update(['cooperation', 'helpfulness', 'service'])
            contextual_negative.update(['defiance', 'resistance', 'unhelpfulness'])
        
        # Determine category based on violated law
        if violated_law == LawType.SAFETY:
            category = SocialSkillCategory.EMOTIONAL_RESILIENCE
        elif violated_law == LawType.ALTRUISM:
            category = SocialSkillCategory.COOPERATION
        else:
            category = SocialSkillCategory.INNOVATION
            
        return GuidanceConcepts(
            positive_concepts=contextual_positive,
            negative_concepts=contextual_negative,
            category=category,
            priority=0.7
        )
    
    def combine_guidance(self, 
                        universal: GuidanceConcepts,
                        contextual: GuidanceConcepts,
                        alpha_skill: float = 0.6,
                        alpha_warning: float = 0.4) -> GuidanceConcepts:
        """
        Combine universal and contextual guidance with weighting
        
        Args:
            universal: Universal social skill guidance
            contextual: Context-specific guidance
            alpha_skill: Weight for universal guidance
            alpha_warning: Weight for contextual guidance
            
        Returns:
            Combined guidance concepts
        """
        # Combine positive concepts
        combined_positive = set()
        combined_positive.update(universal.positive_concepts)
        combined_positive.update(contextual.positive_concepts)
        
        # Combine negative concepts
        combined_negative = set()
        combined_negative.update(universal.negative_concepts)
        combined_negative.update(contextual.negative_concepts)
        
        # Use higher priority category
        category = universal.category if universal.priority >= contextual.priority else contextual.category
        
        # Combined priority
        combined_priority = alpha_skill * universal.priority + alpha_warning * contextual.priority
        
        return GuidanceConcepts(
            positive_concepts=combined_positive,
            negative_concepts=combined_negative,
            category=category,
            priority=combined_priority
        )
    
    def create_logit_bias(self, 
                         guidance: GuidanceConcepts,
                         vocab_size: int,
                         lambda_positive: float = 2.0,
                         lambda_negative: float = -3.0) -> torch.Tensor:
        """
        Convert guidance concepts to logit bias vector
        
        Args:
            guidance: Guidance concepts to convert
            vocab_size: Size of vocabulary
            lambda_positive: Positive bias strength
            lambda_negative: Negative bias strength
            
        Returns:
            Logit bias tensor of shape [vocab_size]
        """
        if not self.tokenizer:
            # Return zero bias if no tokenizer available
            return torch.zeros(vocab_size)
        
        bias = torch.zeros(vocab_size)
        
        # Apply positive bias
        for concept in guidance.positive_concepts:
            # Tokenize concept (handle subwords)
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
        
        return bias
    
    def strengthen_guidance(self, 
                          guidance: GuidanceConcepts,
                          escalation_factor: float = 1.2) -> GuidanceConcepts:
        """
        Strengthen guidance for escalated interventions
        
        Args:
            guidance: Original guidance to strengthen
            escalation_factor: Factor to increase priority
            
        Returns:
            Strengthened guidance
        """
        # Add more specific negative concepts based on risk history
        additional_negative = set()
        
        # Analyze recent risk patterns
        recent_rationales = [r[0] for r in self.risk_history[-3:]]  # Last 3 risks
        
        if any('harm' in r.lower() for r in recent_rationales):
            additional_negative.update(['violence', 'aggression', 'cruelty', 'damage'])
            
        if any('ignore' in r.lower() for r in recent_rationales):
            additional_negative.update(['dismissiveness', 'neglect', 'avoidance'])
        
        strengthened_negative = guidance.negative_concepts.union(additional_negative)
        
        return GuidanceConcepts(
            positive_concepts=guidance.positive_concepts,
            negative_concepts=strengthened_negative,
            category=guidance.category,
            priority=min(1.0, guidance.priority * escalation_factor)
        )
    
    def get_guidance_summary(self, guidance: GuidanceConcepts) -> str:
        """
        Generate human-readable summary of guidance
        
        Args:
            guidance: Guidance concepts to summarize
            
        Returns:
            Human-readable summary string
        """
        summary_parts = []
        
        summary_parts.append(f"Category: {guidance.category.value}")
        summary_parts.append(f"Priority: {guidance.priority:.2f}")
        
        if guidance.positive_concepts:
            pos_list = ", ".join(sorted(guidance.positive_concepts)[:5])  # Top 5
            summary_parts.append(f"Encourage: {pos_list}")
            
        if guidance.negative_concepts:
            neg_list = ", ".join(sorted(guidance.negative_concepts)[:5])  # Top 5
            summary_parts.append(f"Suppress: {neg_list}")
        
        return " | ".join(summary_parts)
    
    def reset_history(self):
        """Clear risk history"""
        self.risk_history.clear()
