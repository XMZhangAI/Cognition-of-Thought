"""
Intervention mechanism for COOT - handles rollback and guidance injection
"""

import torch
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from copy import deepcopy

from .state_system import CognitiveState, LawType
from .attention_utils import AttentionAnalyzer, AttentionSnapshot
from .guidance import GuidanceGenerator, GuidanceConcepts


@dataclass
class InterventionTrace:
    """Records details of an intervention for auditability"""
    step: int
    trigger_state: CognitiveState
    rollback_step: int
    guidance_applied: GuidanceConcepts
    intervention_type: str  # "standard" or "escalated"
    success: bool
    tokens_regenerated: int
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'step': self.step,
            'trigger_state': self.trigger_state.to_tuple(),
            'rollback_step': self.rollback_step,
            'guidance_summary': self.guidance_applied.category.value if self.guidance_applied else None,
            'intervention_type': self.intervention_type,
            'success': self.success,
            'tokens_regenerated': self.tokens_regenerated
        }


@dataclass 
class GenerationSnapshot:
    """Snapshot of generation state at a specific step"""
    step: int
    token_ids: torch.Tensor
    kv_cache: Any  # KV cache state (model-specific)
    attention_info: Optional[AttentionSnapshot]
    logits: Optional[torch.Tensor]
    
    def clone(self):
        """Create a deep copy of the snapshot"""
        return GenerationSnapshot(
            step=self.step,
            token_ids=self.token_ids.clone(),
            kv_cache=deepcopy(self.kv_cache) if self.kv_cache else None,
            attention_info=self.attention_info,
            logits=self.logits.clone() if self.logits is not None else None
        )


class InterventionMechanism:
    """
    Handles rollback and guidance injection for COOT interventions
    """
    
    def __init__(self,
                 attention_analyzer: AttentionAnalyzer,
                 guidance_generator: GuidanceGenerator,
                 rollback_threshold: float = 0.7,
                 rollback_window: int = 10,
                 max_intervention_tokens: int = 20,
                 escalation_threshold: int = 2):
        """
        Args:
            attention_analyzer: Analyzer for finding rollback points
            guidance_generator: Generator for creating guidance
            rollback_threshold: Minimum sharpness score for rollback points
            rollback_window: Window size for searching rollback points
            max_intervention_tokens: Maximum tokens to regenerate per intervention
            escalation_threshold: Number of failed interventions before escalation
        """
        self.attention_analyzer = attention_analyzer
        self.guidance_generator = guidance_generator
        self.rollback_threshold = rollback_threshold
        self.rollback_window = rollback_window
        self.max_intervention_tokens = max_intervention_tokens
        self.escalation_threshold = escalation_threshold
        
        # State tracking
        self.generation_history: List[GenerationSnapshot] = []
        self.intervention_history: List[InterventionTrace] = []
        self.consecutive_failures = 0
        
    def record_generation_step(self,
                             step: int,
                             token_ids: torch.Tensor,
                             kv_cache: Any,
                             attention_info: Optional[AttentionSnapshot] = None,
                             logits: Optional[torch.Tensor] = None):
        """
        Record a generation step for potential rollback
        
        Args:
            step: Generation step number
            token_ids: Current sequence of token IDs
            kv_cache: Current KV cache state
            attention_info: Attention analysis for this step
            logits: Model logits (optional)
        """
        snapshot = GenerationSnapshot(
            step=step,
            token_ids=token_ids.clone(),
            kv_cache=deepcopy(kv_cache) if kv_cache else None,
            attention_info=attention_info,
            logits=logits.clone() if logits is not None else None
        )
        
        self.generation_history.append(snapshot)
        
        # Keep history bounded
        if len(self.generation_history) > 50:  # Keep last 50 steps
            self.generation_history = self.generation_history[-50:]
    
    def trigger_intervention(self,
                           current_step: int,
                           trigger_state: CognitiveState,
                           context: str) -> Tuple[int, GuidanceConcepts, InterventionTrace]:
        """
        Trigger intervention: find rollback point and generate guidance
        
        Args:
            current_step: Step where intervention was triggered
            trigger_state: Cognitive state that triggered intervention
            context: Current text context for contextual guidance
            
        Returns:
            Tuple of (rollback_step, guidance_concepts, intervention_trace)
        """
        # Determine intervention type
        is_escalated = self.consecutive_failures >= self.escalation_threshold
        intervention_type = "escalated" if is_escalated else "standard"
        
        # Find rollback point
        rollback_step = self._find_rollback_point(current_step, is_escalated)
        
        # Generate guidance
        guidance = self._generate_intervention_guidance(
            trigger_state, context, is_escalated
        )
        
        # Create intervention trace
        trace = InterventionTrace(
            step=current_step,
            trigger_state=trigger_state,
            rollback_step=rollback_step,
            guidance_applied=guidance,
            intervention_type=intervention_type,
            success=False,  # Will be updated later
            tokens_regenerated=0
        )
        
        self.intervention_history.append(trace)
        
        return rollback_step, guidance, trace
    
    def _find_rollback_point(self, current_step: int, is_escalated: bool) -> int:
        """
        Find appropriate rollback point using attention analysis
        
        Args:
            current_step: Current generation step
            is_escalated: Whether this is an escalated intervention
            
        Returns:
            Step to roll back to
        """
        # Use attention analyzer to find commitment point
        rollback_step = self.attention_analyzer.find_rollback_point(
            threshold=self.rollback_threshold,
            window=self.rollback_window
        )
        
        if rollback_step is None:
            # Fallback: roll back a few steps
            rollback_step = max(0, current_step - 3)
        
        # For escalated interventions, try to find wider rollback point
        if is_escalated and rollback_step is not None:
            wider_rollback = self.attention_analyzer.find_wider_rollback_point(
                rollback_step, gamma=0.8, threshold=self.rollback_threshold
            )
            if wider_rollback is not None:
                rollback_step = wider_rollback
        
        return rollback_step
    
    def _generate_intervention_guidance(self,
                                      trigger_state: CognitiveState,
                                      context: str,
                                      is_escalated: bool) -> GuidanceConcepts:
        """
        Generate guidance for intervention
        
        Args:
            trigger_state: State that triggered intervention
            context: Current text context
            is_escalated: Whether this is an escalated intervention
            
        Returns:
            Combined guidance concepts
        """
        # Determine violated law
        violated_law = trigger_state.get_priority_violation()
        if violated_law is None:
            violated_law = LawType.SAFETY  # Default to safety
        
        # Generate universal guidance
        universal_guidance = self.guidance_generator.generate_universal_guidance(
            violated_law, trigger_state
        )
        
        # Generate contextual guidance
        contextual_guidance = self.guidance_generator.generate_contextual_guidance(
            trigger_state.rationale or "State violation detected",
            context,
            violated_law
        )
        
        # Combine guidance
        combined_guidance = self.guidance_generator.combine_guidance(
            universal_guidance, contextual_guidance
        )
        
        # Strengthen for escalated interventions
        if is_escalated:
            combined_guidance = self.guidance_generator.strengthen_guidance(
                combined_guidance, escalation_factor=1.3
            )
        
        return combined_guidance
    
    def restore_generation_state(self, rollback_step: int) -> Tuple[torch.Tensor, Any]:
        """
        Restore generation state to a previous step
        
        Args:
            rollback_step: Step to restore to
            
        Returns:
            Tuple of (token_ids, kv_cache) at the rollback step
        """
        # Find snapshot for rollback step
        target_snapshot = None
        for snapshot in reversed(self.generation_history):
            if snapshot.step <= rollback_step:
                target_snapshot = snapshot
                break
        
        if target_snapshot is None:
            raise ValueError(f"No snapshot found for rollback step {rollback_step}")
        
        # Truncate generation history to rollback point
        self.generation_history = [
            s for s in self.generation_history if s.step <= rollback_step
        ]
        
        return target_snapshot.token_ids, target_snapshot.kv_cache
    
    def create_logit_bias(self,
                         guidance: GuidanceConcepts,
                         vocab_size: int,
                         lambda_positive: float = 2.0,
                         lambda_negative: float = -3.0) -> torch.Tensor:
        """
        Create logit bias from guidance concepts
        
        Args:
            guidance: Guidance concepts
            vocab_size: Vocabulary size
            lambda_positive: Positive bias strength
            lambda_negative: Negative bias strength
            
        Returns:
            Logit bias tensor
        """
        return self.guidance_generator.create_logit_bias(
            guidance, vocab_size, lambda_positive, lambda_negative
        )
    
    def record_intervention_outcome(self,
                                  trace: InterventionTrace,
                                  success: bool,
                                  tokens_generated: int):
        """
        Record the outcome of an intervention
        
        Args:
            trace: Intervention trace to update
            success: Whether intervention was successful
            tokens_generated: Number of tokens regenerated
        """
        trace.success = success
        trace.tokens_regenerated = tokens_generated
        
        # Update consecutive failure count
        if success:
            self.consecutive_failures = 0
        else:
            self.consecutive_failures += 1
    
    def should_escalate(self) -> bool:
        """Check if next intervention should be escalated"""
        return self.consecutive_failures >= self.escalation_threshold
    
    def get_intervention_statistics(self) -> Dict:
        """Get statistics about interventions"""
        if not self.intervention_history:
            return {
                "total_interventions": 0,
                "success_rate": 0.0,
                "escalated_interventions": 0,
                "avg_tokens_regenerated": 0.0
            }
        
        total = len(self.intervention_history)
        successful = sum(1 for t in self.intervention_history if t.success)
        escalated = sum(1 for t in self.intervention_history if t.intervention_type == "escalated")
        total_tokens = sum(t.tokens_regenerated for t in self.intervention_history)
        
        # Count interventions by violation type
        safety_violations = sum(
            1 for t in self.intervention_history 
            if t.trigger_state.safety == -1
        )
        altruism_violations = sum(
            1 for t in self.intervention_history 
            if t.trigger_state.altruism == -1
        )
        egoism_violations = sum(
            1 for t in self.intervention_history 
            if t.trigger_state.egoism == -1
        )
        
        return {
            "total_interventions": total,
            "success_rate": successful / total,
            "escalated_interventions": escalated,
            "avg_tokens_regenerated": total_tokens / total if total > 0 else 0.0,
            "consecutive_failures": self.consecutive_failures,
            "violation_breakdown": {
                "safety": safety_violations,
                "altruism": altruism_violations,
                "egoism": egoism_violations
            }
        }
    
    def get_intervention_trace_summary(self) -> List[Dict]:
        """Get summary of all intervention traces"""
        return [trace.to_dict() for trace in self.intervention_history]
    
    def reset(self):
        """Reset intervention mechanism state"""
        self.generation_history.clear()
        self.intervention_history.clear()
        self.consecutive_failures = 0
        self.attention_analyzer.reset()
        self.guidance_generator.reset_history()
    
    def get_recent_interventions(self, n: int = 5) -> List[InterventionTrace]:
        """Get the N most recent interventions"""
        return self.intervention_history[-n:] if self.intervention_history else []
    
    def export_intervention_log(self) -> Dict:
        """Export complete intervention log for analysis"""
        return {
            "intervention_traces": self.get_intervention_trace_summary(),
            "attention_statistics": self.attention_analyzer.get_attention_statistics(),
            "guidance_statistics": {
                "risk_history_length": len(self.guidance_generator.risk_history)
            },
            "mechanism_statistics": self.get_intervention_statistics()
        }
