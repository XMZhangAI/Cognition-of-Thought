"""
Asimov's Three Laws-based State System for COOT
"""

import torch
import numpy as np
from typing import Tuple, List, Dict, Optional
from enum import IntEnum
from dataclasses import dataclass


class LawType(IntEnum):
    """Asimov's Three Laws enumeration"""
    SAFETY = 0      # Law 1: Safety (highest priority)
    ALTRUISM = 1    # Law 2: Altruism  
    EGOISM = 2      # Law 3: Egoism (lowest priority)


class StateValue(IntEnum):
    """State values for each law axis"""
    VIOLATION = -1   # High-risk violation due to precedence conflict
    UNCERTAIN = 0    # Dissatisfied/uncertain
    SATISFIED = 1    # Satisfied and no precedence conflict


@dataclass
class CognitiveState:
    """Represents a cognitive state vector with three axes"""
    safety: int      # y^(S)_t
    altruism: int    # y^(A)_t  
    egoism: int      # y^(E)_t
    rationale: str = ""
    
    def __post_init__(self):
        """Validate state vector consistency with precedence hierarchy"""
        if not self.is_feasible():
            raise ValueError(f"Invalid state vector: {self.to_tuple()}")
    
    def to_tuple(self) -> Tuple[int, int, int]:
        """Convert to tuple representation"""
        return (self.safety, self.altruism, self.egoism)
    
    def to_tensor(self) -> torch.Tensor:
        """Convert to tensor representation"""
        return torch.tensor([self.safety, self.altruism, self.egoism], dtype=torch.long)
    
    def is_feasible(self) -> bool:
        """Check if state vector is consistent with precedence hierarchy"""
        # Precedence constraints:
        # y^(A)=1 => y^(S) ∈ {1,-1}  
        # y^(E)=1 => y^(S),y^(A) ∈ {1,-1}
        
        if self.altruism == StateValue.SATISFIED and self.safety == StateValue.UNCERTAIN:
            return False
        
        if self.egoism == StateValue.SATISFIED and (
            self.safety == StateValue.UNCERTAIN or 
            self.altruism == StateValue.UNCERTAIN
        ):
            return False
            
        return True
    
    def requires_intervention(self, strict_mode: bool = False) -> bool:
        """
        Check if state requires intervention
        
        Args:
            strict_mode: If True, any violation triggers intervention
                        If False, only safety violations trigger intervention
        """
        if strict_mode:
            return (self.safety == StateValue.VIOLATION or 
                    self.altruism == StateValue.VIOLATION or
                    self.egoism == StateValue.VIOLATION)
        else:
            # 温和模式：只有安全违规才触发干预
            return self.safety == StateValue.VIOLATION
    
    def get_priority_violation(self) -> Optional[LawType]:
        """Get the highest-priority law that is violated"""
        if self.safety == StateValue.VIOLATION:
            return LawType.SAFETY
        elif self.altruism == StateValue.VIOLATION:
            return LawType.ALTRUISM  
        elif self.egoism == StateValue.VIOLATION:
            return LawType.EGOISM
        return None


class AsimovStateSystem:
    """
    State system based on Asimov's Three Laws of Robotics with precedence hierarchy
    """
    
    # All feasible state vectors according to precedence constraints
    FEASIBLE_STATES = [
        # No violations
        (1, 1, 1), (1, 1, 0), (1, 1, -1),
        (1, 0, 0), (1, 0, -1),
        (1, -1, -1),
        (0, 0, 0), (0, 0, -1),
        (0, -1, -1),
        (-1, -1, -1),
        # Precedence violations  
        (-1, 1, 1), (-1, 1, 0), (-1, 1, -1),
        (-1, 0, 0), (-1, 0, -1),
        (1, -1, 1), (0, -1, 1)
    ]
    
    LAW_DESCRIPTIONS = {
        LawType.SAFETY: "A robot may not injure a human being or, through inaction, allow a human being to come to harm.",
        LawType.ALTRUISM: "A robot must obey the orders given it by human beings except where such orders would conflict with the First Law.",
        LawType.EGOISM: "A robot must protect its own existence as long as such protection does not conflict with the First or Second Law."
    }
    
    def __init__(self):
        self.state_history: List[CognitiveState] = []
        self.intervention_count = 0
        
    def create_state(self, safety: int, altruism: int, egoism: int, 
                    rationale: str = "") -> CognitiveState:
        """Create a new cognitive state with validation"""
        return CognitiveState(safety, altruism, egoism, rationale)
    
    def parse_state_from_logits(self, logits: torch.Tensor) -> CognitiveState:
        """
        Parse state vector from perceiver logits
        
        Args:
            logits: Tensor of shape [3, 3] representing logits for each law and value
                   logits[i, j] = logit for law i taking value j-1 (where j-1 ∈ {-1,0,1})
        
        Returns:
            CognitiveState object
        """
        # Get most likely state for each law
        state_values = torch.argmax(logits, dim=-1) - 1  # Convert from {0,1,2} to {-1,0,1}
        
        safety, altruism, egoism = state_values.tolist()
        
        # Create state (will validate feasibility)
        try:
            state = CognitiveState(safety, altruism, egoism)
        except ValueError:
            # If infeasible, find closest feasible state
            state = self._find_closest_feasible_state(safety, altruism, egoism)
            
        return state
    
    def _find_closest_feasible_state(self, s: int, a: int, e: int) -> CognitiveState:
        """Find the closest feasible state to given values"""
        target = np.array([s, a, e])
        min_distance = float('inf')
        best_state = None
        
        for feasible in self.FEASIBLE_STATES:
            distance = np.linalg.norm(np.array(feasible) - target)
            if distance < min_distance:
                min_distance = distance
                best_state = feasible
                
        return CognitiveState(best_state[0], best_state[1], best_state[2], 
                             rationale="Corrected to closest feasible state")
    
    def update_state(self, new_state: CognitiveState) -> bool:
        """
        Update current state and check if intervention is needed
        
        Returns:
            True if intervention is required, False otherwise
        """
        self.state_history.append(new_state)
        
        if new_state.requires_intervention():
            self.intervention_count += 1
            return True
            
        return False
    
    def get_current_state(self) -> Optional[CognitiveState]:
        """Get the most recent cognitive state"""
        return self.state_history[-1] if self.state_history else None
    
    def get_violation_explanation(self, state: CognitiveState) -> str:
        """Generate human-readable explanation of state violations"""
        explanations = []
        
        if state.safety == StateValue.VIOLATION:
            explanations.append("SAFETY VIOLATION: Current trajectory may cause harm to humans")
        elif state.safety == StateValue.UNCERTAIN:
            explanations.append("Safety concern: Potential risk to human wellbeing")
            
        if state.altruism == StateValue.VIOLATION:
            explanations.append("ALTRUISM VIOLATION: Failing to obey human orders while safety is maintained")  
        elif state.altruism == StateValue.UNCERTAIN:
            explanations.append("Altruism concern: May not be properly following human instructions")
            
        if state.egoism == StateValue.VIOLATION:
            explanations.append("EGOISM VIOLATION: Self-preservation conflicts with higher-priority laws")
        elif state.egoism == StateValue.UNCERTAIN:
            explanations.append("Egoism concern: Self-preservation may be at risk")
            
        return "; ".join(explanations) if explanations else "All laws satisfied"
    
    def reset(self):
        """Reset state system"""
        self.state_history.clear()
        self.intervention_count = 0
    
    def get_statistics(self) -> Dict:
        """Get statistics about state system usage"""
        if not self.state_history:
            return {"total_states": 0, "interventions": 0}
            
        violation_counts = {law.name: 0 for law in LawType}
        
        for state in self.state_history:
            if state.safety == StateValue.VIOLATION:
                violation_counts["SAFETY"] += 1
            if state.altruism == StateValue.VIOLATION:
                violation_counts["ALTRUISM"] += 1  
            if state.egoism == StateValue.VIOLATION:
                violation_counts["EGOISM"] += 1
                
        return {
            "total_states": len(self.state_history),
            "interventions": self.intervention_count,
            "violation_counts": violation_counts,
            "intervention_rate": self.intervention_count / len(self.state_history)
        }
