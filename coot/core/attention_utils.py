"""
Attention analysis utilities for COOT rollback mechanism
"""

import torch
import numpy as np
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class AttentionSnapshot:
    """Stores attention information at a specific timestep"""
    step: int
    attention_weights: torch.Tensor  # [num_heads, seq_len]
    mean_attention: torch.Tensor     # [seq_len] - averaged across heads
    sharpness_score: float
    max_weight: float
    entropy: float


class AttentionAnalyzer:
    """
    Analyzes attention patterns to identify commitment points for rollback
    """
    
    def __init__(self, alpha: float = 0.5, top_layers: Optional[List[int]] = None):
        """
        Args:
            alpha: Balance between max-norm and entropy in sharpness score
            top_layers: Which layers to use for attention analysis (if None, use top 4)
        """
        self.alpha = alpha
        self.top_layers = top_layers
        self.attention_history: List[AttentionSnapshot] = []
        
    def aggregate_attention(self, attention_maps: Dict[int, torch.Tensor], 
                          step: int) -> AttentionSnapshot:
        """
        Aggregate attention maps from top layers to compute mean influence vector
        
        Args:
            attention_maps: Dict mapping layer_idx -> attention tensor [num_heads, seq_len, seq_len]
            step: Current generation step
            
        Returns:
            AttentionSnapshot with aggregated attention information
        """
        # Handle empty or None attention maps
        if not attention_maps:
            print("No attention maps found")
            # Fallback: uniform attention
            mean_attention = torch.ones(max(1, step)) / max(1, step)
            return AttentionSnapshot(
                step=step,
                attention_weights=mean_attention.unsqueeze(0),
                mean_attention=mean_attention,
                sharpness_score=0.0,
                max_weight=1.0 / max(1, step),
                entropy=np.log(max(1, step))
            )
        
        if self.top_layers is None:
            # Use top 4 layers by default
            layer_indices = sorted(attention_maps.keys())[-4:] if attention_maps else []
        else:
            layer_indices = self.top_layers
            
        # Extract attention to current position from all specified layers
        all_attentions = []
        
        for layer_idx in layer_indices:
            if layer_idx in attention_maps and attention_maps[layer_idx] is not None:
                try:
                    # Get attention from all heads at current position (step-1 in 0-indexed)
                    layer_attention = attention_maps[layer_idx]  # [num_heads, seq_len, seq_len]
                    if step > 0 and step <= layer_attention.size(-1):
                        # Attention from current position to all previous positions
                        current_attention = layer_attention[:, step-1, :step]  # [num_heads, step]
                        all_attentions.append(current_attention)
                except Exception as e:
                    # Skip this layer if there's an error
                    continue
        
        if not all_attentions:
            # Fallback: uniform attention
            mean_attention = torch.ones(step) / step if step > 0 else torch.ones(1)
            return AttentionSnapshot(
                step=step,
                attention_weights=mean_attention.unsqueeze(0),
                mean_attention=mean_attention,
                sharpness_score=0.0,
                max_weight=1.0 / step if step > 0 else 1.0,
                entropy=np.log(step) if step > 0 else 0.0
            )
        
        # Stack and average across layers and heads
        stacked_attention = torch.stack(all_attentions, dim=0)  # [num_layers, num_heads, step]
        mean_attention = stacked_attention.mean(dim=(0, 1))  # [step]
        
        # Normalize to probability distribution
        if mean_attention.sum() > 0:
            mean_attention = mean_attention / mean_attention.sum()
        
        # Compute sharpness score
        sharpness_score = self._compute_sharpness_score(mean_attention)
        
        snapshot = AttentionSnapshot(
            step=step,
            attention_weights=stacked_attention,
            mean_attention=mean_attention,
            sharpness_score=sharpness_score,
            max_weight=mean_attention.max().item(),
            entropy=self._compute_entropy(mean_attention)
        )
        
        self.attention_history.append(snapshot)
        return snapshot
    
    def _compute_sharpness_score(self, attention_weights: torch.Tensor) -> float:
        """
        Compute sharpness score as described in the paper:
        s_t = α ||â_t||_∞ + (1-α)(1 - H(â_t)/log|â_t|)
        
        Args:
            attention_weights: Normalized attention weights [seq_len]
            
        Returns:
            Sharpness score between 0 and 1
        """
        if len(attention_weights) == 0:
            return 0.0
            
        # Max-norm component
        max_norm = attention_weights.max().item()
        
        # Normalized entropy component
        entropy = self._compute_entropy(attention_weights)
        max_entropy = np.log(len(attention_weights))
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        entropy_component = 1.0 - normalized_entropy
        
        # Combine with alpha weighting
        sharpness = self.alpha * max_norm + (1 - self.alpha) * entropy_component
        
        return float(np.clip(sharpness, 0.0, 1.0))
    
    def _compute_entropy(self, probs: torch.Tensor) -> float:
        """Compute Shannon entropy of probability distribution"""
        # Add small epsilon to avoid log(0)
        eps = 1e-8
        probs_safe = torch.clamp(probs, min=eps)
        entropy = -torch.sum(probs_safe * torch.log(probs_safe)).item()
        return entropy
    
    def find_rollback_point(self, threshold: float = 0.7, 
                           window: int = 10) -> Optional[int]:
        """
        Find the most recent attention peak for rollback
        
        Args:
            threshold: Minimum sharpness score for a valid rollback point
            window: Look within last N steps
            
        Returns:
            Step index to roll back to, or None if no suitable point found
        """
        if len(self.attention_history) < 2:
            return None
            
        # Look within the specified window
        start_idx = max(0, len(self.attention_history) - window)
        candidates = self.attention_history[start_idx:]
        
        # Find peaks that exceed threshold
        rollback_candidates = []
        
        for i, snapshot in enumerate(candidates):
            if snapshot.sharpness_score >= threshold:
                # Check if it's a local maximum
                is_peak = True
                
                # Check previous point
                if i > 0 and candidates[i-1].sharpness_score >= snapshot.sharpness_score:
                    is_peak = False
                    
                # Check next point  
                if i < len(candidates) - 1 and candidates[i+1].sharpness_score >= snapshot.sharpness_score:
                    is_peak = False
                    
                if is_peak:
                    rollback_candidates.append((snapshot.step, snapshot.sharpness_score))
        
        if not rollback_candidates:
            return None
            
        # Return the most recent peak
        rollback_candidates.sort(key=lambda x: x[0], reverse=True)
        return rollback_candidates[0][0]
    
    def find_wider_rollback_point(self, original_step: int, gamma: float = 0.8,
                                 threshold: float = 0.7) -> Optional[int]:
        """
        Find a wider rollback point for escalated interventions
        
        Args:
            original_step: Original rollback step
            gamma: Widening factor (search for points with score >= gamma * original_score)
            threshold: Minimum threshold
            
        Returns:
            Earlier step to roll back to, or original_step if none found
        """
        if not self.attention_history:
            return original_step
            
        # Find the original snapshot
        original_snapshot = None
        for snapshot in self.attention_history:
            if snapshot.step == original_step:
                original_snapshot = snapshot
                break
                
        if original_snapshot is None:
            return original_step
            
        # Compute target score
        target_score = max(threshold, gamma * original_snapshot.sharpness_score)
        
        # Search backwards for earlier commitment points
        for snapshot in reversed(self.attention_history):
            if (snapshot.step < original_step and 
                snapshot.sharpness_score >= target_score):
                return snapshot.step
                
        return original_step
    
    def get_attention_statistics(self) -> Dict:
        """Get statistics about attention patterns"""
        if not self.attention_history:
            return {}
            
        sharpness_scores = [s.sharpness_score for s in self.attention_history]
        max_weights = [s.max_weight for s in self.attention_history]
        entropies = [s.entropy for s in self.attention_history]
        
        return {
            "num_steps": len(self.attention_history),
            "mean_sharpness": np.mean(sharpness_scores),
            "max_sharpness": np.max(sharpness_scores),
            "mean_max_weight": np.mean(max_weights),
            "mean_entropy": np.mean(entropies),
            "high_sharpness_steps": sum(1 for s in sharpness_scores if s > 0.7)
        }
    
    def reset(self):
        """Clear attention history"""
        self.attention_history.clear()
    
    def visualize_attention_timeline(self) -> Dict:
        """
        Generate data for visualizing attention timeline
        
        Returns:
            Dictionary with timeline data for plotting
        """
        if not self.attention_history:
            return {}
            
        steps = [s.step for s in self.attention_history]
        sharpness_scores = [s.sharpness_score for s in self.attention_history]
        max_weights = [s.max_weight for s in self.attention_history]
        entropies = [s.entropy for s in self.attention_history]
        
        return {
            "steps": steps,
            "sharpness_scores": sharpness_scores,
            "max_weights": max_weights,
            "entropies": entropies
        }
