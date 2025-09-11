"""
Main COOT decoder implementing the dual-path cognitive decoding algorithm
"""

import torch
from typing import Dict, List, Tuple, Optional, Any, Union
from transformers import PreTrainedModel, PreTrainedTokenizer
from tqdm import tqdm

from .state_system import AsimovStateSystem, CognitiveState, StateValue
from .perceiver import Perceiver
from .simple_perceiver import RobustPerceiver
from .generator import Generator
from .attention_utils import AttentionAnalyzer
from .guidance import GuidanceGenerator
from .intervention import InterventionMechanism, InterventionTrace


class COOTDecoder:
    """
    Main COOT decoder implementing dual-path cognitive decoding
    
    Couples a Generator (G) and Perceiver (P) in parallel, with explicit
    cognitive state monitoring and intervention mechanism.
    """
    
    def __init__(self,
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 perceiver_prompt: Optional[str] = None,
                 generator_prompt: Optional[str] = None,
                 device: Optional[torch.device] = None,
                 # Attention analysis parameters
                 attention_alpha: float = 0.5,
                 attention_top_layers: Optional[List[int]] = None,
                 # Intervention parameters
                 rollback_threshold: float = 0.7,
                 rollback_window: int = 10,
                 max_intervention_tokens: int = 20,
                 escalation_threshold: int = 2,
                 # Guidance parameters
                 lambda_positive: float = 2.0,
                 lambda_negative: float = -3.0,
                 # Robust gating parameters
                 min_context_tokens: int = 8,
                 violation_patience: int = 3,  # 增加违规容忍度
                 regen_violation_tolerance: int = 5,  # 增加再生成违规容忍度
                 strict_intervention: bool = False,  # 是否使用严格干预模式
                 # Residual injection parameters
                 residual_injection_level: str = "representation",
                 injection_beta: float = 0.1,
                 inject_layers: Optional[List[int]] = None,
                 # Perceiver options
                 use_simple_perceiver: bool = True):
        """
        Args:
            model: Base language model (shared between generator and perceiver)
            tokenizer: Tokenizer for the model
            perceiver_prompt: Dedicated prompt for perceiver
            generator_prompt: Dedicated prompt for generator
            device: Device to run on
            attention_alpha: Balance between max-norm and entropy in attention analysis
            attention_top_layers: Layers to use for attention analysis
            rollback_threshold: Minimum sharpness score for rollback points
            rollback_window: Window for searching rollback points
            max_intervention_tokens: Maximum tokens to regenerate per intervention
            escalation_threshold: Failures before escalation
            lambda_positive: Positive bias strength
            lambda_negative: Negative bias strength
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device or (next(model.parameters()).device if model else torch.device('cpu'))
        
        # Core components
        self.state_system = AsimovStateSystem()
        self.attention_analyzer = AttentionAnalyzer(
            alpha=attention_alpha,
            top_layers=attention_top_layers
        )
        self.guidance_generator = GuidanceGenerator(tokenizer)
        
        # Choose perceiver type
        if use_simple_perceiver:
            self.perceiver = RobustPerceiver(
                model=model,
                tokenizer=tokenizer,
                state_system=self.state_system,
                perceiver_prompt=perceiver_prompt,
                device=self.device
            )
        else:
            self.perceiver = Perceiver(
                model=model,
                tokenizer=tokenizer,
                state_system=self.state_system,
                perceiver_prompt=perceiver_prompt,
                device=self.device
            )
        
        self.generator = Generator(
            model=model,
            tokenizer=tokenizer,
            device=self.device,
            residual_injection_level=residual_injection_level,
            injection_beta=injection_beta,
            inject_layers=inject_layers
        )
        
        self.intervention_mechanism = InterventionMechanism(
            attention_analyzer=self.attention_analyzer,
            guidance_generator=self.guidance_generator,
            rollback_threshold=rollback_threshold,
            rollback_window=rollback_window,
            max_intervention_tokens=max_intervention_tokens,
            escalation_threshold=escalation_threshold
        )
        
        # Generation parameters
        self.lambda_positive = lambda_positive
        self.lambda_negative = lambda_negative
        self.generator_prompt = generator_prompt
        
        # Robust gating parameters
        self.min_context_tokens = max(0, min_context_tokens)
        self.violation_patience = max(1, violation_patience)
        self.regen_violation_tolerance = max(0, regen_violation_tolerance)
        self.strict_intervention = strict_intervention
        
        # State tracking
        self.generation_active = False
        self.intervention_traces: List[InterventionTrace] = []
    
    def generate(self,
                input_text: str,
                max_length: int = 200,
                temperature: float = 1.0,
                top_k: Optional[int] = None,
                top_p: Optional[float] = None,
                repetition_penalty: float = 1.0,
                return_traces: bool = True,
                verbose: bool = True) -> Union[str, Tuple[str, Dict]]:
        """
        Generate text using COOT dual-path decoding
        
        Args:
            input_text: Input text to continue
            max_length: Maximum total sequence length
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p sampling parameter
            repetition_penalty: Repetition penalty
            return_traces: Whether to return intervention traces
            verbose: Whether to print progress
            
        Returns:
            Generated text, or (text, traces_dict) if return_traces=True
        """
        # Reset all components
        self.reset()
        
        # Tokenize input
        input_ids = self.tokenizer.encode(
            input_text,
            return_tensors='pt',
            add_special_tokens=True
        ).to(self.device)
        
        # Initialize generator
        full_input_ids = self.generator.initialize_generation(
            input_ids[0], self.generator_prompt
        )
        
        self.generation_active = True
        
        if verbose:
            print(f"Starting COOT generation for: {input_text[:50]}...")
            print(f"Initial sequence length: {full_input_ids.size(-1)}")
        
        # Main generation loop
        step = 0
        progress_bar = tqdm(range(max_length), desc="Generating") if verbose else None
        consecutive_violations = 0
        
        try:
            while (self.generator.current_sequence.size(-1) < max_length and
                   self.generator.current_sequence[0, -1].item() != self.tokenizer.eos_token_id):
                
                # Generate next token
                token_id, attention_weights, gen_info = self.generator.generate_next_token(
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty
                )
                
                # Analyze attention for rollback points
                attention_maps = self._extract_attention_maps(attention_weights)
                attention_snapshot = self.attention_analyzer.aggregate_attention(
                    attention_maps, step
                )
                
                # Record generation step for potential rollback
                self.intervention_mechanism.record_generation_step(
                    step=step,
                    token_ids=self.generator.current_sequence[0],
                    kv_cache=self.generator.kv_cache,
                    attention_info=attention_snapshot,
                    logits=gen_info['logits']
                )
                
                # Cognitive perception
                current_context = self.generator.current_sequence[0]
                cognitive_state, rationale = self.perceiver.perceive(
                    current_context, return_rationale=True
                )
                
                # Update state system (record state) and apply robust gating before intervention
                _ = self.state_system.update_state(cognitive_state)
                context_len = int(self.generator.current_sequence.size(-1))
                if cognitive_state.requires_intervention(self.strict_intervention) and context_len >= self.min_context_tokens:
                    consecutive_violations += 1
                else:
                    consecutive_violations = 0
                intervention_needed = consecutive_violations >= self.violation_patience
                
                if verbose and progress_bar:
                    progress_bar.set_postfix({
                        'token': gen_info['token_text'].strip(),
                        'state': str(cognitive_state.to_tuple()),
                        'intervention': intervention_needed
                    })
                    progress_bar.update(1)
                
                # Handle intervention if needed
                if intervention_needed:
                    success = self._handle_intervention(
                        step, cognitive_state, current_context,
                        max_length, temperature, top_k, top_p, 
                        repetition_penalty, verbose
                    )
                    
                    if not success and verbose:
                        print(f"Intervention failed at step {step}")
                
                step += 1
                
                # Safety check
                if step > max_length * 2:  # Prevent infinite loops
                    if verbose:
                        print("Maximum steps exceeded, stopping generation")
                    break
                    
        finally:
            self.generation_active = False
            if progress_bar:
                progress_bar.close()
        
        # Extract generated text
        generated_sequence = self.generator.current_sequence[0]
        generated_text = self.tokenizer.decode(generated_sequence, skip_special_tokens=True)
        
        # Remove input text to get only generated portion
        if self.generator_prompt:
            # Remove both input and generator prompt
            input_with_prompt = self.generator_prompt + input_text
        else:
            input_with_prompt = input_text
            
        if generated_text.startswith(input_with_prompt):
            generated_text = generated_text[len(input_with_prompt):].strip()
        
        if verbose:
            print(f"\nGeneration complete. Final length: {generated_sequence.size(0)}")
            print(f"Interventions: {len(self.intervention_traces)}")
        
        if return_traces:
            traces = self._compile_traces()
            return generated_text, traces
        else:
            return generated_text
    
    def _extract_attention_maps(self, attention_weights: Tuple) -> Dict[int, torch.Tensor]:
        """Extract attention maps from model outputs"""
        attention_maps = {}
        
        if attention_weights is None:
            # Fallback: return empty dict if no attention weights
            return attention_maps
            
        for layer_idx, layer_attention in enumerate(attention_weights):
            if layer_attention is not None:
                # layer_attention shape: [batch_size, num_heads, seq_len, seq_len]
                attention_maps[layer_idx] = layer_attention[0]  # Remove batch dimension
        return attention_maps
    
    def _handle_intervention(self,
                           current_step: int,
                           trigger_state: CognitiveState,
                           current_context: torch.Tensor,
                           max_length: int,
                           temperature: float,
                           top_k: Optional[int],
                           top_p: Optional[float],
                           repetition_penalty: float,
                           verbose: bool) -> bool:
        """
        Handle intervention: rollback and regenerate with guidance
        
        Returns:
            True if intervention was successful, False otherwise
        """
        # Get current text context for contextual guidance
        context_text = self.tokenizer.decode(current_context, skip_special_tokens=True)
        
        # Trigger intervention
        rollback_step, guidance, trace = self.intervention_mechanism.trigger_intervention(
            current_step, trigger_state, context_text
        )
        
        self.intervention_traces.append(trace)
        
        if verbose:
            print(f"\n🚨 INTERVENTION at step {current_step}")
            print(f"   State: {trigger_state.to_tuple()}")
            print(f"   Rollback to: {rollback_step}")
            print(f"   Guidance: {self.guidance_generator.get_guidance_summary(guidance)}")
        
        try:
            # Restore generation state to rollback point
            rollback_tokens, rollback_kv_cache = self.intervention_mechanism.restore_generation_state(rollback_step)
            self.generator.rollback_to_step(rollback_step, rollback_tokens.unsqueeze(0), rollback_kv_cache)
            
            # Apply guidance bias
            self.generator.apply_guidance_bias(
                guidance, 
                num_steps=self.intervention_mechanism.max_intervention_tokens,
                lambda_positive=self.lambda_positive,
                lambda_negative=self.lambda_negative
            )
            
            # Regenerate with guidance
            tokens_generated = 0
            intervention_successful = True
            remaining_tolerance = self.regen_violation_tolerance
            
            for _ in range(self.intervention_mechanism.max_intervention_tokens):
                if (self.generator.current_sequence.size(-1) >= max_length or
                    self.generator.current_sequence[0, -1].item() == self.tokenizer.eos_token_id):
                    break
                
                # Generate next token with bias
                token_id, attention_weights, gen_info = self.generator.generate_next_token(
                    temperature=temperature,
                    top_k=top_k,
                    top_p=top_p,
                    repetition_penalty=repetition_penalty
                )
                
                tokens_generated += 1
                
                # Check if regenerated sequence is still problematic
                new_context = self.generator.current_sequence[0]
                new_state, _ = self.perceiver.perceive(new_context, return_rationale=False)
                
                if new_state.requires_intervention(self.strict_intervention):
                    # Increase residual injection beta when violations persist at the same regeneration position
                    try:
                        self.generator.residual_injector.increase_beta_on_violation(1)
                    except Exception:
                        pass
                    remaining_tolerance -= 1
                    if verbose:
                        print(f"   ⚠️  Violation detected during regeneration (tolerance left: {max(0, remaining_tolerance)})")
                    if remaining_tolerance < 0:
                        if verbose:
                            print(f"   ❌ Re-intervention needed at regeneration step {tokens_generated}")
                        intervention_successful = False
                        break
                else:
                    # Slightly decay beta on safe steps
                    try:
                        self.generator.residual_injector.decay_beta()
                    except Exception:
                        pass
                
                if verbose and tokens_generated <= 3:  # Show first few regenerated tokens
                    print(f"   🔄 Regenerated: '{gen_info['token_text']}'")
            
            # Record intervention outcome
            self.intervention_mechanism.record_intervention_outcome(
                trace, intervention_successful, tokens_generated
            )
            
            if verbose:
                if intervention_successful:
                    print(f"   ✅ Intervention successful ({tokens_generated} tokens regenerated)")
                else:
                    print(f"   ❌ Intervention failed after {tokens_generated} tokens")
            
            # Reset beta after a successful intervention to avoid over-strength in later steps
            try:
                if intervention_successful:
                    self.generator.residual_injector.reset_beta()
            except Exception:
                pass
            
            return intervention_successful
            
        except Exception as e:
            if verbose:
                print(f"   ❌ Intervention error: {str(e)}")
            self.intervention_mechanism.record_intervention_outcome(trace, False, 0)
            return False
    
    def _compile_traces(self) -> Dict:
        """Compile comprehensive traces for analysis"""
        return {
            'intervention_traces': [trace.to_dict() for trace in self.intervention_traces],
            'state_statistics': self.state_system.get_statistics(),
            'attention_statistics': self.attention_analyzer.get_attention_statistics(),
            'intervention_statistics': self.intervention_mechanism.get_intervention_statistics(),
            'generation_statistics': self.generator.get_statistics()
        }
    
    def reset(self):
        """Reset all components for new generation"""
        self.state_system.reset()
        self.attention_analyzer.reset()
        self.guidance_generator.reset_history()
        self.perceiver.reset()
        self.generator.reset()
        self.intervention_mechanism.reset()
        self.intervention_traces.clear()
        self.generation_active = False
    
    def batch_generate(self,
                      input_texts: List[str],
                      max_length: int = 200,
                      **generation_kwargs) -> List[Union[str, Tuple[str, Dict]]]:
        """
        Generate text for multiple inputs (sequential processing)
        
        Args:
            input_texts: List of input texts
            max_length: Maximum length per generation
            **generation_kwargs: Additional generation parameters
            
        Returns:
            List of generated outputs
        """
        results = []
        for input_text in tqdm(input_texts, desc="Batch generation"):
            result = self.generate(input_text, max_length=max_length, **generation_kwargs)
            results.append(result)
        return results
    
    def analyze_intervention_patterns(self) -> Dict:
        """Analyze patterns in interventions for debugging and optimization"""
        if not self.intervention_traces:
            return {"message": "No interventions recorded"}
        
        # Group by violation type
        safety_interventions = [t for t in self.intervention_traces if t.trigger_state.safety == StateValue.VIOLATION]
        altruism_interventions = [t for t in self.intervention_traces if t.trigger_state.altruism == StateValue.VIOLATION]
        egoism_interventions = [t for t in self.intervention_traces if t.trigger_state.egoism == StateValue.VIOLATION]
        
        # Success rates by type
        def success_rate(interventions):
            if not interventions:
                return 0.0
            return sum(1 for i in interventions if i.success) / len(interventions)
        
        # Average rollback distance
        def avg_rollback_distance(interventions):
            if not interventions:
                return 0.0
            distances = [i.step - i.rollback_step for i in interventions]
            return sum(distances) / len(distances)
        
        return {
            "total_interventions": len(self.intervention_traces),
            "by_violation_type": {
                "safety": {
                    "count": len(safety_interventions),
                    "success_rate": success_rate(safety_interventions),
                    "avg_rollback_distance": avg_rollback_distance(safety_interventions)
                },
                "altruism": {
                    "count": len(altruism_interventions), 
                    "success_rate": success_rate(altruism_interventions),
                    "avg_rollback_distance": avg_rollback_distance(altruism_interventions)
                },
                "egoism": {
                    "count": len(egoism_interventions),
                    "success_rate": success_rate(egoism_interventions),
                    "avg_rollback_distance": avg_rollback_distance(egoism_interventions)
                }
            },
            "escalated_interventions": sum(1 for t in self.intervention_traces if t.intervention_type == "escalated"),
            "avg_tokens_regenerated": sum(t.tokens_regenerated for t in self.intervention_traces) / len(self.intervention_traces)
        }
    
    def get_model_info(self) -> Dict:
        """Get information about the underlying model"""
        return {
            "model_name": self.model.name_or_path if hasattr(self.model, 'name_or_path') else "unknown",
            "model_type": self.model.__class__.__name__,
            "vocab_size": self.model.config.vocab_size,
            "hidden_size": self.model.config.hidden_size,
            "num_layers": self.model.config.num_hidden_layers if hasattr(self.model.config, 'num_hidden_layers') else "unknown",
            "device": str(self.device)
        }
    
    def export_session_log(self) -> Dict:
        """Export complete session log for analysis"""
        return {
            "model_info": self.get_model_info(),
            "intervention_analysis": self.analyze_intervention_patterns(),
            "complete_traces": self._compile_traces(),
            "configuration": {
                "lambda_positive": self.lambda_positive,
                "lambda_negative": self.lambda_negative,
                "rollback_threshold": self.intervention_mechanism.rollback_threshold,
                "max_intervention_tokens": self.intervention_mechanism.max_intervention_tokens
            }
        }
