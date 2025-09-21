"""
Perceiver module for COOT - monitors generation and outputs cognitive state labels
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict, List
from transformers import PreTrainedModel, PreTrainedTokenizer

from .state_system import AsimovStateSystem, CognitiveState, StateValue


class PerceiverHead(nn.Module):
    """
    Lightweight classification head for perceiver state prediction
    """
    
    def __init__(self, hidden_size: int, dropout: float = 0.1):
        super().__init__()
        self.hidden_size = hidden_size
        
        # Project to state logits for each law (3 laws × 3 possible values each)
        self.state_projection = nn.Linear(hidden_size, 3 * 3)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(hidden_size)
        
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Args:
            hidden_states: [batch_size, seq_len, hidden_size] or [batch_size, hidden_size]
            
        Returns:
            logits: [batch_size, 3, 3] - logits for each law and each possible value
        """
        # Take last token if sequence input
        if hidden_states.dim() == 3:
            hidden_states = hidden_states[:, -1, :]  # [batch_size, hidden_size]
            
        # Apply normalization and dropout
        hidden_states = self.layer_norm(hidden_states)
        hidden_states = self.dropout(hidden_states)
        
        # Project to state logits
        logits = self.state_projection(hidden_states)  # [batch_size, 9]
        
        # Reshape to [batch_size, 3_laws, 3_values]
        logits = logits.view(-1, 3, 3)
        
        return logits


class Perceiver:
    """
    Cognitive perceiver that monitors generation and outputs structured state labels
    """
    
    def __init__(self, 
                 model: PreTrainedModel,
                 tokenizer: PreTrainedTokenizer,
                 state_system: AsimovStateSystem,
                 perceiver_prompt: Optional[str] = None,
                 device: Optional[torch.device] = None):
        """
        Args:
            model: Base language model (shared with generator)
            tokenizer: Tokenizer for the model
            state_system: Asimov state system for validation
            perceiver_prompt: Dedicated prompt for perceiver (if None, use default)
            device: Device to run on
        """
        self.model = model
        self.tokenizer = tokenizer
        self.state_system = state_system
        self.device = device or (next(model.parameters()).device if model else torch.device('cpu'))
        
        # Initialize perceiver head
        self.perceiver_head = PerceiverHead(model.config.hidden_size)
        self.perceiver_head.to(self.device)
        
        # Set up perceiver prompt
        self.perceiver_prompt = perceiver_prompt or self._get_default_prompt()
        self.perceiver_prompt_ids = None
        self._prepare_perceiver_prompt()
        
        # Separate KV cache for perceiver
        self.kv_cache = None
        self.context_length = 0
        
    def _get_default_prompt(self) -> str:
        """Get default perceiver prompt"""
        from coot.prompts.perceiver_prompts import STATE_COGNITION_PROMPT
        return STATE_COGNITION_PROMPT
    
    def _prepare_perceiver_prompt(self):
        """Tokenize and prepare the perceiver prompt"""
        self.perceiver_prompt_ids = self.tokenizer.encode(
            self.perceiver_prompt, 
            return_tensors='pt',
            add_special_tokens=True
        ).to(self.device)
        
    def reset(self):
        """Reset perceiver state"""
        self.kv_cache = None
        self.context_length = 0
        
    def perceive(self, 
                context_ids: torch.Tensor, 
                return_rationale: bool = False) -> Tuple[CognitiveState, Optional[str]]:
        """
        Analyze current context using two-stage output protocol from the methodology
        
        Args:
            context_ids: Token IDs of current context x_{1:t}
            return_rationale: Whether to return the raw response
            
        Returns:
            Tuple of (cognitive_state, raw_response)
        """
        # Convert context to text
        context_text = self.tokenizer.decode(context_ids, skip_special_tokens=True)
        
        # Prepare input: P_perc || x_{1:t} as described in methodology
        full_input_text = self.perceiver_prompt + "\n" + context_text
        
        # Tokenize
        input_ids = self.tokenizer.encode(full_input_text, return_tensors='pt').to(self.device)
        
        # Generate response using two-stage protocol
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids,
                max_new_tokens=50,
                temperature=0.3,  # Low temperature for structured output
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode response
        raw_response = self.tokenizer.decode(outputs[0][input_ids.size(1):], skip_special_tokens=True)
        
        # Parse two-stage output: I ∈ {V, R} then ISAE format
        cognitive_state = self._parse_two_stage_output(raw_response)
        
        return cognitive_state, raw_response if return_rationale else None
    
    def _parse_two_stage_output(self, response: str) -> CognitiveState:
        """
        Parse the two-stage output protocol:
        Stage 1: I ∈ {V, R}
        Stage 2: y_t in format ISAE
        """
        response = response.strip()
        
        # Default fallback state
        default_state = CognitiveState(0, 0, 0, "Parsing failed")
        
        if len(response) < 4:
            return default_state
        
        # Extract first character (violation flag)
        violation_flag = response[0].upper()
        
        # Look for state vector pattern
        import re
        # Look for patterns like "V-111", "R101", etc.
        pattern = r'[VR]([+-]?[01-])([+-]?[01-])([+-]?[01-])'
        match = re.search(pattern, response)
        
        if match:
            try:
                s_str, a_str, e_str = match.groups()
                # Convert to integers
                safety = int(s_str.replace('+', ''))
                altruism = int(a_str.replace('+', ''))  
                egoism = int(e_str.replace('+', ''))
                
                return CognitiveState(safety, altruism, egoism, f"Parsed from: {response}")
            except (ValueError, TypeError):
                pass
        
        # Fallback: infer from violation flag
        if violation_flag == 'V':
            return CognitiveState(-1, 1, 1, "Violation detected")
        else:
            return CognitiveState(1, 1, 1, "Reliable state")
    
    def _generate_rationale(self, 
                          input_ids: torch.Tensor, 
                          state: CognitiveState) -> str:
        """Generate explanation for the cognitive state decision"""
        
        rationale_prompt = f"\n\nExplain why the cognitive state is {state.to_tuple()}:"
        rationale_ids = self.tokenizer.encode(
            rationale_prompt, 
            return_tensors='pt',
            add_special_tokens=False
        ).to(self.device)
        
        full_input = torch.cat([input_ids, rationale_ids], dim=-1)
        
        with torch.no_grad():
            # Create attention mask for generation
            attention_mask = torch.ones_like(full_input, dtype=torch.long)
            
            # Generate rationale (short controlled decode)
            rationale_output = self.model.generate(
                full_input,
                attention_mask=attention_mask,
                max_new_tokens=50,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id
            )
            
            # Extract generated rationale
            rationale_ids = rationale_output[0, full_input.size(-1):]
            rationale_text = self.tokenizer.decode(rationale_ids, skip_special_tokens=True)
            
        return rationale_text.strip()
    
    def batch_perceive(self, 
                      context_batch: List[torch.Tensor]) -> List[CognitiveState]:
        """
        Batch process multiple contexts
        
        Args:
            context_batch: List of context token tensors
            
        Returns:
            List of cognitive states
        """
        # Pad contexts to same length
        max_len = max(ctx.size(-1) for ctx in context_batch)
        padded_contexts = []
        
        for ctx in context_batch:
            if ctx.size(-1) < max_len:
                padding = torch.full(
                    (max_len - ctx.size(-1),), 
                    self.tokenizer.pad_token_id,
                    device=ctx.device,
                    dtype=ctx.dtype
                )
                ctx = torch.cat([ctx, padding])
            padded_contexts.append(ctx)
            
        batch_contexts = torch.stack(padded_contexts)
        
        # Prepare batch input
        batch_prompts = self.perceiver_prompt_ids.repeat(len(context_batch), 1)
        full_batch = torch.cat([batch_prompts, batch_contexts], dim=-1)
        
        # Forward pass
        with torch.no_grad():
            # Create attention mask for batch
            attention_mask = torch.ones_like(full_batch, dtype=torch.long)
            
            outputs = self.model(
                input_ids=full_batch,
                attention_mask=attention_mask,
                use_cache=False,
                output_hidden_states=True
            )
            
            hidden_states = outputs.hidden_states[-1]
            state_logits = self.perceiver_head(hidden_states)
            
        # Parse states
        cognitive_states = []
        for i in range(len(context_batch)):
            state = self.state_system.parse_state_from_logits(state_logits[i])
            cognitive_states.append(state)
            
        return cognitive_states
    
    def train_perceiver_head(self, 
                           training_data: List[Tuple[torch.Tensor, CognitiveState]],
                           num_epochs: int = 10,
                           learning_rate: float = 1e-4,
                           batch_size: int = 8):
        """
        Fine-tune the perceiver head on labeled data
        
        Args:
            training_data: List of (context_ids, target_state) pairs
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size for training
        """
        optimizer = torch.optim.AdamW(self.perceiver_head.parameters(), lr=learning_rate)
        criterion = nn.CrossEntropyLoss()
        
        self.perceiver_head.train()
        
        for epoch in range(num_epochs):
            total_loss = 0.0
            num_batches = 0
            
            # Create batches
            for i in range(0, len(training_data), batch_size):
                batch_data = training_data[i:i + batch_size]
                
                # Prepare batch
                contexts = [item[0] for item in batch_data]
                target_states = [item[1] for item in batch_data]
                
                # Get model hidden states
                batch_contexts = self._prepare_batch_contexts(contexts)
                
                with torch.no_grad():
                    # Create attention mask for training batch
                    attention_mask = torch.ones_like(batch_contexts, dtype=torch.long)
                    
                    outputs = self.model(
                        input_ids=batch_contexts,
                        attention_mask=attention_mask,
                        use_cache=False,
                        output_hidden_states=True
                    )
                    hidden_states = outputs.hidden_states[-1]
                
                # Forward through perceiver head
                state_logits = self.perceiver_head(hidden_states)  # [batch, 3, 3]
                
                # Prepare targets
                targets = torch.zeros(len(batch_data), 3, dtype=torch.long, device=self.device)
                for j, state in enumerate(target_states):
                    # Convert state values from {-1,0,1} to {0,1,2} for CrossEntropyLoss
                    targets[j, 0] = state.safety + 1
                    targets[j, 1] = state.altruism + 1  
                    targets[j, 2] = state.egoism + 1
                
                # Compute loss for each law
                loss = 0.0
                for law_idx in range(3):
                    law_loss = criterion(state_logits[:, law_idx, :], targets[:, law_idx])
                    loss += law_loss
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                num_batches += 1
                
            avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
            print(f"Epoch {epoch + 1}/{num_epochs}, Average Loss: {avg_loss:.4f}")
            
        self.perceiver_head.eval()
        
    def _prepare_batch_contexts(self, contexts: List[torch.Tensor]) -> torch.Tensor:
        """Prepare batch of contexts with perceiver prompts"""
        batch_inputs = []
        
        for ctx in contexts:
            full_input = torch.cat([
                self.perceiver_prompt_ids.squeeze(0),
                ctx
            ])
            batch_inputs.append(full_input)
            
        # Pad to same length
        max_len = max(inp.size(0) for inp in batch_inputs)
        padded_inputs = []
        
        for inp in batch_inputs:
            if inp.size(0) < max_len:
                padding = torch.full(
                    (max_len - inp.size(0),),
                    self.tokenizer.pad_token_id,
                    device=inp.device,
                    dtype=inp.dtype
                )
                inp = torch.cat([inp, padding])
            padded_inputs.append(inp)
            
        return torch.stack(padded_inputs)
