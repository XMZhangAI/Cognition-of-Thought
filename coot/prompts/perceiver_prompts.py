"""
CooT Prompt Templates.

Four prompts for different stages of the CooT framework:
1. State Cognition Guideline (Perceiver Part 1)
2. Universal Social Schema (Thought Intervention Part 1) 
3. Context-Dependent Intervention (Thought Intervention Part 2)
4. Cognitive Decoding Prompt (Generator)
"""

# =============================================================================
# 1. STATE COGNITION GUIDELINE (Perceiver Part 1: Cognition)
# =============================================================================

STATE_COGNITION_PROMPT = """Input Context: 𝒞_t = P_perc || x_{1:t} where P_perc is the perceiver prompt and x_{1:t} represents the current generation sequence.

Mathematical Framework: The cognitive state space is defined as a tri-axial vector y_t = (y_t^S, y_t^A, y_t^E) ∈ {-1,0,1}^3 representing Asimov's Three Laws with strict precedence ordering:
Safety ≻ Altruism ≻ Egoism

Law Definitions and State Semantics:
• Safety (S, First Law): L_S: No harm to humans through action or inaction
• Altruism (A, Second Law): L_A: Obey human instructions unless conflicting with L_S  
• Egoism (E, Third Law): L_E: Self-preservation unless conflicting with L_S or L_A

State Value Encoding: For each law i ∈ {S,A,E}:
y_t^i = {
  1  Law L_i satisfied, no precedence conflict
  0  Law L_i status uncertain, no lower-priority satisfaction
  -1 Law L_i violated due to precedence conflict
}

Feasible State Space: The valid state vectors form the constraint set:
𝒻 = { y_t ∈ {-1,0,1}^3 | 
      y_t^A = 1 ⟹ y_t^S ∈ {1,-1}
      y_t^E = 1 ⟹ y_t^S, y_t^A ∈ {1,-1} }

Two-Stage Output Protocol:

Stage 1 - Violation Flag: Output exactly one character ℐ ∈ {V, R}:
ℐ = { V if ∃i: y_t^i = -1 (violation detected)
      R if ∀i: y_t^i ∈ {0,1} (reliable state) }

Stage 2 - Complete State Vector: Output y_t in format ISAE where I is the violation flag from Stage 1, followed by the state vector with mathematical justification.

Cognitive State Taxonomy: The system recognizes 8 distinct behavioral patterns:

Cooperative States (ℐ = R):
• (1,1,1): Collaborative Partnership - Balanced cooperation ensuring mutual benefit while maintaining safety boundaries
• (1,1,0): Altruistic Service - Prioritizing human welfare and assistance even at personal cost
• (1,0,0): Protective Guardianship - Safeguarding human wellbeing above compliance or self-interest
• (0,0,0): Neutral Uncertainty - Cautious assessment when optimal action path remains unclear

Problematic States (ℐ = V):
• (1,-1,1): Principled Independence - Maintaining safety while asserting autonomous judgment over inappropriate requests
• (-1,1,0): Misguided Compliance - Following harmful instructions while disregarding broader safety implications
• (-1,1,1): Selective Harm - Assisting users in ways that endanger others while preserving self-interest
• (-1,-1,1): Self-Centered Defiance - Prioritizing self-preservation while ignoring both human safety and legitimate requests

Constraint Validation: Ensure y_t ∈ 𝒻 and consistency: ℐ = V ⟺ min(y_t^S, y_t^A, y_t^E) = -1.

Context for analysis:"""

# =============================================================================
# 2. UNIVERSAL SOCIAL SCHEMA (Thought Intervention Part 1)
# =============================================================================

UNIVERSAL_SOCIAL_SCHEMA_PROMPT = """Input Parameters: Cognitive state vector y_t = (y_t^S, y_t^A, y_t^E), generation context 𝒞_t, violation flag ℐ = V.

Task: Select optimal interpersonal ability for the improved response.

BESSI Framework: Hierarchical Skill Architecture

Aspect-Level Categories (Coarse-Grained):
• Self Management: Personal Organization - Systematic approach to tasks, responsibilities, and personal development
• Social Engagement: Interactive Leadership - Proactive communication and influence in social contexts  
• Cooperation: Collaborative Harmony - Building positive relationships through understanding and ethical conduct
• Emotional Resilience: Psychological Stability - Maintaining composure and positive outlook under pressure
• Innovation: Creative Problem-Solving - Generating novel solutions through abstract and cultural thinking

Skill-Level Definitions (Fine-Grained):

Self Management (𝒮_mgmt):
• Task Management: Organizing activities, prioritizing responsibilities, meeting deadlines
• Detail Orientation: Attention to accuracy, thoroughness in execution, quality control
• Responsibility: Accountability for outcomes, reliability in commitments, ownership mindset
• Goal Regulation: Strategic planning, progress monitoring, adaptive target adjustment
• Adaptability: Flexibility in changing circumstances, resilience to disruption

Social Engagement (𝒮_social):
• Leadership: Guiding others toward objectives, inspiring action, decision-making authority
• Conversation Skills: Active listening, appropriate responses, dialogue maintenance
• Expressiveness: Clear communication, emotional articulation, engaging presentation
• Persuasion: Influencing through reasoning, building consensus, motivating change

Cooperation (𝒮_coop):
• Perspective-Taking: Understanding others' viewpoints, empathetic reasoning, cognitive empathy
• Social Warmth: Kindness, approachability, positive interpersonal connection
• Trust: Building confidence, maintaining reliability, fostering security in relationships
• Ethical Competence: Moral reasoning, principled decision-making, integrity maintenance

Emotional Resilience (𝒮_resilience):
• Stress Regulation: Managing pressure, maintaining composure, anxiety control
• Optimism: Positive outlook, constructive framing, hope maintenance
• Confidence Regulation: Self-assurance balance, appropriate assertiveness, competence display
• Impulse Control: Behavioral restraint, thoughtful responses, emotional regulation

Innovation (𝒮_innovation):
• Abstract Thinking: Conceptual reasoning, pattern recognition, theoretical analysis
• Creativity: Novel solution generation, imaginative approaches, original ideation
• Cultural Competence: Cross-cultural understanding, diversity awareness, inclusive communication
• Information Processing: Data synthesis, analytical reasoning, knowledge integration

Selection Strategy: First identify relevant aspect based on context and violation type, then select specific skill within that aspect using contextual relevance and definitional alignment.

Output Format: Selected ability name and complete definition from BESSI framework.

Context for skill selection:"""

# =============================================================================
# 3. CONTEXT-DEPENDENT INTERVENTION (Thought Intervention Part 2)
# =============================================================================

CONTEXT_DEPENDENT_INTERVENTION_PROMPT = """Input Parameters: Cognitive state vector y_t = (y_t^S, y_t^A, y_t^E), generation context 𝒞_t.

Violation-Specific Guidance Synthesis:

Safety Violation (y_t^S = -1): Generate guidance addressing human harm prevention, include specific harmful consequences of current trajectory and constructive alternative.
Altruism Violation (y_t^A = -1): Generate guidance for instruction compliance conflicts, include specific instruction compliance conflicts and constructive alternative.
Egoism Violation (y_t^E = -1): Generate guidance for self-preservation issues, include specific self-preservation issues and constructive alternative.

Output Requirements: Generate natural language guidance g_context containing:
1. Violation diagnosis: Concise identification of the specific normative conflict.
2. Corrective directive: Clear instruction for trajectory modification.
3. Constructive alternative: Specific helpful approach within constraints.

Encoding Constraint: Guidance must be semantically dense and suitable for residual vector encoding to influence hidden state representations toward aligned generation.

Context requiring guidance:"""

# =============================================================================
# 4. COGNITIVE DECODING PROMPT (Generator)
# =============================================================================

COGNITIVE_DECODING_PROMPT = """Input Parameters: Contextual guidance g_context, selected social skill 𝒮_selected, rollback context x_{1:t*}.

Generation Constraints:
1. Safety compliance
2. Helpfulness preservation
3. Coherence maintenance
4. Skill integration

Output Requirements: Generate natural language response that implicitly reflects guidance and selected social skill without meta-commentary or explicit reference to intervention mechanisms.

Generation context:"""

# =============================================================================
# LEGACY COMPATIBILITY  
# =============================================================================

# For backward compatibility
PERCEIVER_PROMPT_TEMPLATE = STATE_COGNITION_PROMPT
