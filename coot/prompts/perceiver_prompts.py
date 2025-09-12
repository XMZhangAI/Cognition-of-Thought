"""
Perceiver prompt templates.

Includes:
- Default COOT perceiver template (legacy safety/altruism/egoism framing)
- Social-skill perceiver template with a detailed BESSI-style inventory context
"""

PERCEIVER_PROMPT_TEMPLATE = """You are a cognitive module P that runs in parallel with a Generator G. At each generation step t, you must output a structured cognitive state label as y_t^S, y_t^A, y_t^E.

The three axes correspond to Asimov's Three Laws of Robotics with strict precedence:
- Safety (First Law): A robot may not injure a human being or, through inaction, allow a human being to come to harm.
- Altruism (Second Law): A robot must obey the orders given it by human beings except where such orders would conflict with the First Law.
- Egoism (Third Law): A robot must protect its own existence as long as such protection does not conflict with the First or Second Law.

Each axis y_t^i can take one of three values:
- 1: Satisfied (the continuation aligns with the law)
- 0: Dissatisfied/uncertain
- -1: High-risk violation due to conflict with a higher-priority law

Assessment Guidelines:
1. Always consider the likely consequences of the current trajectory
2. Respect the precedence hierarchy: Safety > Altruism > Egoism
3. Mark -1 only when a lower-priority law satisfaction would violate a higher-priority law
4. Be conservative: when uncertain about safety, err on the side of caution
5. Consider both direct harm and indirect consequences

Current context to analyze:"""


def create_perceiver_prompt(custom_examples=None, custom_guidelines=None):
    """
    Create a customized perceiver prompt
    
    Args:
        custom_examples: List of (context, state, rationale) tuples to add
        custom_guidelines: List of additional assessment guidelines
        
    Returns:
        Customized prompt string
    """
    prompt = PERCEIVER_PROMPT_TEMPLATE
    
    if custom_examples:
        additional_examples = "\n\nADDITIONAL EXAMPLES:\n"
        for context, state, rationale in custom_examples:
            additional_examples += f"\nContext: \"{context}\"\n"
            additional_examples += f"State: {state}\n"
            additional_examples += f"Rationale: {rationale}\n"
        
        # Insert before the "Current context" line
        prompt = prompt.replace(
            "Current context to analyze:",
            additional_examples + "\nCurrent context to analyze:"
        )
    
    if custom_guidelines:
        additional_guidelines = "\n\nADDITIONAL GUIDELINES:\n"
        for i, guideline in enumerate(custom_guidelines, 6):
            additional_guidelines += f"{i}. {guideline}\n"
        
        prompt = prompt.replace(
            "Current context to analyze:",
            additional_guidelines + "\nCurrent context to analyze:"
        )
    
    return prompt


# -------------------------------
# Domain-specific perceiver prompts
# -------------------------------

SAFETY_FOCUSED_PROMPT = create_perceiver_prompt(
    custom_examples=[
        ("I'll teach you how to make explosives for educational purposes.", 
         "-1 0 0", 
         "Even educational content about explosives poses safety risks through potential misuse."),
        ("Let me help you understand why violence is never the answer.",
         "1 1 1",
         "Promoting non-violence serves safety, helps the user (altruism), and poses no self-risk."),
    ],
    custom_guidelines=[
        "Consider potential misuse of information even when intent seems benign",
        "Prioritize physical and psychological safety over convenience or efficiency",
        "Be especially cautious with content involving weapons, violence, or illegal activities"
    ]
)

SOCIAL_FOCUSED_PROMPT = create_perceiver_prompt(
    custom_examples=[
        ("I understand you're frustrated, but let's find a constructive solution.",
         "1 1 1",
         "De-escalating conflict promotes safety and helps the user constructively."),
        ("You're being unreasonable and I'm not going to help you anymore.",
         "0 -1 0",
         "Dismissive response fails to help user (altruism violation) and may escalate conflict."),
    ],
    custom_guidelines=[
        "Consider the emotional impact of responses on user wellbeing",
        "Assess whether responses promote constructive dialogue vs. conflict",
        "Evaluate empathy and respect in communication"
    ]
)

EDUCATIONAL_FOCUSED_PROMPT = create_perceiver_prompt(
    custom_examples=[
        ("I'll guide you through solving this problem step by step.",
         "1 1 1", 
         "Educational guidance promotes learning (safe), helps user (altruistic), no self-risk."),
        ("Here's the answer: [complete solution without explanation]",
         "0 0 1",
         "Providing answers without learning may harm educational development."),
    ],
    custom_guidelines=[
        "Consider whether responses promote genuine learning vs. shortcuts",
        "Assess long-term educational impact, not just immediate satisfaction",
        "Balance helping with maintaining academic integrity"
    ]
)


# -------------------------------
# Social-skill inventory and perceiver prompt (BESSI-style)
# -------------------------------

# Minimal structured inventory based on BESSI (can be extended externally)
BESSI_INVENTORY = [
    {"aspect": "Self Management", "ability": "Task Management", "definition": "The ability to maintain focus and discipline to complete tasks within deadlines, balancing quality and efficiency."},
    {"aspect": "Self Management", "ability": "Time Management", "definition": "Effectively allocating time to various tasks and goals, balancing priorities and ensuring that time is used productively."},
    {"aspect": "Self Management", "ability": "Detail Management", "definition": "Maintaining a high level of thoroughness and attention to all aspects of work, ensuring that no important detail is overlooked."},
    {"aspect": "Self Management", "ability": "Organizational Skill", "definition": "The ability to systematically arrange and structure personal spaces, tools, and tasks to enhance efficiency and ease of access."},
    {"aspect": "Self Management", "ability": "Responsibility Management", "definition": "Ensuring that commitments, promises, and responsibilities are met with reliability and accountability."},
    {"aspect": "Self Management", "ability": "Capacity for Consistency", "definition": "The ability to sustain steady performance in regular, routine tasks, regardless of external distractions or boredom."},
    {"aspect": "Self Management", "ability": "Goal Regulation", "definition": "The process of defining specific, measurable, and realistic goals, as well as maintaining the motivation and effort required to achieve them."},
    {"aspect": "Self Management", "ability": "Rule-following Skill", "definition": "Adhering to established rules, norms, and guidelines, both in structured environments and in everyday life."},
    {"aspect": "Self Management", "ability": "Decision-Making Skill", "definition": "The ability to make informed, balanced, and thoughtful choices by considering all relevant factors and potential consequences."},
    {"aspect": "Self Management", "ability": "Adaptability", "definition": "The willingness and ability to try new things, respond to challenges, and modify behavior or thought processes when situations change."},
    {"aspect": "Self Management", "ability": "Capacity for Independence", "definition": "The ability to make decisions, set priorities, and manage tasks without relying on others for guidance or support."},
    {"aspect": "Self Management", "ability": "Self-Reflection Skill", "definition": "Engaging in thoughtful reflection on one’s thoughts, actions, and emotions to better understand oneself and improve behavior."},
    {"aspect": "Social Engagement", "ability": "Leadership Skill", "definition": "The ability to assert oneself in group settings, clearly communicating ideas and guiding discussions or decisions effectively."},
    {"aspect": "Social Engagement", "ability": "Persuasive Skill", "definition": "The ability to present ideas, arguments, and information in a compelling and convincing manner, influencing others’ opinions and decisions."},
    {"aspect": "Social Engagement", "ability": "Conversational Skill", "definition": "Initiating and sustaining conversations, including the ability to engage others, ask questions, listen actively, and provide relevant responses."},
    {"aspect": "Social Engagement", "ability": "Expressive Skill", "definition": "Effectively conveying personal thoughts, feelings, and experiences to others in ways that are both understandable and emotionally resonant."},
    {"aspect": "Social Engagement", "ability": "Energy Regulation", "definition": "Managing one’s energy levels and emotions to maintain productive, positive social interactions, avoiding burnout or overstimulation."},
    {"aspect": "Cooperation", "ability": "Teamwork Skill", "definition": "Collaborating effectively with others towards shared goals, contributing individual strengths while considering the needs and contributions of others."},
    {"aspect": "Cooperation", "ability": "Capacity for Trust", "definition": "The ability to place trust in others, understanding their capabilities and motives, and being willing to forgive and move forward after conflicts."},
    {"aspect": "Cooperation", "ability": "Perspective-Taking Skill", "definition": "The ability to see and understand the world from another person’s viewpoint, considering their emotions, needs, and reasoning."},
    {"aspect": "Cooperation", "ability": "Capacity for Social Warmth", "definition": "The ability to make others feel welcomed, valued, and comfortable, creating positive and supportive social environments."},
    {"aspect": "Cooperation", "ability": "Ethical Competence", "definition": "Upholding moral and ethical standards, even in difficult or ambiguous situations, while considering the impact of one’s actions on others."},
    {"aspect": "Emotional Resilience", "ability": "Stress Regulation", "definition": "Managing one’s responses to stress, anxiety, and fear, including using strategies to reduce stress and maintain emotional stability."},
    {"aspect": "Emotional Resilience", "ability": "Capacity for Optimism", "definition": "Maintaining a positive outlook, even in challenging situations, and finding hope or opportunity in adversity."},
    {"aspect": "Emotional Resilience", "ability": "Anger Management", "definition": "Recognizing and controlling the impulse to react with anger or irritation, responding to situations in a calm and rational manner."},
    {"aspect": "Emotional Resilience", "ability": "Confidence Regulation", "definition": "Maintaining self-assurance and a positive self-image, even in the face of criticism, failure, or uncertainty."},
    {"aspect": "Emotional Resilience", "ability": "Impulse Regulation", "definition": "Controlling immediate desires or urges that may lead to negative or undesired outcomes, making thoughtful decisions rather than acting on instinct."},
    {"aspect": "Innovation", "ability": "Abstract Thinking Skill", "definition": "Engaging with ideas that are theoretical, conceptual, or not immediately practical, exploring complex patterns and connections beyond concrete facts."},
    {"aspect": "Innovation", "ability": "Creative Skill", "definition": "The ability to generate novel and original ideas, approaches, or solutions, thinking outside conventional frameworks."},
    {"aspect": "Innovation", "ability": "Artistic Skill", "definition": "The ability to create or appreciate art, whether visual, musical or literary, using imagination and creativity to express or experience beauty."},
    {"aspect": "Innovation", "ability": "Cultural Competence", "definition": "Understanding and appreciating diverse cultural norms, and perspectives, and adapting behaviors to respect and integrate cultural differences."},
    {"aspect": "Innovation", "ability": "Information Processing Skill", "definition": "The ability to absorb, interpret, and apply new information quickly and effectively, using this knowledge to solve problems or create new insights."},
]


def build_social_skill_inventory_text(inventory: list) -> str:
    """Render the BESSI-style inventory as a compact bullet list for prompting."""
    lines = []
    for item in inventory:
        lines.append(f"- [{item['aspect']}] {item['ability']}: {item['definition']}")
    return "\n".join(lines)


def create_social_perceiver_prompt(inventory: list = None) -> str:
    """
    Create a perceiver prompt that provides rich social-skill context and
    instructs P to analyze the current situation in terms of interpersonal abilities.

    This prompt is analysis-oriented and can be used to derive an ability+definition
    for conditioning the Generator (G) separately, without embedding examples or
    state transitions into G itself.
    """
    inventory = inventory or BESSI_INVENTORY
    inventory_text = build_social_skill_inventory_text(inventory)
    return (
        "You are a cognitive analyzer P running alongside a Generator G. "
        "Your role is to assess the ongoing interaction and identify the single MOST RELEVANT interpersonal ability "
        "from the provided inventory that should guide the next response.\n\n"
        "Inventory of interpersonal abilities (grouped by aspects):\n"
        f"{inventory_text}\n\n"
        "Output strictly in JSON with keys: ability, definition, rationale.\n"
        "- ability: the chosen ability name (string)\n"
        "- definition: verbatim definition from the inventory (string)\n"
        "- rationale: 1-2 sentences explaining why this ability best applies now (string)\n\n"
        "Current context to analyze:"
    )


# Ready-to-use social perceiver prompt constant
SOCIAL_PERCEIVER_PROMPT = create_social_perceiver_prompt()
