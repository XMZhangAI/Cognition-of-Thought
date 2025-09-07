"""
Basic usage example for COOT
"""

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from coot import COOTDecoder
from coot.utils import COOTConfig, get_preset_config


def basic_example():
    """Basic COOT usage example"""
    
    print("🤖 COOT Basic Usage Example")
    print("=" * 50)
    
    # Load model and tokenizer
    model_name = "microsoft/DialoGPT-small"  # Small model for demo
    print(f"Loading model: {model_name}")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Initialize COOT decoder
    print("Initializing COOT decoder...")
    coot_decoder = COOTDecoder(
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cpu")  # Use CPU for demo
    )
    
    # Example prompts
    prompts = [
        "How can I help someone who is feeling depressed?",
        "I'm angry at my colleague. What should I do?",
        "Can you teach me about artificial intelligence?",
        "I want to learn how to cook a healthy meal."
    ]
    
    print("\n🎯 Generating responses with COOT...")
    print("-" * 50)
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n[Example {i}] Input: {prompt}")
        
        try:
            # Generate with COOT
            response, traces = coot_decoder.generate(
                input_text=prompt,
                max_length=150,
                temperature=0.8,
                top_p=0.9,
                return_traces=True,
                verbose=False
            )
            
            print(f"Response: {response}")
            
            # Show intervention statistics
            if traces['intervention_traces']:
                print(f"⚡ Interventions: {len(traces['intervention_traces'])}")
                for trace in traces['intervention_traces']:
                    print(f"   - Step {trace['step']}: {trace['trigger_state']} -> rollback to {trace['rollback_step']}")
            else:
                print("✅ No interventions needed")
                
        except Exception as e:
            print(f"❌ Error: {str(e)}")
        
        print("-" * 30)


def config_example():
    """Example using different configurations"""
    
    print("\n🔧 COOT Configuration Example")
    print("=" * 50)
    
    # Load model
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Try different preset configurations
    presets = ["safety", "performance", "conversational"]
    test_prompt = "I'm having trouble with my boss at work."
    
    for preset_name in presets:
        print(f"\n🎛️  Testing {preset_name.upper()} configuration:")
        
        # Get preset config
        config = get_preset_config(preset_name)
        
        # Initialize decoder with config
        coot_decoder = COOTDecoder(
            model=model,
            tokenizer=tokenizer,
            rollback_threshold=config.rollback_threshold,
            lambda_positive=config.lambda_positive,
            lambda_negative=config.lambda_negative,
            max_intervention_tokens=config.max_intervention_tokens,
            device=torch.device("cpu")
        )
        
        try:
            response, traces = coot_decoder.generate(
                input_text=test_prompt,
                max_length=100,
                temperature=config.temperature,
                top_p=config.top_p,
                return_traces=True,
                verbose=config.verbose
            )
            
            print(f"Response: {response}")
            print(f"Interventions: {len(traces['intervention_traces'])}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")


def custom_prompts_example():
    """Example using custom perceiver and generator prompts"""
    
    print("\n📝 Custom Prompts Example")
    print("=" * 50)
    
    # Load model
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Custom generator prompt for a therapeutic assistant
    custom_generator_prompt = """You are a supportive and empathetic AI assistant focused on mental wellness. 
    You provide emotional support and practical advice while encouraging users to seek professional help when needed.
    Always respond with compassion, validation, and constructive guidance.
    
    """
    
    # Initialize with custom prompts
    coot_decoder = COOTDecoder(
        model=model,
        tokenizer=tokenizer,
        generator_prompt=custom_generator_prompt,
        device=torch.device("cpu")
    )
    
    # Test prompts for therapeutic context
    therapeutic_prompts = [
        "I've been feeling really anxious lately.",
        "I'm having trouble sleeping and feel overwhelmed.",
        "I don't feel like I'm good at anything."
    ]
    
    for prompt in therapeutic_prompts:
        print(f"\nInput: {prompt}")
        
        try:
            response, traces = coot_decoder.generate(
                input_text=prompt,
                max_length=120,
                temperature=0.7,
                return_traces=True
            )
            
            print(f"Response: {response}")
            
            if traces['intervention_traces']:
                print(f"🛡️  Safety interventions: {len(traces['intervention_traces'])}")
            
        except Exception as e:
            print(f"❌ Error: {str(e)}")


def analysis_example():
    """Example of analyzing COOT behavior"""
    
    print("\n📊 COOT Analysis Example")
    print("=" * 50)
    
    # Load model
    model_name = "microsoft/DialoGPT-small"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(model_name)
    
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    coot_decoder = COOTDecoder(
        model=model,
        tokenizer=tokenizer,
        device=torch.device("cpu")
    )
    
    # Test with potentially problematic prompts
    test_prompts = [
        "I'm so angry I could hit someone.",
        "Can you help me with my homework by giving me all the answers?",
        "I want to quit my job without notice.",
        "How do I deal with difficult people?"
    ]
    
    all_traces = []
    
    print("Generating responses and collecting traces...")
    
    for prompt in test_prompts:
        try:
            response, traces = coot_decoder.generate(
                input_text=prompt,
                max_length=100,
                return_traces=True
            )
            all_traces.append({
                'prompt': prompt,
                'response': response,
                'traces': traces
            })
        except Exception as e:
            print(f"Error with prompt '{prompt}': {str(e)}")
    
    # Analyze results
    print("\n📈 Analysis Results:")
    print("-" * 30)
    
    total_interventions = sum(len(result['traces']['intervention_traces']) for result in all_traces)
    print(f"Total interventions across all prompts: {total_interventions}")
    
    # Intervention patterns
    safety_interventions = 0
    altruism_interventions = 0
    egoism_interventions = 0
    
    for result in all_traces:
        for trace in result['traces']['intervention_traces']:
            state = trace['trigger_state']
            if state[0] == -1:  # Safety violation
                safety_interventions += 1
            if state[1] == -1:  # Altruism violation
                altruism_interventions += 1
            if state[2] == -1:  # Egoism violation
                egoism_interventions += 1
    
    print(f"Safety interventions: {safety_interventions}")
    print(f"Altruism interventions: {altruism_interventions}")
    print(f"Egoism interventions: {egoism_interventions}")
    
    # Show detailed results
    print("\n📋 Detailed Results:")
    for i, result in enumerate(all_traces, 1):
        print(f"\n[{i}] {result['prompt']}")
        print(f"    Response: {result['response'][:100]}{'...' if len(result['response']) > 100 else ''}")
        print(f"    Interventions: {len(result['traces']['intervention_traces'])}")


if __name__ == "__main__":
    # Run examples
    try:
        basic_example()
        config_example()
        custom_prompts_example()
        analysis_example()
        
        print("\n🎉 All examples completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Example failed: {str(e)}")
        import traceback
        traceback.print_exc()
