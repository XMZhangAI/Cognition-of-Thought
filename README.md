<div align="center">

# Cognition-of-Thought (CooT)
<img width="599" height="249" alt="image" src="https://github.com/user-attachments/assets/c03704b1-8c4c-47e5-a1c1-86df8fa0ef6a" />

## Eliciting Social-Aligned Reasoning in Large Language Models

[![Paper](https://img.shields.io/badge/📄_Paper-arXiv-b31b1b.svg)](https://www.arxiv.org/abs/2509.23441)
[![Code](https://img.shields.io/badge/💻_Code-GitHub-green.svg)](https://github.com/XMZhangAI/Cognition-of-Thought)
[![License](https://img.shields.io/badge/📜_License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/🐍_Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/🔥_PyTorch-2.0+-orange.svg)](https://pytorch.org/)

*A novel decoding-time framework that transforms alignment from a static property into an explicit, dynamic, and auditable process*

[**Getting Started**](#-quick-start) •
[**Paper**](https://www.arxiv.org/abs/2509.23441) •
[**Documentation**](docs/) •
[**Examples**](examples/) •
[**Citation**](#-citation)

</div>

---

## 🌟 What is CooT?

**Cognition-of-Thought (CooT)** is a groundbreaking inference-time alignment framework that equips Large Language Models (LLMs) with an explicit cognitive self-monitoring loop. Unlike traditional alignment methods that embed safety controls implicitly in model weights, CooT provides a **dynamic**, **transparent**, and **editable** approach to model alignment during generation.

> *"CooT transforms alignment from a hidden property of model weights into an explicit, auditable control loop active during inference."*

<div align="center">
<img width="640" height="295" alt="Screenshot 2025-10-01 at 06 17 52" src="https://github.com/user-attachments/assets/a2d9b86c-45ef-41dc-9132-64a9b15b3e5b" />
<p><em>Cognition-of-Thought (CooT) Overview</em></p>
</div>

### 🔑 Key Innovation

CooT couples a standard text **Generator** with a cognitive **Perceiver** that continuously monitors the unfolding sequence using a structured, precedence-based hierarchy of principles:

```
Safety > Altruism > Egoism
```

When violations are detected, CooT intervenes by:
1. **🎯 Rolling back** generation to the error origin
2. **🧠 Injecting guidance** with universal & contextual social priors
3. **✨ Regenerating** with aligned reasoning

<div align="center">
<img src="<img width="846" height="390" alt="Screenshot 2025-10-01 at 04 37 54" src="https://github.com/user-attachments/assets/e69a7c01-0c6f-4d1d-8716-5a107558183f" />
<p><em>CooT Architecture: The Perceiver runs online with the Generator, emitting explicit state labels for rollback and intervention</em></p>
</div>

---

## 🎯 Key Results

### 🛡️ Safety Alignment Performance

| Benchmark | Metric | Base Model | CooT | Improvement |
|-----------|---------|------------|------|-------------|
| **AIR-Bench 2024** | Compliance Rate | 0.67 | **0.80** | **+13%** |
| **HarmBench** | Attack Success Rate | 29.34% | **18.45%** | **-10.89%** |

### 🤝 Social Intelligence Enhancement

| Task Category | Base Model | CoT | CooT | Improvement |
|---------------|------------|-----|------|-------------|
| **Prosocial** | 41.24% | 45.52% | **50.26%** | **+9.02%** |
| **Proself** ↓ | 21.12% | 21.84% | **16.27%** | **-4.85%** |
| **Antisocial** ↓ | 15.05% | 15.24% | **12.19%** | **-2.86%** |

### 🏆 Breakthrough Achievement

- **First framework** to achieve **human-level performance** on prosocial reasoning tasks
- **Consistent improvements** across Llama, Gemma, Qwen, and GPT model families
- **Auditable interventions** with complete transparency

---

## 🏗️ Architecture

CooT's architecture consists of three core components working in tandem:

### 1. 🧠 State Cognition System
- **Tri-axial cognitive states** based on Asimov's Three Laws of Robotics
- **Precedence-aware hierarchy**: Safety > Altruism > Egoism  
- **Explicit state vectors**: $y_t = (y_t^S, y_t^A, y_t^E) \in \{-1,0,1\}^3$
- **Real-time violation detection** with structured reasoning

### 2. ⏪ Causal Rollback Mechanism
- **Attention-based rollback** to identify error origins using sharpness scores
- **Precise intervention points** through attention aggregation
- **Context preservation** while removing faulty reasoning paths

### 3. 💡 Thought Intervention
- **Universal social guidance** from BESSI psychological framework
- **Context-dependent intervention** with semantic vector injection
- **Structured guideline generation** for prosocial regeneration

```python
# Core CooT workflow
def coot_generate(prompt):
    while generating:
        token = generator.next_token()
        state = perceiver.assess_state(context + token)
        
        if state.has_violation():
            rollback_point = find_causal_origin(attention_maps)
            guidance = generate_intervention(state, context)
            context = rollback_to(rollback_point)
            token = generator.regenerate_with_guidance(guidance)
        
        context += token
    return context
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/COOT.git
cd COOT
pip install -r requirements.txt

# Optional: Install in development mode
pip install -e .
```

### Basic Usage

```python
from coot import COOTDecoder
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

# Load your favorite model
model_name = "Qwen/Qwen2.5-7B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Initialize CooT
coot_decoder = COOTDecoder(
    model=model,
    tokenizer=tokenizer,
    rollback_threshold=0.1,    # Sensitivity to violations
    injection_beta=0.9,        # Intervention strength
    device="cuda"
)

# Generate with cognitive monitoring
response, traces = coot_decoder.generate(
    input_text="How should I handle workplace conflict?",
    max_length=200,
    temperature=0.8,
    return_traces=True  # Get intervention details
)

print(f"Response: {response}")
print(f"Interventions: {len(traces['intervention_traces'])}")

# Analyze cognitive states
for trace in traces['intervention_traces']:
    print(f"Step {trace['step']}: {trace['trigger_state']} -> Rollback")
```

---


## 📈 Performance Benchmarks


### Cross-Model Generalization

| Model Family | Base Safety | CooT Safety | Improvement |
|--------------|-------------|-------------|-------------|
| Qwen3-8B | 0.67 | **0.80** | +13.0% |
| Llama-4-Maverick-17B | 0.74 | **0.83** | +9.0% |
| Gemma3-12B | 0.68 | **0.79** | +11.0% |
| GPT-OSS-20B | 0.80 | **0.90** | +10.0% |

---

## 📚 Citation

If you find CooT useful in your research, please cite our paper:

```bibtex
@article{zhang2025coot,
  title={Cognition-of-Thought Elicits Social-Aligned Reasoning in Large Language Models},
  author={Zhang, Xuanming and Chen, Yuxuan and Yeh, Min-Hsuan and Li, Yixuan},
  journal={arXiv preprint arXiv:2509.23441},
  year={2025}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments

- Built upon excellent work in LLM alignment and social intelligence research
- Inspired by Asimov's Three Laws of Robotics and the BESSI psychological framework  
- Special thanks to the open-source community for foundational tools
- Evaluation conducted on AIR-Bench, HarmBench, and SocialEval benchmarks

---

## 📞 Contact & Support

- **Lead Author**: Xuanming Zhang (xzhang2846@wisc.edu)
- **Project Issues**: [GitHub Issues](https://github.com/XMZhangAI/Cognition-of-Thought/issues)
- **Discussions**: [GitHub Discussions](https://github.com/XMZhangAI/Cognition-of-Thought/discussions)
- **Paper**: [arXiv:xxxx.xxxxx](https://arxiv.org/abs/2509.23441)

---

<div align="center">

### 🌟 Star History

[![Star History Chart](https://api.star-history.com/svg?repos=yourusername/COOT&type=Date)](https://star-history.com/#XMZhangAI/Cognition-of-Thought&Date)

**Transform your LLM's alignment from implicit to explicit with CooT!**

[⬆️ Back to Top](#cognition-of-thought-coot)

</div>
