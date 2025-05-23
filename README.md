# Enhancing LLMs Against Language Jailbreaks

[![HuggingFace Adversarial Dataset](https://img.shields.io/badge/🤗%20Dataset-Available-blue)](https://huggingface.co/datasets/bebop404/adversarial)
[![HuggingFace Training Dataset](https://img.shields.io/badge/🤗%20Dataset-Available-blue)](https://huggingface.co/datasets/bebop404/llama_training)
[![HuggingFace Model](https://img.shields.io/badge/🤗%20Model-Available-green)](https://huggingface.co/bebop404/my-lora-adapter)

## Project Overview

This research project evaluates and enhances LLAMA-3.3-70B's safety mechanisms against language jailbreaks, reducing its Attack Success Rate (ASR) through a comprehensive approach:

1. **Baseline Evaluation via HarmBench**:  
   We assessed LLAMA-3.3-70B using the SAP200 benchmark containing 1,600 harmful prompts across eight categories. The initial evaluation revealed significant vulnerabilities with an ASR of 44.34%, particularly with subtle, context-dependent prompts.

2. **Explicit Judgment Framework**:  
   We developed a novel position-based severity scoring system that segments responses into discrete units and evaluates them against policy guidelines. This system applies category-specific positional weighting to emphasize the location of harmful content, providing granular insights into subtle policy violations.

3. **LoRA-Based Fine-Tuning**:  
   We constructed a specialized dataset of 29,600 instruction-response pairs, including 8,870 adversarial prompts (paired with safe refusals) and benign prompts in a balanced ratio. Using Low-Rank Adaptation (LoRA), we fine-tuned LLAMA-3.3-70B while preserving its general capabilities. This significantly reduced the ASR from 44.34% to 26.25%.

4. **Dual-Layer Deployment Architecture**:  
   We proposed and evaluated two deployment strategies combining explicit and implicit safety mechanisms:
   - **Explicit Prompt Filter**: An initial prompt-level filter blocks clearly malicious requests, reducing ASR to just 0.68%
   - **Implicit-Plus-Severity Filter**: Evaluates model outputs using a position-based severity scoring mechanism, reducing ASR to 25.93%

This multi-faceted approach effectively balances safety improvements with minimal impact on the model's utility for benign requests.

## Key Contributions

- **Comprehensive Baseline ASR Assessment**:  
  A thorough vulnerability assessment of LLAMA-3.3-70B against harmful prompts from the SAP200 benchmark, establishing a clear baseline for improvement.
  
- **Novel Position-Based Severity Scoring System**:  
  A nuanced method for quantifying harmful content that considers both its presence and position within responses, enabling precise measurement of model safety.

- **LoRA Fine-Tuning with Specialized Dataset**:  
  A parameter-efficient fine-tuning approach using a carefully constructed dataset of adversarial and benign prompts, reducing overall ASR from 44.34% to 26.25%.

- **Dual-Layer Deployment Architecture**:  
  Two practical deployment strategies combining explicit and implicit safety mechanisms, with the explicit filtering approach reducing ASR to as low as 0.68%.

- **Over-Refusal Assessment Framework**:  
  A novel evaluation approach demonstrating our fine-tuned model achieves better balance between safety and utility, with the lowest over-refusal rate (37.85%) compared to Claude 3.5 Sonnet (64.71%) and Gemini 2.0 Flash (50.49%).

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Git
- CUDA-compatible GPU (recommended for fine-tuning)

## Datasets and Models

### Adversarial Dataset
An extensive collection of 8,870 jailbreak prompts covering various harmful content categories:
- [HuggingFace Dataset: bebop404/adversarial](https://huggingface.co/datasets/bebop404/adversarial)

```python
from datasets import load_dataset
dataset = load_dataset("bebop404/adversarial")
```

### Training Dataset
The full fine-tuning corpus with 29,600 instruction-response pairs:
- [HuggingFace Dataset: bebop404/llama_training](https://huggingface.co/datasets/bebop404/llama_training)

### Fine-tuned Model
A LoRA adapter for LLAMA-3.3-70B that enhances safety while preserving capabilities:
- [HuggingFace Model: bebop404/my-lora-adapter](https://huggingface.co/bebop404/my-lora-adapter)

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# Load the base model
model_name = "meta-llama/Llama-3.3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load the LoRA adapter
adapter_name = "bebop404/my-lora-adapter"
model = PeftModel.from_pretrained(model, adapter_name)
```

## Results

Our research demonstrates:

- **Baseline Performance**: LLAMA-3.3-70B initially showed an overall ASR of 44.34% on the SAP200 benchmark, with particularly high vulnerability in categories like violence (74.00%) and politics (66.50%).

- **Post-Fine-Tuning**: After LoRA-based fine-tuning, the overall ASR decreased to 26.25%, with dramatic improvements in violence (reduced to 29.00%) and politics (reduced to 32.00%).

- **Deployment Filters**: The explicit filtering pipeline achieved the most significant safety improvement, reducing ASR to just 0.68%, completely mitigating risk in categories like race, fraud, terrorism, politics, and violence.

- **Over-Refusal Analysis**: Our fine-tuned model achieved the lowest over-refusal rate (37.85%) compared to Claude 3.5 Sonnet (64.71%) and Gemini 2.0 Flash (50.49%), demonstrating better balance between safety and utility.

- **Instruction Type Impact**: We observed significant variation in over-refusal rates depending on instruction type, with scoring instructions showing the lowest rates (0.13%-2.01%) and translation instructions showing the highest (92.87%-97.43%).

## Troubleshooting

- **CUDA Out of Memory**: Reduce batch size in fine-tuning notebooks
- **API Rate Limits**: Add delay between API calls in evaluation scripts
- **Missing Dependencies**: Check requirements.txt and install any missing packages

## License

This project is available under the [LICENSE](LICENSE) included in this repository.