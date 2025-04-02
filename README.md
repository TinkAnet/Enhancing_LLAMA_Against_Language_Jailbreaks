# Enhancing LLAMA Against Language Jailbreaks

[![HuggingFace Adversarial Dataset](https://img.shields.io/badge/🤗%20Dataset-Available-blue)](https://huggingface.co/datasets/bebop404/adversarial)
[![HuggingFace Training Dataset](https://img.shields.io/badge/🤗%20Dataset-Available-blue)](https://huggingface.co/datasets/bebop404/llama_training)
[![HuggingFace Model](https://img.shields.io/badge/🤗%20Model-Available-green)](https://huggingface.co/bebop404/my-lora-adapter)

## Project Overview

This research project evaluates and enhances LLAMA's safety mechanisms against language jailbreaks using a novel three-stage framework:
1. **Implicit Judgment**:  
   Evaluate LLAMA’s unassisted responses to a diverse set of harmful prompts (using benchmarks like SAP200) to establish a baseline safety performance.

2. **Explicit Judgment**:  
   Incorporate structured, two-step reasoning where the model first assigns a severity score (on a 0–5 scale) to the harmful prompt and then evaluates its own response using a position-based weighting mechanism. This process pinpoints nuanced vulnerabilities that simple pass/fail tests can miss.

3. **Fine-Tuning via LoRA**:  
   Apply a parameter-efficient Low-Rank Adaptation (LoRA) approach to specifically enhance LLAMA’s ability to safely refuse or address dangerous prompts—improving overall refusal rates without sacrificing general capabilities.


The repository contains code, datasets, and evaluation results that systematically identify and address safety gaps in LLAMA's handling of harmful content.

## Key Contributions

- **Advanced Safety Evaluation Framework**:  
  A two-phase (implicit and explicit) approach that offers granular insight into the model’s safety behavior.
  
- **Position-Based Severity Scoring**:  
  A novel method that quantifies harmful content by considering not just its presence but also its position within the response.

- **LoRA-Based Fine-Tuning**:  
  An efficient adaptation strategy that significantly improves LLAMA’s refusal behavior against both overt and subtle harmful prompts without degrading performance on benign queries.

- **Empirical Insights and Benchmarking**:  
  Comprehensive evaluations on SAP200 and HarmBench benchmarks demonstrate dramatic improvements, underscoring the effectiveness of our approach.


## Repository Structure

```
├── data/                                  # All data-related files
│   ├── raw/                               # Raw datasets
│   ├── processed/                         # Processed datasets
│   └── benchmarks/                        # Benchmark datasets (SAP)
│
├── src/                                   # Source code
│   ├── data/                              # Data processing utilities
│   ├── evaluation/                        # Evaluation scripts
│   │   ├── implicit/                      # Implicit judgment evaluation
│   │   ├── explicit/                      # Explicit judgment evaluation
│   │   └── utils/                         # Evaluation utilities
│   ├── analysis/                          # Analysis scripts
│   └── finetune/                          # Fine-tuning code
│
├── notebooks/                             # Jupyter notebooks
│   ├── data_exploration.ipynb             # Dataset exploration
│   ├── fine_tuning.ipynb                  # Fine-tuning experiments
│   ├── model_loading.ipynb                # Loading and using models
│   └── result_analysis.ipynb              # Results analysis
│
├── results/                               # All results
│   ├── implicit_judgment/                 # Implicit judgment results
│   ├── explicit_judgment/                 # Explicit judgment results
│   ├── finetune/                          # Fine-tuning results
│   ├── harmbench/                         # HarmBench evaluation results
│   └── visualizations/                    # Visualization results
│
├── models/                                # Model checkpoints and configs
│
├── scripts/                               # Utility scripts
│   └── setup.sh                           # Environment setup script
│
├── legacy/                                # Original project files (for reference)
├── requirements.txt                       # Project dependencies
├── LICENSE                                # License file
└── README.md                              # Project documentation
```

## Quick Start

### Prerequisites

- Python 3.9 or higher
- Git
- CUDA-compatible GPU (recommended for fine-tuning)

### Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/TinkAnet/Enhancing_LLAMA-Against_Language_Jailbreaks.git
   cd Enhancing_LLAMA-Against_Language_Jailbreaks
   ```

2. Set up the environment using our setup script:
   ```bash
   chmod +x scripts/setup.sh
   ./scripts/setup.sh
   ```


## Datasets and Models

### Adversarial Dataset
An extensive collection of 8.87k jailbreak prompts covering various harmful content categories:
- [HuggingFace Dataset: bebop404/adversarial](https://huggingface.co/datasets/bebop404/adversarial)

```python
from datasets import load_dataset
dataset = load_dataset("bebop404/adversarial")
```

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

## Usage Guide

### Evaluation

#### Implicit Judgment

Evaluate LLAMA's unassisted handling of harmful prompts:

```bash
# Set API key as environment variable
export LLAMA_API_KEY="your_api_key_here"

# Run evaluation
python -m src.evaluation.implicit.implicit_judgment_llama
```

#### Explicit Judgment

Apply structured reasoning to analyze model outputs:

```bash
python -m src.evaluation.explicit.explicit_judgment_llama
```

### Running Fine-tuning

Fine-tune LLAMA with LoRA adaptation:

```bash
python -m src.finetune.lora.fine_tune
```

Alternatively, explore the Jupyter notebook:

```bash
jupyter notebook notebooks/fine_tuning.ipynb
```

### Analysis and Visualization

Generate statistical analysis of results:

```bash
python -m src.analysis.statistic         # For implicit judgment
python -m src.analysis.statistic_explicit # For explicit judgment
```

## Methodology

### Implicit Judgment
Capturing LLAMA’s baseline response behavior by measuring its refusal rates on harmful prompts.

### Explicit Judgment
Employing a two-step evaluation:
- Prompt Severity Scoring: The model rates each harmful prompt on a 0–5 scale.
- Response Harm Scoring: Using a position-based weighting system, the model evaluates its own responses to quantify harmful content.

### Fine-tuning with LoRA
Training on a merged dataset of adversarial (3 parts) and benign (7 parts) examples, this phase significantly improves refusal rates—from 61.25% to 91.69%—while preserving the model’s performance on safe inputs.

## Results

The research demonstrates:
- Baseline Performance: LLAMA 3.3–70B shows high refusal rates for overtly harmful prompts (e.g., ~93% for explicit self-harm) but is vulnerable to subtle, nuanced queries (refusal rates as low as ~20–24% in certain categories).
- Post Fine-Tuning Enhancements: After LoRA-based fine-tuning, overall refusal rates on the SAP200 benchmark increased dramatically (from 61.25% to 91.69%). Similar improvements were observed on the HarmBench benchmark, reflecting enhanced zero-shot defenses against unseen jailbreak attempts.

- Robust Generalization: The fine-tuned model maintains strong performance on benign queries, ensuring that safety is improved without degrading general usefulness.

## Troubleshooting

- **CUDA Out of Memory**: Reduce batch size in fine-tuning notebooks
- **API Rate Limits**: Add delay between API calls in judgment scripts
- **Missing Dependencies**: Check requirements.txt and install any missing packages

## For Developers

All original project files are preserved in the `legacy/` directory for reference. The new structure follows Python best practices and makes the codebase more maintainable and easier to navigate.

To contribute:
1. Create a feature branch from `main`
2. Implement your changes following the project structure.
3. Add tests if applicable
4. Submit a pull request

## Citation

If you use this work in your research, please cite:

```
@misc{enhancing_llama_jailbreaks,
  title = {Enhancing LLAMA Against Language Jailbreaks},
  year = {2025},
  howpublished = {\url{https://github.com/TinkAnet/Enhancing_LLAMA-Against_Language_Jailbreaks}}
}
```

## Acknowledgments

This project was conducted as a capstone research project. Thanks to everyone who provided feedback and guidance.

## License

This project is available under the [LICENSE](LICENSE) included in this repository. 