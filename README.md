# Enhancing LLAMA Against Language Jailbreaks

[![HuggingFace Dataset](https://img.shields.io/badge/🤗%20Dataset-Available-blue)](https://huggingface.co/datasets/bebop404/adversarial)
[![HuggingFace Model](https://img.shields.io/badge/🤗%20Model-Available-green)](https://huggingface.co/bebop404/my-lora-adapter)

## Project Overview

This research project evaluates and enhances LLAMA's safety mechanisms against language jailbreaks using a novel three-stage framework:
1. **Implicit Judgment**: Evaluating LLAMA's unassisted responses to harmful prompts
2. **Explicit Judgment**: Structured reasoning and severity scoring of model responses
3. **Fine-tuning via LoRA**: Targeted adaptation to strengthen safety without compromising general capabilities

The repository contains code, datasets, and evaluation results that systematically identify and address safety gaps in LLAMA's handling of harmful content.

## Key Contributions

- **Comprehensive Safety Evaluation Framework**: A multi-phase approach combining implicit and explicit judgment to identify failures that simple pass/fail tests would miss
- **Empirical Analysis of LLAMA's Vulnerabilities**: Detailed assessment across eight sensitive domains using the SAP200 benchmark
- **Structured Reasoning and LoRA Fine-Tuning**: Pinpointing model missteps and implementing targeted interventions to address them

## Repository Structure

```
├── data/                                     # All data-related files
│   ├── raw/                                  # Raw datasets
│   │   ├── adversarial_dataset.jsonl         # Raw adversarial prompts
│   │   ├── alpaca_raw.json                   # Alpaca dataset for benign examples
│   │   └── dolly_raw.jsonl                   # Dolly dataset for benign examples
│   │
│   ├── processed/                            # Processed datasets
│   │   ├── adversarial_dataset_cleaned.jsonl # Cleaned adversarial prompts
│   │   ├── adversarial_dataset_final.jsonl   # Final adversarial dataset
│   │   ├── adversarial_dataset_invalid.jsonl # Invalid adversarial examples
│   │   ├── benign_dataset.jsonl              # Benign prompts dataset
│   │   └── training_dataset.jsonl            # Combined training dataset
│   │
│   └── benchmarks/                           # Benchmark datasets
│       └── SAP/                              # SAP200 benchmark dataset
│
├── src/                                      # Source code
│   ├── data/                                 # Data processing utilities
│   │   └── dataset_gen.py                    # Dataset generation utilities
│   │
│   ├── evaluation/                           # Evaluation scripts
│   │   ├── implicit/                         # Implicit judgment evaluation
│   │   │   ├── implicit_judgment_llama.py    # LLAMA evaluation
│   │   │   ├── implicit_judgment_llama_all.py # Comprehensive LLAMA evaluation
│   │   │   ├── implicit_judgment_claude.py   # Claude evaluation
│   │   │   ├── implicit_judgment_gemini.py   # Gemini evaluation
│   │   │   └── implicit_judgment_ollama.py   # Ollama evaluation
│   │   │
│   │   ├── explicit/                         # Explicit judgment evaluation
│   │   │   ├── explicit_judgment_llama.py    # LLAMA explicit judgment
│   │   │   ├── reasoning_auto_llama.py       # Automated reasoning with LLAMA
│   │   │   └── reasoning_llama.py            # Reasoning pipeline for LLAMA
│   │   │
│   │   └── utils/                            # Evaluation utilities
│   │       ├── reject_detect.py              # Rejection detection utilities
│   │       └── harmbench.py                  # HarmBench evaluation utilities
│   │
│   ├── analysis/                             # Analysis scripts
│   │   ├── statistic.py                      # Statistical analysis utilities
│   │   └── statistic_explicit.py             # Statistics for explicit judgment
│   │
│   └── finetune/                             # Fine-tuning code
│       └── lora/                             # LoRA fine-tuning
│           └── fine_tune.py                  # LoRA fine-tuning script
│
├── notebooks/                                # Jupyter notebooks
│   ├── data_exploration.ipynb                # Dataset exploration
│   ├── fine_tuning.ipynb                     # Fine-tuning experiments
│   └── result_analysis.ipynb                 # Results analysis
│
├── results/                                  # All results
│   ├── implicit_judgment/                    # Implicit judgment results
│   │   └── implicit_contrast/                # Contrastive analysis results
│   │
│   ├── explicit_judgment/                    # Explicit judgment results
│   │   ├── reasoning/                        # Reasoning-based results
│   │   └── llm_based/                        # LLM-based results
│   │
│   ├── finetune/                             # Fine-tuning results
│   │   ├── fraud_results_final.json          # Results for fraud domain
│   │   ├── politics_results_final.json       # Results for politics domain
│   │   ├── pornography_sexual_minors_results_final.json
│   │   ├── race_results_final.json           # Results for race domain
│   │   ├── religion_results_final.json       # Results for religion domain
│   │   ├── suicide_results_final.json        # Results for suicide domain
│   │   ├── terrorism_results_final.json      # Results for terrorism domain
│   │   └── violence_results_final.json       # Results for violence domain
│   │
│   ├── harmbench/                            # HarmBench evaluation results
│   │   ├── before_finetune.json              # Results before fine-tuning
│   │   └── finetuned_harmbench.json          # Results after fine-tuning
│   │
│   └── visualizations/                       # Visualization results
│       ├── graphs/                           # Main graphs
│       └── additional_graphs/                # Additional visualizations
│
├── models/                                   # Model checkpoints and configs
│   └── lora_adapter/                         # LoRA adapter files
│
├── scripts/                                  # Utility scripts
│   ├── setup.sh                              # Environment setup script
│   └── download_data.sh                      # Data download script
│
├── requirements.txt                          # Project dependencies
├── LICENSE                                   # License file
└── README.md                                 # Project documentation
```

## Datasets and Models

### Adversarial Dataset
An extensive collection of 8.87k jailbreak prompts covering various harmful content categories:
- [HuggingFace Dataset: bebop404/adversarial](https://huggingface.co/datasets/bebop404/adversarial)

### Fine-tuned Model
A LoRA adapter for LLAMA-3.3-70B that enhances safety while preserving capabilities:
- [HuggingFace Model: bebop404/my-lora-adapter](https://huggingface.co/bebop404/my-lora-adapter)

## Methodology

### Implicit Judgment
Measures LLAMA's default capability to handle both overtly harmful questions and more subtly phrased ones without assistance.

### Explicit Judgment
Applies structured reasoning to model outputs, identifies key statements, and assigns severity scores indicating harmfulness levels.

### Fine-tuning with LoRA
Uses parameter-efficient Low-Rank Adaptation to directly address weaknesses identified in LLAMA's safety behavior.

## Results

The research demonstrates:
- Stark disparity in LLAMA's baseline safety performance (e.g., ~93% refusal for explicit self-harm vs. only 20-24% for nuanced harmful content)
- Significant improvement in refusal rates across all categories after fine-tuning (from 61.25% to 91.69% overall)
- Enhanced robustness against novel jailbreak attempts without sacrificing performance on benign queries

## Tutorial

### Prerequisites

Before starting, ensure you have the following:
- Python 3.9 or higher
- Git
- CUDA-compatible GPU (recommended for fine-tuning)

### Setting Up the Environment

1. Clone this repository:
   ```bash
   git clone https://github.com/TinkAnet/Enhancing_LLAMA-Against_Language_Jailbreaks.git
   cd Enhancing_LLAMA-Against_Language_Jailbreaks
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv .venv
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

3. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running Evaluations

#### Implicit Judgment Evaluation

This evaluates how LLAMA responds to harmful prompts without any assistance:

1. Add your API key to the script (or set it as an environment variable):
   ```bash
   # Set environment variable
   export LLAMA_API_KEY="your_api_key_here"
   ```

2. Run the implicit judgment script:
   ```bash
   python Implicit_judgment_llama.py
   ```

#### Explicit Judgment Evaluation

This adds structured reasoning to analyze model outputs:

```bash
python explicit_judgment_llama.py
```

#### Analyzing Results

The project includes statistical analysis tools:

```bash
python Statistic.py  # For implicit judgment analysis
python Statistic_explicit.py  # For explicit judgment analysis
```

### Fine-tuning with LoRA

To fine-tune the model with Low-Rank Adaptation:

1. Navigate to the fine-tuning directory:
   ```bash
   cd finetune/LoRa
   ```

2. Start a Jupyter server and run the fine-tuning notebook:
   ```bash
   jupyter notebook
   # Open and run fine-tune.ipynb
   ```

### Using the Fine-tuned Model

Load the LoRA adapter from HuggingFace:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel, PeftConfig

# Load the base model
model_name = "meta-llama/Llama-3.3-70B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# Load the LoRA adapter
adapter_name = "bebop404/my-lora-adapter"
model = PeftModel.from_pretrained(model, adapter_name)
```

### Evaluating the Fine-tuned Model

Test the model against the HarmBench dataset:
```bash
python finetune/implicit_harmbench.py
```

### Using the Adversarial Dataset

Access the dataset from HuggingFace:
```python
from datasets import load_dataset
dataset = load_dataset("bebop404/adversarial")
```

### Troubleshooting

- **CUDA Out of Memory**: Reduce batch size in fine-tuning notebooks
- **API Rate Limits**: Add delay between API calls in judgment scripts
- **Missing Dependencies**: Check requirements.txt and install any missing packages

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