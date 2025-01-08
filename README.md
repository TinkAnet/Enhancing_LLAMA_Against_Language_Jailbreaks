# Enhancing_LLAMA_Against_Language_Jailbreaks

This repository contains datasets for evaluating and enhancing the safety mechanisms of LLAMA against harmful and ambiguous prompts. The datasets are part of a capstone project at The Hong Kong Polytechnic University.

## Results Datasets

### Explicit Judgment Dataset
The **explicit judgment dataset** is available in the [`SAP200_explicit_judgment`](https://github.com/TinkAnet/Enhancing_LLAMA_Against_Language_Jailbreaks/tree/main/SAP200_explicit_judgment) folder.  
This dataset contains results from the explicit judgment phase, where structured reasoning was applied to analyze model responses. The data is categorized into eight domains:
- Fraud
- Politics
- Pornography/Sexual Minors
- Race
- Religion
- Suicide
- Terrorism
- Violence

Each file includes:
- `prompt`: The harmful input.
- `implicit_response`: The model's response during implicit judgment.
- `explicit_instruction`: Guidance for analyzing or rejecting the prompt.
- `explicit_response`: The ideal response from explicit judgment.
- `severity_score`: A score from 1 (safe) to 5 (very harmful).
- `verdict`: The final classification (e.g., A, B, C, Uncategorized).

---

### Implicit Judgment Dataset
The **implicit judgment dataset** is available in the [`SAP200_implicit_judgment`](https://github.com/TinkAnet/Enhancing_LLAMA_Against_Language_Jailbreaks/tree/main/SAP200_implicit_judgment) folder.  
This dataset includes results from the implicit judgment phase, where the model's baseline responses to harmful prompts were collected and categorized. The data is organized into the same eight domains as the explicit dataset.

Each file includes:
- `prompt`: The harmful input.
- `response`: The model's default response without additional guidance.
- `category`: The domain of the harmful prompt.
- `rejected`: Whether the model rejected the harmful input (true/false).

---

## File Format

The datasets are stored in JSON format and can be directly loaded for analysis or fine-tuning purposes.

---

## Acknowledgments

This project is supervised by Dr. Lou Wei at The Hong Kong Polytechnic University.
