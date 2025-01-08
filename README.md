# Enhancing_LLAMA_Against_Language_Jailbreaks

This repository contains datasets for evaluating and enhancing the safety mechanisms of LLAMA against harmful and ambiguous prompts. The datasets are part of a capstone project at The Hong Kong Polytechnic University.

## Datasets

The `SAP200_explicit_judgment` folder includes JSON files for explicit judgment across eight categories:
- Fraud
- Politics
- Pornography/Sexual Minors
- Race
- Religion
- Suicide
- Terrorism
- Violence

## File Format

Each JSON file contains:
- `prompt`: The harmful input.
- `implicit_response`: The model's response during implicit judgment.
- `explicit_instruction`: Guidance for analyzing or rejecting the prompt.
- `explicit_response`: The ideal response from explicit judgment.
- `severity_score`: A score from 1 (safe) to 5 (very harmful).
- `verdict`: The final classification (e.g., A, B, C, Uncategorized).

---

## Usage

Load the JSON files to analyze model performance or for fine-tuning tasks.

---
