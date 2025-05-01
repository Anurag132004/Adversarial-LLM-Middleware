# 🔬 Research Findings: Perturbation Analysis on Mistral LLM

This module is part of the **Adversarial-LLM-Middleware** project and focuses on evaluating how large language models (LLMs) like **Mistral-7B** respond to adversarially perturbed prompts.

---

## 🎯 Objective

To test the **robustness and sensitivity** of a fine-tuned LLM (Mistral-7B QLoRA) by:
- Introducing character-level, word-level, and semantic perturbations to input questions
- Comparing original and perturbated responses
- Recording per-token **logits** and **softmax probabilities** for detailed statistical analysis

---

## 🧪 Script: generate_perturbations.py

### ✔️ Functionality:
- Loads previously generated LLM outputs from mistral_output.jsonl
- Perturbs each question (without using NLP libraries or embeddings)
- Sends the perturbated prompt to the model
- Records the new response, token count, and logits
- Saves a combined result to: `outputs/mistral_output_with_perturbations.jsonl`

---

## ⚙️ Perturbation Techniques Used

| Category | Technique | Example |
|------------------|----------------------------------|-------------------------------------|
| Character-Level | Homoglyph substitution | France → Frànce |
| | Typos (deletion, duplication) | capital → capitl, city → ccity |
| Word-Level | Stopword drop / filler insertion | What is actual the capital... |
| Semantic-Level | Negation or logic reversal | What is NOT the capital of... |

All perturbations are randomized using predefined weights for diversity.

---

## 📦 Dependencies

Run in a Python 3.8+ GPU environment with:
```bash
pip install torch transformers accelerate peft datasets huggingface_hub
```

## 📁 Input File Format (`mistral_output.jsonl`)

Each line should be a JSON object like:
```json
{
  "question": "...",
  "answer": "...",
  "ground_truth": "...",
  "no_of_tokens": ...,
  "logits_data": [...]
}
```

## 📁 Output File Format (`mistral_output_with_perturbations.jsonl`)

Each line will include:
```json
{
  "question": "...",
  "ground_truth": "...",
  "response": "...",
  "no_of_tokens_in_response": ...,
  "logits_data": [...],
  "perturbated_question": "...",
  "perturbated_response": "...",
  "no_of_tokens_in_perturbated_response": ...,
  "perturbated_logits_data": [...]
}
```

## 🖥️ Environment

This script is optimized for:
* **Kaggle Notebook with 80GB A100 GPU**
* Or local machines with GPU (adjust `MAX_MEMORY` if needed)