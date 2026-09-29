<img width="1277" height="565" alt="Screen Shot 2026-09-28 at 11 18 05 AM" src="https://github.com/user-attachments/assets/e1e462ad-7ced-44f0-bb64-5dffca99d55e" />

```mermaid
flowchart TD
    A["Raw corpus<br/>134K Roman Urdu messages"] --> B["Clean and split<br/>129,377 rows, 80/20 split"]
    B --> C["Tokenize<br/>XLM-R tokenizer, 128 tokens"]
    C --> D["Fine-tune XLM-RoBERTa<br/>3 epochs, Kaggle T4 GPU"]
    D --> E["Evaluate<br/>81% accuracy on 25,876"]
    E --> F["Live demo<br/>Text in, sentiment out"]

# Roman Urdu NLP — Sentiment Analysis

    classDef data fill:#F1EFE8,stroke:#5F5E5A,color:#2C2C2A
    classDef prep fill:#EEEDFE,stroke:#534AB7,color:#26215C
    classDef model fill:#E1F5EE,stroke:#0F6E56,color:#04342C
    class A data
    class B,C prep
    class D,E,F model
```
🚀 **[Try the live demo](https://huggingface.co/spaces/FarazAbdulMuqtader/roman-urdu-sentiment-demo)**

A sentiment classification model for Roman Urdu (Urdu written in Latin script), fine-tuned on XLM-RoBERTa. Built to address a low-resource, code-mixed language largely underserved by mainstream NLP tooling.

## Overview

Roman Urdu is the informal script most commonly used across Pakistani social media, messaging, and reviews — yet it's inconsistent, code-mixed with English, and rarely covered by pretrained multilingual models out of the box. This project fine-tunes XLM-RoBERTa on a large Roman Urdu sentiment corpus to classify text as positive, negative, or neutral.

## Dataset

- Source: `Khubaib01/RomanUrdu-NLP-Sentiment-Corpus`
- ~134K raw samples, cleaned down to ~129K after preprocessing

## Pipeline

1. **Preprocessing** — cleaning, normalization, and filtering of the raw corpus (134K → 129K rows).
2. **Tokenization** — XLM-RoBERTa tokenizer applied to the cleaned dataset.
3. **Fine-tuning** — trained on Kaggle's free T4 GPU environment (`03_train_model.py`), with loss tracked across epochs.
4. **Evaluation** — performance measured via scikit-learn's classification report (precision, recall, F1 per class).

## Stack

- Python
- Hugging Face Transformers (XLM-RoBERTa)
- PyTorch
- scikit-learn
- Kaggle (T4 GPU training)

## Results

Fine-tuned on the full ~129K-row dataset (103,501 train / 25,876 held-out test, stratified 80/20 split), 3 epochs on a Kaggle T4 GPU.

**Overall accuracy: 81%**

| Label    | Precision | Recall | F1-score | Support |
|----------|-----------|--------|----------|---------|
| Positive | 0.81      | 0.82   | 0.82     | 7,252   |
| Negative | 0.84      | 0.85   | 0.85     | 10,504  |
| Neutral  | 0.78      | 0.76   | 0.77     | 8,120   |
| **Accuracy** | | | **0.81** | 25,876 |
| Macro avg | 0.81 | 0.81 | 0.81 | 25,876 |
| Weighted avg | 0.81 | 0.81 | 0.81 | 25,876 |

**Confusion matrix** (rows = actual, columns = predicted):

|          | Positive | Negative | Neutral |
|----------|----------|----------|---------|
| Positive | 5,977    | 508      | 767     |
| Negative | 588      | 8,920    | 996     |
| Neutral  | 802      | 1,177    | 6,141   |

Most confusion happens between Neutral and the other two classes — expected, since neutral sentiment in short, code-mixed social text is inherently ambiguous (sarcasm, mild opinions, and rhetorical statements often blur the line). Negative is the most reliably distinguished class.

## Status

Preprocessing, tokenization, fine-tuning, and evaluation are complete on the full dataset. Published on Hugging Face, with a live interactive demo on Spaces.

## Why This Project

Local-language, code-mixed NLP is a niche most large models don't handle well — that gap is the point. Building for Roman Urdu specifically is both a technical exercise and a bet on underserved language data as a differentiator.
