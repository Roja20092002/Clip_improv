# Multimodal Hateful Meme Detection using Multi-Task Learning

## Overview

This project detects hateful memes by jointly analyzing image and text content using a Transformer-based Multi-Task Learning (MTL) architecture.

The system combines:

- CLIP for multimodal representation learning
- BERT for textual understanding
- ViT (Vision Transformer) for visual feature extraction
- Multi-Task Learning framework for improved generalization across hate-related tasks

## Features

- Multimodal meme understanding
- Hate speech classification
- Multi-task learning architecture
- Transformer-based feature extraction
- Confusion matrix and performance visualization
- Evaluation on multiple hateful meme datasets

## Architecture

Text Input
↓
BERT Encoder
↓
Feature Fusion → MTL Network → Classification

Image Input
↓
CLIP / ViT Encoder
↓


## Project Structure

```text
HATEFUL_MEME_DETECTION
│
├── assets/
├── configs/
├── evaluation/
├── models/
├── training/
├── utils/
│
├── test_mami_dataset.py
├── test_mmhs150k_dataset.py
├── test_mtl_forward.py
├── test_multioff_dataset.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

## Technologies Used

- Python
- PyTorch
- Hugging Face Transformers
- CLIP
- BERT
- Vision Transformer (ViT)
- NumPy
- Pandas
- Scikit-Learn

## Datasets

- MAMI
- MMHS150K
- MultiOFF

## Results

Example outputs and confusion matrices are available in the `assets/` directory.

## Installation

```bash
git clone <repo-url>
cd HATEFUL_MEME_DETECTION

pip install -r requirements.txt
```

## Run Training

```bash
python training/train.py
```
