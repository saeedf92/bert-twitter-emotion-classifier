# Sentiment Analysis with BERT for Twitter Emotion Classification

This repository provides an end-to-end deep learning pipeline for multi-class sentiment and emotion classification of Twitter posts using **BERT (Bidirectional Encoder Representations from Transformers)**. The project leverages **PyTorch** and the **Hugging Face Transformers** library to fine-tune a pre-trained BERT model on the SMILE Twitter Emotion Dataset.

The workflow covers data preprocessing, tokenization, model training, evaluation, and performance analysis, serving as a practical example of transfer learning for natural language processing (NLP) tasks.

---

## Overview

Emotion detection from social media text is a challenging NLP problem due to the informal and context-dependent nature of online communication. This project demonstrates how transformer-based language models can be adapted for emotion classification by fine-tuning a pre-trained BERT model on annotated Twitter data.

The pipeline includes:

* Exploratory Data Analysis (EDA)
* Data cleaning and label preprocessing
* BERT tokenization and input encoding
* PyTorch dataset and dataloader creation
* Fine-tuning a pre-trained BERT model
* Performance evaluation using classification metrics
* Class-level accuracy analysis

---

## Features

* Fine-tuning of the `bert-base-uncased` model
* Multi-class emotion classification
* Automated tokenization using BERT tokenizer
* GPU-compatible PyTorch training workflow
* Learning rate scheduling and optimization
* Validation and performance tracking
* Reproducible training configuration

---

## Prerequisites

To run this project, you should have:

* Python 3.8+
* Basic knowledge of Python and machine learning
* Familiarity with PyTorch and transformer models

### Required Packages

```bash
pip install torch pandas numpy transformers scikit-learn tqdm
```

---

## Dataset

This project uses the **SMILE Twitter Emotion Dataset**:

> Wang, B., Tsakalidis, A., Liakata, M., Zubiaga, A., Procter, R., & Jensen, E. (2016). *SMILE Twitter Emotion Dataset*. Figshare. https://doi.org/10.6084/m9.figshare.3187909.v2

### Emotion Classes

After preprocessing, the following emotion categories are retained:

| Label | Emotion      |
| ----- | ------------ |
| 0     | happy        |
| 1     | not-relevant |
| 2     | angry        |
| 3     | disgust      |
| 4     | sad          |
| 5     | surprise     |

### Data Cleaning

The following preprocessing steps are applied:

* Remove tweets with multiple emotion labels (e.g., `happy|surprise`)
* Exclude uncoded entries (`nocode`)
* Convert emotion labels into numerical class IDs
* Preserve class distributions during train/validation splitting

---

## Project Workflow

### 1. Exploratory Data Analysis

* Load the dataset using Pandas
* Inspect label distributions
* Clean and standardize annotations

### 2. Data Splitting

* Stratified train-validation split
* Validation set size: 15%

### 3. Tokenization and Encoding

* BERT tokenizer (`BertTokenizer`)
* Maximum sequence length: 256 tokens
* Automatic padding and truncation

### 4. Model Initialization

* `BertForSequenceClassification`
* Pre-trained checkpoint: `bert-base-uncased`
* Output layer adapted for six emotion classes

### 5. DataLoader Construction

* Random sampling for training
* Sequential sampling for validation
* Efficient batch processing with PyTorch

### 6. Training Setup

* Optimizer: AdamW
* Learning rate scheduler with linear decay
* GPU acceleration when available

### 7. Model Training

* Forward and backward propagation
* Gradient updates
* Validation loss monitoring
* Model checkpoint evaluation

### 8. Performance Evaluation

* Weighted F1-score
* Overall validation accuracy
* Per-class accuracy analysis

---

## Training Configuration

| Parameter               | Value             |
| ----------------------- | ----------------- |
| Model                   | bert-base-uncased |
| Epochs                  | 10                |
| Learning Rate           | 1e-5              |
| Optimizer               | AdamW             |
| Training Batch Size     | 4                 |
| Validation Batch Size   | 32                |
| Maximum Sequence Length | 256               |
| Random Seed             | 17                |

---

## Validation Results

Class-wise validation performance:

| Emotion         | Accuracy  |
| --------------- | --------- |
| 😊 Happy        | 163 / 171 |
| 😐 Not Relevant | 20 / 32   |
| 😡 Angry        | 7 / 9     |
| 🤢 Disgust      | 0 / 1     |
| 😢 Sad          | 4 / 5     |
| 😲 Surprise     | 2 / 5     |

> Note: Some classes contain very few validation samples, making performance estimates less reliable for those categories.

---

## Future Improvements

* Hyperparameter optimization
* Class imbalance handling
* Cross-validation experiments
* Comparison with RoBERTa and DistilBERT
* Confusion matrix visualization
* Model deployment using Hugging Face Spaces or Streamlit

---

## References

* Hugging Face Transformers Documentation
* BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
* McCormickML BERT Fine-Tuning Tutorial
* Hugging Face `run_glue.py` training examples

---

## Acknowledgments

This project was part of the Coursera Guided Project "Sentiment Analysis with Deep Learning using BERT" created by Ari Anastassiou through the Coursera Project Network. The project provided the foundation for understanding transformer-based natural language processing, BERT fine-tuning, and emotion classification using Twitter data.

I would like to thank Ari Anastassiou, Coursera, and the open-source communities behind PyTorch and Hugging Face Transformers for making deep learning tools and educational resources accessible to learners.
