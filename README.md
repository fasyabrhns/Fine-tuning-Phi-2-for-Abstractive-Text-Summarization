# Fine-tuning Phi-2 for Abstractive Text Summarization

## 👥 Team Information

**Course:** Deep Learning  
**Institution:** Telkom University  

| Name | NIM |
| :--- | :--- |
| **Fasya Burhanis Syauqi** | 1103223054 |
| **Muhammad Muhammad Farhan** | [ISI NIM DISINI] |

---

## 🎯 Purpose

This repository contains the implementation of **Task 3: Generative AI**, specifically demonstrating the **instruction-tuned fine-tuning** of Microsoft's **Phi-2** language model (2.7B parameters). 

The goal is to adapt this Small Language Model (SLM) for **Abstractive Text Summarization** using the **XSum dataset**. We utilize **QLoRA** (Quantized Low-Rank Adaptation) to achieve memory-efficient training on consumer-grade GPUs (like T4 in Google Colab).

## 🔍 Project Overview

### The Task: Abstractive Summarization
Unlike extractive summarization (which copies sentences), this model learns to generate concise, human-like summaries by understanding the semantic meaning of the source text.

### The Model: Microsoft Phi-2
* **Parameters:** 2.7 Billion
* **Architecture:** Decoder-only Transformer
* **Why Phi-2?** It offers "textbook-quality" reasoning capabilities in a compact size, making it ideal for fine-tuning experiments without requiring massive H100 GPUs.

### The Dataset: XSum
* **Source:** BBC News articles.
* **Target:** Single-sentence professional summaries.
* **Challenge:** Requires high-level abstraction and rewriting skills.

---

## 📊 Technical Approach

### Model Configuration
* **Base Model:** `microsoft/phi-2`
* **Quantization:** 4-bit NormalFloat (NF4) via `bitsandbytes` to reduce memory usage.
* **Fine-tuning Method:** PEFT (Parameter-Efficient Fine-Tuning) using **LoRA**.

### Training Hyperparameters
* **LoRA Rank (r):** 16
* **LoRA Alpha:** 32
* **Target Modules:** `q_proj`, `k_proj`, `v_proj`, `dense`, `fc1`, `fc2`
* **Optimizer:** Paged AdamW 8-bit
* **Precision:** FP16 (Mixed Precision)

---

## 📁 Repository Structure

```text
.
├── notebooks/
│   └── task-3-deep-learning.ipynb    # Main Jupyter Notebook for Training & Inference
├── requirements.txt                  # Python dependencies
└── README.md                         # Project documentation
