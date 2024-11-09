# LongLoRA Reimplementation for Water Management Policy Analysis

## Overview

This project reimplements and adapts the LongLoRA (Long Low-Rank Adaptation) methodology for efficient fine-tuning of long-context large language models, specifically applied to water management policy analysis. It extends the context length of pre-trained LLMs using Shifted Sparse Attention (S2-Attn) and an improved Low-Rank Adaptation (LoRA) framework.

## Features

- Extends context length of LLaMA2 models up to 32,768 tokens
- Implements Shifted Sparse Attention (S2-Attn) for efficient processing of long sequences
- Utilizes Low-Rank Adaptation (LoRA) for parameter-efficient fine-tuning
- Optimized for water management policy analysis tasks

## Requirements

- Python 3.8+
- PyTorch 1.9+
- Transformers 4.21+
- NVIDIA GPU with at least 24GB memory (e.g., NVIDIA A30)

  ## Project Structure

- `src/`: Source code for the LongLoRA implementation
- `model.py`: LongLoRA model architecture
- `attention.py`: Implementation of Shifted Sparse Attention
- `lora.py`: Low-Rank Adaptation modules
- `train.py`: Script for fine-tuning the model
- `evaluate.py`: Script for evaluating the model on test data
- `utils/`: Utility functions and data processing scripts

## Results

Our reimplementation achieves:
- Context length extension up to 32,768 tokens
- Improved performance on water management policy analysis tasks
- Efficient utilization of GPU resources through gradient checkpointing

For detailed results and analysis, please refer to our paper (link to be added).

