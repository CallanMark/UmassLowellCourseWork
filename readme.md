# Deep Learning for Natural Language Processing 

## Homework 1 - Linear Classifier for Text Classification

This notebook implements a linear classifier for text classification using the IMDB dataset, exploring text vectorization techniques including CountVectorizer and TfidfVectorizer. The implementation demonstrates handling of out-of-vocabulary words and evaluates model performance using accuracy metrics. The code includes data preprocessing, feature extraction, and model training with cross-validation.

## Homework 2 - Word2Vec Implementation

This notebook implements the Word2Vec model from scratch using PyTorch, demonstrating the Skip-gram architecture and efficient vector operations through matrix multiplication. The implementation includes optimizations for training speed and numerical stability, with evaluation of word similarities on a Star Wars text corpus. The model successfully learns meaningful word embeddings that capture semantic relationships between words in the corpus.

## Homework 3 - Neural Networks and SVM

This notebook implements a two-layer neural network and SVM classifier for image classification on CIFAR-10, exploring different optimization techniques and hyperparameter tuning for improved model performance. The implementation includes forward and backward propagation, loss computation, and gradient updates for both architectures. The models are evaluated using accuracy metrics and compared in terms of training time and classification performance.

## Homework 4 - Neural Network Text Classifier

This notebook implements a fully-connected neural network classifier for text classification using PyTorch. The implementation includes techniques like batch normalization, dropout, and L2 regularization, with a focus on proper project structure and model evaluation. The code demonstrates text preprocessing, model training with early stopping, and interactive inference capabilities.

## Homework 5 - Transformer Language Model

This notebook implements a transformer-based language model from scratch using PyTorch, starting with the implementation of multi-head self-attention mechanisms and building up to a complete transformer architecture. The implementation includes training on the Tiny Shakespeare dataset, demonstrating character-level language modeling with techniques like causal masking, positional encoding, and efficient batching. The code explores hyperparameter optimization through wandb integration, model evaluation using perplexity metrics, and text generation capabilities.

## Homework 6 - Transformer Machine Translation Model

This notebook implements a transformer-based machine translation model from scratch using PyTorch, building upon the transformer architecture from HW5 to create an encoder-decoder model for English-German translation. The implementation includes training custom tokenizers, implementing cross-attention mechanisms, and handling sequence-to-sequence tasks with proper padding and masking. The model was trained on the WMT-14 English-German dataset using CS department GPU servers, with hyperparameter optimization through wandb integration and evaluation using BLEU scores.

## Course Overview

This repository contains my work for the Deep Learning for Natural Language Processing course taught by [Anna Rumshisky](https://scholar.google.com/citations?user=_Q1uzVYAAAAJ&hl=en). The course covered fundamental concepts in deep learning and NLP, including:

- Distributional semantics and word embeddings (Word2Vec)
- Neural network architectures (MLPs, CNNs, RNNs)
- Attention mechanisms and Transformer models
- Transfer learning and pre-training strategies
- Parameter-efficient fine-tuning (PEFT)
- Model alignment and RLHF
- Long-context handling and position embeddings

For the final project, I gave a 15-minute presentation on the paper "Distilling Step-by-Step! Outperforming Larger Language Models with Less Training Data and Smaller Model Sizes", which explored knowledge distillation techniques for creating more efficient language models.

I achieved a B+ in this course, which provided a comprehensive foundation in modern NLP techniques and architectures. The course culminated in a final exam covering topics from word embeddings to advanced concepts like RLHF and model alignment among others.

# Social Computing 

## Homework 1 - Reddit Data Analysis

This notebook implements a Reddit data collection and analysis system using the PRAW API, featuring a scraper that collects posts and comments based on user queries, and an analyzer that computes statistical metrics including mean scores, variance, and correlation between post and comment scores. The implementation includes data visualization using seaborn and matplotlib to plot score distributions and correlations, with additional functionality to identify top and bottom performing posts based on their scores.

## Homework 2 - Netowork analysis of academic paper citations

This notebook implements a network analysis system using NetworkX to analyze academic paper citation networks, featuring weak and strong tie analysis through degree centrality calculations and visualization of network resilience. The implementation includes centrality analysis using degree, closeness, and betweenness metrics with Pearson correlation calculations, along with network visualization tools to demonstrate the impact of removing different types of ties on network structure.
