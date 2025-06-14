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

## Homework 6 - Multiclass Text Classification

This notebook implements a multiclass text classifier using scikit-learn's Pipeline and FeatureUnion components, combining word-level and character-level features through TF-IDF vectorization. The implementation includes custom text preprocessing with URL and handle normalization, along with word clustering capabilities to handle out-of-vocabulary words. The model uses logistic regression with balanced class weights and achieves competitive performance on the validation set using balanced accuracy metrics.

## Homework 8 - Node2Vec Implementation

This notebook implements the Node2Vec algorithm for learning node embeddings in social networks, featuring parameter optimization for walk length, dimensions, and bias parameters to capture both structural and homophily relationships. The implementation includes optional node feature integration through feature concatenation and normalization, with evaluation using cosine similarity metrics on target node pairs. The code demonstrates efficient parallel processing through worker threads and handles missing features gracefully through zero padding.

## Final Project - Graph-Based Fake News Detection Using GNNs

This project implements an ensemble-based approach for fake news detection on social media using graph neural networks (GNNs). Working with two team members, we combined Graph Attention Networks (GAT), Heterogeneous Graph Attention Networks (HAN), and Relational Graph Convolutional Networks (RGCN) to leverage structural and relational patterns in news propagation graphs from the User Preference-aware Fake News Detection (UPFD) dataset.

Our key contributions include:
- An ensemble framework combining GAT, HAN, and RGCN architectures to jointly exploit attention over neighbors, heterogeneous node semantics, and relation-specific transformations
- Implementation of four ensemble strategies: voting, averaging, concatenation, and transformation
- Hyperparameter optimization using Optuna to find optimal model configurations for each dataset
- Comprehensive analysis and visualizations of propagation patterns

The model achieved 86.0% accuracy and 86.1% F1-score on PolitiFact and 96.5% accuracy and 96.6% F1-score on GossipCop, outperforming the UPFD baseline by 1.35% accuracy and 1.45% F1-score on PolitiFact. Visualizations of attention mechanisms and outlier graphs provided insights into model behavior, highlighting the importance of structural features in distinguishing fake from real news.


# Course Overview

This repository contains my work for the Social Computing course taught by [Hadi Amiri](https://scholar.google.com/citations?user=fyUpZ5EAAAAJ&hl=en). 


The course included both theoretical paper reviews and practical programming assignments. While some homeworks focused on mathematical proofs and paper analysis (not shown here), the programming assignments demonstrated hands-on implementation of key social computing concepts.

I achieved an A in this course, which provided a strong foundation in analyzing social networks and implementing computational methods for social data analysis. The course culminated in a final project applying graph neural networks to fake news detection, combining multiple architectures to achieve SOTA comparable results.


