# AI-Powered Mathematical Olympiad Problem Solver

## Table of Contents

- [Introduction](#introduction)
- [Data Collection](#data-collection)
- [Data Preprocessing](#data-preprocessing)
- [Dataset Structure](#dataset-structure)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Model Training and Evaluation](#model-training-and-evaluation)
- [Finetuning Deepseekmath](#finetuning-deepseekmath)
- [Self-Consistency Chain Of Thoughts (SC-CoT)](#self-consistency-chain-of-thoughts-sc-cot)
- [Results](#results)
- [Conclusion](#conclusion)

## Introduction

### Objective

- Develop a model that accurately solves mathematical problems from the AI Mathematical Olympiad competition.

### Goal

- Exceed the performance of the current Gemma 7B benchmark of 3/50 and achieve at least 20% accuracy.

### Key Components

- **Problem Interpretation**: Develop a system that accurately interprets and understands natural language descriptions of Math Olympiad problems.
- **Mathematical Reasoning**: Implement algorithms to process and reason through mathematical concepts, theorems, and logic.
- **Evaluation**: Establish a robust evaluation mechanism to assess the accuracy and reliability of the solutions provided by the model.

### Expected Outcome

By the end of this project, we anticipate having a robust model capable of solving a wide range of Math Olympiad problems with high accuracy. This model will not only serve as a powerful tool for students and educators but also pave the way for further advancements in AI-driven mathematical problem-solving.

## Data Collection

### Sources

- **Kaggle**: Problems and solutions contributed by the Kaggle community.
- **American Invitational Mathematics Examination (AIME)**: Problems providing a high standard of difficulty essential for training the model.

## Data Preprocessing

- **Data Merge**: Combined datasets into a single comprehensive dataset.
- **Normalization**: Standardized formatting and notation across all records.
- **Question Filtering**: Selected questions with integer answers.
- **Answer Extraction**: Retrieved integer answers where answers were missing or not provided.
- **Cleaning**: Removed extraneous information to maintain data quality and relevance.

## Dataset Structure

The training dataset consists of approximately 9000 records, each containing three main components:

- **Questions**: Math Olympiad problems presented in natural language, covering algebra, geometry, number theory, and combinatorics.
- **Solutions**: Detailed step-by-step solutions illustrating the logical and mathematical reasoning required to arrive at the correct answer.
- **Answer**: The final integer answer.

## Exploratory Data Analysis (EDA)

- **Basic Information**: Total entries, unique questions, solutions, and answers.
- **Statistics**: Mean, standard deviation, minimum, and maximum lengths of questions and solutions.
- **Visualizations**: Histograms and word clouds of question and solution lengths and common terms.

## Model Training and Evaluation

### Techniques

- **Zero-Shot Learning**
- **Few-Shot Learning**
- **Retrieval-Augmented Generation (RAG)**
- **Parameter-Efficient Fine-Tuning (PEFT)**
- **Fine-Tune + Self Consistency - Chain of Thoughts(SC-CoT)**

### Evaluation

- Accuracy assessed using a subset of 10 complex Olympiad problems.

## Finetuning Deepseekmath

### Training Setup

- **Model**: Deepseek Math Model (Causal LM with 4-bit quantization).
- **Training Parameters**: Batch size: 1, Gradient accumulation steps: 4, Epochs: 1, Learning rate: 1e-4, Optimizer: Paged AdamW 8-bit, Scheduler: Cosine, Warmup ratio: 0.01.
- **Training**: Define parameters, use weights and biases for tracking and train the model

### Evaluation

- **Test Accuracy**: 10% on 10 unseen problems.

## Self-Consistency Chain Of Thoughts (SC-CoT)

- Enhanced reasoning capability through self-consistency, sampling diverse reasoning paths using different decoding strategies and integrating SymPy for calculations.

### Configuration Settings

- **n_repetitions**: 15
- **TOTAL_TOKENS**: 2048
- **TIME_LIMIT**: 31,500 seconds
- **MODEL**: Fine-tuned DeepSeek math model with memory optimization

## Results

- **Accuracy**: 10% on 10 unseen Olympiad problems.
- Zero Shot 0% accuracy
- Few Shot 0% accuracy
- RAG 10% accuracy
- FINE TUNE(QLORA) 10% accuracy
- FINE TUNE+ SC-COT 20% accuracy
- Further analysis required to determine areas for improvement in logical development and execution.

## Conclusion

- Successfully engineered an AI model approaching initial accuracy goals.
- Leveraged advanced reasoning methods, including Zero-Shot, Few-Shot, RAG, and Finetuning.
- Future efforts to refine the model's precision and broaden its application spectrum for educational impact.
- Advanced techno7iques like FINE TUNE+ SC-COT achived high accuracy
- Zero Shot and Few Shot achived or showed low performance
