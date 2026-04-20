# AWS Toronto Summit 2026 - AIM311 Builder Session

## Repo Overview
This repository contains materials for the AWS Toronto Summit 2026 Bedrock Open Weight Model Builder session, focusing on demonstrating the capabilities and advantages of open-weight models on Amazon Bedrock.

Originally developed for re:Invent 2025 and updated for Toronto Summit (June 3, 2026).

### Session Title
Optimize open weight models for low-latency, cost-effective AI apps

### Session Abstract
Open-weight models deliver exceptional performance while offering customization control. Organizations can process sensitive data locally, deploy models tailored to specific requirements, and scale efficiently at lower latency and cost. However, maximizing these benefits requires strategic decisions—poor choices waste resources and compromise results. This session provides a practical framework for using open-weight models in Amazon Bedrock. Learn to evaluate and select the ideal model for your specific use cases, understand the trade-offs between different models and sizes, and identify deployment patterns that balance cost and latency. We'll demonstrate optimization techniques and architect solutions for real-world workloads, including agentic applications.

### Session Details
- **Session ID:** AIM311
- **Content Level:** L300
- **Key Messages**
  - **Low latency** across models
  - **Cost comparison** between different model options
  - **Accuracy differences** between standard and fine-tuned models

## Pillars for LLM Model Evaluation

### 1. Operational Metrics (coverd by Lab 1 & Lab 2)
- **Cost per token processed**: Economic efficiency of model usage
- **Latency**: Response time and processing speed, like time to first token
- **Throughput**: Number of requests handled per unit time

### 2. Features & Usability (covered by Lab 1)
- **Context window size**: Maximum input length the model can process
- **Integrations**: Compatibility with existing systems and workflows
- **Ecosystem tools**: Supporting libraries, frameworks, and utilities
- **Multimodality**: Support for text, images, audio, and other data types

### 3. Performance & Quality (covered by Lab 2)
- **Reasoning ability**: Model's capacity for logical thinking and problem-solving
- **Accuracy**: Correctness of responses and factual information
- **Creativity**: Ability to generate novel and innovative content
- **Language**: Quality of language generation and comprehension
- **Adaptability**: Flexibility to handle diverse tasks and contexts
- **Fine-tuning or custom training options**: Customization capabilities

## Repo Setup and Flow

### Create Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Requirements
```bash
pip install -r requirements.txt
```

### Lab 1: Model Selection & API Comparison
Compare APIs and open-weight models (NVIDIA Nemotron, GPT OSS, Qwen, DeepSeek, Kimi K2.5) to showcase Amazon Bedrock's capabilities.

**Files:**
- [`lab1/Lab1a_-_Model_Selection_Framework.ipynb`](lab1/Lab1a_-_Model_Selection_Framework.ipynb)
- [`lab1/Lab1b_-_API_Integration_Options.ipynb`](lab1/Lab1b_-_API_Integration_Options.ipynb)

### Lab 2: Performance Evaluation
Evaluate quality, latency, and accuracy metrics with focus on tool calling and agentic tasks using automated and LLM-as-a-Judge methodology.

**Files:**
- [`lab2/Lab2a_-_Automatic_model_evaluation.ipynb`](lab2/Lab2a_-_Automatic_model_evaluation.ipynb)
- [`lab2/Lab2b_-_LLM_as_a_judge_evaluation.ipynb`](lab2/Lab2b_-_LLM_as_a_judge_evaluation.ipynb)

## Technical Resources

### Benchmarking & Evaluation
- [Model Latency Benchmarking](https://github.com/aws-samples/amazon-bedrock-samples/tree/main/model-latency-benchmarking)
- [Automatic Model Evaluation](https://github.com/aws-samples/Meta-Llama-on-AWS/blob/main/model-evaluation/Amazon%20Bedrock/Automatic_model_evaluation_v2.ipynb)



## Models Covered

### Lab 1 (Model Selection & API Integration)
- **NVIDIA Nemotron 3 Super** (120B MoE) — Complex reasoning and agentic workflows
- **NVIDIA Nemotron Nano 30B** — Cost-effective general-purpose
- **Kimi K2.5** (Moonshot AI) — Multimodal (text + vision)
- **GPT OSS 120B / 20B** (OpenAI) — Reasoning and cost-effective tiers
- **Qwen3 235B MoE / 32B** — Reasoning with thinking mode / enterprise
- **DeepSeek V3.2** — Cost-optimized high-volume
- **Qwen3 Coder 480B / 30B** (bonus) — Code generation

### Lab 2 (Evaluation)
- **Qwen3 32B** & **GPT OSS 20B** — Automatic Model Evaluation (Lab 2a)
- **NVIDIA Nemotron Nano 9B v2** & **Mistral 7B Instruct** — LLM-as-a-Judge (Lab 2b)
- **Mistral Large** — Evaluator model (Lab 2b)


