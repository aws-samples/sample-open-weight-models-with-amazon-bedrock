# Optimize Open Weight Models for Low-Latency, Cost-Effective AI Apps

This repository accompanies a hands-on workshop focusing on demonstrating the capabilities and advantages of open-weight models on Amazon Bedrock - first delivered at re:Invent 2025 (Builder Session AIM311).

## Overview

Open-weight models deliver exceptional performance while offering customization control. Organizations can process sensitive data locally, deploy models tailored to specific requirements, and scale efficiently at lower latency and cost. However, maximizing these benefits requires strategic decisions: Poor choices waste resources and compromise results. This session provides a practical framework for using open-weight models in Amazon Bedrock. Learn to evaluate and select the ideal model for your specific use cases, understand the trade-offs between different models and sizes, and identify deployment patterns that balance cost and latency. We'll demonstrate optimization techniques and architect solutions for real-world workloads, including agentic applications.

Refer to the [workshop instructions](https://catalog.us-east-1.prod.workshops.aws/workshops/121757da-34db-4eb3-b86d-c1bb24e85a14) for guidance on how to follow along in an AWS-hosted event.

## Pillars for LLM Model Evaluation

To select the right model for a use-case, it's important to evaluate which option performs best from a holistic perspective: Considering factors like speed and price as well as whatever measures of output quality or accuracy apply to your project.

We propose a set of pillars to guide holistic evaluation:

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

## Getting Started

For AWS-hosted events, we typically test and run these notebooks in [JupyterLab](https://docs.aws.amazon.com/sagemaker/latest/dg/studio-updated-jl.html) on [SageMaker AI Studio](https://aws.amazon.com/sagemaker/ai/studio/). However, they may run in other environments too so long as you install the required dependencies and configure your [AWS CLI credentials](https://docs.aws.amazon.com/cli/latest/userguide/cli-configure-files.html).

Your environment will need Python (v3.12+).

If running on a multi-project environment, you may wish to set up a [virtual environment](https://docs.python.org/3/library/venv.html):

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

Install the required libraries (from [pyproject.toml](pyproject.toml) into your Python environment:
```bash
pip install .
```

### Lab 1: Model Selection & API Comparison
Compare APIs and open-weight models (NVIDIA Nemotron, GPT OSS, Qwen, DeepSeek, Kimi K2.5) to showcase Amazon Bedrock's capabilities.

**Files:**
- [`lab1/Lab1a_-_Model_Selection_Framework.ipynb`](lab1/Lab1a_-_Model_Selection_Framework.ipynb)
- [`lab1/Lab1b_-_API_Integration_Options.ipynb`](lab1/Lab1b_-_API_Integration_Options.ipynb)

### Lab 2: Performance Evaluation
Evaluate quality, latency, and accuracy metrics with focus on tool calling and agentic tasks using automated and LLM-as-a-Judge methodology.

**Files:**
- [`lab2/Lab2_Automatic_Evaluation.ipynb`](lab2/Lab2_Automatic_Evaluation.ipynb)
- [`lab2/Lab2Extension_Classical_Metrics.ipynb`](lab2/Lab2Extension_Classical_Metrics.ipynb)

### Lab 3: Automatic Prompt Optimization
Automatically optimize prompt templates to increase the quality of responses between different models - enabling fairer comparison of different models' capabilities without needing to tune prompts manually for each one.

**Files:**
- [`lab3/Lab3_Prompt_Optimization.ipynb`](lab3/Lab3_Prompt_Optimization.ipynb)

### Lab 4: Agent Optimization with AgentCore
Observe, evaluate, and optimize your AI agents deployed on [Amazon Bedrock AgentCore](https://aws.amazon.com/bedrock/agentcore/) to deliver high-quality results even when using cost-optimized models.

**Files:**
- [`lab4/Lab4_AgentCore_Optimization.ipynb`](lab4/Lab4_AgentCore_Optimization.ipynb)

## Technical Resources

### Benchmarking & Evaluation
- [Model Latency Benchmarking](https://github.com/aws-samples/amazon-bedrock-samples/tree/main/model-latency-benchmarking)
- [Automatic Model Evaluation](https://github.com/aws-samples/Meta-Llama-on-AWS/blob/main/model-evaluation/Amazon%20Bedrock/Automatic_model_evaluation_v2.ipynb)
- Amazon Bedrock AgentCore [introductory workshop](https://catalog.workshops.aws/agentcore-getting-started), [developer guide](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html), and [code samples](https://github.com/awslabs/agentcore-samples)


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
