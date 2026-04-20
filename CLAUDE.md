# CLAUDE.md

## Project Overview

AWS Builders Session **AIM311** — "Optimizing open weight models for low-latency, cost-effective AI apps" (L300).
Originally built for re:Invent 2025, adapted for **AWS Toronto Summit (June 3, 2026)**.

## Model Lineup (Toronto Summit)

### Lab 1 (Model Selection & API Integration)

| Model | Model ID | Type | Context |
|-------|----------|------|---------|
| NVIDIA Nemotron 3 Super | `nvidia.nemotron-super-3-120b` | Text, 120B MoE (12B active) | 128K |
| NVIDIA Nemotron Nano 30B | `nvidia.nemotron-nano-3-30b` | Text | 128K |
| Kimi K2.5 | `moonshotai.kimi-k2.5` | Multimodal (text + vision) | 128K |
| GPT OSS 120B | `openai.gpt-oss-120b-1:0` | Text | 128K |
| GPT OSS 20B | `openai.gpt-oss-20b-1:0` | Text | 128K |
| Qwen3 235B MoE | `qwen.qwen3-235b-a22b-2507-v1:0` | Text | 128K |
| Qwen3 32B | `qwen.qwen3-32b-v1:0` | Text | 128K |
| DeepSeek V3.2 | `deepseek.v3.2` | Text | 128K |
| Qwen3 Coder 480B (bonus) | `qwen.qwen3-coder-480b-a35b-v1:0` | Text | 256K |
| Qwen3 Coder 30B (bonus) | `qwen.qwen3-coder-30b-a3b-v1:0` | Text | 256K |

### Lab 2a (Automatic Evaluation)
- Qwen3 32B (`qwen.qwen3-32b-v1:0`) and GPT OSS 20B (`openai.gpt-oss-20b-1:0`)

### Lab 2b (LLM-as-a-Judge)
- Generator: NVIDIA Nemotron Nano 9B v2 (`nvidia.nemotron-nano-9b-v2`) and Mistral 7B Instruct (`mistral.mistral-7b-instruct-v0:2`)
- Evaluator: Mistral Large (`mistral.mistral-large-2402-v1:0`)

## Repository Structure

```
lab1/
  Lab1a_-_Model_Selection_Framework.ipynb   # Use-case-driven model selection, pricing, benchmarks
  Lab1b_-_API_Integration_Options.ipynb     # Invoke API, Converse API, ChatCompletions API
  extract_bedrock_pricing.py                # Pulls live pricing from AWS Pricing API
  llm_compare_jupyter_clean.py              # LLM comparison helper (latency, cost, throughput)
  img/                                      # Slide images

lab2/
  Lab2a_-_Automatic_model_evaluation.ipynb  # Bedrock Automatic Model Evaluation jobs
  Lab2b_-_LLM_as_a_judge_evaluation.ipynb   # LLM-as-Judge with custom datasets
  requirements.txt
```

## Key Technical Details

- **Region**: Default `us-west-2`; some models need region fallback
- **All new models use ON_DEMAND inference** (no inference profile prefix needed)
- **APIs used**: Bedrock Invoke, Converse, and ChatCompletions (OpenAI-compatible)
- **Multimodal demo** in Lab 1b uses **Kimi K2.5** (the only multimodal model in the lineup)
- **Dependencies**: boto3, rich, pandas, Pillow, openai, langchain, langgraph, crewai, sagemaker, seaborn, matplotlib
- **Pricing helper** (`extract_bedrock_pricing.py`): Has manual model-ID-to-name mappings — update when adding/removing models
- **Comparison tool** (`llm_compare_jupyter_clean.py`): Model IDs passed directly

## Conventions

- Notebooks use Jupyter `.ipynb` format with markdown cells for instructions
- Helper modules are plain Python files imported by notebooks
- Lab 2 notebooks create AWS resources (S3 buckets, IAM roles, evaluation jobs)
- Cost analysis sections assume 1M requests/month scenarios
