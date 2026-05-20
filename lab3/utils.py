"""Helpers for the Advanced Prompt Optimization (APO) workshop notebooks."""

from __future__ import annotations

from datetime import datetime
import json
import os
import re
import subprocess
from copy import deepcopy
from typing import Any, Callable

# External Dependencies:
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from IPython.display import Markdown
from upath import UPath

DATASETS_ABSPATH = UPath(os.path.abspath(os.path.dirname(__file__))) / "datasets"
DEFAULT_S3_PREFIX = "apo-demo"

bedrock = boto3.client("bedrock")


def resolve_sample_data_template(
    dataset_id: str,
    bucket_name: str,
    s3_datasets_prefix: str = f"{DEFAULT_S3_PREFIX}/datasets",
) -> dict:
    """Read one of the sample datasets and resolve relative S3 asset links to real S3 URIs

    Args:
        dataset_id: ID of a sample dataset, corresponding to a folder name under lab3/datasets
        bucket_name: Name of the S3 bucket being used for the demo
        s3_prefix: Folder prefix in S3 where the contents of lab3/datasets have been uploaded
    """
    
    with (DATASETS_ABSPATH / dataset_id / "sample_data.template.json").open() as f:
        dataset = json.loads(f.read())
    
    for sample in dataset["evaluationSamples"]:
        for mmvar in sample.get("inputVariablesMultimodal", []):
            if len(mmvar.keys()) != 1:
                raise ValueError(
                    "An entry in inputVariablesMultimodal should have exactly one field (the "
                    "variable name). Got: %s" % (len(mmvar.keys()),)
                )
            mmvar_name = next(k for k in mmvar.keys())

            # Replace relative 's3Asset' filename with full S3 URI on our target bucket:
            if "s3Asset" in mmvar[mmvar_name]:
                mmvar[mmvar_name]["s3Uri"] = "/".join([
                    "s3:/",
                    bucket_name,
                    s3_datasets_prefix,
                    dataset_id,
                    mmvar[mmvar_name].pop("s3Asset")
                ])
    return dataset


def print_dataset_without_samples(dataset: dict) -> None:
    dataset = deepcopy(dataset)
    dataset["evaluationSamples"] = ["{OMITTED FROM PRINT-OUT}"]
    print(json.dumps(dataset, indent=2))


def find_apo_job(
    job_arn: re.Pattern | str | None = None,
    job_name: re.Pattern | str | None = None,
    job_status: re.Pattern | str | None = None,
    sort_by: Literal["CreationTime"] = "CreationTime",
    sort_order: Literal["Ascending", "Descending"] = "Descending",
) -> str:
    paginator = bedrock.get_paginator("list_advanced_prompt_optimization_jobs")
    for page in paginator.paginate(sortBy=sort_by, sortOrder=sort_order):
        for summ in page["jobSummaries"]:
            if job_arn:
                if isinstance(job_arn, re.Pattern):
                    if not job_arn.match(summ["jobArn"]):
                        continue
                else:
                    if job_arn != summ["jobArn"]:
                        continue
            if job_name:
                if isinstance(job_name, re.Pattern):
                    if not job_name.match(summ["jobName"]):
                        continue
                else:
                    if job_name != summ["jobName"]:
                        continue
            if job_status:
                if isinstance(job_status, re.Pattern):
                    if not job_status.match(summ["jobStatus"]):
                        continue
                else:
                    if job_status != summ["jobStatus"]:
                        continue
            return summ["jobArn"]
    raise StopIteration("No APO job found matching the given criteria")


def flatten_results(results: list[dict]) -> list[dict]:
    """Flatten result JSONL into one dict per (template, model) result row."""
    rows: list[dict] = []
    for d in results:
        base_template = d.get("promptTemplate")
        template_id = d.get("promptTemplateId") or d.get("templateId")
        for r in d.get("promptOptimizationResults", []):
            original_metrics = r.get("originalPromptMetrics") or {}
            optimized_metrics = r.get("optimizedPromptMetrics") or {}
            row = {
                "templateId": template_id,
                "originalTemplate": base_template,
                "optimizedTemplate": r.get("optimizedPromptTemplate"),
                "modelId": r.get("modelId"),
                "metricLabel": d.get("customEvaluationMetricLabel"),
                "status": r.get("status"),
                "failureReason": r.get("failureReason"),
            }
            for base_metric_name in ["Score", "InputTokens", "OutputTokens", "TtftInSec"]:
                val_original = original_metrics.get("average" + base_metric_name)
                row["original" + base_metric_name] = val_original
                val_optimized = optimized_metrics.get("average" + base_metric_name)
                row["optimized" + base_metric_name] = val_optimized
                row[base_metric_name[0].lower() + base_metric_name[1:] + "Delta"] = (
                    None if val_original is None or val_optimized is None else val_optimized - val_original
                )
            rows.append(row)
    return rows


def render_prompt_diff(parsed_row: dict, heading_md: str | None = None) -> Markdown:
    """Collapsible original vs optimized template sections."""
    orig = parsed_row.get("originalTemplate") or "(not present)"
    opt = parsed_row.get("optimizedTemplate") or "(not present)"
    return Markdown(
        f"{heading_md or ''}\n"
        f"<details><summary><b>Original template</b> ({len(orig)} chars)</summary>\n\n"
        f"```\n{orig}\n```\n</details>\n\n"
        f"<details><summary><b>Optimized template</b> ({len(opt)} chars)</summary>\n\n"
        f"```\n{opt}\n```\n</details>"
    )

def pre_run():
    """Pre-create APO jobs with analogous configuration to the notebook"""
    import sagemaker

    sm_sess = sagemaker.Session()
    bucket_name = sm_sess.default_bucket()

    # Subfolders for dataset inputs and result outputs:
    s3_datasets_prefix = f"{DEFAULT_S3_PREFIX}/pre-run/datasets"
    s3_results_prefix = f"{DEFAULT_S3_PREFIX}/pre-run/results"

    subprocess.run(
        [
            "aws",
            "s3",
            "sync",
            f"{DATASETS_ABSPATH}/",
            f"s3://{bucket_name}/{s3_datasets_prefix}"
        ]
    )

    dataset_id = "mathvista"
    dataset = resolve_sample_data_template(
        dataset_id=dataset_id,
        bucket_name=bucket_name,
        s3_datasets_prefix=s3_datasets_prefix,
    )
    dataset["steeringCriteria"] = [
        "State the final answer first on its own line, in the format `Answer: <X>` (a number, an option letter, or short text).",
        "After the answer line, provide step-by-step mathematical derivations grounded in the image, with each derivation step rendered in LaTeX (`$...$` for inline math, `$$...$$` for display equations).",
        "Reasoning must reference specific visual elements of the image (axes, labels, shapes, colors, table cells, etc.) where relevant.",
        "Do not output any prose outside of the answer line and the LaTeX-formatted derivation steps.",
    ]

    dataset_s3_uri = f"s3://{bucket_name}/{s3_datasets_prefix}/{dataset_id}/sample_steering.resolved.jsonl"
    
    with UPath(dataset_s3_uri).open("w") as f:
        # Note this is actually a JSON-Lines file - so must be rendered all on one line:
        f.write(json.dumps(dataset))
    print(f"Uploaded to {dataset_s3_uri}\n")
    
    job_name = f"demo-{dataset_id}-steering-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    output_s3uri = f"s3://{bucket_name}/{s3_results_prefix}/{job_name}/"  # (Must have trailing slash)
    print(f"Starting job name {job_name} ...")
    create_resp = bedrock.create_advanced_prompt_optimization_job(
        jobName=job_name,
        modelConfigurations=[
            {"modelId": "moonshotai.kimi-k2.5"},
            {"modelId": "qwen.qwen3-vl-235b-a22b"},
            {"modelId": "us.anthropic.claude-sonnet-4-6"},
        ],
        inputConfig={"s3Uri": dataset_s3_uri},
        outputConfig={"s3Uri": output_s3uri},
    )
    job_arn = create_resp["jobArn"]
    print(f"Job ARN: {job_arn}")
