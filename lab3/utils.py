"""Helpers for the Advanced Prompt Optimization (APO) workshop notebooks."""

from __future__ import annotations

import json
import re
from copy import deepcopy
from typing import Any, Callable

# External Dependencies:
import boto3
from botocore.config import Config
from botocore.exceptions import ClientError
from IPython.display import Markdown
from upath import UPath


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
    
    with open(f"datasets/{dataset_id}/sample_data.template.json") as f:
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
