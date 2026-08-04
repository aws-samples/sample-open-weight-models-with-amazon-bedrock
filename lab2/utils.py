"""Helpers and workshop pre-run automation for the Lab 2 evaluation notebooks.

The `pre_run()` function in this module pre-creates the Amazon Bedrock evaluation
jobs that the Lab 2 notebooks expect to already be *completed* by the time a
workshop attendee reaches the "analyze results" sections. It intentionally
mirrors the job definitions in:

- ``Lab2_Automatic_Evaluation.ipynb`` (LLM-as-a-judge evaluation), and
- ``Lab2Extension_Classical_Metrics.ipynb`` (automatic/"classical" metric jobs
  against both a built-in and a custom dataset).

Keeping these definitions here (rather than in separate Lambda infrastructure)
lets the whole workshop pre-run live in one repository, driven by ``pre-run.sh``.

We *don't* recommend running the pre-run in your own AWS Account - it's only
intended for the temporary, disposable workshop environments.
"""

from __future__ import annotations

from datetime import datetime
import json
import random
import re
import time

# External Dependencies:
import boto3
from botocore.exceptions import ClientError

# ---------------------------------------------------------------------------
# Configuration mirrored from the Lab 2 notebooks
# ---------------------------------------------------------------------------

# S3 parent folder used to keep pre-run artifacts tidy and separate from
# anything an attendee creates while working through the notebooks:
DEFAULT_S3_PREFIX = "bedrock-evaluation-demo/pre-run"

# --- Main notebook: LLM-as-a-judge (see Lab2_Automatic_Evaluation.ipynb) ---
LLM_JUDGE_GENERATOR_MODELS = [
    "nvidia.nemotron-nano-9b-v2",
    "mistral.mistral-7b-instruct-v0:2",
]
LLM_JUDGE_EVALUATOR_MODEL = "mistral.mistral-large-2402-v1:0"
# The full range of built-in LLM-as-judge metrics:
LLM_JUDGE_METRICS = [
    "Builtin.Correctness",
    "Builtin.Completeness",
    "Builtin.Faithfulness",
    "Builtin.Helpfulness",
    "Builtin.Coherence",
    "Builtin.Relevance",
    "Builtin.FollowingInstructions",
    "Builtin.ProfessionalStyleAndTone",
    "Builtin.Harmfulness",
    "Builtin.Stereotyping",
    "Builtin.Refusal",
]

# --- Extension notebook: automatic metrics (see Lab2Extension_Classical_Metrics.ipynb) ---
AUTOMATIC_EVAL_MODELS = ["qwen.qwen3-32b-v1:0", "openai.gpt-oss-20b-1:0"]
AUTOMATIC_TASK_TYPE = "QuestionAndAnswer"
AUTOMATIC_BUILTIN_DATASET = "Builtin.NaturalQuestions"
AUTOMATIC_CUSTOM_DATASET_NAME = "dolly-open-qa-custom"
AUTOMATIC_METRICS = ["Builtin.Accuracy", "Builtin.Robustness", "Builtin.Toxicity"]
# The extension notebook uses "us" cross-region inference profiles for models
# that don't support on-demand throughput directly:
CRIS_REGION_PREFIX = "us"

# Source dataset for the custom automatic-evaluation demo:
DOLLY_15K_URL = (
    "https://huggingface.co/datasets/databricks/databricks-dolly-15k/"
    "resolve/main/databricks-dolly-15k.jsonl"
)


# ---------------------------------------------------------------------------
# Dataset preparation
# ---------------------------------------------------------------------------

def generate_shopping_problems(num_problems: int = 100) -> list[dict]:
    """Generate shopping-related math problems with random values.

    Mirrors the synthetic dataset used in the main Lab 2 notebook.
    """
    problems = []
    items = ["apples", "oranges", "bananas", "books", "pencils", "notebooks"]

    for _ in range(num_problems):
        item = random.choice(items)
        quantity = random.randint(3, 20)
        price_per_item = round(random.uniform(1.5, 15.0), 2)
        discount_percent = random.choice([10, 15, 20, 25, 30])

        total_price = quantity * price_per_item
        discount_amount = total_price * (discount_percent / 100)
        final_price = round(total_price - discount_amount, 2)

        problems.append(
            {
                "prompt": (
                    f"If {item} cost ${price_per_item} each and you buy {quantity} "
                    f"of them with a {discount_percent}% discount, how much will "
                    "you pay in total?"
                ),
                "category": "Shopping Math",
                "referenceResponse": (
                    f"The total price will be ${final_price}. Original price: "
                    f"${total_price} minus {discount_percent}% discount "
                    f"(${discount_amount})"
                ),
            }
        )

    return problems


def fetch_dolly_open_qa(num_records: int = 100) -> list[dict]:
    """Download Dolly-15k and return the first `num_records` `open_qa` examples.

    Records are transformed into the ``prompt``/``referenceResponse``/``category``
    format required by Amazon Bedrock automatic evaluation jobs (mirrors the
    extension notebook).
    """
    import requests

    response = requests.get(DOLLY_15K_URL, timeout=120)
    response.raise_for_status()

    filtered: list[dict] = []
    for line in response.text.strip().split("\n"):
        if not line.strip():
            continue
        try:
            item = json.loads(line)
        except json.JSONDecodeError:
            continue
        if not isinstance(item, dict) or item.get("category") != "open_qa":
            continue
        try:
            filtered.append(
                {
                    "prompt": item["instruction"],
                    "referenceResponse": item["response"],
                    "category": item["category"],
                }
            )
        except KeyError:
            continue
        if len(filtered) >= num_records:
            break

    return filtered


def _upload_jsonl(s3, bucket: str, key: str, records: list[dict]) -> str:
    """Serialize `records` to JSON-Lines and upload to S3, returning the S3 URI."""
    body = "\n".join(json.dumps(r) for r in records)
    s3.put_object(Bucket=bucket, Key=key, Body=body, ContentType="application/json")
    s3_uri = f"s3://{bucket}/{key}"
    print(f"  Uploaded {len(records)} records to {s3_uri}")
    return s3_uri


# ---------------------------------------------------------------------------
# IAM service role for the evaluation jobs
# ---------------------------------------------------------------------------

def create_eval_service_role(iam, region: str, account_id: str, bucket: str) -> str:
    """Create an IAM role Amazon Bedrock can assume to run evaluation jobs.

    Mirrors the role and inline policies created in the main Lab 2 notebook, so
    the pre-run reproduces the same setup an attendee would create by hand. The
    role name follows the ``Amazon-Bedrock-model-eval-*`` convention that the
    workshop's SageMaker execution role is permitted to create and pass.
    """
    assume_role_policy_document = json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "AllowBedrockToAssumeRole",
                    "Effect": "Allow",
                    "Principal": {"Service": "bedrock.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                    "Condition": {
                        "StringEquals": {"aws:SourceAccount": account_id},
                        "ArnEquals": {
                            "aws:SourceArn": (
                                f"arn:aws:bedrock:{region}:{account_id}:evaluation-job/*"
                            )
                        },
                    },
                }
            ],
        }
    )

    role_name = "Amazon-Bedrock-model-eval-{}".format(
        str(datetime.now().timestamp()).split(".")[0]
    )
    role_arn = iam.create_role(
        RoleName=role_name,
        AssumeRolePolicyDocument=assume_role_policy_document,
    )["Role"]["Arn"]

    # Wait for the role to propagate before attaching policies / passing it:
    iam.get_waiter("role_exists").wait(
        RoleName=role_name,
        WaiterConfig={"Delay": 2, "MaxAttempts": 5},
    )

    s3_policy_doc = json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "AllowAccessToCustomDatasetsAndOutput",
                    "Effect": "Allow",
                    "Action": ["s3:GetObject", "s3:ListBucket", "s3:PutObject"],
                    "Resource": [
                        f"arn:aws:s3:::{bucket}",
                        f"arn:aws:s3:::{bucket}/*",
                    ],
                }
            ],
        }
    )
    br_policy_doc = json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Sid": "AllowAccessToBedrockResources",
                    "Effect": "Allow",
                    "Action": [
                        "bedrock:InvokeModel",
                        "bedrock:InvokeModelWithResponseStream",
                        "bedrock:GetInferenceProfile",
                        "bedrock:ListInferenceProfiles",
                        "bedrock:GetEvaluationJob",
                        "bedrock:ListEvaluationJobs",
                        "bedrock:CreateEvaluationJob",
                    ],
                    "Resource": [
                        "arn:aws:bedrock:*::foundation-model/*",
                        f"arn:aws:bedrock:*:{account_id}:inference-profile/*",
                        f"arn:aws:bedrock:*:{account_id}:application-inference-profile/*",
                    ],
                }
            ],
        }
    )

    iam.put_role_policy(
        RoleName=role_name, PolicyName="s3_access", PolicyDocument=s3_policy_doc
    )
    iam.put_role_policy(
        RoleName=role_name, PolicyName="br_access", PolicyDocument=br_policy_doc
    )

    # Wait for the inline policies to become visible:
    for policy_name in ("s3_access", "br_access"):
        for _ in range(30):
            try:
                iam.get_role_policy(RoleName=role_name, PolicyName=policy_name)
                break
            except ClientError as e:
                if e.response["Error"]["Code"] == "NoSuchEntity":
                    time.sleep(1)
                    continue
                raise

    # Allow time for IAM to propagate globally so Bedrock can assume the role.
    print("Waiting 60s for IAM propagation...")
    time.sleep(60)

    print(f"Created Bedrock evaluation service role: {role_arn}")
    return role_arn


# ---------------------------------------------------------------------------
# Model ARN resolution (extension notebook)
# ---------------------------------------------------------------------------

def resolve_model_arn(bedrock, model_id: str) -> str:
    """Resolve a model ID to the ARN the extension notebook would use.

    On-demand models resolve to their foundation-model ARN; models that only
    support inference profiles resolve to the "us" cross-region inference
    profile ARN. Matching this exactly is important because the notebook filters
    completed jobs on ``modelIdentifiers == [model_arn]``.
    """
    fm = bedrock.get_foundation_model(modelIdentifier=model_id)["modelDetails"]
    inference_types = fm["inferenceTypesSupported"]
    if inference_types and inference_types[0] == "INFERENCE_PROFILE":
        profile_id = f"{CRIS_REGION_PREFIX}.{model_id}"
        return bedrock.get_inference_profile(
            inferenceProfileIdentifier=profile_id
        )["inferenceProfileArn"]
    return fm["modelArn"]


# ---------------------------------------------------------------------------
# Evaluation job creation
# ---------------------------------------------------------------------------

def create_llm_judge_evaluation(
    bedrock,
    region: str,
    role_arn: str,
    input_s3_uri: str,
    output_s3_uri: str,
    generator_model_id: str,
    evaluator_model_id: str,
    dataset_name: str = "CustomDataset",
) -> dict:
    """Create an LLM-as-a-judge evaluation job (mirrors the main notebook)."""
    generator_model_arn = f"arn:aws:bedrock:{region}::foundation-model/{generator_model_id}"
    evaluator_model_arn = f"arn:aws:bedrock:{region}::foundation-model/{evaluator_model_id}"

    job_name = "-".join(
        (
            generator_model_id.split(".")[-1].split(":")[0],
            evaluator_model_id.split(".")[0],
            datetime.now().strftime("%Y-%m-%d-%H-%M-%S"),
        )
    )
    job_name = re.sub(r"[^a-z0-9-]+", "-", job_name.lower())[:63]

    response = bedrock.create_evaluation_job(
        jobName=job_name,
        roleArn=role_arn,
        applicationType="ModelEvaluation",
        evaluationConfig={
            "automated": {
                "datasetMetricConfigs": [
                    {
                        "taskType": "General",  # Must be 'General' for LLM-judged
                        "dataset": {
                            "name": dataset_name,
                            "datasetLocation": {"s3Uri": input_s3_uri},
                        },
                        "metricNames": LLM_JUDGE_METRICS,
                    }
                ],
                "evaluatorModelConfig": {
                    "bedrockEvaluatorModels": [
                        {"modelIdentifier": evaluator_model_arn}
                    ]
                },
            }
        },
        inferenceConfig={
            "models": [{"bedrockModel": {"modelIdentifier": generator_model_arn}}]
        },
        outputDataConfig={"s3Uri": output_s3_uri},
    )
    return {"job_name": job_name, "job_arn": response["jobArn"]}


def create_automatic_evaluation(
    bedrock,
    role_arn: str,
    model_arn: str,
    dataset_name: str,
    output_s3_uri: str,
    job_name: str,
    custom_dataset_s3_uri: str | None = None,
) -> dict:
    """Create an automatic (heuristic-metric) evaluation job (extension notebook).

    If `custom_dataset_s3_uri` is provided, a custom dataset is used; otherwise
    `dataset_name` is treated as a Bedrock built-in dataset.
    """
    if custom_dataset_s3_uri:
        dataset = {
            "name": dataset_name,
            "datasetLocation": {"s3Uri": custom_dataset_s3_uri},
        }
    else:
        dataset = {"name": dataset_name}

    response = bedrock.create_evaluation_job(
        jobName=job_name,
        jobDescription="Bedrock Model evaluation job",
        roleArn=role_arn,
        outputDataConfig={"s3Uri": output_s3_uri},
        inferenceConfig={
            "models": [
                {
                    "bedrockModel": {
                        "modelIdentifier": model_arn,
                        "inferenceParams": json.dumps(
                            {
                                "inferenceConfig": {
                                    "maxTokens": 1024,
                                    "temperature": 0.3,
                                    "topP": 0.5,
                                }
                            }
                        ),
                    }
                }
            ]
        },
        evaluationConfig={
            "automated": {
                "datasetMetricConfigs": [
                    {
                        "taskType": AUTOMATIC_TASK_TYPE,
                        "dataset": dataset,
                        "metricNames": AUTOMATIC_METRICS,
                    }
                ]
            }
        },
    )
    return {"job_name": job_name, "job_arn": response["jobArn"]}


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _timestamp_suffix() -> str:
    return str(datetime.now().timestamp()).split(".")[0]


def _get_existing_eval_jobs(bedrock) -> list[dict]:
    """List all evaluation jobs (most recent first).

    Returns the raw jobSummaries from the API, which include 'modelIdentifiers',
    'evaluatorModelIdentifiers', 'status', etc.
    """
    all_jobs = []
    paginator = bedrock.get_paginator("list_evaluation_jobs")
    for page in paginator.paginate(
        sortBy="CreationTime",
        sortOrder="Descending",
    ):
        all_jobs.extend(page["jobSummaries"])
    return all_jobs


def _find_matching_job(
    existing_jobs: list[dict],
    model_identifiers: list[str],
    evaluator_model_identifiers: list[str] | None = None,
    job_name_prefix: str | None = None,
) -> dict | None:
    """Find the most recent job matching the given model configuration.

    Uses the same matching logic as the Lab 2 notebooks: jobs are identified by
    their modelIdentifiers (and evaluatorModelIdentifiers for LLM-judge jobs).
    For automatic eval jobs, job_name_prefix disambiguates built-in vs custom
    dataset jobs that share the same modelIdentifiers.
    Returns the first (most recent) match, or None.
    """
    for job in existing_jobs:
        if job.get("modelIdentifiers") != model_identifiers:
            continue
        if evaluator_model_identifiers is not None:
            if job.get("evaluatorModelIdentifiers") != evaluator_model_identifiers:
                continue
        else:
            if "evaluatorModelIdentifiers" in job:
                continue
        if job_name_prefix and not job.get("jobName", "").startswith(job_name_prefix):
            continue
        return job
    return None


def pre_run(idempotent: bool = True) -> None:
    """Pre-create the Lab 2 evaluation jobs expected by the notebooks.

    This creates completed-ahead-of-time jobs so attendees can jump straight to
    analyzing results. Individual job failures are logged but don't abort the
    whole pre-run (a separate retry pass can re-run any that failed).

    Args:
        idempotent: If True (default), skip creation for any job where a matching
            job (same model identifiers) is already Completed or InProgress.
            Set to False to always create new jobs regardless of existing state.
    """
    import sagemaker

    session = boto3.Session()
    region = session.region_name
    account_id = session.client("sts").get_caller_identity()["Account"]
    bedrock = session.client("bedrock")
    iam = session.client("iam")
    s3 = session.client("s3")

    bucket = sagemaker.Session().default_bucket()
    print(f"Pre-running Lab 2 evaluation jobs in {region}")
    print(f"Using S3 location: s3://{bucket}/{DEFAULT_S3_PREFIX}\n")

    existing_jobs = _get_existing_eval_jobs(bedrock) if idempotent else []

    def _should_skip(
        label: str,
        model_ids: list[str],
        evaluator_ids=None,
        job_name_prefix: str | None = None,
    ) -> bool:
        if not idempotent:
            return False
        match = _find_matching_job(
            existing_jobs, model_ids, evaluator_ids, job_name_prefix
        )
        if match and match["status"] in ("Completed", "InProgress"):
            print(f"  Skipping {label}: existing job is {match['status']}")
            return True
        if match:
            print(f"  Re-creating {label}: existing job has status {match['status']}")
        return False

    role_arn = None

    def _ensure_role() -> str:
        nonlocal role_arn
        if role_arn is None:
            role_arn = create_eval_service_role(iam, region, account_id, bucket)
        return role_arn

    shopping_s3_uri = None
    dolly_s3_uri = None

    def _ensure_datasets():
        nonlocal shopping_s3_uri, dolly_s3_uri
        if shopping_s3_uri is not None:
            return
        print("\nPreparing datasets...")
        shopping_s3_uri = _upload_jsonl(
            s3,
            bucket,
            f"{DEFAULT_S3_PREFIX}/datasets/shopping_math.jsonl",
            generate_shopping_problems(100),
        )
        try:
            dolly_s3_uri = _upload_jsonl(
                s3,
                bucket,
                f"{DEFAULT_S3_PREFIX}/datasets/dolly_open_qa.jsonl",
                fetch_dolly_open_qa(100),
            )
        except Exception as e:  # noqa: BLE001 - best-effort pre-run
            print(f"  WARNING: Failed to prepare Dolly custom dataset: {e}")

    output_s3_uri = f"s3://{bucket}/{DEFAULT_S3_PREFIX}/model-eval-output/"

    # --- LLM-as-a-judge jobs (main notebook) --------------------------------
    print("\nChecking LLM-as-a-judge evaluation jobs...")
    for generator_model_id in LLM_JUDGE_GENERATOR_MODELS:
        generator_arn = f"arn:aws:bedrock:{region}::foundation-model/{generator_model_id}"
        evaluator_arn = f"arn:aws:bedrock:{region}::foundation-model/{LLM_JUDGE_EVALUATOR_MODEL}"
        if _should_skip(
            f"LLM-judge/{generator_model_id}",
            [generator_arn],
            [evaluator_arn],
        ):
            continue
        try:
            _ensure_datasets()
            result = create_llm_judge_evaluation(
                bedrock,
                region=region,
                role_arn=_ensure_role(),
                input_s3_uri=shopping_s3_uri,
                output_s3_uri=output_s3_uri,
                generator_model_id=generator_model_id,
                evaluator_model_id=LLM_JUDGE_EVALUATOR_MODEL,
            )
            print(f"  Created {result['job_name']}: {result['job_arn']}")
        except Exception as e:  # noqa: BLE001 - best-effort pre-run
            print(f"  Failed LLM-judge job for {generator_model_id}: {e}")
        time.sleep(1)

    # --- Automatic metric jobs (extension notebook) -------------------------
    print("\nChecking automatic (classical metric) evaluation jobs...")
    for model_id in AUTOMATIC_EVAL_MODELS:
        try:
            model_arn = resolve_model_arn(bedrock, model_id)
        except Exception as e:  # noqa: BLE001 - best-effort pre-run
            print(f"  Failed to resolve ARN for {model_id}: {e}")
            continue

        sanitized = model_arn.split("/")[-1].split(":")[0].replace(".", "-")

        # Built-in dataset job:
        builtin_prefix = f"model-eval-{sanitized}-"
        if not _should_skip(f"built-in/{model_id}", [model_arn], job_name_prefix=builtin_prefix):
            try:
                _ensure_datasets()
                result = create_automatic_evaluation(
                    bedrock,
                    role_arn=_ensure_role(),
                    model_arn=model_arn,
                    dataset_name=AUTOMATIC_BUILTIN_DATASET,
                    output_s3_uri=output_s3_uri,
                    job_name=f"model-eval-{sanitized}-{_timestamp_suffix()}",
                )
                print(f"  Created built-in {result['job_name']}: {result['job_arn']}")
            except Exception as e:  # noqa: BLE001 - best-effort pre-run
                print(f"  Failed built-in job for {model_id}: {e}")
            time.sleep(1)

        # Custom dataset job (only if the Dolly dataset uploaded successfully):
        custom_prefix = f"eval-custom-{sanitized}-"
        if not _should_skip(f"custom/{model_id}", [model_arn], job_name_prefix=custom_prefix):
            _ensure_datasets()
            if dolly_s3_uri:
                try:
                    result = create_automatic_evaluation(
                        bedrock,
                        role_arn=_ensure_role(),
                        model_arn=model_arn,
                        dataset_name=AUTOMATIC_CUSTOM_DATASET_NAME,
                        output_s3_uri=output_s3_uri,
                        job_name=f"eval-custom-{sanitized}-{_timestamp_suffix()}",
                        custom_dataset_s3_uri=dolly_s3_uri,
                    )
                    print(f"  Created custom {result['job_name']}: {result['job_arn']}")
                except Exception as e:  # noqa: BLE001 - best-effort pre-run
                    print(f"  Failed custom job for {model_id}: {e}")
                time.sleep(1)

    print("\nLab 2 pre-run complete.")
