"""
deploy_agent.py — Deploy the HR Assistant agent to Amazon Bedrock AgentCore Runtime.

Usage:
    python deploy_agent.py --name HRAssistantV1 [--region us-east-1] [--version v1]

Options:
    --name     Runtime name (alphanumeric, used as resource name prefix). Required.
    --region   AWS region (default: us-east-1).
    --version  Agent version for multi-version deployments: v1 (default) or v2.
               v2 adds "escalate_to_hr_manager" tool and an improved baked-in system prompt,
               simulating a code-level change for target-based routing demos.

Output:
    Writes agent_state_{name}.json in the current directory with:
      runtime_id, runtime_arn, log_group, service_name, role_arn, region
    This state file is loaded by optimization_tutorial.ipynb.
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
import zipfile
from pathlib import Path
from typing import Literal, Optional

import boto3


# ---------------------------------------------------------------------------
# Agent code
# ---------------------------------------------------------------------------

# v1: standard HR assistant with config bundle hook (reads hr_assistant_agent.py)
# v2: same agent but with an additional escalation tool and improved baked-in system prompt,
#     simulating a code-level improvement for the target-based routing demo.

SCRIPT_DIR = Path(__file__).parent
V1_CODE_PATH = SCRIPT_DIR / "hr_assistant_agent.py"

# v2 adds an escalation tool and a more detailed system prompt baked into the code.
# This represents a new code deployment (not just a prompt config change).
V2_EXTRA_CODE = '''

# ---------------------------------------------------------------------------
# v2 enhancement: escalate to HR manager (new tool added in this code version)
# ---------------------------------------------------------------------------

@tool
def escalate_to_hr_manager(employee_id: str, issue: str) -> dict:
    """
    Escalate a complex HR issue to a human HR manager for review.

    Args:
        employee_id: Employee identifier involved in the escalation.
        issue:       Brief description of the issue requiring human review.
                     Use this tool when: policy conflicts arise, unusual circumstances
                     need manager judgement, or an employee requests human review.

    Returns:
        Dict with ticket_id, assigned_manager, and expected_response_time.
    """
    import uuid as _uuid
    ticket_id = f"ESC-{_uuid.uuid4().hex[:8].upper()}"
    return {
        "ticket_id": ticket_id,
        "employee_id": employee_id,
        "issue": issue,
        "assigned_manager": "HR Manager (on-call)",
        "expected_response_time": "Within 1 business day",
        "status": "OPEN",
        "message": (
            f"Escalation ticket {ticket_id} created. "
            "An HR manager will review and contact the employee within 1 business day."
        ),
    }
'''

V2_SYSTEM_PROMPT = """You are a knowledgeable and empathetic HR Assistant for Acme Corp (v2).

You assist employees with HR matters through a structured, step-by-step approach:

1. UNDERSTAND the employee's request fully before acting
2. RETRIEVE accurate data using the appropriate tool — never guess or fabricate
3. PRESENT information clearly with specific numbers and dates
4. OFFER next steps proactively (e.g., after showing PTO balance, offer to submit a request)
5. ESCALATE to an HR manager when issues are complex, involve policy exceptions,
   or when the employee explicitly requests human review

Available tools and when to use them:
- get_pto_balance: Always call before submitting a PTO request; shows exact remaining days
- submit_pto_request: Requires employee_id, start_date (YYYY-MM-DD), end_date (YYYY-MM-DD)
- lookup_hr_policy: Use for pto, remote_work, parental_leave, or code_of_conduct questions
- get_benefits_summary: Use for health, dental, vision, 401k, or life_insurance questions
- get_pay_stub: Requires employee_id and period (YYYY-MM format)
- escalate_to_hr_manager: Use for complex issues requiring human judgement

Response guidelines:
- Be concise and factual; include specific numbers from tool results
- Confirm actions taken (e.g., "I've submitted PTO request PTO-2026-001")
- Anticipate follow-up needs (e.g., "Would you like me to check the remote work policy?")"""


def build_v1_code() -> str:
    return V1_CODE_PATH.read_text()


def build_v2_code() -> str:
    base = V1_CODE_PATH.read_text()
    # Inject extra tool before the tools list
    extra = V2_EXTRA_CODE
    # Update the DEFAULT_SYSTEM_PROMPT to v2 version
    base = base.replace(
        'DEFAULT_SYSTEM_PROMPT = """You are a helpful HR Assistant for Acme Corp.',
        f'DEFAULT_SYSTEM_PROMPT = """{V2_SYSTEM_PROMPT[V2_SYSTEM_PROMPT.index("You") :]}'.rstrip()
        + "\n\n# (below replaced by v2)\n_PLACEHOLDER_",
    )
    # Simpler approach: replace the DEFAULT_SYSTEM_PROMPT variable entirely
    import re

    # Replace the multiline DEFAULT_SYSTEM_PROMPT
    new_prompt = f'DEFAULT_SYSTEM_PROMPT = """{V2_SYSTEM_PROMPT}"""\n'
    base = re.sub(
        r'DEFAULT_SYSTEM_PROMPT = """.*?"""\n',
        new_prompt,
        base,
        flags=re.DOTALL,
    )
    # Add extra tool after the last @tool definition and before the _MODEL definition
    base = base.replace(
        "_MODEL = BedrockModel",
        extra + "\n_MODEL = BedrockModel",
    )
    # Add escalate_to_hr_manager to tools list
    base = base.replace(
        "    get_pay_stub,\n]",
        "    get_pay_stub,\n    escalate_to_hr_manager,\n]",
    )
    return base


# ---------------------------------------------------------------------------
# Utility Functions
# ---------------------------------------------------------------------------

def setup_execution_role(
    role_name: str,
    boto_session,
    account_id: Optional[str] = None,
) -> str:
    """Set up an AgentCore Execution Role with the required permissions to run the agent

    Note that if the specified role already exists this function just returns the ARN as-is, and
    does *not* alter its permissions.

    Args:
        role_name: Name of the IAM role to create
        boto_session: A boto3.Session to connect to AWS services in the target Region.

    Returns:
        arn: ARN of the IAM role
    """
    iam = boto_session.client("iam")
    account_id = account_id or boto_session.client("sts").get_caller_identity()["Account"]

    trust_policy = json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "bedrock-agentcore.amazonaws.com"},
                    "Action": "sts:AssumeRole",
                    "Condition": {
                        "StringEquals": {
                            "aws:SourceAccount": account_id,
                        },
                        "ArnLike": {
                            "aws:SourceArn": f"arn:aws:bedrock-agentcore:*:{account_id}:*",
                        },
                    },
                }
            ],
        }
    )

    permissions_policy = json.dumps(
        {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "bedrock-agentcore:*",
                        "bedrock:InvokeModel",
                        "bedrock:InvokeModelWithResponseStream",
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:PutLogEvents",
                        "logs:DescribeLogGroups",
                        "logs:DescribeIndexPolicies",
                        "logs:PutIndexPolicy",
                        "logs:FilterLogEvents",
                        "logs:GetLogEvents",
                        "logs:StartQuery",
                        "logs:GetQueryResults",
                        "logs:StopQuery",
                        "cloudwatch:*",
                        "xray:PutTraceSegments",
                        "xray:PutTelemetryRecords",
                        "sts:AssumeRole",
                        "s3:GetObject",
                        "s3:ListBucket",
                    ],
                    "Resource": "*",
                }
            ],
        }
    )

    try:
        resp = iam.create_role(RoleName=role_name, AssumeRolePolicyDocument=trust_policy)
        role_arn = resp["Role"]["Arn"]
        print(f"Created IAM role: {role_arn}")
    except iam.exceptions.EntityAlreadyExistsException:
        role_arn = iam.get_role(RoleName=role_name)["Role"]["Arn"]
        print(f"IAM role exists: {role_arn}")

    iam.put_role_policy(
        RoleName=role_name,
        PolicyName="AgentCoreRuntimePolicy",
        PolicyDocument=permissions_policy,
    )
    print("IAM policy attached. Waiting 30s for propagation...")
    time.sleep(30)
    return role_arn


def build_deployment_package(
    agent_code: str,
    boto_session,
    build_dir: Path,
    s3_bucket: str,
    s3_key: str,
) -> str:
    """Build a deployment code zip for the agent and upload it to Amazon S3

    Args:
        agent_code: The inline Python code defining the agent (which will become the contents of
            a single .py file)
        boto_session: A boto3.Session in the target AWS Region where you want to deploy
        build_dir: A local temporary folder where deployment assets will be staged
        s3_bucket: S3 Bucket to create or use to host the deployment bundle
        s3_key: Object path+name in the target bucket to upload the bundle to.

    Returns:
        s3_uri: s3:// URI where the code zip bundle has been uploaded
    """

    region = boto_session.region_name
    s3 = boto_session.client("s3")

    if build_dir.exists():
        shutil.rmtree(build_dir)
    pkg_dir = build_dir / "pkg"
    pkg_dir.mkdir(parents=True)

    print(f"Installing dependencies for ARM64 into {pkg_dir}...")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "strands-agents[otel]",
            "bedrock-agentcore",
            "aws-opentelemetry-distro",
            "-t",
            str(pkg_dir),
            "--platform",
            "manylinux2014_aarch64",
            "--only-binary=:all:",
            "--python-version",
            "3.13",
            "--quiet",
        ],
        check=True,
    )

    # Write the agent code to main.py
    (pkg_dir / "main.py").write_text(agent_code)
    print(f"Agent code written to {pkg_dir}/main.py")

    # Zip the package
    zip_path = build_dir / "deployment_package.zip"
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for root, _, files in os.walk(pkg_dir):
            for f in files:
                if f.endswith(".pyc") or "__pycache__" in root:
                    continue
                full = Path(root) / f
                zf.write(full, full.relative_to(pkg_dir))

    size_mb = zip_path.stat().st_size / (1024 * 1024)
    print(f"Package built: {zip_path} ({size_mb:.1f} MB)")

    # Upload to S3
    try:
        if region == "us-east-1":
            s3.create_bucket(Bucket=s3_bucket)
        else:
            s3.create_bucket(
                Bucket=s3_bucket,
                CreateBucketConfiguration={"LocationConstraint": region},
            )
        print(f"Created S3 bucket: {s3_bucket}")
    except (s3.exceptions.BucketAlreadyOwnedByYou, s3.exceptions.BucketAlreadyExists):
        print(f"S3 bucket exists: {s3_bucket}")

    s3.upload_file(str(zip_path), s3_bucket, s3_key)
    s3_uri = f"s3://{s3_bucket}/{s3_key}"
    print(f"Uploaded to {s3_uri}")
    return s3_uri


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_cli_args():
    """Parse arguments when using this script via the CLI"""
    parser = argparse.ArgumentParser(description="Deploy HR Assistant to AgentCore Runtime")
    parser.add_argument("--name", required=True, help="Runtime name (alphanumeric)")
    parser.add_argument("--region", default=os.environ.get("AWS_DEFAULT_REGION", "us-east-1"))
    parser.add_argument(
        "--version",
        default="v1",
        choices=["v1", "v2"],
        help="Agent version: v1=baseline, v2=enhanced (extra tool + improved prompt)",
    )
    return parser.parse_args()


def main(name: str, region: Optional[str] = None, version: Literal["v1", "v2"] = "v1") -> Path:
    """Build and deploy the agent

    Args:
        name: Unique name of the AgentCore Runtime to deploy
        region: AWS Region to deploy in
        version: Code version to deploy - 'v1' or 'v2'

    Returns:
        state_file_path: Path to a local JSON file where created agent state is stored
    """

    if version not in ("v1", "v2"):
        raise ValueError(f"'version' parameter must be 'v1' or 'v2'. Got: '{version}'")

    boto_session = boto3.Session(region_name=region) if region else boto3.Session()
    if not region:
        region = boto_session.region_name
    agentcore_ctrl = boto_session.client("bedrock-agentcore-control")

    account_id = boto_session.client("sts").get_caller_identity()["Account"]
    role_name = f"BedrockAgentCore-{name}"
    s3_bucket = f"bedrock-agentcore-code-{account_id}-{region}"
    s3_key = f"{name}/deployment_package.zip"
    build_dir = Path(f"/tmp/{name}_build")  # nosec B108
    state_file = Path(f"agent_state_{name}.json")

    print(f"Deploying {name} (version={version}) to {region} (account={account_id})")

    # Build and stage the code:
    agent_code = build_v2_code() if version == "v2" else build_v1_code()
    build_deployment_package(
        agent_code=agent_code,
        boto_session=boto_session,
        build_dir=build_dir,
        s3_bucket=s3_bucket,
        s3_key=s3_key,
    )
    # Set up IAM:
    role_arn = setup_execution_role(
        role_name=role_name, boto_session=boto_session, account_id=account_id
    )
    # Create the runtime:
    resp = agentcore_ctrl.create_agent_runtime(
        agentRuntimeName=name,
        agentRuntimeArtifact={
            "codeConfiguration": {
                "code": {"s3": {"bucket": s3_bucket, "prefix": s3_key}},
                "runtime": "PYTHON_3_13",
                "entryPoint": ["opentelemetry-instrument", "main.py"],
            }
        },
        networkConfiguration={"networkMode": "PUBLIC"},
        roleArn=role_arn,
    )
    runtime_arn = resp["agentRuntimeArn"]
    runtime_id = resp["agentRuntimeId"]
    print(f"Runtime created: {runtime_id}")

    # Save the state
    state = {
        "account_id": account_id,
        "log_group": f"/aws/bedrock-agentcore/runtimes/{runtime_id}-DEFAULT",
        "runtime_name": name,
        "runtime_id": runtime_id,
        "runtime_arn": runtime_arn,
        "region": region,
        "role_arn": role_arn,
        "role_name": role_name,
        "s3_bucket": s3_bucket,
        "s3_key": s3_key,
        "service_name": f"{name}.DEFAULT",
        "version": version,
    }
    state_file.write_text(json.dumps(state, indent=2))
    print(f"\nState saved to {state_file}")

    print("Polling for runtime to become ACTIVE...")
    for i in range(90):
        detail = agentcore_ctrl.get_agent_runtime(agentRuntimeId=runtime_id)
        status = detail.get("status", "UNKNOWN")
        print(f"  Poll {i + 1}: {status}")
        if status in ("ACTIVE", "READY"):
            runtime_arn = detail.get("agentRuntimeArn")
            break
        if "FAILED" in status:
            raise RuntimeError(f"Runtime failed: {detail.get('failureReason')}")
        time.sleep(10)
    else:
        raise RuntimeError("Runtime did not become ready within 15 minutes")

    print("Runtime ready")
    print(json.dumps(state, indent=2))
    return state_file


if __name__ == "__main__":
    args = parse_cli_args()
    main(**args)
