#!/bin/bash
# This script is used in temporary AWS environments to pre-run jobs to accelerate the workshops.
# We *don't* recommend running it in your own AWS Account!

set -euxo pipefail

if command -v uv &> /dev/null; then
  uv venv --allow-existing
  uv sync
  source .venv/bin/activate
elif command -v pip &> /dev/null; then
  pip install -e .
else
  echo "Error: neither uv nor pip found" >&2
  exit 1
fi

rc=0

# Lab 2: Pre-create the Bedrock evaluation jobs (LLM-as-judge + automatic metrics)
# so attendees can jump straight to analyzing completed results.
(cd lab2 && python -c "import utils; utils.pre_run()") || { echo "Lab 2 pre-run failed (exit $?)"; rc=1; }

# Lab 3: Pre-create the Advanced Prompt Optimization (APO) jobs.
(cd lab3 && python -c "import utils; utils.pre_run()") || { echo "Lab 3 pre-run failed (exit $?)"; rc=1; }

exit $rc
