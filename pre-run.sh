#!/bin/bash
# This script is used in temporary AWS environments to pre-run jobs to accelerate the workshops.
# We *don't* recommend running it in your own AWS Account!

set -eux pipefail

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

cd lab3
python -c "import utils; utils.pre_run()"
