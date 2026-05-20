#!/bin/bash
# This script is used in temporary AWS environments to pre-run jobs to accelerate the workshops.
# We *don't* recommend running it in your own AWS Account!

set -eux pipefail

pip install -e .

cd lab3
python -c "import utils; utils.pre_run()"
