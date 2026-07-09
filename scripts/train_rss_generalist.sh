#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

export XLA_PYTHON_CLIENT_MEM_FRACTION=0.90
export OPENPI_DATA_HOME="$PWD/.cache/openpi"

.venv/bin/python3 scripts/train.py pi05_rss_generalist \
  --exp-name rss_generalist \
  --fsdp-devices 8 \
  --batch-size 32 \
  --num-train-steps 500000 \
  --save-interval 50000 \
  --keep-period 100000 \
  --eval-interval 2000 \
  --num-eval-batches 10 \
  --tensorboard-enabled \
  --no-wandb-enabled \
  --no-overwrite \
  --resume
