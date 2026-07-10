#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-"$PWD/.cache/openpi"}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"

PYTHON_BIN=${PYTHON_BIN:-}
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x .venv/bin/python3 ]]; then
    PYTHON_BIN=.venv/bin/python3
  else
    PYTHON_BIN=python3
  fi
fi

CONFIG=${CONFIG:-pi05_seal-water-bottle-cap_memory42_phase2_bc_lr5e6}
CHECKPOINT_DIR=${CHECKPOINT_DIR:-"$PWD/checkpoints/pi05_seal-water-bottle-cap_memory42_phase2_bc_lr5e6/seal_memory42_60_120_water120_phase2_bc_lr5e6/35000"}
PORT=${PORT:-8000}
DEFAULT_PROMPT=${DEFAULT_PROMPT:-"Place the lid on the cup, align the threads, and twist clockwise to tighten."}

if [[ ! -e "$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model" ]]; then
  echo "Missing offline tokenizer: $OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model" >&2
  exit 1
fi
if [[ ! -e "$CHECKPOINT_DIR/params/_METADATA" ]]; then
  echo "Missing checkpoint params metadata: $CHECKPOINT_DIR/params/_METADATA" >&2
  exit 1
fi
if [[ ! -e "$CHECKPOINT_DIR/assets/seal-water-bottle-cap/expert-success-hil-suffix-mix-data/norm_stats.json" ]]; then
  echo "Missing checkpoint norm stats under: $CHECKPOINT_DIR/assets" >&2
  exit 1
fi

echo "Serving CONFIG=$CONFIG"
echo "Serving CHECKPOINT_DIR=$CHECKPOINT_DIR"
echo "Serving PORT=$PORT"

exec "$PYTHON_BIN" scripts/serve_policy.py \
  --port "$PORT" \
  --default-prompt "$DEFAULT_PROMPT" \
  policy:checkpoint \
  --policy.config "$CONFIG" \
  --policy.dir "$CHECKPOINT_DIR"
