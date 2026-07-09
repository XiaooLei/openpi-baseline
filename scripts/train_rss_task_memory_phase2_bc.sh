#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG=${CONFIG:?CONFIG must be set by a task wrapper}
EXP_NAME=${EXP_NAME:?EXP_NAME must be set by a task wrapper}
TASK_SLUG=${TASK_SLUG:?TASK_SLUG must be set by a task wrapper}
PHASE2_SLUG=${PHASE2_SLUG:?PHASE2_SLUG must be set by a task wrapper}
CHECKPOINT_EXTRACT_DIR=${CHECKPOINT_EXTRACT_DIR:?CHECKPOINT_EXTRACT_DIR must be set by a task wrapper}
CHECKPOINT_TAR=${CHECKPOINT_TAR:-}
RSS_DATA_ROOT=${RSS_DATA_ROOT:-/inspire/qb-ilm/project/gjjproject/public/xl/data/rss_challenge}
RSS_BASELINE_CHECKPOINT_ROOT=${RSS_BASELINE_CHECKPOINT_ROOT:-/inspire/qb-ilm/project/gjjproject/public/xl/data/baseline_checkpoints}
RSS_SECOND_SUBMIT_CHECKPOINT_ROOT=${RSS_SECOND_SUBMIT_CHECKPOINT_ROOT:-$RSS_BASELINE_CHECKPOINT_ROOT/2nd-submit}

FSDP_DEVICES=${FSDP_DEVICES:-auto}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-300000}
SAVE_INTERVAL=${SAVE_INTERVAL:-20000}
KEEP_PERIOD=${KEEP_PERIOD:-50000}
EVAL_INTERVAL=${EVAL_INTERVAL:-500}
NUM_EVAL_BATCHES=${NUM_EVAL_BATCHES:-10}
NUM_HELDIN_EVAL_BATCHES=${NUM_HELDIN_EVAL_BATCHES:-5}
LOG_INTERVAL=${LOG_INTERVAL:-50}
XLA_PYTHON_CLIENT_MEM_FRACTION=${XLA_PYTHON_CLIENT_MEM_FRACTION:-0.90}

export XLA_PYTHON_CLIENT_MEM_FRACTION
export OPENPI_DATA_HOME=${OPENPI_DATA_HOME:-"$PWD/.cache/openpi"}
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export WANDB_MODE=${WANDB_MODE:-offline}
export RSS_DATA_ROOT
export RSS_BASELINE_CHECKPOINT_ROOT
export RSS_SECOND_SUBMIT_CHECKPOINT_ROOT

TOKENIZER_PATH="$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
CHECKPOINT_PATH="$CHECKPOINT_EXTRACT_DIR/params"
NORM_STATS_PATH="$CHECKPOINT_EXTRACT_DIR/assets/$TASK_SLUG/expert-success-hil-suffix-mix-data/norm_stats.json"
EXPERT_DATASET_PATH="$RSS_DATA_ROOT/raw/$TASK_SLUG/expert-success-hil-suffix-mix-data/meta/info.json"
PHASE2_DATASET_PATH="$RSS_DATA_ROOT/recap/phase2/${PHASE2_SLUG}_hil_split/train/meta/info.json"

if [[ ! -e "$TOKENIZER_PATH" ]]; then
  echo "Missing required offline tokenizer: $TOKENIZER_PATH" >&2
  exit 1
fi
if [[ ! -e "$CHECKPOINT_PATH/_METADATA" || ! -e "$NORM_STATS_PATH" ]]; then
  if [[ -z "$CHECKPOINT_TAR" || ! -e "$CHECKPOINT_TAR" ]]; then
    echo "Missing extracted checkpoint and checkpoint archive." >&2
    echo "Expected extracted params: $CHECKPOINT_PATH/_METADATA" >&2
    echo "Expected norm stats: $NORM_STATS_PATH" >&2
    echo "Checkpoint archive: ${CHECKPOINT_TAR:-unset}" >&2
    exit 1
  fi
  echo "Extracting checkpoint archive to $CHECKPOINT_EXTRACT_DIR"
  mkdir -p "$CHECKPOINT_EXTRACT_DIR"
  tar -xzf "$CHECKPOINT_TAR" -C "$CHECKPOINT_EXTRACT_DIR"
fi

for required_path in "$CHECKPOINT_PATH" "$NORM_STATS_PATH" "$EXPERT_DATASET_PATH" "$PHASE2_DATASET_PATH"; do
  if [[ ! -e "$required_path" ]]; then
    echo "Missing required offline file: $required_path" >&2
    exit 1
  fi
done

PYTHON_BIN=${PYTHON_BIN:-}
if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x .venv/bin/python3 ]]; then
    PYTHON_BIN=.venv/bin/python3
  else
    PYTHON_BIN=python3
  fi
fi

JAX_DEVICE_COUNT=$("$PYTHON_BIN" - <<'PY'
import jax
print(jax.device_count())
PY
)
if [[ "$FSDP_DEVICES" == "auto" ]]; then
  FSDP_DEVICES="$JAX_DEVICE_COUNT"
fi
if (( JAX_DEVICE_COUNT % FSDP_DEVICES != 0 )); then
  echo "Invalid FSDP_DEVICES=$FSDP_DEVICES for JAX device count $JAX_DEVICE_COUNT." >&2
  echo "Set FSDP_DEVICES to a divisor of $JAX_DEVICE_COUNT, e.g. FSDP_DEVICES=$JAX_DEVICE_COUNT." >&2
  exit 1
fi
if (( BATCH_SIZE % JAX_DEVICE_COUNT != 0 )); then
  echo "Invalid BATCH_SIZE=$BATCH_SIZE for JAX device count $JAX_DEVICE_COUNT." >&2
  echo "Set BATCH_SIZE to a multiple of $JAX_DEVICE_COUNT." >&2
  exit 1
fi

mkdir -p logs
LOG_FILE="logs/${CONFIG}_${EXP_NAME}_$(date '+%Y%m%d-%H%M%S').log"

echo "Using PYTHON_BIN=$PYTHON_BIN"
echo "Detected JAX devices: $JAX_DEVICE_COUNT"
echo "Using FSDP_DEVICES=$FSDP_DEVICES"
echo "Using BATCH_SIZE=$BATCH_SIZE"
echo "Using CONFIG=$CONFIG"
echo "Using EXP_NAME=$EXP_NAME"

if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "DRY_RUN=1, checks passed; not starting training."
  exit 0
fi

"$PYTHON_BIN" scripts/train.py "$CONFIG" \
  --exp-name "$EXP_NAME" \
  --fsdp-devices "$FSDP_DEVICES" \
  --batch-size "$BATCH_SIZE" \
  --num-train-steps "$NUM_TRAIN_STEPS" \
  --save-interval "$SAVE_INTERVAL" \
  --keep-period "$KEEP_PERIOD" \
  --eval-interval "$EVAL_INTERVAL" \
  --num-eval-batches "$NUM_EVAL_BATCHES" \
  --num-heldin-eval-batches "$NUM_HELDIN_EVAL_BATCHES" \
  --log-interval "$LOG_INTERVAL" \
  --tensorboard-enabled \
  --no-wandb-enabled \
  --no-overwrite \
  --resume \
  2>&1 | tee -a "$LOG_FILE"
