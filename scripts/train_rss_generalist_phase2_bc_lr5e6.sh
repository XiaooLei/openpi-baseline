#!/bin/bash
set -euo pipefail

cd "$(dirname "$0")/.."

CONFIG=${CONFIG:-pi05_rss_generalist_phase2_bc_lr5e6}
EXP_NAME=${EXP_NAME:-rss_generalist_450k_phase2_bc_lr5e6}
LOCAL_DATA_ROOT="$(dirname "$PWD")/data"
DEFAULT_RSS_DATA_ROOT="$LOCAL_DATA_ROOT/rss_challenge"
DEFAULT_RSS_BASELINE_CHECKPOINT_ROOT="$LOCAL_DATA_ROOT/baseline_checkpoints"
if [[ ! -e "$DEFAULT_RSS_DATA_ROOT" ]]; then
  DEFAULT_RSS_DATA_ROOT=/inspire/qb-ilm/project/gjjproject/public/xl/data/rss_challenge
fi
if [[ ! -e "$DEFAULT_RSS_BASELINE_CHECKPOINT_ROOT" ]]; then
  DEFAULT_RSS_BASELINE_CHECKPOINT_ROOT=/inspire/qb-ilm/project/gjjproject/public/xl/data/baseline_checkpoints
fi
RSS_DATA_ROOT=${RSS_DATA_ROOT:-$DEFAULT_RSS_DATA_ROOT}
RSS_BASELINE_CHECKPOINT_ROOT=${RSS_BASELINE_CHECKPOINT_ROOT:-$DEFAULT_RSS_BASELINE_CHECKPOINT_ROOT}
DEFAULT_GENERALIST_CHECKPOINT_DIR="$RSS_BASELINE_CHECKPOINT_ROOT/pi05_rss_generalist_450000/jax/450000"
if [[ ! -e "$DEFAULT_GENERALIST_CHECKPOINT_DIR/params/_METADATA" ]]; then
  DEFAULT_GENERALIST_CHECKPOINT_DIR=/inspire/qb-ilm/project/gjjproject/public/xl/rss-challenge/checkpoints/pi05_rss_generalist_450000/jax/450000
fi
RSS_GENERALIST_CHECKPOINT_DIR=${RSS_GENERALIST_CHECKPOINT_DIR:-$DEFAULT_GENERALIST_CHECKPOINT_DIR}
DEFAULT_GENERALIST_ASSETS_DIR="$RSS_GENERALIST_CHECKPOINT_DIR/assets"
if [[ ! -e "$DEFAULT_GENERALIST_ASSETS_DIR/rss2026_multitask/norm_stats.json" ]]; then
  DEFAULT_GENERALIST_ASSETS_DIR="$RSS_GENERALIST_CHECKPOINT_DIR/assets/v21"
fi
RSS_GENERALIST_ASSETS_DIR=${RSS_GENERALIST_ASSETS_DIR:-$DEFAULT_GENERALIST_ASSETS_DIR}

FSDP_DEVICES=${FSDP_DEVICES:-auto}
BATCH_SIZE=${BATCH_SIZE:-32}
NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-100000}
SAVE_INTERVAL=${SAVE_INTERVAL:-5000}
KEEP_PERIOD=${KEEP_PERIOD:-5000}
EVAL_INTERVAL=${EVAL_INTERVAL:-1000}
NUM_EVAL_BATCHES=${NUM_EVAL_BATCHES:-15}
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
export RSS_GENERALIST_CHECKPOINT_DIR
export RSS_GENERALIST_ASSETS_DIR
export PYTHONPATH="$PWD/src:$PWD/packages/openpi-client/src${PYTHONPATH:+:$PYTHONPATH}"

TOKENIZER_PATH="$OPENPI_DATA_HOME/big_vision/paligemma_tokenizer.model"
CHECKPOINT_PATH="$RSS_GENERALIST_CHECKPOINT_DIR/params/_METADATA"
NORM_STATS_PATH="$RSS_GENERALIST_ASSETS_DIR/rss2026_multitask/norm_stats.json"
REQUIRED_PATHS=(
  "$TOKENIZER_PATH"
  "$CHECKPOINT_PATH"
  "$NORM_STATS_PATH"
  "$RSS_DATA_ROOT/raw/insert-mouse-battery/expert-success-hil-suffix-mix-data/meta/info.json"
  "$RSS_DATA_ROOT/raw/seal-water-bottle-cap/expert-success-hil-suffix-mix-data/meta/info.json"
  "$RSS_DATA_ROOT/raw/tower-of-hanoi-game/expert-success-hil-suffix-mix-data/meta/info.json"
  "$RSS_DATA_ROOT/recap/phase2/insert_mouse_battery_hil_split/train/meta/info.json"
  "$RSS_DATA_ROOT/recap/phase2/seal_water_bottle_cap_hil_split/train/meta/info.json"
  "$RSS_DATA_ROOT/recap/phase2/tower_of_hanoi_game_hil_split/train/meta/info.json"
)

for required_path in "${REQUIRED_PATHS[@]}"; do
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
echo "Using RSS_GENERALIST_CHECKPOINT_DIR=$RSS_GENERALIST_CHECKPOINT_DIR"
echo "Using RSS_GENERALIST_ASSETS_DIR=$RSS_GENERALIST_ASSETS_DIR"
echo "Using NUM_EVAL_BATCHES=$NUM_EVAL_BATCHES"
echo "Using NUM_HELDIN_EVAL_BATCHES=$NUM_HELDIN_EVAL_BATCHES"

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
