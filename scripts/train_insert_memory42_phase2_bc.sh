#!/bin/bash
set -euo pipefail

export CONFIG=${CONFIG:-pi05_insert-mouse-battery_memory42_phase2_bc}
export EXP_NAME=${EXP_NAME:-insert_mouse_battery_memory42_phase2_bc}
export TASK_SLUG=insert-mouse-battery
export PHASE2_SLUG=insert_mouse_battery
export CHECKPOINT_TAR=${CHECKPOINT_TAR:-/inspire/qb-ilm/project/gjjproject/public/xhc/checkpoints/2nd-submit/mouse_80k_assets_params.tar.gz}
export CHECKPOINT_EXTRACT_DIR=${CHECKPOINT_EXTRACT_DIR:-${RSS_SECOND_SUBMIT_CHECKPOINT_ROOT:-${RSS_BASELINE_CHECKPOINT_ROOT:-/inspire/qb-ilm/project/gjjproject/public/xl/data/baseline_checkpoints}/2nd-submit}/mouse_80k}

exec "$(dirname "$0")/train_rss_task_memory_phase2_bc.sh"
