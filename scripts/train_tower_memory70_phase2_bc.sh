#!/bin/bash
set -euo pipefail

export CONFIG=${CONFIG:-pi05_tower-of-hanoi-game_memory70_phase2_bc}
export EXP_NAME=${EXP_NAME:-tower_of_hanoi_game_memory70_phase2_bc}
export TASK_SLUG=tower-of-hanoi-game
export PHASE2_SLUG=tower_of_hanoi_game
export CHECKPOINT_TAR=${CHECKPOINT_TAR:-/inspire/qb-ilm/project/gjjproject/public/xhc/checkpoints/2nd-submit/hanoi_200k_assets_params.tar.gz}
export CHECKPOINT_EXTRACT_DIR=${CHECKPOINT_EXTRACT_DIR:-${RSS_SECOND_SUBMIT_CHECKPOINT_ROOT:-${RSS_BASELINE_CHECKPOINT_ROOT:-/inspire/qb-ilm/project/gjjproject/public/xl/data/baseline_checkpoints}/2nd-submit}/hanoi_200k}

exec "$(dirname "$0")/train_rss_task_memory_phase2_bc.sh"
