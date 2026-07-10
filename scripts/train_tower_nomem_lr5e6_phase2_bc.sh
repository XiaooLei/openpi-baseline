#!/bin/bash
set -euo pipefail

export CONFIG=${CONFIG:-pi05_tower-of-hanoi-game_nomem_phase2_bc_lr5e6}
export EXP_NAME=${EXP_NAME:-tower_of_hanoi_game_nomem_phase2_bc_lr5e6_20k}
export TASK_SLUG=tower-of-hanoi-game
export PHASE2_SLUG=tower_of_hanoi_game
export CHECKPOINT_TAR=${CHECKPOINT_TAR:-/inspire/qb-ilm/project/gjjproject/public/xhc/checkpoints/2nd-submit/hanoi_200k_assets_params.tar.gz}
export CHECKPOINT_EXTRACT_DIR=${CHECKPOINT_EXTRACT_DIR:-/inspire/hdd/global_user/gongjingjing-25039/xl/data/baseline_checkpoints/2nd-submit/hanoi_200k}
export NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-50000}

exec "$(dirname "$0")/train_rss_task_memory_phase2_bc.sh"
