#!/bin/bash
set -euo pipefail

export CONFIG=${CONFIG:-pi05_seal-water-bottle-cap_state32mem18_expert-success_lr5e6}
export EXP_NAME=${EXP_NAME:-seal_state32mem18_60_120_water120_expert_success_lr5e6}
export NUM_TRAIN_STEPS=${NUM_TRAIN_STEPS:-60000}

exec "$(dirname "$0")/train_seal_memory_phase2_bc.sh"
