#!/bin/bash
set -euo pipefail

export CONFIG=${CONFIG:-pi05_seal-water-bottle-cap_memory70_phase2_bc_lr5e6}
export EXP_NAME=${EXP_NAME:-seal_memory70_15_30_60_120_water120_phase2_bc_lr5e6}

exec "$(dirname "$0")/train_seal_memory_phase2_bc.sh"
