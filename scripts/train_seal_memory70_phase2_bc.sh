#!/bin/bash
set -euo pipefail

export CONFIG=${CONFIG:-pi05_seal-water-bottle-cap_memory70_phase2_bc}
export EXP_NAME=${EXP_NAME:-seal_memory70_phase2_q60_stride_bc}

exec "$(dirname "$0")/train_seal_memory_phase2_bc.sh"
