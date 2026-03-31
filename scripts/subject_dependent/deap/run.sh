#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$SCRIPT_DIR/../../tmux_launch.sh"
TRAIN_SCRIPT="$SCRIPT_DIR/../../train.sh"

"$LAUNCHER" \
    acrnn_subject_dependent_deap \
    deap_valence \
    "$TRAIN_SCRIPT" \
    --dataset deap \
    --mode subject_dependent \
    --target valence \
    "$@"

"$LAUNCHER" \
    acrnn_subject_dependent_deap \
    deap_arousal \
    "$TRAIN_SCRIPT" \
    --dataset deap \
    --mode subject_dependent \
    --target arousal \
    "$@"
