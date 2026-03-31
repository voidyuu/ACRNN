#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LAUNCHER="$SCRIPT_DIR/../../tmux_launch.sh"
TRAIN_SCRIPT="$SCRIPT_DIR/../../train.sh"

"$LAUNCHER" \
    acrnn_subject_independent_dreamer \
    dreamer_valence \
    "$TRAIN_SCRIPT" \
    --dataset dreamer \
    --mode subject_independent \
    --target valence \
    "$@"

"$LAUNCHER" \
    acrnn_subject_independent_dreamer \
    dreamer_arousal \
    "$TRAIN_SCRIPT" \
    --dataset dreamer \
    --mode subject_independent \
    --target arousal \
    "$@"

"$LAUNCHER" \
    acrnn_subject_independent_dreamer \
    dreamer_dominance \
    "$TRAIN_SCRIPT" \
    --dataset dreamer \
    --mode subject_independent \
    --target dominance \
    "$@"
