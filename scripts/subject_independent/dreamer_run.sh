#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="${ACRNN_TMUX_SESSION:-acrnn-subject-independent-dreamer-$(date +%Y%m%d_%H%M%S)}"

exec "$SCRIPT_DIR/../../tmux_targets.sh" \
    "$SESSION_NAME" \
    dreamer \
    subject_independent \
    valence,arousal,dominance \
    --epochs 200 \
    --batch-size 16 \
    --learning-rate 2e-4 \
    --weight-decay 1e-2 \
    --optimizer adamw \
    --scheduler plateau \
    --normalization channel \
    --train-sampling shuffle \
    --loss-class-weighting balanced \
    --grad-clip 1.0 \
    --patience 20 \
    --min-epochs 20 \
    --log-every 10 \
    --num-workers 0 \
    "$@"
