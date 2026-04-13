#!/bin/bash
set -euo pipefail

DREAMER_RUN_ARGS=(
    --epochs 200
    --batch-size 16
    --learning-rate 2e-4
    --weight-decay 1e-2
    --optimizer adamw
    --scheduler plateau
    --normalization channel
    --train-sampling shuffle
    --loss-class-weighting balanced
    --grad-clip 1.0
    --patience 20
    --min-epochs 20
    --log-every 10
    --num-workers 0
)

run_dreamer() {
    local script_dir
    local session_name

    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    session_name="${ACRNN_TMUX_SESSION:-acrnn-subject-dependent-dreamer-$(date +%Y%m%d_%H%M%S)}"

    exec "$script_dir/../../tmux_targets.sh" \
        "$session_name" \
        dreamer \
        subject_dependent \
        valence,arousal,dominance \
        "${DREAMER_RUN_ARGS[@]}" \
        "$@"
}

if [ "${BASH_SOURCE[0]}" = "$0" ]; then
    run_dreamer "$@"
fi
