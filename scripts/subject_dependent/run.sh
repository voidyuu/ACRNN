#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="${ACRNN_TMUX_SESSION:-acrnn-subject-dependent-$(date +%Y%m%d_%H%M%S)}"

ACRNN_TMUX_ATTACH=never "$SCRIPT_DIR/../tmux_targets.sh" \
    "$SESSION_NAME" \
    deap \
    subject_dependent \
    valence,arousal \
    "$@"

ACRNN_TMUX_ATTACH=never "$SCRIPT_DIR/../tmux_targets.sh" \
    "$SESSION_NAME" \
    dreamer \
    subject_dependent \
    valence,arousal,dominance \
    "$@"

if [ -z "${TMUX:-}" ] && [ -t 1 ]; then
    tmux attach-session -t "$SESSION_NAME"
else
    echo "Created tmux windows in session '$SESSION_NAME'. Attach with: tmux attach-session -t $SESSION_NAME"
fi
