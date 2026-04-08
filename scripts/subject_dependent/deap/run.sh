#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SESSION_NAME="${ACRNN_TMUX_SESSION:-acrnn-subject-dependent-deap-$(date +%Y%m%d_%H%M%S)}"

exec "$SCRIPT_DIR/../../tmux_targets.sh" \
    "$SESSION_NAME" \
    deap \
    subject_dependent \
    valence,arousal \
    "$@"
