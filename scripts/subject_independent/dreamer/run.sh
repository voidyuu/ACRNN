#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

"$SCRIPT_DIR/../../train.sh" \
    --dataset dreamer \
    --mode subject_independent \
    --target valence,arousal,dominance \
    "$@"
