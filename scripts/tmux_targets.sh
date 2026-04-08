#!/bin/bash
set -euo pipefail

if [ "$#" -lt 4 ]; then
    echo "Usage: $0 <session-name> <dataset> <mode> <target1,target2,...> [extra train args...]" >&2
    exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/train.sh"

SESSION_NAME="$1"
DATASET="$2"
MODE="$3"
TARGETS_CSV="$4"
shift 4

ATTACH_MODE="${ACRNN_TMUX_ATTACH:-auto}"

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required to launch one window per target." >&2
    exit 1
fi

IFS=',' read -r -a TARGETS <<< "$TARGETS_CSV"
if [ "${#TARGETS[@]}" -eq 0 ]; then
    echo "At least one target is required." >&2
    exit 1
fi

session_exists=0
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    session_exists=1
fi

created_windows=0
for raw_target in "${TARGETS[@]}"; do
    target="${raw_target//[[:space:]]/}"
    if [ -z "$target" ]; then
        continue
    fi

    cmd=(
        "$TRAIN_SCRIPT"
        --dataset "$DATASET"
        --mode "$MODE"
        --target "$target"
        "$@"
    )
    quoted_cmd=$(printf '%q ' "${cmd[@]}")
    window_name="${DATASET}-${target}"

    if [ "$session_exists" -eq 0 ]; then
        tmux new-session -d -s "$SESSION_NAME" -n "$window_name" "$quoted_cmd"
        session_exists=1
    else
        tmux new-window -t "$SESSION_NAME:" -n "$window_name" "$quoted_cmd"
    fi

    created_windows=$((created_windows + 1))
    echo "Created tmux window '$window_name' in session '$SESSION_NAME'."
done

if [ "$created_windows" -eq 0 ]; then
    echo "No non-empty targets were provided." >&2
    exit 1
fi

case "$ATTACH_MODE" in
    always)
        tmux attach-session -t "$SESSION_NAME"
        ;;
    never)
        echo "Created $created_windows tmux window(s). Attach with: tmux attach-session -t $SESSION_NAME"
        ;;
    auto)
        if [ -z "${TMUX:-}" ] && [ -t 1 ]; then
            tmux attach-session -t "$SESSION_NAME"
        else
            echo "Created $created_windows tmux window(s). Attach with: tmux attach-session -t $SESSION_NAME"
        fi
        ;;
    *)
        echo "Unsupported ACRNN_TMUX_ATTACH value: $ATTACH_MODE" >&2
        exit 1
        ;;
esac
