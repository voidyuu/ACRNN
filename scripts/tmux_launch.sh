#!/bin/bash
set -euo pipefail

if [ "$#" -lt 3 ]; then
    echo "Usage: $0 <session-prefix> <window-name> <command> [args...]" >&2
    exit 1
fi

SESSION_PREFIX="$1"
WINDOW_NAME="$2"
shift 2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

sanitize_name() {
    printf '%s' "$1" | tr -cs '[:alnum:]_-' '_'
}

create_window() {
    local target_session="$1"
    local target_window="$2"
    shift 2
    tmux new-window -t "${target_session}:" -n "$target_window" "$(printf '%q ' "$@")"
}

SESSION_PREFIX="$(sanitize_name "$SESSION_PREFIX")"
WINDOW_NAME="$(sanitize_name "$WINDOW_NAME")"

if [ -n "${TMUX:-}" ]; then
    SESSION_NAME="$(tmux display-message -p '#S')"
    create_window "$SESSION_NAME" "$WINDOW_NAME" "$@"
    echo "Started tmux window '$WINDOW_NAME' in existing session '$SESSION_NAME'."
    exit 0
fi

SESSION_NAME="${SESSION_PREFIX}"
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    create_window "$SESSION_NAME" "$WINDOW_NAME" "$@"
    echo "Started tmux window '$WINDOW_NAME' in existing session '$SESSION_NAME'."
    echo "Attach with: tmux attach -t $SESSION_NAME"
    exit 0
fi

tmux new-session -d -s "$SESSION_NAME" -n "$WINDOW_NAME" "$(printf '%q ' "$@")"
echo "Started tmux session '$SESSION_NAME' with window '$WINDOW_NAME'."
echo "Attach with: tmux attach -t $SESSION_NAME"
