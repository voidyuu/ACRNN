#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_SCRIPT="$SCRIPT_DIR/../train.sh"
SESSION_NAME="${ACRNN_TMUX_SESSION:-acrnn-subject-dependent-$(date +%Y%m%d_%H%M%S)}"
BARK_URL="https://bark.cassiel.cc/QkQuG22vPDvthVLE7MLQcX"
ATTACH_MODE="${ACRNN_TMUX_ATTACH:-auto}"

if ! command -v tmux >/dev/null 2>&1; then
    echo "tmux is required to launch one window per target." >&2
    exit 1
fi

session_exists=0
if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    session_exists=1
fi

created_windows=0

launch_target() {
    local dataset="$1"
    local target="$2"
    shift 2

    local window_name="${dataset}-${target}"
    local cmd=(
        /bin/bash
        -c
        'set -euo pipefail
TRAIN_SCRIPT="$1"
DATASET="$2"
TARGET="$3"
BARK_URL="$4"
shift 4

LOG_FILE="$(mktemp -t acrnn_subject_dependent_run.XXXXXX.log)"
cleanup() {
    rm -f "$LOG_FILE"
}
trap cleanup EXIT

set +e
"$TRAIN_SCRIPT" --dataset "$DATASET" --mode subject_dependent --target "$TARGET" "$@" 2>&1 | tee "$LOG_FILE"
status=${PIPESTATUS[0]}
set -e

result_line="$(grep "^Final result - ${DATASET}/${TARGET}:" "$LOG_FILE" | tail -n 1 || true)"
if [ "$status" -eq 0 ]; then
    title="ACRNN ${DATASET}/${TARGET} 完成"
    if [ -n "$result_line" ]; then
        body="$result_line"
    else
        body="${DATASET}/${TARGET} 运行完成，但未找到最终结果行。"
    fi
else
    title="ACRNN ${DATASET}/${TARGET} 失败"
    body="${DATASET}/${TARGET} 运行失败 (exit=${status})"
    if [ -n "$result_line" ]; then
        body="${body}\n\n已完成结果:\n${result_line}"
    fi
    tail_output="$(tail -n 20 "$LOG_FILE")"
    body="${body}\n\n最后日志:\n${tail_output}"
fi

payload="$(python3 - "$title" "$body" <<'"'"'PY'"'"'
import json
import sys

title = sys.argv[1]
body = sys.argv[2]
print(json.dumps({"title": title, "body": body}, ensure_ascii=False))
PY
)"

curl -X POST "$BARK_URL" \
    -H "Content-Type: application/json; charset=utf-8" \
    -d "$payload" >/dev/null || true

exit "$status"'
        --
        "$TRAIN_SCRIPT"
        "$dataset"
        "$target"
        "$BARK_URL"
        "$@"
    )
    if [ "$session_exists" -eq 0 ]; then
        tmux new-session -d -s "$SESSION_NAME" -n "$window_name" "${cmd[@]}"
        session_exists=1
    else
        tmux new-window -t "$SESSION_NAME:" -n "$window_name" "${cmd[@]}"
    fi

    created_windows=$((created_windows + 1))
    echo "Created tmux window '$window_name' in session '$SESSION_NAME'."
}

launch_target deap valence "$@"
launch_target deap arousal "$@"
launch_target dreamer valence "$@"
launch_target dreamer arousal "$@"
launch_target dreamer dominance "$@"

if [ "$created_windows" -eq 0 ]; then
    echo "No targets were launched." >&2
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
