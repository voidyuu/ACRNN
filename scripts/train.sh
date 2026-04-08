#!/bin/bash
set -euo pipefail

export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

VALID_MODES=(subject_dependent subject_independent)

DATASET_SPECS=("dreamer")
MODE_SPECS=("subject_dependent")
TARGET_SPECS=("valence" "arousal")
CACHE_DIR_OVERRIDE=""
USE_FAST_PRESET=0
PRESET_ARGS=()
EXTRA_ARGS=()

python_runner=()
acrnn_runner=()

print_help() {
    cat <<'EOF'
Usage: scripts/train.sh [script-options] [acrnn-options]

Script options:
  --dataset, --datasets VALUE    Dataset list: deap,dreamer or all
  --mode, --modes VALUE          Mode list: subject_dependent,subject_independent or all
  --target, --targets VALUE      Target list: comma-separated names or all
  --fast                         Use the quick debug preset
  --cache-dir PATH               Override cache dir for a single dataset
  --help                         Show this help message

Any other arguments are forwarded to the `acrnn` CLI.

This script is responsible for dataset/mode/target selection and automatic
preprocessing. It also applies a training preset automatically.
By default it uses the stronger validation-driven training setup.
With `--fast` it uses a lighter debug preset for quick smoke tests.

Examples:
  scripts/train.sh --dataset dreamer --mode all --target all
  scripts/train.sh --fast --dataset deap,dreamer --mode subject_independent --target valence,arousal
  scripts/train.sh --dataset deap --mode subject_dependent --target liking --subject-id 1
EOF
}

split_csv() {
    local raw="$1"
    local out_var="$2"
    local item
    local values=()

    if ! [[ "$out_var" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
        echo "Invalid output variable name: $out_var" >&2
        exit 1
    fi

    IFS=',' read -r -a values <<< "$raw"
    for item in "${!values[@]}"; do
        values[$item]="${values[$item]//[[:space:]]/}"
    done

    eval "$out_var=()"
    for item in "${values[@]}"; do
        eval "$out_var+=(\"\$item\")"
    done
}

contains_value() {
    local needle="$1"
    shift
    local value
    for value in "$@"; do
        if [ "$value" = "$needle" ]; then
            return 0
        fi
    done
    return 1
}

read_lines_into_array() {
    local out_var="$1"
    shift
    local line

    if ! [[ "$out_var" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
        echo "Invalid output variable name: $out_var" >&2
        exit 1
    fi

    eval "$out_var=()"
    while IFS= read -r line; do
        eval "$out_var+=(\"\$line\")"
    done < <("$@")
}

resolve_runtime() {
    if command -v uv >/dev/null 2>&1; then
        python_runner=(uv run python)
        acrnn_runner=(uv run acrnn)
        return
    fi

    if [ -x "$SCRIPT_DIR/../.venv/bin/python" ] && [ -x "$SCRIPT_DIR/../.venv/bin/acrnn" ]; then
        python_runner=("$SCRIPT_DIR/../.venv/bin/python")
        acrnn_runner=("$SCRIPT_DIR/../.venv/bin/acrnn")
        return
    fi

    echo "Unable to find a runnable environment. Install uv or create .venv with acrnn installed." >&2
    exit 1
}

default_cache_dir_for_dataset() {
    local dataset="$1"
    "${python_runner[@]}" -c "from acrnn.config import get_default_cache_dir; print(get_default_cache_dir('$dataset'))"
}

read_config_tuple_into_array() {
    local out_var="$1"
    local python_expr="$2"
    local line

    if ! [[ "$out_var" =~ ^[A-Za-z_][A-Za-z0-9_]*$ ]]; then
        echo "Invalid output variable name: $out_var" >&2
        exit 1
    fi

    eval "$out_var=()"
    while IFS= read -r line; do
        eval "$out_var+=(\"\$line\")"
    done < <("${python_runner[@]}" -c "$python_expr")
}

resolve_cache_dir() {
    local dataset="$1"
    if [ -n "$CACHE_DIR_OVERRIDE" ]; then
        echo "$CACHE_DIR_OVERRIDE"
    else
        default_cache_dir_for_dataset "$dataset"
    fi
}

ensure_cache_ready() {
    local dataset="$1"
    local cache_dir="$2"

    if [ -n "$(find "$cache_dir" -mindepth 1 -print -quit 2>/dev/null)" ]; then
        return
    fi

    echo "Cache for dataset '$dataset' is empty, running preprocessor..."
    case "$dataset" in
        deap)
            "${python_runner[@]}" -m acrnn.data.deap_preprocesser --cache-dir "$cache_dir"
            ;;
        dreamer)
            "${python_runner[@]}" -m acrnn.data.dreamer_preprocesser --cache-dir "$cache_dir"
            ;;
    esac
}

expand_datasets() {
    local spec
    local expanded=()
    local valid_datasets=()
    read_config_tuple_into_array valid_datasets "from acrnn.config import VALID_DATASETS; print(*VALID_DATASETS, sep='\\n')"
    for spec in "${DATASET_SPECS[@]}"; do
        if [ "$spec" = "all" ]; then
            expanded+=("${valid_datasets[@]}")
        else
            if ! contains_value "$spec" "${valid_datasets[@]}"; then
                echo "Invalid dataset: $spec" >&2
                exit 1
            fi
            expanded+=("$spec")
        fi
    done
    printf '%s\n' "${expanded[@]}" | awk '!seen[$0]++'
}

expand_modes() {
    local spec
    local expanded=()
    for spec in "${MODE_SPECS[@]}"; do
        if [ "$spec" = "all" ]; then
            expanded+=("${VALID_MODES[@]}")
        else
            if ! contains_value "$spec" "${VALID_MODES[@]}"; then
                echo "Invalid mode: $spec" >&2
                exit 1
            fi
            expanded+=("$spec")
        fi
    done
    printf '%s\n' "${expanded[@]}" | awk '!seen[$0]++'
}

expand_targets_for_dataset() {
    local dataset="$1"
    local available_targets=()
    local spec
    local expanded=()

    read_config_tuple_into_array available_targets "from acrnn.config import get_valid_targets; print(*get_valid_targets('$dataset'), sep='\\n')"

    for spec in "${TARGET_SPECS[@]}"; do
        if [ "$spec" = "all" ]; then
            expanded+=("${available_targets[@]}")
        else
            if ! contains_value "$spec" "${available_targets[@]}"; then
                echo "Invalid target '$spec' for dataset '$dataset'" >&2
                exit 1
            fi
            expanded+=("$spec")
        fi
    done

    printf '%s\n' "${expanded[@]}" | awk '!seen[$0]++'
}

while [ "$#" -gt 0 ]; do
    case "$1" in
        --dataset|--datasets)
            shift
            split_csv "${1:-}" DATASET_SPECS
            ;;
        --mode|--modes)
            shift
            split_csv "${1:-}" MODE_SPECS
            ;;
        --target|--targets)
            shift
            split_csv "${1:-}" TARGET_SPECS
            ;;
        --cache-dir)
            shift
            CACHE_DIR_OVERRIDE="${1:-}"
            EXTRA_ARGS+=("--cache-dir" "$CACHE_DIR_OVERRIDE")
            ;;
        --fast)
            USE_FAST_PRESET=1
            ;;
        --help|-h)
            print_help
            exit 0
            ;;
        *)
            EXTRA_ARGS+=("$1")
            ;;
    esac
    shift
done

if [ "$USE_FAST_PRESET" -eq 1 ]; then
    PRESET_ARGS+=(
        --epochs 25
        --batch-size 16
        --learning-rate 2e-4
        --weight-decay 1e-2
        --optimizer adamw
        --scheduler plateau
        --validation-split 0.1
        --normalization channel
        --train-sampling balanced
        --loss-class-weighting balanced
        --grad-clip 1.0
        --threshold-min-precision 0.65
        --threshold-min-recall 0.65
        --patience 8
        --min-epochs 10
        --log-every 5
        --num-workers 0
    )
else
    PRESET_ARGS+=(
        --epochs 80
        --batch-size 16
        --learning-rate 2e-4
        --weight-decay 1e-2
        --optimizer adamw
        --scheduler plateau
        --validation-split 0.1
        --normalization channel
        --train-sampling balanced
        --loss-class-weighting balanced
        --grad-clip 1.0
        --threshold-min-precision 0.65
        --threshold-min-recall 0.65
        --patience 15
        --min-epochs 20
        --log-every 10
        --num-workers 0
    )
fi

if [ -n "$CACHE_DIR_OVERRIDE" ]; then
    read_lines_into_array _datasets_for_cache_check expand_datasets
    if [ "${#_datasets_for_cache_check[@]}" -ne 1 ]; then
        echo "--cache-dir can only be used when exactly one dataset is selected." >&2
        exit 1
    fi
fi

resolve_runtime

read_lines_into_array DATASETS expand_datasets
read_lines_into_array MODES expand_modes

for dataset in "${DATASETS[@]}"; do
    cache_dir="$(resolve_cache_dir "$dataset")"
    ensure_cache_ready "$dataset" "$cache_dir"

    read_lines_into_array TARGETS expand_targets_for_dataset "$dataset"

    for mode in "${MODES[@]}"; do
        for target in "${TARGETS[@]}"; do
            echo
            echo "============================================================"
            echo "Launching training: dataset=$dataset mode=$mode target=$target"
            echo "============================================================"

            cmd=(
                "${acrnn_runner[@]}"
                --dataset "$dataset"
                --mode "$mode"
                --target "$target"
            )
            cmd+=("${PRESET_ARGS[@]}")
            cmd+=("${EXTRA_ARGS[@]}")
            "${cmd[@]}"
        done
    done
done
