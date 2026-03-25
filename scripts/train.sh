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
EXTRA_ARGS=()

print_help() {
    cat <<'EOF'
Usage: scripts/train.sh [script-options] [acrnn-options]

Script options:
  --dataset, --datasets VALUE    Dataset list: deap,dreamer or all
  --mode, --modes VALUE          Mode list: subject_dependent,subject_independent or all
  --target, --targets VALUE      Target list: comma-separated names or all
  --fast                         Use the larger training preset
  --cache-dir PATH               Override cache dir for a single dataset
  --help                         Show this help message

Any other arguments are forwarded to `uv run acrnn`.

This script is responsible for dataset/mode/target selection and automatic
preprocessing. It also applies a training-size preset automatically.
With `--fast` it uses the larger preset; without `--fast` it uses the
lightweight preset.

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

default_cache_dir_for_dataset() {
    local dataset="$1"
    uv run python -c "from acrnn.config import get_default_cache_dir; print(get_default_cache_dir('$dataset'))"
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
    done < <(uv run python -c "$python_expr")
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
            uv run python -m acrnn.data.deap_preprocesser --cache-dir "$cache_dir"
            ;;
        dreamer)
            uv run python -m acrnn.data.dreamer_preprocesser --cache-dir "$cache_dir"
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
    EXTRA_ARGS+=(
        --epochs 1
        --batch-size 128
        --patience 20
        --log-every 1
    )
else
    EXTRA_ARGS+=(
        --epochs 500
        --batch-size 512
        --patience 20
        --num-workers 4
    )
fi

if [ -n "$CACHE_DIR_OVERRIDE" ]; then
    read_lines_into_array _datasets_for_cache_check expand_datasets
    if [ "${#_datasets_for_cache_check[@]}" -ne 1 ]; then
        echo "--cache-dir can only be used when exactly one dataset is selected." >&2
        exit 1
    fi
fi

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
                uv run acrnn
                --dataset "$dataset"
                --mode "$mode"
                --target "$target"
            )
            cmd+=("${EXTRA_ARGS[@]}")
            "${cmd[@]}"
        done
    done
done
