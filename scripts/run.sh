#!/bin/bash
set -e

export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com

uv run acrnn \
    --target valence \
    --epochs 500 \
    --batch-size 256 \
    --patience 20 \
    --num-workers 4

uv run acrnn \
    --target arousal \
    --epochs 500 \
    --batch-size 256 \
    --patience 20 \
    --num-workers 4
