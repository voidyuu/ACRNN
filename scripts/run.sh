#!/bin/bash
set -e

export HF_ENDPOINT=https://hf-mirror.com

uv run acrnn --target valence --epochs 500 --batch-size 32
uv run acrnn --target arousal --epochs 500 --batch-size 32
