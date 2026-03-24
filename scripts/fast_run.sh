export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com

uv run acrnn --target valence --epochs 3 --batch-size 8 --log-every 1
