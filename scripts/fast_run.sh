export UV_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export HF_ENDPOINT=https://hf-mirror.com

uv run acrnn \
    --target valence \
    --epochs 1 \
    --batch-size 128 \
    --log-every 1 \
    --patience 20 \
    --save-dir ""
