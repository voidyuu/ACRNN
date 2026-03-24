# ACRNN
对论文“EEG-Based Emotion Recognition via Channel-Wise Attention and Self Attention”里的模型复现

Overall Subject-Dependent Accuracy(arousal): 0.9568 ± 0.0357

Overall Subject-Dependent Accuracy(valence): 0.9633 ± 0.0305

## 项目结构

```text
ACRNN/
├── src/acrnn/
│   ├── cli.py
│   ├── model.py
│   ├── training.py
│   └── __main__.py
├── train_arousal.py
├── train_valence.py
└── main.py
```

## 运行方式

统一入口：

```bash
uv run python -m acrnn arousal
uv run python -m acrnn valence
```

兼容旧脚本：

```bash
uv run python train_arousal.py
uv run python train_valence.py
```

可选参数示例：

```bash
uv run python -m acrnn arousal --device cpu --epochs 10 --batch-size 8
```
