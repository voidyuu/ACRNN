from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .split import DataSplit


class DreamerDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        self._X = X
        self._y = y

    def __len__(self) -> int:
        return len(self._y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self._X[idx], dtype=torch.float32)  # (14, 256)
        # y: original labels are 1 / 2 — shift to 0 / 1 for CrossEntropyLoss
        y = torch.tensor(int(self._y[idx]), dtype=torch.long) - 1
        return x, y


@dataclass(frozen=True)
class DreamerLoaders:
    train: DataLoader
    test: DataLoader | None


def build_dreamer_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    split: DataSplit,
    batch_size: int = 32,
    num_workers: int = 0,
) -> DreamerLoaders:
    train = DataLoader(
        DreamerDataset(X[split.train_idx], y[split.train_idx]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test: DataLoader | None = None
    if split.test_idx is not None:
        test = DataLoader(
            DreamerDataset(X[split.test_idx], y[split.test_idx]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    return DreamerLoaders(train=train, test=test)
