from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .split import DataSplit


class ArrayDataset(Dataset):
    def __init__(
        self,
        X: np.ndarray,
        y: np.ndarray,
        label_transform: Callable[[np.generic | int | float], int] | None = None,
    ) -> None:
        self._X = X
        self._y = y
        self._label_transform = label_transform

    def __len__(self) -> int:
        return len(self._y)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self._X[idx], dtype=torch.float32)
        raw_label = self._y[idx]
        label = raw_label if self._label_transform is None else self._label_transform(raw_label)
        y = torch.tensor(int(label), dtype=torch.long)
        return x, y


@dataclass(frozen=True)
class LoaderBundle:
    train: DataLoader
    test: DataLoader | None


def build_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    split: DataSplit,
    batch_size: int = 32,
    num_workers: int = 0,
    dataset_factory: Callable[[np.ndarray, np.ndarray], Dataset] | None = None,
) -> LoaderBundle:
    if dataset_factory is None:
        dataset_factory = lambda features, labels: ArrayDataset(features, labels)

    train = DataLoader(
        dataset_factory(X[split.train_idx], y[split.train_idx]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    test: DataLoader | None = None
    if split.test_idx is not None:
        test = DataLoader(
            dataset_factory(X[split.test_idx], y[split.test_idx]),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    return LoaderBundle(train=train, test=test)
