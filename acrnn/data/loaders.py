from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

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


def _apply_channel_standardization(
    train_X: np.ndarray,
    arrays: list[np.ndarray | None],
) -> list[np.ndarray | None]:
    mean = train_X.mean(axis=(0, 2), keepdims=True)
    std = train_X.std(axis=(0, 2), keepdims=True)
    std = np.where(std < 1e-5, 1.0, std)

    standardised: list[np.ndarray | None] = []
    for array in arrays:
        if array is None:
            standardised.append(None)
            continue
        standardised.append(((array - mean) / std).astype(np.float32))
    return standardised


def build_dataloaders(
    X: np.ndarray,
    y: np.ndarray,
    split: DataSplit,
    batch_size: int = 32,
    num_workers: int = 0,
    seed: int = 42,
    normalization: str = "none",
    train_sampling: str = "balanced",
    dataset_factory: Callable[[np.ndarray, np.ndarray], Dataset] | None = None,
) -> LoaderBundle:
    del seed

    if dataset_factory is None:
        dataset_factory = lambda features, labels: ArrayDataset(features, labels)

    train_X = X[split.train_idx]
    train_y = y[split.train_idx]
    test_X = X[split.test_idx] if split.test_idx is not None else None
    test_y = y[split.test_idx] if split.test_idx is not None else None

    if normalization == "channel":
        train_X, test_X = _apply_channel_standardization(
            train_X,
            [train_X, test_X],
        )
        assert train_X is not None
    elif normalization != "none":
        raise ValueError(
            f"normalization must be 'none' or 'channel', got {normalization!r}"
        )

    sampler = None
    shuffle = True
    if train_sampling == "balanced":
        class_counts = np.bincount(train_y.astype(np.int64), minlength=2)
        valid_classes = class_counts > 0
        if valid_classes.sum() >= 2:
            class_weights = np.zeros_like(class_counts, dtype=np.float64)
            class_weights[valid_classes] = 1.0 / class_counts[valid_classes]
            sample_weights = class_weights[train_y.astype(np.int64)]
            sampler = WeightedRandomSampler(
                weights=torch.as_tensor(sample_weights, dtype=torch.double),
                num_samples=len(train_y),
                replacement=True,
            )
            shuffle = False
    elif train_sampling != "shuffle":
        raise ValueError(
            f"train_sampling must be 'shuffle' or 'balanced', got {train_sampling!r}"
        )

    train = DataLoader(
        dataset_factory(train_X, train_y),
        batch_size=batch_size,
        shuffle=shuffle,
        sampler=sampler,
        num_workers=num_workers,
    )

    test: DataLoader | None = None
    if test_X is not None and test_y is not None:
        test = DataLoader(
            dataset_factory(test_X, test_y),
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
        )

    return LoaderBundle(train=train, test=test)
