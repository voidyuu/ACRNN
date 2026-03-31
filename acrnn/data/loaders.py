from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from .split import DataSplit, split_grouped_train_indices


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

    @property
    def labels(self) -> np.ndarray:
        return self._y

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(self._X[idx], dtype=torch.float32)
        raw_label = self._y[idx]
        label = raw_label if self._label_transform is None else self._label_transform(raw_label)
        y = torch.tensor(int(label), dtype=torch.long)
        return x, y


@dataclass(frozen=True)
class LoaderBundle:
    train: DataLoader
    val: DataLoader | None
    test: DataLoader | None


def _split_train_indices(
    train_idx: np.ndarray,
    y: np.ndarray,
    validation_split: float,
    seed: int,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if validation_split <= 0.0:
        return train_idx, None
    if not 0.0 < validation_split < 1.0:
        raise ValueError(
            f"validation_split must be in [0, 1), got {validation_split}"
        )

    if groups is not None:
        return split_grouped_train_indices(
            train_idx=train_idx,
            y=y,
            groups=groups,
            validation_split=validation_split,
            seed=seed,
        )

    train_labels = y[train_idx]
    class_values = np.unique(train_labels)
    rng = np.random.default_rng(seed)

    train_parts: list[np.ndarray] = []
    val_parts: list[np.ndarray] = []

    for class_value in class_values:
        class_members = train_idx[train_labels == class_value].copy()
        rng.shuffle(class_members)

        if len(class_members) < 2:
            train_parts.append(class_members)
            continue

        val_size = int(round(len(class_members) * validation_split))
        val_size = max(1, min(len(class_members) - 1, val_size))

        val_parts.append(class_members[:val_size])
        train_parts.append(class_members[val_size:])

    inner_train_idx = np.concatenate(train_parts)
    rng.shuffle(inner_train_idx)

    if not val_parts:
        return inner_train_idx, None

    val_idx = np.concatenate(val_parts)
    rng.shuffle(val_idx)
    return inner_train_idx, val_idx


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
    groups: np.ndarray | None = None,
    batch_size: int = 32,
    num_workers: int = 0,
    validation_split: float = 0.0,
    seed: int = 42,
    normalization: str = "none",
    dataset_factory: Callable[[np.ndarray, np.ndarray], Dataset] | None = None,
) -> LoaderBundle:
    if dataset_factory is None:
        dataset_factory = lambda features, labels: ArrayDataset(features, labels)

    train_idx, val_idx = _split_train_indices(
        split.train_idx,
        y,
        validation_split=validation_split,
        seed=seed,
        groups=groups,
    )

    train_X = X[train_idx]
    train_y = y[train_idx]
    val_X = X[val_idx] if val_idx is not None else None
    val_y = y[val_idx] if val_idx is not None else None
    test_X = X[split.test_idx] if split.test_idx is not None else None
    test_y = y[split.test_idx] if split.test_idx is not None else None

    if normalization == "channel":
        train_X, val_X, test_X = _apply_channel_standardization(
            train_X,
            [train_X, val_X, test_X],
        )
        assert train_X is not None
    elif normalization != "none":
        raise ValueError(
            f"normalization must be 'none' or 'channel', got {normalization!r}"
        )

    train = DataLoader(
        dataset_factory(train_X, train_y),
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
    )

    val: DataLoader | None = None
    if val_X is not None and val_y is not None:
        val = DataLoader(
            dataset_factory(val_X, val_y),
            batch_size=batch_size,
            shuffle=False,
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

    return LoaderBundle(train=train, val=val, test=test)
