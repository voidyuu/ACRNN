from __future__ import annotations

from dataclasses import dataclass

import numpy as np

VALID_FOLDS = range(5)  # fold_0 … fold_4


@dataclass(frozen=True)
class DataSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray | None


def validate_fold(fold: int | None) -> None:
    if fold is not None and fold not in VALID_FOLDS:
        raise ValueError(f"fold must be an integer 0–4 or None, got {fold!r}")


def build_fold_split(
    num_examples: int,
    fold: int | None,
    test_idx: np.ndarray | None = None,
) -> DataSplit:
    validate_fold(fold)

    if fold is None:
        return DataSplit(train_idx=np.arange(num_examples), test_idx=None)

    if test_idx is None:
        raise ValueError("test_idx is required when fold is not None")

    train_idx = np.delete(np.arange(num_examples), test_idx)
    return DataSplit(train_idx=train_idx, test_idx=test_idx)
