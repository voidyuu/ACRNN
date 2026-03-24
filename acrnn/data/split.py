from __future__ import annotations

from dataclasses import dataclass

import numpy as np

@dataclass(frozen=True)
class DataSplit:
    train_idx: np.ndarray
    test_idx: np.ndarray | None


def build_index_split(
    num_examples: int,
    test_idx: np.ndarray | None = None,
) -> DataSplit:
    if test_idx is None:
        return DataSplit(train_idx=np.arange(num_examples), test_idx=None)

    train_idx = np.delete(np.arange(num_examples), test_idx)
    return DataSplit(train_idx=train_idx, test_idx=test_idx)
