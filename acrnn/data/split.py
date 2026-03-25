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
    """Build a simple train/test split from explicit test indices.

    Parameters
    ----------
    num_examples:
        Total number of examples in the dataset.
    test_idx:
        Indices to place in the test set.  All remaining indices become the
        training set.  Pass *None* to put everything in training (no test set).

    Returns
    -------
    DataSplit
        A frozen dataclass with ``train_idx`` and ``test_idx`` arrays.
    """
    if test_idx is None:
        return DataSplit(train_idx=np.arange(num_examples), test_idx=None)

    train_idx = np.delete(np.arange(num_examples), test_idx)
    return DataSplit(train_idx=train_idx, test_idx=test_idx)


def build_kfold_splits(
    num_examples: int,
    k: int = 10,
    shuffle: bool = True,
    seed: int | None = 42,
) -> list[DataSplit]:
    """Build *k* :class:`DataSplit` objects for k-fold cross-validation.

    The dataset is partitioned into *k* roughly equal folds.  Each returned
    :class:`DataSplit` holds out one fold as the test set and uses the
    remaining *k-1* folds for training.

    Parameters
    ----------
    num_examples:
        Total number of examples in the dataset.
    k:
        Number of folds (default: 10).
    shuffle:
        Whether to shuffle the indices before partitioning (default: ``True``).
    seed:
        Random seed used for shuffling; ignored when *shuffle* is ``False``
        (default: 42).

    Returns
    -------
    list[DataSplit]
        A list of *k* :class:`DataSplit` objects.  The *i*-th element
        corresponds to using fold *i* as the test set.

    Example
    -------
    >>> splits = build_kfold_splits(1000, k=5)
    >>> for fold, split in enumerate(splits):
    ...     print(fold, len(split.train_idx), len(split.test_idx))
    0 800 200
    1 800 200
    2 800 200
    3 800 200
    4 800 200
    """
    if k < 2:
        raise ValueError(f"k must be at least 2, got {k}")
    if num_examples < k:
        raise ValueError(f"num_examples ({num_examples}) must be >= k ({k})")

    indices = np.arange(num_examples)
    if shuffle:
        rng = np.random.default_rng(seed)
        rng.shuffle(indices)

    # Distribute the remainder across the first (num_examples % k) folds.
    fold_sizes = np.full(k, num_examples // k, dtype=int)
    fold_sizes[: num_examples % k] += 1

    splits: list[DataSplit] = []
    current = 0
    for fold_size in fold_sizes:
        test_idx = indices[current : current + fold_size]
        train_idx = np.concatenate([indices[:current], indices[current + fold_size :]])
        splits.append(DataSplit(train_idx=train_idx, test_idx=test_idx))
        current += fold_size

    return splits
