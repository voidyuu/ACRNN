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


def _validate_group_inputs(
    y: np.ndarray,
    groups: np.ndarray,
) -> None:
    if y.ndim != 1:
        raise ValueError(f"y must be 1-D, got shape {y.shape!r}")
    if groups.ndim != 1:
        raise ValueError(f"groups must be 1-D, got shape {groups.shape!r}")
    if len(y) != len(groups):
        raise ValueError(
            "y and groups must have the same length, "
            f"got len(y)={len(y)} and len(groups)={len(groups)}"
        )


def _extract_group_metadata(
    y: np.ndarray,
    groups: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    _validate_group_inputs(y, groups)

    unique_groups, inverse = np.unique(groups, return_inverse=True)
    group_labels = np.empty(len(unique_groups), dtype=y.dtype)
    group_sizes = np.zeros(len(unique_groups), dtype=np.int64)

    for group_idx in range(len(unique_groups)):
        member_mask = inverse == group_idx
        member_labels = np.unique(y[member_mask])
        if len(member_labels) != 1:
            raise ValueError(
                "Each group must contain a single class label for grouped "
                "stratification; found mixed labels in group "
                f"{unique_groups[group_idx]!r}: {member_labels.tolist()}"
            )
        group_labels[group_idx] = member_labels[0]
        group_sizes[group_idx] = int(member_mask.sum())

    return unique_groups, inverse, group_labels, group_sizes


def split_grouped_train_indices(
    train_idx: np.ndarray,
    y: np.ndarray,
    groups: np.ndarray,
    validation_split: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray | None]:
    if validation_split <= 0.0:
        return train_idx, None
    if not 0.0 < validation_split < 1.0:
        raise ValueError(
            f"validation_split must be in [0, 1), got {validation_split}"
        )

    train_idx = np.asarray(train_idx, dtype=np.int64)
    train_y = y[train_idx]
    train_groups = groups[train_idx]
    unique_groups, _, group_labels, _ = _extract_group_metadata(train_y, train_groups)

    rng = np.random.default_rng(seed)
    val_group_ids: list[int | np.integer] = []

    for class_value in np.unique(group_labels):
        class_group_ids = unique_groups[group_labels == class_value].copy()
        rng.shuffle(class_group_ids)

        if len(class_group_ids) < 2:
            continue

        val_size = int(round(len(class_group_ids) * validation_split))
        val_size = max(1, min(len(class_group_ids) - 1, val_size))
        val_group_ids.extend(class_group_ids[:val_size].tolist())

    if not val_group_ids:
        shuffled_train_idx = train_idx.copy()
        rng.shuffle(shuffled_train_idx)
        return shuffled_train_idx, None

    val_mask = np.isin(train_groups, np.asarray(val_group_ids))
    inner_train_idx = train_idx[~val_mask]
    val_idx = train_idx[val_mask]

    inner_train_idx = inner_train_idx.copy()
    val_idx = val_idx.copy()
    rng.shuffle(inner_train_idx)
    rng.shuffle(val_idx)

    return inner_train_idx, val_idx if len(val_idx) > 0 else None


def build_group_stratified_kfold_splits(
    y: np.ndarray,
    groups: np.ndarray,
    k: int = 10,
    shuffle: bool = True,
    seed: int | None = 42,
) -> list[DataSplit]:
    if k < 2:
        raise ValueError(f"k must be at least 2, got {k}")

    y = np.asarray(y)
    groups = np.asarray(groups)
    unique_groups, inverse, group_labels, group_sizes = _extract_group_metadata(y, groups)

    if len(unique_groups) < k:
        raise ValueError(
            f"Number of unique groups ({len(unique_groups)}) must be >= k ({k})"
        )

    rng = np.random.default_rng(seed)
    fold_group_indices: list[list[int]] = [[] for _ in range(k)]
    fold_sizes = np.zeros(k, dtype=np.int64)
    class_values = np.unique(group_labels)
    fold_class_sizes = {
        class_value: np.zeros(k, dtype=np.int64) for class_value in class_values
    }

    class_order = sorted(
        class_values,
        key=lambda class_value: int(np.sum(group_labels == class_value)),
    )

    for class_value in class_order:
        class_group_indices = np.flatnonzero(group_labels == class_value)
        if shuffle:
            class_group_indices = class_group_indices.copy()
            rng.shuffle(class_group_indices)
        class_group_indices = sorted(
            class_group_indices.tolist(),
            key=lambda group_idx: int(group_sizes[group_idx]),
            reverse=True,
        )

        for group_idx in class_group_indices:
            best_fold = min(
                range(k),
                key=lambda fold_idx: (
                    int(fold_class_sizes[class_value][fold_idx]),
                    int(fold_sizes[fold_idx]),
                    len(fold_group_indices[fold_idx]),
                ),
            )
            fold_group_indices[best_fold].append(group_idx)
            fold_sizes[best_fold] += group_sizes[group_idx]
            fold_class_sizes[class_value][best_fold] += group_sizes[group_idx]

    all_indices = np.arange(len(y))
    splits: list[DataSplit] = []
    for test_group_indices in fold_group_indices:
        test_mask = np.isin(inverse, np.asarray(test_group_indices, dtype=np.int64))
        test_idx = all_indices[test_mask]
        train_idx = all_indices[~test_mask]
        splits.append(DataSplit(train_idx=train_idx, test_idx=test_idx))

    return splits
