from __future__ import annotations

import numpy as np

from .loaders import ArrayDataset, build_dataloaders
from .source import load_index_array, load_npy_bundle
from .split import build_index_split

TARGET_TO_REPO: dict[str, str] = {
    "arousal": "monster-monash/DREAMERA",
    "valence": "monster-monash/DREAMERV",
}

TARGET_TO_NAME: dict[str, str] = {
    "arousal": "DREAMERA",
    "valence": "DREAMERV",
}

VALID_TARGETS = frozenset(TARGET_TO_REPO)
VALID_FOLDS = range(5)  # fold_0 … fold_4


def validate_target(target: str) -> None:
    if target not in VALID_TARGETS:
        raise ValueError(f"target must be one of {list(TARGET_TO_REPO)}, got {target!r}")


def validate_fold(fold: int | None) -> None:
    if fold is not None and fold not in VALID_FOLDS:
        raise ValueError(f"fold must be an integer 0–4 or None, got {fold!r}")


def load_dreamer_arrays(target: str, cache_dir: str | None = None) -> tuple[np.ndarray, np.ndarray]:
    validate_target(target)

    repo_id = TARGET_TO_REPO[target]
    dataset_name = TARGET_TO_NAME[target]
    bundle = load_npy_bundle(
        repo_id=repo_id,
        X_filename=f"{dataset_name}_X.npy",
        y_filename=f"{dataset_name}_y.npy",
        cache_dir=cache_dir,
    )
    return bundle.X, bundle.y


def load_dreamer_test_indices(
    target: str,
    fold: int,
    cache_dir: str | None = None,
) -> np.ndarray:
    validate_target(target)
    validate_fold(fold)
    return load_index_array(
        repo_id=TARGET_TO_REPO[target],
        filename=f"test_indices_fold_{fold}.txt",
        cache_dir=cache_dir,
    )


def _dreamer_label_transform(label: np.generic | int | float) -> int:
    # DREAMER labels are stored as 1 / 2, while CrossEntropyLoss expects 0 / 1.
    return int(label) - 1


class DreamerDataset(ArrayDataset):
    def __init__(self, X: np.ndarray, y: np.ndarray) -> None:
        super().__init__(X, y, label_transform=_dreamer_label_transform)


class DreamerDataloader:
    """DataLoaders for the DREAMER EEG dataset.

    Args:
        target:      ``"arousal"`` (DREAMERA) or ``"valence"`` (DREAMERV).
        fold:        Cross-validation fold index (0–4). When ``None`` the full
                     dataset is loaded and ``test`` will be ``None``.
        batch_size:  Batch size for both loaders.
        num_workers: Worker processes passed to ``DataLoader``.
        cache_dir:   Optional HuggingFace cache directory override.

    Attributes:
        train: DataLoader for the training split.
        test:  DataLoader for the test split, or ``None`` when *fold* is ``None``.
    """

    def __init__(
        self,
        target: str,
        fold: int | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        cache_dir: str | None = None,
    ) -> None:
        validate_target(target)
        validate_fold(fold)

        X, y = load_dreamer_arrays(target, cache_dir=cache_dir)
        test_idx = None if fold is None else load_dreamer_test_indices(target, fold, cache_dir=cache_dir)
        split = build_index_split(len(y), test_idx=test_idx)
        loaders = build_dataloaders(
            X,
            y,
            split,
            batch_size=batch_size,
            num_workers=num_workers,
            dataset_factory=DreamerDataset,
        )

        self.train = loaders.train
        self.test = loaders.test
