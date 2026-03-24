from __future__ import annotations

from .loaders import DreamerDataset, build_dreamer_dataloaders
from .source import load_dreamer_arrays, load_fold_indices, validate_target
from .split import VALID_FOLDS, build_fold_split, validate_fold


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

        arrays = load_dreamer_arrays(target, cache_dir=cache_dir)
        test_idx = None if fold is None else load_fold_indices(target, fold, cache_dir=cache_dir)
        split = build_fold_split(len(arrays.y), fold=fold, test_idx=test_idx)
        loaders = build_dreamer_dataloaders(
            arrays.X,
            arrays.y,
            split,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        self.train = loaders.train
        self.test = loaders.test
