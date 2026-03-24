from __future__ import annotations

import numpy as np
import torch
from huggingface_hub import hf_hub_download
from torch.utils.data import DataLoader, Dataset

TARGET_TO_REPO: dict[str, str] = {
    "arousal": "monster-monash/DREAMERA",
    "valence": "monster-monash/DREAMERV",
}

TARGET_TO_NAME: dict[str, str] = {
    "arousal": "DREAMERA",
    "valence": "DREAMERV",
}

VALID_FOLDS = range(5)  # fold_0 … fold_4


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


def _download(repo_id: str, filename: str, cache_dir: str | None) -> str:
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )


class DreamerDataloader:
    """DataLoaders for the DREAMER EEG dataset downloaded directly from HuggingFace.

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

    Example::

        # Full dataset
        dl = DreamerDataloader("valence")
        for X, y in dl.train: ...

        # Fold-based cross-validation
        dl = DreamerDataloader("arousal", fold=2)
        for X, y in dl.train: ...
        for X, y in dl.test: ...
    """

    def __init__(
        self,
        target: str,
        fold: int | None = None,
        batch_size: int = 32,
        num_workers: int = 0,
        cache_dir: str | None = None,
    ) -> None:
        if target not in TARGET_TO_REPO:
            raise ValueError(
                f"target must be one of {list(TARGET_TO_REPO)}, got {target!r}"
            )
        if fold is not None and fold not in VALID_FOLDS:
            raise ValueError(
                f"fold must be an integer 0–4 or None, got {fold!r}"
            )

        repo = TARGET_TO_REPO[target]
        name = TARGET_TO_NAME[target]

        X = np.load(_download(repo, f"{name}_X.npy", cache_dir))
        y = np.load(_download(repo, f"{name}_y.npy", cache_dir))

        if fold is None:
            train_idx = np.arange(len(y))
            test_idx = None
        else:
            fold_file = _download(repo, f"test_indices_fold_{fold}.txt", cache_dir)
            test_idx = np.loadtxt(fold_file, dtype=int)
            train_idx = np.delete(np.arange(len(y)), test_idx)

        self.train = DataLoader(
            DreamerDataset(X[train_idx], y[train_idx]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        self.test: DataLoader | None = None
        if test_idx is not None:
            self.test = DataLoader(
                DreamerDataset(X[test_idx], y[test_idx]),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
