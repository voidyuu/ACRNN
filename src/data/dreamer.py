from __future__ import annotations

import torch
from datasets import load_dataset as hf_load_dataset
from torch.utils.data import DataLoader, Dataset

TARGET_TO_REPO: dict[str, str] = {
    "arousal": "monster-monash/DREAMERA",
    "valence": "monster-monash/DREAMERV",
}

VALID_FOLDS = range(5)  # fold_0 … fold_4


class DreamerDataset(Dataset):

    def __init__(self, hf_split) -> None:
        self._data = hf_split

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self._data[idx]
        # X: (14, 256) float32
        x = torch.tensor(sample["X"], dtype=torch.float32)
        # y: original labels are 1 / 2 — shift to 0 / 1 for CrossEntropyLoss
        y = torch.tensor(sample["y"], dtype=torch.long) - 1
        return x, y


class DreamerDataloader:

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
        config = "full" if fold is None else f"fold_{fold}"

        hf_ds = hf_load_dataset(repo, config, trust_remote_code=True, cache_dir=cache_dir)

        self.train = DataLoader(
            DreamerDataset(hf_ds["train"]),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )

        self.test: DataLoader | None = None
        if fold is not None:
            self.test = DataLoader(
                DreamerDataset(hf_ds["test"]),
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )
