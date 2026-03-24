from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from huggingface_hub import hf_hub_download

TARGET_TO_REPO: dict[str, str] = {
    "arousal": "monster-monash/DREAMERA",
    "valence": "monster-monash/DREAMERV",
}

TARGET_TO_NAME: dict[str, str] = {
    "arousal": "DREAMERA",
    "valence": "DREAMERV",
}


@dataclass(frozen=True)
class DreamerArrays:
    X: np.ndarray
    y: np.ndarray


def validate_target(target: str) -> None:
    if target not in TARGET_TO_REPO:
        raise ValueError(f"target must be one of {list(TARGET_TO_REPO)}, got {target!r}")


def download_dataset_file(repo_id: str, filename: str, cache_dir: str | None) -> str:
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )


def load_dreamer_arrays(target: str, cache_dir: str | None = None) -> DreamerArrays:
    validate_target(target)

    repo = TARGET_TO_REPO[target]
    name = TARGET_TO_NAME[target]
    X = np.load(download_dataset_file(repo, f"{name}_X.npy", cache_dir))
    y = np.load(download_dataset_file(repo, f"{name}_y.npy", cache_dir))
    return DreamerArrays(X=X, y=y)


def load_fold_indices(
    target: str,
    fold: int,
    cache_dir: str | None = None,
) -> np.ndarray:
    validate_target(target)

    repo = TARGET_TO_REPO[target]
    fold_file = download_dataset_file(repo, f"test_indices_fold_{fold}.txt", cache_dir)
    return np.loadtxt(fold_file, dtype=int)
