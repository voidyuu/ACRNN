from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from huggingface_hub import hf_hub_download


@dataclass(frozen=True)
class ArrayBundle:
    X: np.ndarray
    y: np.ndarray


def download_dataset_file(repo_id: str, filename: str, cache_dir: str | None) -> str:
    return hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        repo_type="dataset",
        cache_dir=cache_dir,
    )


def load_npy_bundle(
    repo_id: str,
    X_filename: str,
    y_filename: str,
    cache_dir: str | None = None,
) -> ArrayBundle:
    X = np.load(download_dataset_file(repo_id, X_filename, cache_dir))
    y = np.load(download_dataset_file(repo_id, y_filename, cache_dir))
    return ArrayBundle(X=X, y=y)


def load_index_array(
    repo_id: str,
    filename: str,
    cache_dir: str | None = None,
) -> np.ndarray:
    fold_file = download_dataset_file(repo_id, filename, cache_dir)
    return np.loadtxt(fold_file, dtype=int)
