from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
from torch.utils.data import DataLoader, Dataset

from ..config import DEAP_CACHE_DIR, DEAP_TARGETS
from .loaders import LoaderBundle, build_dataloaders
from .split import DataSplit, build_kfold_splits

# ── Constants ─────────────────────────────────────────────────────────────────


#: Valid emotion-dimension names that can be used as training targets.
VALID_TARGETS: set[str] = set(DEAP_TARGETS)

#: All 32 subject IDs (1-indexed).
VALID_SUBJECTS: list[int] = list(range(1, 33))

#: Fold indices for leave-one-subject-out evaluation (one fold per subject).
VALID_FOLDS_INDEPENDENT: list[int] = list(range(32))

#: Fold indices for within-subject 10-fold cross-validation.
VALID_FOLDS_DEPENDENT: list[int] = list(range(10))

#: Sampling frequency of the preprocessed DEAP recordings (Hz).
DEAP_SFREQ: float = 128.0

#: Number of EEG channels retained after preprocessing.
DEAP_N_CHANNELS: int = 32

#: Mapping from target name to its column index in the ``y_raw`` array.
_LABEL_COL: dict[str, int] = {
    "valence": 0,
    "arousal": 1,
    "dominance": 2,
    "liking": 3,
}

# ── Internal helpers ──────────────────────────────────────────────────────────


def _load_subject_cache(
    subject_id: int,
    cache_dir: Path,
) -> tuple[np.ndarray, np.ndarray]:

    path = cache_dir / f"s{subject_id:02d}.npz"
    if not path.exists():
        raise FileNotFoundError(
            f"Cached file not found: '{path}'\n"
            "Run the preprocessor first:\n"
            "    python -m acrnn.data.deap_preprocesser"
        )
    npz = np.load(path)
    X = npz["X"].astype(np.float32)
    y_raw = npz["y_raw"].astype(np.float32)
    return X, y_raw


def _binarise(
    y_raw_col: np.ndarray,
    threshold: float,
) -> np.ndarray:
    return (y_raw_col >= threshold).astype(np.int64)


# ── Public array-loading API ──────────────────────────────────────────────────


def load_deap_arrays(
    target: str,
    subject_ids: list[int] | None = None,
    cache_dir: str | Path = DEAP_CACHE_DIR,
    threshold: float = 5.0,
) -> tuple[np.ndarray, np.ndarray]:
    if target not in VALID_TARGETS:
        raise ValueError(
            f"target must be one of {sorted(VALID_TARGETS)}, got {target!r}"
        )

    cache_dir = Path(cache_dir)
    ids: list[int] = subject_ids if subject_ids is not None else VALID_SUBJECTS

    if not ids:
        raise ValueError("subject_ids must contain at least one subject ID.")

    invalid = [s for s in ids if s not in VALID_SUBJECTS]
    if invalid:
        raise ValueError(f"Subject IDs out of valid range [1, 32]: {invalid}")

    col = _LABEL_COL[target]
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for sid in ids:
        X_sub, y_raw_sub = _load_subject_cache(sid, cache_dir)
        X_parts.append(X_sub)
        y_parts.append(_binarise(y_raw_sub[:, col], threshold))

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    return X, y


# ── High-level DataLoader wrapper ─────────────────────────────────────────────


class DeapDataloader:
    #: DataLoader for the training partition.
    train: DataLoader

    #: DataLoader for the test partition (``None`` when no test set exists).
    test: DataLoader | None

    def __init__(
        self,
        target: str,
        mode: str = "subject_dependent",
        fold: int = 0,
        subject_id: int | None = None,
        n_folds: int = 10,
        threshold: float = 5.0,
        cache_dir: str | Path = DEAP_CACHE_DIR,
        batch_size: int = 32,
        num_workers: int = 0,
        dataset_factory: Callable[[np.ndarray, np.ndarray], Dataset] | None = None,
    ) -> None:
        # ── Validate arguments ────────────────────────────────────────────────
        if target not in VALID_TARGETS:
            raise ValueError(
                f"target must be one of {sorted(VALID_TARGETS)}, got {target!r}"
            )
        if mode not in {"subject_independent", "subject_dependent"}:
            raise ValueError(
                "mode must be 'subject_independent' or 'subject_dependent', "
                f"got {mode!r}"
            )

        cache_dir = Path(cache_dir)

        # ── Build split depending on evaluation mode ──────────────────────────
        if mode == "subject_independent":
            self._init_subject_independent(
                target=target,
                fold=fold,
                threshold=threshold,
                cache_dir=cache_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                dataset_factory=dataset_factory,
            )
        else:  # subject_dependent
            self._init_subject_dependent(
                target=target,
                fold=fold,
                subject_id=subject_id,
                n_folds=n_folds,
                threshold=threshold,
                cache_dir=cache_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                dataset_factory=dataset_factory,
            )

    # ── Private initialisation helpers ────────────────────────────────────────

    def _init_subject_independent(
        self,
        target: str,
        fold: int,
        threshold: float,
        cache_dir: Path,
        batch_size: int,
        num_workers: int,
        dataset_factory: Callable[[np.ndarray, np.ndarray], Dataset] | None,
    ) -> None:
        """Leave-one-subject-out split."""
        n_subjects = len(VALID_SUBJECTS)
        if not 0 <= fold < n_subjects:
            raise ValueError(
                f"fold must be in [0, {n_subjects - 1}] for "
                f"subject_independent mode, got {fold}"
            )

        test_subject_id = VALID_SUBJECTS[fold]
        train_subject_ids = [s for s in VALID_SUBJECTS if s != test_subject_id]

        # Load training and test data separately, then concatenate so that
        # build_dataloaders can index into a single array pair.
        X_train, y_train = load_deap_arrays(
            target, train_subject_ids, cache_dir, threshold
        )
        X_test, y_test = load_deap_arrays(
            target, [test_subject_id], cache_dir, threshold
        )

        n_train = len(y_train)
        n_test = len(y_test)

        X_all = np.concatenate([X_train, X_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)

        split = DataSplit(
            train_idx=np.arange(n_train),
            test_idx=np.arange(n_train, n_train + n_test),
        )

        bundle: LoaderBundle = build_dataloaders(
            X_all,
            y_all,
            split,
            batch_size=batch_size,
            num_workers=num_workers,
            dataset_factory=dataset_factory,
        )
        self.train = bundle.train
        self.test = bundle.test

    def _init_subject_dependent(
        self,
        target: str,
        fold: int,
        subject_id: int | None,
        n_folds: int,
        threshold: float,
        cache_dir: Path,
        batch_size: int,
        num_workers: int,
        dataset_factory: Callable[[np.ndarray, np.ndarray], Dataset] | None,
    ) -> None:
        """Within-subject k-fold split."""
        if subject_id is None:
            raise ValueError(
                "subject_id must be provided when mode='subject_dependent'."
            )
        if subject_id not in VALID_SUBJECTS:
            raise ValueError(f"subject_id must be in [1, 32], got {subject_id}")
        if n_folds < 2:
            raise ValueError(f"n_folds must be at least 2, got {n_folds}")
        if not 0 <= fold < n_folds:
            raise ValueError(
                f"fold must be in [0, {n_folds - 1}] for "
                f"subject_dependent mode with n_folds={n_folds}, got {fold}"
            )

        X_all, y_all = load_deap_arrays(target, [subject_id], cache_dir, threshold)

        # build_kfold_splits returns a list of DataSplit objects; pick the
        # requested fold.
        splits = build_kfold_splits(len(y_all), k=n_folds)
        split = splits[fold]

        bundle: LoaderBundle = build_dataloaders(
            X_all,
            y_all,
            split,
            batch_size=batch_size,
            num_workers=num_workers,
            dataset_factory=dataset_factory,
        )
        self.train = bundle.train
        self.test = bundle.test

    # ── Convenience properties ────────────────────────────────────────────────

    @property
    def n_train(self) -> int:
        """Number of training samples."""
        return len(self.train.dataset)  # type: ignore[arg-type]

    @property
    def n_test(self) -> int:
        """Number of test samples (0 if no test set)."""
        if self.test is None:
            return 0
        return len(self.test.dataset)  # type: ignore[arg-type]

    def __repr__(self) -> str:
        test_info = f"{self.n_test}" if self.test is not None else "None"
        return (
            f"{self.__class__.__name__}("
            f"n_train={self.n_train}, "
            f"n_test={test_info}, "
            f"batch_size={self.train.batch_size})"
        )
