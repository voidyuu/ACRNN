from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Callable

import numpy as np
import scipy.io
from torch.utils.data import DataLoader, Dataset

from ..config import (
    DREAMER_CACHE_DIR,
    DREAMER_MAT_PATH,
    DREAMER_TARGETS,
    get_default_threshold,
)
from .loaders import LoaderBundle, build_dataloaders
from .split import DataSplit, build_group_stratified_kfold_splits

# ── Constants ─────────────────────────────────────────────────────────────────


#: Valid emotion-dimension names that can be used as training targets.
VALID_TARGETS: set[str] = set(DREAMER_TARGETS)

#: All 23 subject IDs (1-indexed).
VALID_SUBJECTS: list[int] = list(range(1, 24))

#: Fold indices for leave-one-subject-out evaluation (one fold per subject).
#: Exported as ``VALID_FOLDS`` for direct use in ``trainer.cross_validate_model``.
VALID_FOLDS_INDEPENDENT: list[int] = list(range(23))

#: Alias consumed by ``trainer.cross_validate_model``.
VALID_FOLDS: list[int] = VALID_FOLDS_INDEPENDENT

#: Fold indices for within-subject 10-fold cross-validation.
VALID_FOLDS_DEPENDENT: list[int] = list(range(10))

#: Sampling frequency of the DREAMER EEG recordings (Hz).
DREAMER_SFREQ: float = 128.0

#: Number of EEG channels retained after preprocessing.
DREAMER_N_CHANNELS: int = 14

#: Number of emotion-eliciting trials per subject.
DREAMER_N_TRIALS: int = 18

#: EEG channel names (EMOTIV EPOC headset layout).
DREAMER_CHANNEL_NAMES: list[str] = [
    "AF3",
    "F7",
    "F3",
    "FC5",
    "T7",
    "P7",
    "O1",
    "O2",
    "P8",
    "T8",
    "FC6",
    "F4",
    "F8",
    "AF4",
]

#: Mapping from target name to its column index in the ``y_raw`` array.
_LABEL_COL: dict[str, int] = {target: idx for idx, target in enumerate(DREAMER_TARGETS)}


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
            "    python -m acrnn.data.dreamer_preprocesser"
        )
    npz = np.load(path)
    X = npz["X"].astype(np.float32)
    y_raw = npz["y_raw"].astype(np.int8)
    return X, y_raw


def _binarise(
    y_raw_col: np.ndarray,
    threshold: float,
) -> np.ndarray:
    """Convert a 1-D array of integer scores (1 – 5) to binary class labels.

    Parameters
    ----------
    y_raw_col:
        1-D array of raw DREAMER scores on the 1 – 5 scale.
    threshold:
        Scores **≥ threshold** → 1 (high affect);
        scores **< threshold** → 0 (low affect).

    Returns
    -------
    np.ndarray
        1-D ``int64`` array with values in ``{0, 1}``.
    """
    return (y_raw_col >= threshold).astype(np.int64)


def _build_trial_groups(
    trial_window_counts: np.ndarray,
    group_offset: int = 0,
) -> np.ndarray:
    if trial_window_counts.ndim != 1:
        raise ValueError(
            "trial_window_counts must be 1-D, "
            f"got shape {trial_window_counts.shape!r}"
        )

    trial_groups = np.repeat(
        np.arange(len(trial_window_counts), dtype=np.int64),
        trial_window_counts,
    )
    return trial_groups + group_offset


@lru_cache(maxsize=1)
def _load_trial_sample_counts() -> tuple[tuple[int, ...], ...]:
    mat = scipy.io.loadmat(str(DREAMER_MAT_PATH), squeeze_me=True)
    dataset = mat["DREAMER"]
    subjects: list = list(dataset["Data"].item())

    sample_counts: list[tuple[int, ...]] = []
    for subject_struct in subjects:
        eeg = subject_struct["EEG"].item()
        stimuli_arr = eeg["stimuli"].item()
        sample_counts.append(
            tuple(int(stimulus_raw.shape[0]) for stimulus_raw in stimuli_arr)
        )

    return tuple(sample_counts)


def _resolve_subject_trial_window_counts(
    subject_id: int,
    window_samples: int,
) -> np.ndarray:
    sample_counts = _load_trial_sample_counts()[subject_id - 1]
    trial_window_counts = np.asarray(
        [sample_count // window_samples for sample_count in sample_counts],
        dtype=np.int64,
    )
    if len(trial_window_counts) != DREAMER_N_TRIALS:
        raise ValueError(
            f"Expected {DREAMER_N_TRIALS} DREAMER trial counts, got {len(trial_window_counts)}"
        )
    return trial_window_counts


# ── Public array-loading API ──────────────────────────────────────────────────


def load_dreamer_arrays(
    target: str,
    subject_ids: list[int] | None = None,
    cache_dir: str | Path = DREAMER_CACHE_DIR,
    threshold: float | None = None,
    return_groups: bool = False,
) -> tuple[np.ndarray, np.ndarray] | tuple[np.ndarray, np.ndarray, np.ndarray]:

    if target not in VALID_TARGETS:
        raise ValueError(
            f"target must be one of {sorted(VALID_TARGETS)}, got {target!r}"
        )

    threshold = get_default_threshold("dreamer", target) if threshold is None else threshold
    cache_dir = Path(cache_dir)
    ids: list[int] = subject_ids if subject_ids is not None else VALID_SUBJECTS

    if not ids:
        raise ValueError("subject_ids must contain at least one subject ID.")

    invalid = [s for s in ids if s not in VALID_SUBJECTS]
    if invalid:
        raise ValueError(
            f"Subject IDs out of valid range [1, {len(VALID_SUBJECTS)}]: {invalid}"
        )

    col = _LABEL_COL[target]
    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    group_offset = 0

    for sid in ids:
        X_sub, y_raw_sub = _load_subject_cache(sid, cache_dir)
        X_parts.append(X_sub)
        y_parts.append(_binarise(y_raw_sub[:, col], threshold))
        if return_groups:
            trial_window_counts = _resolve_subject_trial_window_counts(
                subject_id=sid,
                window_samples=int(X_sub.shape[-1]),
            )
            if int(trial_window_counts.sum()) != len(X_sub):
                raise ValueError(
                    "Resolved DREAMER trial window counts do not match the cached "
                    f"window total for subject {sid}: "
                    f"expected {int(trial_window_counts.sum())}, got {len(X_sub)}"
                )
            group_parts.append(
                _build_trial_groups(
                    trial_window_counts=trial_window_counts,
                    group_offset=group_offset,
                )
            )
            group_offset += DREAMER_N_TRIALS

    X = np.concatenate(X_parts, axis=0)
    y = np.concatenate(y_parts, axis=0)
    if return_groups:
        groups = np.concatenate(group_parts, axis=0)
        return X, y, groups
    return X, y


# ── High-level DataLoader wrapper ─────────────────────────────────────────────


class DreamerDataloader:
    #: DataLoader for the training partition.
    train: DataLoader

    #: DataLoader for the validation partition (``None`` when disabled).
    val: DataLoader | None

    #: DataLoader for the test partition (``None`` when no test set exists).
    test: DataLoader | None

    def __init__(
        self,
        target: str,
        mode: str = "subject_dependent",
        fold: int = 0,
        subject_id: int | None = None,
        n_folds: int = 10,
        threshold: float | None = None,
        cache_dir: str | Path = DREAMER_CACHE_DIR,
        batch_size: int = 32,
        num_workers: int = 0,
        validation_split: float = 0.0,
        normalization: str = "none",
        seed: int = 42,
        dataset_factory: Callable[[np.ndarray, np.ndarray], Dataset] | None = None,
    ) -> None:
        # ── Validate common arguments ─────────────────────────────────────────
        if target not in VALID_TARGETS:
            raise ValueError(
                f"target must be one of {sorted(VALID_TARGETS)}, got {target!r}"
            )
        if mode not in {"subject_independent", "subject_dependent"}:
            raise ValueError(
                "mode must be 'subject_independent' or 'subject_dependent', "
                f"got {mode!r}"
            )

        threshold = get_default_threshold("dreamer", target) if threshold is None else threshold
        cache_dir = Path(cache_dir)

        # ── Dispatch to the appropriate initialiser ───────────────────────────
        if mode == "subject_independent":
            self._init_subject_independent(
                target=target,
                fold=fold,
                threshold=threshold,
                cache_dir=cache_dir,
                batch_size=batch_size,
                num_workers=num_workers,
                validation_split=validation_split,
                normalization=normalization,
                seed=seed,
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
                validation_split=validation_split,
                normalization=normalization,
                seed=seed,
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
        validation_split: float,
        normalization: str,
        seed: int,
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

        # Load training and test data separately, then concatenate into a
        # single array pair so build_dataloaders can index into them with
        # the DataSplit indices.
        X_train, y_train, train_groups = load_dreamer_arrays(
            target,
            train_subject_ids,
            cache_dir,
            threshold,
            return_groups=True,
        )
        X_test, y_test, test_groups = load_dreamer_arrays(
            target,
            [test_subject_id],
            cache_dir,
            threshold,
            return_groups=True,
        )

        n_train = len(y_train)
        n_test = len(y_test)

        X_all = np.concatenate([X_train, X_test], axis=0)
        y_all = np.concatenate([y_train, y_test], axis=0)
        groups_all = np.concatenate(
            [
                train_groups,
                test_groups + (int(train_groups.max()) + 1 if len(train_groups) > 0 else 0),
            ],
            axis=0,
        )

        split = DataSplit(
            train_idx=np.arange(n_train),
            test_idx=np.arange(n_train, n_train + n_test),
        )

        bundle: LoaderBundle = build_dataloaders(
            X_all,
            y_all,
            split,
            groups=groups_all,
            batch_size=batch_size,
            num_workers=num_workers,
            validation_split=validation_split,
            seed=seed,
            normalization=normalization,
            dataset_factory=dataset_factory,
        )
        self.train = bundle.train
        self.val = bundle.val
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
        validation_split: float,
        normalization: str,
        seed: int,
        dataset_factory: Callable[[np.ndarray, np.ndarray], Dataset] | None,
    ) -> None:
        """Within-subject k-fold split."""
        if subject_id is None:
            raise ValueError(
                "subject_id must be provided when mode='subject_dependent'."
            )
        if subject_id not in VALID_SUBJECTS:
            raise ValueError(
                f"subject_id must be in [1, {len(VALID_SUBJECTS)}], got {subject_id}"
            )
        if n_folds < 2:
            raise ValueError(f"n_folds must be at least 2, got {n_folds}")
        if not 0 <= fold < n_folds:
            raise ValueError(
                f"fold must be in [0, {n_folds - 1}] for "
                f"subject_dependent mode with n_folds={n_folds}, got {fold}"
            )

        X_all, y_all, groups = load_dreamer_arrays(
            target,
            [subject_id],
            cache_dir,
            threshold,
            return_groups=True,
        )

        # Keep entire trials together so windows from one trial cannot leak
        # across train/validation/test partitions.
        splits = build_group_stratified_kfold_splits(y_all, groups, k=n_folds, seed=seed)
        split = splits[fold]

        bundle: LoaderBundle = build_dataloaders(
            X_all,
            y_all,
            split,
            groups=groups,
            batch_size=batch_size,
            num_workers=num_workers,
            validation_split=validation_split,
            seed=seed,
            normalization=normalization,
            dataset_factory=dataset_factory,
        )
        self.train = bundle.train
        self.val = bundle.val
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
        test_info = str(self.n_test) if self.test is not None else "None"
        return (
            f"{self.__class__.__name__}("
            f"n_train={self.n_train}, "
            f"n_test={test_info}, "
            f"batch_size={self.train.batch_size})"
        )
