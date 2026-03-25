from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import scipy.io

from ..config import DREAMER_CACHE_DIR, DREAMER_MAT_PATH
from .preprocess_utils import (
    combine_windowed_parts,
    compute_baseline_template,
    remove_baseline_template,
    repeat_labels,
    samples_for_duration,
    segment_signal,
)

# ── Module-level logger ───────────────────────────────────────────────────────
_LOG = logging.getLogger(__name__)

# ── DREAMER dataset constants ─────────────────────────────────────────────────
#: Sampling frequency of DREAMER EEG recordings (Hz).
DREAMER_SFREQ: float = 128.0

#: Number of EEG channels recorded in DREAMER.
DREAMER_N_CHANNELS: int = 14

#: Number of subjects in the DREAMER dataset.
DREAMER_N_SUBJECTS: int = 23

#: Number of video-clip trials per subject.
DREAMER_N_TRIALS: int = 18

#: Length of each resting-state baseline recording (samples).
DREAMER_BASELINE_SAMPLES: int = 7808  # 61 s × 128 Hz

#: Number of label dimensions stored per window: [valence, arousal, dominance].
DREAMER_N_LABEL_DIMS: int = 3

#: Default window length (seconds).  Matches ACRNN's ``14 × 384`` input.
DEFAULT_WINDOW_SECS: float = 3.0


#: Channel names in order (informational, not used during preprocessing).
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


# ── Low-level helpers ─────────────────────────────────────────────────────────


def _n_window(window_secs: float, sfreq: float = DREAMER_SFREQ) -> int:
    """Return the integer number of samples per window."""
    return samples_for_duration(window_secs, sfreq)


# ── MATLAB struct helpers ─────────────────────────────────────────────────────


def _load_mat(mat_path: Path) -> tuple[list, float]:

    if not mat_path.exists():
        raise FileNotFoundError(
            f"DREAMER.mat not found at '{mat_path}'.  "
            "Make sure --mat-path points to the correct location."
        )

    _LOG.info("Loading '%s' …", mat_path)
    mat = scipy.io.loadmat(str(mat_path), squeeze_me=True)
    D = mat["DREAMER"]

    sfreq = float(D["EEG_SamplingRate"].item())
    subjects: list = list(D["Data"].item())

    _LOG.info(
        "  %d subjects  |  %d trials  |  %.0f Hz",
        len(subjects),
        DREAMER_N_TRIALS,
        sfreq,
    )
    return subjects, sfreq


# ── Per-subject preprocessing ─────────────────────────────────────────────────


def preprocess_subject(
    subject_struct,
    window_secs: float = DEFAULT_WINDOW_SECS,
    sfreq: float = DREAMER_SFREQ,
) -> tuple[np.ndarray, np.ndarray]:

    # ── Unpack EEG and label arrays ───────────────────────────────────────────
    eeg = subject_struct["EEG"].item()
    stimuli_arr = eeg["stimuli"].item()  # (18,) of (T_v, 14) arrays
    baseline_arr = eeg["baseline"].item()  # (18,) of (7808, 14) arrays

    scores_v = subject_struct["ScoreValence"].item().astype(np.int8)  # (18,)
    scores_a = subject_struct["ScoreArousal"].item().astype(np.int8)  # (18,)
    scores_d = subject_struct["ScoreDominance"].item().astype(np.int8)  # (18,)
    trial_labels = np.column_stack((scores_v, scores_a, scores_d))

    win_n = _n_window(window_secs, sfreq)

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for stimulus_raw, baseline_raw, label in zip(
        stimuli_arr,
        baseline_arr,
        trial_labels,
        strict=True,
    ):
        # Raw arrays: time × channels → (T_v, 14) and (7808, 14)
        stimulus = stimulus_raw.astype(np.float32).T  # (14, T_v)
        baseline = baseline_raw.astype(np.float32).T  # (14, 7808)

        # ── 1. Baseline-template removal (paper Eq. 1–2) ────────────────────
        baseline_template = compute_baseline_template(baseline, sfreq)
        stimulus_bc = remove_baseline_template(stimulus, baseline_template, sfreq)

        # ── 2. Segmentation ───────────────────────────────────────────────────
        windows = segment_signal(stimulus_bc, win_n)  # (n_win, 14, win_n)

        X_parts.append(windows)
        y_parts.append(repeat_labels(label, len(windows)))

    return combine_windowed_parts(X_parts, y_parts)


# ── Batch processing ──────────────────────────────────────────────────────────


def preprocess_all(
    mat_path: Path = DREAMER_MAT_PATH,
    cache_dir: Path = DREAMER_CACHE_DIR,
    window_secs: float = DEFAULT_WINDOW_SECS,
    overwrite: bool = False,
) -> None:

    cache_dir.mkdir(parents=True, exist_ok=True)

    subjects, sfreq = _load_mat(mat_path)
    win_n = _n_window(window_secs, sfreq)

    _LOG.info(
        "Preprocessing %d subjects → '%s'  (window=%g s = %d samples)",
        len(subjects),
        cache_dir,
        window_secs,
        win_n,
    )

    for subj_idx, subj_struct in enumerate(subjects):
        subject_tag = f"s{subj_idx + 1:02d}"
        out_path = cache_dir / f"{subject_tag}.npz"

        if out_path.exists() and not overwrite:
            _LOG.info("  [skip] %s – already cached", subject_tag)
            print(f"  [skip]  {subject_tag}  →  {out_path}  (use --overwrite to redo)")
            continue

        print(f"  Processing {subject_tag} … ", end="", flush=True)

        X, y_raw = preprocess_subject(
            subj_struct,
            window_secs=window_secs,
            sfreq=sfreq,
        )

        np.savez_compressed(out_path, X=X, y_raw=y_raw)

        print(
            f"done  "
            f"X={X.shape} ({X.dtype})  "
            f"y_raw={y_raw.shape} ({y_raw.dtype})  "
            f"→  {out_path.name}"
        )
        _LOG.info(
            "  Saved %s: X=%s  y_raw=%s",
            out_path,
            X.shape,
            y_raw.shape,
        )


# ── CLI ───────────────────────────────────────────────────────────────────────


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m acrnn.data.dreamer_preprocesser",
        description=(
            "Offline DREAMER preprocessor.  "
            "Reads DREAMER.mat, applies the paper-aligned baseline-template "
            "removal and 3-second segmentation pipeline, and writes per-subject "
            ".npz cache files for fast loading during training."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--mat-path",
        type=Path,
        default=DREAMER_MAT_PATH,
        metavar="PATH",
        help="Path to the DREAMER.mat file.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DREAMER_CACHE_DIR,
        metavar="PATH",
        help="Output directory for preprocessed .npz cache files.",
    )
    parser.add_argument(
        "--window-secs",
        type=float,
        default=DEFAULT_WINDOW_SECS,
        metavar="SECS",
        help="Window length in seconds.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Re-process subjects even if a cached .npz already exists.",
    )
    parser.add_argument(
        "--log-level",
        default="WARNING",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Python logging level.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(levelname)s  %(name)s  %(message)s",
    )

    win_n = _n_window(args.window_secs)
    print("=" * 58)
    print("  DREAMER Offline Preprocessor")
    print("=" * 58)
    print(f"  MAT file   : {args.mat_path}")
    print(f"  Cache dir  : {args.cache_dir}")
    print(
        f"  Window     : {args.window_secs} s  =  {win_n} samples"
        f" @ {DREAMER_SFREQ:.0f} Hz"
    )
    print(
        f"  Baseline   : {DREAMER_BASELINE_SAMPLES} samples"
        f" = {DREAMER_BASELINE_SAMPLES / DREAMER_SFREQ:.0f} s"
        " (averaged into a 1 s template and subtracted per second)"
    )
    print(f"  EEG chans  : {DREAMER_N_CHANNELS}  ({', '.join(DREAMER_CHANNEL_NAMES)})")
    print(f"  Overwrite  : {args.overwrite}")
    print()

    preprocess_all(
        mat_path=args.mat_path,
        cache_dir=args.cache_dir,
        window_secs=args.window_secs,
        overwrite=args.overwrite,
    )

    print()
    print("=" * 58)
    print("  Preprocessing complete.")
    print(f"  Cache directory: {args.cache_dir}")
    print("=" * 58)


if __name__ == "__main__":
    main()
