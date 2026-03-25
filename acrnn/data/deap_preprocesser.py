from __future__ import annotations

import argparse
import logging
import pickle
from pathlib import Path

import numpy as np

from .preprocess_utils import (
    combine_windowed_parts,
    compute_baseline_template,
    remove_baseline_template,
    repeat_labels,
    samples_for_duration,
    segment_signal,
)

# MNE is listed as a project dependency; import lazily inside the
# filter branch so the rest of the module works even without it.

# ── Module-level logger ───────────────────────────────────────────────────────
_LOG = logging.getLogger(__name__)

# ── DEAP dataset constants ────────────────────────────────────────────────────
#: Sampling frequency of the (pre-processed) DEAP recordings.
DEAP_SFREQ: float = 128.0

#: Number of EEG channels to retain (the remaining 8 are peripheral signals).
DEAP_N_EEG_CHANNELS: int = 32

#: Total number of trials (video clips) per subject.
DEAP_N_TRIALS: int = 40

#: Duration of the pre-stimulus baseline that is removed (seconds).
DEAP_BASELINE_SECS: float = 3.0

#: Number of label dimensions: [valence, arousal, dominance, liking].
DEAP_N_LABEL_DIMS: int = 4

#: Default window length used for segmentation (seconds).
DEFAULT_WINDOW_SECS: float = 3.0

#: Default cache directory (relative to the working directory).
DEFAULT_CACHE_DIR: Path = Path("data/deap/cache")

#: Default raw data directory.
DEFAULT_DATA_DIR: Path = Path("data/deap")


# ── Low-level helpers ─────────────────────────────────────────────────────────


def _n_baseline(
    sfreq: float = DEAP_SFREQ, baseline_secs: float = DEAP_BASELINE_SECS
) -> int:
    """Return the number of baseline samples to discard."""
    return samples_for_duration(baseline_secs, sfreq)


def _n_window(window_secs: float, sfreq: float = DEAP_SFREQ) -> int:
    """Return the number of samples per window."""
    return samples_for_duration(window_secs, sfreq)


def _apply_bandpass(
    trial: np.ndarray,
    sfreq: float,
    lowcut: float,
    highcut: float,
) -> np.ndarray:
    """Apply a zero-phase FIR bandpass filter to one trial via MNE.

    Parameters
    ----------
    trial:
        EEG data, shape ``(n_channels, n_times)``, any float dtype.
    sfreq:
        Sampling frequency in Hz.
    lowcut:
        Lower pass-band edge in Hz.
    highcut:
        Upper pass-band edge in Hz.

    Returns
    -------
    np.ndarray
        Filtered data with the same shape as *trial*, dtype ``float32``.
    """
    try:
        import mne  # noqa: PLC0415
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "mne is required for bandpass filtering.  Install it with: pip install mne"
        ) from exc

    filtered = mne.filter.filter_data(
        data=trial.astype(np.float64),
        sfreq=sfreq,
        l_freq=lowcut,
        h_freq=highcut,
        method="fir",
        fir_design="firwin",
        verbose=False,
    )
    return filtered.astype(np.float32)


# ── Per-subject preprocessing ─────────────────────────────────────────────────


def preprocess_subject(
    dat_path: Path,
    window_secs: float = DEFAULT_WINDOW_SECS,
    apply_filter: bool = False,
    lowcut: float = 4.0,
    highcut: float = 45.0,
) -> tuple[np.ndarray, np.ndarray]:

    # ── 1. Load pickle ────────────────────────────────────────────────────────
    with open(dat_path, "rb") as fh:
        subject = pickle.load(fh, encoding="latin1")

    raw_data: np.ndarray = subject["data"]  # (40, 40, 8064)
    raw_labels: np.ndarray = subject["labels"]  # (40, 4)

    # ── 2. Channel selection: keep first 32 EEG channels ─────────────────────
    eeg_data = raw_data[:, :DEAP_N_EEG_CHANNELS, :].astype(np.float32)  # (40, 32, 8064)
    labels = raw_labels.astype(np.float32)  # (40, 4)

    baseline_n = _n_baseline()
    win_n = _n_window(window_secs)

    X_parts: list[np.ndarray] = []
    y_parts: list[np.ndarray] = []

    for trial, label in zip(eeg_data, labels, strict=True):
        # ── 3. Baseline-template removal (paper Eq. 1–2) ────────────────────
        baseline = trial[:, :baseline_n]  # (32, 384)
        stimulus = trial[:, baseline_n:]  # (32, 7680)
        baseline_template = compute_baseline_template(baseline, DEAP_SFREQ)
        trial = remove_baseline_template(stimulus, baseline_template, DEAP_SFREQ)

        # ── 4. Optional bandpass filter ───────────────────────────────────────
        if apply_filter:
            trial = _apply_bandpass(trial, DEAP_SFREQ, lowcut, highcut)

        # ── 5. Segmentation ───────────────────────────────────────────────────
        windows = segment_signal(trial, win_n)  # (n_windows, 32, win_n)

        X_parts.append(windows)
        y_parts.append(repeat_labels(label, len(windows)))

    return combine_windowed_parts(X_parts, y_parts)


# ── Batch processing ──────────────────────────────────────────────────────────


def preprocess_all(
    data_dir: Path = DEFAULT_DATA_DIR,
    cache_dir: Path = DEFAULT_CACHE_DIR,
    window_secs: float = DEFAULT_WINDOW_SECS,
    apply_filter: bool = False,
    lowcut: float = 4.0,
    highcut: float = 45.0,
    overwrite: bool = False,
) -> None:

    cache_dir.mkdir(parents=True, exist_ok=True)

    dat_files = sorted(data_dir.glob("s*.dat"))
    if not dat_files:
        raise FileNotFoundError(
            f"No subject .dat files found in '{data_dir}'.  "
            "Make sure --data-dir points to the directory containing s01.dat … s32.dat."
        )

    win_n = _n_window(window_secs)
    _LOG.info(
        "Preprocessing %d subject files → '%s'  (window=%g s = %d samples)",
        len(dat_files),
        cache_dir,
        window_secs,
        win_n,
    )

    for dat_path in dat_files:
        subject_tag = dat_path.stem  # e.g. "s01"
        out_path = cache_dir / f"{subject_tag}.npz"

        if out_path.exists() and not overwrite:
            _LOG.info("  [skip] %s – already cached at '%s'", subject_tag, out_path)
            print(f"  [skip]  {subject_tag}  →  {out_path}  (use --overwrite to redo)")
            continue

        print(f"  Processing {subject_tag} … ", end="", flush=True)

        X, y_raw = preprocess_subject(
            dat_path=dat_path,
            window_secs=window_secs,
            apply_filter=apply_filter,
            lowcut=lowcut,
            highcut=highcut,
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


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m acrnn.data.deap_preprocesser",
        description=(
            "Offline DEAP preprocessor.  "
            "Reads raw .dat files, applies the paper-aligned baseline-template "
            "removal and 3-second segmentation pipeline, and writes per-subject "
            ".npz cache files for fast loading during training."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        metavar="PATH",
        help="Directory containing raw s01.dat … s32.dat files.",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=DEFAULT_CACHE_DIR,
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
        "--filter",
        dest="apply_filter",
        action="store_true",
        default=False,
        help=(
            "Apply an additional 4–45 Hz FIR bandpass filter via MNE.  "
            "DEAP is already filtered; enable only if you need a fresh filter pass."
        ),
    )
    parser.add_argument(
        "--lowcut",
        type=float,
        default=4.0,
        metavar="HZ",
        help="Lower pass-band edge for the optional filter (Hz).",
    )
    parser.add_argument(
        "--highcut",
        type=float,
        default=45.0,
        metavar="HZ",
        help="Upper pass-band edge for the optional filter (Hz).",
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
    print("=" * 56)
    print("  DEAP Offline Preprocessor")
    print("=" * 56)
    print(f"  Data dir   : {args.data_dir}")
    print(f"  Cache dir  : {args.cache_dir}")
    print(
        f"  Window     : {args.window_secs} s  =  {win_n} samples @ {DEAP_SFREQ:.0f} Hz"
    )
    print(
        f"  Baseline   : {DEAP_BASELINE_SECS} s  =  {_n_baseline()} samples"
        " (averaged into a 1 s template and subtracted per second)"
    )
    print(f"  EEG chans  : {DEAP_N_EEG_CHANNELS}  (peripheral channels discarded)")
    print(
        "  Filter     : "
        + (
            f"ON  ({args.lowcut}–{args.highcut} Hz FIR)"
            if args.apply_filter
            else "OFF  (DEAP is already 4–45 Hz filtered)"
        )
    )
    print(f"  Overwrite  : {args.overwrite}")
    print()

    preprocess_all(
        data_dir=args.data_dir,
        cache_dir=args.cache_dir,
        window_secs=args.window_secs,
        apply_filter=args.apply_filter,
        lowcut=args.lowcut,
        highcut=args.highcut,
        overwrite=args.overwrite,
    )

    print()
    print("=" * 56)
    print("  Preprocessing complete.")
    print(f"  Cache directory: {args.cache_dir}")
    print("=" * 56)


if __name__ == "__main__":
    main()
