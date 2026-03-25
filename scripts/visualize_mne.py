from __future__ import annotations

import argparse

import mne
import numpy as np

from acrnn.config import DREAMER_CACHE_DIR
from acrnn.data.dreamer_loader import (
    DREAMER_CHANNEL_NAMES,
    DREAMER_SFREQ,
    VALID_TARGETS,
    load_dreamer_arrays,
)

_DEFAULT_CACHE_DIR = str(DREAMER_CACHE_DIR)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize one DREAMER EEG sample with MNE",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("index", type=int, help="Sample index to visualize")
    parser.add_argument(
        "--target",
        choices=sorted(VALID_TARGETS),
        default="valence",
        help="Dataset target to load.",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=_DEFAULT_CACHE_DIR,
        help=(
            "Directory containing the preprocessed .npz cache files "
            "produced by dreamer_preprocesser.py."
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Visible time span in seconds for the raw plot window.",
    )
    return parser.parse_args()


def build_raw(sample: np.ndarray) -> mne.io.RawArray:
    if sample.ndim != 2:
        raise ValueError(
            f"Expected a 2D sample shaped (channels, timepoints), got {sample.shape!r}"
        )
    if sample.shape[0] != len(DREAMER_CHANNEL_NAMES):
        raise ValueError(
            f"Expected {len(DREAMER_CHANNEL_NAMES)} channels, got shape {sample.shape!r}"
        )

    info = mne.create_info(
        ch_names=DREAMER_CHANNEL_NAMES,
        sfreq=DREAMER_SFREQ,
        ch_types="eeg",
    )
    raw = mne.io.RawArray(sample, info)
    raw.set_montage("standard_1020")
    return raw


def main() -> None:
    args = parse_args()
    X, y = load_dreamer_arrays(args.target, cache_dir=args.cache_dir)

    if not 0 <= args.index < len(X):
        raise IndexError(f"index must be in [0, {len(X) - 1}], got {args.index}")

    sample = X[args.index]
    label = y[args.index]
    raw = build_raw(sample)

    print(f"Target      : {args.target}")
    print(f"Cache dir   : {args.cache_dir}")
    print(f"Sample index: {args.index}")
    print(f"Label       : {label}  (0 = low, 1 = high)")
    print(f"Shape       : {sample.shape}  (channels × timepoints)")
    print(f"Duration    : {sample.shape[1] / DREAMER_SFREQ:.2f} s")

    raw.plot(
        duration=min(args.duration, sample.shape[1] / DREAMER_SFREQ),
        scalings="auto",
        title=f"DREAMER sample {args.index} | {args.target} = {label}",
        show=True,
        block=True,
    )
    raw.plot_sensors(show_names=True)


if __name__ == "__main__":
    main()
