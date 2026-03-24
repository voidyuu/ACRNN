from __future__ import annotations

import argparse

import mne
import numpy as np

from acrnn.data.dreamer import VALID_TARGETS, load_dreamer_arrays

# DREAMER uses these 14 EEG channels in the DEAP-style headset layout.
DREAMER_CHANNEL_NAMES = [
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
DREAMER_SFREQ = 128.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize one DREAMER EEG sample with MNE")
    parser.add_argument("index", type=int, help="Sample index to visualize")
    parser.add_argument(
        "--target",
        choices=sorted(VALID_TARGETS),
        default="valence",
        help="Dataset target to load (default: valence)",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Optional HuggingFace cache directory override",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Visible time span in seconds for the raw plot window (default: 10.0)",
    )
    return parser.parse_args()


def build_raw(sample: np.ndarray) -> mne.io.RawArray:
    if sample.ndim != 2:
        raise ValueError(f"Expected a 2D sample shaped (channels, timepoints), got {sample.shape!r}")
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
    print(f"Sample index: {args.index}")
    print(f"Label       : {label}")
    print(f"Shape       : {sample.shape}")
    print(f"Duration    : {sample.shape[1] / DREAMER_SFREQ:.2f} s")

    raw.plot(
        duration=min(args.duration, sample.shape[1] / DREAMER_SFREQ),
        scalings="auto",
        title=f"DREAMER sample {args.index} ({args.target})",
        show=True,
        block=True,
    )
    raw.plot_sensors(show_names=True)


if __name__ == "__main__":
    main()
