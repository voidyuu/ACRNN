from __future__ import annotations

import numpy as np


def samples_for_duration(duration_secs: float, sfreq: float) -> int:
    """Return the integer number of samples in a duration."""
    return round(duration_secs * sfreq)


def samples_per_second(sfreq: float) -> int:
    """Return the integer number of samples in one second."""
    return round(sfreq)


def segment_signal(
    signal: np.ndarray,
    window_samples: int,
) -> np.ndarray:
    """Split a channel-first signal into fixed-length windows.

    Parameters
    ----------
    signal:
        EEG array with shape ``(n_channels, n_times)``.
    window_samples:
        Number of samples in each output window.

    Returns
    -------
    np.ndarray
        Array with shape ``(n_windows, n_channels, window_samples)``.
        Trailing samples shorter than one full window are discarded.
    """
    n_channels, n_times = signal.shape
    n_windows = n_times // window_samples
    trimmed = signal[:, : n_windows * window_samples]
    reshaped = trimmed.reshape(n_channels, n_windows, window_samples)
    return reshaped.transpose(1, 0, 2)


def repeat_labels(label: np.ndarray, count: int) -> np.ndarray:
    """Repeat one label row ``count`` times."""
    return np.repeat(label[None, :], count, axis=0)


def combine_windowed_parts(
    X_parts: list[np.ndarray],
    y_parts: list[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Concatenate accumulated feature and label chunks."""
    return np.concatenate(X_parts, axis=0), np.concatenate(y_parts, axis=0)


def _reshape_into_seconds(signal: np.ndarray, sfreq: float) -> np.ndarray:
    """Reshape a channel-first signal into one-second slices.

    Parameters
    ----------
    signal:
        EEG array with shape ``(n_channels, n_times)``.
    sfreq:
        Sampling frequency in Hz.

    Returns
    -------
    np.ndarray
        Array with shape ``(n_seconds, n_channels, samples_per_second)``.
        Trailing samples shorter than one second are discarded.
    """
    second_n = samples_per_second(sfreq)
    n_channels, n_times = signal.shape
    n_seconds = n_times // second_n
    if n_seconds < 1:
        raise ValueError(
            f"Need at least one full second of EEG, got shape {signal.shape!r} at {sfreq} Hz."
        )

    trimmed = signal[:, : n_seconds * second_n]
    reshaped = trimmed.reshape(n_channels, n_seconds, second_n)
    return reshaped.transpose(1, 0, 2)


def compute_baseline_template(
    baseline: np.ndarray,
    sfreq: float,
) -> np.ndarray:
    """Compute the per-second baseline template from Eq. (1) in ACRNN.

    The paper averages one-second baseline chunks into a single
    ``(n_channels, samples_per_second)`` template, which is then
    subtracted from each one-second trial slice.
    """
    baseline_seconds = _reshape_into_seconds(baseline, sfreq)
    return baseline_seconds.mean(axis=0).astype(np.float32)


def remove_baseline_template(
    trial: np.ndarray,
    baseline_template: np.ndarray,
    sfreq: float,
) -> np.ndarray:
    """Apply Eq. (2) from the ACRNN paper to one trial.

    The trial is first split into one-second slices, each slice is baseline
    corrected using ``baseline_template``, and the corrected slices are then
    concatenated back to a channel-first signal.
    """
    trial_seconds = _reshape_into_seconds(trial, sfreq)
    corrected = trial_seconds - baseline_template[None, :, :]
    return corrected.transpose(1, 0, 2).reshape(trial.shape[0], -1).astype(np.float32)
