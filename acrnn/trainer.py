from __future__ import annotations

import json
from copy import deepcopy
from dataclasses import dataclass
from pathlib import Path
from time import time
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import VALID_DATASETS, get_default_threshold
from .data import (
    DEAP_VALID_FOLDS_INDEPENDENT,
    DEAP_VALID_SUBJECTS,
    DEAP_VALID_TARGETS,
    DREAMER_VALID_FOLDS_INDEPENDENT,
    DREAMER_VALID_SUBJECTS,
    DREAMER_VALID_TARGETS,
    DeapDataloader,
    DreamerDataloader,
)
from .model import ACRNN
from .utils import make_timestamp_label, resolve_device

VALID_MODES = {"subject_dependent", "subject_independent"}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    dataloader_cls: Callable[..., DeapDataloader | DreamerDataloader]
    valid_targets: set[str]
    valid_subjects: list[int]
    independent_folds: list[int]


_DATASET_CONFIGS: dict[str, DatasetConfig] = {
    "deap": DatasetConfig(
        name="deap",
        dataloader_cls=DeapDataloader,
        valid_targets=DEAP_VALID_TARGETS,
        valid_subjects=DEAP_VALID_SUBJECTS,
        independent_folds=DEAP_VALID_FOLDS_INDEPENDENT,
    ),
    "dreamer": DatasetConfig(
        name="dreamer",
        dataloader_cls=DreamerDataloader,
        valid_targets=DREAMER_VALID_TARGETS,
        valid_subjects=DREAMER_VALID_SUBJECTS,
        independent_folds=DREAMER_VALID_FOLDS_INDEPENDENT,
    ),
}


def _infer_input_shape(train_loader: DataLoader) -> tuple[int, int]:
    """Infer ``(channels, timepoints)`` from one training batch."""
    xb, _ = next(iter(train_loader))
    if xb.ndim != 3:
        raise ValueError(
            f"Expected batched EEG tensors with shape (batch, channels, timepoints), got {tuple(xb.shape)}."
        )
    return int(xb.shape[1]), int(xb.shape[2])


def _get_dataset_config(dataset: str) -> DatasetConfig:
    try:
        return _DATASET_CONFIGS[dataset]
    except KeyError as exc:
        raise ValueError(
            f"dataset must be one of {sorted(VALID_DATASETS)}, got {dataset!r}"
        ) from exc


def _resolve_eval_runs(
    config: DatasetConfig,
    mode: str,
    subject_id: int | None,
    n_folds: int,
) -> list[tuple[int | None, int]]:
    if mode not in VALID_MODES:
        raise ValueError(f"mode must be one of {sorted(VALID_MODES)}, got {mode!r}")

    if mode == "subject_independent":
        if subject_id is not None:
            raise ValueError("subject_id cannot be used with subject_independent mode.")
        return [(None, fold) for fold in config.independent_folds]

    if n_folds < 2:
        raise ValueError(f"n_folds must be at least 2, got {n_folds}")

    if subject_id is not None:
        if subject_id not in config.valid_subjects:
            raise ValueError(
                f"subject_id must be one of {config.valid_subjects}, got {subject_id}"
            )
        subject_ids = [subject_id]
    else:
        subject_ids = config.valid_subjects

    return [(sid, fold) for sid in subject_ids for fold in range(n_folds)]


def _build_dataloader(
    config: DatasetConfig,
    target: str,
    mode: str,
    fold: int,
    subject_id: int | None,
    n_folds: int,
    threshold: float,
    cache_dir: str | Path | None,
    batch_size: int,
    num_workers: int,
) -> DeapDataloader | DreamerDataloader:
    kwargs: dict[str, object] = {
        "target": target,
        "mode": mode,
        "fold": fold,
        "threshold": threshold,
        "batch_size": batch_size,
        "num_workers": num_workers,
    }
    if cache_dir is not None:
        kwargs["cache_dir"] = cache_dir
    if mode == "subject_dependent":
        kwargs["subject_id"] = subject_id
        kwargs["n_folds"] = n_folds
    return config.dataloader_cls(**kwargs)


def _resolve_test_subject_id(
    config: DatasetConfig,
    mode: str,
    run_subject_id: int | None,
    fold: int,
) -> int:
    if mode == "subject_independent":
        return config.valid_subjects[fold]
    if run_subject_id is None:
        raise ValueError("run_subject_id must be set in subject_dependent mode.")
    return run_subject_id


def _save_subject_accuracy_plots(
    output_dir: Path | None,
    dataset: str,
    target: str,
    mode: str,
    subject_scores: dict[int, list[float]],
) -> None:
    if output_dir is None or not subject_scores:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    subject_ids = sorted(subject_scores)
    mean_scores = [float(np.mean(subject_scores[sid])) for sid in subject_ids]
    std_scores = [float(np.std(subject_scores[sid])) for sid in subject_ids]

    x = np.arange(len(subject_ids))

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(x, mean_scores, marker="o", linewidth=2, color="#F58518")
    ax.fill_between(
        x,
        np.maximum(0.0, np.array(mean_scores) - np.array(std_scores)),
        np.minimum(1.0, np.array(mean_scores) + np.array(std_scores)),
        color="#F58518",
        alpha=0.15,
    )
    ax.set_title(f"{dataset.upper()} {target} {mode} subject accuracy")
    ax.set_xlabel("Subject")
    ax.set_ylabel("Accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(x)
    ax.set_xticklabels([str(sid) for sid in subject_ids])
    ax.grid(axis="y", linestyle="--", alpha=0.3)
    fig.tight_layout()
    line_path = output_dir / "subject_accuracy_line.png"
    fig.savefig(line_path, dpi=200)
    plt.close(fig)

    print("\n  Subject accuracy plot saved:")
    print(f"    Line chart: {line_path}")


def _save_metrics(
    output_dir: Path | None,
    dataset: str,
    target: str,
    mode: str,
    overall_mean: float,
    overall_std: float,
) -> None:
    if output_dir is None:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics = {
        "dataset": dataset,
        "target": target,
        "mode": mode,
        "overall_mean": overall_mean,
        "overall_std": overall_std,
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"  Metrics saved: {metrics_path}")


def _save_best_model(
    output_dir: Path | None,
    dataset: str,
    target: str,
    mode: str,
    best_run: tuple[int | None, int, float, dict[str, torch.Tensor]] | None,
    epochs: int,
    batch_size: int,
    threshold: float,
) -> None:
    if output_dir is None or best_run is None:
        return

    best_subject, best_fold, best_acc, best_state = best_run
    output_dir.mkdir(parents=True, exist_ok=True)
    filename = output_dir / "weight.pt"

    torch.save(
        {
            "state_dict": best_state,
            "dataset": dataset,
            "target": target,
            "mode": mode,
            "subject_id": best_subject,
            "fold": best_fold,
            "test_acc": best_acc,
            "epochs": epochs,
            "batch_size": batch_size,
            "threshold": threshold,
        },
        filename,
    )
    subject_text = (
        "all-subject sweep" if best_subject is None else f"subject {best_subject}"
    )
    print(
        f"\n  Best weights ({subject_text}, fold {best_fold + 1}, acc {best_acc * 100:.2f}%) saved -> {filename}"
    )


class EarlyStopping:
    """Stops training when the monitored loss stops improving."""

    def __init__(self, patience: int = 0, min_delta: float = 0.0) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self._best = float("inf")
        self._counter = 0

    def step(self, loss: float) -> bool:
        if loss < self._best - self.min_delta:
            self._best = loss
            self._counter = 0
            return True
        self._counter += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.patience > 0 and self._counter >= self.patience

    @property
    def best(self) -> float:
        return self._best


def train_model(
    model: ACRNN,
    train_loader: DataLoader,
    test_loader: DataLoader | None,
    device: torch.device,
    epochs: int = 200,
    log_every: int = 1,
    patience: int = 0,
) -> dict[str, torch.Tensor]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    early_stopping = EarlyStopping(patience=patience)
    best_state_dict = deepcopy(model.state_dict())

    epoch_w = len(str(epochs))
    dataset_size = len(train_loader.dataset)  # type: ignore[arg-type]

    t_start = time()

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        correct = 0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)
            preds = torch.argmax(logits, dim=1)
            correct += (preds == yb).sum().item()

        avg_loss = epoch_loss / dataset_size
        train_acc = correct / dataset_size
        test_acc: float | None = None

        if test_loader is not None:
            test_acc = evaluate_model(model, test_loader, device)

        is_best = early_stopping.step(avg_loss)
        if is_best:
            best_state_dict = deepcopy(model.state_dict())

        if (epoch + 1) % log_every == 0:
            elapsed = time() - t_start
            best_mark = "  <- best" if is_best else ""
            test_acc_text = (
                f" | Test Acc: {test_acc * 100:5.1f}%" if test_acc is not None else ""
            )
            print(
                f"  Epoch {epoch + 1:{epoch_w}d}/{epochs}"
                f" | Loss: {avg_loss:.4f}"
                f" | Train Acc: {train_acc * 100:5.1f}%"
                f"{test_acc_text}"
                f" | Best Loss: {early_stopping.best:.4f}"
                f" | Elapsed: {elapsed:6.1f}s"
                f"{best_mark}"
            )

        if early_stopping.should_stop:
            print(
                f"  Early stopping at epoch {epoch + 1} because loss did not improve for {patience} epochs."
            )
            break

    return best_state_dict


def evaluate_model(
    model: ACRNN, test_loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds)
            all_targets.append(yb)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    return (all_preds == all_targets).sum().item() / len(all_preds)


def cross_validate_model(
    dataset: str,
    target: str,
    mode: str = "subject_dependent",
    subject_id: int | None = None,
    n_folds: int = 10,
    threshold: float | None = None,
    cache_dir: str | Path | None = None,
    device: str | None = None,
    epochs: int = 200,
    batch_size: int = 10,
    num_workers: int = 0,
    log_every: int = 1,
    patience: int = 0,
    save_dir: str | None = "checkpoints",
) -> tuple[float, float]:
    config = _get_dataset_config(dataset)
    if target not in config.valid_targets:
        raise ValueError(
            f"target must be one of {sorted(config.valid_targets)} for dataset {dataset!r}, got {target!r}"
        )

    training_device = resolve_device(device)
    threshold = get_default_threshold(dataset, target) if threshold is None else threshold
    eval_runs = _resolve_eval_runs(config, mode, subject_id, n_folds)

    all_run_acc: list[float] = []
    subject_scores: dict[int, list[float]] = {}
    best_run: tuple[int | None, int, float, dict[str, torch.Tensor]] | None = None

    for run_idx, (run_subject_id, fold) in enumerate(eval_runs, start=1):
        print(f"\n{'=' * 60}")
        print(
            f"  Run {run_idx}/{len(eval_runs)}"
            f"  |  dataset: {dataset}"
            f"  |  target: {target}"
            f"  |  mode: {mode}"
            f"  |  device: {training_device}"
        )
        if run_subject_id is None:
            print(f"  Fold: {fold + 1}/{len(config.independent_folds)}")
        else:
            print(f"  Subject: {run_subject_id}  |  Fold: {fold + 1}/{n_folds}")
        print(f"{'=' * 60}")

        dl = _build_dataloader(
            config=config,
            target=target,
            mode=mode,
            fold=fold,
            subject_id=run_subject_id,
            n_folds=n_folds,
            threshold=threshold,
            cache_dir=cache_dir,
            batch_size=batch_size,
            num_workers=num_workers,
        )
        num_channels, num_timepoints = _infer_input_shape(dl.train)

        train_size = len(dl.train.dataset)  # type: ignore[arg-type]
        print(f"  Train samples: {train_size}")
        print(f"  Input shape : {num_channels} x {num_timepoints}")

        model = ACRNN(
            reduce=2,
            k=40,
            num_channels=num_channels,
            num_timepoints=num_timepoints,
        ).to(training_device)

        best_state = train_model(
            model,
            dl.train,
            dl.test,
            training_device,
            epochs=epochs,
            log_every=log_every,
            patience=patience,
        )
        model.load_state_dict(best_state)

        assert dl.test is not None
        test_size = len(dl.test.dataset)  # type: ignore[arg-type]
        acc = evaluate_model(model, dl.test, training_device)

        print(f"\n  Test samples : {test_size}")
        print(f"  Accuracy     : {acc * 100:.2f}%")

        all_run_acc.append(acc)
        test_subject_id = _resolve_test_subject_id(config, mode, run_subject_id, fold)
        subject_scores.setdefault(test_subject_id, []).append(acc)

        if best_run is None or acc > best_run[2]:
            best_run = (run_subject_id, fold, acc, best_state)

    if subject_scores and (mode == "subject_independent" or subject_id is None):
        print(f"\n{'-' * 60}")
        print("  Per-subject summary")
        print(f"{'-' * 60}")
        for sid in sorted(subject_scores):
            scores = subject_scores[sid]
            mean = float(np.mean(scores))
            std = float(np.std(scores))
            print(f"  Subject {sid:02d}: {mean * 100:.2f}% +- {std * 100:.2f}%")

    overall_mean = float(np.mean(all_run_acc))
    overall_std = float(np.std(all_run_acc))
    timestamp_label = make_timestamp_label()
    output_dir = (
        None
        if save_dir is None
        else Path(save_dir) / mode / dataset / target / timestamp_label
    )

    print(f"\n{'=' * 60}")
    print(f"  Overall result  |  dataset: {dataset}  |  target: {target}")
    print(f"  Accuracy: {overall_mean * 100:.2f}% +- {overall_std * 100:.2f}%")
    print(f"{'=' * 60}")

    _save_subject_accuracy_plots(
        output_dir=output_dir,
        dataset=dataset,
        target=target,
        mode=mode,
        subject_scores=subject_scores,
    )
    _save_metrics(
        output_dir=output_dir,
        dataset=dataset,
        target=target,
        mode=mode,
        overall_mean=overall_mean,
        overall_std=overall_std,
    )

    _save_best_model(
        output_dir=output_dir,
        dataset=dataset,
        target=target,
        mode=mode,
        best_run=best_run,
        epochs=epochs,
        batch_size=batch_size,
        threshold=threshold,
    )

    return overall_mean, overall_std
