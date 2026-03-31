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
METRIC_NAMES = ("accuracy", "precision", "recall", "f1")
VALID_CLASS_WEIGHTING = {"none", "balanced"}


@dataclass(frozen=True)
class DatasetConfig:
    name: str
    dataloader_cls: Callable[..., DeapDataloader | DreamerDataloader]
    valid_targets: set[str]
    valid_subjects: list[int]
    independent_folds: list[int]


@dataclass(frozen=True)
class EvalMetrics:
    accuracy: float
    precision: float
    recall: float
    f1: float

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
        }


@dataclass(frozen=True)
class EvalResult:
    metrics: EvalMetrics
    confusion_matrix: np.ndarray
    loss: float | None = None
    decision_threshold: float = 0.5


@dataclass(frozen=True)
class TrainResult:
    state_dict: dict[str, torch.Tensor]
    decision_threshold: float
    best_epoch: int
    best_score: float


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
    validation_split: float,
    normalization: str,
    seed: int,
) -> DeapDataloader | DreamerDataloader:
    kwargs: dict[str, object] = {
        "target": target,
        "mode": mode,
        "fold": fold,
        "threshold": threshold,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "validation_split": validation_split,
        "normalization": normalization,
        "seed": seed,
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


def _safe_divide(numerator: float, denominator: float) -> float:
    if denominator == 0:
        return 0.0
    return numerator / denominator


def _compute_eval_metrics(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> EvalMetrics:
    preds = preds.to(torch.long)
    targets = targets.to(torch.long)

    tp = float(((preds == 1) & (targets == 1)).sum().item())
    fp = float(((preds == 1) & (targets == 0)).sum().item())
    fn = float(((preds == 0) & (targets == 1)).sum().item())
    correct = float((preds == targets).sum().item())
    total = float(targets.numel())

    precision = _safe_divide(tp, tp + fp)
    recall = _safe_divide(tp, tp + fn)
    f1 = _safe_divide(2.0 * precision * recall, precision + recall)

    return EvalMetrics(
        accuracy=_safe_divide(correct, total),
        precision=precision,
        recall=recall,
        f1=f1,
    )


def _compute_confusion_matrix(
    preds: torch.Tensor,
    targets: torch.Tensor,
) -> np.ndarray:
    preds_np = preds.cpu().numpy().astype(np.int64)
    targets_np = targets.cpu().numpy().astype(np.int64)
    confusion = np.zeros((2, 2), dtype=np.int64)

    for target, pred in zip(targets_np, preds_np, strict=True):
        confusion[target, pred] += 1

    return confusion


def _make_metric_store() -> dict[str, list[float]]:
    return {name: [] for name in METRIC_NAMES}


def _compute_balanced_class_weights(
    train_loader: DataLoader,
    num_labels: int = 2,
) -> torch.Tensor | None:
    dataset = train_loader.dataset  # type: ignore[assignment]
    labels = getattr(dataset, "labels", None)

    if labels is None:
        counts = torch.zeros(num_labels, dtype=torch.float32)
        for _, yb in train_loader:
            counts += torch.bincount(yb.to(torch.long), minlength=num_labels).to(torch.float32)
    else:
        label_tensor = torch.as_tensor(labels, dtype=torch.long)
        counts = torch.bincount(label_tensor, minlength=num_labels).to(torch.float32)

    nonzero = counts > 0
    if nonzero.sum().item() < 2:
        return None

    weights = torch.ones(num_labels, dtype=torch.float32)
    weights[nonzero] = counts[nonzero].sum() / (float(nonzero.sum().item()) * counts[nonzero])
    return weights


def _score_metrics(metrics: EvalMetrics) -> float:
    # Keep accuracy primary while still penalising collapse through F1.
    return (metrics.accuracy * 0.75) + (metrics.f1 * 0.25)


def _predict_from_probabilities(
    positive_probabilities: torch.Tensor,
    decision_threshold: float,
) -> torch.Tensor:
    return (positive_probabilities >= decision_threshold).to(torch.long)


def _collect_probabilities(
    model: ACRNN,
    loader: DataLoader,
    device: torch.device,
    criterion: nn.Module | None = None,
) -> tuple[torch.Tensor, torch.Tensor, float | None]:
    model.eval()
    all_probabilities: list[torch.Tensor] = []
    all_targets: list[torch.Tensor] = []
    total_loss = 0.0
    total_examples = 0

    with torch.no_grad():
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            logits = model(xb)
            probabilities = torch.softmax(logits, dim=1)[:, 1]
            all_probabilities.append(probabilities.cpu())
            all_targets.append(yb.cpu())

            if criterion is not None:
                total_loss += float(criterion(logits, yb).item()) * xb.size(0)
                total_examples += xb.size(0)

    average_loss = (
        total_loss / total_examples if criterion is not None and total_examples > 0 else None
    )
    return torch.cat(all_probabilities), torch.cat(all_targets), average_loss


def _result_from_probabilities(
    positive_probabilities: torch.Tensor,
    targets: torch.Tensor,
    decision_threshold: float,
    loss: float | None = None,
) -> EvalResult:
    preds = _predict_from_probabilities(positive_probabilities, decision_threshold)
    return EvalResult(
        metrics=_compute_eval_metrics(preds, targets),
        confusion_matrix=_compute_confusion_matrix(preds, targets),
        loss=loss,
        decision_threshold=decision_threshold,
    )


def _select_decision_threshold(
    positive_probabilities: torch.Tensor,
    targets: torch.Tensor,
    min_precision: float,
    min_recall: float,
) -> EvalResult:
    candidate_thresholds = torch.linspace(0.25, 0.75, steps=51)
    best_result: EvalResult | None = None
    best_score = float("-inf")
    fallback_result: EvalResult | None = None
    fallback_score = float("-inf")

    for threshold in candidate_thresholds.tolist():
        result = _result_from_probabilities(
            positive_probabilities,
            targets,
            decision_threshold=float(threshold),
        )
        score = _score_metrics(result.metrics)
        if score > fallback_score:
            fallback_score = score
            fallback_result = result

        if (
            result.metrics.precision >= min_precision
            and result.metrics.recall >= min_recall
            and score > best_score
        ):
            best_score = score
            best_result = result

    return best_result or fallback_result or _result_from_probabilities(
        positive_probabilities,
        targets,
        decision_threshold=0.5,
    )


def _summarise_metric_store(metric_store: dict[str, list[float]]) -> dict[str, dict[str, float]]:
    return {
        name: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
        }
        for name, values in metric_store.items()
    }


def _save_subject_metric_plots(
    output_dir: Path | None,
    dataset: str,
    target: str,
    mode: str,
    subject_scores: dict[int, dict[str, list[float]]],
) -> None:
    if output_dir is None or not subject_scores:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    subject_ids = sorted(subject_scores)
    x = np.arange(len(subject_ids))

    fig, axes = plt.subplots(2, 2, figsize=(14, 9), sharex=True)
    metric_styles = {
        "accuracy": ("Accuracy", "#F58518"),
        "precision": ("Precision", "#4C78A8"),
        "recall": ("Recall", "#54A24B"),
        "f1": ("F1 Score", "#E45756"),
    }

    for ax, metric_name in zip(axes.flat, METRIC_NAMES, strict=True):
        label, color = metric_styles[metric_name]
        mean_scores = [
            float(np.mean(subject_scores[sid][metric_name])) for sid in subject_ids
        ]
        std_scores = [
            float(np.std(subject_scores[sid][metric_name])) for sid in subject_ids
        ]
        ax.plot(x, mean_scores, marker="o", linewidth=2, color=color)
        ax.fill_between(
            x,
            np.maximum(0.0, np.array(mean_scores) - np.array(std_scores)),
            np.minimum(1.0, np.array(mean_scores) + np.array(std_scores)),
            color=color,
            alpha=0.15,
        )
        ax.set_title(label)
        ax.set_ylim(0.0, 1.0)
        ax.set_xticks(x)
        ax.set_xticklabels([str(sid) for sid in subject_ids])
        ax.grid(axis="y", linestyle="--", alpha=0.3)

    axes[1, 0].set_xlabel("Subject")
    axes[1, 1].set_xlabel("Subject")
    axes[0, 0].set_ylabel("Score")
    axes[1, 0].set_ylabel("Score")
    fig.suptitle(f"{dataset.upper()} {target} {mode} subject metrics", fontsize=14)
    fig.tight_layout()
    metrics_plot_path = output_dir / "subject_metrics_grid.png"
    fig.savefig(metrics_plot_path, dpi=200)
    plt.close(fig)

    print("\n  Subject metric plot saved:")
    print(f"    Grid chart: {metrics_plot_path}")


def _save_metrics(
    output_dir: Path | None,
    dataset: str,
    target: str,
    mode: str,
    overall_metrics: dict[str, dict[str, float]],
    confusion_matrix: np.ndarray,
) -> None:
    if output_dir is None:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "metrics.json"
    metrics = {
        "dataset": dataset,
        "target": target,
        "mode": mode,
        "overall_metrics": overall_metrics,
        "confusion_matrix": confusion_matrix.tolist(),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2) + "\n", encoding="utf-8")

    print(f"  Metrics saved: {metrics_path}")


def _save_confusion_matrix_plot(
    output_dir: Path | None,
    dataset: str,
    target: str,
    mode: str,
    confusion_matrix: np.ndarray,
) -> None:
    if output_dir is None:
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    im = ax.imshow(confusion_matrix, cmap="Blues", vmin=0)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Pred 0", "Pred 1"])
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["True 0", "True 1"])
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_title(f"{dataset.upper()} {target} {mode} confusion matrix")

    max_value = int(confusion_matrix.max()) if confusion_matrix.size else 0
    threshold = max_value / 2.0 if max_value > 0 else 0.0
    for row in range(confusion_matrix.shape[0]):
        for col in range(confusion_matrix.shape[1]):
            value = int(confusion_matrix[row, col])
            text_color = "white" if value > threshold else "black"
            ax.text(col, row, str(value), ha="center", va="center", color=text_color)

    fig.tight_layout()
    output_path = output_dir / "confusion_matrix.png"
    fig.savefig(output_path, dpi=200)
    plt.close(fig)

    print("  Confusion matrix plot saved:")
    print(f"    Heatmap: {output_path}")


def _save_best_model(
    output_dir: Path | None,
    dataset: str,
    target: str,
    mode: str,
    best_run: tuple[int | None, int, EvalMetrics, dict[str, torch.Tensor], float] | None,
    epochs: int,
    batch_size: int,
    threshold: float,
) -> None:
    if output_dir is None or best_run is None:
        return

    best_subject, best_fold, best_metrics, best_state, decision_threshold = best_run
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
            "test_acc": best_metrics.accuracy,
            "test_precision": best_metrics.precision,
            "test_recall": best_metrics.recall,
            "test_f1": best_metrics.f1,
            "epochs": epochs,
            "batch_size": batch_size,
            "threshold": threshold,
            "decision_threshold": decision_threshold,
        },
        filename,
    )
    subject_text = (
        "all-subject sweep" if best_subject is None else f"subject {best_subject}"
    )
    print(
        f"\n  Best weights ({subject_text}, fold {best_fold + 1}, acc {best_metrics.accuracy * 100:.2f}%) saved -> {filename}"
    )


def train_model(
    model: ACRNN,
    train_loader: DataLoader,
    val_loader: DataLoader | None,
    device: torch.device,
    epochs: int = 200,
    log_every: int = 1,
    min_epochs: int = 20,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-2,
    optimizer_name: str = "adamw",
    scheduler_name: str = "plateau",
    grad_clip_norm: float = 1.0,
    threshold_min_precision: float = 0.65,
    threshold_min_recall: float = 0.65,
    class_weighting: str = "balanced",
) -> TrainResult:
    if class_weighting not in VALID_CLASS_WEIGHTING:
        raise ValueError(
            f"class_weighting must be one of {sorted(VALID_CLASS_WEIGHTING)}, got {class_weighting!r}"
        )

    criterion_weights = (
        _compute_balanced_class_weights(train_loader)
        if class_weighting == "balanced"
        else None
    )
    criterion = nn.CrossEntropyLoss(
        weight=criterion_weights.to(device) if criterion_weights is not None else None
    )

    if criterion_weights is not None:
        weights_text = ", ".join(f"{float(weight):.4f}" for weight in criterion_weights)
        print(f"  Class weights: [{weights_text}]")
    elif class_weighting == "balanced":
        print("  Class weights: skipped (training fold contains a single class)")

    if optimizer_name == "adam":
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )
    else:
        raise ValueError(
            f"optimizer_name must be 'adam' or 'adamw', got {optimizer_name!r}"
        )

    if scheduler_name == "none":
        scheduler: torch.optim.lr_scheduler.LRScheduler | torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
    elif scheduler_name == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(1, epochs),
        )
    elif scheduler_name == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max" if val_loader is not None else "min",
            factor=0.5,
            min_lr=1e-5,
        )
    else:
        raise ValueError(
            f"scheduler_name must be one of ['none', 'cosine', 'plateau'], got {scheduler_name!r}"
        )

    best_state_dict = deepcopy(model.state_dict())
    best_threshold = 0.5
    best_epoch = 0

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
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)
            preds = _predict_from_probabilities(
                torch.softmax(logits, dim=1)[:, 1],
                decision_threshold=0.5,
            )
            correct += (preds == yb).sum().item()

        avg_loss = epoch_loss / dataset_size
        train_acc = correct / dataset_size
        val_result: EvalResult | None = None

        if val_loader is not None:
            val_probabilities, val_targets, val_loss = _collect_probabilities(
                model,
                val_loader,
                device,
                criterion=criterion,
            )
            tuned_val_result = _select_decision_threshold(
                val_probabilities,
                val_targets,
                min_precision=threshold_min_precision,
                min_recall=threshold_min_recall,
            )
            val_result = EvalResult(
                metrics=tuned_val_result.metrics,
                confusion_matrix=tuned_val_result.confusion_matrix,
                loss=val_loss,
                decision_threshold=tuned_val_result.decision_threshold,
            )
            monitor_value = _score_metrics(val_result.metrics)
        else:
            monitor_value = avg_loss

        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            scheduler.step(monitor_value)
        elif scheduler is not None:
            scheduler.step()

        is_best = early_stopping.step(monitor_value)
        if is_best:
            best_state_dict = deepcopy(model.state_dict())
            if val_result is not None:
                best_threshold = val_result.decision_threshold
            best_epoch = epoch + 1

        if (epoch + 1) % log_every == 0:
            elapsed = time() - t_start
            best_mark = "  <- best" if is_best else ""
            monitor_text = (
                f" | Val Loss: {val_result.loss:.4f}"
                f" | Val Acc: {val_result.metrics.accuracy * 100:5.1f}%"
                f" | Precision: {val_result.metrics.precision * 100:5.1f}%"
                f" | Recall: {val_result.metrics.recall * 100:5.1f}%"
                f" | F1: {val_result.metrics.f1 * 100:5.1f}%"
                f" | Thr: {val_result.decision_threshold:.2f}"
                if val_result is not None and val_result.loss is not None
                else ""
            )
            best_metric_text = (
                f" | Best Score: {early_stopping.best:.4f}"
                if val_result is not None
                else f" | Best Loss: {early_stopping.best:.4f}"
            )
            print(
                f"  Epoch {epoch + 1:{epoch_w}d}/{epochs}"
                f" | Loss: {avg_loss:.4f}"
                f" | Train Acc: {train_acc * 100:5.1f}%"
                f"{monitor_text}"
                f"{best_metric_text}"
                f" | LR: {optimizer.param_groups[0]['lr']:.2e}"
                f" | Elapsed: {elapsed:6.1f}s"
                f"{best_mark}"
            )



    return TrainResult(
        state_dict=best_state_dict,
        decision_threshold=best_threshold,
        best_epoch=best_epoch,
        best_score=early_stopping.best,
    )


def evaluate_model(
    model: ACRNN,
    test_loader: DataLoader,
    device: torch.device,
    decision_threshold: float = 0.5,
    criterion: nn.Module | None = None,
) -> EvalResult:
    positive_probabilities, targets, loss = _collect_probabilities(
        model,
        test_loader,
        device,
        criterion=criterion,
    )
    return _result_from_probabilities(
        positive_probabilities,
        targets,
        decision_threshold=decision_threshold,
        loss=loss,
    )


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
    batch_size: int = 16,
    num_workers: int = 0,
    log_every: int = 1,
    min_epochs: int = 20,
    learning_rate: float = 2e-4,
    weight_decay: float = 1e-2,
    optimizer_name: str = "adamw",
    scheduler_name: str = "plateau",
    grad_clip_norm: float = 1.0,
    validation_split: float = 0.1,
    normalization: str = "channel",
    threshold_min_precision: float = 0.65,
    threshold_min_recall: float = 0.65,
    class_weighting: str = "balanced",
    seed: int = 42,
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

    all_run_metrics = _make_metric_store()
    subject_scores: dict[int, dict[str, list[float]]] = {}
    best_run: tuple[int | None, int, EvalMetrics, dict[str, torch.Tensor], float] | None = None
    overall_confusion_matrix = np.zeros((2, 2), dtype=np.int64)

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
            validation_split=validation_split,
            normalization=normalization,
            seed=seed + fold + (0 if run_subject_id is None else run_subject_id * 100),
        )
        num_channels, num_timepoints = _infer_input_shape(dl.train)

        train_size = len(dl.train.dataset)  # type: ignore[arg-type]
        val_size = len(dl.val.dataset) if dl.val is not None else 0  # type: ignore[arg-type]
        print(f"  Train samples: {train_size}")
        if dl.val is not None:
            print(f"  Val samples   : {val_size}")
        print(f"  Input shape : {num_channels} x {num_timepoints}")

        model = ACRNN(
            reduce=2,
            k=40,
            num_channels=num_channels,
            num_timepoints=num_timepoints,
        ).to(training_device)

        train_result = train_model(
            model,
            dl.train,
            dl.val,
            training_device,
            epochs=epochs,
            log_every=log_every,
            min_epochs=min_epochs,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            optimizer_name=optimizer_name,
            scheduler_name=scheduler_name,
            grad_clip_norm=grad_clip_norm,
            threshold_min_precision=threshold_min_precision,
            threshold_min_recall=threshold_min_recall,
            class_weighting=class_weighting,
        )
        model.load_state_dict(train_result.state_dict)

        assert dl.test is not None
        test_size = len(dl.test.dataset)  # type: ignore[arg-type]
        eval_result = evaluate_model(
            model,
            dl.test,
            training_device,
            decision_threshold=train_result.decision_threshold,
        )
        metrics = eval_result.metrics

        print(f"\n  Test samples : {test_size}")
        print(f"  Best epoch   : {train_result.best_epoch}")
        print(f"  Decision thr : {train_result.decision_threshold:.2f}")
        print(f"  Accuracy     : {metrics.accuracy * 100:.2f}%")
        print(f"  Precision    : {metrics.precision * 100:.2f}%")
        print(f"  Recall       : {metrics.recall * 100:.2f}%")
        print(f"  F1 Score     : {metrics.f1 * 100:.2f}%")
        print(f"  Confusion MX : {eval_result.confusion_matrix.tolist()}")

        for metric_name, metric_value in metrics.as_dict().items():
            all_run_metrics[metric_name].append(metric_value)
        overall_confusion_matrix += eval_result.confusion_matrix
        test_subject_id = _resolve_test_subject_id(config, mode, run_subject_id, fold)
        subject_metric_store = subject_scores.setdefault(test_subject_id, _make_metric_store())
        for metric_name, metric_value in metrics.as_dict().items():
            subject_metric_store[metric_name].append(metric_value)

        if best_run is None or metrics.accuracy > best_run[2].accuracy:
            best_run = (
                run_subject_id,
                fold,
                metrics,
                train_result.state_dict,
                train_result.decision_threshold,
            )

    if subject_scores and (mode == "subject_independent" or subject_id is None):
        print(f"\n{'-' * 60}")
        print("  Per-subject summary")
        print(f"{'-' * 60}")
        for sid in sorted(subject_scores):
            summary = _summarise_metric_store(subject_scores[sid])
            print(
                f"  Subject {sid:02d}: "
                f"Acc {summary['accuracy']['mean'] * 100:.2f}% +- {summary['accuracy']['std'] * 100:.2f}%"
                f" | Prec {summary['precision']['mean'] * 100:.2f}%"
                f" | Rec {summary['recall']['mean'] * 100:.2f}%"
                f" | F1 {summary['f1']['mean'] * 100:.2f}%"
            )

    overall_metrics = _summarise_metric_store(all_run_metrics)
    timestamp_label = make_timestamp_label()
    output_dir = (
        None
        if save_dir is None
        else Path(save_dir) / mode / dataset / target / timestamp_label
    )

    print(f"\n{'=' * 60}")
    print(f"  Overall result  |  dataset: {dataset}  |  target: {target}")
    print(
        f"  Accuracy : {overall_metrics['accuracy']['mean'] * 100:.2f}% +- {overall_metrics['accuracy']['std'] * 100:.2f}%"
    )
    print(
        f"  Precision: {overall_metrics['precision']['mean'] * 100:.2f}% +- {overall_metrics['precision']['std'] * 100:.2f}%"
    )
    print(
        f"  Recall   : {overall_metrics['recall']['mean'] * 100:.2f}% +- {overall_metrics['recall']['std'] * 100:.2f}%"
    )
    print(
        f"  F1 Score : {overall_metrics['f1']['mean'] * 100:.2f}% +- {overall_metrics['f1']['std'] * 100:.2f}%"
    )
    print(f"  Confusion : {overall_confusion_matrix.tolist()}")
    print(f"{'=' * 60}")

    _save_subject_metric_plots(
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
        overall_metrics=overall_metrics,
        confusion_matrix=overall_confusion_matrix,
    )
    _save_confusion_matrix_plot(
        output_dir=output_dir,
        dataset=dataset,
        target=target,
        mode=mode,
        confusion_matrix=overall_confusion_matrix,
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

    return overall_metrics["accuracy"]["mean"], overall_metrics["accuracy"]["std"]
