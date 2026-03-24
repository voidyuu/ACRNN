from copy import deepcopy
from datetime import datetime
from pathlib import Path
from time import time

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import VALID_FOLDS, DreamerDataloader
from .model import ACRNN
from .utils import resolve_device

# DREAMER EEG: 14 channels × 256 timepoints (2 s at 128 Hz)
_DREAMER_CHANNELS = 14
_DREAMER_TIMEPOINTS = 256

VALID_TARGETS = {"valence", "arousal"}


def train_model(
    model: ACRNN,
    train_loader: DataLoader,
    device: torch.device,
    epochs: int = 500,
    log_every: int = 10,
) -> dict[str, torch.Tensor]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_loss = float("inf")
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
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()

            epoch_loss += loss.item() * xb.size(0)
            preds = torch.argmax(output, dim=1)
            correct += (preds == yb).sum().item()

        avg_loss = epoch_loss / dataset_size
        train_acc = correct / dataset_size

        is_best = avg_loss < best_loss
        if is_best:
            best_loss = avg_loss
            best_state_dict = deepcopy(model.state_dict())

        if (epoch + 1) % log_every == 0:
            elapsed = time() - t_start
            best_mark = "  ← best" if is_best else ""
            print(
                f"  Epoch {epoch + 1:{epoch_w}d}/{epochs}"
                f" | Loss: {avg_loss:.4f}"
                f" | Train Acc: {train_acc * 100:5.1f}%"
                f" | Best Loss: {best_loss:.4f}"
                f" | Elapsed: {elapsed:6.1f}s"
                f"{best_mark}"
            )

    return best_state_dict


def evaluate_model(
    model: ACRNN, test_loader: DataLoader, device: torch.device
) -> float:
    model.eval()
    all_preds, all_targets = [], []

    with torch.no_grad():
        for xb, yb in test_loader:
            xb, yb = xb.to(device), yb.to(device)
            outputs = model(xb)
            preds = torch.argmax(outputs, dim=1)
            all_preds.append(preds)
            all_targets.append(yb)

    all_preds = torch.cat(all_preds)
    all_targets = torch.cat(all_targets)
    return (all_preds == all_targets).sum().item() / len(all_preds)


def cross_validate_model(
    target: str,
    device: str | None = None,
    epochs: int = 500,
    batch_size: int = 32,
    num_workers: int = 0,
    log_every: int = 10,
    save_dir: str | None = "checkpoints",
) -> tuple[float, float]:
    if target not in VALID_TARGETS:
        raise ValueError(f"target must be one of {VALID_TARGETS}, got {target!r}")

    training_device = resolve_device(device)
    all_fold_acc: list[float] = []
    fold_results: list[tuple[int, float, dict[str, torch.Tensor]]] = []
    num_folds = len(VALID_FOLDS)

    for fold in VALID_FOLDS:
        print(f"\n{'=' * 52}")
        print(
            f"  Fold {fold + 1}/{num_folds}  |  target: {target}  |  device: {training_device}"
        )
        print(f"{'=' * 52}")

        dl = DreamerDataloader(
            target,
            fold=fold,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        train_size = len(dl.train.dataset)  # type: ignore[arg-type]
        print(f"  Train samples: {train_size}")

        model = ACRNN(
            reduce=2,
            k=40,
            num_channels=_DREAMER_CHANNELS,
            num_timepoints=_DREAMER_TIMEPOINTS,
        ).to(training_device)

        best_state = train_model(
            model,
            dl.train,
            training_device,
            epochs=epochs,
            log_every=log_every,
        )
        model.load_state_dict(best_state)

        assert dl.test is not None
        test_size = len(dl.test.dataset)  # type: ignore[arg-type]
        acc = evaluate_model(model, dl.test, training_device)

        print(f"\n  Test samples : {test_size}")
        print(f"  Fold {fold + 1} Test Accuracy: {acc * 100:.2f}%")

        all_fold_acc.append(acc)
        fold_results.append((fold, acc, best_state))

    overall_mean = float(np.mean(all_fold_acc))
    overall_std = float(np.std(all_fold_acc))

    print(f"\n{'=' * 52}")
    print(f"  {num_folds}-Fold CV  |  {target}")
    print(f"  Accuracy: {overall_mean * 100:.2f}% ± {overall_std * 100:.2f}%")
    print(f"{'=' * 52}")

    if save_dir is not None:
        best_fold, best_acc, best_state = max(fold_results, key=lambda x: x[1])
        save_path = Path(save_dir) / target
        filename = (
            save_path
            / f"{datetime.fromtimestamp(time()).strftime('%Y-%m-%d %H:%M')}.pt"
        )

        torch.save(
            {
                "state_dict": best_state,
                "fold": best_fold,
                "test_acc": best_acc,
                "target": target,
                "epochs": epochs,
            },
            filename,
        )
        print(
            f"\n  Best weights (fold {best_fold + 1}, acc {best_acc * 100:.2f}%) saved → {filename}"
        )

    return overall_mean, overall_std
