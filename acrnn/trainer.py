from copy import deepcopy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .data import DreamerDataloader, VALID_FOLDS
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
) -> dict[str, torch.Tensor]:
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    best_loss = float("inf")
    best_state_dict = deepcopy(model.state_dict())

    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0

        for xb, yb in train_loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            output = model(xb)
            loss = criterion(output, yb)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * xb.size(0)

        avg_epoch_loss = epoch_loss / len(train_loader.dataset)  # type: ignore[arg-type]
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss:.4f}")
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            best_state_dict = deepcopy(model.state_dict())

    return best_state_dict


def evaluate_model(model: ACRNN, test_loader: DataLoader, device: torch.device) -> float:
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
) -> tuple[float, float]:
    if target not in VALID_TARGETS:
        raise ValueError(f"target must be one of {VALID_TARGETS}, got {target!r}")

    training_device = resolve_device(device)
    all_fold_acc = []
    num_folds = len(VALID_FOLDS)

    for fold in VALID_FOLDS:
        print(f"\n========== Fold {fold + 1}/{num_folds} ({target}) ==========")

        dl = DreamerDataloader(
            target,
            fold=fold,
            batch_size=batch_size,
            num_workers=num_workers,
        )

        model = ACRNN(
            reduce=2,
            k=40,
            num_channels=_DREAMER_CHANNELS,
            num_timepoints=_DREAMER_TIMEPOINTS,
        ).to(training_device)

        best_state = train_model(model, dl.train, training_device, epochs=epochs)
        model.load_state_dict(best_state)

        assert dl.test is not None
        acc = evaluate_model(model, dl.test, training_device)
        print(f"Fold {fold + 1} accuracy: {acc:.4f}")
        print("=" * 50)
        all_fold_acc.append(acc)

    overall_mean = float(np.mean(all_fold_acc))
    overall_std = float(np.std(all_fold_acc))
    print(f"\n=== {num_folds}-Fold CV Accuracy ({target}): {overall_mean:.4f} ± {overall_std:.4f} ===")
    return overall_mean, overall_std
