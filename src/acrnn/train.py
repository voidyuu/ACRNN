
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch
from sklearn.model_selection import KFold
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from utils import resolve_device
from .model import ACRNN

LABEL_INDEX = {
    "valence": 0,
    "arousal": 1,
}

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

        avg_epoch_loss = epoch_loss / len(train_loader.dataset)
        if (epoch + 1) % 100 == 0:
            print(f"Epoch {epoch + 1} average loss: {avg_epoch_loss}")
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


def load_subject_data(data_dir: Path, subject_id: int, target: str) -> tuple[torch.Tensor, torch.Tensor]:
    subject_path = data_dir / f"s{subject_id:02d}.pth"
    data = torch.load(subject_path)
    inputs = data["x"].squeeze(1)
    labels = data["y"][:, LABEL_INDEX[target]]
    return inputs, labels


def cross_validate_model(
    target: str,
    data_dir: str = "../../data/data_preprocessed_ACRNN",
    device: str | None = None,
    epochs: int = 500,
    batch_size: int = 16,
    n_splits: int = 10,
    num_subjects: int = 32,
) -> tuple[float, float]:
    if target not in LABEL_INDEX:
        raise ValueError(f"Unsupported target: {target}")

    training_device = resolve_device(device)
    data_path = Path(data_dir)
    all_subject_acc = []

    for subject_id in range(1, num_subjects + 1):
        print(f"\n=== Subject {subject_id} ({target}) ===")
        X_subset, y_subset = load_subject_data(data_path, subject_id, target)

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        all_acc = []

        for fold, (train_val_idx, test_idx) in enumerate(kf.split(X_subset), start=1):
            print(f"\n========== Fold {fold}/{n_splits} ==========")

            X_train_val, y_train_val = X_subset[train_val_idx], y_subset[train_val_idx]
            X_test, y_test = X_subset[test_idx], y_subset[test_idx]

            train_ds = TensorDataset(X_train_val, y_train_val)
            test_ds = TensorDataset(X_test, y_test)

            train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(test_ds, batch_size=batch_size)

            model = ACRNN(reduce=2, k=40).to(training_device)
            best_model = train_model(model, train_loader, training_device, epochs=epochs)

            model.load_state_dict(best_model)
            acc = evaluate_model(model, test_loader, training_device)

            print(f"Fold {fold} - Accuracy: {acc:.4f}")
            print("=======================================\n")
            all_acc.append(acc)

        mean_acc = np.mean(all_acc)
        std_acc = np.std(all_acc)
        all_subject_acc.append(mean_acc)
        print(f"\nFinal {n_splits}-Fold Accuracy: {mean_acc:.4f} ± {std_acc:.4f}")

    overall_mean = np.mean(all_subject_acc)
    overall_std = np.std(all_subject_acc)
    print(f"\n=== Overall Subject-Dependent Accuracy ({target}): {overall_mean:.4f} ± {overall_std:.4f} ===")
    return float(overall_mean), float(overall_std)
