from __future__ import annotations

import argparse

from .config import DEAP_TARGETS, DEFAULT_SAVE_DIR, DREAMER_TARGETS, VALID_DATASETS
from .trainer import cross_validate_model

_TARGET_CHOICES = sorted(set(DEAP_TARGETS) | set(DREAMER_TARGETS))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train ACRNN on the DEAP or DREAMER dataset with paper-aligned defaults"
    )
    parser.add_argument(
        "--dataset",
        choices=list(VALID_DATASETS),
        default="dreamer",
        help="Dataset to evaluate",
    )
    parser.add_argument(
        "--target",
        choices=_TARGET_CHOICES,
        required=True,
        help="Emotion dimension to classify",
    )
    parser.add_argument(
        "--mode",
        choices=["subject_dependent", "subject_independent"],
        default="subject_dependent",
        help="Evaluation protocol",
    )
    parser.add_argument(
        "--subject-id",
        type=int,
        default=None,
        help="Subject to evaluate in subject_dependent mode; omit to run all subjects",
    )
    parser.add_argument(
        "--n-folds",
        type=int,
        default=10,
        help="Number of folds for subject_dependent evaluation (default: 10)",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=None,
        help="Binary-label threshold; defaults to 5.0 for DEAP and 4.0 for DREAMER",
    )
    parser.add_argument(
        "--cache-dir",
        type=str,
        default=None,
        help="Override the default dataset cache directory",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=200,
        help="Training epochs per fold (default: 200)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size (default: 10)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=0,
        help="DataLoader worker processes (default: 0)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device override, e.g. 'cpu', 'cuda', 'mps' (default: auto-detect)",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=1,
        help="Print metrics every N epochs (default: 1)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=0,
        help="Early stopping patience in epochs; 0 disables it (default: 0)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=str(DEFAULT_SAVE_DIR),
        help="Directory for saving best weights under dataset/target/mode subfolders",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Dataset   : {args.dataset}")
    print(f"Target    : {args.target}")
    print(f"Mode      : {args.mode}")
    print(f"Subject   : {args.subject_id if args.subject_id is not None else 'all'}")
    print(f"Folds     : {args.n_folds}")
    print(f"Threshold : {args.threshold if args.threshold is not None else 'dataset default'}")
    print(f"Cache dir : {args.cache_dir or 'dataset default'}")
    print(f"Epochs    : {args.epochs}")
    print(f"Batch     : {args.batch_size}")
    print(f"Workers   : {args.num_workers}")
    print(f"Device    : {args.device or 'auto'}")
    print(f"Log every : {args.log_every} epochs")
    print(f"Patience  : {args.patience if args.patience > 0 else 'disabled'}")
    print(f"Save dir  : {args.save_dir or '(disabled)'}")
    print()

    mean, std = cross_validate_model(
        dataset=args.dataset,
        target=args.target,
        mode=args.mode,
        subject_id=args.subject_id,
        n_folds=args.n_folds,
        threshold=args.threshold,
        cache_dir=args.cache_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        log_every=args.log_every,
        patience=args.patience,
        save_dir=args.save_dir or None,
    )

    print(f"\nFinal result - {args.dataset}/{args.target}: {mean:.4f} +- {std:.4f}")
