from __future__ import annotations

import argparse

from .config import (
    DEAP_TARGETS,
    DEFAULT_SAVE_DIR,
    DREAMER_TARGETS,
    VALID_DATASETS,
    get_default_threshold,
)
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
        help="Binary-label threshold; defaults to the configured dataset/target-specific value",
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
        default=16,
        help="Batch size (default: 16)",
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
        default=5,
        help="Print metrics every N epochs (default: 5)",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience in epochs (default: 15)",
    )
    parser.add_argument(
        "--min-epochs",
        type=int,
        default=20,
        help="Minimum epochs to run before early stopping can trigger (default: 20)",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=2e-4,
        help="Optimizer learning rate (default: 2e-4)",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-2,
        help="Weight decay used by AdamW (default: 1e-2)",
    )
    parser.add_argument(
        "--optimizer",
        choices=["adam", "adamw"],
        default="adamw",
        help="Optimizer to use (default: adamw)",
    )
    parser.add_argument(
        "--scheduler",
        choices=["none", "cosine", "plateau"],
        default="plateau",
        help="Learning-rate scheduler (default: plateau)",
    )
    parser.add_argument(
        "--grad-clip",
        type=float,
        default=1.0,
        help="Gradient clipping norm; 0 disables clipping (default: 1.0)",
    )
    parser.add_argument(
        "--validation-split",
        type=float,
        default=0.1,
        help="Fraction of the training fold held out for validation and threshold tuning (default: 0.1)",
    )
    parser.add_argument(
        "--normalization",
        choices=["none", "channel"],
        default="channel",
        help="Input normalization strategy (default: channel)",
    )
    parser.add_argument(
        "--threshold-min-precision",
        type=float,
        default=0.65,
        help="Minimum validation precision when scanning decision thresholds (default: 0.65)",
    )
    parser.add_argument(
        "--threshold-min-recall",
        type=float,
        default=0.65,
        help="Minimum validation recall when scanning decision thresholds (default: 0.65)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for validation splits (default: 42)",
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
    default_threshold = get_default_threshold(args.dataset, args.target)

    print(f"Dataset   : {args.dataset}")
    print(f"Target    : {args.target}")
    print(f"Mode      : {args.mode}")
    print(f"Subject   : {args.subject_id if args.subject_id is not None else 'all'}")
    print(f"Folds     : {args.n_folds}")
    print(
        f"Threshold : {args.threshold if args.threshold is not None else f'default ({default_threshold})'}"
    )
    print(f"Cache dir : {args.cache_dir or 'dataset default'}")
    print(f"Epochs    : {args.epochs}")
    print(f"Batch     : {args.batch_size}")
    print(f"Workers   : {args.num_workers}")
    print(f"Device    : {args.device or 'auto'}")
    print(f"Log every : {args.log_every} epochs")
    print(f"Patience  : {args.patience if args.patience > 0 else 'disabled'}")
    print(f"Min epochs: {args.min_epochs}")
    print(f"LR        : {args.learning_rate}")
    print(f"W decay   : {args.weight_decay}")
    print(f"Optim     : {args.optimizer}")
    print(f"Sched     : {args.scheduler}")
    print(f"Grad clip : {args.grad_clip}")
    print(f"Val split : {args.validation_split}")
    print(f"Norm      : {args.normalization}")
    print(f"Thr floor : prec>={args.threshold_min_precision}, rec>={args.threshold_min_recall}")
    print(f"Seed      : {args.seed}")
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
        min_epochs=args.min_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        optimizer_name=args.optimizer,
        scheduler_name=args.scheduler,
        grad_clip_norm=args.grad_clip,
        validation_split=args.validation_split,
        normalization=args.normalization,
        threshold_min_precision=args.threshold_min_precision,
        threshold_min_recall=args.threshold_min_recall,
        seed=args.seed,
        save_dir=args.save_dir or None,
    )

    print(f"\nFinal result - {args.dataset}/{args.target}: {mean:.4f} +- {std:.4f}")
