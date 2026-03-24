import argparse


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train and evaluate ACRNN.")
    parser.add_argument("target", choices=["arousal", "valence"], help="Prediction target.")
    parser.add_argument(
        "--data-dir",
        default="../../data/data_preprocessed_ACRNN",
        help="Directory containing subject .pth files.",
    )
    parser.add_argument("--device", default=None, help="Torch device, for example cuda or cpu.")
    parser.add_argument("--epochs", type=int, default=500, help="Training epochs per fold.")
    parser.add_argument("--batch-size", type=int, default=16, help="Batch size.")
    parser.add_argument("--n-splits", type=int, default=10, help="Number of CV folds.")
    parser.add_argument("--num-subjects", type=int, default=32, help="Number of subjects to evaluate.")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    from .training import cross_validate_model

    cross_validate_model(
        target=args.target,
        data_dir=args.data_dir,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        n_splits=args.n_splits,
        num_subjects=args.num_subjects,
    )
