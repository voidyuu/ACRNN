import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from acrnn.trainer import cross_validate_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ACRNN on the DREAMER EEG dataset")
    parser.add_argument(
        "--target",
        choices=["valence", "arousal"],
        help="Emotion dimension to classify",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=500,
        help="Training epochs per fold (default: 500)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size (default: 32)",
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    print(f"Target   : {args.target}")
    print(f"Epochs   : {args.epochs}")
    print(f"Batch    : {args.batch_size}")
    print(f"Workers  : {args.num_workers}")
    print(f"Device   : {args.device or 'auto'}")
    print()

    mean, std = cross_validate_model(
        target=args.target,
        device=args.device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print(f"\nFinal result — {args.target}: {mean:.4f} ± {std:.4f}")


if __name__ == "__main__":
    main()
