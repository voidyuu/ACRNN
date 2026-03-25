from __future__ import annotations

from pathlib import Path

PROJECT_ROOT: Path = Path(__file__).resolve().parent.parent
DATA_DIR: Path = PROJECT_ROOT / "data"
DEFAULT_SAVE_DIR: Path = PROJECT_ROOT / "checkpoints"

VALID_DATASETS: tuple[str, ...] = ("deap", "dreamer")
DEAP_TARGETS: tuple[str, ...] = ("valence", "arousal", "dominance", "liking")
DREAMER_TARGETS: tuple[str, ...] = ("valence", "arousal", "dominance")

DEAP_DATA_DIR: Path = DATA_DIR / "deap"
DEAP_CACHE_DIR: Path = DEAP_DATA_DIR / "cache"

DREAMER_DATA_DIR: Path = DATA_DIR / "dreamer"
DREAMER_MAT_PATH: Path = DREAMER_DATA_DIR / "DREAMER.mat"
DREAMER_CACHE_DIR: Path = DREAMER_DATA_DIR / "cache"


def get_default_cache_dir(dataset: str) -> Path:
    if dataset == "deap":
        return DEAP_CACHE_DIR
    if dataset == "dreamer":
        return DREAMER_CACHE_DIR
    raise ValueError(f"Unsupported dataset: {dataset!r}")


def get_valid_targets(dataset: str) -> tuple[str, ...]:
    if dataset == "deap":
        return DEAP_TARGETS
    if dataset == "dreamer":
        return DREAMER_TARGETS
    raise ValueError(f"Unsupported dataset: {dataset!r}")
