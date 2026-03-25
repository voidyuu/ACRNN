from .deap_loader import (
    VALID_FOLDS_DEPENDENT as DEAP_VALID_FOLDS_DEPENDENT,
)
from .deap_loader import (
    VALID_FOLDS_INDEPENDENT as DEAP_VALID_FOLDS_INDEPENDENT,
)
from .deap_loader import (
    VALID_SUBJECTS as DEAP_VALID_SUBJECTS,
)
from .deap_loader import (
    VALID_TARGETS as DEAP_VALID_TARGETS,
)
from .deap_loader import (
    DeapDataloader,
    load_deap_arrays,
)
from .dreamer_loader import (
    VALID_FOLDS,
    DreamerDataloader,
    load_dreamer_arrays,
)
from .dreamer_loader import (
    VALID_FOLDS_DEPENDENT as DREAMER_VALID_FOLDS_DEPENDENT,
)
from .dreamer_loader import (
    VALID_FOLDS_INDEPENDENT as DREAMER_VALID_FOLDS_INDEPENDENT,
)
from .dreamer_loader import (
    VALID_SUBJECTS as DREAMER_VALID_SUBJECTS,
)
from .dreamer_loader import (
    VALID_TARGETS as DREAMER_VALID_TARGETS,
)
from .loaders import ArrayDataset, LoaderBundle, build_dataloaders
from .split import DataSplit, build_index_split, build_kfold_splits

__all__ = [
    # ── loaders ──────────────────────────────────────────────────────────────
    "ArrayDataset",
    "LoaderBundle",
    "build_dataloaders",
    # ── split ─────────────────────────────────────────────────────────────────
    "DataSplit",
    "build_index_split",
    "build_kfold_splits",
    # ── DEAP ──────────────────────────────────────────────────────────────────
    "DeapDataloader",
    "DEAP_VALID_TARGETS",
    "DEAP_VALID_SUBJECTS",
    "DEAP_VALID_FOLDS_INDEPENDENT",
    "DEAP_VALID_FOLDS_DEPENDENT",
    "load_deap_arrays",
    # ── DREAMER ───────────────────────────────────────────────────────────────
    "DreamerDataloader",
    "DREAMER_VALID_TARGETS",
    "DREAMER_VALID_SUBJECTS",
    "DREAMER_VALID_FOLDS_INDEPENDENT",
    "DREAMER_VALID_FOLDS_DEPENDENT",
    "VALID_FOLDS",  # consumed by trainer.cross_validate_model
    "load_dreamer_arrays",
]
