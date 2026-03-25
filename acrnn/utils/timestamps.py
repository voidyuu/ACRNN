from __future__ import annotations

from datetime import datetime

DEFAULT_TIMESTAMP_FORMAT = "%Y-%m-%d_%H-%M-%S"


def make_timestamp_label(fmt: str = DEFAULT_TIMESTAMP_FORMAT) -> str:
    """Return a filesystem-friendly timestamp label for saved artifacts."""
    return datetime.now().strftime(fmt)
