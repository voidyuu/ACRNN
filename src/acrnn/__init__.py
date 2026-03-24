from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .model import ACRNN


def __getattr__(name: str):
    if name == "ACRNN":
        from .model import ACRNN

        return ACRNN
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = ["ACRNN"]
