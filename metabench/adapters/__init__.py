"""Adapter registry."""

from __future__ import annotations

from .base import Adapter, FitRecord
from .pimeta_adapter import PimetaAdapter

ALL: dict[str, type] = {
    "pimeta": PimetaAdapter,
}


def get(name: str) -> Adapter:
    try:
        return ALL[name]()
    except KeyError:
        raise KeyError(
            f"unknown adapter {name!r}; known: {sorted(ALL)}") from None


__all__ = ["ALL", "Adapter", "FitRecord", "get"]
