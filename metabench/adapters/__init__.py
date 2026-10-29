"""Adapter registry."""

from __future__ import annotations

from .base import Adapter, FitRecord
from .pimeta_adapter import PimetaAdapter
from .pimeta_region_adapter import PimetaRegionAdapter

ALL: dict[str, type] = {
    "pimeta": PimetaAdapter,
    "pimeta-region/chi2": PimetaRegionAdapter,
    "pimeta-region/f": PimetaRegionAdapter,
    "pimeta-region/hk": PimetaRegionAdapter,
}


class _Bound:
    """A registry entry plus the argument it was registered with."""

    def __init__(self, factory, **kwargs):
        self._factory = factory
        self._kwargs = kwargs

    def __call__(self):
        return self._factory(**self._kwargs)


def get(name: str) -> Adapter:
    if name == "pimeta":
        return PimetaAdapter()
    if name.startswith("pimeta-region/"):
        return PimetaRegionAdapter(method=name.split("/", 1)[1])
    raise KeyError(f"unknown adapter {name!r}; known: {sorted(ALL)}")


__all__ = ["ALL", "Adapter", "FitRecord", "get"]
