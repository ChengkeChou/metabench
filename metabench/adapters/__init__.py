"""Adapter registry."""

from __future__ import annotations

from .base import Adapter, FitRecord
from .pimeta_adapter import PimetaAdapter
from .pimeta_cov_adapter import VARIANTS as COV_VARIANTS
from .pimeta_cov_adapter import PimetaCovAdapter
from .pimeta_region_adapter import PimetaRegionAdapter

ALL: dict[str, type] = {
    "pimeta": PimetaAdapter,
    "pimeta-region/chi2": PimetaRegionAdapter,
    "pimeta-region/f": PimetaRegionAdapter,
    "pimeta-region/hk": PimetaRegionAdapter,
    **{f"pimeta-cov/{v}": PimetaCovAdapter for v in COV_VARIANTS},
    "pimeta-cov/hyper-lkj2": PimetaCovAdapter,
    "pimeta-cov/hyper-pinned": PimetaCovAdapter,
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
    if name.startswith("pimeta-cov/"):
        rest = name.split("/", 1)[1]
        if rest.endswith("-pinned"):
            return PimetaCovAdapter(variant=rest[:-len("-pinned")],
                                    pinned=True)
        if "-lkj" in rest:
            variant, eta = rest.split("-lkj", 1)
            return PimetaCovAdapter(variant=variant, lkj_eta=float(eta))
        return PimetaCovAdapter(variant=rest)
    raise KeyError(f"unknown adapter {name!r}; known: {sorted(ALL)}")


__all__ = ["ALL", "Adapter", "FitRecord", "get"]
