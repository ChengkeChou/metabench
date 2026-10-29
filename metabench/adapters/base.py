"""Adapters: what a run is allowed to know about the software under test.

An adapter may only call public entry points.  If a metric needs a quantity the
public API does not expose, that is a finding about the API, not a reason to
reach inside.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Protocol


@dataclass
class FitRecord:
    """What one replicate produced, in the terms the metrics speak.

    ``failure`` is set when the fit did not produce a usable answer.  The
    record is still written: a replicate that failed is evidence about the
    method's reliability, and dropping it silently is the one thing the metrics
    exist to prevent.
    """

    estimate: float | None = None
    se: float | None = None
    ci_lb: float | None = None
    ci_ub: float | None = None
    pi_lb: float | None = None
    pi_ub: float | None = None
    pi_t_lb: float | None = None
    pi_t_ub: float | None = None
    tau2: float | None = None
    converged: bool | None = None
    boundary_hit: bool | None = None
    valid_covariance: bool | None = None
    warnings: int = 0
    failure: str | None = None
    #: a joint region, recorded as `(x - center)' cov^-1 (x - center) <= crit`
    #: so that containment is decided by the metrics and not by the adapter
    region_center: list[float] | None = None
    region_cov: list[list[float]] | None = None
    region_crit: float | None = None
    marginal_lb: list[float] | None = None
    marginal_ub: list[float] | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class Adapter(Protocol):
    """One implementation, driven through its public API."""

    name: str
    version: str

    def fit(self, data, *, level: float) -> FitRecord:
        ...
