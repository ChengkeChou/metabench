"""Scenario registry."""

from __future__ import annotations

from .base import Scenario, fingerprint
from .covariance import SCENARIOS as _COVARIANCE
from .covariance import CovarianceData, simulate as simulate_covariance
from .multivariate import SCENARIOS as _MULTIVARIATE
from .multivariate import MultivariateData, simulate as simulate_multivariate
from .univariate import SCENARIOS as _UNIVARIATE
from .univariate import UnivariateData, simulate as simulate_univariate

ALL: dict[str, Scenario] = {**_UNIVARIATE, **_MULTIVARIATE, **_COVARIANCE}


def get(scenario_id: str) -> Scenario:
    try:
        return ALL[scenario_id]
    except KeyError:
        raise KeyError(
            f"unknown scenario {scenario_id!r}; known: {sorted(ALL)}") from None


def simulate(scenario: Scenario, rng):
    """Dispatch to the family's simulator."""
    if scenario.family == "univariate":
        return simulate_univariate(scenario.dgm, rng)
    if scenario.family == "multivariate":
        return simulate_multivariate(scenario.dgm, rng)
    if scenario.family == "covariance":
        return simulate_covariance(scenario.dgm, rng)
    raise ValueError(f"no simulator for family {scenario.family!r}")


__all__ = ["ALL", "CovarianceData", "MultivariateData", "Scenario",
           "UnivariateData", "fingerprint", "get", "simulate"]
