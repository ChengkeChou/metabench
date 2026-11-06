"""The univariate family: one effect per study, one heterogeneity variance.

Scenarios mirror the parameters `pimeta` records in
``tests/validation/baseline/*.json`` so that a run here can be compared with
the numbers already published, and they are written down in ADEMP terms so a
second implementation can reproduce them without reading `pimeta` at all.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import Scenario


@dataclass(frozen=True)
class UnivariateData:
    """One replicated data set, plus the estimands of that replication."""

    yi: np.ndarray
    vi: np.ndarray
    #: the true mean effect, the estimand of a confidence interval
    mu: float
    #: the true effect of one new study, the estimand of a prediction interval
    theta_new: float


def simulate(dgm: dict[str, Any], rng: np.random.Generator) -> UnivariateData:
    """Draw one data set from the mechanism ``dgm``.

    ``theta_i ~ N(mu, tau2)``, ``yi ~ N(theta_i, se_i^2)`` with ``se_i`` fixed
    by ``se`` and ``se_kind``.  The new study's true effect is drawn from the
    same between-study distribution, which is the estimand a prediction
    interval is supposed to cover.
    """
    k = int(dgm["k"])
    mu = float(dgm["mu"])
    tau2 = float(dgm["tau2"])
    se = dgm["se"]
    se_kind = dgm.get("se_kind", "equal")

    if se_kind == "equal":
        se_i = np.full(k, float(se))
    elif se_kind == "unequal":
        base = float(se)
        se_i = base * np.linspace(0.5, 1.5, k)
    elif se_kind == "lognormal":
        se_i = float(se) * np.exp(rng.normal(0.0, 0.3, k))
    else:                                                    # pragma: no cover
        raise ValueError(f"unknown se_kind: {se_kind!r}")

    theta = rng.normal(mu, np.sqrt(tau2), k)
    yi = rng.normal(theta, se_i)
    vi = se_i ** 2

    outlier = dgm.get("outlier")
    if outlier:
        # one study replaced by a value `shift` standard errors away, the
        # sensitivity scenario `pimeta.validation` records
        index = int(outlier["index"])
        yi[index] = mu + float(outlier["shift"]) * se_i[index]

    theta_new = float(rng.normal(mu, np.sqrt(tau2)))
    return UnivariateData(yi=yi, vi=vi, mu=mu, theta_new=theta_new)


def scenario(scenario_id: str, *, k: int, mu: float, tau2: float, se: float,
             se_kind: str = "equal", outlier: dict | None = None,
             notes: str = "") -> Scenario:
    """Build a scenario record from the DGM parameters."""
    dgm = {"k": k, "mu": mu, "tau2": tau2, "se": se, "se_kind": se_kind}
    if outlier:
        dgm["outlier"] = outlier
    return Scenario(
        scenario_id=scenario_id,
        family="univariate",
        dgm=dgm,
        estimand={"target": "mu", "value": mu,
                  "note": "the mean effect; a prediction interval is checked "
                          "against the true effect of one new study instead"},
        notes=notes,
    )


#: The six cells `pimeta.validation` already measures, with the same numbers,
#: so a run here reproduces a published baseline before it is trusted.
SCENARIOS: dict[str, Scenario] = {
    "univariate/null": scenario(
        "univariate/null", k=20, mu=0.0, tau2=0.04, se=0.2,
        notes="size: the mean effect is truly zero, mild heterogeneity"),
    "univariate/tiny": scenario(
        "univariate/tiny", k=5, mu=0.2, tau2=0.04, se=0.2,
        notes="k=5, where every variance estimator is unstable"),
    "univariate/alt": scenario(
        "univariate/alt", k=20, mu=0.3, tau2=0.04, se=0.2,
        notes="power: a real effect of 0.3.  REML is location-equivariant, so "
              "this cell must reproduce univariate/null exactly -- a useful "
              "check on the harness itself"),
    "univariate/unequal": scenario(
        "univariate/unequal", k=20, mu=0.2, tau2=0.04, se=0.2,
        se_kind="unequal", notes="within-study precision varies fourfold"),
    "univariate/outlier": scenario(
        "univariate/outlier", k=20, mu=0.2, tau2=0.04, se=0.2,
        outlier={"index": 0, "shift": 6.0},
        notes="one study six standard errors away from the mean"),
    "univariate/fixed": scenario(
        "univariate/fixed", k=20, mu=0.2, tau2=0.0, se=0.2,
        notes="no heterogeneity: tau2 on the boundary"),
}


#: The prediction-interval sweep.  A prediction interval covers the *true
#: effect of one new study*, which is a different estimand from the mean, and
#: the classical Higgins-Thompson-Spiegelhalter interval charges the t
#: distribution with ``k - 1`` degrees of freedom for the sampling of tau^2.
#: The deficit of a joint region was ``O(1/k)``; this sweep measures whether a
#: single interval has the same signature, and where it becomes nominal.
#:
#: ``k`` runs from 5 (the smallest size a meta-analysis is ever fitted at) to
#: 100; ``tau2`` covers a mild (I^2 ~ 50 % at se = 0.2) and a strong
#: heterogeneity case, because the interval's width is dominated by tau^2 and a
#: boundary estimate is the failure mode at small ``k``.
PI_CELLS: dict[str, dict[str, float]] = {
    "k05-tau004": {"k": 5, "tau2": 0.04},
    "k05-tau013": {"k": 5, "tau2": 0.13},
    "k10-tau004": {"k": 10, "tau2": 0.04},
    "k10-tau013": {"k": 10, "tau2": 0.13},
    "k20-tau004": {"k": 20, "tau2": 0.04},
    "k20-tau013": {"k": 20, "tau2": 0.13},
    "k50-tau004": {"k": 50, "tau2": 0.04},
    "k50-tau013": {"k": 50, "tau2": 0.13},
    "k100-tau004": {"k": 100, "tau2": 0.04},
    "k100-tau013": {"k": 100, "tau2": 0.13},
}

for _cell, _spec in PI_CELLS.items():
    SCENARIOS[f"univariate-pi/{_cell}"] = scenario(
        f"univariate-pi/{_cell}", k=int(_spec["k"]), mu=0.2,
        tau2=float(_spec["tau2"]), se=0.2,
        notes=f"prediction-interval sweep: k={int(_spec['k'])}, "
              f"tau2={_spec['tau2']}, equal within-study se")
