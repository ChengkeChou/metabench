"""The multivariate family: two correlated outcomes per study.

The estimand of a joint prediction region is the *vector* of true effects of a
new study, and the estimand of a joint confidence region is the mean vector;
both are recorded per replicate, because a region that covers the mean says
nothing about whether it covers a new study.

The data are laid out study-major: rows ``i*J .. i*J+J-1`` are study ``i``'s
outcomes, the model matrix is ``~ outcome - 1`` (one column per outcome) and
the sampling covariance is block diagonal per study.  Correlated outcomes are
therefore carried by the between-study ``UN`` matrix, which is the object the
region is built from.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import Scenario


@dataclass(frozen=True)
class MultivariateData:
    yi: np.ndarray          # (k*J,)
    V: np.ndarray           # (k*J, k*J), block diagonal per study
    X: np.ndarray           # (k*J, J), one column per outcome
    frame: dict[str, list]  # the `data=` argument rma_mv needs
    mu: np.ndarray          # (J,) true mean vector
    theta_new: np.ndarray   # (J,) true effects of one new study


def simulate(dgm: dict[str, Any], rng: np.random.Generator) -> MultivariateData:
    """Draw one data set: ``theta_i ~ N(mu, D)``, ``yi ~ N(theta_i, V_i)``."""
    k = int(dgm["k"])
    J = int(dgm.get("J", 2))
    mu = np.asarray(dgm["mu"], dtype=float)
    tau2 = np.asarray(dgm["tau2"], dtype=float)
    rho = float(dgm.get("rho", 0.3))
    se = np.asarray(dgm["se"], dtype=float)
    if mu.size != J or tau2.size != J or se.size != J:
        raise ValueError("mu, tau2 and se must have J entries")

    sd = np.sqrt(tau2)
    R = np.eye(J)
    for i in range(J):
        for j in range(i + 1, J):
            R[i, j] = R[j, i] = rho
    D = np.outer(sd, sd) * R

    theta = rng.multivariate_normal(mu, D, k)
    yi = np.empty((k, J))
    V = np.zeros((k * J, k * J))
    for i in range(k):
        yi[i] = rng.normal(theta[i], se)
        V[i * J:(i + 1) * J, i * J:(i + 1) * J] = np.diag(se ** 2)

    theta_new = rng.multivariate_normal(mu, D)
    X = np.zeros((k * J, J))
    X[np.arange(k * J), np.tile(np.arange(J), k)] = 1.0
    frame = {
        "study": [f"s{i:03d}" for i in range(k) for _ in range(J)],
        "outcome": [f"o{j}" for _ in range(k) for j in range(J)],
    }
    return MultivariateData(yi=yi.ravel(), V=V, X=X, frame=frame, mu=mu,
                            theta_new=theta_new)


def scenario(scenario_id: str, *, k: int, tau2: tuple[float, ...],
             se: tuple[float, ...], rho: float, mu: tuple[float, ...] = (0.0, 0.0),
             notes: str = "") -> Scenario:
    return Scenario(
        scenario_id=scenario_id,
        family="multivariate",
        dgm={"k": k, "J": len(mu), "mu": list(mu), "tau2": list(tau2),
             "se": list(se), "rho": rho},
        estimand={"target": "theta_new",
                  "note": "the true effects of one new study; the joint "
                          "confidence region is checked against mu instead"},
        notes=notes,
    )


#: The first grid: study count against outcome correlation, with the same
#: heterogeneity the univariate cells use.
SCENARIOS: dict[str, Scenario] = {
    "multivariate/k05-rho30": scenario(
        "multivariate/k05-rho30", k=5, tau2=(0.04, 0.09), se=(0.20, 0.25),
        rho=0.3, notes="k=5: the region has to estimate a 2x2 UN matrix from "
                       "five studies"),
    "multivariate/k10-rho30": scenario(
        "multivariate/k10-rho30", k=10, tau2=(0.04, 0.09), se=(0.20, 0.25),
        rho=0.3, notes="the cell the chi-square reference is known to miss"),
    "multivariate/k10-rho70": scenario(
        "multivariate/k10-rho70", k=10, tau2=(0.04, 0.09), se=(0.20, 0.25),
        rho=0.7, notes="stronger correlation: the region is an elongated "
                       "ellipse and its area is very sensitive to rho"),
    "multivariate/k20-rho30": scenario(
        "multivariate/k20-rho30", k=20, tau2=(0.04, 0.09), se=(0.20, 0.25),
        rho=0.3),
    "multivariate/k20-rho70": scenario(
        "multivariate/k20-rho70", k=20, tau2=(0.04, 0.09), se=(0.20, 0.25),
        rho=0.7),
    "multivariate/k50-rho30": scenario(
        "multivariate/k50-rho30", k=50, tau2=(0.04, 0.09), se=(0.20, 0.25),
        rho=0.3, notes="the asymptotic reference should be adequate here; if "
                       "it is not, the problem is not the sample size"),
    # -- more outcomes: the covariance has J(J+1)/2 parameters, so J=3 needs 6
    # and J=5 needs 15 estimated from the same k studies.  These cells exist to
    # find out whether the small-k multiplier measured at J=2 carries over.
    "multivariate/k10-J3-rho30": scenario(
        "multivariate/k10-J3-rho30", k=10, tau2=(0.04, 0.06, 0.09),
        se=(0.20, 0.22, 0.25), rho=0.3, mu=(0.0, 0.0, 0.0),
        notes="J=3 at k=10: six variance parameters from ten studies"),
    "multivariate/k20-J3-rho30": scenario(
        "multivariate/k20-J3-rho30", k=20, tau2=(0.04, 0.06, 0.09),
        se=(0.20, 0.22, 0.25), rho=0.3, mu=(0.0, 0.0, 0.0)),
    "multivariate/k20-J5-rho30": scenario(
        "multivariate/k20-J5-rho30", k=20,
        tau2=(0.04, 0.05, 0.06, 0.075, 0.09),
        se=(0.20, 0.21, 0.22, 0.235, 0.25), rho=0.3,
        mu=(0.0, 0.0, 0.0, 0.0, 0.0),
        notes="J=5 at k=20: fifteen variance parameters"),
}
