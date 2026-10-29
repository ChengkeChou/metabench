"""Family D: the UN covariance matrix itself.

Family B asked whether an *interval* covers.  This family asks a narrower and
more basic question about the same model: when `rma_mv` fits an unstructured
between-study covariance, is the matrix it returns

* a covariance matrix at all (symmetric, positive definite, variances >= 0);
* the maximiser of the likelihood it claims to maximise;
* the same matrix whatever parameterisation is used to reach it;
* and only then, close to the truth?

The estimand is the whole matrix `Sigma`, so the truth is recorded as a matrix
and every metric is computed against it.  The design keeps `V_s` diagonal and
the means at zero: the covariance is the only thing being judged, and a
correlated `V_s` or a moderator would give a second explanation for a
disagreement.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .base import Scenario


@dataclass(frozen=True)
class CovarianceData:
    yi: np.ndarray          # (k*J,)
    V: np.ndarray           # (k*J, k*J), block diagonal per study
    X: np.ndarray           # (k*J, J), one column per outcome
    frame: dict[str, list]  # the `data=` argument rma_mv needs
    Sigma: np.ndarray       # (J, J) the true between-study covariance
    mu: np.ndarray          # (J,) true mean (all zeros here)


def _correlation(J: int, rho: Any) -> np.ndarray:
    """Equicorrelation from a scalar, or a matrix from a nested list."""
    if np.isscalar(rho):
        R = np.eye(J)
        for i in range(J):
            for j in range(i + 1, J):
                R[i, j] = R[j, i] = float(rho)
        return R
    R = np.asarray(rho, dtype=float)
    if R.shape != (J, J):
        raise ValueError(f"'rho' must give a {J}x{J} correlation matrix")
    return R


def simulate(dgm: dict[str, Any], rng: np.random.Generator) -> CovarianceData:
    """``theta_i ~ N(0, Sigma)``, ``y_i ~ N(theta_i, diag(se^2))``."""
    k = int(dgm["k"])
    sd = np.asarray(dgm["sd"], dtype=float)
    J = int(sd.size)
    se = np.asarray(dgm["se"], dtype=float)
    if se.size != J:
        raise ValueError("'se' must have one entry per outcome")
    R = _correlation(J, dgm.get("rho", 0.3))
    Sigma = np.outer(sd, sd) * R

    theta = rng.multivariate_normal(np.zeros(J), Sigma, k)
    yi = np.empty((k, J))
    V = np.zeros((k * J, k * J))
    for i in range(k):
        yi[i] = rng.normal(theta[i], se)
        block = np.diag(se ** 2)
        V[i * J:(i + 1) * J, i * J:(i + 1) * J] = block

    X = np.zeros((k * J, J))
    X[np.arange(k * J), np.tile(np.arange(J), k)] = 1.0
    frame = {
        "study": [f"s{i:03d}" for i in range(k) for _ in range(J)],
        "outcome": [f"o{j}" for _ in range(k) for j in range(J)],
    }
    return CovarianceData(yi=yi.ravel(), V=V, X=X, frame=frame,
                          Sigma=Sigma, mu=np.zeros(J))


def scenario(scenario_id: str, *, k: int, sd: tuple[float, ...],
             se: tuple[float, ...], rho: Any = 0.3,
             notes: str = "") -> Scenario:
    J = len(sd)
    return Scenario(
        scenario_id=scenario_id,
        family="covariance",
        dgm={"k": k, "J": J, "sd": list(sd), "se": list(se),
             "rho": rho if np.isscalar(rho) else [[float(x) for x in row]
                                                  for row in np.asarray(rho)]},
        estimand={"target": "Sigma",
                  "note": "the whole between-study covariance: each entry is "
                          "judged, and so is the matrix's conditioning"},
        notes=notes,
    )


#: Screening design.  Every cell varies *one* thing that decides whether the
#: search is easy: the number of parameters (J), the strength and the sign of
#: the correlation, and whether a variance component is near the boundary.
SCENARIOS: dict[str, Scenario] = {
    "covariance/k10-J2-rho30": scenario(
        "covariance/k10-J2-rho30", k=10, sd=(0.20, 0.30), se=(0.20, 0.25),
        rho=0.3,
        notes="J=2: three parameters, the only size where the per-pair box is "
              "the whole cone.  The control cell for the bigger matrices."),
    "covariance/k10-J3-rho30": scenario(
        "covariance/k10-J3-rho30", k=10, sd=(0.20, 0.30, 0.25),
        se=(0.20, 0.25, 0.30), rho=0.3,
        notes="the smallest J where the box is no longer the cone"),
    "covariance/k10-J3-rho70": scenario(
        "covariance/k10-J3-rho70", k=10, sd=(0.20, 0.30, 0.25),
        se=(0.20, 0.25, 0.30), rho=0.7,
        notes="a nearly collinear set of outcomes: the surface is flat along "
              "the direction that decreases all three correlations"),
    "covariance/k10-J3-mixed": scenario(
        "covariance/k10-J3-mixed", k=10, sd=(0.25, 0.18, 0.30),
        se=(0.20, 0.25, 0.30),
        rho=[[1.0, 0.5, -0.4], [0.5, 1.0, 0.1], [-0.4, 0.1, 1.0]],
        notes="one negative correlation and one near zero: the cell a "
              "parameterisation can fail on without failing a symmetric one"),
    "covariance/k30-J3-rho30": scenario(
        "covariance/k30-J3-rho30", k=30, sd=(0.20, 0.30, 0.25),
        se=(0.20, 0.25, 0.30), rho=0.3,
        notes="same as k10-J3-rho30 with three times the studies: separates a "
              "search problem from an information problem"),
    "covariance/k30-J5-rho30": scenario(
        "covariance/k30-J5-rho30", k=30,
        sd=(0.20, 0.30, 0.25, 0.15, 0.22),
        se=(0.20, 0.25, 0.30, 0.18, 0.22), rho=0.3,
        notes="15 variance parameters from 30 studies: the largest screening "
              "cell, and the one where a Cholesky route has the most room to "
              "get stuck"),
    "covariance/k30-J5-rho70": scenario(
        "covariance/k30-J5-rho70", k=30,
        sd=(0.20, 0.30, 0.25, 0.15, 0.22),
        se=(0.20, 0.25, 0.30, 0.18, 0.22), rho=0.7,
        notes="15 parameters with a nearly rank-deficient truth"),
    "covariance/k30-J3-tiny": scenario(
        "covariance/k30-J3-tiny", k=30, sd=(0.03, 0.30, 0.25),
        se=(0.20, 0.25, 0.30), rho=0.3,
        notes="the first variance component is 100x smaller than the second: "
              "the boundary cell, where a variance estimate sits at zero"),
}
