"""One model, written down from the algebra, fitted three ways.

The model is the multivariate random-effects model that family D judges::

    y_s ~ N(theta_s, V_s),   theta_s ~ N(mu, Sigma),   s = 1..k

with `Sigma` unstructured (UN, `J(J+1)/2` parameters), one mean per outcome,
and the sampling covariances `V_s` known.  Stacking the k studies with `X` the
(kJ x J) matrix of outcome indicators,

    M = V + X Sigma X',

and the restricted log-likelihood is

    -2 l_R(Sigma) = (k-p) log 2pi - log|X'X| + log|M| + log|X'M^-1 X| + r'M^-1 r

with `r` the residuals at the GLS estimate of `mu`.  It is written out here in
full rather than called from `pimeta`, so that a disagreement between the two
is evidence and not a tautology.  (The formula is the one metafor uses with
`REMLf = TRUE`; `un_reml_nll` is also checked against `pimeta`'s reported
log-likelihood in `tests/test_reference_un.py` -- a *parity* check, which shows
the two maximise the same thing but is not a correctness argument on its own.)

## Why three parameterisations

The same maximum can be approached through very different surfaces, and the
surface -- not the maximum -- is what makes an optimiser stop early:

``hyper``
    ``Sigma = D R D`` with ``R = L L'`` from the unit-row hyperspherical map
    ``L = L(z)``, ``z`` unbounded, ``D = diag(exp(a))``.  No constraints, so
    the optimiser never meets a wall: a correlation at ``+-1`` is a finite
    ``z``, and a variance going to zero is ``a -> -inf``.

``external``
    the ``(tau2, rho)`` vector ``pimeta`` exposes: ``tau2 = exp(a)``,
    ``rho = tanh(u)`` per pair, with an LKJ-shaped cliff returning
    ``1e5 + 1e7 (1e-6 - lam_min)^2`` when the implied ``R`` is not positive
    definite.  Deliberately the *same* cliff ``pimeta`` uses, so that its cost
    is measured rather than argued about.

``chol``
    a lower-triangular factor of ``Sigma`` itself, ``Sigma = L L'`` with
    ``L_ii = exp(a)``: ``pimeta``'s own route with the box replaced by a
    positive parameter, so the optimum is the same but no bound is needed.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from scipy import optimize

Parameterisation = Literal["hyper", "external", "chol"]

_CLIFF = 1e5
_CLIFF_SLOPE = 1e7


# --------------------------------------------------------------------------- #
# the likelihood                                                              #
# --------------------------------------------------------------------------- #
class UNLikelihood:
    """The restricted likelihood of the UN model, with the data work done once.

    The model is block diagonal by study -- ``M = blockdiag(V_s + Sigma)`` and
    every column of `X` is an outcome indicator -- so ``X'M^-1 X = sum_s
    M_s^-1`` and ``r'M^-1 r = sum_s r_s' M_s^-1 r_s``.  Evaluating it per block
    and evaluating those blocks *together*, as one ``(k, J, J)`` batch, is what
    makes a reference that has to run once per replicate affordable: an
    optimiser calls this thousands of times per replicate.

    Note that the random-effects covariance is ``blockdiag(Sigma)`` over
    studies, *not* ``X Sigma X'`` -- the latter couples any two rows that share
    an outcome, including rows from different studies, which is a different
    model.

    The layout assumed -- row ``i*J + j`` is outcome ``j`` of study ``i`` -- is
    the one :mod:`metabench.scenarios.covariance` produces and the one
    `pimeta` receives.
    """

    def __init__(self, y: np.ndarray, V: np.ndarray, X: np.ndarray) -> None:
        y = np.asarray(y, dtype=float).ravel()
        self.X = np.asarray(X, dtype=float)
        self.n_obs, self.J = self.X.shape
        self.k = self.n_obs // self.J
        if self.k * self.J != self.n_obs:
            raise ValueError(f"{self.n_obs} rows cannot be split into studies "
                             f"of {self.J} outcomes")
        self.Y = y.reshape(self.k, self.J)
        V = np.asarray(V, dtype=float)
        self.V_blocks = np.asarray([
            V[s * self.J:(s + 1) * self.J, s * self.J:(s + 1) * self.J]
            for s in range(self.k)])
        sign, logdet_XtX = np.linalg.slogdet(self.X.T @ self.X)
        self._logdet_XtX = float(logdet_XtX) if sign > 0 else None
        # The scale below is what "large" means for these data.  Beyond
        # `_SCALE_LIMIT` times it, log|M| and log|X'M^-1 X| are two numbers of
        # order kJ*log(lambda) that nearly cancel, and in floating point they
        # can cancel *past* the true minimum: a fit was observed to report
        # nll = -2648.9 on data whose best possible value is 9.3, i.e. a
        # likelihood better than any likelihood.  Rejecting the region is the
        # only honest response -- the number there is arithmetic, not
        # statistics.
        self.scale = max(float(np.max(np.diag(V))),
                         float(np.max(np.var(self.Y, axis=0, ddof=1))),
                         1e-12)

    _SCALE_LIMIT = 1e4

    def nll(self, Sigma: np.ndarray) -> float:
        Sigma = np.asarray(Sigma, dtype=float)
        if float(np.max(np.diag(Sigma))) > self._SCALE_LIMIT * self.scale:
            return 1e30
        M = self.V_blocks + Sigma                      # (k, J, J) broadcast
        sign, logdet = np.linalg.slogdet(M)
        if np.any(sign <= 0) or not np.all(np.isfinite(logdet)):
            return 1e30
        if self._logdet_XtX is None:
            return 1e30
        W = np.linalg.inv(M)
        XtWX = W.sum(axis=0)                           # X_s is the identity
        XtWy = np.einsum("sij,sj->i", W, self.Y)
        sign, logdet_XtWX = np.linalg.slogdet(XtWX)
        if sign <= 0 or not np.isfinite(logdet_XtWX):
            return 1e30
        beta = np.linalg.solve(XtWX, XtWy)
        r = self.Y - beta
        rss = float(np.einsum("si,sij,sj->", r, W, r))
        return 0.5 * (
            (self.n_obs - self.J) * math.log(2 * math.pi)
            - self._logdet_XtX
            + float(logdet.sum())
            + float(logdet_XtWX)
            + rss
        )


def un_reml_nll(y: np.ndarray, V: np.ndarray, X: np.ndarray,
                Sigma: np.ndarray) -> float:
    """`-l_R` for the UN model; a non-positive-definite `M` gives `1e30`.

    A thin wrapper over :class:`UNLikelihood` for one-off evaluations.
    Returned as a negative log-likelihood, i.e. what every optimiser here
    minimises.
    """
    return UNLikelihood(y, V, X).nll(Sigma)


# --------------------------------------------------------------------------- #
# parameterisations: a flat vector -> Sigma                                   #
# --------------------------------------------------------------------------- #
def pairs(J: int) -> list[tuple[int, int]]:
    """Row-major `(i, j)` with `j < i`, the order the `z` map consumes."""
    return [(i, j) for i in range(1, J) for j in range(i)]


def corr_from_z(J: int, z: np.ndarray) -> np.ndarray:
    """`R = L L'` with `L` the unit-row hyperspherical factor."""
    z = np.asarray(z, dtype=float).ravel()
    order = pairs(J)
    L = np.zeros((J, J))
    L[0, 0] = 1.0
    for i in range(1, J):
        acc = 1.0
        for j in range(i):
            value = float(z[order.index((i, j))])
            L[i, j] = acc * math.cos(value)
            acc *= math.sin(value)
        L[i, i] = acc
    R = L @ L.T
    R = 0.5 * (R + R.T)
    np.fill_diagonal(R, 1.0)
    return R


def z_from_corr(R: np.ndarray) -> np.ndarray:
    """The inverse of :func:`corr_from_z` for a positive-definite `R`."""
    R = np.asarray(R, dtype=float)
    J = R.shape[0]
    L = np.linalg.cholesky(R)
    out: list[float] = []
    for i in range(1, J):
        for j in range(i):
            tail = L[i, j:]
            denom = float(np.sqrt(np.sum(tail ** 2)))
            ratio = 1.0 if denom <= 0.0 else float(
                np.clip(L[i, j] / denom, -1.0, 1.0))
            out.append(math.acos(ratio))
    return np.asarray(out, dtype=float)


def _sigma_chol(theta: np.ndarray, J: int) -> np.ndarray:
    """`Sigma = L L'` with `L_ii = exp(theta_i)` and free off-diagonals."""
    L = np.zeros((J, J))
    for i in range(J):
        L[i, i] = math.exp(min(float(theta[i]), 15.0))
    idx = J
    for j in range(J):
        for i in range(j + 1, J):
            L[i, j] = float(theta[idx])
            idx += 1
    return L @ L.T


def _sigma_hyper(theta: np.ndarray, J: int) -> np.ndarray:
    sd = np.exp(np.clip(np.asarray(theta[:J], dtype=float), -15.0, 15.0))
    R = corr_from_z(J, theta[J:])
    return np.outer(sd, sd) * R


def _sigma_external(theta: np.ndarray, J: int) -> np.ndarray | None:
    """`None` when the implied correlation matrix is not positive definite."""
    tau2 = np.exp(np.clip(np.asarray(theta[:J], dtype=float), -15.0, 15.0))
    sd = np.sqrt(tau2)
    R = np.eye(J)
    for value, (i, j) in zip(np.asarray(theta[J:], dtype=float), pairs(J)):
        R[i, j] = R[j, i] = math.tanh(value)
    if float(np.linalg.eigvalsh(R)[0]) <= 1e-6:
        return None
    return np.outer(sd, sd) * R


def build(theta: np.ndarray, J: int,
          variant: Parameterisation) -> np.ndarray | None:
    if variant == "hyper":
        return _sigma_hyper(theta, J)
    if variant == "external":
        return _sigma_external(theta, J)
    if variant == "chol":
        return _sigma_chol(theta, J)
    raise ValueError(f"unknown variant {variant!r}")


def to_external(Sigma: np.ndarray) -> np.ndarray:
    """A covariance matrix -> `(tau2, rho)` in `pimeta`'s column-major order."""
    Sigma = np.asarray(Sigma, dtype=float)
    J = Sigma.shape[0]
    tau2 = np.diag(Sigma).copy()
    sd = np.sqrt(np.maximum(tau2, 1e-300))
    rho = [float(np.clip(Sigma[i, j] / (sd[i] * sd[j]), -1.0, 1.0))
           for i, j in itertools.combinations(range(J), 2)]
    return np.concatenate([tau2, np.asarray(rho, dtype=float)])


# --------------------------------------------------------------------------- #
# the fit                                                                     #
# --------------------------------------------------------------------------- #
@dataclass
class UNFit:
    """The best point the reference could find, and the evidence it looked."""

    variant: str
    sigma: np.ndarray
    nll: float
    theta: np.ndarray
    start_nlls: list[float] = field(default_factory=list)
    n_at_best: int = 1
    spread: float = 0.0
    n_evaluations: int = 0

    @property
    def tau2(self) -> np.ndarray:
        return np.diag(self.sigma).copy()

    @property
    def rho(self) -> np.ndarray:
        return to_external(self.sigma)[self.sigma.shape[0]:]


def _start_vector(y: np.ndarray, V: np.ndarray, X: np.ndarray, J: int,
                  variant: Parameterisation) -> np.ndarray:
    """A data-driven start: method-of-moments heterogeneity, zero correlation."""
    n_obs, J = X.shape
    k = n_obs // J
    Y = np.asarray(y, dtype=float).reshape(k, J)
    between = float(np.mean(np.var(Y, axis=0, ddof=1)))
    within = float(np.mean(np.diag(V)))
    entry = max(between - within, 1e-3)
    n_z = J * (J - 1) // 2
    if variant == "hyper":
        return np.concatenate([np.full(J, 0.5 * math.log(entry)),
                               np.full(n_z, math.pi / 4.0)])
    if variant == "external":
        return np.concatenate([np.full(J, math.log(entry)), np.zeros(n_z)])
    return np.concatenate([np.full(J, 0.5 * math.log(entry)), np.zeros(n_z)])


def _perturbations(start: np.ndarray, n_starts: int) -> list[np.ndarray]:
    """Deterministic perturbations of the data-driven start.

    Deterministic on purpose: a reference that draws its starts from a stream
    reports a different "how many starts found the best point" on every run,
    and that count is one of the things being measured.
    """
    out = [start]
    for i in range(1, n_starts):
        scale = 1.5 if i % 2 else 0.75
        step = np.array([(1.0 if (j + i) % 2 else -1.0)
                         for j in range(start.size)])
        perturbed = start.copy()
        perturbed += scale * step * (0.75 if i % 3 else 1.5)
        out.append(perturbed)
    return out


def fit_un(y: np.ndarray, V: np.ndarray, X: np.ndarray, *,
           variant: Parameterisation = "hyper", n_starts: int = 6,
           fixed_rho: np.ndarray | None = None,
           fixed_tau2: np.ndarray | None = None) -> UNFit:
    """Maximise `l_R` from many starts, under any of the three maps.

    `fixed_rho` / `fixed_tau2` mirror `pimeta`'s `rho=` / `sigma2=` and make a
    *constrained* comparison possible: pinning a parameter removes the surface
    a free one has to be searched over, so agreement while pinned and
    disagreement while free localises the problem to the search.
    """
    y = np.asarray(y, dtype=float).ravel()
    V = np.asarray(V, dtype=float)
    X = np.asarray(X, dtype=float)
    J = X.shape[1]
    likelihood = UNLikelihood(y, V, X)

    starts = _perturbations(_start_vector(y, V, X, J, variant), n_starts)
    mask = np.ones(starts[0].size, dtype=bool)
    if fixed_tau2 is not None:
        fixed_tau2 = np.asarray(fixed_tau2, dtype=float)
        starts = [np.concatenate([np.log(np.maximum(fixed_tau2, 1e-12)),
                                  s[J:]]) for s in starts]
        mask[:J] = False
    if fixed_rho is not None:
        R = np.eye(J)
        for value, (i, j) in zip(np.asarray(fixed_rho, dtype=float), pairs(J)):
            R[i, j] = R[j, i] = float(value)
        z = z_from_corr(R)
        starts = [np.concatenate([s[:J], z]) for s in starts]
        mask[J:] = False

    def nll(theta: np.ndarray) -> float:
        Sigma = build(theta, J, variant)
        if Sigma is None:                          # only `external` can fail
            R = np.eye(J)
            for value, (i, j) in zip(theta[J:], pairs(J)):
                R[i, j] = R[j, i] = math.tanh(value)
            lam = float(np.linalg.eigvalsh(R)[0])
            return _CLIFF + _CLIFF_SLOPE * max(0.0, 1e-6 - lam) ** 2
        return likelihood.nll(Sigma)

    best_nll = math.inf
    best_theta = starts[0].copy()
    start_nlls: list[float] = []
    evaluations = 0
    for start in starts:
        free = start[mask]

        def wrapped(theta_free: np.ndarray, _start=start) -> float:
            theta = _start.copy()
            theta[mask] = theta_free
            return nll(theta)

        if free.size == 0:
            value = nll(start)
            point = np.zeros(0)
            evaluations += 1
        else:
            # A hard budget matters here: with 15 variance parameters and 30
            # studies the surface is flat enough that L-BFGS-B can spend
            # tens of thousands of evaluations gaining nothing, and the
            # reference is run once per replicate.  The budget is per start,
            # and the best of the starts is what gets reported.
            result = optimize.minimize(
                wrapped, free, method="L-BFGS-B",
                options={"maxiter": 1000, "maxfun": 4000,
                         "ftol": 1e-12, "gtol": 1e-9, "maxls": 40})
            value = float(result.fun)
            point = np.asarray(result.x, dtype=float)
            evaluations += int(getattr(result, "nfev", 0) or 0)
        start_nlls.append(value)
        if value < best_nll:
            best_nll = value
            best_theta = start.copy()
            best_theta[mask] = point

    Sigma = build(best_theta, J, variant)
    if Sigma is None:                                            # pragma: no cover
        raise RuntimeError("the reference optimum is not a covariance matrix")
    at_best = sum(1 for value in start_nlls if value - best_nll < 1e-6)
    return UNFit(variant=variant, sigma=np.asarray(Sigma, dtype=float),
                 nll=float(best_nll), theta=best_theta,
                 start_nlls=start_nlls, n_at_best=int(at_best),
                 spread=float(max(start_nlls) - min(start_nlls)),
                 n_evaluations=evaluations)
