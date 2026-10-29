"""`pimeta`'s UN fit, measured against an independent optimum.

The adapter calls two things and records both:

* ``ma.rma_mv(...)`` through its public arguments only, taking the fitted
  matrix from ``g_matrix()``; and
* :func:`metabench.reference.fit_un` on the *same* data, in a chosen
  parameterisation and from several starts.

Because both numbers are the value of the *same* restricted log-likelihood at a
particular `Sigma`, their difference is a statement that needs no shared code
to be trusted: if `l(Sigma_pimeta) < l(Sigma_reference)`, the fit that the
library returns is not the maximiser, whatever its own convergence flag says.

The reference is deliberately not allowed to see the library's answer: it
starts from the data, exactly as a user would have to.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np

import pimeta as ma

from ..reference import fit_un, un_reml_nll
from .base import FitRecord

VARIANTS = ("hyper", "external", "chol")


def _rho_from(Sigma: np.ndarray) -> list[float]:
    Sigma = np.asarray(Sigma, dtype=float)
    sd = np.sqrt(np.maximum(np.diag(Sigma), 1e-300))
    return [float(np.clip(Sigma[i, j] / (sd[i] * sd[j]), -1.0, 1.0))
            for i, j in itertools.combinations(range(Sigma.shape[0]), 2)]


class PimetaCovAdapter:
    name = "pimeta-cov"
    version = getattr(ma, "__version__", "unknown")

    def __init__(self, variant: str = "hyper", n_starts: int = 6,
                 method: str = "REML", lkj_eta: float = 0.0,
                 pinned: bool = False) -> None:
        if variant not in VARIANTS:
            raise ValueError(f"unknown reference variant {variant!r}; "
                             f"known: {list(VARIANTS)}")
        if method not in ("REML", "ML"):
            raise ValueError("method must be 'REML' or 'ML'")
        self.variant = variant
        self.n_starts = int(n_starts)
        self.method = method
        self.lkj_eta = float(lkj_eta)
        self.pinned = bool(pinned)
        suffix = ""
        if self.lkj_eta:
            suffix += f"-lkj{self.lkj_eta:g}"
        if self.pinned:
            suffix += "-pinned"
        self.name = f"pimeta-cov/{variant}{suffix}"

    def fit(self, data, *, level: float) -> FitRecord:
        record = FitRecord()
        truth_rho = None
        if self.pinned:
            # `rho=` fixes the correlation rather than starting from it, and
            # pinning it also moves the fit off the Cholesky route -- which is
            # the point: it is the comparison that separates "the search over
            # the correlations failed" from "the correlations are hard".
            truth_rho = _rho_from(data.Sigma)
        try:
            with warnings.catch_warnings(record=True) as caught:
                warnings.simplefilter("always")
                fit = ma.rma_mv(data.yi, V=data.V, mods=data.X, data=data.frame,
                                random="~ outcome | study", struct="UN",
                                intercept=False, method=self.method,
                                lkj_eta=self.lkj_eta, rho=truth_rho)
            record.warnings = len(caught)
            matrix, _names = fit.g_matrix()
        except Exception as exc:                              # noqa: BLE001
            record.failure = f"{type(exc).__name__}: {exc}"
            return record
        if matrix is None:                                    # pragma: no cover
            record.failure = "the fit reported no between-study covariance"
            return record

        Sigma = np.asarray(matrix, dtype=float)
        tau2 = np.diag(Sigma).copy()
        rho = _rho_from(Sigma)

        # --- layer 3 on the library's answer -------------------------------
        symmetric = bool(np.allclose(Sigma, Sigma.T, rtol=0.0, atol=0.0))
        eigenvalues = np.linalg.eigvalsh(0.5 * (Sigma + Sigma.T))
        mapping = np.outer(np.sqrt(np.maximum(tau2, 0.0)),
                           np.sqrt(np.maximum(tau2, 0.0)))
        R = np.eye(Sigma.shape[0])
        for value, (i, j) in zip(rho, itertools.combinations(
                range(Sigma.shape[0]), 2)):
            R[i, j] = R[j, i] = value
        mapping_gap = float(np.max(np.abs(Sigma - mapping * R)))

        # --- the independent optimum on the same data ----------------------
        studies = np.asarray(data.frame["study"])
        reference = fit_un(data.yi, data.V, data.X, variant=self.variant,
                           n_starts=self.n_starts, fixed_rho=truth_rho,
                           groups=studies)
        ll_at_fit = un_reml_nll(data.yi, data.V, data.X, Sigma)

        record.sigma_hat = [[float(x) for x in row] for row in Sigma]
        record.tau2 = float(tau2[0])
        record.converged = bool(getattr(fit, "converged", True))
        record.valid_covariance = bool(symmetric and eigenvalues.min() > 0.0
                                       and np.all(tau2 > 0.0))
        record.boundary_hit = bool(np.any(tau2 <= 1e-10)
                                   or np.any(np.abs(rho) >= 1.0 - 1e-6))
        record.extra = {
            "variant": self.variant,
            "n_starts": self.n_starts,
            "lkj_eta": self.lkj_eta,
            "pinned": self.pinned,
            "tau2_hat": [float(x) for x in tau2],
            "rho_hat": [float(x) for x in rho],
            "min_eig_hat": float(eigenvalues.min()),
            "symmetric": symmetric,
            "mapping_gap": mapping_gap,
            "ll_library": float(getattr(fit, "ll", float("nan"))),
            "ll_at_fit": float(-ll_at_fit),
            "ll_ref_opt": float(-reference.nll),
            "gap": float(ll_at_fit - reference.nll),
            "ref_sigma": [[float(x) for x in row] for row in reference.sigma],
            "ref_rho": [float(x) for x in reference.rho],
            "ref_n_at_best": int(reference.n_at_best),
            "ref_spread": float(reference.spread),
            "ref_min_eig": float(np.linalg.eigvalsh(reference.sigma).min()),
        }
        return record
