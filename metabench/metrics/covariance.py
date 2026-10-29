"""Metrics for a matrix-valued estimand.

Three questions, kept apart on purpose:

1. **Is it a covariance matrix?** -- `non_pd_rate`, `asymmetric_rate`.
2. **Is it the maximum?** -- `gap` is `l(Sigma_hat) - l(Sigma_reference)`, both
   evaluated by the same independent likelihood.  `suboptimal_rate` counts the
   replicate where that gap exceeds the tolerance, `better_rate` the one where
   the library beat the reference (which would mean the *reference* failed).
3. **Is it close to the truth?** -- relative Frobenius error, bias of the
   variances and the correlations, and the conditioning of the estimate against
   the truth's.

The middle question is the one a convergence flag cannot answer, and the first
one is cheap enough that there is no reason to leave it to inspection.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

import numpy as np

#: A gap below this is arithmetic noise from two different but equivalent
#: routes to the same number; measured across the grid it is ~1e-10.
GAP_TOL = 1e-6


def _prop(count: int, total: int) -> dict[str, float | None]:
    if total <= 0:
        return {"estimate": None, "mcse": None, "halfwidth_95": None}
    p = count / total
    mcse = math.sqrt(p * (1.0 - p) / total)
    return {"estimate": p, "mcse": mcse, "halfwidth_95": 1.96 * mcse}


def _numbers(records: Iterable[dict], key: str) -> list[float]:
    out = []
    for record in records:
        value = (record.get("extra") or {}).get(key)
        if value is not None and np.isfinite(value):
            out.append(float(value))
    return out


def _quantile(values: Sequence[float], q: float) -> float | None:
    return None if not values else float(np.quantile(values, q))


def summarise(records: Sequence[dict[str, Any]]) -> dict[str, Any]:
    reps = len(records)
    failures = sum(1 for r in records if r.get("failure"))
    usable = [r for r in records if r.get("sigma_hat") is not None]
    n = len(usable)

    gaps = _numbers(usable, "gap")
    spreads = _numbers(usable, "ref_spread")
    not_at_best = sum(1 for r in usable
                      if (r.get("extra") or {}).get("ref_n_at_best", 1) < 3)

    # --- validity (question 1) ------------------------------------------
    non_pd = sum(1 for r in usable if not r.get("valid_covariance"))
    asymmetric = sum(1 for r in usable
                     if not (r.get("extra") or {}).get("symmetric", True))
    boundary = sum(1 for r in usable if r.get("boundary_hit"))

    # --- accuracy against the truth (question 3) ------------------------
    frob: list[float] = []
    tau_bias: list[float] = []
    rho_bias: list[float] = []
    conditioning: list[float] = []
    for r in usable:
        truth = r.get("truth_cov")
        if truth is None:
            continue
        T = np.asarray(truth, dtype=float)
        S = np.asarray(r["sigma_hat"], dtype=float)
        denom = float(np.linalg.norm(T))
        if denom > 0:
            frob.append(float(np.linalg.norm(S - T) / denom))
        tau_true = np.diag(T)
        tau_hat = np.diag(S)
        with np.errstate(divide="ignore", invalid="ignore"):
            rel = (tau_hat - tau_true) / tau_true
        tau_bias.extend(float(x) for x in rel if np.isfinite(x))
        pairs = [(i, j) for i in range(T.shape[0]) for j in range(i + 1,
                                                               T.shape[0])]
        sd_t = np.sqrt(np.maximum(tau_true, 1e-300))
        sd_h = np.sqrt(np.maximum(tau_hat, 1e-300))
        for i, j in pairs:
            rho_bias.append(float(S[i, j] / (sd_h[i] * sd_h[j])
                                  - T[i, j] / (sd_t[i] * sd_t[j])))
        lam_t = float(np.linalg.eigvalsh(T)[0])
        lam_h = float(np.linalg.eigvalsh(0.5 * (S + S.T))[0])
        if lam_t > 0:
            conditioning.append(lam_h / lam_t)

    mapping = _numbers(usable, "mapping_gap")
    ll_mismatch = sum(
        1 for r in usable
        if abs((r.get("extra") or {}).get("ll_library", 0.0)
               - (r.get("extra") or {}).get("ll_at_fit", 0.0)) > 1e-6)
    runtimes = [float(r["runtime"]) for r in usable if r.get("runtime")]

    return {
        "reps": reps,
        "usable": n,
        "runtime": {
            "median": _quantile(runtimes, 0.5),
            "p90": _quantile(runtimes, 0.9),
        },
        "rates": {
            "failure_rate": failures / reps if reps else None,
            "non_pd_rate": non_pd / n if n else None,
            "asymmetric_rate": asymmetric / n if n else None,
            "boundary_rate": boundary / n if n else None,
            "suboptimal_rate": (sum(1 for g in gaps if g > GAP_TOL) / len(gaps)
                                if gaps else None),
            "better_rate": (sum(1 for g in gaps if g < -GAP_TOL) / len(gaps)
                            if gaps else None),
            "multistart_conflict_rate": (not_at_best / n) if n else None,
            "ll_mismatch_rate": (ll_mismatch / n) if n else None,
        },
        "gap": {
            "median": _quantile(gaps, 0.5),
            "q90": _quantile(gaps, 0.9),
            "max": max(gaps) if gaps else None,
        },
        "ref_spread": {
            "median": _quantile(spreads, 0.5),
            "q90": _quantile(spreads, 0.9),
        },
        "frobenius": {
            "median": _quantile(frob, 0.5),
            "q90": _quantile(frob, 0.9),
        },
        "tau2_relative_bias": float(np.mean(tau_bias)) if tau_bias else None,
        "rho_bias": float(np.mean(rho_bias)) if rho_bias else None,
        "min_eig_ratio": {
            "median": _quantile(conditioning, 0.5),
            "q10": _quantile(conditioning, 0.1),
        },
        "mapping_gap_max": max(mapping) if mapping else None,
        "proportions": {
            "non_pd": _prop(non_pd, n),
            "boundary": _prop(boundary, n),
            "suboptimal": _prop(sum(1 for g in gaps if g > GAP_TOL), len(gaps)),
            "multistart_conflict": _prop(not_at_best, n),
        },
    }
