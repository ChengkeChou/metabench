"""Metrics, always recomputed from raw replicate records.

Two rules, both learned from the evaluations this harness exists for:

* a replicate that failed stays in the data and is *counted*; coverage is
  reported twice, conditional on a usable fit and with failures counted as
  misses, because quietly dropping failures makes a fragile method look good;
* every proportion carries its Monte Carlo standard error, so that a 0.917 from
  sixty replicates is not read like a 0.917 from two thousand.
"""

from __future__ import annotations

import math
from typing import Any, Iterable, Sequence

Record = dict[str, Any]


def mcse_proportion(p: float, n: int) -> float:
    """Monte Carlo standard error of a proportion."""
    if n <= 0 or not math.isfinite(p):
        return float("nan")
    return math.sqrt(p * (1.0 - p) / n)


def mcse_halfwidth(n: int, p: float = 0.95) -> float:
    """Half-width of the 95 % Monte Carlo interval for a coverage estimate."""
    return 1.959963984540054 * mcse_proportion(p, n)


def _usable(records: Sequence[Record]) -> list[Record]:
    return [r for r in records
            if r.get("failure") is None
            and r.get("ci_lb") is not None and r.get("ci_ub") is not None]


def coverage(records: Sequence[Record], *, lo: str, hi: str,
             truth: str = "truth") -> dict[str, Any]:
    """Coverage of ``[lo, hi]`` for the per-record ``truth``.

    ``conditional`` uses the usable records only; ``with_failures`` keeps the
    whole denominator, so a method that fails often cannot hide it.  Both are
    returned, and neither is presented without the other.
    """
    usable = [r for r in records if r.get(lo) is not None
              and r.get(hi) is not None and r.get("failure") is None]
    hits = sum(1 for r in usable if r[lo] <= r[truth] <= r[hi])
    n = len(usable)
    total = len(records)
    p = hits / n if n else float("nan")
    p_all = hits / total if total else float("nan")
    return {
        "hits": hits,
        "reps_used": n,
        "reps_total": total,
        "estimate": p,
        "mcse": mcse_proportion(p, n),
        "halfwidth_95": mcse_halfwidth(n, p if n else 0.95),
        "with_failures": p_all,
        "mcse_with_failures": mcse_proportion(p_all, total),
    }


def mean_width(records: Sequence[Record], *, lo: str, hi: str) -> float:
    widths = [r[hi] - r[lo] for r in records
              if r.get(lo) is not None and r.get(hi) is not None]
    return float(sum(widths) / len(widths)) if widths else float("nan")


def bias(records: Sequence[Record], *, truth: str = "truth") -> dict[str, float]:
    values = [r["estimate"] for r in _usable(records)]
    if not values:
        return {"estimate": float("nan"), "mcse": float("nan")}
    mean = sum(values) / len(values)
    if len(values) > 1:
        var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
        mcse = math.sqrt(var / len(values))
    else:
        mcse = float("nan")
    target = records[0][truth]
    return {"estimate": mean - target, "mcse": mcse,
            "mean_estimate": mean, "truth": target}


def dispersion(records: Sequence[Record]) -> dict[str, float]:
    """Empirical SD of the estimates against the mean reported standard error."""
    estimates = [r["estimate"] for r in _usable(records)]
    ses = [r["se"] for r in _usable(records) if r.get("se") is not None]
    if len(estimates) < 2 or not ses:
        return {"empirical_sd": float("nan"), "mean_reported_se": float("nan"),
                "se_ratio": float("nan")}
    mean = sum(estimates) / len(estimates)
    var = sum((v - mean) ** 2 for v in estimates) / (len(estimates) - 1)
    empirical_sd = math.sqrt(var)
    mean_se = sum(ses) / len(ses)
    return {
        "empirical_sd": empirical_sd,
        "mean_reported_se": mean_se,
        "se_ratio": mean_se / empirical_sd if empirical_sd > 0 else float("nan"),
    }


def rates(records: Sequence[Record]) -> dict[str, float]:
    """Failure, boundary, non-convergence and invalid-covariance rates."""
    total = len(records)
    if not total:
        return {}
    def rate(key: str, predicate) -> float:
        return sum(1 for r in records if predicate(r)) / total
    return {
        "failure_rate": rate("failure", lambda r: r.get("failure") is not None),
        "boundary_rate": rate("boundary", lambda r: bool(r.get("boundary_hit"))),
        "non_converged_rate": rate(
            "converged", lambda r: r.get("converged") is False),
        "invalid_covariance_rate": rate(
            "cov", lambda r: r.get("valid_covariance") is False),
    }


def summarise(records: Iterable[Record]) -> dict[str, Any]:
    """Every metric this harness reports, from one list of raw records."""
    rows = list(records)
    return {
        "reps": len(rows),
        "ci": coverage(rows, lo="ci_lb", hi="ci_ub", truth="truth"),
        "pi": coverage(rows, lo="pi_lb", hi="pi_ub", truth="truth_new"),
        "pi_t": coverage(rows, lo="pi_t_lb", hi="pi_t_ub", truth="truth_new"),
        "ci_width": mean_width(rows, lo="ci_lb", hi="ci_ub"),
        "pi_width": mean_width(rows, lo="pi_lb", hi="pi_ub"),
        "pi_t_width": mean_width(rows, lo="pi_t_lb", hi="pi_t_ub"),
        "bias": bias(rows),
        "dispersion": dispersion(rows),
        "rates": rates(rows),
    }
