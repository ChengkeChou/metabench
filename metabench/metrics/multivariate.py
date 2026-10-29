"""Metrics for an ellipsoidal region, evaluated at the truth.

The region is `(x - center)' cov^-1 (x - center) <= crit`.  Containment is
computed here, from the three recorded quantities, so a region cannot report
its own coverage.  The volume proxy is ``crit**(J/2) * sqrt(det(cov))`` up to
the unit-sphere constant -- monotone in the volume and free of the dimension
bookkeeping, which is all a paired comparison needs.
"""

from __future__ import annotations

import math
from typing import Any, Sequence

import numpy as np

from .core import mcse_proportion, rates

Record = dict[str, Any]


def _quadform(record: Record, point: Sequence[float]) -> float:
    center = np.asarray(record["region_center"], dtype=float)
    cov = np.asarray(record["region_cov"], dtype=float)
    delta = np.asarray(point, dtype=float) - center
    return float(delta @ np.linalg.solve(cov, delta))


def _covered(record: Record, key: str) -> bool:
    return _quadform(record, record[key]) <= record["region_crit"]


def joint_coverage(records: Sequence[Record], *, key: str = "truth_new_vec"
                   ) -> dict[str, Any]:
    usable = [r for r in records
              if r.get("failure") is None and r.get("region_center") is not None]
    hits = sum(1 for r in usable if _covered(r, key))
    n = len(usable)
    total = len(records)
    p = hits / n if n else float("nan")
    return {
        "hits": hits,
        "reps_used": n,
        "reps_total": total,
        "estimate": p,
        "mcse": mcse_proportion(p, n),
        "with_failures": hits / total if total else float("nan"),
    }


def marginal_coverage(records: Sequence[Record], *,
                      key: str = "truth_new_vec") -> list[dict[str, Any]]:
    """Per-outcome coverage of the interval the region implies."""
    usable = [r for r in records
              if r.get("failure") is None and r.get("marginal_lb") is not None]
    if not usable:
        return []
    dim = len(usable[0]["marginal_lb"])
    out = []
    for j in range(dim):
        hits = sum(1 for r in usable
                   if r["marginal_lb"][j] <= r[key][j] <= r["marginal_ub"][j])
        p = hits / len(usable)
        out.append({"outcome": j, "hits": hits, "reps_used": len(usable),
                    "estimate": p, "mcse": mcse_proportion(p, len(usable))})
    return out


def volume(records: Sequence[Record]) -> float:
    """Mean volume proxy: ``crit**(J/2) * sqrt(det(cov))``."""
    values = []
    for r in records:
        if r.get("failure") is not None or r.get("region_cov") is None:
            continue
        cov = np.asarray(r["region_cov"], dtype=float)
        det = float(np.linalg.det(cov))
        if det <= 0:
            continue
        dim = cov.shape[0]
        values.append(r["region_crit"] ** (dim / 2.0) * math.sqrt(det))
    return float(sum(values) / len(values)) if values else float("nan")


def summarise(records: Sequence[Record]) -> dict[str, Any]:
    rows = list(records)
    return {
        "reps": len(rows),
        "joint_new": joint_coverage(rows, key="truth_new_vec"),
        "joint_mean": joint_coverage(rows, key="truth_vec"),
        "marginal_new": marginal_coverage(rows, key="truth_new_vec"),
        "marginal_mean": marginal_coverage(rows, key="truth_vec"),
        "volume": volume(rows),
        "rates": rates(rows),
    }
