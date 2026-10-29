"""Is the small-k under-coverage caused by the boundary?

The screening grid showed the joint region covering 66-88% where it claims 95%,
worst at k = 5.  One candidate mechanism is measurable without new runs: a
replicate whose estimated between-study covariance sits on the boundary
(some tau2 component at zero) gets a *smaller* Sigma, hence a *narrower*
region, hence a miss.  If that is the mechanism, coverage among the
boundary-hit replicates must be much worse than among the interior ones.

Run: python -m tools.diagnose_boundary results
"""

from __future__ import annotations

import json
import pathlib
import sys
from typing import Any

import numpy as np
from scipy import stats


def _contained(row: dict[str, Any]) -> bool | None:
    """Quadratic form of the truth in the recorded ellipsoid -- the metric's job."""
    center = row.get("region_center")
    cov = row.get("region_cov")
    truth = row.get("truth_new_vec")
    crit = row.get("region_crit")
    if not center or not cov or not truth or crit is None:
        return None
    d = np.asarray(truth, dtype=float) - np.asarray(center, dtype=float)
    try:
        w = float(d @ np.linalg.solve(np.asarray(cov, dtype=float), d))
    except np.linalg.LinAlgError:
        return None
    return w <= float(crit)


def _ratio(k: int, n: int, z: float = 1.96) -> float:
    p = k / n
    return z * float(np.sqrt(p * (1 - p) / n))


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = pathlib.Path(argv[0] if argv else "results")
    rows: dict[tuple[str, str], list[dict]] = {}
    for raw in sorted(root.rglob("raw_*.jsonl")):
        with raw.open(encoding="utf-8") as handle:
            for line in handle:
                row = json.loads(line)
                key = (row["scenario_id"], row["method_id"].split("/")[1]
                       if "/" in row["method_id"] else row["method_id"])
                rows.setdefault(key, []).append(row)

    print(f"{'scenario':24}{'test':8}{'n':>4}{'covered':>9}"
          f"{'interior':>16}{'on boundary':>16}{'bound%':>9}")
    for (scenario, adapter), group in sorted(rows.items()):
        hit = [r for r in group if r.get("boundary_hit")]
        free = [r for r in group if not r.get("boundary_hit")]
        usable = [(r, _contained(r)) for r in group]
        usable = [(r, c) for r, c in usable if c is not None]
        if not usable:
            continue
        total = len(usable)
        cover = sum(c for _, c in usable)
        ci = [c for r, c in usable if not r.get("boundary_hit")]
        bi = [c for r, c in usable if r.get("boundary_hit")]

        def cell(values: list[bool]) -> str:
            if not values:
                return "—"
            p = sum(values) / len(values)
            return f"{100 * p:.1f}% ±{100 * _ratio(sum(values), len(values)):.1f}"

        print(f"{scenario:24}{adapter:8}{total:>4}"
              f"{100 * cover / total:>8.1f}%{cell(ci):>16}{cell(bi):>16}"
              f"{100 * len(hit) / len(group):>8.1f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
