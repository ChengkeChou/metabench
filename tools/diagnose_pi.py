"""Diagnose the prediction-interval deficit: scale, boundary, and df.

Coverage alone says *that* an interval is short.  This asks *why*, from the raw
records, using three quantities the raw file already holds:

* the ratio of the empirical SD of ``theta_new - mu_hat`` to the SD the reported
  interval implies — the inflation a corrected interval would need;
* coverage split by whether tau^2 landed on the boundary, because a truncated
  tau^2 is the obvious mechanism at small k;
* the same for the ``predtype="t"`` interval, which charges the t distribution
  for the uncertainty in tau^2.

Run: ``python tools/diagnose_pi.py`` from the metabench root.
"""

from __future__ import annotations

import json
import math
import pathlib

from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
CELLS = ("k05-tau004", "k05-tau013", "k10-tau004", "k10-tau013",
         "k20-tau004", "k20-tau013", "k50-tau004", "k50-tau013",
         "k100-tau004", "k100-tau013")


def load(cell: str) -> list[dict]:
    """The largest run for this cell, so screening and formal runs never mix."""
    folder = ROOT / "results" / f"univariate-pi_{cell}" / "pimeta"
    manifests = sorted(folder.glob("manifest_*.json"))
    if not manifests:
        return []
    best = max(manifests,
               key=lambda p: json.loads(p.read_text(encoding="utf-8"))["reps"])
    raw = folder / json.loads(best.read_text(encoding="utf-8"))["raw"]
    rows: list[dict] = []
    with raw.open(encoding="utf-8") as handle:
        rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def sd(values: list[float]) -> float:
    n = len(values)
    if n < 2:
        return float("nan")
    mean = sum(values) / n
    return math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))


def covered(row: dict, lo: str, hi: str) -> bool:
    return row[lo] <= row["truth_new"] <= row[hi]


def main() -> int:
    header = (f"{'cell':<12} {'n':>4} {'cov':>6} {'sd_imp':>7} {'sd_emp':>7} "
              f"{'ratio':>6} {'cov_int':>8} {'cov_bnd':>8} {'n_bnd':>6} "
              f"{'cov_t':>6} {'ratio_t':>8}")
    print(header)
    for cell in CELLS:
        rows = [r for r in load(cell) if r.get("failure") is None]
        if not rows:
            continue
        k = int(cell.split("-")[0][1:])
        q = float(stats.t.ppf(0.975, k - 1))
        widths = [r["pi_ub"] - r["pi_lb"] for r in rows]
        widths_t = [r["pi_t_ub"] - r["pi_t_lb"] for r in rows]
        sd_emp = sd([r["truth_new"] - r["estimate"] for r in rows])
        sd_imp = (sum(widths) / len(widths)) / (2 * q)
        sd_imp_t = (sum(widths_t) / len(widths_t)) / (2 * q)
        interior = [r for r in rows if (r.get("tau2") or 0.0) > 0.0]
        boundary = [r for r in rows if (r.get("tau2") or 0.0) <= 0.0]
        cov_int = (sum(1 for r in interior if covered(r, "pi_lb", "pi_ub")) / len(interior)
                   if interior else float("nan"))
        cov_bnd = (sum(1 for r in boundary if covered(r, "pi_lb", "pi_ub")) / len(boundary)
                   if boundary else float("nan"))
        print(f"{cell:<12} {len(rows):>4} {sum(1 for r in rows if covered(r, 'pi_lb', 'pi_ub')) / len(rows):>6.3f} "
              f"{sd_imp:>7.4f} {sd_emp:>7.4f} {sd_emp / sd_imp:>6.3f} "
              f"{cov_int:>8.3f} {cov_bnd:>8.3f} {len(boundary):>6} "
              f"{sum(1 for r in rows if covered(r, 'pi_t_lb', 'pi_t_ub')) / len(rows):>6.3f} "
              f"{sd_emp / sd_imp_t:>8.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
