"""Blend the point estimate and the upper bound, and see which blend calibrates.

The 95 % upper bound for tau^2 is a large over-correction once it is plugged
into a *prediction* interval: it targets a one-sided coverage statement about
tau^2, not a two-sided statement about a new study.  Measured, it gives 0.99
where 0.95 was asked for, at four times the width at k = 5.

So the usable rule is a blend::

    tau2_used = tau2_hat + lam * (tau2_ub - tau2_hat)

which is the shipped interval at ``lam = 0`` and the upper-bound interval at
``lam = 1``.  This script sweeps ``lam`` on one seed; the chosen value is then
confirmed on a different seed by rerunning the adapter, because a constant picked
and checked on the same replicates is not checked at all.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(r"D:\autometa\metabench")
CELLS = ("k05-tau004", "k05-tau013", "k10-tau004", "k10-tau013",
         "k20-tau004", "k20-tau013", "k50-tau004", "k50-tau013",
         "k100-tau004", "k100-tau013")
LAMBDAS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.75, 1.0)


def load(cell: str) -> list[dict]:
    folder = ROOT / "results" / f"univariate-pi_{cell}" / "pimeta-pi" / "upper95"
    manifests = [(json.loads(p.read_text(encoding="utf-8"))["reps"], p)
                 for p in sorted(folder.glob("manifest_*.json"))]
    if not manifests:
        return []
    _reps, path = max(manifests)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    with (folder / manifest["raw"]).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [r for r in rows if r.get("failure") is None]


def main() -> int:
    cells = {cell: load(cell) for cell in CELLS}
    cells = {k: v for k, v in cells.items() if v}
    print(f"{'cell':<12} " + " ".join(f"{lam:>7.2f}" for lam in LAMBDAS))
    for cell, rows in cells.items():
        dev = np.array([abs(r["truth_new"] - r["estimate"]) for r in rows])
        se = np.array([r["se"] for r in rows])
        tau2 = np.array([r["tau2"] for r in rows])
        used = np.array([r["extra"]["tau2_used"] for r in rows])
        df = int(rows[0]["extra"]["df"])
        from scipy import stats
        q = float(stats.t.ppf(0.975, df))
        parts = []
        for lam in LAMBDAS:
            blended = tau2 + lam * (used - tau2)
            half = q * np.sqrt(np.maximum(blended, 0.0) + se ** 2)
            parts.append(f"{np.mean(dev <= half):>7.3f}")
        print(f"{cell:<12} " + " ".join(parts))
    print("\nlambda = 0 is the shipped interval, 1 is the 95% upper bound")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
