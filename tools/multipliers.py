"""Required critical-value multiplier for every measured cell.

Prints, per cell, the reference's coverage, the 95th percentile of the
pivot, and the multiplier that percentile implies, for both closed-form bases.
Cells whose runs are still in flight simply show what is on disk.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1] / "results"
CELLS = ("multivariate_k05-rho30", "multivariate_k10-rho30",
         "multivariate_k20-rho30", "multivariate_k50-rho30",
         "multivariate_k10-rho70", "multivariate_k20-rho70",
         "multivariate_k10-J3-rho30", "multivariate_k20-J3-rho30",
         "multivariate_k20-J5-rho30")
REFS = ("chi2", "f", "hk")


def load(cell: str, ref: str) -> list[dict]:
    folder = ROOT / cell / "pimeta-region" / ref
    manifests = [(json.loads(p.read_text(encoding="utf-8"))["reps"], p)
                 for p in sorted(folder.glob("manifest_*.json"))]
    if not manifests:
        return []
    _reps, path = max(manifests)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    with (folder / manifest["raw"]).open(encoding="utf-8") as handle:
        rows = [json.loads(line) for line in handle if line.strip()]
    return [r for r in rows if r.get("failure") is None]


def pivot(row: dict) -> float:
    centre = np.asarray(row["region_center"], dtype=float)
    cov = np.asarray(row["region_cov"], dtype=float)
    delta = np.asarray(row["truth_new_vec"], dtype=float) - centre
    try:
        return float(delta @ np.linalg.solve(cov, delta))
    except np.linalg.LinAlgError:
        return float("nan")


def main() -> int:
    print(f"{'cell':<26} {'ref':>4} {'n':>4} {'cov':>6} {'crit':>7} "
          f"{'p95':>7} {'need':>5}")
    for cell in CELLS:
        for ref in REFS:
            rows = load(cell, ref)
            if not rows:
                continue
            values = np.array([pivot(r) for r in rows])
            values = values[np.isfinite(values)]
            crit = float(np.median([r["region_crit"] for r in rows]))
            covered = float(np.mean(values <= crit))
            p95 = float(np.quantile(values, 0.95))
            print(f"{cell:<26} {ref:>4} {len(rows):>4} {covered:>6.3f} "
                  f"{crit:>7.2f} {p95:>7.2f} {p95 / crit:>5.2f}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
