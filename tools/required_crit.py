"""What critical value would have calibrated the region?  Read from raw records.

Coverage under *any* reference can be recomputed from what a raw record already
holds, because the region is an ellipsoid and containment is a single number:

    D2 = (truth - centre)' cov^-1 (truth - centre),   covered  <=>  D2 <= crit

so changing ``crit`` (or scaling ``cov`` by ``c``, which divides ``D2`` by ``c``)
never needs a refit.  This script prints, per cell, the empirical 95 % quantile
of ``D2`` and divides it by the critical value each shipped reference used.  The
ratio is the inflation that reference would need to be calibrated — the quantity
a correction has to reproduce as a function of ``k``.

Run: ``python tools/required_crit.py`` from the metabench root.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(r"D:\autometa\metabench")
CELLS = ("k05-rho30", "k10-rho30", "k10-rho70", "k20-rho30", "k20-rho70",
         "k50-rho30")
REFERENCES = ("chi2", "f", "hk")


def load(cell: str, ref: str) -> list[dict]:
    folder = ROOT / "results" / f"multivariate_{cell}" / "pimeta-region" / ref
    manifests = [(json.loads(p.read_text(encoding="utf-8"))["reps"], p)
                 for p in sorted(folder.glob("manifest_*.json"))]
    if not manifests:
        return []
    _reps, path = max(manifests)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows: list[dict] = []
    with (folder / manifest["raw"]).open(encoding="utf-8") as handle:
        rows.extend(json.loads(line) for line in handle if line.strip())
    return [r for r in rows if r.get("failure") is None]


def mahalanobis(row: dict, target: str = "truth_new_vec") -> float:
    centre = np.asarray(row["region_center"], dtype=float)
    cov = np.asarray(row["region_cov"], dtype=float)
    truth = np.asarray(row[target], dtype=float)
    delta = truth - centre
    try:
        return float(delta @ np.linalg.solve(cov, delta))
    except np.linalg.LinAlgError:
        return float("nan")


def main() -> int:
    print(f"{'cell':<12} {'ref':>5} {'n':>4} {'crit':>7} {'D2 p50':>8} "
          f"{'D2 p95':>8} {'cov':>6} {'need':>6}")
    for cell in CELLS:
        for ref in REFERENCES:
            rows = load(cell, ref)
            if not rows:
                continue
            values = np.array([mahalanobis(r) for r in rows])
            values = values[np.isfinite(values)]
            crit = float(np.median([r["region_crit"] for r in rows]))
            p95 = float(np.quantile(values, 0.95))
            covered = float(np.mean(values <= crit))
            print(f"{cell:<12} {ref:>5} {values.size:>4} {crit:>7.2f} "
                  f"{np.median(values):>8.2f} {p95:>8.2f} {covered:>6.3f} "
                  f"{p95 / crit:>6.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
