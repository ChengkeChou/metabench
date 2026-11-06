"""Per-replicate anatomy of the PI misses: the studentised ratio.

Coverage is the share of ``|theta_new - mu_hat| / half_width <= 1``.  This prints
the quantiles of that ratio and of both of its parts, which is what distinguishes
"the interval is too narrow by a constant" from "a few replicates blow up".
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(r"D:\autometa\metabench")
CELLS = ("k05-tau004", "k10-tau004", "k20-tau004", "k50-tau004")


def load(cell: str) -> list[dict]:
    folder = ROOT / "results" / f"univariate-pi_{cell}" / "pimeta"
    manifests = [(json.loads(p.read_text(encoding="utf-8"))["reps"], p)
                 for p in sorted(folder.glob("manifest_*.json"))]
    _reps, path = max(manifests)
    manifest = json.loads(path.read_text(encoding="utf-8"))
    rows = []
    with (folder / manifest["raw"]).open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return [r for r in rows if r.get("failure") is None]


def q(values, name: str) -> str:
    arr = np.asarray(values, dtype=float)
    qs = np.quantile(arr, [0.05, 0.5, 0.9, 0.95, 0.99])
    return (f"{name:<12} p05 {qs[0]:>7.3f}  p50 {qs[1]:>7.3f}  "
            f"p90 {qs[2]:>7.3f}  p95 {qs[3]:>7.3f}  p99 {qs[4]:>7.3f}")


def main() -> int:
    for cell in CELLS:
        rows = load(cell)
        if not rows:
            continue
        dev = np.array([r["truth_new"] - r["estimate"] for r in rows])
        half = np.array([(r["pi_ub"] - r["pi_lb"]) / 2 for r in rows])
        tau2 = np.array([r["tau2"] for r in rows])
        se = np.array([r["se"] for r in rows])
        ratio = np.abs(dev) / half
        print(f"\n{cell}  n={len(rows)}  coverage={np.mean(ratio <= 1):.3f}")
        print(q(np.abs(dev), "|dev|"))
        print(q(half, "half_width"))
        print(q(ratio, "|dev|/half"))
        print(q(tau2, "tau2_hat"))
        print(q(se, "se(mu)"))
        print(f"corr(tau2, half) {np.corrcoef(tau2, half)[0, 1]:.3f}   "
              f"corr(|dev|, half) {np.corrcoef(np.abs(dev), half)[0, 1]:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
