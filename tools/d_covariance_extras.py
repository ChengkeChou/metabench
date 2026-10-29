"""The family-D quantities that a one-line-per-cell table cannot hold.

Three of them, each answering a question the summary columns only hint at:

* **per component** -- a mean relative bias hides which variance component is
  wrong, and when one true component is 100x smaller than the sampling variance
  the ratio is not even a stable statistic.  What matters there is where the
  estimate actually sits.
* **reported vs found** -- `ll_library` is the likelihood the library reports
  for its own answer; `ll_at_fit` is the reference likelihood at the matrix the
  library returned.  If they differ, the answer reported is not the answer
  found, and every coverage number downstream is measuring the wrong matrix.
* **how often the answer is singular** -- `max |rho| = 1` exactly, or a
  variance at zero, is a boundary answer that the likelihood may or may not
  justify.

    python tools/d_covariance_extras.py results
"""

from __future__ import annotations

import glob
import json
import pathlib
import sys
from typing import Any

import numpy as np


def load(cell_dir: str) -> list[dict[str, Any]]:
    files = sorted(glob.glob(str(pathlib.Path(cell_dir) / "raw_*.jsonl")))
    if not files:
        return []
    with open(files[-1], encoding="utf-8") as handle:
        return [json.loads(line) for line in handle]


def per_component(rows: list[dict], label: str) -> None:
    if not rows:
        return
    truth = np.diag(np.asarray(rows[0]["truth_cov"], dtype=float))
    hat = np.asarray([r["extra"]["tau2_hat"] for r in rows], dtype=float)
    print(f"\n{label}: tau2 by component (truth {np.round(truth, 4)})")
    print(f"{'component':>10}{'truth':>9}{'median':>10}{'q10':>9}{'q90':>9}"
          f"{'share=0':>9}")
    for j in range(hat.shape[1]):
        column = hat[:, j]
        print(f"{j:>10}{truth[j]:>9.4f}{np.median(column):>10.4f}"
              f"{np.quantile(column, 0.1):>9.4f}"
              f"{np.quantile(column, 0.9):>9.4f}"
              f"{100 * np.mean(column <= 1e-10):>8.1f}%")


def reported_vs_found(rows: list[dict], label: str) -> None:
    if not rows:
        return
    difference = np.asarray([
        r["extra"]["ll_library"] - r["extra"]["ll_at_fit"] for r in rows])
    gap = np.asarray([r["extra"]["gap"] for r in rows])
    worst = np.argsort(-gap)[:3]
    print(f"\n{label}: reported answer vs found answer")
    print(f"  ll_library - ll_at_reported: max {difference.max():.3g}, "
          f"share > 1e-6: {100 * np.mean(difference > 1e-6):.1f}%")
    print(f"  gap > 1e-6: {int(np.sum(gap > 1e-6))} of {len(gap)}   "
          f"max {gap.max():.3f}")
    for index in worst:
        if gap[index] <= 1e-6:
            continue
        row = rows[int(index)]
        print(f"    replicate {row['replicate_id']}: gap {gap[index]:.4f}, "
              f"converged={row['converged']}, "
              f"boundary={row['boundary_hit']}, "
              f"warnings={row['warnings']}")


def singularity(rows: list[dict], label: str) -> None:
    if not rows:
        return
    rho = np.asarray([np.max(np.abs(r["extra"]["rho_hat"])) for r in rows])
    tau2 = np.asarray([np.min(r["extra"]["tau2_hat"]) for r in rows])
    min_eig = np.asarray([r["extra"]["min_eig_hat"] for r in rows])
    print(f"\n{label}: boundary answers")
    print(f"  max|rho| = 1: {100 * np.mean(rho >= 1 - 1e-9):.1f}%   "
          f"min tau2 = 0: {100 * np.mean(tau2 <= 1e-10):.1f}%   "
          f"non-PD: {100 * np.mean(min_eig <= 0):.1f}%")
    print(f"  max|rho| median {np.median(rho):.3f}, "
          f"min eigenvalue median {np.median(min_eig):.4f}")


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    root = pathlib.Path(argv[0] if argv else "results")
    for directory in sorted(root.glob("covariance_*")):
        for adapter_dir in sorted(directory.glob("pimeta-cov/*")):
            rows = load(str(adapter_dir))
            label = f"{directory.name}/{adapter_dir.name}"
            per_component(rows, label)
            reported_vs_found(rows, label)
            singularity(rows, label)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
