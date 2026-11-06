"""Compare the bootstrap critical value against the closed-form references.

Reads every raw record for one scenario and prints, per replicate, the critical
value each reference produced, so the ratio can be summarised rather than
guessed.  Also reports how often the region covered, from the recorded ellipsoid
and the recorded truth — the same arithmetic the metrics do.
"""

from __future__ import annotations

import glob
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]


def load(scenario: str, adapter: str) -> list[dict]:
    pattern = ROOT / "results" / scenario / "pimeta-region" / adapter / "raw_*.jsonl"
    rows: list[dict] = []
    for path in sorted(glob.glob(str(pattern))):
        with pathlib.Path(path).open(encoding="utf-8") as handle:
            rows.extend(json.loads(line) for line in handle if line.strip())
    return rows


def inside(row: dict, target: str) -> bool:
    centre = np.asarray(row["region_center"], dtype=float)
    cov = np.asarray(row["region_cov"], dtype=float)
    truth = np.asarray(row[target], dtype=float)
    delta = truth - centre
    try:
        return bool(delta @ np.linalg.solve(cov, delta) <= row["region_crit"])
    except np.linalg.LinAlgError:
        return False


def main(argv: list[str]) -> int:
    scenario = argv[0] if argv else "multivariate_k10-rho30"
    boot = load(scenario, "bootstrap")
    ref = {name: load(scenario, name) for name in ("chi2", "f", "hk")}
    usable = [r for r in boot if r.get("failure") is None]
    print(f"{scenario}: {len(usable)} usable bootstrap records of {len(boot)}")
    if not usable:
        return 1
    crits = np.array([r["region_crit"] for r in usable], dtype=float)
    print(f"bootstrap crit: min {crits.min():.2f}  median {np.median(crits):.2f} "
          f" max {crits.max():.2f}")
    for name, rows in ref.items():
        if not rows:
            continue
        values = np.array([r["region_crit"] for r in rows
                           if r.get("failure") is None], dtype=float)
        shared = min(len(values), len(crits))
        ratio = crits[:shared] / values[:shared]
        new_cov = sum(inside(r, "truth_new_vec") for r in rows) / len(rows)
        mean_cov = sum(inside(r, "truth_vec") for r in rows) / len(rows)
        print(f"  {name:>4}: crit median {np.median(values):.2f}  "
              f"bootstrap/this = {np.median(ratio):.2f}  "
              f"coverage new {new_cov:.3f} mean {mean_cov:.3f}")
    new_cov = sum(inside(r, "truth_new_vec") for r in usable) / len(usable)
    mean_cov = sum(inside(r, "truth_vec") for r in usable) / len(usable)
    print(f"bootstrap coverage: new {new_cov:.3f}  mean {mean_cov:.3f}  "
          f"({len(usable)} reps)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
