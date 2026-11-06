"""Coverage against volume for the joint region, from the largest run per cell.

A region that covers is not automatically a good region: it can cover by being
enormous.  This prints coverage next to the volume proxy
``crit**(J/2) * det(cov)**0.5`` relative to the chi-square region's, so the trade
is visible on one line instead of two tables.
"""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(r"D:\autometa\metabench")
CELLS = ("k05-rho30", "k10-rho30", "k20-rho30", "k50-rho30")
REFS = ("chi2", "f", "hk", "bootstrap")


def best_summary(cell: str, ref: str) -> dict | None:
    folder = ROOT / "results" / f"multivariate_{cell}" / "pimeta-region" / ref
    manifests = [(json.loads(p.read_text(encoding="utf-8"))["reps"], p)
                 for p in sorted(folder.glob("manifest_*.json"))]
    if not manifests:
        return None
    _reps, path = max(manifests)
    summary = folder / path.name.replace("manifest_", "summary_")
    if not summary.exists():
        return None
    data = json.loads(summary.read_text(encoding="utf-8"))
    data["_reps"] = _reps
    return data


def main() -> int:
    print(f"{'cell':<12} {'ref':>10} {'reps':>5} {'coverage':>9} {'+-':>6} "
          f"{'volume':>7} {'vol x':>6}")
    for cell in CELLS:
        base = None
        for ref in REFS:
            data = best_summary(cell, ref)
            if data is None:
                print(f"{cell:<12} {ref:>10} {'--':>5}")
                continue
            metrics = data["metrics"]
            cov = metrics["joint_new"]
            vol = metrics["volume"]
            if base is None:
                base = vol
            p = cov["estimate"]
            half = 1.96 * (p * (1 - p) / max(cov["reps_used"], 1)) ** 0.5
            print(f"{cell:<12} {ref:>10} {data['_reps']:>5} {p:>9.3f} "
                  f"{half:>6.3f} {vol:>7.3f} {vol / base:>6.2f}")
        print()
    print("volume is crit**(J/2)*sqrt(det(cov)): monotone in the ellipsoid's "
          "volume, and 'vol x' is relative to the chi-square region of the same "
          "cell")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
