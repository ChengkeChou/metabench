"""Evaluate candidate small-sample rules for the univariate prediction interval.

The raw records hold ``estimate``, ``se``, ``tau2``, the two intervals and the
truth, so a rule that only rescales the interval can be evaluated without
refitting.  Rules are written as functions of the record and ``k``, evaluated
on one seed and then (by the caller) confirmed on a different seed, because a
rule chosen and validated on the same replicates is not validated at all.

Each rule reports coverage and the mean half-width relative to the shipped
interval, so a rule that buys coverage with width is visible as such.
"""

from __future__ import annotations

import json
import math
import pathlib
import sys

import numpy as np
from scipy import stats

ROOT = pathlib.Path(__file__).resolve().parents[1]
CELLS = ("k05-tau004", "k05-tau013", "k10-tau004", "k10-tau013",
         "k20-tau004", "k20-tau013", "k50-tau004", "k50-tau013",
         "k100-tau004", "k100-tau013")


def load(cell: str, seed: int | None = None) -> list[dict]:
    folder = ROOT / "results" / f"univariate-pi_{cell}" / "pimeta"
    manifests = []
    for path in sorted(folder.glob("manifest_*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if seed is None or manifest["seed"] == seed:
            manifests.append((manifest["reps"], path, manifest))
    if not manifests:
        return []
    _reps, path, manifest = max(manifests, key=lambda item: item[0])
    rows: list[dict] = []
    with (folder / manifest["raw"]).open(encoding="utf-8") as handle:
        rows.extend(json.loads(line) for line in handle if line.strip())
    return [r for r in rows if r.get("failure") is None]


def half_width_status(row: dict, k: int) -> float:
    """The shipped interval's half-width, recovered from its bounds."""
    return (row["pi_ub"] - row["pi_lb"]) / 2.0


def scale_status(row: dict, k: int) -> float:
    return half_width_status(row, k) / float(stats.t.ppf(0.975, k - 1))


def rule_status(row: dict, k: int) -> float:
    return float(stats.t.ppf(0.975, k - 1)) * scale_status(row, k)


def rule_t_k2(row: dict, k: int) -> float:
    return float(stats.t.ppf(0.975, k - 2)) * scale_status(row, k)


def rule_inflate_25(row: dict, k: int) -> float:
    return float(stats.t.ppf(0.975, k - 1)) * scale_status(row, k) * (1 + 2.5 / k)


def rule_tau2_kp(row: dict, k: int) -> float:
    """Re-scale the heterogeneity variance by k/(k-2), the classic df fix."""
    tau2 = float(row["tau2"])
    se = float(row["se"])
    s = math.sqrt(tau2 * k / (k - 2) + se * se) if k > 2 else float("nan")
    return float(stats.t.ppf(0.975, k - 2)) * s


def rule_combo(row: dict, k: int) -> float:
    """Both textbook parts: t_{k-2} and the k/(k-2) variance rescale."""
    tau2 = float(row["tau2"])
    se = float(row["se"])
    s = math.sqrt(tau2 * k / (k - 2) + se * se) if k > 2 else float("nan")
    return float(stats.t.ppf(0.975, k - 1)) * s


RULES = {
    "status": rule_status,
    "t_k-2": rule_t_k2,
    "x(1+2.5/k)": rule_inflate_25,
    "tau2*k/(k-2)+t_k-2": rule_tau2_kp,
    "tau2*k/(k-2)+t_k-1": rule_combo,
}


def main(argv: list[str]) -> int:
    seed = int(argv[0]) if argv else None
    print(f"seed = {seed if seed is not None else 'largest run'}")
    header = f"{'cell':<12} {'n':>5} " + " ".join(f"{name:>20}" for name in RULES)
    print(header)
    totals = {name: [] for name in RULES}
    widths = {name: [] for name in RULES}
    for cell in CELLS:
        rows = load(cell, seed)
        if not rows:
            continue
        k = int(cell.split("-")[0][1:])
        parts = []
        base = np.mean([half_width_status(r, k) for r in rows])
        for name, rule in RULES.items():
            cover, width = [], []
            for r in rows:
                half = rule(r, k)
                if not np.isfinite(half):
                    continue
                cover.append(abs(r["truth_new"] - r["estimate"]) <= half)
                width.append(half)
            p = float(np.mean(cover))
            totals[name].append(p)
            widths[name].append(float(np.mean(width)) / base)
            parts.append(f"{p:>13.3f} ({np.mean(width) / base:>4.2f})")
        print(f"{cell:<12} {len(rows):>5} " + " ".join(parts))
    print()
    for name in RULES:
        if totals[name]:
            print(f"{name:<22} mean coverage {np.mean(totals[name]):.3f}   "
                  f"mean width x {np.mean(widths[name]):.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
