"""Compare the shipped prediction interval with the upper-bound rules.

Reads one run per (cell, adapter), so screening and formal runs never mix, and
prints coverage with its Monte Carlo half-width next to the mean width relative
to the shipped interval — a rule that buys coverage with width is visible.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
CELLS = ("k05-tau004", "k05-tau013", "k10-tau004", "k10-tau013",
         "k20-tau004", "k20-tau013", "k50-tau004", "k50-tau013",
         "k100-tau004", "k100-tau013")
ADAPTERS = (("pimeta", ""), ("pimeta-pi", "upper95"), ("pimeta-pi", "upper80"),
            ("pimeta-pi", "upper95t"))


def load(cell: str, adapter: str, rule: str) -> list[dict]:
    folder = ROOT / "results" / f"univariate-pi_{cell}" / adapter
    if rule:
        folder = folder / rule
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
    print(f"{'cell':<12} " + " ".join(f"{rule or 'shipped':>20}"
                                      for _a, rule in ADAPTERS))
    for cell in CELLS:
        parts = []
        base = None
        for adapter, rule in ADAPTERS:
            rows = load(cell, adapter, rule)
            if not rows:
                parts.append(f"{'--':>20}")
                continue
            covered = [r["pi_lb"] <= r["truth_new"] <= r["pi_ub"] for r in rows]
            width = np.mean([r["pi_ub"] - r["pi_lb"] for r in rows])
            if base is None:
                base = width
            half = 1.96 * float(np.sqrt(np.mean(covered) * (1 - np.mean(covered))
                                        / len(covered)))
            parts.append(f"{np.mean(covered):>12.3f} +-{half:>4.3f} {width / base:>4.2f}")
        print(f"{cell:<12} " + " ".join(parts))
    print("\ncolumns are: coverage +- MC half-width, then mean width relative to "
          "the shipped interval")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
