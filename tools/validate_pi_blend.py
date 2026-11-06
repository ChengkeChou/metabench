"""Compare the shipped interval with the blended rule on the validation seed.

Two seeds, one protocol: the blend weight was chosen on seed A and this prints
coverage on seed B, so a number that only exists because it was tuned to the
first seed cannot survive.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(r"D:\autometa\metabench")
CELLS = ("k05-tau004", "k05-tau013", "k10-tau004", "k10-tau013",
         "k20-tau004", "k20-tau013", "k50-tau004", "k50-tau013",
         "k100-tau004", "k100-tau013")


def load(cell: str, adapter: str, rule: str, seed: int) -> list[dict]:
    folder = ROOT / "results" / f"univariate-pi_{cell}" / adapter
    if rule:
        folder = folder / rule
    for path in sorted(folder.glob("manifest_*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["seed"] != seed:
            continue
        with (folder / manifest["raw"]).open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        return [r for r in rows if r.get("failure") is None]
    return []


def summarise(rows: list[dict]) -> tuple[float, float, float]:
    covered = [r["pi_lb"] <= r["truth_new"] <= r["pi_ub"] for r in rows]
    p = float(np.mean(covered))
    half = 1.96 * float(np.sqrt(p * (1 - p) / len(covered)))
    width = float(np.mean([r["pi_ub"] - r["pi_lb"] for r in rows]))
    return p, half, width


def main() -> int:
    print(f"{'cell':<12} {'shipped':>18} {'blend20 (validation)':>26} {'width x':>8}")
    for cell in CELLS:
        adhoc = load(cell, "pimeta", "", 20261107)
        blend = load(cell, "pimeta-pi", "blend20", 20261107)
        if not adhoc or not blend:
            continue
        p0, h0, w0 = summarise(adhoc)
        p1, h1, w1 = summarise(blend)
        print(f"{cell:<12} {p0:>10.3f} +-{h0:>5.3f} {p1:>16.3f} +-{h1:>5.3f} "
              f"{w1 / w0:>8.2f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
