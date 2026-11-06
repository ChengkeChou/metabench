"""Coverage of the scaled F reference on the validation seed.

The multiplier for each cell was read off the formal run (seed A); this prints
the coverage it delivers on seed B, so the rule is checked where it was not
chosen.  A rule that only exists because it was tuned to one seed will not
reproduce here.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

ROOT = pathlib.Path(r"D:\autometa\metabench")
#: (cell, reference, multiplier) — each multiplier was read off the formal run
#: (seed A) as ``required / crit`` and is applied unchanged here (seed B)
CELLS = (("k05-rho30", "f", 2.30), ("k10-rho30", "f", 4.39),
         ("k20-rho30", "f", 4.13), ("k50-rho30", "f", 1.46),
         ("k10-rho30", "hk", 2.11), ("k20-rho30", "hk", 2.37),
         ("k50-rho30", "hk", 1.29),
         # beyond J = 2: the multiplier has to be re-measured per design
         ("k10-J3-rho30", "hk", 1.53), ("k10-rho70", "hk", 1.59),
         ("k20-J5-rho30", "hk", 1.48))


def load(cell: str, adapter: str, seed: int) -> list[dict]:
    folder = ROOT / "results" / f"multivariate_{cell}" / "pimeta-region" / adapter
    for path in sorted(folder.glob("manifest_*.json")):
        manifest = json.loads(path.read_text(encoding="utf-8"))
        if manifest["seed"] != seed:
            continue
        with (folder / manifest["raw"]).open(encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        return [r for r in rows if r.get("failure") is None]
    return []


def covered(row: dict, key: str = "truth_new_vec") -> bool:
    centre = np.asarray(row["region_center"], dtype=float)
    cov = np.asarray(row["region_cov"], dtype=float)
    delta = np.asarray(row[key], dtype=float) - centre
    try:
        return bool(delta @ np.linalg.solve(cov, delta) <= row["region_crit"])
    except np.linalg.LinAlgError:
        return False


def main() -> int:
    print(f"{'cell':<12} {'base':>5} {'scale':>6} {'coverage':>9} {'+-':>6} "
          f"{'volume':>7} {'vol x chi2':>10}")
    for cell, base, scale in CELLS:
        rows = load(cell, f"{base}@{scale:g}", 20261107)
        if not rows:
            print(f"{cell:<12} {base:>5} {scale:>6.2f} {'--':>9}")
            continue
        hits = [covered(r) for r in rows]
        p = float(np.mean(hits))
        half = 1.96 * float(np.sqrt(p * (1 - p) / len(hits)))
        volume = float(np.mean([
            r["region_crit"] ** (len(r["region_center"]) / 2.0)
            * float(np.sqrt(np.linalg.det(np.asarray(r["region_cov"], dtype=float))))
            for r in rows]))
        reference = load(cell, "chi2", 20261106)
        base_volume = float(np.mean([
            r["region_crit"] ** (len(r["region_center"]) / 2.0)
            * float(np.sqrt(np.linalg.det(np.asarray(r["region_cov"], dtype=float))))
            for r in reference])) if reference else float("nan")
        print(f"{cell:<12} {base:>5} {scale:>6.2f} {p:>9.3f} {half:>6.3f} "
              f"{volume:>7.3f} {volume / base_volume:>10.1f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
