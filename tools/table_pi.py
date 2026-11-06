"""Tabulate the prediction-interval sweep from the raw summaries."""

from __future__ import annotations

import json
import pathlib

ROOT = pathlib.Path(__file__).resolve().parents[1]

rows = []
for folder in sorted((ROOT / "results").glob("univariate-pi_*")):
    run_dir = folder / "pimeta"
    manifests = sorted(run_dir.glob("manifest_*.json"))
    if not manifests:
        continue
    best = max(manifests,
               key=lambda p: json.loads(p.read_text(encoding="utf-8"))["reps"])
    manifest = json.loads(best.read_text(encoding="utf-8"))
    summary_path = run_dir / best.name.replace("manifest_", "summary_")
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    metrics = data["metrics"]
    dgm = data["scenario"]["dgm"]
    rows.append((
        int(dgm["k"]), float(dgm["tau2"]),
        metrics["pi"]["estimate"], metrics["pi"]["halfwidth_95"],
        metrics["pi_t"]["estimate"], metrics["pi_t"]["halfwidth_95"],
        metrics["ci"]["estimate"], metrics["rates"]["boundary_rate"],
        metrics["pi_width"], metrics["pi_t_width"], manifest["reps"],
    ))

rows.sort()
header = (f"{'k':>4} {'tau2':>5} {'n':>5} {'PI':>7} {'+-':>6} {'PIt':>7} {'+-':>6} "
          f"{'CI':>6} {'bnd':>6} {'wPI':>7} {'wPIt':>7}")
print(header)
for r in rows:
    print(f"{r[0]:>4} {r[1]:>5.2f} {r[10]:>5} {r[2]:>7.3f} {r[3]:>6.3f} {r[4]:>7.3f} "
          f"{r[5]:>6.3f} {r[6]:>6.3f} {r[7]:>6.3f} {r[8]:>7.3f} {r[9]:>7.3f}")
