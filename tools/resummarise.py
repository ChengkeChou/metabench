"""Recompute the summaries of a results directory from its raw records.

The runner is dumb on purpose: it writes one JSON line per replicate and a
summary beside it, and every decision about what the numbers *mean* lives in
``metabench.metrics``.  That separation is what makes this tool possible -- a
metric can be corrected, and every run made before the correction can be
re-read with it, without re-simulating anything.

    python tools/resummarise.py results_v116 results_v117p

Each ``raw_*.jsonl`` gets a fresh ``summary_<same timestamp>.json`` written next
to it, so ``compare_runs.py`` and ``reports/table.py`` pick up the corrected
numbers and the old file stays on disk for comparison.
"""

from __future__ import annotations

import json
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from metabench.experiments.run import _summarise_for  # noqa: E402
from metabench.scenarios import get as get_scenario  # noqa: E402


def _manifest_duration(path: pathlib.Path) -> float | None:
    """The wall-clock duration the runner recorded, if the manifest is there."""
    manifest = path.with_name(
        path.name.replace("raw_", "manifest_").replace(".jsonl", ".json"))
    if not manifest.exists():
        return None
    try:
        return float(json.loads(manifest.read_text(encoding="utf-8"))["duration_s"])
    except (KeyError, ValueError):
        return None


def resummarise(path: pathlib.Path) -> pathlib.Path:
    records = [json.loads(line) for line in
               path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise SystemExit(f"{path}: no records")
    first = records[0]
    scenario = get_scenario(first["scenario_id"])
    duration = _manifest_duration(path)
    summary = {
        "scenario": scenario.to_dict(),
        "adapter": first.get("method_id"),
        "adapter_version": first.get("software_version"),
        "seed": first.get("seed"),
        "level": first.get("level"),
        "duration_s": (duration if duration is not None
                       else sum(float(r.get("runtime") or 0.0)
                                for r in records)),
        "metrics": _summarise_for(scenario.family, records),
    }
    out = path.with_name(path.stem.replace("raw_", "summary_") + ".json")
    out.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    return out


def main(argv: list[str]) -> int:
    roots = [pathlib.Path(a) for a in argv[1:]] or [pathlib.Path("results")]
    total = 0
    for root in roots:
        for raw in sorted(root.rglob("raw_*.jsonl")):
            out = resummarise(raw)
            print(f"{raw} -> {out.name}")
            total += 1
    print(f"{total} summaries rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
