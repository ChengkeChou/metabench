"""Run one scenario against one adapter, and write the raw records.

The runner is deliberately dumb: simulate, fit, record, repeat.  Everything
that decides what the numbers *mean* lives in :mod:`metabench.metrics`, so a
metric can be recomputed later from the raw JSONL without re-running anything.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import platform
import sys
import time
from typing import Any

import numpy as np

from .. import __version__ as metabench_version
from ..adapters import get as get_adapter
from ..metrics import summarise
from ..scenarios import get as get_scenario
from ..scenarios import simulate


def environment() -> dict[str, Any]:
    """Versions and platform, recorded in every manifest."""
    import scipy

    try:
        import pimeta
        pimeta_version = pimeta.__version__
    except Exception:                                          # noqa: BLE001
        pimeta_version = None
    return {
        "metabench": metabench_version,
        "pimeta": pimeta_version,
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": __import__("os").cpu_count(),
    }


def run(scenario_id: str, *, adapter_name: str, reps: int, seed: int,
        level: float = 0.95, out: pathlib.Path) -> pathlib.Path:
    scenario = get_scenario(scenario_id)
    adapter = get_adapter(adapter_name)
    rng = np.random.default_rng(seed)

    started = time.time()
    records: list[dict[str, Any]] = []
    for replicate in range(reps):
        data = simulate(scenario, rng)
        t0 = time.perf_counter()
        fit = adapter.fit(data, level=level)
        runtime = time.perf_counter() - t0
        row = {
            "scenario_id": scenario.scenario_id,
            "scenario_digest": scenario.digest,
            "replicate_id": replicate,
            "method_id": f"{adapter.name}/{adapter.__class__.__name__}",
            "software_version": adapter.version,
            "seed": seed,
            "level": level,
            "truth": data.mu,
            "truth_new": data.theta_new,
            "runtime": runtime,
            **fit.to_dict(),
        }
        records.append(row)
    duration = time.time() - started

    run_dir = out / scenario.scenario_id.replace("/", "_") / adapter.name
    run_dir.mkdir(parents=True, exist_ok=True)
    stamp = time.strftime("%Y%m%dT%H%M%S")
    raw_path = run_dir / f"raw_{stamp}.jsonl"
    with raw_path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in records:
            handle.write(json.dumps(row, sort_keys=True) + "\n")

    summary = {
        "scenario": scenario.to_dict(),
        "adapter": adapter.name,
        "adapter_version": adapter.version,
        "seed": seed,
        "level": level,
        "duration_s": duration,
        "metrics": summarise(records),
    }
    (run_dir / f"summary_{stamp}.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (run_dir / f"manifest_{stamp}.json").write_text(
        json.dumps({"environment": environment(), "scenario_digest": scenario.digest,
                    "reps": reps, "seed": seed, "level": level,
                    "raw": raw_path.name, "duration_s": duration},
                   indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return raw_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--scenario", required=True,
                        help="scenario id, e.g. univariate/null")
    parser.add_argument("--adapter", default="pimeta")
    parser.add_argument("--reps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20261029)
    parser.add_argument("--level", type=float, default=0.95)
    parser.add_argument("--out", default="results")
    args = parser.parse_args(argv)

    raw = run(args.scenario, adapter_name=args.adapter, reps=args.reps,
              seed=args.seed, level=args.level,
              out=pathlib.Path(args.out))
    print(f"wrote {raw}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
