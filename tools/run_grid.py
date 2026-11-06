"""Run a grid of (scenario, adapter) cells in parallel processes.

Each cell is an independent ``metabench.experiments.run`` call, so the grid is
embarrassingly parallel and each cell keeps its own seed and its own raw file.
The point of a formal run is replicate count, not elegance: 2000 replicates make
the Monte Carlo half-width 0.0096 at a nominal 0.95, which is what it takes to
call a 0.94 from a 0.95.

Usage::

    python tools/run_grid.py pi --reps 2000
    python tools/run_grid.py region --reps 2000 --workers 6
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from metabench.experiments.run import run  # noqa: E402


def pi_cells() -> list[tuple[str, str]]:
    cells = []
    for k in ("k05", "k10", "k20", "k50", "k100"):
        for tau in ("tau004", "tau013"):
            cells.append((f"univariate-pi/{k}-{tau}", "pimeta"))
    return cells


def region_cells() -> list[tuple[str, str]]:
    cells = []
    for k in ("k05", "k10", "k20"):
        for adapter in ("chi2", "f", "hk"):
            cells.append((f"multivariate/{k}-rho30", f"pimeta-region/{adapter}"))
    return cells


def boot_cells() -> list[tuple[str, str]]:
    """The bootstrap cells: ``n_boot`` refits per replicate, so few replicates."""
    boot = os.environ.get("METABENCH_NBOOT", "200")
    return [(f"multivariate/{k}-rho30", f"pimeta-region/bootstrap@{boot}")
            for k in ("k05", "k10", "k20")]


def pi_rule_cells() -> list[tuple[str, str]]:
    """The candidate prediction-interval rules, on the same ten cells."""
    cells = []
    for k in ("k05", "k10", "k20", "k50", "k100"):
        for tau in ("tau004", "tau013"):
            for rule in ("upper95", "upper80", "upper95t", "blend20"):
                cells.append((f"univariate-pi/{k}-{tau}", f"pimeta-pi/{rule}"))
    return cells


def extended_cells() -> list[tuple[str, str]]:
    """J = 3 and J = 5, plus the rho = 0.7 cells at a formal replicate count."""
    cells = []
    for scenario in ("k10-J3-rho30", "k20-J3-rho30", "k20-J5-rho30",
                     "k10-rho70", "k20-rho70"):
        for adapter in ("chi2", "f", "hk"):
            cells.append((f"multivariate/{scenario}", f"pimeta-region/{adapter}"))
    return cells


def _one(job) -> tuple[str, str, str]:
    scenario, adapter, reps, seed, out = job
    path = run(scenario, adapter_name=adapter, reps=reps, seed=seed,
               out=pathlib.Path(out))
    return scenario, adapter, str(path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("grid", choices=("pi", "region", "boot", "pi-rules",
                                        "extended"))
    parser.add_argument("--reps", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=20261106)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default=str(ROOT / "results"))
    args = parser.parse_args(argv)

    cells = {"pi": pi_cells, "region": region_cells, "boot": boot_cells,
             "pi-rules": pi_rule_cells, "extended": extended_cells}[args.grid]()
    jobs = [(scenario, adapter, args.reps, args.seed, args.out)
            for scenario, adapter in cells]
    with mp.Pool(processes=args.workers) as pool:
        for scenario, adapter, path in pool.imap_unordered(_one, jobs):
            print(f"done {scenario} / {adapter} -> {path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
