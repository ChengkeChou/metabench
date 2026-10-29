"""How the cost of one UN fit grows with the number of observations.

The UN random-effects term is block diagonal by study: `M = blockdiag(V_s +
Sigma)` with a `J x J` block per study.  The restricted likelihood is therefore
`O(k J^3)` to evaluate, and an implementation that materialises the whole
`kJ x kJ` matrix instead pays `O((kJ)^3)`.  At one outcome per study the two
agree; the whole question is what happens when a meta-analysis has ten
outcomes in a hundred studies, where `kJ = 1000`.

This tool times the public entry point and the block-wise reference on the same
data, so the ratio is a property of the implementation and not of the machine.

    python tools/bench_un_scaling.py
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np

import pimeta as ma
from metabench.reference import UNLikelihood
from metabench.scenarios import get, simulate


def _cell(k: int, J: int) -> str:
    if J == 2:
        return f"covariance/k{k:02d}-J2-rho30" if k == 10 else ""
    if J == 3:
        return {10: "covariance/k10-J3-rho30",
                30: "covariance/k30-J3-rho30"}.get(k, "")
    return {30: "covariance/k30-J5-rho30"}.get(k, "")


def _time(fn, repeats: int = 3) -> float:
    best = float("inf")
    for _ in range(repeats):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def bench(k: int, J: int, reps: int = 5) -> dict[str, Any] | None:
    """One row: pimeta's fit against the block-wise likelihood evaluations."""
    scenario_id = _cell(k, J)
    if not scenario_id:
        return None
    scenario = get(scenario_id)
    rng = np.random.default_rng(4242)
    data = simulate(scenario, rng)

    def pimeta_fit():
        return ma.rma_mv(data.yi, V=data.V, mods=data.X, data=data.frame,
                         random="~ outcome | study", struct="UN",
                         intercept=False)

    fit = pimeta_fit()
    G = np.asarray(fit.g_matrix()[0], dtype=float)
    likelihood = UNLikelihood(data.yi, data.V, data.X)

    def reference_eval():
        for _ in range(100):
            likelihood.nll(G)

    fit_seconds = _time(pimeta_fit, repeats=max(1, reps // 2))
    eval_seconds = _time(reference_eval, repeats=2) / 100.0
    return {
        "k": k,
        "J": J,
        "kJ": k * J,
        "params": J * (J + 1) // 2,
        "fit_s": fit_seconds,
        "eval_ms": 1000 * eval_seconds,
    }


def main() -> int:
    cells = [(10, 2), (10, 3), (30, 3), (30, 5)]
    print(f"{'k':>4}{'J':>3}{'kJ':>5}{'params':>7}{'fit (s)':>10}"
          f"{'eval (ms)':>11}{'block evals per fit':>21}")
    for k, J in cells:
        row = bench(k, J)
        if row is None:
            continue
        implied = row["fit_s"] / max(row["eval_ms"], 1e-9) * 1000
        print(f"{row['k']:>4}{row['J']:>3}{row['kJ']:>5}{row['params']:>7}"
              f"{row['fit_s']:>10.3f}{row['eval_ms']:>11.3f}"
              f"{implied:>21.0f}")
    print()
    print("`fit (s)` is one call to rma_mv.  `eval (ms)` is one block-wise "
          "evaluation of the same likelihood.")
    print("The last column is how many of those evaluations one fit costs: "
          "an L-BFGS-B run needs hundreds, not tens of thousands.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
