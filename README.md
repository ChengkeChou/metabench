# metabench

A measurement harness for meta-analysis estimators, built to answer one kind of
question: *under which data-generating mechanisms does this method's interval
cover its estimand at the nominal rate, and when it does not, is the cause the
method, the algorithm, or the simulation?*

It is deliberately independent of `pimeta`'s internals: the harness drives
public entry points through **adapters**, so the same scenario can be run
against `pimeta`, `metafor` (through Rscript), or a reference implementation
written for the purpose. A harness that imports the code under test cannot tell
a bug from a shared assumption.

## The four layers, reported separately

| layer | question | example |
|---|---|---|
| 1. reference | does the estimator compute what it says? | analytic REML on a two-study problem; parameter-space and objective checks |
| 2. statistical | does the interval cover, at what width and bias? | coverage + Monte Carlo SE, mean width, bias, empirical SD against the model SE |
| 3. numerical | is the returned solution a solution? | convergence, local optima from multiple starts, positive-definiteness, boundary and identifiability rates |
| 4. cross-software | do two implementations agree, and where do they not? | paired comparisons on identical data, with the failure rate reported |

A layer-1 pass says nothing about layer 2, and layer-2 under-coverage says
nothing about layer 3 correctness. Collapsing them is how "the software is
tested" becomes a claim no single experiment can support.

## Layout

```
metabench/
├── scenarios/     ADEMP definition: data-generating mechanism + estimand
├── adapters/      one module per implementation, public API only
├── reference/     analytic and independent reference computations
├── metrics/       coverage, width, bias, failure and boundary rates
├── experiments/   runners that write raw records and a summary
├── results/       raw JSONL + summaries, one directory per run (never edited)
├── reports/       markdown tables generated from results
└── manifests/     versions, seeds, fingerprints, environment
```

Every run writes, per replicate: `scenario_id`, `replicate_id`, `method_id`,
`software_version`, `seed`, `estimate`, `se`, `ci`, `pi`, `converged`,
`boundary_hit`, `valid_covariance`, `runtime`, `warnings`, `failure_reason`.
Metrics are recomputed **from those records**, never from a previous summary.

## Two rules the metrics enforce

1. **A failed replicate is data, not a missing value.** Coverage is reported
   twice: conditional on a usable fit, and with failures counted as misses.
   Deleting failures from the denominator makes a fragile method look better
   than it is, and it is the most common way a simulation study flatters its
   subject.
2. **Every proportion carries its Monte Carlo standard error.** At the nominal
   0.95, the half-width of a 95 % interval is roughly `1.96·sqrt(.95·.05/N)`:
   0.055 at N=60, 0.025 at N=300, 0.0096 at N=2000, 0.0043 at N=10000. A
   coverage of 0.917 from 60 replicates is compatible with 0.95; the same
   number from 2000 is not.

## Two-stage simulation design

* **screening**: 300–500 replicates per scenario, to find the regions that
  break and to time the expensive ones;
* **formal**: at least 2000 replicates for the scenarios that matter, 5000 or
  10000 where two methods have to be separated.

The judgement is never "is the point estimate 0.95", but "is the deviation
larger than the Monte Carlo error and than the pre-specified tolerance".

## Validation status per method

A method is not `pass`/`fail`; it is one of

* **validated** — reference and objective checks pass, the pre-specified
  statistical criteria are met, and the failure rate is reported;
* **conditional** — acceptable inside a stated region (for example `k >= 20`)
  and limited outside it;
* **unvalidated** — reference answers are insufficient, coverage is untested,
  or the result depends on the starting point.

`pimeta`'s statistical reliability report is generated from these statuses.

## Running

```bash
python -m metabench.experiments.run --scenario univariate/null --reps 300 \
    --adapter pimeta --seed 20261029 --out results
python -m metabench.reports.table results --match univariate

# a whole grid, one process per cell
python tools/run_grid.py pi        --reps 2000 --workers 5    # family C
python tools/run_grid.py region    --reps 2000 --workers 6    # family B
METABENCH_NBOOT=200 python tools/run_grid.py boot --reps 100  # family B, bootstrap
python tools/run_grid.py pi-rules  --reps 2000 --workers 4    # candidate corrections
```

Python needs `src/pimeta` (or an installed pimeta) and the metabench root on the
path: `PYTHONPATH=<path-to-pimeta>/src:<path-to-metabench>`.

## Families

| id | estimand | adapters | report |
|---|---|---|---|
| `univariate/*` | the mean effect, and a new study's true effect | `pimeta` | — |
| `univariate-pi/*` | a new study's **true** effect only | `pimeta`, `pimeta-pi/{upper95,upper80,upper95t,blend20}` | `C_prediction_interval.md` |
| `multivariate/*` | the mean vector, and a new study's effect vector | `pimeta-region/{chi2,f,hk,bootstrap}` | `B_joint_region.md` |
| `covariance/*` | the whole between-study covariance | `pimeta-cov/*` | `D_covariance.md` |

## Reports

Each report is generated from a run's raw records and states its own
replicate count, seed and Monte Carlo error. Read them in this order.

| report | what it measures | conclusion |
|---|---|---|
| `C_prediction_interval.md` | the univariate prediction interval `pimeta.predict()` returns, against the true effect of a new study | coverage 0.797 / 0.836 / 0.888 / 0.919 / 0.931 at k = 5/10/20/50/100 from 2000 replicates each. The shipped interval under-covers, and the width is uncorrelated with the error it has to cover (0.03) while being almost perfectly correlated with `tau^2_hat` (0.97) — it inherits the heterogeneity estimate, it does not hedge it |
| `B_joint_region.md` | the joint region, screening stage, 500 replicates | finds which cells break; superseded in part by B2 |
| `B2_joint_region_formal.md` | the same cells at 2000 replicates, plus a parametric bootstrap and the critical multiplier that would have calibrated each reference | chi-squared 0.670/0.690/0.768/0.870; Hotelling 0.825/0.741/0.793/0.877; HK 0.972/0.863/0.846/0.893. The mean region is calibrated (0.96–1.00), so the defect is in the prediction step, not in the fit |
| `D_covariance.md` | whether `rma_mv`'s unstructured `Sigma` is a covariance at all, and what it costs | boundary and indefiniteness rates, the semantic split between "not positive definite" and "on the boundary", and runtimes per fit |

A multiplier that repairs the region is **not** a function of `k` alone —
2.11 at `k=10, J=2, rho=.3` but 1.59 at `rho=.7` and 1.53 at `J=3` — which is
why `pimeta` refuses to apply one to a `J` that was never measured. The
per-cell numbers are in `B2`; the shipped calibration is in
`pimeta.src.pimeta.mv.region.MEASURED_CRIT_SCALE`.

## Tools

| tool | question it answers |
|---|---|
| `tools/run_grid.py` | run many cells in parallel, one raw file each |
| `tools/table_pi.py` | coverage table for the prediction-interval sweep |
| `tools/diagnose_pi.py` | is the interval too narrow on average, or conditionally? |
| `tools/anatomy_pi.py` | per-replicate: does the width track the error it must cover? |
| `tools/pi_rules.py` | evaluate rescaling rules without refitting |
| `tools/pi_blend.py` | sweep the blend weight between `tau^2_hat` and its upper bound |
| `tools/compare_pi_rules.py` | the shipped interval against each rule, coverage and width |
| `tools/validate_pi_blend.py` | the same comparison on the *validation* seed |
| `tools/required_crit.py` | what critical value would have calibrated each region |
| `tools/compare_crit.py` | bootstrap critical value against the closed-form ones |
| `tools/multipliers.py` | required multiplier per cell, and whether it is a function of `k` |
| `tools/coverage_volume.py` | coverage bought per unit of region volume |
| `tools/validate_region_scale.py` | the multiplier on the *validation* seed |
| `tools/probe_edf_coverage.py` | does an effective-df reference beat the tabulated multiplier? |

## One protocol worth naming

A tuning constant (a blend weight, an effective df) is chosen on one seed and
then **re-run on a seed that was never used for the choice**. `pi_blend.py` picks,
`validate_pi_blend.py` confirms. A rule that only works on the seed it was tuned
to does not survive that step, and that step is cheap — which is the argument for
doing it every time.

## Requirements

Python 3.10+, `numpy`, `scipy`; `pimeta` on the path (installed from PyPI, or
`src/pimeta` from a checkout). The `metafor` comparison layer additionally needs
`Rscript` with `metafor` installed and is skipped when it is absent. Nothing is
installed as a package — clone and run.

```
git clone https://github.com/ChengkeChou/metabench.git
pip install pimeta
export PYTHONPATH=$PWD/metabench        # PowerShell: $env:PYTHONPATH="$PWD\metabench"
python -m pytest tests                 # 34 tests
```

## Status

Layer 2 is implemented for the univariate family (CI and PI against the true
mean and the true effect of a new study), for the multivariate joint region, and
for the between-study covariance. Layers 3 and 4 for the multivariate and
covariance families are the next section of the plan; the scenario protocol and
the record format are already fixed here, so adding them does not change what a
run means.

Families B and C are written up in `reports/B2_joint_region_formal.md` and
`reports/C_prediction_interval.md`. Their shared conclusion: the **prediction
step** under-states the uncertainty in the heterogeneity estimate, in the
univariate interval and in the joint region alike, and neither a fixed scale
factor nor a different reference distribution repairs it.

Two repairs were measured and only one held:

* for the univariate interval, blending `tau^2_hat` 20 % of the way toward its
  95 % upper bound — validated off-sample, 0.947 at `k=10`, 0.938 at `k=20`;
* for the joint region, multiplying the HK critical value — validated
  off-sample at 0.951–0.957.

An effective-degrees-of-freedom reference (observed information → delta method
to `vech(Sigma)` → Wishart moment match) reaches 0.942–0.950 without a lookup
table, but it is a prototype (`tools/_probe_effective_df.py`) and gives
`nu_eff ≈ 3.5` at `k=10`, so it needs a floor and a formal two-seed run before
it could replace the multiplier.

## Licence

MIT. See `LICENSE`.
