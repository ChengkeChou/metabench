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
```

## Status

Layer 2 is implemented for the univariate family (CI and PI against the true
mean and the true effect of a new study). Layers 3 and 4 for the multivariate
and covariance families are the next section of the plan; the scenario protocol
and the record format are already fixed here, so adding them does not change
what a run means.
