# B — the joint prediction region, screening run

> **Superseded in part by `B2_joint_region_formal.md`**, which re-measures the
> `k = 5, 10, 20` cells at 2000 replicates, measures the parametric bootstrap,
> and validates a scaled-reference correction on an independent seed. The
> findings below all survive; the numbers quoted there are the ones to cite.

Family **B** of the research plan: a joint interval for a *new study's* effect
vector, `pimeta.predict_region`, fitted with `random="~ outcome | study"`,
`struct="UN"`.

This is the **screening** stage (300 replicates per cell), not the formal one.
Its job is to find the cells that break and to fix the measurement before
spending 2000 replicates on them.

## Run

| field | value |
|---|---|
| harness | metabench 0.1.0 |
| implementation | pimeta 1.1.5 |
| python / numpy / scipy | 3.12.3 / 2.5.4 / 1.18.1 |
| platform | Windows-11-10.0.26300-SP0, 32 cores |
| seed | 20261029 (one stream per cell) |
| replicates | 300 per cell |
| level | 0.95 |
| cells | 6 = k ∈ {5, 10, 20, 50} × rho ∈ {0.3, 0.7} (all realised combinations below) |
| references | `chi2` (z), `f` (Hotelling), `hk` (multivariate Hartung-Knapp) |
| scenario digest | `be747cae0f97a4d8` (k10-rho30) |

The DGM: `k` studies, 2 outcomes, within-study sampling covariance `V` with a
known correlation, between-study covariance `Sigma = D R D` with
`D = diag(tau)` and equicorrelation `rho`. Truth for a *new* study is drawn
once per replicate from `N(mu, Sigma)` and stored as `truth_new_vec`; the region
is the ellipsoid `{theta : (theta - mu_hat)' (V_hat + Sigma_hat)^-1
(theta - mu_hat) <= crit}`, and **the metrics decide containment** — the adapter
only records `center`, `cov` and `crit`.

Reproduce:

```bash
python -m metabench.experiments.run --scenario multivariate/k10-rho30 \
    --adapter pimeta-region/f --reps 300 --seed 20261029 --out results
python -m metabench.reports.table results --match multivariate
python tools/diagnose_boundary.py results
```

## Coverage, 300 replicates

`joint (new)` = the truth for a new study's effect vector; `joint (mean)` = the
same region against the population mean `mu` (a check on `mu_hat` and `V_hat`,
not on the prediction step). Half-widths are 1.96 × the Monte Carlo SE.

| scenario | test | joint (new) | joint (mean) | marginal (CI / PI) | volume | boundary |
|---|---|---|---|---|---|---|
| k05-rho30 | chi2 | 68.3% ±0.053 | 93.3% ±0.028 | 90.0% / 90.7% | 0.316 | 10.3% |
| k05-rho30 | f | 82.3% ±0.043 | 99.3% ±0.009 | 96.0% / 94.0% | 0.633 | 10.3% |
| k05-rho30 | hk | 98.0% ±0.016 | 100.0% ±0.000 | 99.7% / 100.0% | 1.520 | 10.3% |
| k10-rho30 | chi2 | 66.3% ±0.053 | 96.3% ±0.021 | 90.7% / 94.0% | 0.292 | 3.0% |
| k10-rho30 | f | 70.7% ±0.052 | 97.3% ±0.018 | 91.3% / 95.3% | 0.376 | 3.0% |
| k10-rho30 | hk | 85.3% ±0.040 | 99.3% ±0.009 | 94.7% / 97.0% | 0.553 | 3.0% |
| k10-rho70 | chi2 | 71.3% ±0.051 | 97.7% ±0.017 | 93.0% / 91.7% | 0.252 | 1.0% |
| k10-rho70 | f | 77.7% ±0.047 | 99.0% ±0.011 | 94.7% / 94.3% | 0.325 | 1.0% |
| k10-rho70 | hk | 87.7% ±0.037 | 100.0% ±0.000 | 97.0% / 96.3% | 0.490 | 1.0% |
| k20-rho30 | chi2 | 75.7% ±0.049 | 98.0% ±0.016 | 94.0% / 93.0% | 0.299 | 0.0% |
| k20-rho30 | f | 79.0% ±0.046 | 98.7% ±0.013 | 95.0% / 95.3% | 0.335 | 0.0% |
| k20-rho30 | hk | 86.3% ±0.039 | 100.0% ±0.000 | 95.7% / 96.3% | 0.401 | 0.0% |
| k20-rho70 | chi2 | 71.3% ±0.051 | 99.3% ±0.009 | 93.3% / 95.7% | 0.229 | 0.3% |
| k20-rho70 | f | 74.0% ±0.050 | 99.3% ±0.009 | 94.3% / 95.7% | 0.256 | 0.3% |
| k20-rho70 | hk | 84.0% ±0.041 | 100.0% ±0.000 | 96.0% / 97.0% | 0.320 | 0.3% |
| k50-rho30 | chi2 | 87.0% ±0.038 | 99.7% ±0.007 | 96.7% / 96.3% | 0.320 | 0.0% |
| k50-rho30 | f | 87.7% ±0.037 | 99.7% ±0.007 | 97.0% / 96.3% | 0.334 | 0.0% |
| k50-rho30 | hk | 89.3% ±0.035 | 100.0% ±0.000 | 97.7% / 97.0% | 0.356 | 0.0% |

Nominal coverage is 95% throughout.

## Finding 1 — the joint region under-covers, and not by a rounding error

Every cell and every reference falls short of 95% on `joint (new)`: 66.3% to
98.0%. At k ≤ 20 the deficit is 9 to 29 points, which is 4 to 10 Monte Carlo
half-widths; the only cell inside its own error band is `hk` at k = 5, and it is
inside on the *wrong* side (over-covering, 5× the volume of `chi2`).

The defect is therefore **statistical, measured, and reproducible**: for a new
study's effect vector, `predict_region` claims 95% and delivers 66–88% over the
range of `k` where meta-analyses actually live.

## Finding 2 — the failure is in the prediction step, not in the fit

`joint (mean)` covers 93.3–100%: the region pointed at `mu` is calibrated, so
`mu_hat` and `V_hat` behave. What fails is adding `Sigma_hat` to the covariance
and treating it as known. That separates the diagnosis cleanly: the estimator of
the mean is fine; the treatment of the between-study covariance is not.

## Finding 3 — the deficit shrinks like 1/k, which names the missing term

`chi2`: 68.3% at k=5 → 66.3% at k=10 → 75.7% at k=20 → 87.0% at k=50.
The pattern is the signature of ignoring the sampling variability of
`Sigma_hat`, whose effect on a Wald region is `O(1/k)`. No choice of reference
distribution removes it: `f` and `hk` shift the number, they do not fix the
trend, and at k=5 `hk` pays for its coverage with a region five times the
volume. The fix has to put `Var(Sigma_hat)` into the region (or use a reference
that is calibrated *for a random Sigma*, such as the parametric bootstrap path
that already exists as `test="bootstrap"` and is not yet measured here).

## Finding 4 — the boundary is real but explains only a few points

Coverage split by whether any `tau2` component landed on the boundary:

| scenario | test | interior | on boundary | boundary rate |
|---|---|---|---|---|
| k05-rho30 | chi2 | 71.0% ±5.4 | 45.2% ±17.5 | 10.3% |
| k05-rho30 | f | 85.1% ±4.3 | 58.1% ±17.4 | 10.3% |
| k10-rho30 | chi2 | 67.0% ±5.4 | 44.4% ±32.5 | 3.0% |
| k20-rho30 | chi2 | 75.7% ±4.9 | — | 0.0% |

A boundary replicate is a region built on a too-small `Sigma_hat`, and it does
miss much more often (45% versus 71% at k=5). But at k=5 it is 10% of
replicates, so it accounts for about 3 of the 27 missing points; at k ≥ 20 the
boundary is essentially absent and the deficit is still 15–20 points. **The
boundary is a second defect, not the first one.** It is nevertheless worth
reporting on its own: a 10% boundary rate at k=5 is a property of the estimator
that no coverage table shows.

## Finding 5 — the library's own warning is more optimistic than the measurement

`predict_region(test="z")` warns that "measured over 300 replications of two
correlated outcomes, the region covered 0.77 at k=10 and 0.92 at k=20 …, 0.94 at
k=40". The harness measures 0.663 at k=10 and 0.757 at k=20 on its DGM. Both
can be true of different data-generating mechanisms, which is exactly why the
warning must state the mechanism and the measure, and why the harness — not a
comment written once — has to supply the number. This is a documentation defect
with a measurable consequence and is fixed in 1.1.6 together with the algorithm.

## Status after this run

| method | status | reason |
|---|---|---|
| `predict_region(test="z")` | **conditional** | calibrated only for `k >= 50` (0.87, still below 0.95) |
| `predict_region(test="t")` | **conditional** | better at small k (0.82 at k=5) but not calibrated anywhere |
| `predict_region(test="hk")` | **conditional** | covers at small k by inflating volume ~5×; under-covers at k=20 |
| the fit itself (`rma_mv`, UN) | **not judged here** | that is family D; the boundary rate above is its first datum |

No method in family B can be marked *validated* on this evidence. That is the
honest outcome of a screening run, and it is the one the plan predicted.

## What this run does not test

* `test="bootstrap"` — hundreds of refits per replicate; needs its own run.
* containment of a *new study's observed* estimates (`observation=S`); only the
  true effects are checked here.
* unbalanced designs, `J != k`, moderators, three-level models.
* anything about the algorithm's stability — that is family D.
