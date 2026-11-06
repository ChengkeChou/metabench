# C — the univariate prediction interval

Family **C** of the research plan: the interval `pimeta.predict()` returns, whose
estimand is
the **true effect of one new study**, `theta_new ~ N(mu, tau^2)`.

This family had never been measured. Family B (the joint region) had, and the
finding there was an under-covering region whose deficit was in the *prediction*
step, not in the fit. C asks whether the single interval shows the same signature
— and it does, with the same mechanism, so the two are reported together.

## Run

| field | value |
|---|---|
| harness | metabench 0.1.0 |
| implementation | pimeta 1.1.9 |
| replicates | 2000 per cell |
| seed | 20261106 (validation on 20261107) |
| level | 0.95 |
| cells | 10 = `k ∈ {5, 10, 20, 50, 100}` × `tau^2 ∈ {0.04, 0.13}` |
| DGM | `theta_i ~ N(0.2, tau^2)`, `y_i ~ N(theta_i, 0.04)`, equal within-study variance |
| references | the shipped interval, and pimeta's `predtype="t"` variant |

At `k = 100` the Monte Carlo half-width is 0.011, so a 0.931 is not a 0.950 with
bad luck; it is a different number.

Reproduce:

```bash
python tools/run_grid.py pi --reps 2000 --seed 20261106 --workers 5
python tools/table_pi.py
python tools/diagnose_pi.py
python tools/anatomy_pi.py
```

## Coverage, 2000 replicates

| k | tau^2 | shipped | `predtype="t"` | CI (same fit) | boundary rate | mean width |
|---|---|---|---|---|---|---|
| 5 | 0.04 | **0.797** ±0.018 | 0.893 ±0.014 | 0.913 | 28.4% | 0.832 |
| 5 | 0.13 | **0.825** ±0.017 | 0.909 ±0.013 | 0.885 | 8.8% | 1.420 |
| 10 | 0.04 | **0.836** ±0.016 | 0.867 ±0.015 | 0.932 | 13.9% | 0.791 |
| 10 | 0.13 | **0.894** ±0.013 | 0.924 ±0.012 | 0.927 | 1.3% | 1.424 |
| 20 | 0.04 | **0.888** ±0.014 | 0.900 ±0.013 | 0.937 | 3.8% | 0.780 |
| 20 | 0.13 | 0.929 ±0.011 | 0.947 ±0.010 | 0.938 | 0.05% | 1.428 |
| 50 | 0.04 | 0.919 ±0.012 | 0.922 ±0.012 | 0.946 | 0.1% | 0.785 |
| 50 | 0.13 | 0.939 ±0.011 | 0.944 ±0.010 | 0.944 | 0% | 1.422 |
| 100 | 0.04 | **0.931** ±0.011 | 0.934 ±0.011 | 0.943 | 0% | 0.783 |
| 100 | 0.13 | 0.940 ±0.010 | 0.943 ±0.010 | 0.949 | 0% | 1.413 |

Nominal coverage is 0.95. Bold cells are more than two Monte Carlo half-widths
short of it.

## Finding 1 — the prediction interval under-covers below k ≈ 50

From 0.797 at `k = 5` to 0.931 at `k = 100`, against a nominal 0.95. The deficit
is 15 points at `k = 5`, 11 at `k = 10`, 6 at `k = 20`, 3 at `k = 50`, and still
2 at `k = 100` — where the half-width is 0.011, so the last point is also real.

For comparison, the **confidence interval from the same fits covers 0.885–0.949**,
and is within two half-widths of nominal at eight of the ten cells (the two
`k = 5` cells are 0.913 and 0.885). The interval about the *mean* is broadly
calibrated; the interval about a *new study* is not. That is the same split
family B found, on a different estimand, which makes it a property of the
prediction step rather than of either model.

`predtype="t"` — the variant that charges the t distribution for the uncertainty
in `tau^2` — is better everywhere (0.893 at `k = 5` instead of 0.797) and still
short at `k ≤ 20`.

## Finding 2 — the deficit concentrates where tau^2 is small relative to sampling error

Both `tau^2 = 0.04` (I^2 ≈ 50 %) cells are worse than their `tau^2 = 0.13`
partners at every `k`, by 3 to 6 points. That is the ordering real meta-analyses
have: the interval is at its least reliable exactly in the common case of
moderate heterogeneity, not in the dramatic one.

## Finding 3 — the width collapses, the distance it has to cover does not

This is the mechanism, and it is visible without any modelling. Per replicate,
the correlation between the interval's half-width and `tau^2_hat` is **0.97**,
while the correlation between the half-width and the distance it actually has to
cover, `|theta_new - mu_hat|`, is **0.03**:

| cell | corr(tau2_hat, half-width) | corr(&#124;dev&#124;, half-width) | |dev| p90 | half-width p05 |
|---|---|---|---|---|
| k05-tau004 | 0.970 | 0.030 | 0.387 | 0.175 |
| k10-tau004 | 0.967 | 0.010 | 0.360 | 0.124 |
| k20-tau004 | 0.973 | 0.015 | 0.344 | 0.126 |
| k50-tau004 | 0.985 | −0.016 | 0.333 | 0.259 |

At `k = 5` the 5th percentile of the half-width is 0.175 while the 90th
percentile of `|theta_new - mu_hat|` is 0.387. Those replicates are inside an
interval half as wide as the error they have to contain. The interval is not
uniformly too narrow — it is *conditionally* much too narrow whenever `tau^2` is
estimated small, and `tau^2` is estimated small often: the estimate is 0 in 28 %
of `k = 5` replicates and 14 % of `k = 10` ones.

Splitting coverage by that event:

| cell | coverage when tau^2_hat > 0 | coverage when tau^2_hat = 0 | share on the boundary |
|---|---|---|---|
| k05-tau004 | 0.907 | 0.521 | 28.4% |
| k10-tau004 | 0.905 | 0.406 | 13.9% |
| k20-tau004 | 0.911 | 0.325 | 3.8% |

A boundary replicate is a *catastrophic* miss (0.33–0.52 coverage), but it is
not the whole story: interior coverage is still 0.905–0.911 rather than 0.95.

## Finding 4 — a scale factor does not fix it

The obvious repair — multiply the half-width by a constant, or widen the variance
by `k/(k-2)`, or use `t` on `k-2` df — was measured rather than argued about
(`tools/pi_rules.py`, evaluated on the raw records of the same run):

| rule | k=5 tau.04 | k=10 tau.04 | k=20 tau.04 | k=50 tau.04 | width cost |
|---|---|---|---|---|---|
| shipped | 0.797 | 0.836 | 0.888 | 0.919 | 1.00 |
| `t` on `k-2` df | 0.846 | 0.841 | 0.889 | 0.919 | 1.03 |
| half-width × (1 + 2.5/k) | 0.906 | 0.883 | 0.909 | 0.929 | 1.19 |
| tau^2 × k/(k−2), `t` on `k-2` | 0.925 | 0.879 | 0.908 | 0.928 | 1.29 |

Even at nearly twice the width, `k = 5` reaches only 0.925. The interval's
problem is not its average scale — it is that the scale is wrong *in the tails*,
where `tau^2_hat` collapses. A constant cannot fix a conditional defect.

## Finding 5 — the repair that follows from the diagnosis: stop the collapse

If the misses happen when `tau^2_hat` collapses, the repair is to stop the width
from collapsing, which means building it from an **upper confidence bound** for
`tau^2` instead of the point estimate. `pimeta.confint_tau2` already computes the
QP profile bound, and at the boundary it returns a median 0.118 instead of 0 —
exactly the quantity needed.

Measured on all ten cells, 2000 replicates (`tools/compare_pi_rules.py`):

| cell | shipped | 95 % upper bound | 90 % upper bound | 95 % UB, `t` on `k-2` |
|---|---|---|---|---|
| k05-tau004 | 0.797 ±0.018 | 0.992 ±0.004 (×3.98) | 0.985 ±0.005 (×3.18) | 0.994 ±0.003 (×4.56) |
| k10-tau004 | 0.836 ±0.016 | 0.994 ±0.003 (×2.35) | 0.986 ±0.005 (×2.04) | 0.995 ±0.003 (×2.39) |
| k20-tau004 | 0.888 ±0.014 | 0.992 ±0.004 (×1.77) | 0.981 ±0.006 (×1.60) | 0.992 ±0.004 (×1.78) |
| k50-tau004 | 0.919 ±0.012 | 0.989 ±0.005 (×1.41) | 0.981 ±0.006 (×1.32) | 0.989 ±0.005 (×1.41) |
| k100-tau004 | 0.931 ±0.011 | 0.981 ±0.006 (×1.27) | 0.975 ±0.007 (×1.21) | 0.981 ±0.006 (×1.27) |

Parentheses are the mean width relative to the shipped interval. The bound
over-corrects: 0.99 where 0.95 was asked, at four times the width at `k = 5`.
That is expected — a one-sided 95 % statement about `tau^2` is not a two-sided
95 % statement about a new study, and treating the two as the same thing buys
coverage with a great deal of width.

## Finding 6 — a partial bound is calibrated at k ≥ 10, and validated off-sample

Blending a fifth of the way from the point estimate to the bound,

```
tau2_used = tau2_hat + 0.2 * (tau2_ub95 - tau2_hat)
```

was chosen on seed 20261106 and then run again on **seed 20261107**, which was
never used for the choice:

| cell | shipped (seed B) | blend 0.2 (seed B) | width × |
|---|---|---|---|
| k05-tau004 | 0.813 ±0.017 | 0.977 ±0.007 | 2.18 |
| k10-tau004 | 0.837 ±0.016 | **0.947** ±0.010 | 1.48 |
| k20-tau004 | 0.878 ±0.014 | **0.938** ±0.011 | 1.24 |

At `k = 10` the blend lands on nominal on a seed it never saw; at `k = 20` it is
1.2 half-widths short; at `k = 5` it is conservative by 3 points at more than
double the width. The direction and the size both reproduce, so the mechanism is
confirmed and the constant is honest — but 0.2 is an empirical calibration on one
DGM, not a theorem, and it should be reported as such wherever it is used.

## What this run does not test

* the parametric bootstrap interval (the same construction family B uses): it is
  the principled version of "let the data decide the quantile", and it costs
  `n_boot` refits per replicate. It is the next run, not this one.
* `J > 1` (the joint region), unbalanced `se`, moderators, three-level models.
* the *observed* effect of a new study (which adds the new study's own sampling
  variance to the interval and has a different, wider target).

## Status after this run

| interval | status | reason |
|---|---|---|
| `predict()` shipped | **conditional** | 0.80–0.93 below `k = 50`; 0.931 even at `k = 100` |
| `predict(predtype="t")` | **conditional** | better everywhere, still 0.87–0.92 below `k = 20` |
| the CI from the same fits | **mostly calibrated** | 0.885–0.949 across `k ∈ [5, 100]`; the two `k = 5` cells are the short ones |
| `tau^2_hat + 0.2(tau^2_ub − tau^2_hat)` | **conditional** | calibrated at `k = 10` off-sample (0.947), conservative at `k = 5`, 0.938 at `k = 20` |
| the 95 % upper bound | **over-corrects** | 0.98–0.99 at 1.2–4.6× the width |

The single interval and the joint region fail for the same reason and to a
similar degree, which is the strongest result here: **the prediction step of this
library under-states the uncertainty in the heterogeneity estimate, and the
deficit is a property of the step, not of a particular reference distribution.**
