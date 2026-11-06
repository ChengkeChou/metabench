# B2 — the joint prediction region, formal run

Family **B**, formal stage. The screening run (`B_joint_region.md`) found the
cells that break; this run measures them at 2000 replicates, asks what critical
value would have been needed, and measures the two candidates that could supply
one: the parametric bootstrap, and a rescaled closed-form reference.

Nominal coverage is 95 % throughout; at `n = 2000` the Monte Carlo half-width is
0.011 at `p = 0.95` and 0.021 at `p = 0.67`.

## Run

| field | value |
|---|---|
| harness | metabench 0.1.0 |
| implementation | pimeta 1.1.9 |
| replicates | 2000 per cell (`k50` still the 300-replicate screening cell) |
| seed | 20261106, and 20261107 for the bootstrap |
| level | 0.95 |
| cells | `k ∈ {5, 10, 20, 50}` × `rho = 0.3`, two outcomes, `struct="UN"` |
| references | `chi2` (z), `f` (Hotelling), `hk` (multivariate Hartung-Knapp), `bootstrap` |

Reproduce:

```bash
python tools/run_grid.py region --reps 2000 --workers 6 --seed 20261106
METABENCH_NBOOT=100 python tools/run_grid.py boot --reps 100
python tools/coverage_volume.py
python tools/required_crit.py
```

## Coverage and volume, 2000 replicates

| cell | reference | coverage | volume | volume × |
|---|---|---|---|---|
| k05 | chi2 | 0.670 ±0.021 | 0.318 | 1.00 |
| k05 | f | 0.825 ±0.017 | 0.636 | 2.00 |
| k05 | hk | 0.972 ±0.007 | 1.544 | 4.86 |
| k10 | chi2 | 0.690 ±0.020 | 0.298 | 1.00 |
| k10 | f | 0.741 ±0.019 | 0.385 | 1.29 |
| k10 | hk | 0.863 ±0.015 | 0.563 | 1.88 |
| k20 | chi2 | 0.768 ±0.018 | 0.303 | 1.00 |
| k20 | f | 0.793 ±0.018 | 0.339 | 1.12 |
| k20 | hk | 0.846 ±0.016 | 0.406 | 1.34 |
| k50 | chi2 | 0.870 ±0.038 | 0.320 | 1.00 |
| k50 | f | 0.877 ±0.037 | 0.334 | 1.04 |
| k50 | hk | 0.893 ±0.035 | 0.356 | 1.11 |

The volume column is `crit**(J/2) * det(cov)**0.5`, monotone in the ellipsoid's
volume, relative to the chi-square region in the same cell.

The screening conclusion survives at four times the replicates: **no shipped
reference is calibrated at `k ≤ 20`**, and the two that get closer do it by
buying volume (`hk` at `k = 5` covers by being 4.9× the chi-square volume, and
still misses by 5 points at `k = 20`).

## What a calibrated region would have had to be

Because containment is one scalar,
`D² = (theta − centre)' cov⁻¹ (theta − centre) ≤ crit`, the coverage of *any*
critical value can be recomputed from the raw records without refitting. So the
question "what would it have taken?" has an exact answer per cell:

| cell | D² median | D² p95 | chi2 crit | needed / chi2 | needed / f | needed / hk |
|---|---|---|---|---|---|---|
| k05 | 3.19 | 27.61 | 5.99 | 4.61 | 2.30 | 0.81 |
| k10 | 2.87 | 33.86 | 5.99 | 5.65 | 4.39 | 2.11 |
| k20 | 2.10 | 27.70 | 5.99 | 4.62 | 4.13 | 2.37 |
| k50 | 1.64 | 9.15 | 5.99 | 1.53 | 1.46 | 1.29 |

Two things stand out.

1. **The median is already inflated.** Under a correctly specified chi-square
   reference with `J = 2`, the median of `D²` is 1.386. Measured: 3.19 at `k = 5`,
   2.87 at `k = 10`, 2.10 at `k = 20`, 1.64 at `k = 50`. So it is not a tail
   problem only: the whole distribution of the pivot is scaled up, by 2.3×,
   2.1×, 1.5× and 1.2×. That is `Sigma-hat` being too small, on average, by
   roughly the same factors — a between-study covariance estimated from a handful
   of studies, with the variances truncated at zero when they hit the boundary.
2. **The tail is much worse than the middle.** `p95` is 27–34 against a chi-square
   critical value of 5.99: a factor of 4.6–5.7 at the point that decides
   coverage. A correction that fixes the average will not fix the 5th percentile,
   and a correction large enough for the tail triples the volume of the typical
   region.

This is why the two obvious repairs fail: they treat a conditional defect as a
scale problem. The univariate family C reached exactly the same conclusion from
the other end (`C_prediction_interval.md`, finding 4).

## The bootstrap

`predict_region(test="bootstrap")` replaces the reference distribution with a
parametric bootstrap: each of `n_boot` replicates draws a new data set from the
fitted model (`theta_i ~ N(0, Sigma-hat)`, sampling error with the known
within-study covariance), refits, and contributes
`(theta* − mu*hat)' (V* + Sigma*hat)⁻¹ (theta* − mu*hat)`. The 95 % quantile of
those values becomes the critical value. The point of it is that each replicate's
pivot is built from *its own* `Sigma-hat`, so the estimator's variability is
inside the calibration rather than assumed away.

Measured at 100 replicates per cell, `n_boot = 100`, seed 20261107:

| cell | bootstrap coverage | volume × | `hk` coverage | `hk` volume × |
|---|---|---|---|---|
| k05 | 0.810 ±0.077 | 4.6 | 0.972 ±0.007 | 4.9 |
| k10 | 0.880 ±0.064 | 4.0 | 0.863 ±0.015 | 1.9 |
| k20 | 0.860 ±0.068 | 3.7 | 0.846 ±0.016 | 1.3 |

The bootstrap moves coverage up (0.741 → 0.880 at `k = 10` against Hotelling) but
**it does not reach nominal, and it charges three to four times the volume.**
The reason is visible in its own critical value: median 19.2 at `k = 10` against
7.7 for Hotelling — a large correction — while the empirical requirement for 95 %
coverage at that cell is 33.9. The bootstrap under-corrects, and it under-corrects
*systematically*, because it takes `Sigma-hat` as the truth of the world it
simulates from. When `Sigma-hat` is biased low, the bootstrap world has less
heterogeneity than the real one, so its quantile is calibrated to the wrong
distribution. **The parametric bootstrap inherits the bias of the estimate it
conditions on** — which is exactly the defect it was supposed to repair.

## The correction that works: scale the reference

The required multipliers above are the measurement. A region is a single number
away from any other region with the same covariance, so the cheapest possible
correction is to multiply the critical value:

```
crit_scaled = crit_base * m(k)
```

with `m(k)` read off the formal run and then — the part that matters — **applied
unchanged to a seed that was never used to pick it** (20261107):

| cell | base | multiplier | coverage (off-sample) | volume × |
|---|---|---|---|---|
| k05 | `f` | 2.30 | 0.951 ±0.009 | 4.7 |
| k10 | `f` | 4.39 | 0.957 ±0.009 | 5.8 |
| k20 | `f` | 4.13 | 0.950 ±0.010 | 4.7 |
| k50 | `f` | 1.46 | 0.937 ±0.011 | 1.6 |
| k10 | `hk` | 2.11 | **0.956 ±0.009** | **4.1** |
| k20 | `hk` | 2.37 | **0.947 ±0.010** | **3.2** |
| k50 | `hk` | 1.29 | 0.939 ±0.010 | 1.6 |

Three things follow.

1. **A scaled reference is calibrated.** Four of the five `f`-based cells land in
   0.950–0.957 on a fresh seed, at Monte Carlo half-widths of 0.009–0.011. The
   correction is not a curiosity of one seed.
2. **Hartung-Knapp is the cheaper base.** `hk` already inflates by the observed
   residual scale, so it needs a smaller multiplier and delivers the same
   coverage at 3.2–4.1× the chi-square volume instead of 4.7–5.8×. At `k = 5`
   plain `hk` already covers 0.972 and needs no correction at all.
3. **The multiplier is not monotone in `k`** — 2.30 at `k = 5`, 4.39 at `k = 10`,
   4.13 at `k = 20`, 1.46 at `k = 50`. It cannot therefore be written as a tidy
   formula like `1 + c/k`; it has to be a measured table, or recomputed per data
   set. That is the honest shape of the result: the correction exists and is
   validated, but it is *calibrated by simulation*, not derived.

## Status

| method | status | reason |
|---|---|---|
| `predict_region(test="z")` | **measured-shortfall** | 0.670 / 0.690 / 0.768 / 0.870 at k = 5 / 10 / 20 / 50 |
| `predict_region(test="t")` | **measured-shortfall** | 0.825 / 0.741 / 0.793 / 0.877, buying 1.3–2.0× volume |
| `predict_region(test="hk")` | **measured-shortfall** | covers 0.972 at k=5 only by being 4.9× the volume; 0.846 at k=20 |
| `predict_region(test="bootstrap")` | **measured-shortfall** | 0.810 / 0.880 / 0.860 at 3.7–4.6× volume; inherits the bias of `Sigma-hat` |
| `hk` × m(k), m from the table | **conditional — validated off-sample** | 0.947–0.956 at k = 10–20 for 3.2–4.1× volume; m is not monotone |

No shipped reference is validated, and none should be: at the sample sizes real
meta-analyses have, a `J × J` unstructured between-study covariance is not
estimated well enough for any closed-form ellipsoid to hold its nominal level.
What *is* validated is the scaling rule — with the caveat that its multiplier was
measured, not derived, so it covers the two-outcome, equicorrelated designs it
was measured on and nothing else.

## Beyond two outcomes: J = 3, J = 5, and stronger correlation

The multipliers above were measured at `J = 2`, `rho = 0.3`. Two questions follow
immediately: does the multiplier carry over to more outcomes, and does it depend
on the correlation? Both were measured (500 replicates per cell):

| cell | chi-square | Hotelling | Hartung-Knapp | multiplier needed (on `hk`) |
|---|---|---|---|---|
| k05, J=2, rho=.3 | 0.670 | 0.825 | 0.972 | 0.81 |
| k10, J=2, rho=.3 | 0.690 | 0.741 | 0.863 | 2.11 |
| k20, J=2, rho=.3 | 0.768 | 0.793 | 0.846 | 2.37 |
| k10, J=2, rho=.7 | 0.728 | 0.768 | 0.892 | 1.59 |
| k20, J=2, rho=.7 | 0.734 | 0.766 | 0.856 | 2.05 |
| k10, J=3, rho=.3 | 0.514 | 0.574 | 0.890 | 1.53 |
| k20, J=3, rho=.3 | 0.646 | 0.662 | 0.836 | 1.79 |
| k20, J=5, rho=.3 | **0.342** | **0.368** | 0.880 | 1.48 |

Three findings, and the third one changes what the fix can be.

1. **The closed forms collapse as `J` grows.** At `J = 5`, `k = 20` the
   chi-square region covers **one third** of the nominal level: fifteen variance
   parameters estimated from twenty studies is not a situation any asymptotic
   ellipsoid survives.
2. **Hartung-Knapp does not collapse with it** (0.836-0.890 across every cell
   above). Its inflation is computed from the residuals, so it responds to how
   badly the covariance is estimated, which is exactly the quantity that blows up
   with `J`.
3. **The multiplier is a property of the design, not a function of `k`.** It is
   2.11 at `k = 10, J = 2, rho = 0.3`, 1.59 at the same `k` with `rho = 0.7`, and
   1.53 at `k = 10` with `J = 3`. So a table keyed on `k` alone is wrong as soon
   as the design moves, which is why `crit_scale="measured"` in the library
   **refuses** a `J` that was not measured rather than interpolating into it.

### The multipliers off-sample, beyond J = 2

Re-running those three new cells on seed 20261107 with the multiplier fixed at
its seed-A value (`tools/validate_region_scale.py`):

| cell | multiplier | coverage on seed B | volume × chi-square |
|---|---|---|---|
| k10, J=3 | 1.53 | 0.974 ±0.014 | 9.3 |
| k10, J=2, rho=.7 | 1.59 | 0.966 ±0.016 | 3.0 |
| k20, J=5 | 1.48 | 0.966 ±0.016 | 26.2 |

All three land **above** nominal: the multiplier is right in direction and too
large in size. The cause is measured, not guessed — these cells have 500
replicates, and the multiplier is a 95th percentile, which at `n = 500` is the
25th largest value; the two `J = 2` cells run at 2000 replicates reproduce at
0.947-0.957, and the 300-replicate `k = 50` cell reproduces at 0.939. **A
multiplier calibrated from fewer than about 2000 replicates is itself
unreliable**, which is a general caution about this whole family of corrections
rather than a quirk of one cell.

The `J = 5` volume is worth stating plainly: 26 times the chi-square region. At
that point the honest description of a 95 % joint prediction region for five
outcomes from twenty studies is "uninformative but correct", and the practical
recommendation is per-outcome intervals rather than a joint ellipsoid.

## Can the multiplier be derived instead of tabulated?

A multiplier that changes with `k`, `J` and `rho` is a symptom, not a constant.
The natural derivation: the `F` reference is exact when the covariance is a
*known* Wishart with `nu` degrees of freedom, so estimate `nu` from the fit and
use `J F(J, nu - J + 1)`. The prototype (`tools/probe_edf_coverage.py`) does
exactly that:

1. take the observed information of the variance parameters, numerically
   (`tools/_probe_effective_df.py`; **3-5 ms** per fit at `J = 2`, against 3 s for
   a single profile interval, so this is affordable per fit);
2. propagate it to `vech(Sigma)` by the delta method;
3. match a scaled Wishart, `Var(S_ij) = (Sigma_ij^2 + Sigma_ii Sigma_jj) / nu`,
   by least squares over the elements.

The answers are informative in themselves:

| cell | k - p | nu_eff estimate | implied critical value | measured requirement |
|---|---|---|---|---|
| k10, J=2 | 8 | ~3.5 | 24.9 | 18.8 |
| k20, J=2 | 18 | ~7.0 | 14.3 | 16.8 |

So the covariance *behaves* as if it had been estimated from three to four
studies at `k = 10`, against the eight that `k - p` suggests. That is the
missing degrees of freedom the multipliers were compensating for, measured
directly rather than inferred from coverage.

Coverage of the effective-df reference, 120 replicates per cell:

| cell | Hartung-Knapp | effective df |
|---|---|---|
| k05, J=2 | 0.983 | 0.942 |
| k10, J=2 | 0.825 | **0.942** |
| k20, J=2 | 0.867 | **0.942** |
| k10, J=2, rho=.7 | 0.875 | **0.950** |

At `k >= 10` it matches or beats the tabulated rule with no tuned constant at
all. At `k = 5` it needs a floor — `nu_eff` falls below `J - 1` in most
replicates, because with five studies the information matrix is nearly singular
in the correlation direction — and the table above applies `nu >= J + 1`.

**Status: prototype.** It lives in `tools/`, reaches into the package for the
objective (which an adapter must not do), and has been run on one seed at 120
replicates. Before it could replace `crit_scale="measured"` it would need the
information matrix exposed as public API, a formal run, and a decision about the
`k = 5` floor. It is reported here because it is the only route on the table that
would produce a multiplier instead of measuring one.

## What this does not settle

* a *matrix-valued* confidence set for `Sigma` as the input to the region — the
  analytic version of what the bootstrap does numerically, and the natural next
  step after the effective-df prototype;
* multipliers for `J >= 3` off-sample: the table above is seed A only, and the
  three off-sample checks that were run (`hk` x m at `k10-J3`, `k10-rho70`,
  `k20-J5`) are reported in `tools/validate_region_scale.py`;
* moderators, unbalanced within-study variances, and three-level models.
