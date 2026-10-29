# D — the UN covariance, screening run

Family **D** of the research plan: when `rma_mv` fits an unstructured
between-study covariance `Sigma` (J(J+1)/2 parameters), is the matrix it
returns

1. a covariance matrix at all — symmetric, positive definite, variances >= 0;
2. the **maximiser** of the likelihood it claims to maximise;
3. the same matrix whatever parameterisation is used to reach it;
4. and, only after those three, close to the truth?

Four different questions, four different measurements.  A convergence flag
answers none of them: **every** wrong fit reported in this document came back
with `converged=True`.

## Design

| field | value |
|---|---|
| harness | metabench 0.1.0 |
| implementation | pimeta 1.1.5 |
| seed | 20261029 (one stream per cell) |
| replicates | 250 per cell; the two J=5 cells are not reported (see *Cost*) |
| cells | 8 |
| estimand | the matrix `Sigma`, not an interval |

The data-generating mechanism is the model itself: `theta_i ~ N(0, Sigma)`,
`y_i ~ N(theta_i, diag(se^2))`, with `Sigma = D R D`, `D = diag(sd)`.  Sampling
covariances are **diagonal** and the means are **zero** on purpose: a
correlated `V` or a moderator would give a second explanation for a
disagreement about `Sigma`, and the point of a screening run is to have one
explanation.

| cell | k | J | what it isolates |
|---|---|---|---|
| `k10-J2-rho30` | 10 | 2 | three parameters; the case where the per-pair box *is* the cone |
| `k10-J3-rho30` | 10 | 3 | the smallest J where the box is no longer the cone |
| `k10-J3-rho70` | 10 | 3 | nearly collinear outcomes: a flat direction |
| `k10-J3-mixed` | 10 | 3 | correlations (0.5, -0.4, 0.1): both signs |
| `k30-J3-rho30` | 30 | 3 | three times the studies |
| `k30-J3-tiny` | 30 | 3 | one true variance 100x smaller than the sampling variance |
| `k30-J5-rho30` | 30 | 5 | 15 parameters |
| `k30-J5-rho70` | 30 | 5 | 15 parameters, nearly rank-deficient truth |

## The reference, and why it is not the same code

`metabench/reference/un_reml.py` writes the restricted likelihood out from the
algebra and optimises it three ways, none of which is `pimeta`'s:

| variant | parameterisation | constraints |
|---|---|---|
| `hyper` | `Sigma = D R D`, `R = L L'` from the unit-row hyperspherical map, `z` unbounded | none |
| `external` | `(tau2, logit rho)` — the vector `pimeta` exposes | `rho` in (-1,1) plus the same `1e5 + 1e7(1e-6-lam)^2` cliff `pimeta` uses |
| `chol` | `Sigma = L L'`, `L_ii = exp(a)`, free off-diagonals | none (positivity by construction) |

Each replicate records **both** answers evaluated by the *same* function:
`l(Sigma_pimeta)` and `l(Sigma_reference_best)`.  Their difference (`gap`) is a
statement that needs no shared code to be believed.  The reference never sees
`pimeta`'s answer; it starts from the data, at six deterministic perturbations
of a method-of-moments start.

Two implementation notes that turned out to matter:

* **The random-effects covariance is block diagonal by study**, not
  `X Sigma X'`.  The latter couples any two rows that share an outcome,
  including rows from different studies — a different model.  Getting this
  wrong produced a 4.4-unit disagreement with `pimeta`'s own likelihood that
  looked like a software defect for an hour.
* **The likelihood is evaluated block by block**: `M = blockdiag(V_s + Sigma)`,
  so `X'M^-1 X = sum_s M_s^-1`.  That is `O(k J^3)` instead of `O((kJ)^3)`,
  and it is what makes a reference that runs once per replicate affordable.

### A guard the harness earned the hard way

The first pass reported `pimeta` as suboptimal in 11.6% of replicates at
k=10, J=3, with one gap of **2658** log-likelihood units.  The gap was real, but
it was the *reference* that was wrong: one start had wandered to
`tau2 ~ 1e43`, where `log|M|` and `log|X'M^-1 X|` are two numbers of order
`kJ log(lambda)` that cancel **past** the true minimum in floating point.  The
reference reported `nll = -2648.9` on data whose best possible value is `9.27`
— a likelihood better than any likelihood.

There is no correct way to score that number, so the region is refused
(`UNLikelihood.nll` returns `1e30` beyond `1e4 x` the data's own scale) and the
case is a regression test (`tests/test_reference_guards.py`).  The lesson
generalises: **a measurement system that can be 2658 units better than the
truth would have declared every honest fit suboptimal**, and the only reason it
was caught is that the harness records what the reference found as well as what
the library returned.

## Cost, and why the J=5 cells carry fewer replicates

`rma_mv` at k=30, J=5 takes about nine seconds per fit, against 0.03 s at
k=30, J=3.  The cause is structural, and it splits into two measurable pieces:

| quantity (k=30, J=5, 150x150) | measured |
|---|---|
| `comp.make_D(params)` | 0.2 ms |
| `_mv_loglik` (one 150x150 inverse plus determinants) | **16.2 ms** |
| the same likelihood evaluated block-wise | 0.109 ms |

The model is block diagonal, so the likelihood is `O(k J^3)`; the
implementation materialises the whole `kJ x kJ` matrix and pays `O((kJ)^3)`,
which is 149x more work per evaluation at this size and grows like `k^2` after
that.  One J=5 fit costs as much as 83,000 block evaluations where an L-BFGS-B
run needs hundreds.

That is why the two J=5 cells are **not reported**: 100 replicates would have
been a multi-hour job and still too few to estimate a 10% rate, and a rate
estimated from a handful of replicates is worse than no rate.  The budget went
where a proportion can be estimated, not where the software is the bottleneck. 
The measurement itself is the first entry for the 1.1.6 performance baseline.

## Results

The `hyper` rows are the primary grid.  `suboptimal` is the share of fits whose
`Sigma` is worse than the reference optimum by more than `1e-6`; `gap med` and
`gap max` are the log-likelihood differences; `multi-start` is the share of
replicates where the reference's own six starts failed to agree, i.e. where the
surface has more than one local maximum.

| scenario | adapter | reps | usable | suboptimal | gap med | gap max | non-PD | boundary | multi-start | Frob med | min-eig | tau2 bias | rho bias |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| covariance/k10-J2-rho30 | pimeta-cov/hyper | 250 | 250 | 2.4% ±0.019 | 0.00 | 0.14 | 2.8% | 38.0% | 0.4% | 0.7289 | 0.194 | 0.052 | 0.073 |
| covariance/k10-J3-mixed | pimeta-cov/hyper | 250 | 250 | 12.4% ±0.041 | 0.00 | 2.60 | 20.4% | 19.6% | 8.4% | 0.8665 | 0.000 | 0.175 | -0.054 |
| covariance/k10-J3-rho30 | pimeta-cov/chol | 150 | 150 | 10.7% ±0.049 | 0.00 | 1.19 | 17.3% | 22.7% | 0.7% | 0.9137 | 0.000 | 0.131 | -0.070 |
| covariance/k10-J3-rho30 | pimeta-cov/external | 150 | 150 | 10.7% ±0.049 | 0.00 | 0.73 | 17.3% | 22.7% | 78.0% | 0.9137 | 0.000 | 0.131 | -0.070 |
| covariance/k10-J3-rho30 | pimeta-cov/hyper | 250 | 250 | 11.6% ±0.040 | 0.00 | 3.03 | 20.8% | 22.8% | 8.4% | 0.9076 | 0.000 | 0.119 | -0.071 |
| covariance/k10-J3-rho30 | pimeta-cov/hyper-lkj2 | 120 | 120 | 100.0% ±0.000 | 0.55 | 1.43 | 4.2% | 7.5% | 8.3% | 0.7628 | 0.410 | 0.087 | -0.187 |
| covariance/k10-J3-rho30 | pimeta-cov/hyper-pinned | 120 | 120 | 0.8% ±0.016 | 0.00 | 0.06 | 28.3% | 28.3% | 0.8% | 0.7056 | 0.233 | 0.055 | -0.059 |
| covariance/k10-J3-rho70 | pimeta-cov/hyper | 250 | 250 | 10.0% ±0.037 | 0.00 | 0.82 | 25.2% | 25.6% | 3.2% | 0.7173 | 0.000 | 0.151 | -0.153 |
| covariance/k30-J3-rho30 | pimeta-cov/hyper | 250 | 250 | 0.8% ±0.011 | 0.00 | 0.74 | 3.2% | 1.2% | 1.2% | 0.5819 | 0.255 | 0.010 | 0.014 |
| covariance/k30-J3-rho30 | pimeta-cov/hyper-lkj2 | 120 | 120 | 100.0% ±0.000 | 0.24 | 0.89 | 0.0% | 0.0% | 0.0% | 0.5378 | 0.548 | 0.021 | -0.060 |
| covariance/k30-J3-rho30 | pimeta-cov/hyper-pinned | 120 | 120 | 0.0% ±0.000 | 0.00 | 0.00 | 3.3% | 3.3% | 0.0% | 0.4606 | 0.726 | -0.001 | -0.007 |
| covariance/k30-J3-tiny | pimeta-cov/hyper | 250 | 250 | 19.2% ±0.049 | 0.00 | 3.66 | 22.0% | 22.0% | 5.6% | 0.5536 | 0.000 | 1.890 | -0.062 |

`suboptimal` and `non-PD` are *pimeta*'s behaviour and are identical across the
three variant rows by construction: the variant changes the **reference**, not
the library.  What the variant rows measure is how good a reference each
parameterisation is, which is Finding 6.

`tau2 bias` is the mean of `(tau2_hat - tau2_true)/tau2_true` over components,
and in the `tiny` cell that ratio is not a stable statistic: its denominator is
100x smaller than the sampling variance.  The per-component table in
`tools/d_covariance_extras.py` is the honest version for that cell — the small
component's median estimate is 0.0037 against a truth of 0.0009, and 16.8% of
fits put it at zero.

Still running when this section was written: the two J=5 cells.  They are not
reported because they could not be afforded, and the reason is Finding 6 in the
cost section: one `rma_mv` call at k=30, J=5 can take minutes once the
Nelder-Mead fallback fires, which makes 100 replicates a multi-hour job.  A
rate that cannot be measured is not a rate, and reporting one from a handful of
replicates would be worse than reporting none.


## Finding 1 — the search is reliable when the model is identified

`k30-J3-rho30`: **0.8% ±1.1** of fits are below the reference optimum (2 of
250).  With 30 studies and six parameters, `pimeta`'s single L-BFGS-B start
finds the maximum essentially always, and 1.2% of the answers sit on the
correlation boundary.

This is the control cell, and it decides how the rest is read: the failure that
follows is not "the optimiser is broken".  It is "the optimiser is fragile
exactly where the surface is hard".

## Finding 2 — at k=10 with J=3, one fit in ten is a worse local point

| cell | suboptimal | gap (median / max) | multi-start conflict |
|---|---|---|---|
| `k10-J3-rho30` | 11.6% ±4.0 | 0.000 / 3.03 | 8.4% |
| `k10-J3-mixed` | 12.4% ±4.1 | 0.000 / 2.60 | 8.4% |
| `k10-J3-rho70` | 10.0% ±3.7 | 0.000 / 0.82 | 3.2% |
| `k30-J3-rho30` | 0.8% ±1.1 | 0.000 / 0.74 | 1.2% |

Two facts make this a finding about the *software* rather than about the model:

* `ll_library - ll_at_reported` is **0.0 in every one of the 1500 records**
  (max 2.7e-13).  The matrix the library reports is exactly the one it
  optimised, so the gap is not a reporting artefact — the fit genuinely stopped
  at a worse point.
* The reference's own six starts disagree in 8.4% of the same replicates
  (`multi-start conflict`), and the disagreeing fits are the failing ones.  The
  surface has more than one local maximum, and a single start finds the other
  one.

A gap of 3.03 log-likelihood units is not a rounding error; at these sample
sizes it is a visibly different covariance matrix.

## Finding 3 — the maximum is often singular, and that is the model, not the search

| cell | `max abs rho = 1` | `min tau2 = 0` | non-PD `Sigma` | median min-eigenvalue |
|---|---|---|---|---|
| `k10-J2-rho30` | 30.0% | 2.8% | 2.8% | 0.0066 |
| `k10-J3-rho30` | 20.4% | 6.0% | 20.8% | 0.0000 |
| `k10-J3-rho70` | 24.8% | 2.4% | 25.2% | 0.0000 |
| `k30-J3-rho30` | 0.8% | 0.4% | 3.2% | 0.0081 |
| `k30-J3-tiny` | 9.2% | 16.8% | 22.0% | 0.0000 |

At k=10 the REML estimate of a 3x3 between-study covariance is singular in a
fifth of data sets, and the median estimate is *exactly* singular.  The
reference finds the same point, so this is not an optimisation failure: with
ten studies and a 3-vector per study, the likelihood genuinely prefers a
rank-deficient `Sigma`.  Any downstream object that assumes a non-singular
`Sigma` — a joint prediction region, a multivariate test, a BLUP with a
singular `G` — is then built on nothing.

That is a statistical property of the model at small k, and no search algorithm
changes it.  The library already ships the tool for it (`lkj_eta`), which is
what Finding 4 measures.

## Finding 4 — `lkj_eta` is a penalty, and both its price and its benefit are measurable

`lkj_eta=2` adds `-(eta-1) log det R` to the objective.  Fitting the same data
both ways (120 replicates per cell):

| | unpenalised | `lkj_eta=2` |
|---|---|---|
| restricted likelihood at the returned `Sigma`, k=10 | the maximum | **0.55 lower (median)** |
| restricted likelihood at the returned `Sigma`, k=30 | the maximum | **0.24 lower (median)** |
| suboptimal by the likelihood's standard | 11.6% | **100%** |
| non-PD `Sigma`, k=10 -> k=30 | 20.8% -> 3.2% | **4.2% -> 0.0%** |
| median min-eigenvalue, k=10 -> k=30 | 0.0000 -> 0.255 | **0.410 -> 0.548** |
| median relative Frobenius error, k=10 | 0.908 | **0.763** |
| median relative Frobenius error, k=30 | 0.582 | **0.538** |

Both halves matter.  A penalised fit is *by construction* not a likelihood
maximum — 100% "suboptimal" is the correct measurement, not a defect — which
is exactly why the plan insists on calling it a penalty and not a prior.  And
it is worth the price: k=10's typical answer goes from **exactly singular** to
positive definite, and the estimate is closer to the truth in Frobenius norm at
both study counts.  A 1.1.6 discussion that treats `lkj_eta` as "a bit of
Bayesian shrinkage" would miss that it is doing real statistical work on a real
defect, and that at k=30 it costs almost nothing.

Both halves matter.  A penalised fit is *by construction* not a likelihood
maximum, so it can never be validated as one — which is exactly why the plan
insists on calling it a penalty and not a prior.  And it is worth the price:
the estimate is closer to the truth in Frobenius norm and the conditioning
improves by two orders of magnitude.  A 1.1.6 discussion that treats `lkj_eta`
as "a bit of Bayesian shrinkage" would miss that it is doing real statistical
work on a real defect.

## Finding 5 — pinning the correlations removes the search failure

Fitting with `rho=` fixed at the true correlations (120 replicates, k=10, J=3),
which also moves the fit off the Cholesky route and onto the `(tau2, rho)`
route:

| | free `rho` | pinned `rho` |
|---|---|---|
| suboptimal | 11.6% | **0.8% ±1.6** |
| gap max | 3.03 | 0.06 |
| non-PD | 20.8% | 28.3% |

With the correlation surface removed, a single start finds the maximum in
99.2% of replicates.  The search difficulty is therefore localised to the
**correlation** dimensions, not to the variances.

## Finding 6 — the parameterisations differ, and the difference names the culprit

Same 150 data sets, same seed, only the *reference*'s parameterisation changes
(the `hyper` row quotes the full 250-replicate cell for the conflict rate, the
other two the 150-replicate paired runs):

| reference | suboptimal | gap max | multi-start conflict |
|---|---|---|---|
| `hyper` (unconstrained, hyperspherical) | 10.7% | **3.03** | 8.4% |
| `chol` (free Cholesky factor) | 10.7% | 1.19 | **0.7%** |
| `external` (`(tau2, logit rho)` + cliff) | 10.7% | 0.73 | **78.0%** |

Read the last column, because it is the one that does not depend on how good
any single answer is: **the `(tau2, rho)` surface with the `1e5 + 1e7(...)`
cliff makes the reference's own six starts disagree in 78% of replicates**,
against 8.4% for the unconstrained map and 0.7% for a free Cholesky factor.  A
surface whose restarts disagree four times out of five is a surface on which a
single start cannot be trusted, and the cliff is what makes it ragged: every
point outside the cone is replaced by a value of a different order, so the
optimiser sees a wall and stops in front of it.

The other two columns say the same thing from the other side.  `gap max = 3.03`
for `hyper` is the largest gap *any* reference found, i.e. `hyper` is the only
one that reaches the best point in the worst replicates; `external` never
finds it there (0.73) and `chol` sometimes does (1.19).

Two conclusions follow, and they point in opposite directions from the obvious
one:

* `pimeta`'s own choice — the Cholesky route for J >= 3 — is the **best** of
  the three, so the parameterisation is not the defect.  It is the single
  start.
* The `(tau2, rho)` + cliff route, which `pimeta` still uses for J = 2 and
  whenever a parameter is pinned, is the worst of the three, so the cliff is a
  real cost and not a stylistic dislike.

## Finding 7 — one fit can emit hundreds of warnings


The `k10-J2-rho30` cell reported 248, 572 and 76 `RuntimeWarning`s in three
replicates (`invalid value encountered in sqrt`, then in `slogdet`, both from
`rma_mv.py`).  They come from evaluating the objective where a Cholesky row has
gone to zero to within rounding, so `diag(L L')` is a tiny **negative** number
and its square root is NaN.  The final answers are unaffected — across 1500
records no returned `Sigma` is non-finite and no reported variance is negative
— but a user fitting a J=2 model gets hundreds of warnings on a fit that
succeeded, and the harmless NaN is then rejected downstream by `slogdet`
rather than by design.

## Status after this run

| method | status | reason |
|---|---|---|
| `rma_mv` UN, k >= 30, well identified | **conditional** | 0.8% suboptimal, 1.2% boundary: acceptable inside this region |
| `rma_mv` UN, k = 10, J >= 3 | **conditional** | 10-12% of fits are worse local points; 20-25% return a singular `Sigma` |
| `rma_mv` UN, one tiny variance component | **conditional** | 19.2% suboptimal; the small component is not identifiable |
| `lkj_eta` | **penalty, measured** | never a likelihood maximum by construction; improves conditioning and Frobenius error |
| `rho=` pinned | **conditional** | 0.8% suboptimal, but 28% of the answers are singular |

No cell reaches *validated*: the boundary rate at k=10 is a property of the
model, and the search failure at k=10 is a property of the code.  Both have to
be addressed before the label is available.

## Consequences for 1.1.6

In the order the measurements justify:

1. **Evaluate the likelihood block-wise when the structure is block diagonal.**
   16.2 ms against 0.109 ms per evaluation at k=30, J=5, improving like `k^2`
   for larger studies.  A pure win with no statistical content, and it is what
   makes 2000-replicate runs affordable.
2. **Multi-start, or a start that is not the method-of-moments point.**  The
   reference needs six starts to be reliable where `pimeta` fails one time in
   ten, and `chol`'s restarts agree 99.3% of the time — so two or three starts
   on `pimeta`'s own parameterisation should recover most of the difference at
   a cost that is now known.
3. **Fix the Nelder-Mead fallback's budget.**  It triggers when one variance
   sits on its bound and the others do not, and then allows 100,000
   evaluations of an objective that costs 16 ms at k=30, J=5.  That is the
   reason the J=5 cells could not be measured here: a single unlucky replicate
   can cost minutes.
3. **Make the boundary a first-class answer.**  Return the conditioning of
   `Sigma` and a boundary flag, and document `lkj_eta` as the remedy with its
   likelihood cost printed.
4. **Replace the `1e5 + 1e7(1e-6-lam)^2` cliff and the `1e300` return in
   `_mv_loglik`** with one barrier of the same order as the likelihood, so that
   a line search cannot mistake a rejection for a feature.  Finding 6 gives the
   number: 78% restart disagreement against 0.7% on a smooth surface.
5. **Close the NaN path**: `sqrt(max(., 0))` in `from_chol_vec`, and reject a
   non-finite matrix before it reaches `slogdet`.

## Implemented in 1.1.6

All five are in, plus the fixture correction the measurement forced:

| item | measurement before | after |
|---|---|---|
| block-wise likelihood | 16.2 ms per evaluation (k=30, J=5) | **0.37 ms** |
| one fit at k=30, J=5 | 9.0 s | **1.0-1.7 s** |
| single start | 11.6% of fits below the maximum | second start on the opposite side of the box |
| stall fallback | up to 100,000 evaluations | 1,000 (unchanged for `GEN`, which needs it) |
| cone barrier | `1e5`, 78% restart disagreement | `1e2`, proportional to the deficit |
| `sqrt` of `diag(L L')` | 248-572 warnings per fit | clipped at zero |
| profile at `rho = -0.7` | metafor's interior point (1.3104116) | the boundary maximum (1.3813334) |

The re-measurement of this grid against 1.1.6 is the next entry; the numbers
above are the ones the changes were chosen from, not the ones they produced.

## Re-measured against 1.1.6 (same seed, same 250 replicates)

`python tools/compare_runs.py results results_v116`

| cell | suboptimal | gap max | non-PD | boundary | Frobenius (median) | fit time (median) |
|---|---|---|---|---|---|---|
| `k10-J2-rho30` | 2.4% → **0.0%** | 0.137 → **0.000** | 2.8% → **0.4%** | 38.0% → 38.0% | 0.729 → 0.729 | 0.09 → 0.39 s |
| `k10-J3-mixed` | 12.4% → **5.2%** | 2.596 → 2.596 | 20.4% → 18.0% | 19.6% → 12.0% | 0.867 → 0.869 | 0.25 → 0.88 s |
| `k10-J3-rho30` | 11.6% → **6.4%** | 3.034 → 3.034 | 20.8% → 23.6% | 22.8% → 18.8% | 0.908 → 0.908 | 0.26 → 1.06 s |
| `k10-J3-rho70` | 10.0% → **5.6%** | 0.824 → 0.824 | 25.2% → 25.2% | 25.6% → 22.8% | 0.717 → 0.717 | 0.29 → 1.10 s |
| `k30-J3-rho30` | 0.8% → 0.8% | 0.742 → 0.742 | 3.2% → 3.6% | 1.2% → 1.2% | 0.582 → 0.582 | 0.23 → 1.63 s |
| `k30-J3-tiny` | 19.2% → **8.8%** | 3.660 → 3.655 | 22.0% → 18.0% | 22.0% → 13.6% | 0.554 → 0.559 | 0.30 → 2.86 s |

Reading it:

* **Where the search was failing, it fails about half as often** (11.6 → 6.4,
  12.4 → 5.2, 10.0 → 5.6, 19.2 → 8.8).  Where it was already working
  (0.8% at k=30, J=3) it is unchanged, which is the control.
* **The boundary rate and the Frobenius error do not move** (38.0 → 38.0,
  0.908 → 0.908, 0.729 → 0.729).  They are properties of the model at these
  study counts, and a change that had moved them would have been a change to
  something else.
* **Fits cost 3 to 10 times more**, because there are two starts.  That is the
  price of the halved failure rate, and it is now *bounded* -- the fallback that
  used to run for minutes is capped.

### The J=5 cells, now measurable

| cell | reps | suboptimal | gap max | non-PD | boundary | Frobenius |
|---|---|---|---|---|---|---|
| `k30-J5-rho30` | 100 | 4.0% ±3.8 | **0.04** | 15.0% | 0.0% | 0.647 |
| `k30-J5-rho70` | 100 | 11.0% ±6.1 | **0.07** | 21.0% | 1.0% | 0.438 |

These could not be run at all before (one fit could take minutes).  They also
show why the *size* of a gap matters more than its existence: at k=30 with 15
variance parameters the surface is flat enough that "suboptimal" fires on
differences of 0.04 to 0.07 log-likelihood units, which is the same matrix to
any purpose.  A tolerance of `1e-6` counts a flat ridge and a wrong basin as
the same event; the table reports the maximum gap beside the rate for exactly
that reason.

### What is left, and it is not the same defect

Every residual failure is on a boundary corner.  The worst one (k=10, J=3,
replicate 230, gap 3.03) is a genuine local maximum of the Cholesky surface:

| | `tau2` | `rho` |
|---|---|---|
| pimeta 1.1.6 | (0, 0.164, 0.087) | (0, 0, 0.276) |
| reference | (0.036, 0.164, 0.110) | (−0.015, **−0.978**, 0.224) |

Both of pimeta's starts stop in the corner where the first variance is exactly
zero; the reference needs a correlation of −0.98 between the second and third
outcomes to escape it, and even then one of its own six starts misses.  Two
starts halve the failure rate; whether a third would reach these corners is a
question for the next grid, not for an argument.


## What this run does not test

* `lkj_eta` as anything other than a penalty: it is measured, not justified,
  and no interval built from a penalised fit is calibrated here.
* `sigma2=` profile likelihood (`profile.rma.mv`), where the pinned component is
  swept rather than held.
* correlated sampling covariances, moderators, three-level models, the `GEN`
  random-slope structure — each is a separate surface.
* whether a boundary estimate is the *statistically* right answer; Finding 3
  measures how often it happens, not whether it should.
* gradient norms at the returned point.  A finite-difference gradient in the
  external `(tau2, rho)` coordinates is the natural next diagnostic and it
  needs a `start=` entry point that the public API does not have yet.
