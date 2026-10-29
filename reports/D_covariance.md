# D — the UN covariance, screening run

Family **D** of the research plan: when `rma_mv` fits an unstructured
between-study covariance `Sigma` (J(J+1)/2 parameters), is the matrix it
returns

1. a covariance matrix at all — symmetric, positive definite, variances >= 0;
2. the **maximiser** of the likelihood it claims to maximise;
3. the same matrix whatever parameterisation is used to reach it;
4. and, only after those three, close to the truth?

These are four different questions and they are answered by four different
measurements.  A convergence flag answers none of them: every "wrong" fit
reported in this document came back with `converged=True`.

## Design

| field | value |
|---|---|
| harness | metabench 0.1.0 |
| implementation | pimeta 1.1.5 |
| seed | 20261029 |
| cells | 8 (see below) |
| replicates | 250 per cell, 100 for the J=5 cells (see the cost note) |
| level | irrelevant here — the estimand is the matrix, not an interval |

The data-generating mechanism is the model itself: `theta_i ~ N(0, Sigma)`,
`y_i ~ N(theta_i, diag(se^2))`, with `Sigma = D R D`, `D = diag(sd)`.  Sampling
covariances are **diagonal** and the means are **zero** on purpose: a
correlated `V` or a moderator would give a second explanation for a
disagreement about `Sigma`, and the point of a screening run is to have one.

| cell | k | J | what it isolates |
|---|---|---|---|
| `k10-J2-rho30` | 10 | 2 | three parameters; the case where the per-pair box *is* the cone |
| `k10-J3-rho30` | 10 | 3 | the smallest J where the box is no longer the cone |
| `k10-J3-rho70` | 10 | 3 | a nearly collinear set of outcomes |
| `k10-J3-mixed` | 10 | 3 | correlations (0.5, -0.4, 0.1): sign and magnitude both vary |
| `k30-J3-rho30` | 30 | 3 | three times the studies: search problem or information problem? |
| `k30-J3-tiny` | 30 | 3 | one variance component 100x smaller than the others |
| `k30-J5-rho30` | 30 | 5 | 15 parameters; the largest screening cell |
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
`l(Sigma_pimeta)` and `l(Sigma_reference_best)`.  Their difference is a
statement that needs no shared code to be believed.  The reference is not
allowed to see `pimeta`'s answer and starts from the data, from six
deterministic perturbations of a method-of-moments start.

Two implementation notes that turned out to matter:

* **The random-effects covariance is block diagonal by study**, not
  `X Sigma X'`.  The latter couples any two rows that share an outcome,
  including rows from different studies, which is a different model.  Getting
  this wrong produced a 4.4-unit disagreement with `pimeta`'s own likelihood
  that looked like a software bug for an hour.
* **The likelihood is evaluated block by block**: `M = blockdiag(V_s + Sigma)`,
  so `X'M^-1 X = sum_s M_s^-1`.  That is `O(k J^3)` instead of `O((kJ)^3)`,
  and it is what makes a reference that runs once per replicate affordable.

### A guard the harness earned the hard way

The first pass reported `pimeta` as suboptimal in 11.6% of replicates at
k=10, J=3, with one gap of **2658** log-likelihood units.  The gap was real, but
it was the *reference* that was wrong: one start had wandered to
`tau2 ~ 1e43`, where `log|M|` and `log|X'M^-1 X|` are two numbers of order
`kJ log(lambda)` that cancel **past** the true minimum in floating point.  The
reference reported `nll = -2648.9` on data whose best possible value is
`9.27` — a likelihood better than any likelihood.

There is no correct way to score that number, so the region is refused
(`UNLikelihood.nll` returns `1e30` beyond `1e4 x` the data's own scale) and
the case is a regression test (`tests/test_reference_guards.py`).  The lesson
generalises: **a measurement system that can be 2658 units better than the
truth would have declared every honest fit suboptimal**, and the only reason
this was caught is that the harness reports what the reference did as well as
what the library did.

## Cost

`rma_mv` at k=30, J=5 takes about nine seconds per fit, against 0.03 s at
k=30, J=3 — the same dataset size, four times the observations, **300x** the
time.  The cause is structural: the objective materialises the whole
`kJ x kJ` matrix and inverts it per evaluation, which is `O((kJ)^3)`, while the
model is block diagonal and costs `O(k J^3)`.  Measured on one dataset with the
same likelihood evaluated block-wise, `pimeta`'s per-evaluation cost is about
116x the block cost (12.7 ms against 0.109 ms), and one J=5 fit costs as much
as 83,000 block evaluations where an L-BFGS-B run needs hundreds.

This is why the J=5 cells carry 100 replicates and the rest carry 250: the
budget went to cells where a proportion can be estimated, not to ones where the
software is the bottleneck.  It is also the first entry for the 1.1.6
performance baseline.

## Results

*(filled from `python -m metabench.reports.table results --match covariance`
once the grid finishes)*

## What this run does not test

* `lkj_eta` as anything other than a penalty: it is measured, not justified.
* `sigma2=` with a profile likelihood (`profile.rma.mv`), where the pinned
  component is swept rather than held.
* correlated sampling covariances, moderators, three-level models, and the
  `GEN` random-slope structure — each is a separate surface.
* whether a boundary estimate is *statistically* the right answer.  That is a
  property of the model at small k, and no search algorithm changes it.
