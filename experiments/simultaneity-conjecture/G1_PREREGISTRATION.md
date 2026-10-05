# Preregistration G1: does a trained control signal partition the vector field into distinct dynamical regimes?

Date: 2026-10-05    Status: **approved by the author on 2026-10-05, final.** Nothing below had been run when this version was committed. From here on, the only changes are dated entries in the deviation log.

Type: confirmatory for the primary outcome and its decision rule. Everything else is secondary or exploratory and labelled so.

Parent document: `CONJECTURE.md` in this folder, section "Dynamical regimes as voices", conjecture G1.

**Relation to earlier results.** The paper claimed that "the control signal partitions the vector field into distinct dynamical regimes with different curvature characteristics." The WikiText-2 replication tested that claim, through solver effort, as H6 and refuted it for the model as built. That refutation stands whatever G1 shows. G1 tests a new conjecture on a new model, built differently:
- it is trained with exact gradients, after T1;
- its control signal is used throughout training, not attached afterwards;
- geometry is measured directly, not through solver effort.

## The claim

**When Model A is trained with exact gradients and its control signal selects between two computations the model needs, the two control values send the ODE block's trajectories through geometrically distinct regimes. Measured on held-out text, the mean trajectory curvature under one control value differs from that under the other by a large effect size, Cohen's d of at least 0.8, in at least 4 of 5 trained models.**

### The explanation behind it

The ODE block's vector field depends on the control u. If u selects a computation the network must carry out differently, training should shape the field so that each control value produces its own kind of flow. Trajectories under one value would then bend differently from trajectories under the other, so the same text would be carried through distinct regimes. If training instead finds a solution in which u only nudges the trajectories, with the same geometry under both values, the claim is false for this architecture.

## Competing hypotheses

| Hypothesis | What it predicts for the primary outcome |
|---|---|
| The claim: distinct regimes | d of at least 0.8 in trained models; small in the shuffled-control models |
| The control nudges without partitioning | d below 0.2, even though the model demonstrably uses u |
| The model ignores u | The manipulation check fails: wrong control costs little loss. G1 is then untestable, not refuted |
| Geometry differs only because u is added to the input | The shuffled-control models show separation as well, so the must-fail control fails |
| The curvature measure is wrong | The analytic checks fail |
| Noise from the choice of windows | The split-half check at a single u value shows separation |

## Refuting result

The claim is refuted if d is below 0.2 in at least 4 of 5 trained models, and the manipulation check shows those models do use u.

## Design

### What the control signal selects

A training signal is needed that the model must use u to meet. Otherwise a model that ignores u would refute G1 for a trivial reason. Each training sequence gets one of two tasks at random, and u says which:

| u (first coordinate; the other three are 0) | Task at every position i |
|---|---|
| +1 | predict the next token, i + 1, as in ordinary language modelling |
| −1 | predict the token after next, i + 2 |

Both tasks use the same text, so the two control values are two readings of one sequence, a minimal analogue of two voices. The text alone does not say which task is wanted, so the model can only do well on both by using u. The control enters as the paper's model has it: `control_embed(u)`, one value per sequence, added inside the vector field. Nothing else in the model sees u: layers 0–1 come before the ODE block, and layers 4–5 and the output head see only its output. So any use the model makes of u has to pass through the block's dynamics. G3 will test multiplicative control separately.

### Models and training

- **Architecture:** the paper's `HybridODETransformer` (`experiments/wikitext2-replication/paper_model.py`), restored verbatim.
- **Exact gradients:** trained with ordinary backpropagation through the four Euler steps, as arm AX in T1.
- **Output:** the shared output head serves both tasks. Only u tells the model which task it is doing.
- **Budget:** 2,879 steps, 10% of the WikiText-103 budget, as in T1.
- **Settings:** batch 32 × 128, AdamW, warm-up then cosine decay, gradient clipping at 1.0, fp16 mixed precision. The peak learning rate is 2.4e-3, the rate T1's sweep chose for AX. It is not re-tuned, and that is a limitation.
- **Data:** the pinned WikiText-103 token files and fingerprints. Each sequence carries 130 tokens, so that both tasks have targets at every position.
- **Main models:** seeds 3001 to 3005, with the task drawn at random per sequence and u telling the truth.
- **Shuffled-control models:** the same seeds and data order, but u drawn independently of the task, so it carries no information. These are the must-fail controls.

10 training runs in all.

### Measuring geometry

Measurement uses the first 256 windows of the validation split, with dropout off and the block integrated by dopri5 at tolerance 1e-5. Each window is integrated once at u = +1 and once at u = −1.

For each window, control value, token position i and each of 17 evenly spaced depths τ from 0 to 1:
- **Velocity:** v = f(h, τ, u), the vector field itself.
- **Acceleration:** a = (∂f/∂h)·v + ∂f/∂τ, computed exactly with one forward-mode Jacobian-vector product. The dynamics couple tokens through attention, so this is computed for the whole sequence state, then read per token.
- **Curvature:** κ = ‖a − (a·v̂)v̂‖ / ‖v‖², for token i, using its components of v and a. This is how sharply the trajectory turns, independent of its speed and of the scale α.

The window's curvature, κ̄(w, u), is the mean of log κ over its tokens and depths. The logarithm is chosen in advance because κ divides by ‖v‖², so a few points where the state barely moves could otherwise dominate a plain average. Points where ‖v‖ is exactly zero are excluded and counted; if they exceed 1% of all points in any model, the report says so. The plain average of κ is reported as secondary measure S0.

## Primary outcome

For each trained model, Cohen's d between u = +1 and u = −1 for κ̄. That is the difference in means over the 256 windows, divided by the pooled standard deviation across windows. The unit of analysis is the trained model, 5 of them.

The same windows are measured under both control values, so the comparison is paired. The primary d nonetheless uses the pooled standard deviation across windows, because the question is whether the two regimes differ by much compared with how much windows differ from each other, not merely whether the difference is consistent. The paired version, the mean difference divided by the standard deviation of the differences, is secondary measure S0b.

## Decision rule

**Preconditions.** If any fails, the result is "untestable", whatever d shows.

1. **Manipulation check: the main models use u.** In at least 4 of 5 main models, on the validation windows, validation loss on the next-token task is higher by at least 0.5 nats with the wrong control (u = −1) than with the right one (u = +1).
2. **Analytic check: the curvature code is right.** Synthetic vector fields whose flows are known curves are passed through the same code, Jacobian-vector product included. It must return the textbook curvature to within 0.1% relative error: a constant field (a straight line) gives 0, a rotation field (a circle of radius r) gives 1/r, and a rotation with constant drift along the axis (a helix of radius r and pitch 2πc) gives r / (r² + c²). One time-dependent field with a known answer is included, so that the ∂f/∂τ term is checked as well.
3. **Must fail: input injection alone does not create regimes.** The shuffled-control models give |d| below 0.2 in at least 4 of 5.
4. **Must match: window sampling alone does not create regimes.** At u = +1 alone, the odd-numbered windows against the even-numbered ones give |d| below 0.4 in every main model. With 128 windows per half, the standard error of d is about 0.125, so a true value of zero passes this in each model with probability about 0.999. The check fails only if the windows are far more heterogeneous than independent sampling allows, which would make the primary d unreliable.

| Outcome | Condition |
|---|---|
| Supported: distinct regimes | \|d\| at least 0.8 in at least 4 of 5 main models |
| Refuted: no distinct regimes | \|d\| below 0.2 in at least 4 of 5 main models |
| Intermediate: some separation | Anything else. All five values are reported, with the shuffled-control values beside them. |

The thresholds 0.2 and 0.8 are the conventional small and large effect sizes, chosen before any run. With 256 windows per control value, the standard error of each d is about 0.09. A model whose true d is zero then falls below 0.2 with probability about 0.97, and one whose true d is 0.8 or more stays far above 0.2, so sampling noise alone is unlikely to move a model from one side of the rule to the other.

## Secondary and exploratory outcomes

- **S1, which control value curves more,** reported per model. There is no directional prediction.
- **S2, curvature by depth:** κ averaged over windows at each of the 17 depths, per u. This shows where in the block the regimes differ.
- **S3, speed:** the mean ‖v‖ per u.
- **S4, divergence:** expansion or contraction of the state, the trace of ∂f/∂h estimated with 8 random probes (Hutchinson's estimator) at 5 depths, per u.
- **S5, solver effort:** dopri5 function evaluations per u at tolerance 1e-3, the paper's own measure, for comparison with H6.
- **S6, a regime classifier, exploratory:** a logistic regression that predicts u from the curvature-by-depth profile alone. It is trained on half the windows and tested on the other half.
- **S7, the intermediate control u = 0, exploratory:** curvature at u = 0, to see whether it lies between the two regimes or forms one of its own.
- **S8, task losses:** both tasks' validation losses for all models, so the cost of learning two readings is visible.
- **S9, curvature where the model actually computes, exploratory:** the same curvature at the four states the trained model's Euler steps visit (τ = 0, 0.25, 0.5 and 0.75), since dopri5 traces the field's exact flow, which the Euler-trained model only approximates.

## Sanity checks before trusting results

- Every data file's fingerprint matches the manifest.
- One training batch of each task is decoded to text, and its targets are checked to be the tokens at i + 1 and i + 2.
- AX's forward pass matches the adjoint version's exactly before training, reusing T1's same-model check.
- Every number in the report is computed by code from saved JSON records. `g1_analyze.py` is written and committed before any real run.

## Compute, where it runs, and stopping rule

- **Where:** one Colab job, `g1_regimes`, on a T4, resumable like T1, with its own Drive folder.
- **Cost:** 10 training runs at about 0.1 hours each, plus the geometry measurement, roughly 2 hours in all.
- **Stopping rule:** exactly these runs and measurements, each once. A crashed run resumes from its checkpoint. Nothing is added after seeing results except as a dated, exploratory entry in the deviation log.

## What each outcome would teach

| Outcome | What we would learn | Next |
|---|---|---|
| Supported | A trained control signal does partition the dynamics into regimes of different curvature, measured directly, in a properly trained model. The paper's geometric picture holds in this new setting; H6's refutation of the original model stands beside it. | G2: do regimes keep strands of text apart, and does their distinctness predict tracking? |
| Refuted, with the manipulation check passed | The model uses u, yet the two readings share one geometry. Additive control steers without partitioning. | G3: multiplicative control per voice, before giving up the geometric picture. |
| Intermediate | Some separation, short of distinct regimes. S2 shows where in the depth it appears. | G3, and a longer training budget, as a new preregistered test. |
| Untestable: the manipulation check fails | The model did not learn to use u at this budget. | A longer budget, or multiplicative control, as a new test. |
| Untestable: the must-fail control fails | Adding u to the input creates geometric differences by itself. The measure needs a baseline correction before it can be used. | Redesign the measure. |

## Limitations known now

- 10% of the training budget, and a learning rate taken from T1 without re-tuning for two tasks.
- Two control values and two tasks. Regimes for strands of meaning in natural text are G2's question, not this one.
- One model size, one dataset, and additive control only.
- Curvature summarises a trajectory's turning. It is not the whole of the geometry; S4 and S6 cover some of the rest.
- Geometry is measured along dopri5's exact flow of the learned field. The trained model computes with four Euler steps, which follow that flow only approximately; S9 looks at the Euler states.
- A difference in curvature between the two control values is necessary for distinct regimes but not sufficient for the polyphonic picture: G1 asks whether the regimes exist, not whether they carry separate strands of meaning.

## Deviation log

### Implementation details fixed before any real run, 2026-10-05

Results seen before this entry: only test-mode runs of `g1_regimes.py` on tiny models (width 32, 60 steps), built to test the code, and the analytic check, which uses no data. No real G1 run had started.

- **Code.** `g1_regimes.py` trains and measures; `g1_analyze.py` applies the decision rule and writes `G1_RESULTS.md`. Both are committed before any real run. The Colab job is `colab/jobs/g1_regimes.py`.
- **Exact gradients** are implemented as in T1: the model's `ODEFlowBlock.forward` is replaced on the instance by a version that calls `torchdiffeq.odeint`. The same-model check, run before training, confirms the exact and adjoint versions give the same weights, the same first-batch loss and the same downstream gradients with u given.
- **Random streams.** For a seed, the windows come from a generator seeded by the seed, the tasks from a second generator (seed × 10 + 1), and the shuffled models' u from a third (seed × 10 + 2). A seed's main and shuffled models therefore start from the same weights and see the same windows and tasks; only u differs.
- **Validation windows** are non-overlapping 128-token inputs, each with the two following tokens so that both tasks have targets at every position. The first 256 are the geometry windows. The manipulation check and S8 use all of them in the split. During training, the curves use the first 512.
- **Points excluded from log curvature.** Points where the speed or the curvature is exactly zero, or the curvature is not finite, are excluded and counted, as the plan says for zero speed. Zero curvature also gives an undefined logarithm, so it is treated the same way.
- **Curvature is computed in float64** from the float32 velocity and acceleration. The model and the integration stay in float32. The dopri5 step limit is 10,000.
- **S4** uses depths 0, 0.25, 0.5, 0.75 and 1, which are 5 of the 17 depths, and Rademacher probes. The trace is divided by the number of tokens. **S5** counts function evaluations per batch of 32 windows, as solver effort depends on the batch. **S6** trains on windows 1, 3, 5… and tests on windows 2, 4, 6…, numbered from 1; both control values of a window fall on the same side.
- **The analytic check** also includes the circle at three times the speed, which must give the same curvature, and a parabola traced by a time-dependent field, which checks the ∂f/∂τ term. The check runs with no data and passed before this entry: worst relative error 2.0e-5. Separately, the forward-mode acceleration was compared with a finite-difference estimate on an untrained tiny model; the median relative difference in curvature was 7e-5.
- **If the analytic check fails,** the job still trains and measures so that the cause can be found, but the verdict is untestable, as preregistered.

## References

Finlay, C., Jacobsen, J.-H., Nurbekyan, L., et al. (2020). *How to train your neural ODE: The world of Jacobian and kinetic regularization*. arXiv. https://doi.org/10.48550/arxiv.2002.02798

Kelly, J., Bettencourt, J., Johnson, M. J., et al. (2020). *Learning differential equations that are easy to solve*. arXiv. https://doi.org/10.48550/arxiv.2007.04504

### Proposed amendment A1, before any real result, 2026-10-05: awaiting the author's approval

Results seen before this entry:
- the test-mode runs above, on tiny models (width 32, 60 steps);
- the geometry of five **untrained** full-size models, seeds 3001–3005 at initialisation, on the first 64 validation windows of the test-mode token file. That file holds the start of the real WikiText-103 validation split.

No full-size model had been trained, and no real G1 run had started.

**The problem.** The preregistered statistic does not measure what the decision rule assumes.
- Each window's value is the mean of log curvature over 128 tokens and 17 depths. That average varies very little from window to window: the standard deviation across windows was 0.019 to 0.023 in the untrained models.
- Cohen's d divides by that spread, so any consistent shift between u = +1 and u = −1 gives a large d, however small the shift.
- In the untrained models, adding ±u through the randomly initialised `control_embed` alone gave d = −3.5, +6.0, −2.9, +0.9 and +3.3. Four of five exceed 0.8 in absolute value.
- The curvature itself differed by factors of only 1.07, 1.16, 1.06, 1.02 and 1.07.

Under the rule as written, the untrained models meet the "supported: distinct regimes" threshold, |d| of at least 0.8 in 4 of 5. Only the manipulation check would stop that verdict. The thresholds 0.2 and 0.8 are conventional small and large effects for spreads like those of individual measurements, not for averages this stable. The must-fail control could catch the problem only if the shuffled models kept their initial shift. If training shrank u's effect in the shuffled models, any modest shift in the main models would count as distinct regimes. Either way the rule would not answer the question asked.

**The amendment.** It measures the size of the difference in curvature itself, on a scale that means something geometrically.

- **A1a. Primary statistic: the curvature ratio.**
  - For each model, the log ratio is the mean over the 256 windows of κ̄(w, +1) − κ̄(w, −1), where κ̄ is the window's mean log curvature, as preregistered.
  - The curvature ratio is exp(|log ratio|), the larger curvature over the smaller, as a geometric mean. A ratio of 2 means trajectories under one control value turn twice as sharply as under the other.
- **A1b. Decision rule.**
  - Supported: distinct regimes, if the ratio is at least 2 in at least 4 of 5 main models.
  - Refuted: no distinct regimes, if the ratio is below 1.25 in at least 4 of 5 main models.
  - Intermediate otherwise.
  - Preconditions 1 (manipulation) and 2 (analytic) are unchanged.
- **A1c. Must fail.** The shuffled-control models must give a ratio below 1.25 in at least 4 of 5.
- **A1d. Must match.** In every main model, the log ratio computed from windows 1, 3, 5… and from windows 2, 4, 6… must differ by less than 0.1, a factor of about 1.1. This is far smaller than the gap between the two thresholds, log 2 − log 1.25 = 0.47.
- **A1e. Secondary measures.**
  - The original statistic, Cohen's d with the pooled standard deviation across windows, is kept as secondary S0c.
  - S2, S7 and S9 report log ratios in place of d. S0 and S0b are unchanged.

**Why these thresholds.**
- Random control embeddings at initialisation gave ratios of 1.02 to 1.16. The refutation threshold, 1.25, sits just above that range: below it, the trained control does no more than an untrained one.
- A ratio of 2, a doubling of how sharply trajectories turn, is the smallest difference I would call distinct regimes rather than a nudge. It lies well clear of both the initial range and the refutation threshold.
- These numbers were chosen after seeing untrained models only. They are a judgement, not a convention, and the report states them as such.

**Code.** `g1_regimes.py` is unchanged: it already records every window's mean log curvature at each control value, so it computes everything both rules need. `g1_analyze.py` applies the amended rule.
- It was tested on synthetic records built to fall in each branch: supported, refuted, intermediate, untestable through the manipulation check, and untestable through the must-fail control. Each gave the expected verdict.
- Synthetic records copying the five untrained models' log ratios come out "refuted", where the original rule would have said "supported".
