# Preregistration G1d: does the model need the farther reading's strong contraction?

Date: 2026-10-06    Status: **approved by the author on 2026-10-06, final.** Nothing below had been run when this version was committed. From here on, the only changes are dated entries in the deviation log.

Type: confirmatory for the primary outcome and its decision rule. Everything else is secondary or exploratory and labelled so.

Parent documents, both in this folder:
- `G1C_NOTE.md`, next conjecture 1;
- `CONJECTURE.md`, section "Dynamical regimes as voices".

**Relation to earlier results.** G1c found that when the control u asks for the token after next (u = −1), the ODE block contracts the state 4 to 10 times faster than when it asks for the next token (u = +1). That shows the difference exists. It does not show the model needs it: the contraction could be a by-product of training, with no work to do. G1d intervenes on it to find out. Whatever G1d shows, the results of G1 and G1c stand.

## The claim

**The strong contraction under the farther reading does work the model needs. If training caps the ODE block's contraction under u = −1 near the level the next-token reading uses, prediction of the token after next gets worse, by at least 0.05 nats more than prediction of the next token does.**

### The explanation behind it

- **The tentative explanation, from G1c's preregistration.** Predicting further ahead is helped by discarding fine detail of the current state and carrying a smoother summary forward. Contraction discards detail. If that is why the farther reading contracts, preventing the contraction should leave the farther task with detail it cannot use, and its loss should rise.
- **Why the next-token task is the comparison.** The next-token reading already contracts weakly, at 0.8 to 1.9 per token in G1c, below the cap. So the cap barely touches it directly. Comparing the change in the two tasks' losses separates a cost specific to the capped reading from any general cost of training with a penalty.

## Competing hypotheses

| Hypothesis | What it predicts |
|---|---|
| The claim: contraction is functional for the farther task | The farther task's loss rises by at least 0.05 nats more than the next-token task's |
| Contraction is a by-product | Capping it costs neither task anything material |
| Contraction is needed, but other layers can take it over | Also little cost. The design cannot separate this from a by-product; the outcome table says so |
| The penalty's machinery changes training by itself | The sham arm, whose penalty never binds, differs from the control arm |
| The cap does not take hold | The manipulation check fails: G1d is then untestable, not refuted |
| Any constraint on the u = −1 dynamics would cost the farther task, not specifically one on contraction | The claim's prediction, with no way to tell the two apart here. Listed as a limitation and a next test |

## Refuting result

The claim is refuted if the cap takes hold (the manipulation check) and the 95% interval for the primary contrast lies wholly below 0.05 nats.

## Design

### Arms

All arms use G1c's training exactly: the paper's `HybridODETransformer` with exact gradients, u = (±1, 0, 0, 0) selecting the task per sequence, 2,879 steps at peak learning rate 2.4e-3, batch 32 × 128, on the pinned WikiText-103 token files. They differ only in a penalty added to the training loss.

| Arm | Penalty |
|---|---|
| **Control** | None. This is G1c's main training, on new seeds. |
| **Capped** | λ · P, where P penalises contraction above a cap of 2.0 per token, on u = −1 sequences only |
| **Sham** | The same penalty computation and λ, with a cap of 10⁶, which never binds. Its gradient is exactly zero, so it differs from the control arm only in the extra random draws and computation. It serves as the null contrast. |

- **Seeds:** 3201 to 3205 for every arm. These seeds have not been used before. A seed fixes the initial weights, the data order and the tasks, so the three arms of a seed differ only in the penalty.
- **Runs:** 15 main runs.

### The penalty

- **When and where.** At every training step, for the sequences in the batch given u = −1, at the four states the Euler solver visits (depths 0, 0.25, 0.5 and 0.75).
- **The estimate.** The contraction rate per token is estimated as c = −zᵀ(∂f/∂h)z / T, with one random ±1 probe vector z per sequence and state, as in Hutchinson's estimator. The vector-Jacobian product is computed by backpropagation with the graph kept, so that the penalty itself can be trained through, as in regularised neural ODEs (Finlay et al., 2020).
- **The penalty.** P is the mean over those sequences and states of max(0, c − 2.0)².
- **Its own random generator.** The probes come from a separate random generator, so they do not shift the dropout masks or the data order.
- **Why a cap of 2.0.** It lies just above the next-token reading's contraction in G1c (0.8 to 1.9) and far below the farther reading's (7.8 to 9.2). So it asks the farther reading to contract no more than the nearer one does, and leaves the nearer one alone.

### Choosing λ, before the main runs

- **The grid.** λ is chosen from {0.01, 0.1, 1, 10} by a pilot on seed 100, which is not a main seed.
- **The rule.** The capped arm is trained once at each λ, and its contraction is measured as in G1c. The chosen λ is the smallest at which the pilot's c̄(−1) is at most 3.0.
- **What the rule looks at.** Only the measured contraction, never a loss, so tuning cannot favour either outcome.
- **If no λ in the grid qualifies,** the main runs are not made, and G1d is reported as untestable: the cap could not be imposed.

### Measuring

- **Losses.** Both tasks' validation losses on the whole validation split, in fp32, each with its right control value: the next-token task with u = +1 and the farther task with u = −1. The test split is not used.
- **Contraction.** Measured exactly as in G1c (dopri5 at tolerance 1e-5, five depths, eight probes) on validation windows 513 to 768, which no earlier geometry used.

## Primary outcome

For each seed:

D = (farther-task loss, capped − control) − (next-token loss, capped − control)

D is how much more the cap costs the farther task than the next-token task. Positive D means the farther task is hurt more.

## Decision rule

**Preconditions.** If any fails, the result is "untestable", whatever D shows.

1. **The cap was imposed.** A λ in the grid met the pilot rule.
2. **Manipulation check: the cap takes hold.** In at least 4 of 5 capped models, c̄(−1) is at most 3.0, measured as in G1c.
3. **Positive control: there is a contraction difference to remove.** In at least 4 of 5 control-arm models, the contraction ratio R = c̄(−1) / c̄(+1) is at least 2, scored as in G1c. This is also a second replication of G1c.
4. **The models use u.** In the control arm, the wrong control costs at least 0.5 nats on the next-token task in at least 4 of 5 models, as in G1.
5. **Null contrast: the penalty's machinery alone does nothing.** The 95% interval for D between the sham and control arms lies wholly within ±0.05 nats.
6. **Analytic check of the training penalty.** On a linear field f(h) = A h with a known trace, the training-time estimator, averaged over many probes, gives the trace within 2%. Its gradient with respect to A matches the analytic gradient of the trace, the identity, within 2%.

The interval for D is a t-interval at 95% with 4 degrees of freedom (critical value 2.776), over the 5 seeds.

| Outcome | Condition |
|---|---|
| Supported: the contraction is functional | The interval for D lies wholly above +0.05 |
| Refuted: no material differential cost | The interval for D lies wholly below +0.05 |
| Some effect, size unclear | The interval excludes 0 but meets neither rule |
| Inconclusive | Everything else |

**Why 0.05 nats.** It is the margin T1 used for "material", kept for consistency. It is about a sixth of what the farther task loses when the model cannot tell the tasks apart (the shuffled-control models: 6.45 against 6.13 in G1 and G1c). In G1 and G1c, the seed-to-seed spread of the difference between the two tasks' losses was 0.004 to 0.005 nats. So the interval's half-width should be near 0.01, and the design can resolve the margin.

**Sensitivity.** If the standard deviation of D exceeds 0.04 nats, as in T1's rule, the report says the design could not detect 0.05 reliably, and "refuted" is not claimed even if the interval fits.

## Secondary and exploratory outcomes

- **S1, each task separately.** The change in the farther-task loss and the change in the next-token loss, capped minus control, each with its own 95% interval.
- **S2, how the cap was met.** Contraction under both control values in the capped models, by depth.
  - **The question.** Did the model lower u = −1's contraction only, or also raise or lower u = +1's?
  - **Also reported:** the G1c ratio R for the capped arm.
- **S3, the penalty during training.** The penalty's value every 50 steps. Did the cap bind throughout, or only early?
- **S4, other geometry in the capped models.** G1's curvature ratio, speed and the curvature-profile classifier, measured on the same windows.
- **S5, the wrong-control costs and α,** for all arms.
- **S6, training time per arm,** to show what the penalty costs in compute.

## Sanity checks before trusting results

- Every data file's fingerprint matches the manifest.
- G1's batch check and same-model check pass for the new seeds.
- **The penalty's probes don't disturb training.** Before training, the sham penalty's gradient is checked to be exactly zero on one batch. The control and sham arms must start with identical weights and see identical first batches.
- Every number in the report is computed by code from saved JSON records. `g1d_analyze.py` is written and committed before any real run, and tested on synthetic records covering every branch of the decision rule.

## Compute, where it runs, and stopping rule

- **Where:** one Colab job, `g1d_functional`, on a T4, resumable like G1c, with its own Drive folder.
- **Cost:** the penalty roughly doubles a step's cost, which the test runs will measure.
  - up to 4 pilot runs;
  - 5 control runs at about 0.09 hours;
  - 10 penalised runs (capped and sham) at about 0.2 hours;
  - 15 contraction measurements at about 1.5 minutes;
  - about 3 to 3.5 hours in all. Two Colab sessions may be needed; the job resumes.
- **Stopping rule:** exactly these runs and measurements, each once. A crashed run resumes from its checkpoint. Nothing is added after seeing results except as a dated, exploratory entry in the deviation log.

## What each outcome would teach

| Outcome | What we would learn | Next |
|---|---|---|
| Supported | The farther reading's strong contraction does work the model needs at this budget, and the rest of the network does not readily take it over. The regime is a mechanism, not a by-product. | Separate "farther" from "harder" (G1c next conjecture 2), and test whether any constraint of matched size on the u = −1 dynamics costs as much (the specificity rival). |
| Refuted | Capping the contraction costs the farther task nothing material. Either it was a by-product, or other layers do the same work when the ODE block cannot. G1c's regime is real but not, by this test, needed. | Find out which of the two: cap the contraction of the whole stack after the embedding, not only the ODE block, as a new test. |
| Some effect, size unclear | The cap costs the farther task something, but whether it is material is not settled. | More seeds, preregistered as a new test. |
| Untestable: the cap could not be imposed, or did not take hold | The penalty cannot hold this property down at these strengths. | Redesign the intervention, for example by constraining the field's form directly. |
| Untestable: the null contrast fails | The penalty's machinery changes training by itself. | Fix the implementation before any conclusion. |
| Untestable: the positive control fails | The contraction difference did not appear in this run's control arm, so there was nothing to remove. G1c's replication would itself be in question. | Report it as a failed replication of G1c, and look for why. |

## Limitations known now

- **Same limits as G1c:**
  - 10% of the training budget, one learning rate (not re-tuned for the penalty), one model size and one dataset;
  - additive control, two tasks;
  - farther and harder confounded.
- **Specificity.** "Supported" would show that capping contraction costs the farther task. It would not show that contraction is special: a constraint of similar strength on some other property of the u = −1 dynamics might cost as much. That needs its own test.
- **Necessity versus sufficiency.** "Refuted" cannot tell a by-product from a function that other layers take over.
- **The cap is set at Euler states, measured along the exact flow.** The cap acts at the four states the trained model's Euler steps visit, while contraction is measured, as in G1c, along dopri5's exact flow. The manipulation check measures whether the cap holds in G1c's sense.
- **The cap level is a judgement.** 2.0 per token is tied to G1c's next-token range, not derived.

## Deviation log

### Implementation details fixed before any real run, 2026-10-06

Results seen before this entry: only test-mode runs of `g1d_functional.py` on tiny models (width 32, 60 steps), unit tests of the penalty, and synthetic records for the analysis. No real G1d run had started.

- **Code.**
  - `g1d_functional.py` runs the pilot, the three arms and the measurements.
  - `g1d_analyze.py` applies the decision rule and writes `G1D_RESULTS.md`.
  - The Colab job is `colab/jobs/g1d_functional.py`.
- **Changes to G1's training function.** `g1_regimes.train_g1` gained two optional arguments: whether u tells the task, and a penalty callable. The exact Euler forward can now keep its states for the penalty, when a flag is set on the block. With the defaults, G1 and G1c's training is unchanged.
- **The penalty is computed in fp32,** outside mixed precision, so that a half-precision overflow cannot turn the sham penalty into a non-number. Its probes come from a generator seeded per step, so they are reproducible on resume and do not touch the global random state.
- **Dropout in the penalty.** The penalty evaluates the vector field in training mode, so its forward pass draws dropout masks. The sham arm therefore differs from the control arm in those draws, which is the difference the null contrast is meant to measure.
- **The pilot** trains λ values in increasing order and stops at the first that qualifies, so it makes at most 4 runs. Its tag is `pilot_lam<λ>`, and it is measured on the same windows as the main runs.
- **The analytic check of the penalty uses 65,536 sequences.** With 4,096, the expected sampling error of the gradient (about 2.2%) was close to the 2% limit, so a correct estimator could fail by chance. With 65,536 it is about 0.5%, and the check gave 0.02% (trace) and 0.58% (gradient).
- **Tests before this entry.**
  - The sham penalty was exactly zero with all-zero gradients on a batch.
  - A few optimiser steps on the capped penalty alone, with a cap set below a tiny model's contraction, drove the contraction below the cap within 10 steps, so the penalty pushes in the intended direction.
  - The job ran end to end on tiny models. Their contraction stays far below the cap, so the capped and sham arms coincided there, as they should.
  - A second run of the job skips finished work, as in G1c.
  - `g1d_analyze.py` gave the expected verdict on synthetic records for all 14 branches: supported; refuted (twice); some effect (twice); inconclusive; inconclusive for insensitivity; and untestable through each of the six preconditions, with the "no λ" case reported separately.
- **Measured cost.** In test mode on a CPU, a penalised run took about twice as long as a plain one.

## References

Finlay, C., Jacobsen, J.-H., Nurbekyan, L., & Oberman, A. M. (2020). *How to train your neural ODE: The world of Jacobian and kinetic regularization*. arXiv. https://doi.org/10.48550/arxiv.2002.02798

Hutchinson, M. F. (1990). A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines. *Communications in Statistics: Simulation and Computation, 19*(2), 433–450. https://doi.org/10.1080/03610919008812866
