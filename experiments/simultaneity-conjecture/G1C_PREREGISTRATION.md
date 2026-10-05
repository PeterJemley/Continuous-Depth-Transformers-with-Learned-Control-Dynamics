# Preregistration G1c: does a trained control signal set how strongly the dynamics contract?

Date: 2026-10-05    Status: **draft, awaiting the author's approval.** Nothing below has been run.

Type: confirmatory for the primary outcome and its decision rule. Everything else is secondary or exploratory and labelled so.

Parent documents, both in this folder:
- `G1_NOTE.md`, section "Exploratory observations" and next conjecture 1;
- `CONJECTURE.md`, section "Dynamical regimes as voices".

**Relation to earlier results.**
- **G1 stays refuted.** It conjectured regimes of distinctly different curvature, and with additive control at this budget they did not appear. G1c does not revisit that.
- **G1c is a new conjecture, prompted by something G1 found by looking.** In all five of G1's trained models, the two control values differed about fivefold in how fast the vector field contracts the state, while their curvature barely differed. That observation was not predicted, so it is tested here:
  - on new trained models, seeds 3101–3105;
  - on validation windows that no G1 geometry used, windows 257–512;
  - against a threshold fixed before any of these models exists.
- **What G1c does not use.** G1's own models and windows are not part of the test.

## The claim

**When Model A is trained with exact gradients and its control u selects between predicting the next token (u = +1) and the token after next (u = −1), the vector field contracts the state at least twice as fast under u = −1 as under u = +1, in at least 4 of 5 trained models.**

### What contraction means here

- **Divergence and contraction rate.** The divergence of the vector field f at a state h is the trace of its Jacobian, ∂f/∂h. It is the rate at which a small volume of nearby states grows as the flow carries it. Negative divergence means nearby states are drawn together: the flow contracts. The contraction rate is minus the divergence.
- **What strong contraction does.** It makes trajectories that start apart end closer together. The block then forgets differences between its inputs, in the sense of contraction analysis for dynamical systems (Lohmiller & Slotine, 1998).

### The explanation behind it, and its status

- **The tentative explanation.** The token after next is harder to predict than the next token: 6.13 against 5.05 nats in G1. Fine detail of the current state helps less with it. A flow that contracts more strongly discards that detail and carries a smoother summary forward. So training on a control value that selects the harder, farther target would push the field under that value toward stronger contraction.
- **What G1c tests.** Only the dynamic claim, not this explanation. The design cannot separate "the farther target" from "the harder target"; that is a limitation, and a separate test.
- **Why the claim is directional.** The direction, u = −1 contracting more, came from G1. Stating it in advance makes the claim easier to refute: a large difference in the other direction counts against it.

## Competing hypotheses

| Hypothesis | What it predicts for the primary outcome |
|---|---|
| The claim: control sets the contraction rate | Ratio of at least 2, u = −1 contracting more; ratio below 1.25 in the shuffled-control models |
| G1's difference was luck of its seeds or windows | Ratio below 1.25 on new seeds and windows |
| The control nudges the dynamics without changing contraction much | Ratio below 1.25, even though the model uses u |
| Contraction differs, but in the other direction | Ratio below 1 |
| Any input change alters contraction, trained or not | The untrained models, or the shuffled ones, show differences too, and a must-fail control fails |
| The trace estimator is biased or too noisy | The analytic checks fail, or two probe sets disagree |
| The model ignores u | The manipulation check fails: G1c is then untestable, not refuted |

## Refuting result

The claim is refuted if the contraction ratio is below 1.25 in at least 4 of 5 trained models, and the manipulation check shows those models use u.

## Design

### Models and training

Everything is as in G1, except the seeds:
- **Model and gradients:** the paper's `HybridODETransformer`, trained with exact gradients.
- **Control:** u = (±1, 0, 0, 0) selects the task per sequence. The control enters additively through `control_embed`, inside the vector field only.
- **Training:** 2,879 steps at peak learning rate 2.4e-3, batch 32 × 128, AdamW, warm-up then cosine decay, gradient clipping at 1.0, fp16 mixed precision, on the pinned WikiText-103 token files.
- **Main models:** seeds 3101 to 3105. These seeds have not been used before.
- **Shuffled-control models:** the same seeds, with u drawn independently of the task. These are must-fail control A.
- **Untrained models:** the main seeds' initial weights, before any training step. These are must-fail control B.

10 training runs in all, as in G1. The code is G1's `g1_regimes.py` training function, unchanged.

### Measuring contraction

- **Windows:** validation windows 257 to 512, numbered from 1, in G1's window definition: non-overlapping 128-token inputs. G1 measured geometry on windows 1 to 256 only.
- **Integration:** dropout off. Each window is integrated by dopri5 at tolerance 1e-5, once at u = +1 and once at u = −1.
- **Depths:** the divergence is estimated at depths 0, 0.25, 0.5, 0.75 and 1, as in G1's measure.
- **Estimator:** Hutchinson's estimator (Hutchinson, 1990), as used for continuous normalising flows (Grathwohl et al., 2018). Divergence is approximated by zᵀ(∂f/∂h)z, averaged over 8 random ±1 probe vectors z, with each Jacobian-vector product computed exactly by forward-mode differentiation. The trace is taken over the whole sequence state, because attention couples the tokens, and divided by the number of tokens.
- **Contraction rate:** c(w, u) is minus that per-token divergence, averaged over the five depths, for window w and control value u.
- **Model-level value:** c̄(u) is the mean of c(w, u) over the 256 windows.

## Primary outcome

For each trained main model, the **contraction ratio** R = c̄(−1) / c̄(+1).

### Calibration: why the rule needs explicit cases

**A ratio is only meaningful when both rates are clearly away from zero.** Before this draft, the contraction measure was run on five **untrained** full-size models: G1's seeds 3001–3005 at initialisation, on the first 64 windows of the test-mode token file. Neither those seeds nor those windows are used by G1c.
- Contraction rates were tiny, between −0.018 and +0.015 per token, and their sign varied.
- Ratios between them ranged from 0.05 to 1.2 and meant nothing.

G1's trained models, by contrast, had rates of 0.87 to 8.5. So the rule treats rates near zero explicitly. The cut-off, 0.1 per token, is:
- more than 5 times the largest untrained magnitude;
- more than 8 times smaller than the smallest trained one seen in G1.

### How R is scored

| Case | Scored as |
|---|---|
| c̄(+1) ≥ 0.1 and c̄(−1) ≥ 0.1 | R = c̄(−1) / c̄(+1) |
| c̄(+1) < 0.1 and c̄(−1) ≥ 0.2: u = −1 contracts, u = +1 is neutral or expanding | Counts as R ≥ 2 |
| c̄(+1) < 0.1 and 0.1 ≤ c̄(−1) < 0.2 | Counts as intermediate: neither rule's threshold |
| c̄(−1) < 0.1: u = −1 does not contract | Counts as R < 1.25 |

## Decision rule

**Preconditions.** If any fails, the result is "untestable", whatever R shows.

1. **Manipulation check: the main models use u.** The wrong control (u = −1) raises next-token validation loss, on the whole validation split, by at least 0.5 nats in at least 4 of 5 main models.
2. **Analytic check: the divergence code is right.** Synthetic linear fields f(h) = A h, with known trace, are passed through the same code, Jacobian-vector products included.
   - **Diagonal A:** ±1 probes make Hutchinson's estimator exact, so the code must return the trace to within 1e-5 relative error.
   - **Dense random A:** averaged over 256 windows and 8 probes, the estimate must lie within 2% of the trace.
   - **G1's curvature check** is also repeated, for the secondary measures.
3. **Must fail A: trained, uninformative control.** The shuffled-control models give R between 0.8 and 1.25, scored by the table above, in at least 4 of 5.
4. **Must fail B: no training.** In the untrained models, |c̄(−1) − c̄(+1)| is below 0.1 in at least 4 of 5. A difference rather than a ratio is used, because the untrained rates sit near zero.
5. **Must match A: window sampling.** In every main model, log R computed from odd-numbered windows and from even-numbered windows differ by less than 0.1.
6. **Must match B: estimator noise.** Every main model is measured a second time with a different set of probe vectors. The two values of log R differ by less than 0.05.

| Outcome | Condition |
|---|---|
| Supported: the control sets the contraction rate | R ≥ 2, as scored above, in at least 4 of 5 main models |
| Refuted | R < 1.25 in at least 4 of 5 main models. This includes contraction differences in the other direction |
| Intermediate | Anything else. All five values are reported, with the controls beside them |

**Why these thresholds.**
- A factor of 2 is the threshold amendment A1 set for "distinct regimes" in G1. Using the same standard means curvature and contraction are judged alike.
- A factor of 1.25 is the threshold A1 set for "no more than a nudge".
- G1's models gave ratios of 4.4 to 5.7. A replication would clear 2 comfortably; a result near 1 would refute.

**Sensitivity.** G1's window-to-window standard deviation of c was 0.13 to 0.36. With 256 windows, the standard error of each c̄ is about 0.02, against rates of about 1 to 8. Window sampling cannot move R across a threshold; must-match A checks this directly.

## Secondary and exploratory outcomes

- **S1, contraction by depth:** c̄ at each of the five depths, per control value.
- **S2, gain or shape:**
  - **Question.** Does u scale the whole Jacobian, or change how much of it is contraction?
  - **Measure.** The normalised contraction is −trace(∂f/∂h) / (√n ‖∂f/∂h‖_F), where n is the state's dimension. It lies between −1 and 1 and does not change if the Jacobian is multiplied by a constant. The Frobenius norm is estimated from the same probes, as the mean of ‖(∂f/∂h)z‖².
  - **Reading.** The ratio of normalised contraction between control values is reported. A ratio near 1 alongside R ≥ 2 would mean the control mostly sets the Jacobian's overall gain.
- **S3, field or states:**
  - **Question.** Does the difference come from the field, or from the states each control value leads to?
  - **Measure.** Each control value's field is also evaluated at the other control value's trajectory states. If c changes little when only the states are swapped, the field itself differs.
- **S4, u = 0, exploratory:** c̄(0), and where it lies between c̄(+1) and c̄(−1) on a log scale. In G1 it lay between them.
- **S5, G1's measures on new seeds and windows:**
  - the curvature ratio;
  - the curvature-profile classifier;
  - speed.
  
  This checks whether G1's refutation, and its other observations, replicate.
- **S6, task losses and α,** as in G1.

## Sanity checks before trusting results

- Every data file's fingerprint matches the manifest.
- The batch check and the same-model check from G1 run again for the new seeds.
- Every number in the report is computed by code from saved JSON records. `g1c_analyze.py` is written and committed before any real run, and tested on synthetic records covering every branch of the decision rule.

## Compute, where it runs, and stopping rule

- **Where:** one Colab job, `g1c_contraction`, on a T4, resumable like G1, with its own Drive folder.
- **Cost:**
  - 10 training runs at about 0.09 hours each;
  - measurement of 10 trained and 5 untrained models, about 2 minutes each;
  - about 1.2 hours in all.
- **Stopping rule:** exactly these runs and measurements, each once. A crashed run resumes from its checkpoint. Nothing is added after seeing results except as a dated, exploratory entry in the deviation log.

## What each outcome would teach

| Outcome | What we would learn | Next |
|---|---|---|
| Supported | A trained control signal sets how strongly the dynamics contract, replicated on new seeds and windows. The regimes picture holds in a different geometric form from the one the paper proposed: contraction, not curvature. S2 and S3 say whether it is gain or shape, field or states. | Test the explanation: separate "farther target" from "harder target" with a third task of matched difficulty. Then G2, with contraction as the regime measure. |
| Refuted, ratio near 1 | G1's contraction difference did not replicate. It belonged to those seeds or windows, or to chance. | Return to G3, multiplicative control, for the regimes picture. |
| Refuted, ratio below 1 | Contraction is set by the control, but not in the predicted direction. The direction in G1 was not stable, and the explanation offered for it is wrong. | A new conjecture about what sets the direction, from the secondary measures. |
| Intermediate | The difference exists but is smaller or less consistent than G1 suggested. | S1 and S3 locate it, before a new test. |
| Untestable: a must-fail control fails | Contraction differences arise without informative training. The measure cannot attribute them to the control. | Redesign the measure. |
| Untestable: manipulation or analytic check fails | The setup or the code did not do what was intended. | Fix and rerun as a new preregistered test. |

## Limitations known now

- **The design follows G1, with the same limits:**
  - 10% of the training budget;
  - one learning rate, one model size and one dataset;
  - additive control, two control values and two tasks.
- **Difficulty is confounded with the target.** The two tasks differ in difficulty as well as in how far ahead they look, and this design cannot separate the two.
- **The trace says little about the rest of the spectrum.** The trace is one number. Two Jacobians with the same trace can have very different eigenvalues. S2 adds only the Frobenius norm.
- **The threshold is reused, not derived.** The factor of 2 comes from amendment A1, a judgement made for curvature, and is reused here for consistency.
- **The direction was observed, not explained.** The direction of the claim came from G1's observation. The explanation offered for it is tentative and is not tested here.

## Deviation log

None yet.

## References

Grathwohl, W., Chen, R. T. Q., Bettencourt, J., Sutskever, I., & Duvenaud, D. (2018). *FFJORD: Free-form continuous dynamics for scalable reversible generative models*. arXiv. https://doi.org/10.48550/arxiv.1810.01367

Hutchinson, M. F. (1990). A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines. *Communications in Statistics: Simulation and Computation, 19*(2), 433–450. https://doi.org/10.1080/03610919008812866

Lohmiller, W., & Slotine, J.-J. E. (1998). On contraction analysis for non-linear systems. *Automatica, 34*(6), 683–696. https://doi.org/10.1016/s0005-1098(98)00019-3
