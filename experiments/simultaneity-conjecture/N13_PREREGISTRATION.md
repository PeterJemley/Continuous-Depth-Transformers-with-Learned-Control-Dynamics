# Preregistration N13: the mixture head's learning signal, separated from its output form

Date: 2026-10-22    Status: **approved by the author, 2026-10-22.** Being built and tested; no calibration run and no real N13 run has started.

Type: confirmatory for N13a (primary), N13b and N13c, each with its own decision rule. N13a and N13c use new models trained for this test; N13b re-measures N12's saved models. Everything else is secondary or exploratory and labelled so.

Parent documents: `N12_NOTE.md` (its section "Why the mixture head does more than escape the ceiling", and next conjecture 1), the necessity audit in `N12_PREREGISTRATION.md` (item 3), and `CRITICAL_REVIEW.md`, open problem 3.

**Relation to earlier results.**
- **What N12 found.** A head whose output is the noisy mixture, 0.2 softmax + 0.8 u, trained on the same 80%-noisy targets as the standard head, closed 71% of the gap to the ideal copy (N12a), and its inner softmax was 0.34 nats from head A on clean targets, against 1.40 for the standard head.
- **What N12 left open.** The necessity audit found, before the run, that the mixture head changes two things at once: what the head can represent, and the learning signal it trains on. N12 did not separate them. Its note proposed two arms, "reweighting without representation" and "representation without reweighting".
- **Why those two arms are the wrong design.** The two effects are not independent in training. For a target y, the gradient of the mixture loss −log q(y), with q = 0.2 p + 0.8 u and p = softmax(l), with respect to the logits l is

  r(y) · (p − e_y), where r(y) = 0.2 p(y) / q(y),

  and (p − e_y) is the gradient of the ordinary cross-entropy. r(y) is the posterior probability, under the head's own prediction, that y is a clean target rather than a random one. This is the chain rule, and it was checked numerically through the job's own code before N12 ran (to within 10⁻⁹; N12's deviation log, audit item 3). So a single-softmax head trained with the discounted cross-entropy, with the weight treated as a constant, receives *exactly* the gradient the mixture head's inner softmax receives. "Reweighting without representation" is the mixture head's inner softmax under another name, and "representation without reweighting" is the standard head read out in mixture form.
- **What follows.** The form can affect training only through the discounting. The separable questions are: whether the pipeline follows the identity in practice; whether the form, applied at readout to a head trained without the discounting, does anything; and whether the discounting works by telling targets apart, or only by making head B pull less on the shared network. N13 tests those three.
- **What stands regardless.** Every earlier verdict stands. N13 adds no claim about N12's verdicts; it says what they were made of.

## The conjecture

**The mixture head's advantage is its learning signal: a discount on each target by the estimated probability that it is clean, in context. The output form matters only at readout.**

- **The mechanism.** The discount r(y) is small for targets the head judges likely to be random and near 1 for targets it judges likely to be clean. At the start of training, when p is flat, r(y) is set by the word frequencies alone: it nearly ignores frequent tokens, which random draws from u mostly are, and keeps rare ones. As p sharpens, the discount becomes context-aware. The standard head, with no discount, fits every target equally and so fits the noise.
- **What it forbids.**
  1. A single-softmax head trained with the discounted cross-entropy differing from the mixture head's inner softmax by more than a replicate of the same training differs from it.
  2. The mixture form, applied at readout to the standard head, closing a material part of the gap.
  3. A head whose signal is scaled down by the prior probability of a clean target, 0.2, with no discrimination between targets, gaining a large part of the mixture head's improvement.
- **The rival it is built to exclude.** The mixture head's discount also lowers head B's gradient on the shared trunk: over targets drawn from the head's own prediction of the noisy target, r(y) averages exactly 0.2, so for a calibrated head the trunk receives about a fifth of head B's undiscounted gradient. N12 found that the standard noisy head costs head A 0.04 nats and the mixture head costs it little or nothing. So the gain could come from head B pulling less on a trunk that head A then shapes better, rather than from head B learning which targets to trust. A constant discount of 0.2 lowers the pull by the same factor without discriminating between targets. AdamW removes a constant scale from head B's own parameters (its per-parameter normalisation), so for head B's readout the constant changes nothing; it changes only head B's share of the trunk's gradient. That is what makes it the control for this rival.

## Competing hypotheses

| Hypothesis | N13a: rw matches mix | N13b: S(n80, mixture readout) | N13c: w02's gain in the inner reading |
|---|---|---|---|
| **The conjecture:** the context-aware discount does the work; the form matters only at readout | Matches | Near zero or negative | Small |
| The pipeline does not follow the chain rule (a bug, or an interaction with mixed precision or clipping) | Differs by far more than a replicate | Any | Any |
| The form does the work at readout: the standard head's softmax is already a fair estimate of the clean reading, and mixing it with u is what brings it to the copy | Matches | A third or more | Any |
| Less pull on the trunk does the work: head B's gain comes from the trunk being shaped by head A | Matches | Near zero | Large |

## The arms

All use N12's model and training, on **N12's seeds 4101–4105**, so that every comparison is paired with N12's saved models of the same seed: the same initial weights, the same batches, the same replaced positions. This reuses models, against the project's preference for new seeds (`CRITICAL_REVIEW.md`, lesson 6), because N13a is an identity test that has no meaning across seeds. The reuse is declared, and no verdict of N12 is re-decided.

| Arm | Head B | Trained for N13? | Role |
|---|---|---|---|
| `n80` | Single softmax, plain cross-entropy on the noisy targets | No: N12's saved model | The standard head; N13b re-measures it in mixture form |
| `n80-mix` | Mixture output, −log q(y) | No: N12's saved model | The reference for N13a |
| `n80-rw` | Single softmax, loss stopgrad(r(y)) · CE(l, y) | Yes | Reweighting alone: by the identity, the mixture head's inner softmax |
| `n80-mix-rep` | Mixture output, −log q(y), trained again from the same seed | Yes | The replicate: how far two runs with identical mathematics differ in this pipeline |
| `n80-w02` | Single softmax, loss 0.2 · CE(l, y) | Yes | The scale-only control: the prior clean-ness as a constant discount |
| `sub20` | N12's saved model | No | Secondary comparisons only |

**Details, fixed now:**
- **`n80-rw`:** at each position, r(y) = 0.2 p(y) / (0.2 p(y) + 0.8 u(y)) is computed from the head's current logits in float32 without gradient, and multiplies that position's cross-entropy; the loss is the mean over positions, as in every arm. Nothing else differs from `n80`.
- **`n80-w02`:** head B's cross-entropy multiplied by the constant 0.2. Nothing else differs from `n80`.
- **`n80-mix-rep`:** N12's `n80-mix` arm run again, with no change of code, seed, data order or noise draws. Any difference from N12's model is the pipeline's own nondeterminism.
- **Every run records,** every 50 steps, the mean of r(y) over the batch, and separately over the positions whose targets were kept and over those whose targets were replaced. For `n80-rw` this is the discount in use; for the other arms it is the same quantity computed from their logits and recorded only. The mask of replaced positions is known in training, so this is exact.
- **The measurement windows** are N12's: the held-out tail's windows 1–512, which no model involved has trained on. N13 must use them, because its comparisons are with N12's models measured there.

## The measures

- **The inner reading against head A.** For each model, the clean-target loss of head B's single softmax (for the mixture head, its inner softmax) minus head A's clean-target loss in the same model, on the measurement windows: Δ_A. In N12, Δ_A was +1.400 for `n80`, +0.338 for `n80-mix` and +0.163 for `sub20` (means over the five seeds).
- **The difference between two models' inner readings,** paired by seed: Δ_A(x) − Δ_A(y). Also, as a secondary measure, the mean per-position symmetrised KL divergence between the two models' inner softmaxes on the measurement windows, which compares the predictions themselves.
- **S in mixture form:** N12's share of the gap closed, computed for the prediction 0.2 softmax + 0.8 u formed from a model's logits, against the same seed's `n80` yardstick. For `n80` itself this is the output form applied at readout to the standard head.
- **The geometric measures** of N12, computed within each model: the inner softmax's agreement with head A, and the rotated and stability controls.

## The claims

**N13a (primary): the pipeline follows the identity.** `n80-rw`'s inner reading matches `n80-mix`'s: |Δ_A(`n80-rw`) − Δ_A(`n80-mix`)| ≤ 0.03 nats in at least 4 of 5 seeds. The tolerance is the size of the hardware effect found in N11's comparison (losses within 0.024 nats between GPUs with identical mathematics).

**N13b: the output form, applied at readout to the standard head, closes none of the gap.** The 97.5% interval for S(`n80`, mixture readout) lies wholly below 0.10.

**N13c: the gain is not from head B pulling less on the shared network.** `n80-w02` improves the inner reading by less than a quarter of the mixture head's improvement: the 97.5% interval for D = Δ_A(`n80-w02`) − Δ_A(`n80`) lies wholly above −0.27 nats, a quarter of N12's recorded improvement of 1.062 nats (1.400 − 0.338) on these same models.

## Refuting results

- **N13a** is refuted if |Δ_A(`n80-rw`) − Δ_A(`n80-mix`)| > 0.10 nats in at least 4 of 5 seeds while the replicate stays within 0.03 (precondition 3): the two trainings differ by far more than the pipeline's own noise.
- **N13b** is refuted if the interval for S(`n80`, mixture readout) lies wholly above 0.33, with a standard deviation of at most 0.05.
- **N13c** is refuted if the interval for D lies wholly below −0.53 nats, half of the mixture head's improvement, with a standard deviation of at most 0.10.

## Design

- **Runs:** 15 training runs (`n80-rw`, `n80-mix-rep`, `n80-w02` on seeds 4101–4105), and measurements of those 15 models and of N12's 15 saved models (`n80`, `n80-mix`, `sub20`), the latter in both readout forms.
- **N12's models** are loaded from the `n12_bottleneck` Drive folder. Their N12 measurement records, committed in `results/n12/`, are the reference for precondition 2.
- **Measured per model:** head A's and head B's inner clean-target losses; the mixture-form prediction's loss on clean and noisy targets and S; the inner softmax's agreement with head A; the rotated control; stability. For each new model, the symmetrised KL to its N12 partner's inner softmax (`n80-rw` and `n80-mix-rep` to `n80-mix`; `n80-w02` to `n80`), per position, on the measurement windows.

## Decision rules

**Preconditions:**
1. **Setup.** The batch, same-model, identical-heads and analytic checks pass; the windows are shifted correctly; no training draw reaches the held-out tail; each new run's batch fingerprint and replaced-positions fingerprint at step 0 equal those recorded for N12's `n80` of the same seed.
2. **Must match.**
   - Re-measuring N12's saved `n80` and `n80-mix` models on the same windows reproduces N12's records: every loss within 0.001 nats, every agreement ratio within 0.01.
   - The constructed mixture check passes in every model.
   - The zero-noise copy matches head A at a ratio of at least 0.999 in every model.
3. **The replicate floor is below the tolerance.** |Δ_A(`n80-mix-rep`) − Δ_A(`n80-mix`)| ≤ 0.03 nats in at least 4 of 5 seeds. If replicates of the same training differ by more, N13a's tolerance cannot tell an identity from a difference, and N13a is untestable.
4. **Must differ.** The rotated control's ratio is at most 0.60 in at least 4 of 5 models of each arm.
5. **The yardstick leaves a gap.** gap(`n80`) ≥ 0.30 nats in every seed (N12 recorded 0.477–0.503).

Preconditions 1, 2 and 4 apply to every claim; 3 to N13a; 5 to N13b.

**N13a:**

| Outcome | Condition |
|---|---|
| Supported | \|Δ_A(rw) − Δ_A(mix)\| ≤ 0.03 in at least 4 of 5 seeds |
| Refuted | > 0.10 in at least 4 of 5 seeds |
| Intermediate | Anything else |

**N13b and N13c** each have four outcomes:

| Outcome | Condition |
|---|---|
| Supported | The 97.5% interval lies wholly on the claim's side of its margin: below 0.10 for N13b, above −0.27 for N13c |
| Refuted | It lies wholly past the refuting margin (above 0.33; below −0.53), with a standard deviation of at most 0.05 (N13b) or 0.10 (N13c) |
| Some effect, size unclear | It excludes 0 but meets neither rule |
| Inconclusive | Everything else |

**Intervals.** N13b and N13c are interval rules, so each uses a Bonferroni-corrected 97.5% t-interval with 4 degrees of freedom. N13a is a count rule.

**Why these thresholds.**
- **0.03 nats** for the identity is the hardware effect's size, the largest difference the project has measured between runs with identical mathematics. **0.10** is three times that.
- **0.10 and 0.33** of the gap for N13b: a tenth is "none, within the measure's noise" (N12's S values had standard deviations of about 0.015), and a third is a material part.
- **A quarter and a half** of the mixture head's improvement for N13c: a control that captures less than a quarter has not found the mechanism; one that captures half has.

## Secondary and exploratory outcomes

- **S1: the discount's discrimination over training.** The mean of r(y) on kept and on replaced positions, every 50 steps, in `n80-rw`. The conjecture's mechanism predicts the two means apart from the start, by frequency, and further apart as training proceeds.
- **S2: the replicate floor.** The symmetrised KL between `n80-mix-rep` and `n80-mix`, and their difference in every measured quantity, for later tests to use as the pipeline's own noise.
- **S3: S in mixture form for `n80-rw` and `n80-w02`,** against N12's 0.713 for `n80-mix`.
- **S4: the cost to head A** in each new arm, against `n80` and `clean` of the same seed (N12: `n80` +0.040, `n80-mix` −0.009). The rival predicts `n80-w02` lowers the cost even if N13c is supported; that would separate the cost to the trunk from head B's reading.
- **S5: the inner softmax's agreement with head A** in each new arm, against N12's values (0.31 for `n80`, 0.81 for `n80-mix`).

## Necessity audit, at drafting (`NECESSITY_AUDIT.md`)

| Claim the design relies on | Features and ground | The check, and whether it works at the boundary |
|---|---|---|
| The identity test presupposes that a replicate of the same training differs by less than the tolerance | Functional; a present mechanism (the pipeline's nondeterminism) | Precondition 3. If the floor is above 0.03, N13a is untestable rather than refuted. At the other boundary, a replicate that is bit-identical makes the floor 0, and the absolute tolerance still applies |
| The discounted cross-entropy has the mixture loss's gradient | Definitional (the chain rule) | Checked numerically before N12; N13a tests the pipeline, not the mathematics. A failure means a bug or a mixed-precision or clipping interaction, not a refutation of the chain rule |
| A constant discount isolates head B's pull on the trunk | Definitional for head B's own parameters (AdamW's normalisation, up to its ε), a present mechanism for the trunk | The control's reading is stated in "The conjecture": it changes head B's share of the trunk's gradient and nothing in head B's own update. If AdamW's ε matters at this gradient scale, `n80-w02` would differ from `n80` in head B's readout even without a trunk effect; S4 and S5 would show it |
| S in mixture form is a noisy-target prediction's measure | Definitional | For `n80`'s mixture readout, 0.2 softmax + 0.8 u is a distribution over the same vocabulary, measured as the copy is. The clean arm is excluded, as in N12 |
| Re-measuring N12's models reproduces N12 | Present mechanism: deterministic forward passes on the same GPU type | Precondition 2, with tolerances. The job runs only on an A100 |
| Comparing inner softmaxes across models | Definitional: predictions at the same positions are comparable, states are not | Only prediction-space comparisons cross models; every geometric measure stays within a model |
| N13c's margins presuppose N12's improvement of 1.062 nats | Definitional: the same models and windows | Exact by construction; the margins are stated as fractions and as nats |
| The discount's effect does not change over training in a way that breaks the identity | Reply 5 (time) | The identity holds at every step, so it does not. S1 records how the discount itself changes, as description |

The critic's strongest objection, two moves deep: *"N13a is a test of arithmetic, not of the world."* The reply: it is a test of whether the explanation N12 gave is the one the pipeline implements, which the chain rule asserts and nothing has yet checked at full length; its failure mode is real (a bug, or mixed precision) and would change what N12 means. *The critic's answer:* then the knowledge is in N13c, and N13a is a precondition in disguise. *The next move:* that is accepted; N13a is primary because N13c's reading depends on it, and because its tolerance, once calibrated, is a measured fact about the pipeline that every later paired test will need.

## Calibration before queuing, including a full-length run

- **A full-length calibration run on the A100,** as its own Colab job, seed 3001: `n80-rw`, `n80-mix-rep` and `n80-w02`, with N12's calibration models (`n80`, `n80-mix`, `sub20` of seed 3001, from the `n12_calib` Drive folder) re-measured, on the calibration windows 513–1024. It reports:
  - the replicate floor for one seed, which decides whether 0.03 is a usable tolerance;
  - the `n80-rw` difference from `n80-mix`;
  - the discount's means on kept and replaced positions over training;
  - `n80-w02`'s values.

  The values for seed 3001 will be seen, and recorded in the deviation log.
- **What the calibration may change:** only the tolerance of N13a and precondition 3, if the replicate floor exceeds 0.03, through an amendment that comes back for approval first. No claim, margin, arm, window or measure changes in response to it.
- **Also:** the untrained checks as in N12; the tiny end-to-end run and the resume test; synthetic records for every branch of the rules.

## Sanity checks before trusting results

- Every data file's fingerprint matches the manifest; the held-out boundary is logged.
- Each new run's fingerprints match N12's; the discount's means are logged.
- Every record names the GPU, the PyTorch version and the code commit. The job runs only on an A100, the GPU N12 ran on.
- Every number in the report is computed by code from saved JSON records. `n13_analyze.py` is written and committed before any real run, and tested on synthetic records covering every branch of the rules.

## Compute, where it runs, and stopping rule

- **Where:** one Colab job, `n13_signal`, on an A100, resumable, after the calibration job `n13_calib`.
- **Cost:** 15 training runs at about 2 minutes each, 30 measurements, and the 15 cross-model comparisons: about 45 minutes. The calibration job: about 15 minutes.
- **Stopping rule:** exactly these runs and measurements, each once. Nothing is added after seeing results except as a dated, exploratory entry in the deviation log.

## What each outcome would teach

| N13a | N13b | N13c | What we would learn | Next |
|---|---|---|---|---|
| Supported | Supported | Supported | The mixture head's advantage is a learning signal that discounts each target by its estimated clean-ness in context. The form matters only at readout, and head B's reduced pull on the trunk is not the mechanism. | The dose of random targets at a fixed number of informative ones, with the discounting head, to test whether the random targets leave the geometric trace N12b and N10c suggest |
| Refuted | Any | Any | The pipeline does not implement the chain rule's identity: a bug or a numerical interaction. N12's explanation of its own result is in doubt until it is found. | Find it: compare the two arms' gradients step by step |
| Supported | Any | Refuted | Head B's reduced pull on the trunk accounts for half or more of the gain. The damage noise does is to the shared network, through head B's gradient, more than to head B's reading. | Measure the trunk directly: head A's cost and the state's geometry under each arm, and N4's cost of sharing at several noise rates |
| Supported | Refuted | Any | The standard head's softmax is closer to a clean estimate than N12 assumed, and the mixture form at readout is what brings it to the copy. N12d's reading needs revisiting. | Compare the standard head's softmax with head A directly, at several noise rates |
| Untestable (precondition 3) | Any | Any | Runs with identical mathematics differ by more than 0.03 nats on this pipeline; the tolerance was wrong, and every paired comparison in the project has that much noise in it. | Amend the tolerance, and record the floor for all later tests |

## Limitations known now

- **Reused seeds and models,** by necessity; declared above.
- **One noise rate, one readout width, one scale, 10% of the budget.**
- **The scale control uses one constant,** 0.2, the prior clean-ness. A constant chosen to match the discount's time-averaged mean would be a second version; S1 will show how far the two differ.
- **The replicate floor is measured on one arm,** `n80-mix`, and taken to hold for the others.
- **N13a's tolerance is a project measurement, not a known property of the hardware.**

## Deviation log

(none yet)

## References

Patrini, G., Rozza, A., Krishna Menon, A., Nock, R., & Qu, L. (2017). Making deep neural networks robust to label noise: A loss correction approach. In *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 2233–2241). IEEE. https://doi.org/10.1109/cvpr.2017.240

Yang, Z., Dai, Z., Salakhutdinov, R., & Cohen, W. W. (2017). *Breaking the softmax bottleneck: A high-rank RNN language model* [Preprint]. arXiv. https://doi.org/10.48550/arxiv.1711.03953
