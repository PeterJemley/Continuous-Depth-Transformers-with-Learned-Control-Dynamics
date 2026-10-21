# Preregistration N12: is the noisy head held back by its readout?

Date: 2026-10-20    Status: **approved by the author, 2026-10-20.** **Amendment A1 (precondition 5's convergence test) approved by the author, 2026-10-21.** Built, tested and calibrated; queued as `n12_bottleneck`. No real N12 run has started.

Type: confirmatory for N12a (primary), N12b, N12c and N12d, each with its own decision rule, on new models trained for this test. Everything else is secondary or exploratory and labelled so.

Parent documents: `N11_NOTE.md` (its two explanations for why the payload saturated, and next conjectures 1 and 2), `N9_NOTE.md`, `N10_NOTE.md`, and `CRITICAL_REVIEW.md` lesson 7.

**Relation to earlier results.**
- **The pattern so far.** Since N7, a head trained on targets with 80% replaced by random words has read different directions from the clean head.
  - N9 showed that it is not the *ideal noisy copy* of the clean head either (agreement 0.31–0.35), although the measure recognises that copy (0.95–1.00). N9b was refuted.
  - N10 found the drift grows with the noise. N11 found that a payload pushing the head towards the copy closed under half of the functional gap, and saturated as its strength rose.
- **What has not been explained.** *Why* a head trained on noisy targets fails to become the ideal copy. Two explanations have been offered so far, after the fact. One is that training on noise is simply worse. The other, N11's first, is the softmax bottleneck. N12 tests the second as a conjecture that could fail.
- **What stands regardless.** Every earlier verdict stands. N9b stays refuted: the trained noisy head is not the ideal copy. N12 asks why.

## The conjecture

**The noisy head drifts because its readout cannot represent what it is trained to predict.**
- **What it is trained to predict.** With 80% of targets replaced by draws from the word frequencies u, the best possible prediction of the noisy target is the mixture 0.2 · p + 0.8 · u, where p is the true next-word distribution.
  - That mixture is exactly the ideal noisy copy built from a perfect clean head.
  - Head A, trained on clean targets, estimates p. So the ideal copy of head A is the best available estimate of what head B should predict.
- **Why a standard head cannot represent it.** Head B is a single softmax on a linear, rank-64 readout of the state. A mixture of two softmax distributions is in general not a single softmax of any linear readout: this is the *softmax bottleneck* (Yang et al., 2017). So head B is trained towards a target outside its reach, and settles on the closest thing it can represent, which reads the state differently.
- **The fix it implies.** Give head B an output that *can* represent the mixture: q_B = 0.2 · softmax(head B's logits) + 0.8 · u, trained on the noisy targets alone. In the label-noise literature this is *forward loss correction* (Patrini et al., 2017). The best q_B is then the ideal copy, and the inner softmax's best value is p, the same thing head A estimates.
- **What it explains, if true.**
  - N9b: the noisy head is not the ideal copy because it cannot be.
  - N10's dose order: the closer the mixture is to a single softmax (less noise), the less the head must distort its reading.
  - N11's saturation: a payload cannot push a single-softmax readout past the bottleneck, however strong.
  - It is hard to vary. It names the head's form as the cause, and it predicts that changing only the form, with the same noisy targets and no payload, removes most of the drift.

**What I suggested before, and why this replaces it.** `N11_NOTE.md` proposed two next tests.
- *The frozen-readout test* is kept, as N12d. It measures the bottleneck's size directly.
- *Anchoring the clean reading* is dropped. With a single-softmax head B, a payload matching head A's own prediction would make head B a second head A. The share-of-the-gap measure would then reward it for being cleaner than the copy, which says nothing about preserving a reading under noise. The mixture head asks the same question without that flaw: its inner softmax is free to become head A's reading, but only if the noisy targets alone lead it there.

## Competing hypotheses

| Hypothesis | S, mixture head (N12a) | Mixture head's agreement with its copy (N12b) | S, data-matched clean control (N12c) | S, best single-softmax fit (N12d) |
|---|---|---|---|---|
| **The conjecture:** the readout's form is the limit | ≥ 0.67 | ≥ 0.85 | ≥ 0.67 | < 0.67 |
| **Scarcity:** with only 20% of targets informative, 10% of the budget is too little | low | low | low, like the mixture head | any |
| **Noise in the learning signal:** random targets disrupt learning beyond carrying no information | low | low | ≥ 0.67 | any |
| **No bottleneck:** a single softmax can represent the copy closely, and N11's payload was limited by its weighting | any | any | any | ≥ 0.67 |

The data-matched control separates the first three. It sees clean targets at exactly the positions the noisy arms leave unreplaced, and none elsewhere: the same informative targets as the noisy arms, without the random ones.

## The arms

All use N7's to N11's model and training at 10% of the budget, on new seeds, 4101–4105. They differ only in head B's targets and output.

| Arm | Head B's targets | Head B's output | Role |
|---|---|---|---|
| `clean` | The next token | Single softmax | Must-match control |
| `n80` | 80% replaced by draws from u | Single softmax | The yardstick, as in N11, and the standard noisy head |
| `n80-mix` | 80% replaced by draws from u | 0.2 · softmax + 0.8 · u, trained by cross-entropy on the noisy targets | The conjecture's fix |
| `sub20` | The next token, at exactly the positions `n80` leaves unreplaced (about 20%); no target elsewhere | Single softmax | The data-matched clean control |

**Details, fixed now:**
- **`n80-mix`:** the mixture uses the same ρ = 0.8 and the same u that generate the noise. Its log-probabilities are computed as N9's ideal copy is: logaddexp(log 0.2 + log-softmax(logits), log 0.8 + log u).
- **`sub20`:** the positions come from the same draw that decides which targets the noisy arms replace, at the same step. So `sub20` has a clean target exactly where `n80` has one, and nothing where `n80` has a random word. The two arms differ only in those random targets. Head B's loss is summed over the chosen positions and divided by the number of *all* positions, so its informative signal per step matches the noisy arms'.
- **The frozen fits,** on each `n80` model after training:
  - Everything is frozen except head B's readout (P_B and W_B), which starts at its trained value.
  - The readout is then trained to match the model's own ideal copy, KL(copy ‖ head B's output), on fresh training windows. There are two versions:
    - `fit-single`, where head B's output is its usual single softmax;
    - `fit-mix`, where it is the mixture above.
  - Both use AdamW at the training's peak learning rate (2.4 × 10⁻³) with its warm-up and cosine shape, for 3,000 steps of 32 windows.
  - `fit-mix` can represent the copy exactly, so it is the fitting procedure's must-match control. `fit-single` measures how close any single-softmax readout of this state can get.

## The windows: a held-out part of the training text

The test split is used up (N11 used its last windows), and every test has used the validation split for its final losses. N12 therefore holds out the **last 1% of the training text**:
- **The boundary:** token 116,740,938 of 117,920,140.
- **What never sees the held-out tail:** no N12 model trains on it, and no fit samples from it. Training batches and fit windows are drawn only from before the boundary.
- **The word frequencies:** u is computed from the text before the boundary, so nothing about the held-out text enters the noise or the copies.
- **What comes from the tail:** it holds 9,212 non-overlapping windows. N12 measures on its windows 1–512. The calibration below uses windows 513–1024, so N12's own windows stay unseen until the real run.

## The measures

**Functional: S, the share of the gap closed, as in N11.**
- Each arm's head B gives a prediction of the *noisy* target:
  - `n80`: its output;
  - `n80-mix`: its mixture output;
  - `sub20`: the same mixture formed from its softmax, 0.2 · softmax + 0.8 · u;
  - the fits: their outputs.
- gap(arm) = that prediction's loss on the clean targets, minus the loss of the yardstick: the ideal copy of the same seed's `n80` head A.
- S(arm) = 1 − gap(arm) / gap(`n80`), per seed.
- For the fits, head A and the trunk are frozen, so their yardstick is the very model they are fitted to.

**Geometric: N9's agreement ratio** of head B's noisy-target prediction with its own model's ideal copy, by the target-free Fisher measure. For `n80-mix` and `sub20` this compares like with like: a mixture-form prediction against a mixture-form copy.

## The claims

**N12a (primary): a head that can represent the mixture keeps the reading, from the noisy targets alone.** The 97.5% interval for S(`n80-mix`) lies wholly above 0.67.

**N12b: and reads the same directions as its ideal copy.** `n80-mix`'s head B has an agreement ratio of at least 0.85 with its own model's ideal copy, in at least 4 of 5 seeds.

**N12c: scarcity alone does not cause the drift.** The 97.5% interval for S(`sub20`) lies wholly above 0.67.

**N12d: the bottleneck is large.** Even fitted directly, to convergence, the best single-softmax readout of the copy closes at most two thirds of the gap: the 97.5% interval for S(`fit-single`) lies wholly below 0.67.

## Refuting results

- **N12a** is refuted if the 97.5% interval for S(`n80-mix`) lies wholly below 0.67, with a standard deviation of at most 0.05.
- **N12b** is refuted if the ratio is at most 0.60 in at least 4 of 5 seeds.
- **N12c** is refuted if the 97.5% interval for S(`sub20`) lies wholly below 0.67, with a standard deviation of at most 0.05.
- **N12d** is refuted if the 97.5% interval for S(`fit-single`) lies wholly above 0.67, with a standard deviation of at most 0.05.

The conjecture as a whole survives only if all four are supported. If N12a and N12b are supported and N12d is refuted, the mixture head works but not because of a large bottleneck, and the explanation must change.

## Design

- **Seeds:** 4101–4105, new. All four arms of a seed share the initial weights and the order of the training data.
- **Runs:** 20 training runs, and 10 frozen fits (two on each `n80` model).
- **Measurement windows:** the held-out tail's windows 1–512, with 8 extra tokens each. No measurement has used them.

**Measured per model:**
- head B's noisy-target prediction's loss on the clean targets, for S;
- the agreement ratio with the model's ideal copy;
- the rotated-readout control;
- split-half stability;
- head A's loss.

The fits record their loss every 50 steps.

## Decision rules

**Preconditions:**
1. **Setup.**
   - The batch, same-model, identical-heads and analytic checks pass.
   - The windows are shifted correctly.
   - No training batch or fit window reaches the held-out tail. The largest offset drawn is recorded and checked.
   - Each noisy arm's replaced fraction at step 0 is within 0.05 of 0.8.
   - All four arms of a seed see the same batches, and `sub20`'s chosen positions at step 0 are exactly `n80`'s unreplaced ones.
2. **Must match.**
   - In `clean`, head B's ratio with head A is at least 0.95, and |head B's clean loss − head A's| ≤ 0.05, each in at least 4 of 5 models.
   - The zero-noise copy matches head A at a ratio of at least 0.999 in every model.
   - *A constructed mixture check:* in every model, a mixture head given head A's readout reproduces the ideal copy exactly. Its ratio is at least 0.999, and its loss differs from the copy's by at most 10⁻⁴. This tests the mixture's code path in both the training loss and the measurement.
3. **Must differ.** The rotated-readout control's ratio is at most 0.60 in at least 4 of 5 models of each arm.
4. **The yardstick leaves a gap to close.** gap(`n80`) ≥ 0.30 nats in every seed. In N11 it was 0.48–0.50.
5. **The fits converged and can fit.**
   - In every fit, the mean loss over its last 300 steps is within 2% of the mean over the 300 before.
   - In every model, `fit-mix` reaches S ≥ 0.95, so the fitting procedure reaches the copy when the form allows it.
6. **Stable subspaces, for the geometric outcomes.** Head B and the copies have a split-half stability of at least 0.80.

Preconditions 1–4 apply to every claim. 5 applies to N12d, and 6 to N12b.

**N12a, N12c and N12d** each have four outcomes:

| Outcome | Condition |
|---|---|
| Supported | The 97.5% interval lies wholly on the claim's side of 0.67: above for N12a and N12c, below for N12d |
| Refuted | It lies wholly on the other side, with a standard deviation of at most 0.05 |
| Some effect, size unclear | It excludes 0 (N12a, N12c) or 1 (N12d) but meets neither rule |
| Inconclusive | Everything else |

**N12b:**

| Outcome | Condition |
|---|---|
| Supported | Ratio ≥ 0.85 in at least 4 of 5 seeds |
| Refuted | Ratio ≤ 0.60 in at least 4 of 5 seeds |
| Intermediate | Anything else |

**Intervals.** N12a, N12c and N12d are interval rules, each with a Bonferroni-corrected 97.5% t-interval with 4 degrees of freedom. N12b is a count rule.

**Why these thresholds.**
- **0.67** is N11a's bar, two thirds of the gap. For N12d, it is the share N11 required of its payload. If no single-softmax readout can reach it, N11a's target was out of reach for that head.
- **0.85** is the bar N9 and N10 used for "the same reading". Converged clean heads score 0.95–0.98; N10's head at 20% noise scored 0.81.
- **The standard-deviation rule of 0.05:** in N11, S varied across seeds with a standard deviation of about 0.01.

## Secondary and exploratory outcomes

- **S1: the inner reading.** For `n80-mix` and `sub20`, the agreement ratio of head B's *inner* softmax with head A directly, and its loss on the clean targets against head A's. If the conjecture holds, the inner softmax should approach head A.
- **S2: the cost to head A** of each arm, against `n80`.
- **S3: the noisy targets.** Each head's loss on the noisy targets. The conjecture predicts `n80-mix` below `n80`, because only the mixture can represent the best noisy prediction.
- **S4: the fits' curves,** and `fit-single`'s divergence against N11's saturated payload (0.038).
- **S5: replication of N11's `n80`:** the agreement ratio and gap(`n80`), on different windows.

## Calibration before queuing, including a full-length dose check

Lesson 7 of `CRITICAL_REVIEW.md`, as sharpened after N11: a check must cover the whole training schedule.
- **Untrained models,** on the calibration windows (513–1024 of the tail):
  - the analytic check, identical heads, the rotated control and the constructed mixture check;
  - recognising copies, and stability.
- **A full-length calibration run on the A100,** before queuing, as its own Colab job. One seed, G1's 3001, all four arms and both fits, on the full schedule, measured on the calibration windows. It checks:
  - that every arm trains without failure, and that `n80-mix`'s loss falls;
  - that `sub20`'s chosen positions match `n80`'s unreplaced ones;
  - that both fits meet precondition 5, and how many steps `fit-single` needs to converge.

  Its S values for seed 3001 will be seen, and recorded in the deviation log.
- **What the calibration may change:** only the fits' number of steps, if precondition 5 fails, through an amendment that comes back for approval first. No claim, threshold or arm changes in response to it.
- **Also:**
  - the tiny end-to-end run, and the resume test;
  - synthetic records for every branch of the rules.

## Sanity checks before trusting results

- Every data file's fingerprint matches the manifest, and the boundary of the held-out tail is logged with its token count.
- Each arm's noise fraction, `sub20`'s chosen positions, and the yardstick for each seed are logged.
- Every record names the GPU, the PyTorch version and the code commit. The job runs only on an A100, the GPU N11 ran on.
- Every number in the report is computed by code from saved JSON records. `n12_analyze.py` is written and committed before any real run, and tested on synthetic records covering every branch of the rules.

## Compute, where it runs, and stopping rule

- **Where:** one Colab job, `n12_bottleneck`, on an A100, resumable, after the calibration job.
- **Cost:** 20 training runs at about 2 minutes each on the A100, plus 10 fits and the measurements: about an hour. The calibration job: about 15 minutes.
- **Stopping rule:** exactly these runs, fits and measurements, each once. Nothing is added after seeing results except as a dated, exploratory entry in the deviation log.

## What each outcome would teach

| N12a | N12b | N12c | N12d | What we would learn | Next |
|---|---|---|---|---|---|
| Supported | Supported | Supported | Supported | The noisy head's drift since N7 was the readout's form, not noisy training. Given a form that can represent what it must predict, a head trained on 80% noise keeps the clean reading. | Return to N10's question with mixture heads: does anchoring function now beat pinning geometry? |
| Supported | Any | Supported | Refuted | The mixture head works, but a single softmax could also have come close, so the bottleneck is not why. N11's saturation then points to the payload's weighting. | Re-examine A1's post-clip weighting |
| Refuted | Any | Refuted | Any | Scarcity: with 20% informative targets at this budget, no form keeps the reading | The same arms at the full budget |
| Refuted | Any | Supported | Any | The random targets themselves disrupt learning, beyond their lack of information | Compare per-position gradient noise across arms |
| Supported | Refuted | Supported | Any | Function is kept, but the directions differ: function and geometry come apart again, now without any payload | Ask which of many equivalent geometries training picks |

## Limitations known now

- **The mixture head is told the noise:** its ρ and u are the ones that generated it. That is right for testing the explanation, but it is not a method for unknown noise.
- **The held-out tail is training text, not a test split.** It comes from the same source, and no N12 model or fit has seen it, but it was never designed as a test set.
- **One noise rate (80%) and one readout width.** The bottleneck should matter less at lower noise, by the conjecture's own logic.
- **The same limits as before:** 10% of the budget, one learning rate, one size, one dataset.

## Deviation log

### Calibration before any real run, 2026-10-21 (no amendment)

Results seen before this entry:
- calibration on **untrained** full-size models (G1's seeds 3001 and 3002), on the calibration windows (513–1024 of the tail);
- test-mode runs on tiny models;
- synthetic records.

The full-length calibration run on the A100 has not started, and no real N12 run has started.

- **The held-out tail:**
  - its boundary is at token 116,740,938 of 117,920,140, and it holds 9,212 windows;
  - u computed from the training region gives 3,390 words that never occur (3,369 for the whole training split).
- **Untrained models,** on the calibration windows:
  - *The estimator:* the analytic check scored 0.994, and identical heads 1.0.
  - *The constructed mixture check* passed exactly in both models: ratio 1.0, loss difference 0.0.
  - *Recognising copies:* head A against its ideal copy scored 0.988 and 0.995, and the zero-noise copy 1.0.
  - *Must differ:* the rotated control scored 0.077 and 0.076.
  - *Stability:* 0.88 to 0.98 for every head and copy.
- **Tiny test-mode runs.**
  - All 8 models trained and were measured, and all 4 fits ran. A rerun skipped all finished work.
  - The calibration mode was interrupted after 26 steps of its first run. With its local folder deleted, as on a new Colab machine, it resumed from the Drive copy and finished.
  - Every training batch and fit window stayed before the boundary.
  - After each fit, the copy's loss was exactly unchanged (difference 0), so the fits leave the trunk and head A untouched.
  - The tiny models are too small to pass the must-match control, and their 60-step fits do not converge. Both were expected.
- **The Colab wrappers** run only on an A100. This was tested with simulated GPU names: a T4 is refused, and the calibration wrapper adds `--calibration`.
- **Synthetic records.** 23 cases covering every branch of the four rules and every precondition gave the intended verdicts.
- **Implementation details, not changes:**
  - The fits use the training's optimiser settings (AdamW, PyTorch's default weight decay, gradients clipped at 1.0). They run in evaluation mode, without dropout, as the measurements do.
  - The replaced positions of `n80`, `n80-mix` and `sub20` all come from N10's per-step draw, seeded by the seed and the step.

### The calibration run and amendment A1, before any real run, 2026-10-21: approved by the author

**The calibration run.**
- *The session:* `colab-results` branch `n12_calib/20261021-095316`, code commit 6f497b3, on an NVIDIA A100-SXM4-40GB with PyTorch 2.11. It ran in 12 minutes, with no resumes and no Drive failures.
- *What it was:* seed 3001, all four arms and both fits, on the full schedule, measured on the tail's windows 513–1024.
- *The report:* `N12_CALIBRATION.md`, generated by `n12_analyze.py` from `results/n12_calib/`.

**What it showed, recorded as the preregistration said it would be:**
- **Preconditions 1–4 and 6 held:**
  - setup;
  - must match: the clean arm 0.971, and the constructed mixture check exact;
  - must differ: the rotated control 0.12–0.13;
  - gap(`n80`) 0.469;
  - stability 0.93–0.99.
- **`fit-single` converged:** a 0.8% change over its last 300 steps, at a fit divergence of 0.034. So 3,000 steps are enough, and the number of steps does not change.
- **`fit-mix` reached the copy** (S = 0.997, agreement 0.998, divergence 0.0001 nats), **but failed the convergence test as written.**
  - Its loss went from 0.000107 to 0.000099 nats between the two 300-step windows. That is a change of 0.000008 nats, but 7% relative.
  - A relative test cannot be met by a loss that is falling towards zero.
  - More steps would not help, and the preregistration allowed the calibration to change only the number of steps.
- **The values for seed 3001, seen and recorded.** They decide nothing.

  | Measure | Value |
  |---|---|
  | S(`n80-mix`) | 0.687 |
  | S(`sub20`) | 0.783 |
  | S(`fit-single`) | 0.614 |
  | S(`fit-mix`) | 0.997 |
  | `n80-mix`'s agreement with its copy | 0.811 |
  | `n80`'s agreement with its copy | 0.325 |

**The amendment.**
- *What it changes.* In precondition 5, a fit has converged if the mean loss over its last 300 steps is within 2% of the mean over the 300 before, **or** that mean is already below 10⁻³ nats, so the fit has reached its target.
  - For `fit-single` the test is unchanged in effect: its loss cannot fall below 10⁻³ unless it reaches the copy, and then it has converged in any sense.
  - Code: `AT_TARGET` in `n12_bottleneck.py` and `n12_analyze.py`. The analysis recomputes the test from each fit's recorded means.
- *Why it is outside what the preregistration foresaw.* The preregistration said the calibration might change only the fits' number of steps. This changes the form of a precondition instead, because no number of steps can fix a relative test on a loss near zero. It is therefore put to the author as an amendment, not made as a calibration adjustment.
- *What had been seen.* The calibration values above, for one seed, on other windows. The amendment touches only the fitting procedure's must-match control. It changes no claim, threshold, arm, window or measure, and no number in the table above.
- *Tests:*
  - the 23 synthetic cases still give the intended verdicts;
  - a new case, `fit-mix` at its target with a 7% relative change, now passes;
  - fits that are unconverged, or that are below S = 0.95, still fail.

### Necessity audit before the real run, 2026-10-21 (no change to any claim, rule, arm, window or measure)

This is the first use of `NECESSITY_AUDIT.md`. Results seen before this entry: the calibration values for seed 3001, recorded above. The audit was done after reading them, and findings 2–4 were prompted partly by them. No real N12 run has started, and the author may object to any of this before it does.

**1. Necessity claims the design relies on, and the checks that cover them.** All are sound:
- the held-out tail is unseen (the largest offset drawn, precondition 1);
- u comes only from the training region (recorded in the summary);
- `sub20` differs from `n80` only in the random targets (the replaced-position fingerprint, precondition 1);
- the yardstick is fixed, and a fit leaves its own yardstick unchanged (the copy's loss was unchanged after the calibration fits);
- the mixture's code path is exact (the constructed check, precondition 2);
- the fitting procedure can reach the copy when the form allows (`fit-mix`, precondition 5);
- the convergence test works at zero loss (amendment A1).

**2. N12d is asymmetric, by reply 4 (hold an argument to its modest conclusion).** N12d speaks of "the best single-softmax readout", but a fit finds one readout, not the best.
- S(`fit-single`) is therefore an existence proof: some single-softmax readout of this state reaches it.
- A refutation (the interval wholly above 0.67) is conclusive, because a readout that good exists.
- Support (the interval wholly below 0.67) shows only that this fitting procedure could not do better, the same procedure that reaches the copy when the form allows. Showing that no single softmax can would need a proof, or more fitting procedures than one.
- A supported N12d is reported in that modest form.

**3. One rival hypothesis was understated, by reply 3 (another mechanism).**
- The mixture head's loss changes the learning signal as well as the output form. Its gradient for a target y is the single softmax's gradient scaled by r(y) = 0.2 p(y) / q(y), the posterior probability that y is a clean target. This was checked through the job's own code, to within 10⁻⁹.
- So forward correction mostly ignores the targets the head already judges to be noise, as `sub20` ignores them entirely.
- The rival "noise in the learning signal" therefore predicts S(`n80-mix`) high, not low as the competing-hypotheses table says. Its row should read: S(`n80-mix`) high, S(`sub20`) at least 0.67, S(`fit-single`) any.
- So N12a does not separate this rival from the conjecture. N12d does: the conjecture needs a single softmax to be capped, and the rival does not.

**4. The conjecture's wording claims more than the design can show, by replies 3 and 4.**
- S(`fit-single`) is the share of `n80`'s gap that some single-softmax readout of the same state can close.
- So the readout's form accounts for at most 1 − S(`fit-single`) of the gap. At least S(`fit-single`) of it comes from how the head was trained.
- In the calibration seed that split was at most 0.39 for the form against at least 0.61 for training.
- If the real seeds are similar, "all four supported" would not mean, as the outcome table's first row says, that the drift since N7 "was the readout's form, not noisy training." It would mean the form accounts for less than half of the gap, and training on noisy targets for the rest.
- That row is read in this light.

**5. An added secondary measure, S6, the decomposition.** It is computed from the preregistered records and decides nothing. It was defined after seeing seed 3001's values (S(`fit-single`) 0.614, S(`n80-mix`) 0.687), and it says so in the report.
- *The form's share at most:* 1 − S(`fit-single`).
- *Training's share at least:* S(`fit-single`).
- *What the mixture head achieved that the single-softmax fit did not:* S(`n80-mix`) − S(`fit-single`).


## References

Patrini, G., Rozza, A., Krishna Menon, A., Nock, R., & Qu, L. (2017). Making deep neural networks robust to label noise: A loss correction approach. In *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 2233–2241). IEEE. https://doi.org/10.1109/cvpr.2017.240

Yang, Z., Dai, Z., Salakhutdinov, R., & Cohen, W. W. (2017). *Breaking the softmax bottleneck: A high-rank RNN language model* [Preprint]. arXiv. https://doi.org/10.48550/arxiv.1711.03953
