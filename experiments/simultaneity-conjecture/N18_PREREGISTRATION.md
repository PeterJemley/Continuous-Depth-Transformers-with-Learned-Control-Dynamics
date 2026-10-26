# Preregistration N18: the loop, or what the judge is trained on

Date: 2026-10-26    Status: **draft for the author's approval.** Not built, not run. Nothing in this file is a finding.

Type: confirmatory for N18a, the single claim, decided on two accuracy axes that must agree. It uses one new judge trained for this test, compared with N17's saved judges and with loop-free judges trained as N17's were. A second new judge, `rw-nfree`, is exploratory, with no claim. Everything else is secondary or exploratory and labelled so.

Parent documents: `N17_NOTE.md` (next conjecture 1; "Limitations"), `N17_PREREGISTRATION.md` (the third head, its checks and the loop-free line), `N14_NOTE.md` (`sub20-rw`, the self-judged weighting of clean targets alone), `N13_NOTE.md` (the discount as the mixture head's learning signal), `N12_NOTE.md` (the standard head on noisy targets).

**Relation to earlier results.**
- **The shortfall.** Head B, trained on 80%-scrambled next-word targets and weighted by its own discount (N13's `n80-rw`, called `rw-b1` here), ends 0.337 nats from head A on clean targets. The oracle mask `sub20` ends at 0.163. The shortfall between them, G, is 0.165–0.179 by seed.
- **What N17 found.** In each new arm, a third head, head C, read the trunk without training it and judged head B's targets.
  - Interpolated to head B's own accuracy, a loop-free judge recovered a third of the shortfall (N17a: 0.334, 97.5% interval [0.318, 0.349], some effect, size unclear). The two loop-free judges trained at full weight on clean targets at a nested 10% and 5% of positions.
  - A peer trained as head B is, judging itself, was as accurate as head B and recovered 0.041 (secondary).
  - It fell short of a loop-free judge of its accuracy by d = −0.290, 95% interval [−0.315, −0.265] (description; N17b was untestable).
- **What N17 left open.** The loop-free judges and the peer differ in two ways at once.
  - *The loop:* the peer weights its own targets by its own prediction; the loop-free judges train at full weight.
  - *The training data:* the loop-free judges train on clean targets only, chosen with the true mask; the peer trains on head B's targets, 80% of them random.

  The course of training and sharpness during it also differ.
- **What "as accurate as" means is not one thing.** A self-judging head is trained by exactly the gradient of the mixture-form likelihood, −log(0.2 p + 0.8 u) (N13's identity). Judged on that likelihood, it can look more accurate than heads that are far more accurate on plain cross-entropy.
  - N14's `sub20-rw`, the self-judged weighting of clean targets alone, ended 0.72–0.76 nats behind head A on plain cross-entropy, against `sub20`'s 0.16.
  - In mixture form it ended 0.042–0.053 behind the ideal copy, ahead of `sub20`'s 0.112–0.123.
  - So a comparison "at matched accuracy" depends on which accuracy is matched. N18 matches on both, and decides only where they agree.
- **What stands regardless.** Every earlier verdict stands. N18 asks how the gap N17 described divides. It does not re-decide N17.

## The conjecture

**The gap between a self-judging judge and a loop-free judge of the same accuracy is the loop's, not the training data's. A judge with the loop but trained on clean targets only falls short of the loop-free line by most of the peer's gap, on either measure of accuracy.**

- **The mechanism.** A judge that weights its own targets by its own prediction gives little weight to the targets it predicts poorly, and so goes on predicting them poorly, whatever data it is given. This would be analogous to the confirmation bias described in self-training (Arazo et al., 2020). A judge trained at full weight has no such loop.
- **What it forbids.** The self-judging judge on clean data falling short of the line by a third of the peer's gap or less, on both axes.
- **The rivals it is built to exclude.**
  - *The training data (D, N17's R5).* What matters is that the peer trains on noisy targets. A self-judging judge on clean data then sits near the line. It predicts N18a refuted.
  - *Both, in part (B).* The loop and the data each account for part of the gap. It predicts the ratio between its margins.
  - *The axis (A).* "Matched accuracy" means different things on the two axes, and a self-judging judge's standing depends on which is used. It predicts the two axes disagreeing. That outcome is preregistered as "depends on the axis", and N14's records make it plausible.
  - *The course of training (R4).* A self-judging judge starts slowly, because its discount weights every target alike at first, and the gap is that slow start. The slow start is the loop's own doing, so R4 predicts what the conjecture predicts, and N18 does not separate them. S3 reports the course.
  - *Sharpness (R3).* Judges of equal accuracy can differ in how far apart they put clean and random targets. S2 reports it over training.

## The judges

N18 fills in N17's two-by-two of judges by the loop and the training data. `rw-cloop` is the claim's judge; `rw-nfree` is exploratory.

| | No loop: trained at full weight | The loop: trained by its own discount |
|---|---|---|
| **Clean targets only** | The rungs: the loop-free line | **`rw-cloop`** (new, N18a): clean targets at K, the same positions as `rw-j20`, weighted by its own discount |
| **Head B's targets, 80% random** | **`rw-nfree`** (new, exploratory): head B's targets at every position, at full weight, its prediction unmixed | `rw-peer` (N17's, saved) |

`rw-cloop` and `rw-j20` are the same cell but for the loop: the same positions, the same clean targets, weighted or not. The test does not compare them directly. It compares `rw-cloop` with the loop-free line at `rw-cloop`'s own accuracy, as N17 compared the peer.

## Competing hypotheses

| Hypothesis | N18a on N17's axis x: R_loop | N18a on the mixture axis x_m: R_loop | `rw-nfree` (exploratory) |
|---|---|---|---|
| **The conjecture:** the loop | Above two thirds | Above two thirds | Near the line |
| D: the training data | Below a third | Below a third | Below the line |
| B: both, in part | Between | Between | Between |
| A: the axis | One side | The other side | Any |
| R4: the course of training | As the conjecture | As the conjecture | Follows `rw-nfree`'s own course (S3) |

## The arms

Every model is N12's: the trunk with head A and head B, each reading the trunk's final state through its own 64-direction readout, at the 80% noise rate. The training, batches and windows are also N12's. The arms use **N12's seeds 4101–4105**, paired with the saved models of the same seed. Seeds and models are reused, as declared in N13–N17. No earlier verdict is re-decided.

Each new arm adds **head C**, the judge, exactly as in N17. It is a third 64-direction readout of the trunk's state, read detached, with its own initial values and its optimisation tied one way to the main group's. Head B's loss is N13's: the mean over all positions of w(y) · CE, with w from head C.

| Arm | Head C's targets | Head C's loss | Head B's weight w(y) | Trained for N18? |
|---|---|---|---|---|
| `rw-cloop` | Clean targets at K (n80's kept 20%), the positions of `rw-j20` and N12's `sub20` | Each target weighted by head C's own discount, summed over K and divided by all positions: N14's convention for `sub20-rw` | 0.2 p_C / (0.2 p_C + 0.8 u), from head C's prediction | Yes |
| `rw-nfree` (exploratory) | Head B's targets at every position | Plain cross-entropy at full weight, as N12's `n80` head | max(1 − 0.8 u / q_C, 0), the unmixed weight (below) | Yes |
| `rw-j100` (a rung) | The clean target at every position | Plain cross-entropy at full weight | 0.2 p_C / (0.2 p_C + 0.8 u) | Only if the calibration's rule names it |
| `rw-j20`, `rw-j025`, `rw-j0125` (rungs) | As in N17: clean targets at N14's nested positions | As in N17 | As in N17 | Only if the rule names them; N17's real run did not train them |
| `rw-peer`, `rw-j10`, `rw-j05` | As in N17 | As in N17 | As in N17 | No: N17's saved models |
| `rw-b1`, `rw-oracle`, `sub20`, `clean`, `sub10` … `sub0125` | — | — | — | No: N13's, N16's, N12's and N14's saved models |

`rw-j100` is needed because, on the mixture axis, a self-judging judge on clean data may be more accurate than every rung N17 used. `rw-j100` trains on more clean data than K, so it is a rung of the loop-free line, not a cell of the two-by-two.

**The unmixed weight** (`rw-nfree`).
- A head trained at full weight on head B's targets learns their distribution, q ≈ 0.2 p + 0.8 u, with p the clean prediction.
- Under that noise model, the probability that a target y is clean is 1 − 0.8 u(y) / q(y). Where q(y) is below 0.8 u(y), the weight is set to 0.
- This inverts the known noise model, the idea behind one of the loss corrections of Patrini et al. (2017).

N12 found that a single softmax trained on noisy targets matches their distribution poorly: its mixture-form gap to the ideal copy was 0.48–0.50 nats. So `rw-nfree`'s weights are likely to be poor, and many of its clean targets floored at 0. A shortfall from it could not tell the training data from the estimator. That is why it carries no claim.

The rungs and the oracle need clean labels, as in N17; they are diagnostics. `rw-nfree` needs none.

## The measures

- **Δ_A, F, G and a,** as in N17.
- **Two accuracy axes for a judge,** each at the end of training, on the measurement windows' clean targets, relative to head A:
  - **x, N17's:** the judge's plain cross-entropy minus head A's.
  - **x_m, the mixture axis:** the mean of −log m(y) minus the same for head A, where m(y) is the judge's mixture-form probability of y. For a judge with a clean prediction p, m(y) = 0.2 p(y) + 0.8 u(y); for `rw-nfree`, m(y) = max(q(y), 0.8 u(y)). Every judge's weight is w(y) = 1 − 0.8 u(y) / m(y), so x_m summarises the quantity the weights are computed from.

  x_m is not a neutral measure. It is the objective a self-judging judge is trained on, so it favours such judges. Since 1 − w = 0.8 u / m, it is driven by targets with weight near 1 and nearly blind to targets with weight near 0, the neglected targets the conjecture names. A judge that optimises it can score below 0, ahead of head A.

  x is defined for every judge except `rw-nfree`. For `rw-nfree` it is reported as description only, with the clean estimate max(q − 0.8 u, 0.01 u) / 0.2, renormalised.
- **The loop-free line, on each axis:** within seed, the rungs' points (axis value, F), sorted and joined by straight lines. F_lf is its value between the rungs; outside them it gives nothing.
- **d(J) = F(J) − F_lf(judge J's axis value)**, on each axis: how far judge J falls short of a loop-free clean-trained judge of its accuracy.
- **R_loop = d(`rw-cloop`) / d(`rw-peer`),** within seed and on each axis: the share of the peer's gap that the loop alone reproduces.

The margins are set on the ratio, not on d itself: N17's lesson that a margin on a difference should be relative to the quantity it is a difference from.

## The claim

**N18a: the loop alone reproduces most of the gap.** The interval for R_loop lies wholly above 0.67 on both axes.

## Refuting result

**N18a** is refuted if the interval for R_loop lies wholly below 0.33 on both axes, with a standard deviation of at most 0.25 on each. A self-judging judge on clean data then falls short of the line by a third of the peer's gap or less, however accuracy is matched.

## Design

- **Runs:**
  - 10 training runs: `rw-cloop` and `rw-nfree` on seeds 4101–4105.
  - Up to 20 more, for the rungs the calibration's rule names that N17's real run did not train (`rw-j100`, `rw-j20`, `rw-j025`, `rw-j0125`).
  - The measurements of those models, and re-measurements of the saved models, each checked against its committed record:
    - N17's `rw-peer`, `rw-j10` and `rw-j05`, with their head C;
    - N13's `n80-rw`, N16's `rw-oracle`, and N12's `sub20` and `clean`;
    - N14's `sub10` … `sub0125`, for S6.
- **Saved models** are loaded from the `n17_loop`, `n13_signal`, `n16_judge`, `n12_bottleneck` and `n14_dose` Drive folders.
- **Code:** `n18_loop.py`, built on `n17_loop.py`'s head C, forward pass, one-way tie, setup checks and judge measures. It adds `rw-cloop`, `rw-nfree` and `rw-j100`, both axes, and `rw-nfree`'s diagnostics. `n18_analyze.py` applies the rule.

## Decision rule

**Preconditions:**
1. **Setup.**
   - N17's checks: batch, same-model, identical-heads, analytic, the weight identity, the forward identity, the passenger check, the copied-peer check, the overflow test, the nesting, the windows, and no training draw in the held-out tail.
   - Detachment, run for each of the three new specs.
   - **The judge-source check for each new spec, with head C at its own initial values.** The weight in head B's loss must equal, bit for bit, the weight from head C's logits by that spec's formula, and must differ from head B's own discount.
   - **The clean-loop loss form.** `rw-cloop`'s head C loss term, before the factor of 0.5 both losses carry, computed from head B's logits instead of head C's, equals N14's `sub20-rw` loss on the same batch: the largest difference over the largest loss at most 10⁻⁶. N14's discount and N15's weight differ by up to 4 × 10⁻⁷.
   - **The noisy-loss form.** `rw-nfree`'s head C loss term, computed from head B's logits instead, equals N12's `n80` head B loss on the same batch, in the same way.
   - **The unmixing identity.** For a constructed judge whose q is exactly 0.2 p_A + 0.8 u, computed in float64, the unmixed weight equals the discount from p_A: the largest difference over the largest weight at most 10⁻⁶, N15's convention. This is algebra; it checks the code, not the real head.
   - Each new run's fingerprints and noise fraction at step 0 equal N12's `n80`'s, and `rw-cloop`'s positions are K.
2. **Must match.** Re-measuring the saved models reproduces their records: losses within 0.001 and ratios within 0.01. The constructed mixture check passes, and the zero-noise copy scores at least 0.999, in every model.
3. **Must differ.** The rotated control scores at most 0.60 in at least 4 of 5 models of each arm.
4. **Head B trained.** In each new arm, Δ_A of head B is at most 1.0 in at least 4 of 5 seeds. Head C carries no training criterion, since a self-judging judge on clean data may collapse on plain cross-entropy, as N14's `sub20-rw` did. Precondition 6 asks only that the line reach it.
5. **The shortfall exists.** G is at least 0.10 nats in every seed.
6. **The line reaches the judges, on each axis.** In at least 4 of 5 seeds, the rungs are ordered by fraction on that axis, at least the axis's gap apart, and bracket both the peer's and `rw-cloop`'s values. The gap is 0.05 on x and 0.03 on x_m.
7. **The peer's gap exists, on each axis.** d(`rw-peer`) is at most −0.10 on x, and at most −0.06 on x_m, in at least 4 of 5 seeds. That keeps the ratio's denominator well away from 0.

The x_m thresholds are about 0.6 of x's. In N12's and N13's records, head B's x_m was about 0.7 of its x for `sub20`, a head without the loop, and about 0.53 for `n80-rw`, a head with it. N17's records hold x only. On each axis the claim pools the seeds that axis's precondition 6 brackets. Precondition 7 is a condition on testability, not a filter on seeds. A pooled seed with a small d(`rw-peer`) widens the interval, which makes a verdict harder, not easier.

**The outcome, from the two axes:**

| Outcome | Condition |
|---|---|
| Supported | R_loop's interval lies wholly above 0.67 on both axes |
| Refuted | It lies wholly below 0.33 on both axes, with a standard deviation of at most 0.25 on each |
| Depends on the axis | It lies wholly above 0.67 on one axis and wholly below 0.33 on the other |
| Some effect, size unclear | None of the above, and both intervals exclude 0 |
| Inconclusive | Everything else |

If precondition 6 or 7 fails on one axis only, the claim is **untestable**, not decided on the other axis alone. The other axis's interval is reported as description.

**Intervals.** Each axis uses a two-sided 95% t-interval over the pooled seeds, with 4 degrees of freedom, or 3 if one seed is not bracketed. A support or a refutation needs both axes to agree, so the chance of a wrong one is no greater than on one axis alone. That is why no correction for the two axes is applied.

**Why these thresholds.**
- **Two thirds and a third** of the peer's gap. Two thirds is "most of it" and a third is "little of it". Between them, both the loop and something else contribute.
- **A standard deviation cap of 0.25** on refutation. A ratio is noisier than a difference. In N17, d(`rw-peer`) on x varied across seeds by 0.02 around −0.29, so a numerator varying by 0.03 gives a ratio varying by about 0.1.
- **What the design can detect.**
  - If the ratio varies by about 0.1 across seeds, the 95% half-width is about 0.12 with five seeds and about 0.16 with four.
  - A ratio is then decisive on an axis above about 0.79 or below about 0.21.
  - Where the axes disagree, the verdict is "depends on the axis".
  - Both outcomes, the claim falling between its margins and the axes disagreeing, are within the design's limits, stated now.

## Secondary and exploratory outcomes

- **S1: each new judge's own recovery,** F, beside its x and x_m.
- **S2: each judge's sharpness and random share,** at the steps 300, 500, 1000, 2000 and 2850, interpolated along the rungs to each judge's accuracy on each axis. N17 found the loop-free judges sharper during training and matched only at the end.
- **S3: each judge's accuracy over training,** from the batch records: the judge's plain cross-entropy minus head A's, for every arm, and the mixture-form loss for the new arms. N17's saved arms recorded only the plain loss.
- **S4: the blind spots,** as in N17: the clean targets each judge ignores (weight below 0.01), measured both ways against head B's own discount, with their set sizes.
- **S5: head A's cost** in each new arm against `clean`.
- **S6: geometry** against the clean-target ladder, as in N17.
- **S7: `rw-nfree`, the loop-free judge on noisy data** (exploratory, no rule):
  - its F;
  - its x_m, and x with the stated floor;
  - d on each axis, and d / d(`rw-peer`) on each axis;
  - the fraction of clean targets its weight floors at 0;
  - its mixture-form gap to the ideal copy.

  Expected now, from N12's records: a large floored fraction, a gap to the copy near N12's 0.48–0.50, and a shortfall that cannot tell the data from the estimator.

## Necessity audit, at drafting (`NECESSITY_AUDIT.md`)

| Claim the design relies on | Features and ground | The check, and whether it works at the boundary |
|---|---|---|
| `rw-cloop` has the loop and clean data only | Definitional: its weights come from its own prediction, and its targets are K's clean ones | The code; the clean-loop loss-form check; the judge-source check with head C at its own values, which a bug taking head B's logits would fail |
| The two axes place a self-judging judge sensibly | A present mechanism, from N14's records | They may not agree. `sub20-rw` sat 0.72–0.76 behind head A on x and 0.042–0.053 behind the copy in mixture form, ahead of `sub20`. So on x_m `rw-cloop` may sit at the accurate end, beyond every rung but `rw-j100`, and on x at the inaccurate end, near `rw-j0125`. Precondition 6 is per axis, and disagreement has its own outcome |
| x_m is the accuracy that matters for judging | Reply 4: hold the argument to its modest conclusion | Every weight is a function of m / u, but x_m is the self-judging judges' own objective, and it is blind to the neglected targets. It is one of two axes, not the axis |
| The ratio is well defined on each axis | Reply 8 at the boundary: a small d(`rw-peer`) makes it unstable | Precondition 7, per axis. A pooled seed with a small gap widens the interval instead of biasing it |
| The rungs reach both judges on both axes | A present mechanism | The calibration's rule chooses rungs per axis, and the real run trains their union. A judge beyond the rungs on either axis makes the claim untestable, not refuted |
| "The loop" is separated from the course of training | Reply 3 | No. The slow start is the loop's own doing (R4). A support is read as "the loop, with the course it causes" |
| `rw-nfree` tests the training data | Reply 3 | No: it also changes the estimator, which N12's records suggest is poor. It is exploratory for that reason |
| Reused models and seeds measure as before | A present mechanism; N17's records | Precondition 2 |
| The test can fail | Reply 8 | It is refuted if `rw-cloop` sits near the line on both axes, as D predicts. If the axes disagree, as A predicts, the verdict says so |

**The critic's strongest objection, two moves deep:** *"Matching on accuracy is ill-defined for a self-judging judge, so neither axis compares like with like."*
- *The reply:* that is why both axes are used and a verdict needs them to agree. A support then holds however accuracy is matched.
- *The critic's answer:* the two axes are only two of many, and agreement between them does not make the matching neutral.
- *The next move:* accepted. A support is reported as "on both axes N17 and N18 use", not as matched in every sense. If the axes disagree, that is reported as a finding about the matching, not explained away. The blind spots (S4) and the course (S3) are reported beside it.

## Calibration before queuing, including a full-length run

- **A full-length calibration run on the A100,** as its own Colab job, on seed 3001 and the calibration windows 513–1024.
  - **It trains:** `rw-cloop`, `rw-nfree` and `rw-j100`.
  - **It re-measures:**
    - N17's calibration models: `rw-peer` and the five rungs, from the `n17_calib` Drive folder;
    - N13's `n80-rw`, N12's `sub20` and `clean`, N15's calibration `rw-oracle`, and N14's `sub10`, `sub05` and `sub025` of seed 3001.
  - **It reports:**
    - every judge's x, x_m, F and d on each axis;
    - R_loop for seed 3001;
    - the setup checks;
    - `rw-nfree`'s diagnostics.

  The values will be seen and recorded in the deviation log, with what they imply for the claim, before the real run.
- **What the calibration may change:** only which rungs the real run uses, by this rule.
  - **On each axis separately:** take the shortest run of adjacent rungs from `rw-j100`, `rw-j20`, `rw-j10`, `rw-j05`, `rw-j025` and `rw-j0125` that, in seed 3001, is ordered by fraction and spans both the peer and `rw-cloop` with room to spare at each end: 0.03 on x, 0.02 on x_m.
  - **Ties** go to the run nearer `rw-j10` and `rw-j05`. A run may have any length.
  - **The real run trains the union** of the two axes' runs, less the rungs N17's real run already trained (`rw-j10`, `rw-j05`).
  - **If no run spans the peer on either axis,** the real run is not queued, and the calibration's values go to the author.
  - **If no run spans `rw-cloop` on an axis,** the claim is recorded as likely untestable, and the real run is not queued; the values go to the author.
- **Nothing else changes in response to the calibration.**
- **Also:**
  - the untrained checks;
  - the tiny end-to-end run, and the rerun and resume tests;
  - synthetic records for every branch of the rule, both axes, and every precondition;
  - an independent review of the code against this preregistration before the real run (N17's lesson 16).

## Sanity checks before trusting results

- Every data file's fingerprint matches the manifest. The held-out boundary is logged.
- Each new run's fingerprints match N12's `n80`'s. The setup checks are logged.
- Every record names the GPU, the PyTorch version and the code commit. The job runs only on an A100.
- Every number in the report is computed by code from saved JSON records. `n18_analyze.py` is written and committed before any real run, and it is tested on synthetic records covering every branch of the rule.

## Compute, where it runs, and stopping rule

- **Where:** one Colab job, `n18_data`, on an A100, resumable. It runs after the calibration job, `n18_calib`.
- **Cost:**
  - Training: 10 runs at about 2.5 minutes each, and up to 20 more for the rungs.
  - About 80 measurements and 55 model loads.
  - In all, 45 to 105 minutes. The calibration job takes about 25 minutes.
- **Stopping rule:** exactly these runs and measurements, each once. Nothing is added after seeing results except as a dated, exploratory entry in the deviation log.

## What each outcome would teach

| N18a | What we would learn | Next |
|---|---|---|
| Supported | Consistent with the loop, with the course it causes, carrying most of the gap: a judge with the loop falls short even on clean data, however its accuracy is matched on these two axes. The training data is not needed for the gap. | Break the loop without a clean head: a floor on the discount |
| Refuted | A self-judging judge on clean data sits near the loop-free line on both axes: the loop alone does not make the gap, and the noisy training data, or what it does to the judge, is needed. | Separate the data from the estimator: a noisy-trained judge with a better model of the noisy distribution than `rw-nfree`'s single softmax |
| Depends on the axis | Whether a judge with the loop falls short depends on what "as accurate as" means. A self-judging judge trades plain accuracy on the targets it neglects for mixture-form accuracy on the rest, so matching judges by one number of accuracy cannot settle the question. | Compare judges target by target, by the weights they give, instead of by one summary of accuracy (S4) |
| Some effect or inconclusive | The loop accounts for part of the gap, or the design's limits, as stated. | More seeds, or rungs nearer `rw-cloop` |
| Untestable | A judge lay outside the rungs on an axis, or the peer's gap vanished on one. | Recalibrate the rungs, or report the decided axis as description |

## Limitations known now

- **The loop is not separated from the course of training it causes** (R4).
- **Matching on accuracy is axis-dependent** (rival A). Two axes are used, and agreement is required; neither is neutral.
- **The training data's own role is tested only through `rw-nfree`,** whose estimator is likely poor. It is exploratory.
- **The rungs and the oracle need clean labels.** `rw-nfree` does not.
- **Reused seeds and models,** by necessity and declared. Five seeds.
- **One noise rate, one readout width, one scale, 10% of the budget.** The discounts and the unmixing are told the noise.

## Deviation log

*Empty: the preregistration has not been approved.*

## References

Arazo, E., Ortego, D., O'Connor, N. E., & McGuinness, K. (2020). Pseudo-labeling and confirmation bias in deep semi-supervised learning. In *2020 International Joint Conference on Neural Networks (IJCNN)* (pp. 1–8). IEEE. https://doi.org/10.1109/ijcnn48605.2020.9207304

Patrini, G., Rozza, A., Krishna Menon, A., Nock, R., & Qu, L. (2017). Making deep neural networks robust to label noise: A loss correction approach. In *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 2233–2241). IEEE. https://doi.org/10.1109/cvpr.2017.240
