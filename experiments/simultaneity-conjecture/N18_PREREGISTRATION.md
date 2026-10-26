# Preregistration N18: the loop, or what the judge is trained on

Date: 2026-10-26    Status: **draft for the author's approval.** Not built, not run. Nothing in this file is a finding.

Type: confirmatory for N18a (primary) and N18b, each with its own decision rule. Both use two new judges trained for this test, compared with N17's saved judges. Everything else is secondary or exploratory and labelled so.

Parent documents: `N17_NOTE.md` (next conjecture 1; "Limitations"), `N17_PREREGISTRATION.md` (the third head, its checks and the loop-free line), `N16_NOTE.md` (the oracle judge), `N14_NOTE.md` (the self-judged weighting of clean targets alone).

**Relation to earlier results.**
- **The shortfall.** Head B, trained on 80%-scrambled next-word targets and weighted by its own discount (N13's `n80-rw`, called `rw-b1` here), ends 0.337 nats from head A on clean targets. The oracle mask `sub20` ends at 0.163. The shortfall between them, G, is 0.165–0.179 by seed.
- **What N17 found.** N17 compared judges of head B's targets, each a third head, head C, that reads the trunk without training it.
  - A loop-free judge interpolated to head B's own accuracy recovered a third of the shortfall (N17a: 0.334, 97.5% interval [0.318, 0.349], some effect, size unclear). Its line ran between two judges trained at full weight on clean targets at a nested 10% and 5% of positions.
  - A peer trained as head B is, judging itself, was as accurate as head B and recovered 0.041 (secondary).
  - The peer fell short of a loop-free judge of its accuracy by d = −0.290 [−0.315, −0.265] (description; N17b was untestable).
- **What N17 left open.** The loop-free judges and the peer differ in two ways at once.
  - *The loop:* the peer weights its own targets by its own prediction; the loop-free judges train at full weight.
  - *The training data:* the loop-free judges train on clean targets only, chosen with the true mask; the peer trains on head B's targets, 80% of them random.

  N17's note named a third and fourth difference, the course of training and sharpness during it.
- **What stands regardless.** Every earlier verdict stands. N18 asks how the gap N17 described divides. It does not re-decide N17.

## The conjecture

**The gap between a self-judging judge and a loop-free judge of the same accuracy is the loop's, not the training data's. A judge with the loop but trained on clean targets only falls short of the loop-free line by most of the peer's gap. A judge with no loop but trained on head B's scrambled targets falls short by little.**

- **The mechanism.** A judge that weights its own targets by its own prediction gives little weight to the targets it predicts poorly, and so goes on predicting them poorly, whatever data it is given. This would be analogous to the confirmation bias described in self-training (Arazo et al., 2020). A judge trained at full weight has no such loop, on clean data or on noisy.
- **What it forbids.**
  1. The self-judging judge on clean data falling short of the line by a third of the peer's gap or less.
  2. The loop-free judge on noisy data falling short of the line by two thirds of the peer's gap or more.
- **The rivals it is built to exclude.**
  - *The training data (D, N17's R5).* What matters is that the peer trains on noisy targets. It predicts N18a refuted and N18b refuted.
  - *Both, adding up (B).* The loop and the data each account for part of the gap. It predicts both ratios between their margins, summing to about 1.
  - *The course of training (R4).* A self-judging judge starts slowly, because its discount weights every target alike at first, and the gap is that slow start. A self-judging judge's slow start is the loop's own doing, so R4 predicts what the conjecture predicts, and N18 does not separate them. S3 reports the course.
  - *Sharpness (R3).* Judges of equal accuracy can differ in how far apart they put clean and random targets. S2 reports it, over training as well as at the end.
  - *The estimator (E).* The loop-free judge on noisy data must turn its noisy prediction into a clean one, which no other judge does (see "The arms"). If that conversion is poor in ways the accuracy axis does not capture, d(`rw-nfree`) is negative for reasons other than the data. The audit records how a refutation of N18b is read.

## The two-by-two

N18 completes a two-by-two of judges by the loop and the training data. The two cells N17 left empty are trained here.

| | No loop: trained at full weight | The loop: trained by its own discount |
|---|---|---|
| **Clean targets only** | The rungs (N17's `rw-j20` … `rw-j0125`): the loop-free line | **`rw-cloop`** (new): clean targets at K, the same positions as `rw-j20`, weighted by its own discount |
| **Head B's targets, 80% random** | **`rw-nfree`** (new): head B's targets at every position, at full weight, its prediction turned into a clean estimate | `rw-peer` (N17's, saved) |

`rw-cloop` and `rw-j20` differ only in the loop: the same positions, the same clean targets, weighted or not. `rw-nfree` and `rw-peer` differ in the loop and in the estimator: the same targets, at full weight or by its own discount.

## Competing hypotheses

| Hypothesis | N18a: R_loop, `rw-cloop`'s shortfall as a share of the peer's | N18b: R_data, `rw-nfree`'s shortfall as a share of the peer's |
|---|---|---|
| **The conjecture:** the loop | Above two thirds | Below a third |
| D: the training data | Below a third | Above two thirds |
| B: both, adding up | Between | Between, the two summing to about 1 |
| R4: the course of training | Above two thirds (as the conjecture) | Below a third (as the conjecture) |
| E: the estimator | Any | Above a third |

## The arms

Every model is N12's: the trunk with head A and head B, each reading the trunk's final state through its own 64-direction readout, at the 80% noise rate. The training, batches and windows are also N12's. The arms use **N12's seeds 4101–4105**, paired with the saved models of the same seed. Seeds and models are reused, as declared in N13–N17. No earlier verdict is re-decided.

Each new arm adds **head C**, the judge, exactly as in N17. It is a third 64-direction readout of the trunk's state, read detached, with its own initial values and its optimisation tied one way to the main group's. Head B's loss is N13's: the mean over all positions of w(y) · CE, with w from head C.

| Arm | Head C's targets | Head C's loss | Head B's weight w(y) | Trained for N18? |
|---|---|---|---|---|
| `rw-cloop` | Clean targets at K (n80's kept 20%), the positions of `rw-j20` and N12's `sub20` | Each target weighted by head C's own discount, summed over K and divided by all positions (N14's convention for `sub20-rw`) | 0.2 p_C / (0.2 p_C + 0.8 u), from head C's prediction | Yes |
| `rw-nfree` | Head B's targets at every position | Plain cross-entropy, at full weight | 1 − 0.8 u / q_C, floored at 0, from head C's prediction q_C (the unmixed weight, below) | Yes |
| `rw-peer`, `rw-j10`, `rw-j05` | As in N17 | As in N17 | As in N17 | No: N17's saved models |
| `rw-j20`, `rw-j025`, `rw-j0125` | As in N17's calibration | As in N17 | As in N17 | Only if the calibration's rule names them; N17's real run did not train them |
| `rw-b1`, `rw-oracle`, `sub20`, `clean`, `sub10` … `sub0125` | — | — | — | No: N13's, N16's, N12's and N14's saved models |

**The unmixed weight.** A head trained at full weight on head B's targets learns their distribution, q ≈ 0.2 p + 0.8 u, where p is the clean prediction. Under that noise model, the probability that a target y is clean is 0.2 p(y) / q(y) = 1 − 0.8 u(y) / q(y). This turns the noisy prediction into a clean one by inverting the known noise model, the idea behind one of the loss corrections of Patrini et al. (2017), which invert the noise matrix. Where q(y) is below 0.8 u(y), the estimate would be negative, and the weight is set to 0. For a judge whose q is exactly 0.2 p + 0.8 u, the unmixed weight equals the discount computed from p. A setup check confirms this.

The rungs and the oracle need clean labels, as in N17; they are diagnostics. `rw-nfree` needs no clean labels, and it is a method as well as a cell of the two-by-two.

## The measures

- **Δ_A, F, G and a,** as in N17.
- **The judge's mixture-form accuracy, x_m.** Every judge's weight on a target depends only on m(y) / u(y), where m(y) is the judge's mixture-form probability of y:
  - m(y) = 0.2 p(y) + 0.8 u(y) for a judge with a clean prediction p: head A, head B judging itself, the rungs, the peer and `rw-cloop`;
  - m(y) = max(q(y), 0.8 u(y)) for `rw-nfree`.

  So the weight is w(y) = 1 − 0.8 u(y) / m(y) in every case. x_m is the mean over the measurement windows' clean targets of −log m(y), minus the same for head A, at the end of training. It is 0 for head A and positive for a judge less accurate than head A. It is defined for every judge without a floor or a choice, which N17's x is not for `rw-nfree`. N17's x is reported beside it as a secondary measure where defined.
- **The loop-free line, on the x_m axis:** within seed, the rungs' points (x_m, F), sorted by x_m and joined by straight lines. F_lf(x_m) is its value between the rungs; outside them it gives nothing.
- **d(J) = F(J) − F_lf(x_m(J))** for each judge J: how far it falls short of a loop-free clean-trained judge of its accuracy.
- **The two ratios,** within seed, against the peer's shortfall:
  - R_loop = d(`rw-cloop`) / d(`rw-peer`);
  - R_data = d(`rw-nfree`) / d(`rw-peer`).

  Each is the share of the peer's gap that one change reproduces. Under additivity they sum to about 1.

The margins are set on the ratios, not on d itself: N17's lesson that a margin on a difference should be relative to the quantity it is a difference from.

## The claims

**N18a (primary): the loop alone reproduces most of the gap.** The interval for R_loop lies wholly above 0.67.

**N18b: the training data alone reproduces little of it.** The interval for R_data lies wholly below 0.33.

## Refuting results

- **N18a** is refuted if the interval for R_loop lies wholly below 0.33, with a standard deviation of at most 0.25. A self-judging judge on clean data then falls short by a third of the peer's gap or less, and the loop is not most of it.
- **N18b** is refuted if the interval for R_data lies wholly above 0.67, with a standard deviation of at most 0.25. A loop-free judge on noisy data then falls short by two thirds of the peer's gap or more, and the data carries most of it.

## Design

- **Runs:**
  - 10 training runs: `rw-cloop` and `rw-nfree` on seeds 4101–4105.
  - Up to 15 more, if the calibration's rule names rungs that N17's real run did not train (`rw-j20`, `rw-j025`, `rw-j0125`).
  - The measurements of those models, and re-measurements of the saved models, each checked against its committed record:
    - N17's `rw-peer`, `rw-j10` and `rw-j05`, with their head C;
    - N13's `n80-rw`, N16's `rw-oracle`, and N12's `sub20` and `clean`;
    - N14's `sub10` … `sub0125`, for S6.
- **Saved models** are loaded from the `n17_loop`, `n13_signal`, `n16_judge`, `n12_bottleneck` and `n14_dose` Drive folders.
- **Code:** `n18_loop.py`, built on `n17_loop.py`'s head C, forward pass, one-way tie, setup checks and judge measures, adding the two new judges and the x_m measure. `n18_analyze.py` applies the rules.

## Decision rules

**Preconditions:**
1. **Setup.**
   - N17's checks: batch, same-model, identical-heads, analytic, the weight identity, the forward identity, detachment, the passenger check, the copied-peer check, the overflow test, the nesting, the windows, and no training draw in the held-out tail.
   - The judge-source check for both new judges, against their own weight formulas.
   - **The unmixing identity:** for a constructed judge whose q is exactly 0.2 p_A + 0.8 u, the unmixed weight equals the discount from p_A, to a relative 10⁻⁶.
   - **The clean-loop loss form:** `rw-cloop`'s head C loss, computed from head B's logits instead, equals N14's `sub20-rw` loss on the same batch, to a relative 10⁻⁶.
   - Each new run's fingerprints and noise fraction at step 0 equal N12's `n80`'s, and `rw-cloop`'s positions are K.
2. **Must match.** Re-measuring the saved models reproduces their records: losses within 0.001 and ratios within 0.01. The constructed mixture check passes, and the zero-noise copy scores at least 0.999, in every model.
3. **Must differ.** The rotated control scores at most 0.60 in at least 4 of 5 models of each arm.
4. **Head B trained.** In each new arm, Δ_A of head B is at most 1.0 in at least 4 of 5 seeds. Head C carries no training criterion, since a self-judging judge on clean data may collapse, as N14's `sub20-rw` did. Precondition 6 asks only that the line reach it.
5. **The shortfall exists.** G is at least 0.10 nats in every seed.
6. **The loop-free line reaches the judges.** In at least 4 of 5 seeds, the rungs' x_m are ordered by fraction, at least 0.05 apart, and bracket the peer's x_m and the claim's judge's x_m: `rw-cloop`'s for N18a, `rw-nfree`'s for N18b.
7. **The peer's gap exists on this axis.** d(`rw-peer`) is at most −0.10 in at least 4 of 5 seeds, so that the ratios have a denominator well away from 0.

Preconditions 1–5 and 7 apply to both claims; 6 applies to each claim for its own judge. Each claim pools the seeds its precondition 6 brackets. Precondition 7 is a condition on testability, not a filter on seeds. A pooled seed with a small d(`rw-peer`) widens the interval, which makes a verdict harder, not easier.

**Each claim has four outcomes:**

| Outcome | Condition |
|---|---|
| Supported | The interval lies wholly on the claim's side of its margin: R_loop above 0.67; R_data below 0.33 |
| Refuted | It lies wholly past the refuting margin, with a standard deviation of at most 0.25: R_loop below 0.33; R_data above 0.67 |
| Some effect, size unclear | It excludes 0 but meets neither rule |
| Inconclusive | Everything else |

**Intervals.** There are two interval rules, so each uses a Bonferroni-corrected t-interval at the 1 − 0.05 / 2 level (97.5%), with 4 degrees of freedom, or 3 if one seed is not bracketed.

**Why these thresholds.**
- **Two thirds and a third** of the peer's gap. Two thirds is "most of it" and a third is "little of it". Between them, both the loop and the data contribute.
- **A standard deviation cap of 0.25** on refutation. A ratio is noisier than a difference. In N17, d(`rw-peer`) varied across seeds by 0.02 around −0.29, so a numerator varying by 0.03 gives a ratio varying by about 0.1. The cap keeps a refutation from resting on scattered seeds.
- **What the design can detect.** If the ratios vary by about 0.1 across seeds, the 97.5% half-width with five seeds is about 0.16. A ratio is then decisive above about 0.83 or below about 0.17. Values between 0.5 and 0.67, or between 0.33 and 0.5, will most likely be "some effect, size unclear", and that is the design's limit, stated now.

## Secondary and exploratory outcomes

- **S1: each new judge's own recovery,** F, beside its x_m and N17's x where defined. `rw-nfree` is a judge that needs no clean labels, so its F also asks the method question.
- **S2: each judge's sharpness and random share,** at the steps 300, 500, 1000, 2000 and 2850, interpolated to each judge's x_m along the rungs. N17 found the loop-free judges sharper during training and matched only at the end.
- **S3: each judge's accuracy over training,** from the batch records: the judge's mixture-form loss on the batch's clean targets minus head A's.
- **S4: the blind spots,** as in N17: the clean targets each judge ignores (weight below 0.01), measured both ways against head B's own discount, with their set sizes.
- **S5: head A's cost** in each new arm against `clean`.
- **S6: geometry** against the clean-target ladder, as in N17.
- **S7: additivity,** R_loop + R_data by seed, as description.
- **S8: the two-by-two on N17's axis.** The same d and ratios with N17's x where it is defined. For `rw-nfree`, x uses the clean estimate max(q − 0.8 u, 0.01 · u) / 0.2, renormalised, and is reported as description only.

## Necessity audit, at drafting (`NECESSITY_AUDIT.md`)

| Claim the design relies on | Features and ground | The check, and whether it works at the boundary |
|---|---|---|
| `rw-cloop` has the loop and clean data only | Definitional: its weights come from its own prediction, and its targets are K's clean ones | The code; the clean-loop loss-form check; the judge-source check. At the boundary, a collapsed judge sits far down the axis, and precondition 6 asks whether the line reaches it |
| `rw-nfree` has no loop | Definitional: its own loss is unweighted | The code. A weak second-order path remains, as in N17: head B's loss, weighted by head C, trains the trunk head C reads |
| The unmixed weight is the right judge for a noisy-trained head | Definitional under the noise model; a present mechanism in practice | The unmixing identity on a constructed judge. Whether the real head's q is close to 0.2 p + 0.8 u is not guaranteed: N12 found a single softmax cannot represent the copy well. This is rival E, reported through S2 and S4 |
| x_m is the accuracy that matters for judging | Definitional: every judge's weight is a function of m / u | It changes the axis from N17's x. S8 reports N17's axis beside it, so a conclusion that depends on the axis shows |
| The ratios are well defined | Reply 8 at the boundary: a small d(`rw-peer`) makes them unstable | Precondition 7 requires the peer's gap in at least 4 of 5 seeds. A pooled seed with a small gap widens the interval instead of biasing it |
| The rungs reach the new judges | A present mechanism: `rw-cloop` may be far less accurate than any rung | The calibration chooses the rungs; precondition 6 checks each claim's judge. A judge beyond the lowest rung makes its claim untestable, not refuted |
| "The loop" is separated from the course of training | Reply 3: another mechanism | No. A self-judging judge's slow start is the loop's doing, so R4 and the conjecture predict the same (S3). A support of N18a is read as "the loop, with the course it causes" |
| `rw-nfree` differs from the peer only in the loop | Reply 3 | No: it also uses a different estimator (rival E). A support of N18b holds whatever the estimator, since it puts `rw-nfree` near the line. A refutation is read with E named |
| Reused models and seeds measure as before | A present mechanism; N17's records | Precondition 2: every reused model reproduces its record |
| The tests can fail | Reply 8: every test | N18a fails if `rw-cloop` sits near the line, as D predicts. N18b fails if `rw-nfree` sits far below it, as D or E predicts |

**The critic's strongest objection, two moves deep:** *"The unmixed judge is a different estimator, not just different data. A shortfall for `rw-nfree` would say nothing about data."*
- *The reply:* x_m matches it on the quantity its weights depend on, and a support of N18b, that it falls short by little, holds whatever the estimator.
- *The critic's answer:* matching the mean of −log m does not match its spread across targets, and a refutation of N18b still could not tell the data from the estimator.
- *The next move:* accepted. A refutation of N18b is reported as "the data, or the estimator that noisy data requires", with S2 and S4 describing how the weights differ. The mirror case is the decisive one: if `rw-cloop` falls short by most of the peer's gap (N18a supported), the loop does it on clean data, where there is no estimator question.

## Calibration before queuing, including a full-length run

- **A full-length calibration run on the A100,** as its own Colab job, on seed 3001 and the calibration windows 513–1024.
  - **It trains:** `rw-cloop` and `rw-nfree`.
  - **It re-measures:**
    - N17's calibration models: `rw-peer` and all five rungs, from the `n17_calib` Drive folder;
    - N13's `n80-rw`, N12's `sub20` and `clean`, N15's calibration `rw-oracle`, and N14's `sub10`, `sub05` and `sub025` of seed 3001.
  - **It reports:**
    - every judge's x_m, F and d;
    - the ratios for seed 3001;
    - the setup checks;
    - the secondary measures.

  The values will be seen and recorded in the deviation log, with what they imply for the claims, before the real run.
- **What the calibration may change:** only which rungs the real run uses, by this rule.
  - The real run uses the shortest run of adjacent rungs from `rw-j20`, `rw-j10`, `rw-j05`, `rw-j025` and `rw-j0125` that meets two conditions in seed 3001: the rungs are ordered by fraction, and they span the x_m of the peer, `rw-cloop` and `rw-nfree` with at least 0.03 to spare at each end.
  - Rungs that N17's real run did not train are trained on the five real seeds with N17's code.
  - If no run spans a judge, the run spanning the others is used, and that judge's claim is recorded as likely untestable before the real run.
  - If neither claim's judge is spanned, the real run is not queued, and the calibration's values go to the author.
- **Nothing else changes in response to the calibration.**
- **Also:** the untrained checks, the tiny end-to-end run, the rerun and resume tests, synthetic records for every branch of the two rules and every precondition, and an independent review of the code against this preregistration before the real run (N17's lesson 16).

## Sanity checks before trusting results

- Every data file's fingerprint matches the manifest. The held-out boundary is logged.
- Each new run's fingerprints match N12's `n80`'s. The setup checks are logged.
- Every record names the GPU, the PyTorch version and the code commit. The job runs only on an A100.
- Every number in the report is computed by code from saved JSON records. `n18_analyze.py` is written and committed before any real run, and it is tested on synthetic records covering every branch of the rules.

## Compute, where it runs, and stopping rule

- **Where:** one Colab job, `n18_data`, on an A100, resumable. It runs after the calibration job, `n18_calib`.
- **Cost:**
  - Training: 10 runs at about 2.5 minutes each, and up to 15 more if the rule adds rungs.
  - About 70 measurements and 55 model loads.
  - In all, 45 to 90 minutes. The calibration job takes about 20 minutes.
- **Stopping rule:** exactly these runs and measurements, each once. Nothing is added after seeing results except as a dated, exploratory entry in the deviation log.

## What each outcome would teach

| N18a | N18b | What we would learn | Next |
|---|---|---|---|
| Supported | Supported | Consistent with the loop, with the course it causes, carrying the gap. A judge with the loop falls short even on clean data, and a judge without it falls short little even on noisy data. The training data is not the cause. | Break the loop without a clean head: a floor on the discount, or `rw-nfree` itself as a method (S1) |
| Refuted | Refuted | The training data carries the gap: a self-judging judge on clean data does about as well as a loop-free one, and a loop-free judge on noisy data falls short (or the estimator noisy data requires, rival E). | Separate the data from the estimator, for example with a noisy-trained judge using a learned noise model |
| Supported | Refuted | Either change alone reproduces most of the gap, so the gap is not additive: the loop and the data each suffice. | A judge with both changes undone in steps, to find where the gap opens |
| Refuted | Supported | Neither change alone reproduces it: the gap needs both, an interaction of the loop with noisy data. | Measure where the self-judging judge's neglect falls on noisy against clean targets (S4) |
| Some effect, inconclusive, or a mix | Any | Both contribute in part, or the design's limits, as stated. S7 says whether the parts add up. | More seeds, or a finer two-by-two |
| Untestable (precondition 6 or 7) | Any | A judge lay outside the rungs, or the peer's gap vanished on this axis. | Recalibrate the rungs; compare on N17's axis (S8) |

## Limitations known now

- **The loop is not separated from the course of training it causes** (R4).
- **`rw-nfree` differs from the peer in its estimator** as well as its loop (rival E).
- **The accuracy axis is new.** S8 reports N17's beside it.
- **The rungs and the oracle need clean labels.** `rw-nfree` does not.
- **Reused seeds and models,** by necessity and declared. Five seeds.
- **One noise rate, one readout width, one scale, 10% of the budget.** The discounts and the unmixing are told the noise.

## Deviation log

*Empty: the preregistration has not been approved.*

## References

Arazo, E., Ortego, D., O'Connor, N. E., & McGuinness, K. (2020). Pseudo-labeling and confirmation bias in deep semi-supervised learning. In *2020 International Joint Conference on Neural Networks (IJCNN)* (pp. 1–8). IEEE. https://doi.org/10.1109/ijcnn48605.2020.9207304

Patrini, G., Rozza, A., Krishna Menon, A., Nock, R., & Qu, L. (2017). Making deep neural networks robust to label noise: A loss correction approach. In *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 2233–2241). IEEE. https://doi.org/10.1109/cvpr.2017.240
