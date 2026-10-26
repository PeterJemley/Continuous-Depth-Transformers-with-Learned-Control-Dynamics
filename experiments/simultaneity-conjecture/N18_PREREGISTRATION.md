# Preregistration N18: the loop, with the judge's data held fixed

Date: 2026-10-26    Status: **draft for the author's approval.** Not built, not run. Nothing in this file is a finding.

Type: confirmatory for N18a, the single claim. It uses one new judge trained for this test, `rw-cloop`, compared with a judge of N17's design trained on the same targets, `rw-j20`, and with N17's saved peer. A second new judge, `rw-nfree`, is exploratory, with no claim. Everything else is secondary or exploratory and labelled so.

Parent documents: `N17_NOTE.md` (next conjecture 1; "Limitations"), `N17_PREREGISTRATION.md` (the third head, its checks and the loop-free line), `N16_NOTE.md` (the two-by-two of judge and random targets), `N14_NOTE.md` (`sub20-rw`, the self-judged weighting of clean targets alone), `N13_NOTE.md` (the discount as the mixture head's learning signal), `N12_NOTE.md` (the standard head on noisy targets).

**Relation to earlier results.**
- **The shortfall.** Head B, trained on 80%-scrambled next-word targets and weighted by its own discount (N13's `n80-rw`, called `rw-b1` here), ends 0.337 nats from head A on clean targets. The oracle mask `sub20` ends at 0.163. The shortfall between them, G, is 0.165–0.179 by seed.
- **What N17 found.** In each new arm, a third head, head C, read the trunk without training it and judged head B's targets.
  - Interpolated to head B's own accuracy, a loop-free judge recovered a third of the shortfall (N17a: 0.334, 97.5% interval [0.318, 0.349], some effect, size unclear). The two loop-free judges trained at full weight on clean targets at a nested 10% and 5% of positions.
  - A peer trained as head B is, judging itself, was as accurate as head B and recovered 0.041 (secondary).
  - It fell short of a loop-free judge of its accuracy by d = −0.290, 95% interval [−0.315, −0.265] (description; N17b was untestable).
- **What N17 left open.** The loop-free judges and the peer differ in two ways at once.
  - *The loop:* the peer weights its own targets by its own prediction; the loop-free judges train at full weight.
  - *The training data:* the loop-free judges train on clean targets only, chosen with the true mask; the peer trains on head B's targets at every position, 80% of them random.

  The course of training and sharpness during it also differ.
- **What N16 found about the loop without random targets** (exploratory). Weighting the clean targets of the kept 20% by head B's own discount, with no random targets (N14's `sub20-rw`), left head B 0.57 nats further from head A than the same targets at full weight (`sub20`). Adding the random targets improved the self-judged head by 0.40 nats. So the loop is costly on clean targets alone, and the random targets may temper it.
- **Why N18 compares directly, not at matched accuracy.** An earlier draft compared a self-judging judge on clean targets with loop-free judges of the same accuracy, as N17 compared the peer. A fresh reader's check found that comparison ill-posed for this judge.
  - N14's `sub20-rw` ended 0.72–0.76 nats behind head A on plain cross-entropy, at the inaccurate end of N17's loop-free judges.
  - In mixture form, the form its discount is computed from, it ended 0.042–0.053 behind its copy, ahead of `sub20`'s 0.112–0.123.
  - If the new judge sits likewise, the two measures place it at opposite ends of the loop-free line, and "a loop-free judge of the same accuracy" names two different judges.
  - The loop is also what makes this judge inaccurate on plain cross-entropy, so matching on it would remove part of the effect under test.

  N18 therefore compares the judge with the loop against the loop-free judge trained on the same targets at the same positions, and reports the matched comparisons as description (S8).
- **What stands regardless.** Every earlier verdict stands. N18 asks whether the loop is enough to produce the shortfall N17 described. It does not re-decide N17.

## The conjecture

**The loop alone is enough to produce most of the peer's shortfall. Given the same clean targets at the same positions, a judge that weights them by its own discount leaves head B with at least two thirds of the shortfall the peer leaves, both measured from the judge trained on those targets at full weight.**

- **The mechanism.** A judge that weights its own targets by its own prediction gives little weight to the targets it predicts poorly, and so goes on predicting them poorly, whatever data it is given. This would be analogous to the confirmation bias described in self-training (Arazo et al., 2020). A judge trained at full weight has no such loop.
- **What it forbids.** The judge with the loop on clean targets costing head B a third of the peer's shortfall or less.
- **What a support would not show.**
  - It would show that the loop is enough to produce the shortfall, with the data fixed. It would not show that the peer's own shortfall is the loop's. With random targets present the loop may act differently (rival I).
  - It would not say whether the loop costs through the judge's plain accuracy or beyond it (rival P).
  - It would not decide the matched-accuracy comparison N17 left open.

**The rivals it is built to exclude, and readings it does not separate.**
- *The training data (D, N17's R5).* What makes the peer fall short is its noisy targets. A judge with the loop on clean targets then costs head B little more than `rw-j20` does: R near 0, refuted.
- *The mixture form (M).* A judge's usefulness follows how well it predicts in mixture form, the quantity its weights are computed from. On that measure a self-judging judge on clean targets may beat the loop-free one, as `sub20-rw` beat `sub20`. Then `rw-cloop` helps head B more than `rw-j20` does: R below 0, refuted.
- *Both, in part (B).* R between a third and two thirds.
- *The random targets temper the loop (I), from N16's two-by-two.* The loop costs more on clean targets alone than with random targets present: R above 1, a support. I predicts what the conjecture predicts, and more.
- *Plain accuracy (P), not separated.*
  - P says a judge's usefulness follows its plain accuracy, whoever trains it.
  - If `rw-cloop` is as inaccurate as `sub20-rw`, it then sits near N17's loop-free line at its plain accuracy. In N17's calibration seed the line lay near F = −0.65 at x = 0.74. That gives R near 2, a support.
  - P and the conjecture predict the same verdict, and N18 does not separate them. N17 already found the peer below the line at its plain accuracy, against P for the peer. S8 describes where `rw-cloop` sits.
- *The course of training (R4), not separated.*
  - A self-judging judge's weights start low, since its prediction starts poor, and the low weights slow its own learning.
  - N17 found the peer slow to start, partly by the loop's own doing.
  - The course is part of what the loop does here, and N18 does not separate them. S3 reports it.
- *Sharpness (R3).* Judges can differ in how far apart they put clean and random targets. S2 reports it over training.

## The judges

N18 fills in N17's two-by-two of judges by the loop and the training data. The claim compares the clean row; the noisy row is exploratory.

| | No loop: trained at full weight | The loop: trained by its own discount |
|---|---|---|
| **Clean targets at K** | **`rw-j20`** (N17's design, trained on the real seeds for N18) | **`rw-cloop`** (new): the same targets at the same positions, each weighted by head C's own discount |
| **Head B's targets, 80% random, at every position** | **`rw-nfree`** (new, exploratory): at full weight, its prediction unmixed | `rw-peer` (N17's, saved) |

- **The claim's comparison, the clean row.** `rw-cloop` and `rw-j20` share the positions K, the clean targets, and the normalisation of head C's loss, summed over K and divided by all positions. Head B's weight is computed from head C's prediction in the same way in both. They differ only in the weight on each term of head C's own loss: 1 in `rw-j20`, head C's own discount in `rw-cloop`.
- **The noisy row's comparison** (`rw-nfree` against `rw-peer`) also changes the estimator, since `rw-nfree` must unmix its prediction, and is exploratory.
- **"The training data"** here changes the noise and the count together: head B's targets at every position, against clean targets at 20% of them.

## Competing hypotheses

| Hypothesis | R (N18a) | Head B under `rw-cloop`, against under `rw-j20` |
|---|---|---|
| **The conjecture:** the loop is enough | Above two thirds | Much further from head A |
| I: the random targets temper the loop | Above 1 | Further from head A than under the peer |
| P: plain accuracy (not separated) | Near 2, if `rw-cloop` is as inaccurate as `sub20-rw` | Much further from head A |
| B: both, in part | Between a third and two thirds | Somewhat further |
| D: the training data | Near 0 | About the same |
| M: the mixture form | Below 0 | Closer to head A |
| R4: the course of training | As the conjecture | As the conjecture |

## The arms

Every model is N12's: the trunk with head A and head B, each reading the trunk's final state through its own 64-direction readout, at the 80% noise rate. The training, batches and windows are also N12's. The arms use **N12's seeds 4101–4105**, paired with the saved models of the same seed. Seeds and models are reused, as declared in N13–N17. No earlier verdict is re-decided.

Each new arm adds **head C**, the judge, exactly as in N17. It is a third 64-direction readout of the trunk's state, read detached, with its own initial values and its optimisation tied one way to the main group's. Head B's loss is N13's: the mean over all positions of w(y) · CE, with w from head C.

| Arm | Head C's targets | Head C's loss | Head B's weight w(y) | Trained for N18? |
|---|---|---|---|---|
| `rw-cloop` | Clean targets at K, the kept 20% | Each target's cross-entropy weighted by head C's own discount on it, without gradient through the weight, summed over K and divided by all positions: N14's convention for `sub20-rw` | 0.2 p_C / (0.2 p_C + 0.8 u), from head C's prediction, as in N17 | Yes |
| `rw-j20` | Clean targets at K: N17's rung at 20%, whose nested draw keeps every position of K | Full weight, summed over K and divided by all positions, as in N17 | As in `rw-cloop` | Yes, on the real seeds, since N17's real run did not train it; in the calibration, again on seed 3001, as a check against N17's |
| `rw-nfree` (exploratory) | Head B's targets at every position | Plain cross-entropy at full weight, the mean over positions: N12's `n80` loss | max(1 − 0.8 u / q_C, 0), the unmixed weight (below) | Yes |
| `rw-peer`, `rw-j10`, `rw-j05` | As in N17 | As in N17 | As in N17 | No: N17's saved models |
| `rw-b1`, `rw-oracle`, `sub20`, `clean`, `sub10` … `sub0125` | — | — | — | No: N13's, N16's, N12's and N14's saved models |

Head C's own discount in `rw-cloop` is computed from its prediction of the clean target at each position of K, by N15's weight function, which N17 uses for head B's weight. It is a different quantity from head B's weight, which head C computes on head B's targets at every position.

**The unmixed weight** (`rw-nfree`).
- A head trained at full weight on head B's targets learns their distribution, q ≈ 0.2 p + 0.8 u, with p the clean prediction.
- Under that noise model, the probability that a target y is clean is 1 − 0.8 u(y) / q(y). Where q(y) is below 0.8 u(y), the weight is set to 0.
- This inverts the known noise model, the idea behind one of the loss corrections of Patrini et al. (2017).

N12 found that a single softmax trained on noisy targets fits their distribution poorly. Its clean-target loss ended 0.48–0.50 nats above the copy's mixture-form loss, and placing it in the mixture form doubled that gap (N13b). So `rw-nfree`'s weights may be poor, with many clean targets weighted 0. A shortfall from it could not tell the training data from the estimator. That is why it carries no claim.

`rw-cloop` and `rw-j20` train on clean targets chosen with the true mask, as N17's rungs did. They are diagnostics, not methods. `rw-nfree` needs no clean labels.

## The measures

- **Δ_A, F, G and a,** as in N17. Δ_A is head B's clean-target loss minus head A's.
- **R = (Δ_A(`rw-cloop`) − Δ_A(`rw-j20`)) / (Δ_A(`rw-peer`) − Δ_A(`rw-j20`)),** within seed.
  - R is the share of the peer's shortfall from `rw-j20` that the loop alone reproduces on clean targets.
  - It equals (F(`rw-j20`) − F(`rw-cloop`)) / (F(`rw-j20`) − F(`rw-peer`)). G cancels, so R does not depend on `rw-b1` or `sub20`.
  - 1 − R is the share that changing the data does with the loop present, from `rw-cloop` to the peer.
- **For description only (S8): two accuracy axes and the loop-free line.** Each is measured at the end of training, on the measurement windows' clean targets, against the same model's head A.
  - **x, N17's:** the judge's plain cross-entropy minus head A's. It is not defined for `rw-nfree`, which has no clean prediction.
  - **x_m, the mixture axis:** the mean of −log m(y) minus the mean of −log(0.2 p_A(y) + 0.8 u(y)), where m is the judge's mixture-form prediction.
    - For a judge with a clean prediction p, m = 0.2 p + 0.8 u.
    - For `rw-nfree`, m = max(q, 0.8 u), normalised to sum to 1. Unnormalised, it sums to more than 1 wherever a weight is floored, and its loss would be biased in its favour.
    - x_m is the objective a self-judging judge is trained on, so it favours such judges. It is driven by targets with weight near 1 and nearly blind to targets with weight near 0.
  - **The loop-free line, and d:** within seed, the rungs' points (axis value, F), sorted and joined by straight lines, with nothing outside them. d(J) = F(J) − the line at judge J's axis value, where the line reaches.

## The claim

**N18a: the loop is enough to produce most of the shortfall.** The interval for R lies wholly above 0.67.

## Refuting result

**N18a** is refuted if the interval for R lies wholly below 0.33, with a standard deviation of at most 0.25. A judge with the loop on clean targets then costs head B a third of the peer's shortfall or less. If the interval also lies wholly below 0, the loop helped on clean targets, as M predicts, and the report says so.

## Design

- **Runs:** 15 training runs, `rw-cloop`, `rw-j20` and `rw-nfree` on seeds 4101–4105.
- **Measurements:** those models, and re-measurements of the saved models, each checked against its committed record:
  - N17's `rw-peer`, `rw-j10` and `rw-j05`, with their head C;
  - N13's `n80-rw`, N16's `rw-oracle`, and N12's `sub20` and `clean`;
  - N14's `sub10` … `sub0125`, for S6.
- **Saved models** are loaded from the `n17_loop`, `n13_signal`, `n16_judge`, `n12_bottleneck` and `n14_dose` Drive folders. The calibration's come from `n17_calib`, `n13_signal`, `n15_sharpen`, `n12_bottleneck` and `n14_dose`.
- **Code:** `n18_loop.py`, built on `n17_loop.py`'s head C, forward pass, one-way tie, setup checks and judge measures. It adds `rw-cloop` and `rw-nfree`, the mixture axis, `rw-nfree`'s diagnostics and the new checks. `n18_analyze.py` applies the rule.

## Decision rule

**Preconditions:**
1. **Setup.**
   - N17's checks: batch, same-model, identical-heads, analytic, the weight identity, the forward identity, the passenger check, the copied-peer check, the overflow test, the windows, and no training draw in the held-out tail.
   - Detachment, run for `rw-cloop` and `rw-nfree`.
   - **The judge-source check for each new spec, with head C at its own initial values.** The weight in head B's loss must equal, bit for bit, the weight from head C's logits by that spec's formula, and must differ from head B's own discount.
   - **The clean-loop loss form.** `rw-cloop`'s head C loss term, before the factor of 0.5 both losses carry, is computed once from head B's logits instead of head C's. It must equal N14's `sub20-rw` loss on the same batch, with a relative difference of at most 10⁻⁶: the largest absolute difference divided by the largest absolute value. N14's discount and N15's weight differ by up to 4 × 10⁻⁷.
   - **The clean-loop gradient.** On the same batch, the gradient of that term with respect to the logits must equal the gradient of N14's `sub20-rw` loss, by the same relative measure. A weight left with its gradient attached would fail this check, though not the check on the loss.
   - **The noisy-loss form.** `rw-nfree`'s head C loss term, computed from head B's logits instead, must equal N12's `n80` head B loss on the same batch, by the same measure.
   - **The unmixing identity.** Take a constructed judge whose q is exactly 0.2 p_A + 0.8 u, computed in float64. Its unmixed weight must equal the discount from p_A, by the same measure, N15's convention. This is algebra; it checks the code, not the real head.
   - **Positions.** Each new run's fingerprints and noise fraction at step 0 must equal N12's `n80`'s. Head C's positions in `rw-cloop` and in `rw-j20` must be exactly K.
   - **Code identity, checked in the calibration.** N18's `rw-j20` of seed 3001 must reproduce N17's calibration record of `rw-j20`: head B's Δ_A and head C's x within 0.001, agreement within 0.01. The largest difference between its weights and N17's saved model is reported; it is expected to be 0 if the GPU and the PyTorch version match.
2. **Must match.** Re-measuring the saved models reproduces their records: losses within 0.001 and ratios within 0.01. The constructed mixture check passes, and the zero-noise copy scores at least 0.999, in every model. x_m of the saved judges is measured afresh and has no record to match.
3. **Must differ.** The rotated control scores at most 0.60 in at least 4 of 5 models of `rw-cloop`, `rw-j20` and the re-measured `rw-peer`.
4. **The runs trained.**
   - In every seed, `rw-j20`'s head B Δ_A and head C x are each at most 1.0, as N17's rungs had to be.
   - `rw-cloop` carries no criterion on how well its heads predict, since their failure is a possible effect of the loop. N14's `sub20-rw` reached 0.72–0.76 on plain cross-entropy.
   - Every logged loss of `rw-cloop` and `rw-j20` is finite.
   - `rw-nfree` enters no precondition.
5. **The peer's shortfall from `rw-j20` exists.** Δ_A(`rw-peer`) − Δ_A(`rw-j20`) is at least 0.05 nats in every seed. That keeps R's denominator well away from 0.

Preconditions 4 and 5 hold in every seed, not in 4 of 5, so that no seed is dropped on a condition involving `rw-j20`, which enters both sides of R.

**The outcome:**

| Outcome | Condition |
|---|---|
| Supported | R's interval lies wholly above 0.67 |
| Refuted | It lies wholly below 0.33, with a standard deviation of at most 0.25 |
| Some effect, size unclear | Neither, and the interval lies wholly above 0 |
| Inconclusive | Everything else |
| Untestable | A precondition fails |

**Interval:** a two-sided 95% t-interval over the five seeds, with 4 degrees of freedom. There is one claim, so no correction applies.

**Why these thresholds.**
- **Two thirds and a third** of the peer's shortfall. Two thirds is "most of it" and a third is "little of it". Between them, both the loop and the data contribute.
- **The denominator's floor, 0.05 nats,** is about half the denominator's expected size.
  - In N17's calibration seed, Δ_A(`rw-peer`) − Δ_A(`rw-j20`) was 0.123.
  - In N17's real run, Δ_A(`rw-peer`) was 0.323–0.338, and Δ_A(`rw-j10`) was 0.252–0.265. `rw-j20` trains on twice `rw-j10`'s targets, so it should lie lower still, and the denominator should be about 0.10.
- **A standard deviation cap of 0.25** on refutation, as a ratio is noisier than a difference.
- **What the design can detect.**
  - In N17's real run, within-seed differences of head B's Δ_A between arms varied across seeds with a standard deviation of 0.004–0.008 nats.
  - If R's numerator varies that much, R's standard deviation is about 0.05–0.1. If it varies as `sub20-rw`'s own accuracy did, by 0.02, it is about 0.2.
  - With five seeds, the 95% half-width is then 0.06–0.25.
  - So an interval clears a margin when R lies above about 0.73–0.92 or below about 0.08–0.27. P predicts about 2 and M below 0, both well beyond either margin.

## Secondary and exploratory outcomes

- **S1: each judge's own recovery,** F, with its Δ_A, x and x_m.
- **S2: each judge's sharpness and random share,** at the steps 300, 500, 1000, 2000 and 2850, compared directly between `rw-cloop` and `rw-j20`, and between `rw-nfree` and `rw-peer`.
- **S3: each judge's accuracy over training,** from the batch records. For every arm, the judge's plain cross-entropy minus head A's; for the new arms, the mixture-form loss as well. N17's saved arms recorded only the plain loss.
- **S4: the blind spots,** as in N17: the clean targets each judge ignores (weight below 0.01), measured both ways against head B's own discount, with their set sizes.
- **S5: head A's cost** in each new arm against `clean`.
- **S6: geometry** against the clean-target ladder, as in N17.
- **S7: `rw-nfree`, the loop-free judge on noisy data** (exploratory, no rule):
  - its F, and its x_m;
  - the fraction of clean targets its weight floors at 0;
  - its mixture-form gap to the ideal copy;
  - the other contrasts of the two-by-two: Δ_A(`rw-nfree`) − Δ_A(`rw-peer`), the loop with noisy data; and Δ_A(`rw-nfree`) − Δ_A(`rw-j20`), the data without the loop. Each also changes the estimator.
- **S8: the matched-accuracy comparisons** (description).
  - Each judge's x and x_m are placed against the loop-free line on each axis, with d where the line reaches it.
  - In the calibration seed, all five of N17's rungs exist. In the real seeds, the line runs through `rw-j20`, `rw-j10` and `rw-j05` only, so it may not reach `rw-cloop` on either axis.
  - This is where P and M would show. P puts `rw-cloop` near the line on x; M puts it near the line on x_m.

## Necessity audit, at drafting (`NECESSITY_AUDIT.md`)

| Claim the design relies on | Features and ground | The check, and whether it works at the boundary |
|---|---|---|
| `rw-cloop` and `rw-j20` differ only in the loop | Definitional: the same positions, targets, normalisation and head B weight; only head C's own weight differs | The code; the positions check; the clean-loop loss and gradient checks, which a weight with its gradient attached would fail; the judge-source check with head C at its own values, which a bug taking head B's logits would fail |
| N18's code trains `rw-j20` as N17's did | A present mechanism | The code-identity check in the calibration, against N17's saved calibration model and record |
| The direct contrast measures the loop | Reply 3: what does it hold fixed? | It holds the data fixed and changes the loop, including what the loop does to the judge's accuracy (P) and course (R4). It is read as "the loop, with what it does", not as the loop beyond accuracy |
| The result transfers to the peer | Reply 3 | No: the loop may act differently with random targets present (I; N16's two-by-two). A support says the loop is enough, not that the peer's shortfall is the loop's |
| The ratio is well defined | Reply 8 at the boundary: a small denominator makes it unstable | Precondition 5, in every seed. The denominator was 0.123 in N17's calibration seed |
| `rw-nfree` tests the training data | Reply 3 | No: it also changes the estimator, which N12's records suggest is poor. It is exploratory for that reason |
| Reused models and seeds measure as before | A present mechanism; N17's records | Precondition 2 |
| The test can fail | Reply 8 | It is refuted if `rw-cloop` costs head B little more than `rw-j20` does, as D predicts, or helps it, as M predicts. P and M predict opposite sides of both margins, and N14's records do not say which holds |

**The critic's strongest objection, two moves deep:** *"N17 left open whether the peer's shortfall at matched accuracy is the loop's. The direct contrast does not answer that."*
- *The reply:* true, and stated. For a self-judging judge on clean targets the matched comparison is ill-posed. N14's records suggest the two measures of accuracy would place it at opposite ends of the loop-free line, and the loop is what makes it inaccurate. The direct contrast is the comparison that is well defined.
- *The critic's answer:* then N18 asks a different question from the one N17 left.
- *The next move:* accepted in part. N18 asks whether the loop, with the data fixed, is enough to produce a shortfall of the peer's size, which N17 could not ask. The matched comparisons are reported as description (S8). A verdict is not read as dividing N17's d.

## Calibration before queuing, including a full-length run

- **A full-length calibration run on the A100,** as its own Colab job, on seed 3001 and the calibration windows 513–1024.
  - **It trains:** `rw-cloop`, `rw-nfree`, and `rw-j20` again, for the code-identity check.
  - **It re-measures:**
    - N17's calibration models: `rw-peer` and the five rungs, for S8 on both axes;
    - N13's `n80-rw`, N12's `sub20` and `clean`, N15's calibration `rw-oracle`, and N14's `sub10`, `sub05` and `sub025` of seed 3001.
  - **It reports:**
    - R for seed 3001;
    - every judge's Δ_A, F, x and x_m;
    - d on each axis where the line reaches;
    - the setup checks and the code-identity check;
    - `rw-nfree`'s diagnostics, and the timings.

  The values will be seen and recorded in the deviation log, with what they imply for the claim, before the real run.
- **What the calibration may change:** nothing in the design. The real run is not queued, and the values go to the author, if any of these happens:
  - a setup check or the code-identity check fails;
  - precondition 4 or 5 fails in seed 3001;
  - a loss in `rw-cloop` is not finite.
- **Also:**
  - the untrained checks;
  - planted bugs that the checks must catch: a weight with its gradient attached in `rw-cloop`, head C judging from head B's logits, `rw-cloop` on every position, and `rw-nfree`'s weight left without its floor;
  - the tiny end-to-end run, and the rerun and resume tests;
  - synthetic records for every branch of the rule and every precondition;
  - an independent review of the code against this preregistration before the real run (N17's lesson 16).

## Sanity checks before trusting results

- Every data file's fingerprint matches the manifest. The held-out boundary is logged.
- Each new run's fingerprints match N12's `n80`'s. The setup checks are logged.
- Every record names the GPU, the PyTorch version and the code commit. The job runs only on an A100.
- Every number in the report is computed by code from saved JSON records. `n18_analyze.py` is written and committed before any real run, and it is tested on synthetic records covering every branch of the rule.

## Compute, where it runs, and stopping rule

- **Where:** one Colab job, `n18_data`, on an A100, resumable. It runs after the calibration job, `n18_calib`.
- **Cost:** 15 training runs at about 2.5 minutes each, with the measurements and model loads. N17's real run, with the same number of training runs, took 45 minutes, so about 45–60 minutes. The calibration job takes about 25 minutes.
- **Stopping rule:** exactly these runs and measurements, each once. Nothing is added after seeing results except as a dated, exploratory entry in the deviation log.

## What each outcome would teach

| N18a | What we would learn | Next |
|---|---|---|
| Supported | With the data held at clean targets, a judge's own loop is enough to cost head B most of what the peer costs it. The noisy data is not needed for a shortfall of that size. An interval wholly above 1 would fit the random targets tempering the loop (I). Whether the loop costs through the judge's plain accuracy or beyond it stays open (P; S8). | Break the loop without a clean head: a floor on the discount |
| Refuted, interval above 0 | With the data fixed, the loop costs head B little, so the peer's shortfall needs its noisy data, or what the noise does with the loop. This is consistent with D. | Separate the data from the estimator: a noisy-trained judge with a better model of the noisy distribution than `rw-nfree`'s single softmax |
| Refuted, interval below 0 | The loop helped on clean targets, as M predicts: a judge trained by its own discount was the better judge. Its usefulness would then follow its mixture-form accuracy more than its plain accuracy. | Test M directly, across judges, by the accuracy that predicts their usefulness |
| Some effect | The loop accounts for part of the shortfall, by between a third and two thirds, or by an amount the design cannot place. | More seeds |
| Inconclusive | The design's limits, as stated. | More seeds |
| Untestable | The peer's shortfall from `rw-j20` vanished, or a run failed to train. | Inspect the failed precondition |

## Limitations known now

- **The loop is changed on clean targets only.** With random targets present it may act differently (I; N16's two-by-two). A support shows the loop is enough, not that the peer's shortfall is the loop's.
- **The direct contrast includes what the loop does to the judge's accuracy and course** (P and R4).
- **The matched-accuracy question N17 left is not decided.** S8 describes it.
- **"The training data" changes noise and count together:** head B's targets at every position, against clean targets at 20%.
- **The training data's own role is tested only through `rw-nfree`,** whose estimator may be poor. It is exploratory.
- **`rw-cloop` and `rw-j20` need clean labels.** They are diagnostics, not methods. `rw-nfree` does not need clean labels.
- **Reused seeds and models,** by necessity and declared. Five seeds.
- **One noise rate, one readout width, one scale, 10% of the budget.** The discounts and the unmixing are told the noise.

## Deviation log

*Empty: the preregistration has not been approved.*

## References

Arazo, E., Ortego, D., O'Connor, N. E., & McGuinness, K. (2020). Pseudo-labeling and confirmation bias in deep semi-supervised learning. In *2020 International Joint Conference on Neural Networks (IJCNN)* (pp. 1–8). IEEE. https://doi.org/10.1109/ijcnn48605.2020.9207304

Patrini, G., Rozza, A., Krishna Menon, A., Nock, R., & Qu, L. (2017). Making deep neural networks robust to label noise: A loss correction approach. In *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 2233–2241). IEEE. https://doi.org/10.1109/cvpr.2017.240
