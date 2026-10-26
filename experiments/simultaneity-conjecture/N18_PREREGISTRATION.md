# Preregistration N18: the loop, with the judge's data and weighting held fixed

Date: 2026-10-26    Status: **draft for the author's approval.** Not built, not run. Nothing in this file is a finding.

Type: confirmatory for N18a, the single claim. It compares two new judges trained for this test, `rw-cloop` and `rw-ctwin`, and scales their difference by the gap between N17's peer and a loop-free judge of N17's design, both trained again here. A further new judge, `rw-nfree`, is exploratory, with no claim. Everything else is secondary or exploratory and labelled so.

Parent documents: `N17_NOTE.md` (next conjecture 1; "Limitations"), `N17_PREREGISTRATION.md` (the third head, its checks and the loop-free line), `N16_NOTE.md` (the two-by-two of judge and random targets; `sub20-oracle`), `N14_NOTE.md` (`sub20-rw`, the self-judged weighting of clean targets alone), `N13_NOTE.md` (the discount as the mixture head's learning signal), `N12_NOTE.md` (the standard head on noisy targets).

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
- **What N16 found about weighting clean targets** (exploratory, on head B itself, with no random targets).
  - Weighting the clean targets of the kept 20% by head A's discount (`sub20-oracle`) left head B 0.124 nats further from head A than the same targets at full weight (`sub20`). Head A's discount has no loop. So weighting clean targets by the discount costs something whoever computes it.
  - Weighting them by head B's own discount (N14's `sub20-rw`) cost 0.45 nats more than head A's (N16c). Head A differs from head B in more than the loop: it is more accurate and sharper. So that comparison left the attribution open.
  - Adding the random targets improved the self-judged head by 0.40 nats. The random targets may temper the loop.
- **Why N18 compares directly, not at matched accuracy.** An earlier draft compared a self-judging judge on clean targets with loop-free judges of the same accuracy, as N17 compared the peer. A fresh reader's check found that comparison ill-posed for this judge.
  - N14's `sub20-rw` ended 0.72–0.76 nats behind head A on plain cross-entropy (0.805 in the calibration seed). That is less accurate than every loop-free judge N17's real run trained, which reached 0.42–0.44 at most; only the calibration's 1.25% judge (0.826) is as inaccurate.
  - In mixture form, the form its discount is computed from, it ended 0.042–0.053 behind its copy, ahead of `sub20`'s 0.112–0.123.
  - If the new judge sits likewise, the two measures place it at opposite ends of the loop-free line, and "a loop-free judge of the same accuracy" names two different judges.

  N18 therefore compares judges trained on the same targets at the same positions, and reports the matched comparisons as description (S8).
- **Why a twin.** A second fresh reader's check found that a direct comparison with `rw-j20`, the judge trained at full weight, would change two things at once: weighting the clean targets by the discount, which N16 found costly without a loop, and taking the weight from the judge itself. N18 separates them with a loop-free twin.
- **What stands regardless.** Every earlier verdict stands. N18 asks whether the loop, with the data and the weighting fixed, is enough to produce a gap the size of the peer's. It does not re-decide N17.

## The conjecture

**The loop alone is enough to produce most of the peer's gap. Given the same clean targets at the same positions, weighted by the same discount formula, a judge whose weights come from its own prediction costs head B at least 0.67 of the peer's gap to `rw-j20` more than a judge whose weights come from a loop-free twin.**

- **The mechanism.** A judge that weights its own targets by its own prediction gives little weight to the targets it predicts poorly, and so goes on predicting them poorly, whatever data it is given. This would be analogous to the confirmation bias described in self-training (Arazo et al., 2020). A judge weighted by a twin that trains at full weight has the same weighting but no such loop: its weights do not depend on what it has itself learned.
- **What it forbids.** R below 0.33: the judge's own loop costing head B a third of the peer's gap or less, beyond what the twin's weighting costs.
- **What a support would not show.**
  - It would show that the loop is enough to produce a gap of that size, with the data and weighting fixed. It would not show that the peer's own gap is the loop's. With random targets present the loop may act differently (rival I).
  - It would not say whether the loop costs through the judge's plain accuracy or beyond it (P).
  - It would not divide the matched-accuracy gap d that N17 described.

**The rivals it is built to exclude, and readings it does not separate.**
- *The weighting (W), from N16's `sub20-oracle`.* Weighting clean targets by the discount costs, whoever computes the weights. Then `rw-cloop` costs head B about what `rw-ctwin` does: R near 0, refuted. The secondary share W measures this weighting directly.
- *The training data (D, N17's R5).* What makes the peer fall short is its noisy targets. Then the loop costs head B little on clean targets: R near 0, refuted.
- *The mixture form (M).* A judge's usefulness follows how well it predicts in mixture form, the quantity its weights are computed from.
  - On that measure, N16's head weighted by head A's discount (`sub20-oracle`, 0.014–0.023 from its copy) was ahead of `sub20-rw` (0.042–0.053), and both were ahead of `sub20` (0.112–0.123).
  - If the judges sit likewise, M predicts R small (near or below a third) and W below 0.
- *Both, in part (B).* R between a third and two thirds.
- *The random targets temper the loop (I), from N16's two-by-two.* The loop costs more on clean targets alone than with random targets present. I predicts R above 1, a support.
- *Plain accuracy (P), not separated.*
  - P says a judge's usefulness follows its plain accuracy, whoever trains it.
  - Suppose `rw-cloop`'s judge is as inaccurate as `sub20-rw` (x of 0.74 in the real seeds, 0.805 in the calibration seed), and `rw-ctwin`'s as `sub20-oracle` (about 0.29).
  - Then N17's calibration line puts R near 1.6–1.9, a support, and W near 0.3.
  - P and the conjecture predict the same verdict, and N18 does not separate them. N17 already found the peer below the line at its plain accuracy, against P for the peer.
- *The course of training (R4), not separated.*
  - A self-judging judge's weights start low, since its prediction starts poor, and the low weights slow its own learning. The twin, at full weight, learns faster, and so does the judge it weights.
  - N17 found the peer slow to start, partly by the loop's own doing. The course is part of what the loop does here. S3 reports it.
- *Sharpness (R3).* Judges can differ in how far apart they put clean and random targets. S2 reports it over training.

## The judges

| Head C's targets | Full weight | Weighted by the discount of a loop-free twin | Weighted by its own discount: the loop |
|---|---|---|---|
| **Clean targets at K** | **`rw-j20`** | **`rw-ctwin`** (new) | **`rw-cloop`** (new) |
| **Head B's targets, 80% random, at every position** | `rw-nfree` (new, exploratory; its weight for head B unmixed) | — | **`rw-peer`** (N17's, trained again here) |

- **The claim's comparison, `rw-cloop` against `rw-ctwin`.** The judges train on the same clean targets at the same positions K. Each target is weighted by the same discount formula, with the same normalisation: summed over K and divided by all positions. Head B's weight is computed from the judge's prediction in the same way in both. The judges start from the same initial values. The one difference is where the weight on the judge's own loss comes from: its own prediction, or the twin's.
- **The twin** is a further readout of the trunk, trained at full weight on the same clean targets, exactly as `rw-j20`'s judge is. It judges nothing, and its weights do not depend on the judge it weights.
- **The scale, the peer's gap to `rw-j20`,** is the difference between the noisy row's loop and the clean row's full weight. In N17's calibration seed, 0.374 of its 0.662 in F lay along the loop-free line between the two judges' plain accuracies, and 0.288 was N17's d. So R counts the loop's whole cost, including through the judge's accuracy.
- **The noisy row's comparison** (`rw-nfree` against `rw-peer`) also changes the estimator, since `rw-nfree` must unmix its prediction, and is exploratory.
- **"The training data"** here changes the noise and the count together: head B's targets at every position, against clean targets at 20% of them.

## Competing hypotheses

| Hypothesis | R (N18a) | W (secondary) |
|---|---|---|
| **The conjecture:** the loop is enough | Above 0.67 | Any |
| I: the random targets temper the loop | Above 1 | Any |
| P: plain accuracy (not separated) | Near 1.6–1.9, if the judges are as inaccurate as N16's heads | Near 0.3 |
| B: both, in part | Between 0.33 and 0.67 | Any |
| W: the weighting | Near 0 | Large |
| D: the training data | Near 0 | Any |
| M: the mixture form | Small, near or below 0.33 | Below 0 |
| R4: the course of training | As the conjecture | — |

## The arms

Every model is N12's: the trunk with head A and head B, each reading the trunk's final state through its own 64-direction readout, at the 80% noise rate. The training, batches and windows are also N12's. The arms use **N12's seeds 4101–4105**, paired with the saved models of the same seed. Seeds and models are reused, as declared in N13–N17. No earlier verdict is re-decided.

Each new arm adds the judge as a further 64-direction readout of the trunk's state, read detached, exactly as N17's head C. `rw-ctwin` adds the twin as well. Every judge starts from N17's initial values for head C. The twin has its own initial values, from a second generator. Both extra heads are tied one way to the main group as one group:
- their losses are scaled by the main scale;
- their gradients are unscaled by the same reciprocal and clipped by the main group's coefficient;
- they step only when the main group steps;
- an overflow in either skips both, and leaves the main group's step alone.

Head B's loss is N13's: the mean over all positions of w(y) · CE, with w from the judge.

| Arm | The judge's targets | The judge's loss | Head B's weight w(y) | Trained for N18? |
|---|---|---|---|---|
| `rw-cloop` | Clean targets at K, the kept 20% | Each target's cross-entropy weighted by the judge's own discount on it, without gradient through the weight, summed over K and divided by all positions: N14's convention for `sub20-rw` | 0.2 p_J / (0.2 p_J + 0.8 u), from the judge's prediction p_J, as in N17 | Yes |
| `rw-ctwin` | Clean targets at K | The same, weighted by the twin's discount on each target instead: N16's convention for `sub20-oracle`, with the twin in head A's place | As in `rw-cloop` | Yes; the twin trains at full weight on K, as `rw-j20`'s judge |
| `rw-j20` | Clean targets at K: N17's rung at 20%, whose nested draw keeps every position of K | Full weight, summed over K and divided by all positions, as in N17 | As in `rw-cloop` | Yes, on the real seeds, as N17's real run did not train it, and again on seed 3001 |
| `rw-peer` | Head B's targets at every position | Each weighted by the judge's own discount, the mean over positions, as in N17 | As in `rw-cloop` | Yes, again, so that all four arms of R train in one session; N17's saved peer is re-measured as a reference |
| `rw-nfree` (exploratory) | Head B's targets at every position | Plain cross-entropy at full weight, the mean over positions: N12's `n80` loss | max(1 − 0.8 u / q_J, 0), the unmixed weight (below) | Yes |
| `rw-j10`, `rw-j05` | As in N17 | As in N17 | As in N17 | No: N17's saved models, for S8 |
| `rw-b1`, `rw-oracle`, `sub20`, `clean`, `sub10` … `sub0125` | — | — | — | No: N13's, N16's, N12's and N14's saved models |

The weight on a judge's own loss in `rw-cloop`, and the twin's in `rw-ctwin`, is computed on the clean target at each position of K, by N15's weight function, which N17 uses for head B's weight. It is a different quantity from head B's weight, which the judge computes on head B's targets at every position.

**The unmixed weight** (`rw-nfree`).
- A head trained at full weight on head B's targets learns their distribution, q ≈ 0.2 p + 0.8 u, with p the clean prediction.
- Under that noise model, the probability that a target y is clean is 1 − 0.8 u(y) / q(y). Where q(y) is below 0.8 u(y), the weight is set to 0.
- This inverts the known noise model, the idea behind one of the loss corrections of Patrini et al. (2017).

N12 found that a single softmax trained on noisy targets fits their distribution poorly. Its clean-target loss ended 0.48–0.50 nats above the copy's mixture-form loss, and placing it in the mixture form doubled that gap (N13b). So `rw-nfree`'s weights may be poor, with many clean targets weighted 0. A shortfall from it could not tell the training data from the estimator. That is why it carries no claim.

`rw-cloop`, `rw-ctwin` and `rw-j20` train on clean targets chosen with the true mask, as N17's rungs did. They are diagnostics, not methods. `rw-nfree` needs no clean labels.

## The measures

- **Δ_A, F, G and a,** as in N17. Δ_A is head B's clean-target loss minus head A's.
- **R = (Δ_A(`rw-cloop`) − Δ_A(`rw-ctwin`)) / (Δ_A(`rw-peer`) − Δ_A(`rw-j20`)),** within seed: the loop's cost to head B on clean targets, as a share of the peer's gap to `rw-j20`. G cancels from it.
- **W = (Δ_A(`rw-ctwin`) − Δ_A(`rw-j20`)) / (Δ_A(`rw-peer`) − Δ_A(`rw-j20`))** (secondary): the weighting's cost without the loop, as a share of the same gap. The rest, 1 − R − W = (Δ_A(`rw-peer`) − Δ_A(`rw-cloop`)) / (the same gap), is what changing the data does with the loop present.
- **For description only (S8): two accuracy axes and the loop-free line.** Each is measured at the end of training, on the measurement windows' clean targets, against the same model's head A.
  - **x, N17's:** the judge's plain cross-entropy minus head A's. It is not defined for `rw-nfree`, which has no clean prediction.
  - **x_m, the mixture axis:** the mean of −log m(y) minus the mean of −log(0.2 p_A(y) + 0.8 u(y)), where m is the judge's mixture-form prediction.
    - For a judge with a clean prediction p, m = 0.2 p + 0.8 u.
    - For `rw-nfree`, m = max(q, 0.8 u), normalised to sum to 1. Unnormalised, it sums to more than 1 wherever a weight is floored, and its loss would be biased in its favour.
    - x_m is the objective a self-judging judge is trained on, so it favours such judges. It is driven by targets with weight near 1 and nearly blind to targets with weight near 0.
  - **The loop-free line, and d:** within seed, the rungs' points (axis value, F), sorted and joined by straight lines, with nothing outside them. d(J) = F(J) − the line at judge J's axis value, where the line reaches.

## The claim

**N18a: the loop is enough to produce most of the peer's gap.** The interval for R lies wholly above 0.67.

## Refuting result

**N18a** is refuted if the interval for R lies wholly below 0.33. The judge's own loop then costs head B a third of the peer's gap or less, beyond what the twin's weighting costs. The report says which reading the interval fits:
- if it contains 0, the loop's cost cannot be told from none, as W and D predict;
- if it lies wholly between 0 and 0.33, the loop costs a little;
- if it lies wholly below 0, the loop helped.

## Design

- **Runs:** 25 training runs: `rw-cloop`, `rw-ctwin`, `rw-j20`, `rw-peer` and `rw-nfree`, on seeds 4101–4105.
- **Measurements:** those models, and re-measurements of the saved models, each checked against its committed record:
  - N17's `rw-peer`, `rw-j10` and `rw-j05`, with their head C;
  - N13's `n80-rw`, N16's `rw-oracle`, and N12's `sub20` and `clean`;
  - N14's `sub10` … `sub0125`, for S6.
- **Saved models** are loaded from the `n17_loop`, `n13_signal`, `n16_judge`, `n12_bottleneck` and `n14_dose` Drive folders. The calibration's come from `n17_calib`, `n12_calib`, `n13_calib`, `n14_calib` and `n15_calib`.
- **Code:** `n18_loop.py`, built on `n17_loop.py`'s head C, forward pass, one-way tie, setup checks and judge measures. It adds:
  - `rw-cloop`, `rw-ctwin` (with the twin) and `rw-nfree`;
  - the one-way tie extended to two extra heads;
  - the mixture axis, and `rw-nfree`'s diagnostics;
  - the new checks.

  `n18_analyze.py` applies the rule.

## Decision rule

**Preconditions:**
1. **Setup.** The loss and gradient checks below are run in float32, as in training. "Relative difference" means the largest absolute difference divided by the largest absolute value.
   - **N17's checks.** Batch, same-model, identical-heads, analytic, the weight identity, the forward identity, the copied-peer check, the overflow test, the windows, and no training draw in the held-out tail. Also the passenger check, run with two passenger heads: `rw-b1` trained with a judge and a twin that judge nothing must be bit-identical to `rw-b1` trained without them.
   - **The overflow test, extended.** An infinite gradient placed in the twin must skip both extra heads' steps and not the main group's.
   - **Detachment,** run for `rw-cloop`, `rw-ctwin` and `rw-nfree`. The trunk receives no gradient from either extra head, and the twin receives none from the judge's loss.
   - **Head B's weight source,** for each new spec, with the extra heads at their own initial values. The weight in head B's loss must equal, bit for bit, the weight computed from the judge's logits by that spec's formula. It must differ from head B's own discount, and in `rw-ctwin` from the twin's.
   - **The judge's own weight source,** with the extra heads at their own initial values.
     - In `rw-cloop`, the weight on the judge's own loss must equal, bit for bit, the discount from the judge's own logits on the clean targets at K, and must differ from head B's.
     - In `rw-ctwin`, it must equal the discount from the twin's logits, and must differ from the judge's own.
   - **The loss and gradient forms.** Each check substitutes other logits into the term, as listed. The term is taken before the factor of 0.5 both losses carry. The tolerance is a relative difference of at most 10⁻⁵, for both the term and its gradient with respect to the logits. N13's discount and N15's weight, which N17 uses, differed by a relative 4.4–4.9 × 10⁻⁷ in N17's checks. N14's discount is computed in log space, and the loss's relative error is larger where weights are small. A weight left with its gradient attached would fail the gradient check, though not the check on the loss.

     | Term | Logits substituted | Must match, on the same batch |
     |---|---|---|
     | `rw-cloop`'s judge term | Head B's | N14's `sub20-rw` loss |
     | `rw-ctwin`'s judge term | Head B's for the judge, head A's for the twin | N16's `sub20-oracle` loss |
     | `rw-nfree`'s judge term | Head B's | N12's `n80` head B loss |
   - **The unmixing identity and floor**, in float64, against a reference written separately from the training code.
     - Take a constructed judge whose q is exactly 0.2 p_A + 0.8 u. Its unmixed weight must equal the discount from p_A, to a relative difference of 10⁻⁶ (N15's convention).
     - Take a constructed q that lies below 0.8 u at some targets. The weight must be exactly 0 there.
   - **Positions.** Each new run's fingerprints and noise fraction at step 0 must equal N12's `n80`'s. The judge's positions in `rw-cloop`, `rw-ctwin` and `rw-j20`, and the twin's, must be exactly K.
   - **Code identity.**
     - *At build time,* N18's code and N17's code must train `rw-j20` and `rw-peer` to bit-identical weights in the tiny setup.
     - *In both runs,* N18's `rw-j20` (calibration) and `rw-peer` (both) are compared with N17's saved models of the same seed. If the GPU and the PyTorch version match N17's records, they must be bit-identical. If not, the differences are reported, and code identity rests on the build-time test. No looser tolerance would be meaningful. In N17's calibration, `rw-b1` was retrained in N17's code on the same GPU and PyTorch version as N13's saved model. It differed from that model by 0.002 in Δ_A and 0.02 in agreement. N17's code computes the weight by N15's function, which differs from N13's arithmetic by less than 5 × 10⁻⁷.
   - **The exploratory arm.** A failed check that concerns only `rw-nfree` (its detachment, its weight source, its loss form, the unmixing) makes `rw-nfree`'s results unreportable. It does not make N18a untestable.
2. **Must match.** Re-measuring the saved models reproduces their records: losses within 0.001 and ratios within 0.01. The constructed mixture check passes, and the zero-noise copy scores at least 0.999, in every model. x_m of the saved judges is measured afresh and has no record to match.
3. **Must differ.** The rotated control scores at most 0.60 in at least 4 of 5 models of each of `rw-cloop`, `rw-ctwin`, `rw-j20` and `rw-peer`. This drops no seed; failing it makes the claim untestable.
4. **The runs trained,** in every seed.
   - `rw-j20`'s and `rw-peer`'s head B Δ_A and judge x are each at most 1.0, as N17's rungs and peer had to be. So is the twin's x.
   - The judges of `rw-cloop` and `rw-ctwin` carry no criterion on how well they predict, since a poor judge is a possible effect of the loop or of the weighting. N14's `sub20-rw` reached 0.72–0.76 on plain cross-entropy.
   - Every logged loss of the four arms is finite.
5. **The peer's gap to `rw-j20` exists.** Δ_A(`rw-peer`) − Δ_A(`rw-j20`) is at least 0.05 nats in every seed. That keeps R's denominator well away from 0.

Preconditions 4 and 5 hold in every seed, not in 4 of 5. No seed is then dropped on a condition involving an arm that enters R.

**The outcome:**

| Outcome | Condition |
|---|---|
| Supported | R's interval lies wholly above 0.67 |
| Refuted | It lies wholly below 0.33 |
| Some effect, size unclear | Neither, and the interval excludes 0 (here, lies wholly above 0) |
| Inconclusive | Everything else |
| Untestable | A precondition fails |

R above 1, the loop on clean targets costing more than the peer's whole gap, counts as support. The report reads it against I.

**Interval:** a two-sided 95% t-interval over the five seeds, with 4 degrees of freedom. There is one claim, so no correction applies.

**Why these thresholds.**
- **0.67 and 0.33,** two thirds and a third of the peer's gap. Two thirds is "most of it" and a third is "little of it". Between them, the loop and something else both contribute.
- **No cap on the standard deviation.** Earlier claims capped it for refutation. Here a cap would bite only where the interval's mean is below about 0.02, in the regions W, D and M predict, and so would favour the conjecture.
- **The denominator's floor, 0.05 nats,** is about half the denominator's expected size.
  - In N17's calibration seed, Δ_A(`rw-peer`) − Δ_A(`rw-j20`) was 0.123.
  - In N17's real run, Δ_A(`rw-peer`) was 0.323–0.338, and Δ_A(`rw-j10`) was 0.252–0.265. `rw-j20` trains on twice `rw-j10`'s targets, so it should lie lower still, and the denominator should be about 0.10.
- **What the design can detect.**
  - In N17's real run, within-seed differences of head B's Δ_A between arms varied across seeds with a standard deviation of 0.004–0.008 nats.
  - If R's numerator varies that much, R's standard deviation is about 0.05–0.1. If it varies as `sub20-rw`'s own accuracy did, by 0.02, it is about 0.2.
  - With five seeds, the 95% half-width is then 0.06–0.25.
  - So an interval clears a margin when R lies above about 0.73–0.92 or below about 0.08–0.27.

## Secondary and exploratory outcomes

- **S1: each judge's own recovery,** F, with its Δ_A, x and x_m. W, with its 95% interval.
- **S2: each judge's sharpness and random share,** at the steps 300, 500, 1000, 2000 and 2850. They are compared directly between `rw-cloop`, `rw-ctwin` and `rw-j20`, and between `rw-nfree` and `rw-peer`.
- **S3: each judge's accuracy over training,** from the batch records. For every arm, the judge's plain cross-entropy minus head A's. For the new arms, the mixture-form loss as well, and the twin's plain loss in `rw-ctwin`.
- **S4: the blind spots,** as in N17: the clean targets each judge ignores (weight below 0.01), measured both ways against head B's own discount, with their set sizes.
- **S5: head A's cost** in each new arm against `clean`.
- **S6: geometry** against the clean-target ladder, as in N17.
- **S7: `rw-nfree`, the loop-free judge on noisy data** (exploratory, no rule).
  - Its F and x_m.
  - The fraction of clean targets its weight floors at 0.
  - Its mixture-form gap to the ideal copy.
  - The other contrasts of the table, each of which also changes the estimator: Δ_A(`rw-nfree`) − Δ_A(`rw-peer`), the loop with noisy data; and Δ_A(`rw-nfree`) − Δ_A(`rw-j20`), the data without the loop.
- **S8: the matched-accuracy comparisons** (description).
  - Each judge's x and x_m are placed against the loop-free line on each axis, with d where the line reaches it.
  - In the calibration seed, N17's five rungs reach x 0.826. In the real seeds, the line runs through `rw-j20`, `rw-j10` and `rw-j05` only, up to x of about 0.44, so it will not reach `rw-cloop` if that judge is as inaccurate as `sub20-rw`.
  - On x_m, a judge more accurate than `rw-j20` lies beyond the line's top in every seed.
  - So S8 can show P only in the calibration seed, and M probably nowhere.

## Necessity audit, at drafting (`NECESSITY_AUDIT.md`)

| Claim the design relies on | Features and ground | The check, and whether it works at the boundary |
|---|---|---|
| `rw-cloop` and `rw-ctwin` differ only in where the judge's own weight comes from | Definitional: the same positions, targets, formula, normalisation, initial values and head B weight | The code; the positions check; both weight-source checks, with the extra heads at their own initial values so that a weight taken from the wrong head differs; the loss and gradient forms |
| The twin has no loop | Definitional: it trains at full weight, and the judge's loss sends it no gradient | The detachment check, which includes the twin |
| The twin is a working judge | A present mechanism: it trains as `rw-j20`'s judge does | Precondition 4 (its x at most 1.0); S3 reports its course |
| N18's code trains as N17's did | A present mechanism | The build-time bit-identity test; the comparison with N17's saved models, bit-identical where the environment matches |
| The contrast measures the loop | Reply 3: what does it hold fixed? | The data, the positions and the weighting formula. It includes what the loop does to the judge's accuracy (P) and course (R4). It is read as "the loop, with what it does", not as the loop beyond accuracy |
| The result transfers to the peer | Reply 3 | No: the loop may act differently with random targets present (I; N16's two-by-two). A support says the loop is enough, not that the peer's gap is the loop's |
| The ratio is well defined | Reply 8 at the boundary: a small denominator makes it unstable | Precondition 5, in every seed. The denominator was 0.123 in N17's calibration seed |
| `rw-nfree` tests the training data | Reply 3 | No: it also changes the estimator, which N12's records suggest is poor. It is exploratory for that reason |
| Reused models and seeds measure as before | A present mechanism; N17's records | Precondition 2 |
| The test can fail | Reply 8 | It is refuted if the loop costs head B a third of the peer's gap or less beyond the twin's weighting, as W, D and M predict. N16's records make W a live rival: the weighting alone cost head B itself 0.124 nats |

**The critic's strongest objection, two moves deep:** *"The twin is more accurate than the self-judging judge, so the contrast compares weights from a better predictor with weights from a worse one, not a loop with no loop."*
- *The reply:* the twin trains on the same targets at the same positions, from its own initial values, at full weight. Its accuracy is what a judge on those targets reaches without the loop. The self-judging judge's poorer accuracy is the loop's own doing, so it is part of what the contrast should count.
- *The critic's answer:* then the claim cannot say whether the loop costs head B through the judge's accuracy or through something else.
- *The next move:* accepted, and stated (P, not separated). A support is read as "the loop, with the accuracy it costs", and S8 describes where the judges sit on both axes.

## Calibration before queuing, including a full-length run

- **A full-length calibration run on the A100,** as its own Colab job, on seed 3001 and the calibration windows 513–1024.
  - **It trains:** `rw-cloop`, `rw-ctwin`, `rw-nfree`, `rw-j20` and `rw-peer`. The last two are compared with N17's saved calibration models for code identity, and the four claim arms give seed 3001's R from one session.
  - **It re-measures:**
    - N17's calibration models: `rw-peer` and the five rungs, for S8 on both axes;
    - N13's `n80-rw`, N12's `sub20` and `clean`, N15's calibration `rw-oracle`, and N14's `sub10`, `sub05` and `sub025` of seed 3001.
  - **It reports:**
    - R and W for seed 3001;
    - every judge's Δ_A, F, x and x_m, and the twin's x;
    - d on each axis where the line reaches;
    - the setup checks and the code identity;
    - `rw-nfree`'s diagnostics, and the timings.

  The values will be seen and recorded in the deviation log, with what they imply for the claim, before the real run.
- **What the calibration may change:** nothing in the design. The real run is not queued, and the values go to the author, if any of these happens:
  - a setup check fails, other than one concerning only `rw-nfree`;
  - the code identity fails where the environment matches;
  - precondition 4 or 5 fails in seed 3001.
- **Also:**
  - the untrained checks;
  - planted bugs that the checks must catch:
    - a weight with its gradient attached in `rw-cloop`;
    - the judge's weight taken from head B's logits, or in `rw-ctwin` from the judge's own;
    - head B weighted by the twin;
    - the twin receiving gradient from the judge's loss;
    - a judge or twin on every position;
    - `rw-nfree`'s weight left without its floor;
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
- **Cost:** 25 training runs at about 2.5 minutes each, with the measurements and model loads. N17's real run took 45 minutes for 15 runs, so this one should take about 75–90 minutes. The calibration job takes about 30 minutes.
- **Stopping rule:** exactly these runs and measurements, each once. Nothing is added after seeing results except as a dated, exploratory entry in the deviation log.

## What each outcome would teach

| N18a | What we would learn | Next |
|---|---|---|
| Supported | With the data and the weighting fixed, a judge's own loop is enough to cost head B most of the peer's gap. The noisy data is not needed for a gap that size. An interval wholly above 1 would fit the random targets tempering the loop (I). Whether the loop costs through the judge's plain accuracy or beyond it stays open (P; S8). | Break the loop without a clean head: a floor on the discount |
| Refuted, interval containing 0 | With the data and weighting fixed, the loop's cost to head B cannot be told from none. The peer's gap would then come from its data, or from the weighting; W says how much of it the weighting carries. | If W is large: weight differently, not judge differently. If not: separate the data from the estimator, with a better noisy-trained judge than `rw-nfree`'s single softmax |
| Refuted, interval between 0 and 0.33 | The loop costs a little, and most of the peer's gap lies elsewhere. | As above |
| Refuted, interval below 0 | The loop helped on clean targets: the self-weighted judge was the better judge. That would fit M. | Test M directly: which accuracy predicts a judge's usefulness |
| Some effect | The loop accounts for part of the gap, between a third and two thirds, or by an amount the design cannot place. | More seeds |
| Inconclusive | The design's limits, as stated. | More seeds |
| Untestable | The peer's gap to `rw-j20` vanished, a run failed to train, or a check failed. | Inspect the failed precondition |

## Limitations known now

- **The loop is changed on clean targets only.** With random targets present it may act differently (I; N16's two-by-two). A support shows the loop is enough, not that the peer's gap is the loop's.
- **The contrast includes what the loop does to the judge's accuracy and course** (P and R4).
- **The matched-accuracy question N17 left is not decided.** S8 describes it, and in the real seeds may not reach the new judges.
- **The twin has its own initial values,** so its early weights differ from what the judge's would have been. That difference is part of having no loop, and cannot be removed.
- **"The training data" changes noise and count together:** head B's targets at every position, against clean targets at 20%.
- **The training data's own role is tested only through `rw-nfree`,** whose estimator may be poor. It is exploratory.
- **`rw-cloop`, `rw-ctwin` and `rw-j20` need clean labels.** They are diagnostics, not methods. `rw-nfree` does not need clean labels.
- **Reused seeds and models,** by necessity and declared. Five seeds.
- **One noise rate, one readout width, one scale, 10% of the budget.** The discounts and the unmixing are told the noise.

## Deviation log

*Empty: the preregistration has not been approved.*

## References

Arazo, E., Ortego, D., O'Connor, N. E., & McGuinness, K. (2020). Pseudo-labeling and confirmation bias in deep semi-supervised learning. In *2020 International Joint Conference on Neural Networks (IJCNN)* (pp. 1–8). IEEE. https://doi.org/10.1109/ijcnn48605.2020.9207304

Patrini, G., Rozza, A., Krishna Menon, A., Nock, R., & Qu, L. (2017). Making deep neural networks robust to label noise: A loss correction approach. In *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 2233–2241). IEEE. https://doi.org/10.1109/cvpr.2017.240
