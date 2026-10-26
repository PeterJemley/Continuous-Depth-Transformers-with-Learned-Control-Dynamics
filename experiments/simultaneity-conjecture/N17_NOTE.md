# Result note: N17

Date: 2026-10-26    Preregistration: `N17_PREREGISTRATION.md`    Generated report: `N17_RESULTS.md`    Calibration run: `N17_CALIBRATION.md`

## In brief

N16 found that a head trained on 80%-scrambled next-word answers, each weighted by its own estimate that the answer is genuine, does as well as a head given only the genuine answers when the clean head, head A, makes that estimate instead. But as a judge, head A differs from the scrambled-answer head in at least three ways: it is not caught in a loop of judging by its own predictions, it predicts better, and its judgement is sharper. N17 asked which of the first two carried the result, using a third head as the judge.

- **N17a (primary): some effect, size unclear.** A judge with no loop, interpolated to the scrambled-answer head's own accuracy, recovered a third of the shortfall: 0.334, 97.5% interval [0.318, 0.349]. That is neither the three quarters the conjecture required nor the quarter or less that would have refuted it.
- **N17b: untestable.** Precondition 6b failed in every seed, so the rule could not compare the peer with a loop-free judge.
- **The peer, as description.** A second head that judges itself, as accurate as the scrambled-answer head, recovered almost none of the shortfall (0.04).

So the judge's accuracy at the end of training does not account for N16's result on its own. At equal accuracy, a judge trained at full weight on clean targets recovers a third more of the shortfall than a judge trained the way the scrambled-answer head is. The design does not separate that third into the loop itself, what the judge is trained on, how fast it improves, or how sharp it is during training. How the rest of N16's result divides is description only, since head A's point lies outside the loop-free judges' range.

## The arms and measures

Every model has a shared network, the trunk, and heads that read the next word from the trunk's final state through their own 64-direction readouts.
- **Head A**, the clean head, is trained on the genuine next word.
- **Head B** is trained on targets 80% of which were replaced by words drawn at random in proportion to their frequency, u. The note calls those random targets, and the genuine ones clean targets.
- **Head B's loss** weights each target by a judge's estimate that it is genuine: w(y) = 0.2 p(y) / (0.2 p(y) + 0.8 u(y)), with p the judge's prediction. This weighting is the discount.
- **Head C**, new in N17, is a third head that reads the trunk's state without training the trunk. It is the judge in every new arm.

All arms use N12's five seeds, 4101 to 4105, paired with the saved models of the same seed. A seed is one training run from its own random start. Everything is measured on N12's 512 windows held out from the training text.

| Arm | Who judges head B | How the judge is trained | Does the judge have a loop? | Source |
|---|---|---|---|---|
| `rw-b1` | Head B itself | As head B | Yes: its weights come from its own prediction | N13's `n80-rw`, saved |
| `rw-peer` | Head C, the peer | On head B's targets, weighted by its own discount, as head B is | Yes | New |
| `rw-j10`, `rw-j05` | Head C, a loop-free judge | At full weight on the clean targets at a nested 10% or 5% of positions (N14's positions) | No | New |
| `rw-oracle` | Head A | On every clean target | No | N16's, saved |
| `sub20` | None: random targets dropped | — | — | N12's, saved: the shortfall's reference |
| `clean`, `sub10` … `sub0125` | — | — | — | N12's and N14's, saved: head A's cost and the clean-target ladder |

The loop-free judges are told which targets are clean, as head A is. Like N16's oracle, they are diagnostics, not methods.

| Term | Meaning |
|---|---|
| Nats | The unit of prediction loss |
| Δ_A | A head's loss on the clean targets minus head A's, in nats. Lower is closer to the clean head |
| x, the judge's accuracy | The judge's Δ_A at the end of training, on the measurement windows: a loss gap, so higher means less accurate. x_self = Δ_A(`rw-b1`), head B judging itself; x_peer is the peer's; x = 0 for head A |
| The rungs | The loop-free judges, `rw-j10` and `rw-j05`, named after the rungs of N14's clean-target ladder whose positions they train on |
| G, the shortfall | Δ_A(`rw-b1`) − Δ_A(`sub20`), within a seed |
| F, the recovered fraction | (Δ_A(`rw-b1`) − Δ_A(head B in the arm)) / G. 0 is head B judging itself; 1 is the whole shortfall recovered |
| The loop-free line | Within a seed, the points (x, F) of the loop-free judges joined by a straight line. F_lf(x) is its value at accuracy x, between the rungs only |
| F* | F_lf(x_self): what a loop-free judge exactly as accurate as head B would recover |
| d | F(`rw-peer`) − F_lf(x_peer): how far the peer falls short of a loop-free judge of its own accuracy |
| Sharpness | The mean weight on clean targets over the mean on random ones, from the training batch at the last logged step (2850) |
| Share, s | The random targets' share of head B's summed weight, from the same batch |
| Ignored, blind spot | A clean target is ignored when its weight is below 0.01. A judge's blind spot is the set of clean targets it ignores. Head B's own discount is the weight head B's own prediction would give, measured even where head C is the judge |
| The clean-target ladder, â, b | N14's yardstick: heads trained on fewer and fewer clean targets and no random ones. â is the ladder's agreement at a head's Δ_A, and b = â − a; positive means the head reads head A's directions slightly worse than its accuracy predicts |
| a, the agreement | How far head B's predictions depend on the same directions of the trunk's state as head A's, corrected for noise; 1 for the same directions |
| R1–R5 | The preregistration's rivals: R1 the judge's accuracy; R2 the judge being head B itself; R3 sharpness; R4 the judge's course over training; R5 what the judge is trained on |
| S1–S7 | The preregistration's secondary measures, reported in `N17_RESULTS.md`; S1 is the peer's own recovery |
| The oracle | Head A as the judge, N16's `rw-oracle`. N15's share mechanism was the conjecture that the shortfall is the random weight the discount leaves behind; N16 refuted it |
| The necessity audit | `NECESSITY_AUDIT.md`: a check made before every run from N12 on. It lists each claim in the design of the form "this requires that", so that the cheapest criticism comes before the run |
| Setup checks | Tests run before training. The ones new in N17: head C does not train the trunk; head B's weights come from head C; adding head C changes nothing else (the passenger check); a copy of head B as head C stays identical to it (the copied-peer check); an overflow in head C skips only its own step |

**The rules.** Each claim uses a Bonferroni-corrected 97.5% interval over the five seeds: two rules, so the level is 1 − 0.05 / 2.
- **N17a** is *supported* if F*'s interval lies wholly above 0.75, and *refuted* if wholly below 0.25 with a standard deviation of at most 0.15.
- **N17b** is *supported* if d's interval lies wholly below −0.5, and *refuted* if wholly above −0.25 with a standard deviation of at most 0.15. It is testable only if, in at least 4 of 5 seeds, the loop-free line reaches the peer's accuracy and is at least 0.75 there (precondition 6b). Below that, the conjecture and R2 predict nearly the same d.
- Otherwise a verdict is *some effect, size unclear* if the interval excludes 0, and *inconclusive* if not.

The exploratory intervals are 95% and decide nothing.

- **Changes after approval.** None changed a claim, margin, measure or rule.
  - Seven readings were fixed at build time, among them a forward pass that also returns the trunk's state, with a check that it matches the original bit for bit.
  - Five fixes followed an independent review of the code, made before any calibration value was seen. One made the analysis pool the seeds as the preregistration says.
  - The calibration's one choice, the rungs, fell to the default pair, 10% and 5%.
  - Nothing changed after the calibration.

## What we expected

**The question.** In N16, head A as judge recovered 108% of the shortfall, G = 0.17 nats. Head A is more accurate than head B, and it is not judging by its own predictions. Which mattered?

**The conjecture.** The shortfall is a self-confirming loop in the judge. A judge that weights its own targets by its own discount gives little weight to the clean targets it predicts poorly, so it goes on predicting them poorly, and so does the head it judges. Without the loop, a judge no more accurate than head B would recover most of the shortfall (N17a). With the loop, a judge that is not head B would do far worse than a loop-free judge of its accuracy (N17b). This would be analogous to the confirmation bias described in self-training (Arazo et al., 2020).

**What the rivals predicted.** R1, the judge's accuracy, predicted N17a refuted (F* near 0) and d near 0. R2, the judge being head B itself, predicted N17a supported and d near 0. The preregistration said in advance that N17 could not separate the conjecture from R3, R4 or R5.

**What the calibration suggested.** On one seed and other windows, F* was 0.295 and the loop-free line at the peer's accuracy 0.26. The calibration entry, written before the real run, said N17a would most likely land between its margins and N17b would most likely be untestable.

## What happened

| Claim | Rule | Result (5 seeds) | Verdict |
|---|---|---|---|
| N17a | F* wholly above 0.75 | 0.334 [0.318, 0.349], standard deviation 0.010; 0.325–0.350 by seed | **Some effect, size unclear** |
| N17b | d wholly below −0.5 | Precondition 6b failed: F_lf(x_peer) 0.31–0.37 in every seed, against 0.75 | **Untestable** |

All other preconditions held. Every setup check passed on the A100 with 16-bit training on, and the passenger and copied-peer checks were bit-identical. The 40 re-measured saved models reproduced their records exactly. Every new run used N12's batches and N14's positions. G was 0.165–0.179, and the rungs bracketed head B's accuracy in every seed.

**By arm** (means over seeds):

| Arm | Judge's accuracy, x | Head B's Δ_A | F | Sharpness | Share, s | Agreement, a |
|---|---|---|---|---|---|---|
| `rw-oracle` (head A judges) | 0 | 0.150 | 1.081 | 6.73 | 0.378 | 0.892 |
| `rw-j10` (loop-free, 10%) | 0.288 | 0.259 | 0.449 | 5.78 | 0.415 | 0.831 |
| `rw-b1` (head B judges itself) | 0.337 | 0.337 | 0 | 5.64 | 0.422 | 0.804 |
| `rw-peer` (a second self-judging head) | 0.339 | 0.330 | 0.041 | 5.66 | 0.420 | 0.797 |
| `rw-j05` (loop-free, 5%) | 0.429 | 0.316 | 0.122 | 5.38 | 0.432 | 0.792 |
| `sub20` (random targets dropped) | — | 0.163 | 1 | — | — | 0.896 |

## Which kind of result this is

- **N17a's verdict is "some effect, size unclear", with a narrow interval.** It is ±0.016 wide and sits well clear of both margins. The conjecture, that a loop-free judge as accurate as head B recovers most of the shortfall, was not supported. R1's prediction, F* near 0, failed. A loop-free judge as accurate as head B recovers about a third.
- **N17b is untestable by its precondition.** With the peer's F near or above 0 (−0.007 to +0.071 by seed), d was unlikely to fall far below −F_lf(x_peer). With the loop-free line at 0.33 there, the support margin of −0.5 was out of reach whatever the mechanism. The height condition in precondition 6b was added after a fresh reader's check of the draft to make that case untestable. For the record, without it d's 97.5% interval would have been [−0.322, −0.258], "some effect, size unclear".
  - The margins were set in units of the shortfall, on the conjecture's own expectation that the loop-free line would be high at head B's accuracy. A margin relative to the line's height would have stayed reachable. That is a lesson about the design, recorded here, not a reason to read d against a different margin now.
- **The calibration's warning held.** Both verdicts are what the calibration entry, committed before the real run, said they would most likely be.

## What the results say together

All of this is exploratory: it decides nothing, and N17b's quantities are reported as description.

**The loop-free line.** Between the rungs, F falls as the judge gets less accurate: 0.45 at x = 0.29 and 0.12 at x = 0.43, so 0.33 at head B's own accuracy. Head A's point (x = 0, F = 1.08) lies outside that range. Head A is also a different kind of judge: it trains on every clean target, and it trains the trunk, which head C does not. So the 0.75 between head A's point and the line at head B's accuracy mixes accuracy with sharpness, the course of training and the kind of judge (S7, description only). Even between the rungs, accuracy is not separated from sharpness or course: the 10% judge is sharper than the 5% judge (5.78 against 5.38) and more accurate at every step of training.

**A second self-judging head does nearly as badly as head B judging itself.** The peer was as accurate as head B (x 0.339 against 0.337) and recovered 0.041 of the shortfall (95% interval [0.003, 0.080]). Head B's Δ_A with the peer as judge was 0.007 nats lower than with its own judgement. That is far from what R2 predicted for the peer's own recovery, about what its accuracy gives (S1, secondary). Against a loop-free judge of the peer's accuracy, d = −0.290 [−0.315, −0.265], between −0.32 and −0.27 in every seed; d is description, since N17b was untestable. This is consistent with the shortfall not needing head B to judge itself. But the peer shares head B's trunk, targets and training, and may converge towards head B, so it does not show that.

**At equal accuracy, sharpness and share match at the end of training only.** Interpolated to head B's accuracy, the loop-free judges' sharpness at the last logged step is 5.64 [5.31, 5.97] and their random share 0.421 [0.413, 0.429]; the self-judge's are 5.64 and 0.422. Earlier in training they were sharper and left less random weight: sharpness +0.16 [+0.06, +0.25] and share −0.016 [−0.029, −0.003] at step 500, sharpness +0.18 [+0.09, +0.27] at step 1000. So sharpness during training (R3, with R4) is not excluded as part of the third.

**The blind spots differ less than they first appear.** The peer ignored more clean targets than the loop-free judges (1.04% against 0.77% and 0.87%), so it covers more of head B's own blind spot: 0.83 of it, against 0.79 and 0.81. Against the 5% judge that difference is within noise (+0.02 [−0.02, +0.06]). Measured the other way, as the share of the judge's own blind spot that head B's discount also ignores, the peer and head A rank together (0.76 and 0.74, against 0.72 and 0.66). These do not show the judges differing in which clean targets they ignore.

**The self-judging peer was slow to start.** One step-100 reading, on a training batch, puts the peer's loss 1.33 nats above head A's, against 0.57 and 0.75 for the loop-free judges. From step 200 on, the peer lay between the two rungs (0.36–0.48, against 0.26–0.31 and 0.38–0.47). This is the course difference R4 names. A self-judging head's slow start is partly the loop's own doing, at the start, when its discount weights clean and random targets alike, so the course cannot be separated from the loop here.

**Geometry stayed close to the ladder.** Against the clean-target ladder, head B's agreement with head A sat, on average, within 0.04 of what its accuracy predicts in every arm, though slightly below it. The residuals were +0.010 (`rw-b1`), +0.011 (`rw-oracle`), +0.021 (`rw-peer`), +0.018 (`rw-j10`) and +0.031 (`rw-j05`). The peer's and the 5% judge's 95% intervals exclude 0; the 10% judge's does not.

**Head A's cost** was within noise in every arm: from −0.006 to +0.009 nats against `clean`, every 95% interval including 0.

## What this teaches

- **Matching one difference can leave a clear intermediate answer.** N16 left three differences between its judges unseparated. N17 matched one of them, accuracy at the end of training, and a loop-free judge so matched recovered a third of the shortfall: not zero (F*'s interval lies well above 0), and not the "most" the conjecture claimed. How the rest divides, the design does not say.
- **Margins in the units of a quantity the conjecture expects to be large can go out of reach when it is not.** N17b's margins assumed a high loop-free line. A fresh reader's check of the draft caught the consequence before approval, and the height condition added to precondition 6b made the case untestable rather than leaving the support margin out of reach. Next time, a margin on a difference should be set relative to the quantity it is a difference from.
- **A calibration that predicts its own verdicts is worth stating before the run.** The calibration entry said N17a would most likely land between its margins and N17b would most likely be untestable; both happened. Recording that before the run showed that the verdicts were foreseen.
- **A match at one time is not a match.** At equal accuracy, the judges' sharpness and share matched at the end of training and differed during it. The audit's warning about conditions fixed to one time applies to the secondary measures too.

## What this does not change

- **Every earlier verdict stands.** N16's three supported claims stand by their rules. N17 adds, by its primary rule, that a loop-free judge as accurate as head B recovers a third of N16's shortfall.
- **N16c's label,** "self-reference is the mechanism", was already marked as saying more than its comparison separated. N17's peer, a different self-judging head, recovered almost none of the shortfall (secondary), consistent with self-reference as such not being what costs.

## Limitations

- **Accuracy is matched only at the end of training,** and the self-judging peer was slower early (R4).
- **The loop-free judges train on clean targets only,** chosen with the true mask, while the self-judges train on head B's noisy targets (R5). The third that is not accuracy could come from the loop, from that difference in training, or from the course. N17 does not separate them.
- **N17b was untestable,** so the comparison of the peer with a loop-free judge is description only.
- **The loop-free line is straight between two rungs,** so F* is an interpolation.
- **The loop-free judges and the oracle need clean labels.** Only the peer is a method, and it helped little (0.04; S1).
- **Five seeds, reused with the saved models,** by necessity and declared.
- **One noise rate, one readout width, one scale, 10% of the budget.** The discounts are told the noise.

## Next conjectures

1. **Separate the loop from what the judge is trained on.** A loop-free judge trained on head B's noisy targets would have no loop and head B's training data: for example a standard head on those targets, with u's share subtracted from its prediction to estimate the clean one. If it recovers the third, that is consistent with the loop; if it does not, with the noisy training. Course and sharpness would still not be separated.
2. **Break the loop without a clean head:** a floor on the self-judged discount, so that no clean target falls below a minimum weight (N16's next conjecture 2). That would be a method, aimed at the third that is not accuracy.
3. **Improve the judge's accuracy without clean labels.** Along the loop-free judges, F rose steeply with the judge's accuracy, so a judge more accurate than head B that needs no clean head would be worth testing, for example one averaged over the head's own training.
4. **Return to N10's question with mixture heads** (carried from N12–N16).

## References

Arazo, E., Ortego, D., O'Connor, N. E., & McGuinness, K. (2020). Pseudo-labeling and confirmation bias in deep semi-supervised learning. In *2020 International Joint Conference on Neural Networks (IJCNN)* (pp. 1–8). IEEE. https://doi.org/10.1109/ijcnn48605.2020.9207304

## Records

- **Raw results:** `results/n17/` and `results/n17_log.txt`, from Colab run `n17_loop/20261026-000615`. The calibration's are `results/n17_calib/` and `results/n17_calib_log.txt`, from `n17_calib/20261025-233741`.
- **Code commit:** e2d3de8 for the real run, 4e5015a for the calibration.
- **Analysis:** `n17_analyze.py` writes `N17_RESULTS.md` and `N17_CALIBRATION.md` from those records, with N12's, N13's, N14's and N16's (or N15's calibration) records. The job is `n17_loop.py`.

**How the run went.**
- **One session:** an NVIDIA A100 with 80 GB and PyTorch 2.11. It trained the 15 models, re-measured the 40 saved ones and measured everything in 45 minutes.
- **No problems:** no resumes, no Drive failures, no skipped 16-bit steps, and no steps skipped by head C alone.
- **The re-measurements** reproduced their records exactly, for all 40 saved models of N12, N13, N14 and N16.
