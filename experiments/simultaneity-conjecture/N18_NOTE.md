# Result note: N18

Date: 2026-10-28    Preregistration: `N18_PREREGISTRATION.md`    Generated report: `N18_RESULTS.md`    Calibration run: `N18_CALIBRATION.md`

## In brief

N17 found that a judge trained the way the scrambled-answer head is, judging itself, falls short of a loop-free judge of the same accuracy. But the two kinds of judge differed in two ways at once: the loop, and what they were trained on. N18 asked whether the loop alone is enough. It held the data fixed at clean targets, and held the weighting formula fixed too, using a loop-free twin to supply the weights.

- **N18a: supported.** With the same clean targets, positions and weighting formula, a judge whose weights came from its own prediction cost head B about twice the peer's whole gap. That is R = 2.07, 95% interval [1.85, 2.29], beyond what the same weighting cost when a loop-free twin supplied the weights. The interval lies wholly above 1.
- **The weighting without the loop also cost a good deal** (secondary): W = 0.86 [0.77, 0.94] of the peer's gap.
- **Adding the random targets helped the self-weighting judge** (description): it recovered nearly twice the peer's gap.
- **The loop's cost went beyond its judge's plain accuracy** (secondary): at its own accuracy, the self-weighting judge fell short of the loop-free line by 0.46 of the shortfall.

So the loop is enough to produce a gap of the peer's size, and more, when the judge trains on clean targets alone. The run does not show that the peer's own gap is the loop's. With random targets present, as in the peer, a self-weighting judge does far better, and N18 does not divide that between the loop and the random targets. Nor does it separate the loop from the slow start it causes, or from what the weights do to the weighter's own training.

## The arms and measures

Every model has a shared network, the trunk, and heads that read the next word from the trunk's final state through their own 64-direction readouts.
- **Head A**, the clean head, is trained on the genuine next word.
- **Head B** is trained on targets 80% of which were replaced by words drawn at random in proportion to their frequency, u. The note calls those random targets, and the genuine ones clean targets. The 20% of positions that keep the genuine word are K.
- **Head B's loss** weights each target by a judge's estimate that it is genuine: w(y) = 0.2 p(y) / (0.2 p(y) + 0.8 u(y)), with p the judge's prediction. This weighting is the discount.
- **Head C**, the judge, reads the trunk's state without training the trunk, as in N17.
- **The twin**, new in N18 and used in one arm only, is a further readout of the same kind. It trains at full weight on the clean targets at K, exactly as `rw-j20`'s judge does, and it judges nothing.

All arms use N12's five seeds, 4101 to 4105, paired with the saved models of the same seed. A seed is one training run from its own random start. Everything is measured on N12's 512 windows held out from the training text.

| Arm | What head C trains on | Each target of head C's own loss weighted by | The loop? | Source |
|---|---|---|---|---|
| `rw-cloop` | The clean targets at K | Head C's own discount | Yes | New |
| `rw-ctwin` | The clean targets at K | The twin's discount | No | New |
| `rw-j20` | The clean targets at K | 1 (full weight) | No | N17's design, trained here on the real seeds |
| `rw-peer` | Head B's targets at every position | Head C's own discount | Yes | N17's, trained again here |
| `rw-j025`, `rw-j0125` | The clean targets at a nested 2.5% or 1.25% of positions | 1 | No | New, for S8 |
| `rw-nfree` (exploratory) | Head B's targets at every position | 1; head B's weight is head C's prediction unmixed, max(1 − 0.8 u / q, 0) | No | New |

N17's `rw-j10` and `rw-j05`, N13's `n80-rw` (`rw-b1`, head B judging itself), N16's `rw-oracle`, N12's `sub20` and `clean`, and N14's clean-target ladder were re-measured from their saved models.

`rw-cloop` and `rw-ctwin` start from the same initial values, judge and twin alike, so they are identical at step 0. They differ only in where the weight on head C's own loss comes from. The judges trained on clean targets are told which targets are clean, as head A is. Like N16's oracle, they are diagnostics, not methods.

| Term | Meaning |
|---|---|
| Nats | The unit of prediction loss |
| Δ_A | Head B's loss on the clean targets minus head A's, in nats. Lower is closer to the clean head |
| The peer's gap | Δ_A(`rw-peer`) − Δ_A(`rw-j20`), within a seed: 0.105–0.112 nats |
| R | (Δ_A(`rw-cloop`) − Δ_A(`rw-ctwin`)) / the peer's gap: the loop's cost to head B on clean targets, as a share of the peer's gap |
| W | (Δ_A(`rw-ctwin`) − Δ_A(`rw-j20`)) / the peer's gap: the weighting's cost without the loop |
| 1 − R − W | (Δ_A(`rw-peer`) − Δ_A(`rw-cloop`)) / the peer's gap: what moving from `rw-cloop`'s training to the peer's does with the loop present. The peer's judge trains on `rw-cloop`'s clean targets, with the same weights and scaling, plus the random targets at the other 80% of positions |
| G, the shortfall; F | As in N17: G = Δ_A(`rw-b1`) − Δ_A(`sub20`), 0.165–0.179 nats; F = (Δ_A(`rw-b1`) − Δ_A(arm)) / G, the share of the shortfall recovered |
| x, the judge's plain accuracy | The judge's cross-entropy on the clean targets minus head A's. Higher means less accurate. Not defined for `rw-nfree` |
| x_m, the mixture accuracy | The judge's loss in mixture form, −log(0.2 p + 0.8 u), minus head A's copy's. It favours judges trained by their own discount |
| The loop-free line, d | Within a seed, the points (x, F) of the judges trained at full weight on clean targets, joined by straight lines. d is a judge's F minus the line at its x |
| Sharpness, share | As in N17: the mean weight on clean targets over the mean on random ones, and the random targets' share of head B's summed weight, from the training batch |
| Ignored | A clean target is ignored when its weight is below 0.01 |
| P, M, W, D, I, R3, R4 | The preregistration's rivals and unseparated readings: P, a judge's usefulness follows its plain accuracy; M, it follows its mixture accuracy; W, the weighting costs whoever computes it; D, the training data; I, the random targets temper the loop; R3, sharpness; R4, the course of training |
| S1–S8 | The preregistration's secondary measures, in `N18_RESULTS.md`. S8 tests P directly |

**The rule.** N18a is *supported* if R's two-sided 95% interval over the five seeds lies wholly above 0.67, and *refuted* if wholly below 0.33. Otherwise it is *some effect, size unclear* if the interval excludes 0, and *inconclusive* if not. There is one claim, so no correction applies.

- **Changes after approval.** None changed the claim, its margins, its measures or its rule.
  - Five readings were fixed at build time:
    - each extra head has its own optimiser and skips only its own step on overflow;
    - with one extra head, the step performs N17's operations in N17's order;
    - N14's module is imported for one loss with its side effects undone;
    - `rw-nfree` trains last in each seed, and a failure in it is recorded rather than fatal;
    - a missing comparison with N17's saved models fails the code-identity check.
  - Six fixes followed an independent review of the code, made before any calibration value was seen.
  - In the tiny tests, N18's peer differed from N17's earlier tiny peer by about 10⁻⁵. N17's own code, run on the same machine, gave N18's weights bit for bit, so the cause was the change of host. On the A100, N18's `rw-j20` and `rw-peer` were bit-identical to N17's saved models.
  - Nothing changed after the calibration.

## What we expected

**The question.** N17's peer fell short of a loop-free judge of its accuracy by 0.29 of the shortfall. The peer has the loop and trains on head B's noisy targets; the loop-free judges have neither. Is the loop enough to produce a gap of that size?

**Why the comparison is direct.** An earlier draft compared a self-weighting judge on clean targets with loop-free judges of the same accuracy. A fresh reader's check found that ill-posed.
- N14's head trained this way was very inaccurate on plain cross-entropy but very accurate in mixture form, so "a loop-free judge of the same accuracy" named two different judges.
- A second check found that a comparison with `rw-j20` alone would change two things: weighting the clean targets by the discount, which N16 found costly even with head A computing it, and taking the weight from the judge itself.
- The twin keeps the first and removes the second.

**The conjecture.** A judge that weights its own targets by its own prediction gives little weight to the targets it predicts poorly, and so goes on predicting them poorly. This would be analogous to the confirmation bias described in self-training (Arazo et al., 2020). With the data and weighting fixed, the loop alone would cost head B at least two thirds of the peer's gap.

**What the rivals predicted.**
- W and D predicted R near 0. N16c had already refuted W on head B itself, with head A as the weighter, so a refutation by it would have been surprising.
- M predicted R small but above 0, about 0.1–0.5, and W below 0.
- P predicted about 1.4–1.9 for R and 0.3–0.5 for W, if the judges were as inaccurate as N14's and N16's heads. P and the conjecture predict the same verdict.
- I predicted R above 1.

**What the calibration suggested.** On one seed and other windows, R was 2.10 and W 0.94. The calibration entry, written before the real run, said that if the real seeds sat likewise, R's interval would lie wholly above 0.67 and above 1.

## What happened

| Claim | Rule | Result (5 seeds) | Verdict |
|---|---|---|---|
| N18a | R wholly above 0.67 | 2.07 [1.85, 2.29], standard deviation 0.18; 1.93–2.36 by seed | **Supported** |

All five preconditions held, and every setup check passed on the A100 with 16-bit training on:
- the weight sources;
- the loss and gradient forms against N14's, N16's and N12's losses;
- the passenger check with two passenger heads;
- the per-head overflow test;
- the positions.

N18's peer was bit-identical to N17's saved peer in every seed. The 55 re-measured saved models reproduced their records exactly. The peer's gap was 0.105–0.112 nats, against a floor of 0.05. The twin was exactly as accurate as `rw-j20`'s judge, as it should be (x 0.175 in both).

**By seed** (Δ_A of head B, nats):

| Seed | `rw-cloop` | `rw-ctwin` | `rw-j20` | `rw-peer` | R | W | 1 − R − W |
|---|---|---|---|---|---|---|---|
| 4101 | 0.534 | 0.310 | 0.213 | 0.325 | 2.00 | 0.86 | −1.87 |
| 4102 | 0.514 | 0.308 | 0.217 | 0.323 | 1.94 | 0.85 | −1.79 |
| 4103 | 0.540 | 0.309 | 0.227 | 0.336 | 2.11 | 0.75 | −1.86 |
| 4104 | 0.559 | 0.313 | 0.224 | 0.328 | 2.36 | 0.85 | −2.21 |
| 4105 | 0.548 | 0.332 | 0.227 | 0.338 | 1.93 | 0.95 | −1.88 |

**By judge** (means over seeds):

| Judge (arm) | Plain accuracy, x | Mixture accuracy, x_m | Head B's Δ_A | F | Sharpness at the end | Clean targets ignored |
|---|---|---|---|---|---|---|
| `rw-j20` (full weight, clean) | 0.175 | 0.126 | 0.221 | 0.667 | 6.19 | 0.7% |
| `rw-ctwin` (twin-weighted, clean) | 0.347 | 0.034 | 0.314 | 0.131 | 8.16 | 3.6% |
| `rw-peer` (self-weighted, noisy) | 0.339 | 0.182 | 0.330 | 0.041 | 5.66 | 1.0% |
| `rw-cloop` (self-weighted, clean) | 0.748 | 0.050 | 0.539 | −1.160 | 9.26 | 8.6% |
| `rw-nfree` (full weight, noisy; exploratory) | — | 0.532 | 0.579 | −1.396 | 2.89 | 7.8% |

## Which kind of result this is

- **N18a is supported, with a wide margin.** The interval's lower end, 1.85, is far above 0.67 and above 1, and R was above 1.9 in every seed. A rival that put R near 0 (W, D) or between the margins (M) is not what happened.
- **The result was foreseen.** The calibration entry, committed before the real run, said R would most likely lie wholly above 1 if the real seeds sat like seed 3001. They did: 2.07 against 2.10.
- **P's rough prediction was close for R but not for W.** P put R at 1.4–1.9 and W at 0.3–0.5 under stated assumptions; R was 2.07 and W 0.86. The verdict does not separate P from the conjecture, which is what S8 is for (below).

## What the results say together

All of this is secondary or exploratory: it decides nothing.

**How the peer's gap divides** (description, within seed). From `rw-j20` to the peer, head B's Δ_A rose by 0.11 nats:
- Weighting the clean targets by a loop-free twin's discount cost 0.08–0.11 nats. That is W, 0.86 of the gap.
- Taking the weight from the judge itself instead cost 0.21–0.25 nats more. That is R, about twice the gap.
- Adding the random targets at the other 80% of positions took 0.19–0.23 nats back, about 1.9 times the gap.

Each step changes one thing, so the decomposition is exact, but each step's size depends on the ones before it. The last step is the judge-level counterpart of N16's finding that the random targets improved a self-judged head by 0.40 nats. It is consistent with I, the random targets tempering the loop. But N16 found that they also helped a head weighted by head A, and N18 has no twin-weighted judge on noisy targets, so it cannot say how much of the last step is the loop being tempered.

**The loop's cost goes beyond plain accuracy** (S8, secondary). The two rungs added for S8 reached the self-weighting judge in all five seeds. At its own plain accuracy:
- `rw-cloop` fell short of the loop-free line by d = −0.46 [−0.50, −0.41], against P's prediction of about 0.
- `rw-ctwin` fell short by −0.18 [−0.22, −0.14], so the weighting also costs more than accuracy predicts.
- The peer's d was −0.29 [−0.32, −0.27], as in N17.

So neither judge's usefulness follows its plain accuracy alone.

**Mixture accuracy points the wrong way** (secondary). The twin-weighted judge was the most accurate in mixture form (x_m 0.034, against 0.126 for `rw-j20`) and was less useful to head B. W was above 0 in every seed, against M's prediction.

**The loop's mark is neglect.** At the end of training:
- on the last logged training batch, the self-weighting judge put a weight below 0.01 on 5–10% of its own clean training targets, by seed, against 0.5–1.1% for the twin-weighted judge;
- on the measurement windows it ignored 8.6% of clean targets, against 0.7% for `rw-j20`.

That is the neglect the conjecture names, and much stronger without random targets than in the peer (1.0%), as N16 found for head B.

**Sharper was worse.** At the end of training the self-weighting judge was the sharpest (9.26), then the twin-weighted judge (8.16), `rw-j20` (6.19) and the peer (5.66). Among the three judges trained on clean targets, the order of usefulness was the reverse of the order of sharpness. Sharpness (R3) does not explain the result.

**The slow start is large** (R4, not separated). On the training batch, the self-weighting judge's plain cross-entropy minus head A's was:

| Step | Self-weighting judge | Twin-weighted judge | `rw-j20` |
|---|---|---|---|
| 100 | 2.23 | 0.98 | 0.42 |
| 300 | 1.24 | 0.40 | 0.20 |
| 2850 | 0.79 | 0.36 | 0.19 |

It was still the worst at the end. The slow start is the loop's own doing here, so the course cannot be separated from the loop.

**`rw-nfree`** (exploratory):
- Its mixture gap to the copy was 0.53 [0.52, 0.55], near the 0.48–0.50 expected from N12.
- Only 7.5% of clean targets were floored at 0, fewer than the "many" expected.
- As a judge it did worse than the peer, by 0.25 nats [0.21, 0.29].
- It was also the only arm with a cost to head A: 0.20 nats [0.16, 0.24] against `clean`. Every other arm's head A was within noise.

These say little about the training data, since `rw-nfree` also changes the estimator, as the preregistration warned.

**Geometry stayed close to the ladder.** Head B's agreement with head A was within 0.03 of what its accuracy predicts in every arm but `rw-nfree`, whose head B read head A's directions 0.07 worse than its accuracy predicts.

## What this teaches

- **A comparison matched on accuracy can be the wrong comparison.** For a judge whose loop makes it inaccurate, matching on accuracy removes part of the effect, and the two summaries of accuracy pointed opposite ways. The direct comparison, with the data and weighting fixed, was well defined and decisive. Two fresh-reader checks found the problem before approval.
- **Fix the confound you can name.** The first direct design would have credited the discount weighting's own cost to the loop. On these numbers that cost was 0.86 of the peer's gap, which by itself would have cleared the support margin. The twin removed it.
- **Test the unseparated rival directly when it is cheap.** Ten more runs extended the loop-free line far enough for S8 to reach the self-weighting judge in every seed. That turned "P is not separated" into a secondary test against P.
- **On CPU, the environment record is not the environment.** A change of host changed low-order bits while the record read "matches". A test of bit-identity needs both codes run on one machine.

## What this does not change

- **Every earlier verdict stands.** N17a stands: a loop-free judge as accurate as head B recovers a third of the shortfall. N18 adds that, with clean data, the loop alone is enough to produce a gap larger than the peer's.
- **The peer's own gap is not attributed.** N17's d, the peer falling short of a loop-free judge of its accuracy, may be the loop's in part. But a self-weighting judge does far better with random targets present, and N18 does not divide d.

## Limitations

- **The loop is changed on clean targets only.** With random targets present, a self-weighting judge does far better (1 − R − W = −1.92), and N18 cannot say how much of that is the loop being tempered. A support shows the loop is enough to produce a gap of the peer's size, not that the peer's gap is the loop's.
- **The contrast includes what the loop does to the judge's accuracy and course, and what the weights do to the weighter's own training** (P, R4). S8 shows a cost beyond plain accuracy. It does not separate self-reference from the weights' effect on the weighter, which a yoked arm would.
- **The weighting's own cost (W) is large,** and N18 does not say why. One candidate is the down-weighting of hard clean targets, which N16 found costly with head A as the weighter.
- **`rw-nfree`'s estimator may be poor,** so it does not test the training data. It is exploratory.
- **The judges trained on clean targets need clean labels.** They are diagnostics, not methods.
- **Five seeds, reused with the saved models,** by necessity and declared.
- **One noise rate, one readout width, one scale, 10% of the budget.** The discounts are told the noise.

## Next conjectures

1. **Separate self-reference from the weights' effect on the weighter.** In a yoked arm, the judge is weighted by a twin that is itself trained by its own discount. If the yoked judge costs head B as much as `rw-cloop`, what matters is the weighter's loop, not who the weighter is.
2. **Do the random targets temper the loop, and how?** A twin-weighted judge on head B's noisy targets would say how much of the peer's advantage over `rw-cloop` is the loop being tempered, and how much is the random targets helping any judge. N16 read them as keeping the softmax from neglecting what the discount down-weights, since they are drawn from u. A test of how: replace them, for a self-weighting judge on clean targets, with a loss term pulling its prediction towards u at the other positions. If that recovers what the random targets did, the u-mass is what tempers.
3. **Break the loop without a clean head:** a floor on the self-judged discount, so that no clean target falls below a minimum weight (carried from N16 and N17). The neglect measures above suggest that is where the loop does its damage.
4. **Why does the weighting cost, without a loop?** W was 0.86 of the peer's gap. A twin-weighted judge with a softer discount (a flatter weight, nearer 1 for hard clean targets) would test whether the down-weighting of hard targets is the cost.
5. **Return to N10's question with mixture heads** (carried from N12–N17).

## References

Arazo, E., Ortego, D., O'Connor, N. E., & McGuinness, K. (2020). Pseudo-labeling and confirmation bias in deep semi-supervised learning. In *2020 International Joint Conference on Neural Networks (IJCNN)* (pp. 1–8). IEEE. https://doi.org/10.1109/ijcnn48605.2020.9207304

Patrini, G., Rozza, A., Krishna Menon, A., Nock, R., & Qu, L. (2017). Making deep neural networks robust to label noise: A loss correction approach. In *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 2233–2241). IEEE. https://doi.org/10.1109/cvpr.2017.240

## Records

- **Raw results:** `results/n18/` and `results/n18_log.txt`, from Colab run `n18_data/20261028-120244`. The calibration's are `results/n18_calib/` and `results/n18_calib_log.txt`, from `n18_calib/20261028-113321`.
- **Code commit:** 115aa15 for the real run, b9b8f0c for the calibration.
- **Analysis:** `n18_analyze.py` writes `N18_RESULTS.md` and `N18_CALIBRATION.md` from those records, with N12's, N13's, N14's and N16's (or N15's calibration) records. The job is `n18_loop.py`.

**How the run went.**
- **One session:** an NVIDIA A100 with 80 GB and PyTorch 2.11. It trained the 35 models, re-measured the 55 saved ones and measured everything in 108 minutes.
- **No problems:** no resumes, no Drive failures, and no skipped 16-bit steps, by the main group, the judge or the twin.
- **The re-measurements** reproduced their records exactly, for all 55 saved models of N12, N13, N14, N16 and N17.
