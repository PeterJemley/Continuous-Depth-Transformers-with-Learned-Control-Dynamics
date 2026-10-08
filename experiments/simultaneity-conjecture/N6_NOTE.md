# Result note: N6

Date: 2026-10-08    Preregistration: `N6_PREREGISTRATION.md`    Generated report: `N6_RESULTS.md`    Raw results: `results/n6/` and `results/n6_log.txt`, from `colab-results` branch `n6_halfoverlap/20261008-102605`. Code commit 70b476b.

**How the run went.** One Colab session ran all 20 models and their measurements, with no resumes, no skipped fp16 steps and no Drive failures. Nothing was changed after approval.

## What we expected

Head A always predicts the next token (i + 1); head B predicts token i + k, for k = 1 (the must-fail control), 2, 4 or 8.

- **N6a (primary).** The overlap of the two readings' sensitivity subspaces falls as the readings differ more: o(2) − o(8) at least 0.10.
- **N6b.** At k = 2, the overlap replicates N5's half-overlap: between 0.40 and 0.60 in at least 4 of 5 models.

## What happened

**Both verdicts are untestable: the must-fail precondition failed.**

The rule required the same arm, two heads reading the same target, to reach an overlap of at least 0.95 in at least 4 of 5 models. It did so in 3:

| Seed | Same-arm overlap | Lower split-half stability of its two heads |
|---|---|---|
| 3601 | 0.9498 | 0.916 |
| 3602 | 0.9571 | 0.953 |
| 3603 | 0.9624 | 0.931 |
| 3604 | 0.9396 | 0.899 |
| 3605 | 0.9633 | 0.931 |

Seed 3601 missed by 0.0002. The other preconditions held: the data and setup checks passed, all four arms of each seed saw the same batches, and every head in arms k2, k4 and k8 had a split-half stability of at least 0.89, above the 0.80 threshold.

Under the preregistered rules, neither N6a nor N6b is decided, and nothing below changes that.

## Which kind of failure this is

**A control set too tight for the measurement, and the calibration should have caught it.**

- **Why the same arm fell short.** With 139 windows, the subspaces are estimated with some noise, and noise caps how close even two identical readings can measure. The two seeds that missed are the two whose heads had the least stable subspaces (0.90–0.92).
- **The same-arm overlap, divided by the geometric mean of its two heads' stabilities, averaged 1.02.** In other words, the same-target heads agree as well as the measurement can show. So the control did its job in substance, while missing its number.
- **Why the calibration missed it.** The threshold of 0.95 came from N4, which measured on 256 windows. The calibration checked that each head's stability was attainable with 139 windows, but not what fewer windows would do to the must-fail ceiling. N4's own records already held the warning: overlaps from one half of N4's windows (128 of them, close to N6's 139) went as low as 0.944 for same-target heads.
- **What would be a rescue.** Lowering the threshold now, or counting 0.9498 as 0.95, would be choosing the rule after seeing the result. The verdicts stay untestable.

## Exploratory observations, not confirmatory

These had no decision rule once the precondition failed. They are recorded because they bear directly on the conjecture, and they set the next test.

- **The overlap fell steeply, and in order, as the readings differed more.**

  | k | Mean overlap (5 seeds) | 95% interval | Seeds |
  |---|---|---|---|
  | 1 (same target) | 0.954 | [0.942, 0.967] | |
  | 2 | 0.481 | [0.439, 0.522] | 0.43–0.53 |
  | 4 | 0.305 | [0.285, 0.325] | 0.29–0.33 |
  | 8 | 0.254 | [0.245, 0.263] | 0.24–0.26 |

  - o(2) − o(8) = +0.227, 95% interval [+0.184, +0.269]. That is more than twice the 0.10 the claim required, with every seed between +0.17 and +0.27.
  - Every seed was in the predicted order, o(2) > o(4) > o(8): 5 of 5.
  - At k = 2, the half-overlap reappeared on new seeds and new windows: 0.43–0.53, all within N6b's band.
- **Training moved the readings apart in proportion to how different they were.** All four arms started at an overlap of about 0.48. Training raised the same arm to 0.95, left k = 2 where it started (+0.003), and lowered k = 4 by 0.17 and k = 8 by 0.22. This fits N5's observation that the end point does not depend on the start.
- **The fall is not an estimation artefact.** The farther heads' subspaces were as stable as the nearer ones' (0.93–0.95 on average), and dividing each overlap by its heads' stabilities leaves the same ladder: 1.02, 0.52, 0.33, 0.27.
- **The confound with difficulty remains.** Head B's loss rose with k: 6.23, 6.92 and 7.11 nats, against 5.14 for the next token. So "more different" and "harder to predict" moved together, as the preregistration warned. They do not move in step, though. From k = 4 to k = 8 the loss rose by only 0.19, while the overlap still fell by 0.05.
- **Even the most different reading shares a quarter.** At k = 8 the overlap, 0.25, is still four times what two random 16-dimensional subspaces would share (0.06). Something common, perhaps word frequency or the passage's general content, is read by both.

## What this teaches

- **The conjecture survives a test it could have failed, but not as a confirmation.** The exploratory pattern is what the conjecture predicted, and is large and consistent. A preregistered verdict still needs a new test, with its must-fail control set for the measurement it uses.
- **A methodological lesson for this programme.** A threshold carried from one test to another must be re-checked under the new measurement's conditions: its window count, here, just as N5 showed for its run-to-run spread. Calibration should check what each control can reach, not only what each measure can.

## What this does not change

- All earlier results stand. N5's verdicts stay inconclusive; N6 does not re-score them.
- Nothing here decides N6a or N6b.

## Limitations

- **The same limits as the earlier tests:** 10% of the budget, one learning rate, one size, one dataset, one readout width (64).
- **Distance confounds "different" with "harder".** It is unresolved here.
- **Only 139 windows.** That caused the failed precondition, and it will constrain any further test on the validation split, which has no unused windows left.

## Next

**N6r: the same design, with the control set for its measurement.** On new seeds, with these changes, each fixed before any run:
- **Fresh data.** The overlap is measured on 256 windows of the WikiText-103 *test* split. No overlap or geometry measurement has used that split, and no model was trained or tuned on it. Using it also doubles the windows.
- **The must-fail control on the measurement's own scale.** Two same-target heads count as converged if their overlap, divided by the geometric mean of their split-half stabilities, is at least 0.95. That is the condition that matters: they agree as far as the measurement can show. The raw threshold would be calibrated beforehand, on N4's and N6's existing same-target models, at the new window count.
- **Everything else unchanged:** the arms, the 0.10 threshold for N6a, the 0.40–0.60 band for N6b, and the stability precondition.

## References

Lin, H. W., & Tegmark, M. (2017). Critical behavior in physics and probabilistic formal languages. *Entropy, 19*(7), 299. https://doi.org/10.3390/e19070299
