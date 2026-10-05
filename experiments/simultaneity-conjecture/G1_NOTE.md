# Result note: G1

Date: 2026-10-05    Preregistration: `G1_PREREGISTRATION.md`, with amendment A1    Generated report: `G1_RESULTS.md`    Raw results: `results/g1/` and `results/g1_log.txt`, from `colab-results` branch `g1_regimes/20261005-204242`. Code commit 9374217. One session, no resumes, no Drive failures.

## What we expected

The claim: when Model A is trained with exact gradients and its control u selects between two tasks, the two control values send trajectories through regimes of clearly different curvature. Under amendment A1, that means trajectories under one control value turn at least twice as sharply as under the other, in at least 4 of 5 models.

## What happened

**The preregistered verdict is refuted: no distinct regimes.**

All four preconditions held:
- **The models use u.** Giving the wrong control costs 1.87 to 1.91 nats on the next-token task, in all five main models. The threshold was 0.5.
- **The curvature code is right.** The worst error on the known curves was 0.002%.
- **The must-fail control failed, as required.** The shuffled-control models gave curvature ratios of 1.003 to 1.010.
- **The must-match control matched.** The odd and even halves of the windows agreed to within 0.003 in log ratio.

The curvature ratio between u = +1 and u = −1 in the main models:

| Seed | Ratio | Curves more |
|---|---|---|
| 3001 | 1.09 | u = +1 |
| 3002 | 1.12 | u = +1 |
| 3003 | 1.08 | u = +1 |
| 3004 | 1.27 | u = +1 |
| 3005 | 1.20 | u = +1 |

Four of five are below 1.25 and none reaches 2. Trajectories under the two control values turn 8 to 27% differently. The direction is the same in every model.

## The amendment decided this verdict

Under the rule as first approved, the result would have been **supported**:
- the main models' Cohen's d was 1.0 to 3.0, against a threshold of 0.8, in all five;
- every original precondition also held: the shuffled models gave |d| of 0.07 or less, and the odd-against-even split gave |d| of 0.15 or less.

Amendment A1 replaced d with the curvature ratio before any real run, for a stated reason. In five untrained models, a random control embedding alone gave |d| of 0.9 to 6.0 and ratios of 1.02 to 1.16. The trained ratios here, 1.08 to 1.27, overlap that range, and their d values are no larger than the untrained ones. So d could not tell a trained regime from an untrained nudge, which is the problem A1 was written to fix.

The difference the amendment makes is reported here, not hidden. A reader who prefers the original rule can apply it to the same records: S0c in `G1_RESULTS.md` gives every d value.

## What this teaches

- **What training did produce.** The shuffled models show no curvature difference at all, ratio 1.01 or less. So the main models' difference is caused by training on a control that carries information. It is consistent: the paired d_z is 1.6 to 4.5, and the direction is the same in every model.
- **What it did not produce.** The difference is small. Under one control value, trajectories turn about 1.1 to 1.3 times as sharply as under the other.
- **The conjecture as stated is refuted at this scale.** It said the control partitions the field into regimes of distinctly different curvature. With additive control, exact gradients and two tasks at 10% of the budget, it does not.
- **Prediction from the outcome table:** "Control in this architecture shifts the dynamics without partitioning them, at least by these measures." For curvature, that is what happened.

## What this rules out, and what it does not

- **Ruled out, for this setup:** large curvature differences between control values, as the paper described them, arising from additive control trained in from the start.
- **Not ruled out:**
  - other kinds of geometric difference (see below);
  - multiplicative control (G3);
  - longer training;
  - more than two control values.
- **H6's refutation of the original model stands.** G1 was a new test on a new model, and it does not change that.

## Exploratory observations, not confirmatory

These are secondary measures. None of them had a threshold, and none decides anything. They are recorded because they bear on what to test next.

- **The two control values differ strongly in contraction, not turning** (S4).
  - The divergence of the field, the trace of its Jacobian, is −7.6 to −8.5 per token at u = −1 and −1.4 to −1.8 at u = +1.
  - So the state contracts 4.4 to 5.7 times faster per unit depth under u = −1, in every main model.
  - In the shuffled models the two control values agree to within 3%.
  - The conjecture note's G1 named "curvature and Jacobian spectrum". The preregistration made curvature primary, and divergence is one summary of the Jacobian spectrum.
- **Speed also differs** (S3). Trajectories move 14 to 26% faster at u = −1.
- **The curvature difference sits at the start of the block** (S2). It is a factor of 1.08 to 1.63 at depth 0, fades steadily, and is about 1.02 at depth 1. The control acts mainly as the block begins its work.
- **The curvature profile still identifies the control value** (S6). A classifier using only the 17-depth curvature profile tells u = +1 from u = −1 with 91 to 98% accuracy on held-out windows. In the shuffled models it is at chance, 48 to 52%. Small differences, consistently placed, are enough to classify; distinct regimes need more than that.
- **u = 0, never seen in training, gives a third kind of motion** (S7). Its trajectories curve more than under either trained value, by factors of about 1.3 to 1.7. Its losses on both tasks lie between those of the right and wrong control.
- **α grew to 0.20–0.22 in the main models** and 0.09–0.12 in the shuffled ones. The model that uses its control signal uses its ODE block more.
- **Not knowing the task costs about 0.4 nats.** The shuffled models' next-token loss was 5.46–5.49, against 5.05–5.06 for the main models with the right control.

These observations were found after looking at the results. A conjecture built on them needs new data to test it.

## Limitations

- **Budget and design.** 10% of the training budget, one learning rate taken from T1, one model size, additive control, two control values and two tasks.
- **The two tasks differ in difficulty.** The token after next is harder to predict (6.13 against 5.05 nats). Some of the differences between control values may come from difficulty, not from the kind of computation.
- **The thresholds of 2 and 1.25 are judgements,** fixed before the run from untrained models. A different threshold for "distinct" would read the same numbers differently. The numbers themselves are reported in full.

## Next conjectures

1. **Control separates contraction, not turning.** A trained control value sets how strongly the block contracts the state. The prediction is a divergence ratio of at least 2 between control values in trained models, and below 1.25 in the shuffled models. It needs its own preregistration, with new seeds and windows not used here, because the observation came from looking at these results. If it holds, the regimes picture survives in a different geometric form; if not, the observation was specific to these models.
2. **G3, multiplicative control.** Does control that enters the vector field multiplicatively, per strand, produce large curvature differences where additive control did not? This is the outcome table's planned next step after a refutation.
3. **G2, regimes as voices,** still waits on TV1's tracking measure. This result says the regimes to look for may differ in contraction rather than curvature.
