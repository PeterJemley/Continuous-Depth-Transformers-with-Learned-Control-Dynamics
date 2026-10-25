# Result note: G1

Date: 2026-10-05    Preregistration: `G1_PREREGISTRATION.md`, with amendment A1    Generated report: `G1_RESULTS.md`

## In brief

G1 asked whether Model A, trained with exact gradients and with its control signal u choosing between two readings (the next word, or the word after next), sends the paths through its continuous block along regions of clearly different curvature. The preregistered verdict, under amendment A1, is refuted: paths under one control value turned only 1.08 to 1.27 times as sharply as under the other, where at least 2 was required in 4 of 5 models. Under the rule as first approved, which used Cohen's d, the verdict would have been supported, so the amendment, made before any real run, decided it. The small difference was consistent across models and absent when u carried no information, and an exploratory measure found a larger one in how strongly the flow contracts the state, 4.4 to 5.7 times.

## The models and the measures

G1 trains Model A with exact gradients (ordinary backpropagation, not the adjoint shortcut), at 10% of the WikiText-103 training budget, 2,879 steps. Its control u is a signal of four numbers, of which only the first is used. u is added inside the vector field of the continuous block (the ODE block). The vector field is the learned function that says how the state changes with depth. Nothing else in the model sees u. For each sequence, u says which task, or reading, the model must perform. Strictly the tasks predict tokens, which are words or pieces of words.

| u | Task at every position |
|---|---|
| +1 | Predict the next token: the next-token task |
| −1 | Predict the token after next, a harder task: 6.13 against 5.05 nats |
| 0 | Never used in training; measured only as S7 |

| Models | What u carries |
|---|---|
| Main, seeds 3001–3005 | The true task, so a model must use u to do both tasks well |
| Shuffled-control, the same seeds and data | Nothing: u is drawn at random, independently of the task. These are the must-fail control. |
| Untrained, seeds 3001–3005 at initialisation | Measured before any training, with a random control embedding; used only to set amendment A1's thresholds |

- **Regime:** the paper's word for a region of the flow with its own kind of motion. G1's claim was that the two control values produce distinct regimes.
- **Trajectory:** the path a token's state takes through the block as depth τ runs from 0 (where the block starts) to 1 (where it ends). Trajectories are traced with dropout off and an accurate solver (dopri5), on 256 validation windows of 128 tokens, at 17 evenly spaced depths.
- **Curvature, κ:** how sharply a trajectory turns, whatever its speed.
- **Curvature ratio, the primary measure under amendment A1:** for each window, the mean log curvature at u = +1 minus that at u = −1; averaged over windows, this is the log ratio. A positive log ratio means trajectories curve more at u = +1. The ratio is the larger curvature over the smaller, so a ratio of 2 means trajectories under one control value turn twice as sharply as under the other.
- **Cohen's d (S0c), the primary measure as first approved:** the difference in mean curvature between the two control values, divided by how much windows differ from one another. **The paired d_z (S0b)** divides the same difference by the spread of each window's own difference instead.
- **Jacobian:** how a small change in the state changes its velocity under the vector field. Its spectrum is the set of its eigenvalues. Its trace is the **divergence**: positive when nearby states spread apart, negative when they are drawn together, which is contraction.
- **Secondary measures:** S1 says which control value curves more. S2 is curvature by depth. S3 is speed, the mean size of a trajectory's velocity. S4 is divergence per token, estimated with 8 random probe vectors (Hutchinson's estimator) at 5 depths. S6 is a classifier that predicts u from a window's 17-depth curvature profile; it is trained on half the windows and tested on the rest, and chance is 50%. S7 is curvature at u = 0.
- **α:** a learned scale on the block's update, which starts at 0.1.
- **Wrong-control cost:** how much worse the next-token loss is with u = −1 than with u = +1.
- **Other codes:** A1 is the amendment to the decision rule. H6 is the WikiText-2 replication's test of the paper's claim of distinct regimes, made through solver effort; it refuted the claim for the paper's model. T1 is the earlier test that supplied the learning rate. G2 (regimes as voices: strands of a text carried in distinct regimes) and G3 (a control signal per voice, entering the field by multiplication) are untested conjectures in `CONJECTURE.md`. TV1 is a planned measure of how well a model tracks several strands of a text.

**The decision rule, under amendment A1.** Four preconditions must hold, or the result is *untestable*:
- the wrong control costs at least 0.5 nats on the next-token task in at least 4 of 5 main models;
- the curvature code returns the curvature of known curves to within 0.1%;
- must fail: the shuffled-control models give ratios below 1.25 in at least 4 of 5;
- must match: in every main model, the log ratios from the odd-numbered and the even-numbered windows differ by less than 0.1.

The claim is *supported* if the ratio is at least 2 in at least 4 of 5 main models, *refuted* if it is below 1.25 in at least 4 of 5, and *intermediate* otherwise. There is no interval: the unit is the trained model. The thresholds are judgements, set from the untrained models. As first approved, the rule used |d| instead: at least 0.8 in 4 of 5 for *supported*, below 0.2 for *refuted*.

*The summary, this key and the records section at the end were added on 2026-10-25 so that the note can be read on its own. Nothing else was changed.*

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

## Records

- **Raw results:** `results/g1/` and `results/g1_log.txt`, from the `colab-results` branch, run `g1_regimes/20261005-204242`.
- **Run:** code commit 9374217. One session, no resumes, no Drive failures.
- **Analysis:** `g1_analyze.py` writes `G1_RESULTS.md` from those records; the job is `g1_regimes.py`.
