# Result note: G1c

Date: 2026-10-06    Preregistration: `G1C_PREREGISTRATION.md`    Generated report: `G1C_RESULTS.md`

## In brief

G1c asked whether the control signal u sets how strongly Model A's continuous block draws nearby states together, its contraction, a difference that G1 had noticed only as an exploratory observation. On new seeds and new text windows, the block contracted the state 4.1 to 10.2 times faster under the farther reading (u = −1, the word after next) than under the next-word reading (u = +1), against a threshold of 2 in at least 4 of 5 models. The preregistered verdict is supported, and models whose control carried no information, or that were untrained, showed none of the difference. The note stresses that this shows contraction differs, not that the model needs it, and G1's curvature refutation replicated here as a secondary measure.

## The models and the measures

The setup is G1's. Model A is trained with exact gradients (ordinary backpropagation, not the adjoint shortcut), at 10% of the WikiText-103 training budget. Its control u is added inside the vector field of the continuous block (the ODE block), and nothing else in the model sees it. The vector field is the learned function that says how the state changes with depth. For each sequence, u says which task, or reading, the model must perform. Strictly the tasks predict tokens, which are words or pieces of words.

| u | Task at every position |
|---|---|
| +1 | Predict the next token: the next-token reading |
| −1 | Predict the token after next: the farther-ahead, or farther, reading, also called the skip task |
| 0 | Never used in training; measured only as S4 |

| Models | Role |
|---|---|
| Main, seeds 3101–3105 | u tells the task. These seeds were not used in G1. |
| Shuffled-control, the same seeds | u is drawn at random, independently of the task, so it carries no information. Must-fail control A. |
| Untrained, the main seeds at initialisation | Measured before any training step. Must-fail control B. |

- **Regime:** the paper's word for a region of the flow with its own kind of motion.
- **Measurement:** on validation windows 257 to 512 of 128 tokens, which G1 did not use, with dropout off and an accurate solver (dopri5).
- **Jacobian and divergence:** the Jacobian of the vector field says how a small change in the state changes its velocity. Its trace is the divergence: positive when nearby states spread apart, negative when they are drawn together. It is estimated per token with 8 random probe vectors (Hutchinson's estimator), at depths 0, 0.25, 0.5, 0.75 and 1, where 0 is the start of the block and 1 its end.
- **Contraction rate, c:** minus the divergence, averaged over the five depths: how strongly the flow draws nearby states together. c̄(u) is its mean over the 256 windows, for control value u.
- **Contraction ratio, R, the primary measure:** R = c̄(−1) / c̄(+1). Rates below 0.1 per token are too near zero for a ratio, so the plan scores them by fixed cases. If c̄(+1) is below 0.1 and c̄(−1) is at least 0.2, the model counts as R ≥ 2. If c̄(+1) is below 0.1 and c̄(−1) is from 0.1 to below 0.2, it counts as neither. If c̄(−1) is below 0.1, it counts as R < 1.25.
- **Log ratio:** the logarithm of R, used by the must-match checks.
- **Secondary measures:**
  - S1 is contraction by depth.
  - S2 is the normalised contraction: contraction divided by the Jacobian's overall size, its Frobenius norm (the square root of the sum of its squared entries). Scaling the whole Jacobian leaves it unchanged. A ratio near 1 would mean u sets only the Jacobian's overall strength, its gain; a larger ratio means u changes its shape.
  - S3 evaluates each control value's field at the other value's trajectory states, to tell a difference in the field from a difference in where the trajectories go.
  - S4 places u = 0 between the two trained values on a log scale, 0 at u = +1 and 1 at u = −1.
  - S5 repeats G1's measures: the curvature ratio (how much more sharply trajectories turn under one control value), the curvature-profile classifier (which predicts u from how sharply trajectories turn at each depth; chance is 50%) and speed.
- **α:** a learned scale on the block's update, which starts at 0.1.
- **Wrong-control cost:** how much worse the next-token loss is with u = −1 than with u = +1.
- **Other codes:** G1 is the test that found this pattern and refuted regimes of different curvature. Its amendment A1 set the thresholds reused here. G2 (regimes as voices: strands of a text carried in distinct regimes) is an untested conjecture in `CONJECTURE.md`, and TV1 is a planned measure of how well a model tracks several strands of a text.

**The decision rule.** Six preconditions must hold, or the result is *untestable*:
- the wrong control costs at least 0.5 nats on the next-token task in at least 4 of 5 main models;
- the analytic check: the divergence code is exact to 1e-5 on a diagonal linear field and within 2% on a dense one, and G1's curvature check passes again;
- must fail A: the shuffled-control models give R between 0.8 and 1.25 in at least 4 of 5;
- must fail B: in at least 4 of 5 untrained models, the two contraction rates differ by less than 0.1;
- must match A: in every main model, the log ratios from odd-numbered and even-numbered windows differ by less than 0.1;
- must match B: in every main model, a second set of probe vectors gives a log ratio within 0.05 of the first.

The claim is *supported* if R ≥ 2 in at least 4 of 5 main models. It is *refuted* if R < 1.25 in at least 4 of 5, which includes a difference in the other direction. Anything else is *intermediate*. There is no interval: the unit is the trained model. The thresholds of 2 and 1.25 are reused from G1's amendment A1, as judgements.

*The summary, this key and the records section at the end were added on 2026-10-25 so that the note can be read on its own. Nothing else was changed.*

## What we expected

The claim: when Model A is trained with exact gradients and its control u selects between the next token (u = +1) and the token after next (u = −1), the vector field contracts the state at least twice as fast under u = −1, in at least 4 of 5 models.

The direction and the size were taken from an exploratory observation in G1. They were tested here on new seeds (3101–3105) and new validation windows (257–512), against thresholds fixed before any of these models existed.

## What happened

**The preregistered verdict is supported: the control sets the contraction rate.** All six preconditions held:

- **The models use u.** The wrong control costs 1.88 to 1.92 nats on the next-token task.
- **The code is right.** On the diagonal field the divergence estimate was exact to 1.4e-8. On the dense field it was within 0.19%. The curvature check also passed.
- **Must fail A, trained but uninformative control.** The shuffled-control models gave ratios of 0.997 to 1.031.
- **Must fail B, no training.** The untrained models' two contraction rates differed by at most 0.005, against a limit of 0.1.
- **Must match A, windows.** Odd and even windows agreed to within 0.025 in log ratio, against a limit of 0.1.
- **Must match B, estimator.** A second set of probe vectors agreed to within 0.023 in log ratio, against a limit of 0.05.

The contraction ratio in the main models:

| Seed | Contraction at u = +1 | Contraction at u = −1 | Ratio |
|---|---|---|---|
| 3101 | 1.13 | 8.20 | 7.3 |
| 3102 | 1.91 | 7.82 | 4.1 |
| 3103 | 1.77 | 9.16 | 5.2 |
| 3104 | 1.77 | 8.08 | 4.6 |
| 3105 | 0.80 | 8.16 | 10.2 |

Every model clears the threshold of 2 by at least a factor of 2. G1's models gave 4.4 to 5.7; these give 4.1 to 10.2.

## What this teaches

- **A trained control signal does set how strongly the dynamics contract.** Under the farther-ahead reading, u = −1, the block draws nearby states together four to ten times faster than under the next-token reading. This replicated on new seeds and new windows, and it appears only when training gives the control meaning: shuffled-control and untrained models show none of it.
- **The regimes picture holds, in a different geometric form from the one the paper proposed.** The paper described regimes of different curvature. G1 refuted that for this setup, and S5 below shows the refutation replicating. G1c finds regimes of different contraction instead: how much the flow forgets the differences between its inputs, not how sharply its trajectories turn.
- **The paper's claims are unchanged.** Its refuted claims about the original model stay refuted, and G1 stays refuted. G1c is a new conjecture, prompted by a G1 observation and confirmed on fresh data. It explains G1's pattern: the dynamics differ, but in contraction rather than curvature. It does not save G1's claim.

## Secondary measures

None of these had a threshold. They are reported because they bear on what the result means. G1c's preregistration set out the questions they answer.

- **Shape, not only gain (S2).**
  - If the control only scaled the Jacobian as a whole, the normalised contraction would be the same under both control values.
  - It is 2.9 to 7.0 times larger under u = −1. So the control changes how much of the field's local behaviour is contraction, not just its overall strength.
  - In the shuffled models the normalised ratio is 1.00 to 1.03.
- **The field differs, not only the states (S3).**
  - At u = +1's trajectory states, u = −1's field still contracts at 9.0 to 11.0, as much as or more than at its own states.
  - At u = −1's states, u = +1's field contracts at 0.3 to 1.3, near its own rates. *[Corrected 2026-10-25: lower than its own rates in every model (0.30 to 1.31 against 0.80 to 1.91), and near them only beside u = −1's rates of about 8.]*
  - So the difference belongs to the two fields themselves, not to the different places their trajectories go.
- **Where it is strongest (S1).**
  - The ratio is largest at the start of the block, 8.1 at depth 0, and falls to 4.5 at depth 1.
  - Under u = −1 the contraction rate stays near 8 throughout; under u = +1 it rises from 1.0 to 1.7.
  - G1's curvature difference was also concentrated at the start.
- **u = 0 lies in the middle (S4).**
  - The control value never seen in training contracts at 3.2 to 3.9, between the two trained values. On a log scale it sits 0.48 to 0.60 of the way from u = +1 to u = −1, in every model.
  - Contraction behaves like a dial that u turns, at least at this one untrained point.
  - In G1, u = 0's curvature lay outside both trained values, so the two measures behave differently here.
- **G1's curvature result replicates (S5).**
  - On the new seeds and windows the curvature ratio is 1.00 to 1.25, all below 1.25 (seed 3103 is the closest, at 1.2499), so G1's refutation holds.
  - The direction of the curvature difference was not consistent: three models curve more at u = +1, two at u = −1. In G1 all five curved more at u = +1.
  - The curvature-profile classifier again reached 91 to 98%, and trajectories again moved faster at u = −1, by 16 to 28%.
- **Shuffled models contract like next-token mode.**
  - Models that cannot tell the tasks apart contract at 0.7 to 1.0, close to the main models' u = +1 rates, not between the two modes.
  - The strong contraction appears only when the model knows it is reading for the farther target.
- **α.** It grew to 0.18–0.22 in the main models and 0.08–0.11 in the shuffled ones, as in G1.

## Limitations

- **Farther and harder are confounded.** The token after next is both farther ahead and harder to predict (6.13 against 5.05 nats). G1c shows the control sets contraction; it cannot say which property of the task produces it. The tentative explanation in the preregistration, that a harder or farther target is better served by forgetting fine detail, is still untested.
- **The design follows G1, with the same limits:**
  - 10% of the training budget;
  - one learning rate, one model size and one dataset;
  - additive control, two trained control values and two tasks.
- **The trace is one summary of the Jacobian.** S2's normalisation adds the Frobenius norm, but the full spectrum was not measured.
- **The threshold is a judgement.** The factor of 2 was reused from amendment A1 for consistency, not derived. Every model cleared it by at least a factor of 2, so the verdict does not hinge on the exact value.
- **Contraction is shown to differ, not to do any work.** Whether the model needs it in order to predict the farther target is untested.

## Next conjectures

1. **Contraction is functional.**
   - The claim: the model needs strong contraction to predict the farther target.
   - The test: train with a penalty that holds the two control values' contraction equal, for example a Jacobian-trace regulariser in the spirit of Finlay et al. (2020).
   - The prediction: the skip task's loss rises by a preregistered margin while the next-token loss does not. If neither loss moves, contraction is a by-product and not a mechanism.
2. **Farther or harder.**
   - Add a third task that separates distance from difficulty: for example, the next token with the current token hidden. That is as near as the next token, but harder.
   - Contraction should follow difficulty if the tentative explanation is right, and distance if it is not.
3. **u as a dial.**
   - Train with a continuous u between −1 and +1 that sets a mixture of the two targets.
   - The prediction: contraction varies monotonically with u. S4's midpoint is one untrained point, found after looking at the results.
4. **G2, regimes as voices.** This now has a candidate regime measure: contraction. The question becomes whether strands of a text that a model tracks separately are carried under different contraction rates. It still needs the tracking measure, TV1.

## References

Finlay, C., Jacobsen, J.-H., Nurbekyan, L., & Oberman, A. M. (2020). *How to train your neural ODE: The world of Jacobian and kinetic regularization*. arXiv. https://doi.org/10.48550/arxiv.2002.02798

## Records

- **Raw results:** `results/g1c/` and `results/g1c_log.txt`, from the `colab-results` branch, run `g1c_contraction/20261005-225340`.
- **Run:** code commit ca3dcb5. One session, no resumes, no Drive failures, no deviations from the plan.
- **Analysis:** `g1c_analyze.py` writes `G1C_RESULTS.md` from those records; the job is `g1c_contraction.py`.
