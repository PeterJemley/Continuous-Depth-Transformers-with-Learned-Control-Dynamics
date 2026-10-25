# Result note: N2

Date: 2026-10-06    Preregistration: `N2_PREREGISTRATION.md`, with amendment A1    Generated report: `N2_RESULTS.md`

## In brief

N2 gave one model two heads that read its internal state at once, one predicting the next token and the other the token after next. It asked whether the continuous block's flow draws nearby states together (contracts) at least twice as fast along the farther reading's directions as along the nearer reading's (N2a), and whether forcing the two rates to be equal costs at least 0.05 nats (N2b). The farther reading's directions contracted only 14 to 25% faster, so N2a is refuted, and N2b, which required N2a supported, is untestable. An exploratory measurement found that the two readings rely on partly separate directions: they overlap 0.57 to 0.63, against 0.98 to 0.99 for two heads reading the same target.

## The arms and measures

Every arm is Model A, the paper's model, trained with exact gradients and no control signal for 2,879 steps (10% of the WikiText-103 budget), on seeds 3401 to 3405. Each model has two heads, full-width and untied, that read the same final state, and it trains on the mean of the two heads' losses. Reading A is the next token (i + 1). Reading B, the farther reading, is the token after next (i + 2).

| Arm | Head A reads | Head B reads | Training |
|---|---|---|---|
| dual | i + 1 | i + 2 | The full 2,879 steps, with a copy saved halfway, at step 1,440 |
| dual-same (the must-fail control) | i + 1 | i + 1 | The full 2,879 steps: two heads, one reading |
| equalised | i + 1 | i + 2 | Resumes from that seed's dual copy at step 1,440, with a penalty that pushes the two readings' contraction rates to be equal |
| sham (the null contrast) | i + 1 | i + 2 | Resumes from the same copy and computes the same penalty at zero weight, so the penalty changes nothing |

- **The ODE block** is Model A's continuous block, in which the state flows smoothly through depth by a learned differential equation (an ODE).
- **A reading's directions** (its subspace, U_A or U_B) are the 16 directions of the block's 256-number output that the reading's loss is most sensitive to. They are the top eigenvectors of the gradient covariance of that head's loss, estimated on validation windows 1,025 to 1,280. A window is one fixed-length passage of text.
- **The contraction rate** (c_A, c_B) is how strongly the block's flow draws nearby states together within a reading's directions, per direction. A higher rate means the flow forgets differences in those directions faster: a shorter timescale. It is measured on windows 1,281 to 1,536, and c̄ is its average.
- **The ratio** R_sub = c̄_B / c̄_A is the farther reading's rate over the nearer one's (the table's "B over A"). A rate below 0.002 per direction counts as near zero, and is not divided by (amendment A1).
- **Overlap** is how much two readings rely on the same directions: the mean squared cosine of the angles between their two subspaces. It is 1 when they are identical and about 0.06 for random directions.
- **Overall contraction** is the contraction rate of the whole state, summed over all 256 directions.
- **λ** is the weight on the equalising penalty. A pilot run on seed 100 was to try 0.01, 0.1, 1 and 10 in turn and keep the first that brought the ratio within 1.25. The first, 0.01, did, so no other value was run.
- **The combined loss** is L_A + L_B, the two heads' validation losses added. D_joint, N2b's measure, is the combined loss of the equalised model minus that of the dual model, per seed, on the whole validation split.
- **Must fail:** the dual-same models must give a ratio from 0.8 to below 1.25 in at least 4 of 5, since two heads reading the same target must not split. **Must match:** subspaces estimated from the odd and from the even windows must give log ratios within 0.1 of each other, in every dual model. **The analytic checks:** on linear fields with a known answer, the estimator must come within 2% on average (the "dense error") and be exact on a diagonal field, and the training penalty's gradient is checked too (the "gradient" error).
- **G1, G1c, G1d and N1** are earlier tests, each with a note in this folder. **N3** (or G3) is a proposed test of multiplicative control.

N2a is *supported* if the ratio is at least 2 in at least 4 of 5 dual models, *refuted* if it is below 1.25 in at least 4 of 5, and *intermediate* otherwise. N2b is *untestable* unless N2a is supported, the equalising takes hold (a ratio within 1.25 either way in at least 4 of 5 equalised models), and the sham arm's interval lies within ±0.05. Its margin of practical importance is 0.05 nats of combined loss, and its interval is a 95% t-interval over the five seeds. It is *supported* if D_joint's interval lies wholly above +0.05, *refuted* if wholly below +0.05 with a standard deviation of at most 0.04, *some effect, size unclear* if it excludes 0 but meets neither rule, and *inconclusive* otherwise.

*The summary, this key and the records section at the end were added on 2026-10-25 so that the note can be read on its own. Nothing else was changed.*

**How the run went.** The job needed more than one Colab session.
- The earlier sessions saved their finished runs and checkpoints to Drive but did not reach the upload step. Their logs are therefore not in the repository.
- The final session resumed from Drive, finished the rest, and uploaded all 50 result records, including the earlier sessions' work.
- No Drive failures were recorded, and nothing was changed after approval.

## What we expected

**N2a.** When one state must serve two readings at once, predicting the next token (A) and the token after next (B), the directions each reading relies on contract at different rates. The farther reading's directions contract at least twice as fast, in at least 4 of 5 models.

**N2b.** Forcing those rates to be equal costs the model at least 0.05 nats.

## What happened

**N2a is refuted.** The farther reading's directions contract only 14 to 25% faster than the nearer reading's:

| Seed | Ratio (B over A) |
|---|---|
| 3401 | 1.24 |
| 3402 | 1.25 |
| 3403 | 1.14 |
| 3404 | 1.22 |
| 3405 | 1.25 |

Four of five are below 1.25, and none comes near 2. The controls held:
- **Must fail.** Two heads reading the *same* target gave ratios of 0.995 to 0.999.
- **Must match.** Subspaces from odd and even windows agreed to within 0.02 in log ratio.
- **The analytic checks passed:** dense error 0.12%, diagonal exact, gradient 0.41%.

**N2b is untestable, as preregistered.** It required N2a supported, since there must be a large difference to equalise.
- Its other preconditions held: the pilot found λ = 0.01, the equalised models' ratios were within 1.25, and the sham arm's interval was [−0.001, +0.002].
- As description only: equalising changed the combined loss by +0.0006, 95% interval [−0.0005, +0.0017].
- That says little here, because the dual models' own difference was already small at the end of training.

## What this teaches

- **Coexistence does not make the ODE block give the two readings different timescales**, not at this scale and budget.
  - The difference that appears is consistent: every dual model contracts the farther reading's directions somewhat faster, and the dual-same models show none.
  - But it is a nudge, not a register.
  - This repeats G1's pattern for curvature: a real, consistent, small difference that is not the distinct regime the conjecture predicted.
- **N1 and N2 together.** By the outcome table in `PATHS_AND_UNDERSTANDING.md`, this is the "N1 supported, N2 refuted" row. Neither switching between readings nor holding them together needs depth dynamics at this scale. The readings live in the representation's geometry, not in its dynamics. The table's next step: move the programme to subspaces and attention, and test multiplicative control (N3) before leaving dynamics altogether.

## What this does not change

- All earlier results stand:
  - G1c's contraction regimes exist under switching;
  - G1d showed they are not needed;
  - N1 showed an offset is enough for switching.
- The paper's original claims stay refuted.

## Exploratory observations, not confirmatory

These had no decision rule. They are recorded because they bear on the next conjecture.

- **The two readings occupy partly separate directions.**
  - In the dual models, the two readings' subspaces overlap 0.57 to 0.63, on a scale where 1 means identical.
  - In the dual-same models, two heads trained on the same target, from different random starting weights, end up with subspaces overlapping 0.98 to 0.99.
  - So training makes a reading's directions converge, and two different readings settle into directions that are about 40% distinct.
  - The voices separate in **directions**, not in **timescales**. This is the most interesting finding of the run, and the one to test next.
- **Holding two readings costs the nearer one.** Head A's next-token loss was 0.05 nats worse in the dual models than in the dual-same models: +0.052, 95% interval [+0.040, +0.063]. The two readings compete for the shared state.
- **The timescale difference shrinks with training.**
  - At step 1,440 the ratio was 1.24 to 1.46. By step 2,879 it was 1.14 to 1.25.
  - Whatever distinct timescales appear early, training partly erases them.
- **Overall contraction is lower when the readings differ.** It was 1.07 to 1.30 in the dual models, against 1.26 to 2.13 in the dual-same models.
- **The subspaces were fairly stable** over the second half of training: overlap of 0.82 to 0.89 between step 1,440 and step 2,879.
- **A weakness in N2b's design, recorded for future tests.**
  - The pilot rule required the equalised model's ratio to be within 1.25 at the end. It did not require the *unpenalised* model's ratio to be outside it.
  - Here the dual models drifted to about 1.2 on their own, so the rule was met almost without the penalty doing anything.
  - N2b's precondition, that N2a be supported, caught this, which is why N2b is untestable rather than misleading.
  - A future necessity test should compare against the unpenalised model's own ratio.

## Limitations

- **The same limits as the earlier tests:**
  - 10% of the budget, one learning rate (not re-tuned for two heads), one size and one dataset;
  - two readings that are related, being adjacent future tokens.
- **The subspaces are estimates.** Gradient covariance captures the directions each head is sensitive to, not necessarily where information is stored.
- **One mechanism was tested.** Only per-subspace contraction was measured. Readings could differ in other dynamical properties, such as rotation or the rest of the Jacobian spectrum, that the trace does not see.

## Next conjectures

1. **Voices as directions.**
   - The claim: under coexistence, the readings are separated in direction, and the separation is needed.
   - The test: train dual models while penalising the overlap between the two readings' subspaces, or forcing them to share one subspace.
   - The prediction: forcing overlap costs more than the 0.05-nat margin, and dual-same models are unaffected.
   - This is the polyphony idea in the form this run points to: voices as separable directions in one state, like notes as separable frequencies in one sound.
2. **N3 / G3, multiplicative control.** Before leaving dynamics, give the readings a way to shape the field multiplicatively. Then check whether timescales separate and become necessary.
3. **More than two readings, and less related ones.** The two readings here are adjacent tokens. Readings that differ more, such as the next token against a topic or sentiment label, might separate more strongly in direction or in timescale.

## Records

- **Raw results:** `results/n2/` and `results/n2_log.txt`, from the `colab-results` branch, run `n2_coexist/20261006-164621`.
- **Code:** commit a7724b7. `n2_coexist.py` trains and measures the models, and `n2_analyze.py` writes `N2_RESULTS.md` from the records; the Colab job is `colab/jobs/n2_coexist.py`.
