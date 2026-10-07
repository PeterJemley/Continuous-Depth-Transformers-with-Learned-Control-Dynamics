# Result note: N2

Date: 2026-10-06    Preregistration: `N2_PREREGISTRATION.md`, with amendment A1    Generated report: `N2_RESULTS.md`    Raw results: `results/n2/` and `results/n2_log.txt`, from `colab-results` branch `n2_coexist/20261006-164621`. Code commit a7724b7.

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
