# Result note: T1

Date: 2026-10-04    Preregistration: `T1_PREREGISTRATION.md`    Generated report: `T1_RESULTS.md`

## In brief

Model A, the paper's model, trailed an ordinary six-layer transformer of the same width by 0.83 nats at a tenth of the training budget. T1 asked which of Model A's four differences from that baseline carries the gap, by changing one at a time across five models, each trained on five seeds. Training the same model with exact gradients instead of the adjoint shortcut removed about 95% of the gap (0.79 nats). Losing one block's worth of separate weights cost 0.02 nats, reusing one block four times cost nothing beyond the margin, and the continuous block's form cost a further 0.035 nats whose practical size the test could not settle.

## The arms and the contrasts

All five arms share the paper's embedding, width 256, 4 attention heads, and layers 0–1 and 4–5; only what replaces layers 2–3 differs.

| Arm | Layers 2–3 replaced by |
|---|---|
| B6 | Two standard transformer blocks with their own weights: the baseline, an ordinary transformer |
| B5 | One standard block, applied once |
| T4 | One standard block, applied four times (the same weights reused) |
| AX | Model A's continuous block, trained with exact gradients |
| AA | Model A's continuous block, trained with the adjoint shortcut, as in the paper and the WikiText-103 study |

Each contrast is a paired difference in validation loss, run by run; positive means the first-named arm is worse.

| Contrast | What it measures | The guess it tests (`CONJECTURE.md`) |
|---|---|---|
| G = AA − AX (primary) | The cost of the adjoint shortcut | Rival G: the shortcut, not the design, causes the gap |
| P = B5 − B6 | The cost of losing one block's separate weights | Rival P: too few separate weights cause it |
| C2 = T4 − B6 | Whether one block reused four times matches two separate blocks | Conjecture C2, which predicted it would not |
| F = AX − T4 | The cost of the continuous block's form (α scaling, the time embedding, the parallel update, steps of α/4) beyond reusing one block | — |

The margin of practical importance is 0.05 nats, and each interval is a Bonferroni-corrected 98.75% interval over the five runs. A contrast is *material* if its interval lies wholly beyond ±0.05, *negligible* if wholly within, and *some effect, size unclear* if it excludes 0 but meets neither rule.

*The summary, this key and the records section at the end were added on 2026-10-25 so that the note can be read on its own. Nothing else was changed.*

## Controls

All four passed, so the confirmatory contrasts count as preregistered.

- **Construction:** the parameter counts match the definitions.
- **Same model, two backward passes:** identical in all 5 seeds.
- **Null contrast,** B6 against B6 with another initialisation: −0.002, interval [−0.008, +0.005], negligible. With 5 seeds, the design resolves effects well below the 0.05 margin.
- **Injected effect,** B6 at 80% of the budget: +0.127, interval [+0.120, +0.134], detected.

## The four contrasts, as preregistered

| Contrast | Result | Preregistered outcome |
|---|---|---|
| **G: AA − AX (primary)** | +0.792, interval [+0.661, +0.923] | **Material: adjoint training is worse** |
| P: B5 − B6 | +0.020, interval [+0.012, +0.028] | Negligible, and rival P is refuted: about 2% of the whole gap, against a predicted half or more |
| C2: T4 − B6 | +0.007, interval [−0.001, +0.015] | Negligible: C2's forbidden case happened, so **C2 is refuted at this scale** |
| F: AX − T4 | +0.035, interval [+0.018, +0.052] | Some effect, size unclear |

The whole gap at this budget, AA − B6, was +0.834 (exploratory, unadjusted interval [+0.754, +0.914]). The primary contrast G accounts for about 95% of it.

## What this teaches, by the table fixed in advance

- **G material.** Training Model A with adjoint gradients costs about 0.8 nats at 10% of the WikiText-103 budget. Most of the gap between Model A and the baseline at this budget comes from the training method, not the architecture. Trained with exact gradients, Model A was 0.04 nats worse than the baseline here (4.82 against 4.78 on average).
- **P refuted.** Losing one block's weights costs about 0.02 nats. Missing weights do not explain the gap.
- **C2 refuted at this scale.** Repeating one standard block four times matches the two separate blocks it replaces, within 0.05 nats. The rigidity of a repeated function does not cost loss here.
- **F: some effect, size unclear.** The ODE form (α scaling, the time embedding, the parallel update and steps of α/4) costs about 0.035 nats beyond weight tying with exact gradients. The interval excludes zero but reaches just past the margin, so its practical size is not settled.

## What this does not change

- **The paper's refuted claims stay refuted.** They were about Model A as built and trained, which includes the adjoint training. That model was worse than the baseline, by 1.10 nats at the full budget.
- **T0 remains untestable** under its own rules. T1 tests the consequence of a gradient defect; T0 tested the mechanism and gave no verdict. T1's result is not read back into T0.

## Exploratory observations

These are not part of the confirmatory family.

- **α.** With exact gradients, α ended at 0.12 to 0.15, close to its starting value of 0.1. With adjoint gradients, it grew to 0.32 to 0.41 by the same step. This fits T0's exploratory observation that the adjoint gradient of α has the opposite sign to the exact gradient, and it is still exploratory.
- **Learning rate.** AX tolerated, and chose, the highest rate in the grid, 2.4e-3. AA did badly at that rate in stage 1, 6.24 against 5.50 at 6e-4.
- **Training time per run.** AX was about 22% faster than AA (0.093 hours against 0.119), because it skips the adjoint's extra backward integration.

## Limitations

- 10% of the training budget. The full-budget gap was larger: 1.10 nats on test, against 0.83 here on validation.
- B6, B5, T4 and AX all chose the top of the learning-rate grid. Their optima may lie higher, and that may differ by arm.
- One model size, one dataset, one solver configuration, and 5 seeds.

## Next conjectures

1. **Model A trained with exact gradients at the full budget**, against the baseline. This is a new model with new evidence and needs its own preregistration. It does not revisit the paper's claims.
2. **Contrast F's 0.035 nats.** Which part of the ODE form costs it: α, the time embedding, or the parallel update? Each is a single-change test.
3. **The steering conjectures, C1 and C3,** should be tested on exactly-trained models, since adjoint training changes both α and the loss.

## Records

- **Raw results:** `results/t1/`, from the `colab-results` branch, run `t1_ladder/20261004-190740`. This was the second session; the first session's finished runs were resumed from their saved copies on Drive.
- **Analysis:** `t1_analyze.py` writes `T1_RESULTS.md` from those records; the job is `t1_ladder.py`.
