# Result note: T1

Date: 2026-10-04    Preregistration: `T1_PREREGISTRATION.md`    Generated report: `T1_RESULTS.md`    Raw results: `results/t1/`, from `colab-results` branch `t1_ladder/20261004-190740` (second session; the first session's finished runs were resumed from Drive)

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
