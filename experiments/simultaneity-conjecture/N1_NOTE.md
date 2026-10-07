# Result note: N1

Date: 2026-10-06    Preregistration: `N1_PREREGISTRATION.md`    Generated report: `N1_RESULTS.md`    Raw results: `results/n1/` and `results/n1_log.txt`, from `colab-results` branch `n1_offset/20261006-121140`. Code commit ead1207. One session, no resumes, no Drive failures, no deviations from the plan.

## What we expected

**N1b (primary).** A model in which the control u never enters the ODE block's vector field, and is only added as a learned vector to the block's output, does both tasks as well as the model with u inside the field, within 0.05 nats.

**N1a (secondary).** In the trained full model, the control's effect on the block's output is mostly one constant vector.

## What happened

**N1b is supported: an offset is enough.**

| Task | Post − full | 97.5% interval |
|---|---|---|
| Next token (u = +1) | +0.0010 | [−0.008, +0.010] |
| Token after next (u = −1) | +0.0008 | [−0.008, +0.010] |

Both intervals lie well inside the ±0.05 margin, and the spreads (0.006) are far below the sensitivity limit. All three preconditions held:
- **Must match, by construction.** With the control zeroed, all four arms gave bit-identical logits for every seed. With the control present, they differed.
- **The tasks need the control.** The none arm was worse on the farther task by 0.32 nats, 97.5% interval [0.31, 0.34].
- **The full models use u.** The wrong control cost 1.86 to 1.92 nats.

**N1a is untestable, as the test runs foresaw.** The injected multiplicative check read 0.28 to 0.37 in the trained full models, against a limit of 0.1. Under its own rule, N1a decides nothing.

## What this teaches

- **Switching between the two readings needs nothing from the ODE block's dynamics.**
  - A single learned vector per control value, added after the block, does the job as well as letting the control shape the dynamics throughout.
  - So did adding the vector *before* the block, where the dynamics carry it but never see u (S1: differences of −0.001 and +0.000).
- **This explains G1d.** The contraction regime could be removed at no cost because the reading never depended on the dynamics. A fixed shift, read by the later layers, is enough.
- **It also explains G1c, at least in part.** See the exploratory observation on the pre arm below.
- **By the outcome table**, this is the "N1b supported" row: switching between readings needs only a direction, and the ODE dynamics play no needed part in it. The next conjecture the table names is N2: whether dynamics matter when readings must *coexist*, where a single shift cannot keep two readings apart.

## What this does not change

- G1c's finding that the contraction regimes exist stands. The full arm replicated it a third time here: ratios 4.6 to 6.0.
- G1d's finding that they are not needed stands, and N1 is consistent with it.
- G1's refutation, and the paper's refuted claims, stand.

## Exploratory observations, not confirmatory

None of these had a decision rule. N1a's measures are reported as description only, since its precondition failed.

- **The full model's control effect is large and varies with the state.**
  - The offset fraction fell from 0.89–0.93 at initialisation to 0.19–0.23 after training.
  - The top singular direction carries only 30 to 31% of the effect.
  - The effect's size is about half the size of the block's output.
  - So the full model does *not* implement the reading as a constant shift. It implements something state-dependent, and that state-dependent part is not needed: the post model does equally well with a constant shift.
  - This matches the outcome table's row "supported / refuted", but N1a cannot formally decide it.
- **Contraction regimes appear even when the field never sees u.**
  - In the pre arm, the control only moves the block's starting point, and the vector field is identical for both control values. Yet the contraction ratio was 2.5 to 4.4.
  - So contraction differences can arise purely from *where* the state starts, not from the control acting on the field.
  - G1c's S3 found that, *in the full model*, the difference lies in the fields themselves. The two findings are compatible: in the full model the field does differ, but a different starting point alone also produces regimes.
  - This strengthens the reading that the regimes are side effects rather than mechanisms.
- **α is lower when the field does not see u.** It was 0.12 to 0.13 in the post models, against 0.17 to 0.23 in the full models and 0.17 to 0.18 in the pre models. When the control does not act through the block, the block is used less.
- **The none models matched G1c's shuffled-control models,** losing about 0.4 nats on the next-token task and 0.32 on the farther task.

## Limitations

- **The setup is the same as G1–G1d:**
  - 10% of the budget, one learning rate, one model size and one dataset;
  - two tasks, with farther and harder confounded.
- **The test was about switching.** Each sequence is read one way. Whether an offset is enough when two readings must be produced from the same state at once is exactly what N1 does not test.
- **N1a's measure could not separate a shift from a scaling** in trained models, whose outputs carry a large component shared across tokens. A better measure of how the control's effect depends on the state would be needed to make N1a's question decidable.

## Next conjectures

1. **N2: regimes become load-bearing only when readings must coexist.**
   - N1 removes the last reason to think the dynamics matter for switching. So the founding question, simultaneity, is now the sharpest open one.
   - The design: a dual-reading model, where both predictions come from one forward pass and one state. Measure contraction within each reading's subspace, and test its necessity by intervention as in G1d.
   - The must-fail control is a single-reading model.
2. **What the state-dependent part of the control's effect encodes, exploratory and cheap.** The full model computes a large, state-dependent control effect that it does not need. Probing it, for example for current-token identity (path B), would show what it carries.
3. **N3 / G3: multiplicative control.** If additive control lets the model get away with an offset, control that multiplies the field might make the dynamics worth using. The prediction is a G1d-style necessity test that passes.
