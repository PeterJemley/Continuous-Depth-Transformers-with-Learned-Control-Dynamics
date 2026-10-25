# Result note: N1

Date: 2026-10-06    Preregistration: `N1_PREREGISTRATION.md`    Generated report: `N1_RESULTS.md`

## In brief

N1 asked whether switching between the two readings needs anything from the continuous block's dynamics, or whether one learned vector per control value, added after the block, is enough. That offset-only model did both tasks as well as the model with the control inside the block: the differences were +0.0010 and +0.0008 nats, with 97.5% intervals of [−0.008, +0.010], well inside the ±0.05 margin. The primary verdict, N1b, is supported: an offset is enough. The secondary hypothesis N1a, that the trained model's own control effect is mostly one constant vector, is untestable, because the check that its measure could fail read 0.28 to 0.37 against a limit of 0.1.

## The arms and the measures

The setup is G1c's. Model A is trained with exact gradients (ordinary backpropagation, not the adjoint shortcut), at 10% of the WikiText-103 training budget. Its continuous block is also called the ODE block. Inside it, the vector field is the learned function that says how the state changes with depth; the block's dynamics are the motion it produces. The control u says which task, or reading, each sequence has: u = +1 asks for the next token (the next-token task), and u = −1 for the token after next (the farther task). Strictly the tasks predict tokens, which are words or pieces of words. The arms differ only in where the control's learned vector, `control_embed(u)`, enters. Every arm has the same modules and the same number of parameters.

| Arm | Where the control enters |
|---|---|
| full | Inside the vector field, at every evaluation, as in G1, G1c and G1d |
| post | Nowhere inside the block: it is added once to the block's output. This is the offset-only model. |
| pre (secondary) | Added once to the block's input, so the dynamics carry it but the field never sees u |
| none (positive control) | Nowhere: the model is never told the task |

Every arm uses seeds 3301 to 3305, and a seed's arms start from identical weights.

- **N1b, the primary claim:** the post model does both tasks as well as the full model. Its two contrasts are post − full in validation loss, one per task, on the whole validation split.
- **N1a, the secondary hypothesis:** in the trained full model, the control's effect on the block's output is mostly one constant vector, an offset.
- **Δ:** the block's output under u = −1 minus its output under u = +1, per token, on validation windows 769 to 1,024.
- **Offset fraction:** the share of Δ that is one constant vector across all tokens and windows. It is 1 if Δ is the same everywhere, and near 0 if it varies freely.
- **Injected multiplicative check:** N1a's test that its measure can fail. The block's output is scaled by 0.9 against 1.1, a scaling rather than a shift, and the offset fraction must read at most 0.1 on it.
- **Logits:** the model's raw scores for every possible next token.
- **The outcome table:** the preregistration's table of what each result would teach. Its rows are named by the two verdicts, N1b's first, so "supported / refuted" means N1b supported and N1a refuted.
- **Secondary measures:**
  - S1 is pre − full for each task, with 95% intervals.
  - S2 covers the top singular direction's share (how much of Δ lies along its single strongest direction) and Δ's size relative to the block's output.
  - S3 is G1c's contraction ratio in the full and pre arms: how many times more strongly the block draws nearby states together under u = −1 than under u = +1.
  - S4 is the wrong-control costs (how much worse the next-token loss is with the wrong control value), α (a learned scale on the block's update) and training time.
- **Other codes:** G1, G1c and G1d are the earlier tests of curvature, contraction and whether the contraction is needed. N2 asks whether regimes, distinct kinds of motion in the flow, matter when two readings must coexist in one state; a dual-reading model gives both predictions from one forward pass, and a reading's subspace is the set of directions in the state it relies on. N3, or G3, is control that multiplies the vector field instead of adding to it. Path B, in `PATHS_AND_UNDERSTANDING.md`, is a probe of whether the current token's identity is harder to recover after the block.

**The decision rules.** N1b's two contrasts each use a Bonferroni-corrected 97.5% t-interval over the 5 seeds, with 4 degrees of freedom; the correction allows for testing two contrasts together. The margin of practical importance is 0.05 nats. N1b is *supported* if both upper bounds lie below +0.05 and both standard deviations are at most 0.04, the sensitivity limit. It is *refuted* if at least one interval lies wholly above +0.05. It is *some effect, size unclear* if at least one interval excludes 0 on the positive side and neither rule is met, and *inconclusive* otherwise, including a standard deviation above 0.04. Three preconditions must hold, or the result is *untestable*:
- must match, by construction: with the control zeroed, all four arms give bit-identical logits for every seed;
- positive control: for the farther task, none − full has a 97.5% interval wholly above +0.05, so the tasks need the control;
- the full models use u: the wrong control costs at least 0.5 nats on the next-token task in at least 4 of 5 full models.

N1a is *untestable* if the injected check reads above 0.1 in a trained full model. Otherwise it is *supported* if the offset fraction is at least 0.8 in at least 4 of 5 full models, *refuted* if it is at most 0.5 in at least 4 of 5, and *intermediate* otherwise.

*The summary, this key and the records section at the end were added on 2026-10-25 so that the note can be read on its own. Nothing else was changed.*

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
- **This explains G1d.** The contraction regime could be removed at no cost because the reading never depended on the dynamics. A fixed shift, read by the later layers, is enough. *[Corrected 2026-10-25: this goes further than N1 can. N1b shows that a model with no control inside the dynamics does as well; whether the trained model itself works by a shift was untestable (N1a), and this note's exploratory section finds its effect state-dependent. N1 is consistent with G1d, and does not explain it.]*
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

## Records

- **Raw results:** `results/n1/` and `results/n1_log.txt`, from the `colab-results` branch, run `n1_offset/20261006-121140`.
- **Run:** code commit ead1207. One session, no resumes, no Drive failures, no deviations from the plan.
- **Analysis:** `n1_analyze.py` writes `N1_RESULTS.md` from those records; the job is `n1_offset.py`.
