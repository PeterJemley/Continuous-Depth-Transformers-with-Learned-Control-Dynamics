# Result note: G1d

Date: 2026-10-06    Preregistration: `G1D_PREREGISTRATION.md`    Generated report: `G1D_RESULTS.md`

## In brief

G1d asked whether Model A needs the strong contraction that G1c found under the farther reading (u = −1, the word after next). It trained models with a penalty that capped that contraction, cutting it by over 90%, and compared each with an uncapped model from the same seed. The cap cost the farther task no more than the next-word task: the difference D averaged −0.002 nats, with a 95% interval of [−0.007, +0.002], wholly below the 0.05 margin. The preregistered verdict is refuted: the contraction is real but not, by this test, functional, and the test cannot say whether it was a by-product or whether the later layers did the same work.

## The arms and the measures

The setup is G1c's. Model A is trained with exact gradients (ordinary backpropagation, not the adjoint shortcut), at 10% of the WikiText-103 training budget. Its control u is added inside the vector field of the continuous block (the ODE block). The vector field is the learned function that says how the state changes with depth. The trained model computes the block in four fixed steps, its Euler steps. u = +1 asks for the next token (the next-token task or reading), and u = −1 for the token after next (the farther task or reading). Strictly the tasks predict tokens, which are words or pieces of words. The arms differ only in a penalty added to the training loss.

| Arm | Penalty during training |
|---|---|
| Control | None: G1c's training, on new seeds |
| Capped | λ · P, where P penalises contraction above a cap of 2.0 per token, on u = −1 sequences only, at the four states the block's Euler steps visit |
| Sham | The same penalty computation and λ, with a cap of 10⁶ that never binds. Its gradient is exactly zero. It is the null contrast. |

Every arm uses seeds 3201 to 3205. A seed fixes the initial weights, the data order and the tasks, so a seed's three arms differ only in the penalty. A pilot on seed 100 chose λ.

- **Contraction, c̄(u):** G1c's measure of how strongly the block draws nearby states together under control value u: minus the divergence (the trace of the vector field's Jacobian, which says how a small change in the state changes its velocity), per token, averaged over five depths and 256 windows. Here it is measured on validation windows 513 to 768. **R** is G1c's contraction ratio, c̄(−1) / c̄(+1). G1c's scoring table handles rates too near zero for a ratio.
- **λ:** the penalty's strength. The pilot was to try 0.01, 0.1, 1 and 10 in turn and keep the smallest at which the capped model's c̄(−1) was at most 3.0. The first, 0.01, qualified, so no other value was run.
- **D, the primary contrast:** for each seed, (farther-task loss, capped − control) − (next-token loss, capped − control). Positive D means the cap hurts the farther task more. Losses are on the whole validation split, each task with its right control value.
- **The sensitivity limit:** if the standard deviation of D across seeds were above 0.04 nats, the design could not detect 0.05 reliably, and *refuted* would not be claimed.
- **By-product and substitution:** the two explanations the test cannot tell apart, as the note defines them below.
- **Secondary measures:**
  - S1 is each task's change on its own, capped minus control, with its own 95% interval.
  - S2 is contraction under both control values in the capped models, by depth.
  - S3 is the penalty's value every 50 steps.
  - S4 is G1's other geometry: the curvature ratio, speed, and the curvature-profile classifier, which predicts u from how sharply trajectories turn at each depth (chance is 50%).
  - S5 is the wrong-control costs and α, a learned scale on the block's update. S6 is training time.
- **Wrong-control cost:** how much worse a task's loss is with the wrong control value.
- **Paths B, C and D:** directions of work set out in `PATHS_AND_UNDERSTANDING.md`. Path B is a probe of whether the current token's identity is harder to recover after the block under u = −1. Path C is polyphony on a task built for it, such as two interleaved texts. Path D is a model that holds several readings at once, with several flows read out together.

**The decision rule.** The interval for D is a 95% t-interval over the 5 seeds, with 4 degrees of freedom. The margin of practical importance is 0.05 nats. The claim is *supported* if the interval lies wholly above +0.05, and *refuted* if it lies wholly below +0.05. It is *some effect, size unclear* if the interval excludes 0 but meets neither rule, and *inconclusive* otherwise. Six preconditions must hold, or the result is *untestable*:
- the cap was imposed: a λ in the grid met the pilot rule;
- the cap took hold: c̄(−1) is at most 3.0 in at least 4 of 5 capped models;
- there was something to remove: R ≥ 2 in at least 4 of 5 control models;
- the models use u: the wrong control costs at least 0.5 nats on the next-token task in at least 4 of 5 control models;
- the null contrast: the sham arm's D has a 95% interval wholly within ±0.05;
- the analytic check: on a linear field with a known trace, the training penalty's estimate of the trace, and its gradient, are within 2%.

*The summary, this key and the records section at the end were added on 2026-10-25 so that the note can be read on its own. Apart from the corrections dated 2026-10-25, nothing else was changed.*

## What we expected

The claim: the strong contraction under the farther reading (u = −1), found in G1c, does work the model needs. Capping it during training would raise the farther task's loss by at least 0.05 nats more than the next-token task's loss.

## What happened

**The preregistered verdict is refuted: no material differential cost.**

The primary contrast, D = (change in farther-task loss) − (change in next-token loss), capped arm against control:

| Seed | D |
|---|---|
| 3201 | −0.006 |
| 3202 | −0.002 |
| 3203 | −0.006 |
| 3204 | +0.000 |
| 3205 | +0.002 |

The mean is −0.002, and the 95% interval is [−0.007, +0.002]. The interval lies wholly below the 0.05 margin, and the spread (0.0035) is far inside the sensitivity limit. The design could have detected a cost of about a tenth of the margin.

All six preconditions held:
- **The cap was imposed.** The pilot's weakest strength, λ = 0.01, already brought the farther reading's contraction to 0.62.
- **The cap took hold.** In the capped models, the farther reading's contraction was 0.37 to 0.65, against 7.4 to 8.7 in the control models. That removes over 90% of it.
- **There was something to remove.** The control models replicated G1c a second time: contraction ratios of 4.9 to 7.5.
- **The models use u.** The wrong control costs about 1.9 nats in every arm.
- **The penalty's machinery does nothing by itself.** The sham arm's D was +0.0003, interval [−0.002, +0.003].
- **The analytic checks passed:** trace 0.02%, gradient 0.58%.

## What this teaches

- **The strong contraction is not needed** by these models at this budget. The models' contraction under the farther reading was cut by over 90%, and the farther task did not suffer more than the next-token task.
  - Both tasks lost about 0.01 nats. Neither change is distinguishable from zero at 95%, and they are almost equal.
  - The models went on using u just as much: the wrong-control costs were unchanged.
  - The curvature-profile classifier still told the control values apart, 77 to 100%.
- **G1c's regime is real but not, by this test, functional.** By the preregistered outcome table: "Either it was a by-product, or other layers do the same work when the ODE block cannot."
- **G1c's result stands.** The contraction difference exists, and it replicated again here in the control arm. G1d adds that it is not a mechanism the model depends on.

## Which kind of result this is

- **Not a bug.** Every check passed, the cap held by a wide margin, and the sham arm matched the control.
- **Not an insensitive test.** The spread of D was a tenth of the sensitivity limit, and the interval is narrow.
- **The claim is false at this scale and budget, for this model.** The test cannot say which of two explanations is right:
  - by-product: the contraction just happened, and had no work to do;
  - substitution: the later layers did the same work once the ODE block could not.

## Exploratory observations, not confirmatory

- **The cap reached the next-token reading too.**
  - The penalty acted only on u = −1 sequences, with a cap of 2.0. Yet the capped models' contraction under u = +1 also fell, from about 1 to 1.7 down to −0.44 to +0.08: slightly expanding, on average.
  - u = 0 fell too. The whole field became nearly neutral.
  - The weakest penalty strength drove contraction far below the cap, and the penalty was zero for most of training, which suggests contraction was cheap for the model to give up. *[Corrected 2026-10-25: the records show the penalty above zero at 27 to 36 of the 58 logged steps, and zero at the other 22 to 31, including every logged step up to step 500. So it was zero for about half of training, not most. The report's table shows only the first, middle and last values, which were mostly zero.]*
- **A small difference remains.**
  - In the capped models, u = −1 still contracts a little more than u = +1, by about 0.4 to 0.9. G1c's scoring table would still classify it as "u = −1 contracts, u = +1 does not".
  - So the test removed the *large* difference, not every difference. It cannot rule out that a small residual difference is what the model uses.
- **α fell slightly:** 0.17 to 0.20 in the capped models, against 0.19 to 0.22 in the control models.
- **The cost of the penalty in compute:** 0.12 hours per run, against 0.086 for a plain run, about 1.4 times.

## What this rules out, and what it does not

- **Ruled out, for this setup:** that the model *needs* the large contraction difference between readings in order to predict the farther target.
- **Not ruled out:**
  - that later layers do the forgetting when the ODE block cannot (substitution);
  - that the small residual difference matters;
  - that contraction matters at a larger scale or budget;
  - that it matters when several readings must coexist, as opposed to being switched between.

## Limitations

- **The design follows G1c:**
  - 10% of the budget, one learning rate, one model size and one dataset;
  - additive control, two tasks, and farther and harder confounded.
- **The cap removed over 90% of the difference, not all of it.**
- **Necessity only.** The test asked whether the contraction is necessary. Whether it helps when it is free was not asked, and the absolute costs of about 0.01 nats are too small to read.

## Next conjectures

The preregistered outcome table's next step, and the paths in `PATHS_AND_UNDERSTANDING.md`, adjusted by this result:

1. **Substitution or by-product.**
   - Cap contraction across the whole stack after the embedding, not only in the ODE block, as a new preregistered test. If the farther task then suffers, the forgetting was being done elsewhere.
   - A cheap exploratory first look: measure whether the capped models' later blocks reduce the differences between inputs more strongly than the control models' do. G1d's final weights are saved on Drive.
2. **Is anything being forgotten at all?**
   - The probe test of path B, on the G1c or G1d models: is the current token's identity less recoverable after the ODE block under u = −1?
   - If the capped models keep that difference, the forgetting is done by something other than contraction.
3. **What the control actually changes.**
   - The control's effect is fully present in the capped models: wrong-control costs and classifier accuracy are unchanged. So whatever implements the two readings is not the contraction difference.
   - Finding what it is, for example by tracing u's effect from the ODE block through the later layers, is the next mechanistic question.
4. **Polyphony (paths C and D)** is unaffected. G1d says the regimes found so far are not load-bearing when the model *switches* between readings. Whether regimes matter when readings must run *at the same time* is still open, and that was always the founding question.

## Records

- **Raw results:** `results/g1d/` and `results/g1d_log.txt`, from the `colab-results` branch, run `g1d_functional/20261006-003944`.
- **Run:** code commit 935f8a9. One session, no resumes, no Drive failures, no deviations from the plan.
- **Second session:** a second session the next morning (`g1d_functional/20261006-093947`), after a runtime reconnect, found every run finished and skipped them all. Its 48 result files are byte-for-byte identical to the first session's.
- **Analysis:** `g1d_analyze.py` writes `G1D_RESULTS.md` from those records; the job is `g1d_functional.py`.
