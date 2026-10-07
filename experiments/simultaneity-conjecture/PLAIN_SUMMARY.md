# What we have done, in plain language

Date: 2026-10-06, updated 2026-10-07 with N5. Covers the work through N5. The mathematical and geometrical details are in the numbered notes at the end, so the main text can be read without them. Every number comes from a result note in this repository; the notes say which.

## The starting point: a paper's claims, tested and refuted

The paper proposed a language model with a twist. In the middle of an ordinary transformer, a few layers are replaced by a *continuous* block. Instead of passing a word's representation through a fixed number of steps, the block lets it flow smoothly through "depth", as a ball rolls along a surface whose slopes the model learns.[^1] A small control signal can tilt that surface, and the paper claimed this steered what the model writes, without costing accuracy or speed.

We first tried to reproduce the claims, with plans written and fixed before any run.
- **On a small dataset (WikiText-2), eight times over,** the model was about as accurate as an ordinary transformer, but most of the other claims failed. Steering worked reliably in only 5 of 8 runs. The model was slower, not as fast. A learned scale settled near 0.17, not 0.065. The geometric claim, that the control splits the flow into distinct regions, was not borne out by the paper's own measure.
- **On a dataset about fifty times larger (WikiText-103), trained properly,** the model was clearly worse than an ordinary transformer: by 1.10 nats, in all six runs.[^2]

## Why it failed

The block was trained with a memory-saving shortcut for computing its learning signal, the "adjoint" method. Training the same model with the exact signal instead closed about 95% of the gap: the shortcut alone cost about 0.8 nats.[^3] Trained exactly, the model was only 0.04 nats behind an ordinary transformer at that budget. So the idea was not the main problem; the way it was trained was.

## The real aim: reading the way we hear music

The project began with a conjecture of the author's, not with the paper. When we hear music, we hear several voices at once, each a continuous line, and we follow them all together. Language arrives one word at a time, and that order is right, but a reader still holds many strands at once: characters, arguments, tone, who knows what. The question is whether a model can hold its readings that way, simultaneously and distinctly, and whether continuous dynamics, like the paper's block, help it do so.

From here on, every model was trained with exact gradients, and given two "readings" to perform: predicting the next word, and predicting the word after that.

## The tests, one at a time

Each test was a question the world could answer "no", with the plan, the threshold and the decision rule fixed before any run, and controls built so that the test could fail.

1. **G1. Does the control signal send the flow through regions of different curvature, as the paper suggested?** No. Under one reading the paths through the block bend only 8 to 27% more sharply than under the other. The prediction was at least twice as sharply.[^4] *Refuted.*
2. **G1c. Does it set how strongly the flow pulls nearby states together?** Yes, strongly. Under the farther-ahead reading, the block draws nearby states together 4 to 10 times faster.[^5] *Supported*, and it replicated twice more in later tests.
3. **G1d. Does the model need that stronger pull?** No. Removing over 90% of it during training cost the farther reading nothing extra (−0.002 nats).[^6] *Refuted.* The pattern is real, but not load-bearing.
4. **N1. Then how does the model switch between readings?** With a simple shift. Replacing the control's influence on the flow with one fixed vector added after the block did just as well, within 0.001 nats.[^7] *Supported.* Switching between readings needs nothing from the dynamics.
5. **N2. When both readings must be held at once, does the flow give each its own pace, the way voices differ in tempo?** No. The farther reading's directions settle only 14 to 25% faster than the nearer one's, against a predicted factor of 2.[^8] *Refuted.*

   But N2 noticed something it had not predicted. The two readings lean on partly *different directions* in the model's internal space, overlapping only about 0.6, while two copies of the same reading overlap about 0.99.[^9]
6. **N4. Are the readings kept apart by direction, and does the model need that?** Yes. Forcing both readings to be read through the same 64 directions cost them 0.088 nats more than the same restriction cost a single reading, consistently across five runs.[^10] *Supported.* This is the first test in which a way of holding two readings at once turned out to be needed.

7. **N5. Is N4's cost a competition for directions?** We could not tell. If it were, giving the readout all 256 directions should have removed the cost. It shrank to about a third (+0.030 nats) and was positive in every run, but the runs varied too much for a verdict either way. At narrow widths, training itself became erratic: some runs simply learned worse, which swamped the comparison.[^11] *Inconclusive.* The plan had assumed the runs would vary as little as they did in N4, and they did not.

   N5 did find something it had not predicted, and it held in all 20 models that read two different targets. The two readings always ended up sharing about half of their most important directions (overlap 0.42 to 0.53), whether they started with more in common (0.7) or less (0.2). Training pushed them apart in the first case and together in the second. Two copies of one reading converged instead.[^12]

## What we now understand

- **The voices are separated by direction, not by tempo.** When one internal state must serve two readings, the readings occupy partly different directions within it, and squeezing them into the same directions costs accuracy. They do not get different timescales in the continuous block.
- **Two related voices share about half their space, and training finds that balance.** The next word and the word after settle at roughly half-overlap from either side (N5). Why the balance sits there, and whether it shifts for readings that differ more, is the most promising open question.
- **Why sharing costs is not yet settled.** N5 could not tell competition for directions from a clash of learning signals; some cost may remain even when nothing is restricted.
- **The continuous dynamics are not where the readings live,** at this scale. Switching readings needs only a shift (N1). Holding them at once relies on geometry, the arrangement of directions, rather than on the flow's dynamics (N2, N4).
- **This matches the music analogy better than the original picture did.** Listeners follow several voices more easily when the voices differ in timbre, the quality that tells instruments apart (the evidence is in `CONJECTURE.md`). Here, a direction plays the part of timbre.
- **Refutations taught as much as successes.** Each "no" ruled out a mechanism and pointed to the next question. None was rescued after the fact. Calibration on untrained models caught problems in three tests (G1, N2, N4) before any real run. In G1, the original measure would have called an untrained model a success.

## Limits

- Small models (about 30 million parameters, or 23 to 43 million with the two-reading heads), one dataset, 10% of the full training budget, and one learning rate.
- Only two readings, and closely related ones: the next word and the word after.
- N4 restricted the model's final readout, not the continuous block, and with only 64 of 256 directions. That is a tight squeeze.

## What comes next

The candidates, in the order N5 suggests:
- **Test the half-overlap pattern** as a prediction that can fail: readings that differ more (the next word against a topic label) should settle at less overlap, and readings that differ less at more.
- **Settle N5's main question with enough runs:** about ten runs at full width would tell a cost of 0.03 from none.
- **More voices at once,** three or four readings, at the width where training is consistent.
- **Move the restriction into the continuous block,** which would tie the results back to the original conjecture.

---

## Notes

[^1]: **The continuous block.** A word's representation h, a list of 256 numbers, evolves through "depth" τ from 0 to 1 by the differential equation dh/dτ = α·F(h, τ, u), where F is a small learned network, α a learned scale and u the control signal. The model solves it in 4 small steps (Euler's method). The paper calls the model "Model A"; it is the paper's `HybridODETransformer`. Sources: `../../MATH_EXPLAINER.md`, `../../model.py`.

[^2]: **Nats and intervals.** Accuracy is measured as loss, the average of −ln p over every word, where p is the probability the model gave the word that actually came next. A difference of 0.05 nats, the margin used throughout, is about a 5% change in how uncertain the model is (e^0.05 ≈ 1.05). Intervals are 95% t-intervals across independent runs (5 to 8 seeds), paired when the runs share starting conditions. WikiText-103: Model A minus baseline +1.10 nats, 95% interval [+0.97, +1.23], 6 seeds, Model A worse in all 6. Sources: `../wikitext2-replication/RESULTS.md`, `../wikitext103-training/RESULTS.md`.

[^3]: **Adjoint against exact gradients.** The adjoint method recomputes the flow backwards to save memory, instead of storing it, so its learning signal is an approximation. Cost of training with it rather than exactly: +0.79 nats, 95% interval [+0.66, +0.92], about 95% of the 0.83-nat gap at 10% of the budget. A direct comparison of the two gradients themselves (T0) was untestable under its own rules, so how the approximation goes wrong is not established. Sources: `../too-linear-conjecture/T1_NOTE.md`, `../too-linear-conjecture/T0_NOTE.md`.

[^4]: **Curvature.** For a path h(τ) with velocity v = dh/dτ and acceleration a = d²h/dτ², curvature is κ = √(|v|²|a|² − (v·a)²) / |v|³, how sharply the path turns. G1 compared mean curvature under the two control values: ratios 1.08 to 1.27 in five models; at least 2 was required in 4 of 5. Amendment A1 replaced an effect-size measure (Cohen's d) with this ratio before any real run, because untrained models gave d up to 6.0 and would have passed. Source: `G1_NOTE.md`.

[^5]: **Contraction.** The block's flow has a Jacobian J = ∂F/∂h, which says how a small change in the state changes its velocity. Its trace, the divergence of the flow, measures whether nearby states spread apart (positive) or are drawn together (negative). Contraction is minus the divergence, −tr(J), per token. It is estimated without forming J, by Hutchinson's method: tr(J) ≈ average of zᵀJz over random vectors z of ±1s, with Jz computed by forward-mode differentiation. Ratios (u = −1 over u = +1): 4.1 to 10.2 in five new models; shuffled-control models 0.997 to 1.031. Source: `G1C_NOTE.md`.

[^6]: **Capping contraction.** During training, the loss was given an extra term λ·max(0, c − 2)², where c is the contraction estimate on farther-reading sequences, so contraction above 2 per token was penalised. The capped models' contraction was 0.37–0.65, against 7.4–8.7 in uncapped models trained alongside them. The measure D was the change in the farther reading's loss minus the change in the next-word loss: −0.002, 95% interval [−0.007, +0.002]. Source: `G1D_NOTE.md`.

[^7]: **An offset.** In the "post" model, u never enters F. The block's output is h(1) + e(u), where e(u) is a learned vector, one per control value. Differences from the full model: +0.0010 and +0.0008 nats for the two readings, 97.5% intervals within [−0.008, +0.010]. Source: `N1_NOTE.md`.

[^8]: **Timescales within a reading's directions.** For each reading k, its sensitive directions were found first (note 9). Contraction within them was then measured as c_k = −E[zᵀJz]/r, with probes z = U_k ε confined to that reading's r = 16 directions. A higher rate means the flow forgets differences in those directions faster: a shorter timescale. Ratios (farther over nearer): 1.14 to 1.25; at least 2 was required. Source: `N2_NOTE.md`.

[^9]: **A reading's directions, and how much two readings share.** For each reading, the gradient g of its loss with respect to the block's output is collected for every word, and the matrix G = Σ g gᵀ is formed. Its top 16 eigenvectors, the columns of U, span the directions the reading is most sensitive to. The overlap of two readings is ‖U_Aᵀ U_B‖²_F / 16, the mean squared cosine of the principal angles between the two subspaces. It is 1 for identical subspaces, and about 16/256 ≈ 0.06 for random ones. N2: 0.57 to 0.63 for two readings, 0.98 to 0.99 for two copies of one. Source: `N2_NOTE.md`.

[^10]: **The shared-directions test.** Each reading's prediction is made through a projection: logits_k = W_k P_k z, where z is the final state after normalisation (256 numbers), P_k a 64 × 256 matrix and W_k maps to the vocabulary. "Separate" lets P_A and P_B differ; "tied" forces P_A = P_B. P_B started as a copy of P_A, so the two versions were identical until training. The measure is a difference of differences, which removes the general cost of a narrow readout: D = (J_dual-tied − J_dual-sep) − (J_same-tied − J_same-sep), with J the sum of the two readings' losses. D = +0.088, 95% interval [+0.078, +0.097]. Overlap at the block's output: 0.46–0.49 for two readings, against 0.95–0.97 for two copies of one; untrained, 0.455–0.501. Sources: `N4_NOTE.md`, `results/n4_exploratory/untrained_overlap.json`.

[^11]: **Why the full width decides.** At width 256, P is a square matrix, invertible from the start. Then W_k P can be any matrix W_k could be on its own, so tied and separate readouts can compute exactly the same things. Any remaining cost of tying must come from training, not from a lack of directions. N5's primary test: D at width 256, supported if its 97.5% interval lies below 0.02, refuted if it lies above 0.05. Result: D(256) = +0.030, 97.5% interval [−0.004, +0.065], positive in all 5 seeds; inconclusive. N5b, D(32) − D(128) = −0.063, interval [−0.32, +0.20]; inconclusive. The standard deviation of D across seeds was 0.022 at width 256, 0.060 at 128 and 0.194 at 32, against about 0.008 at 64 in N4. Sources: `N5_PREREGISTRATION.md`, `N5_NOTE.md`.

[^12]: **Half-overlap.** The overlap of note 9, for two different readings, by readout width (5 seeds each): untrained 0.67–0.74, trained 0.42–0.48 at width 32; untrained 0.45–0.50, trained 0.46–0.49 at 64; untrained 0.28–0.31, trained 0.44–0.53 at 128; untrained 0.18–0.20, trained 0.47–0.53 at 256. Two heads reading the same target: 0.87–0.99 after training. Exploratory; no decision rule. Source: `N5_NOTE.md`.
