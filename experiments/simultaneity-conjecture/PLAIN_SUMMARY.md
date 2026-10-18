# What we have done, in plain language

Date: 2026-10-06, updated 2026-10-07 with N5, 2026-10-08 with one added limit, N6 and N6r, 2026-10-09 with N7, 2026-10-10 with N8 and N9, and 2026-10-18 with N10. Covers the work through N10. The mathematical and geometrical details are in the numbered notes at the end, so the main text can be read without them. Every number comes from a result note in this repository; the notes say which. A critical review of the same work, asking where each guess could have failed and where the method fell short, is in `CRITICAL_REVIEW.md`.

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

8. **N6. Does that balance depend on how different the two readings are?** The test could not give a verdict. Its control, two heads reading the same target, had to agree almost perfectly (an overlap of at least 0.95) in 4 of 5 runs, and did so in 3. One run missed by 0.0002. The threshold had been carried over from N4, which measured on almost twice as many passages. With fewer passages, measurement noise keeps even identical readings from scoring as high, and that was not checked in advance.[^13] *Untestable.*

   What the runs showed, though, is what the conjecture predicted. Head B read the word 2, 4 or 8 places ahead. The further ahead, the less the two readings shared: about half at 2 places, under a third at 4, a quarter at 8. That held in the same order in every run. Training started all the pairs at the same point and moved each to its own balance. A new test with its control set correctly did turn this into a verdict: see N6r, next.

9. **N6r. The same test, with its control fixed.** On new runs, measured on fresh passages from the test split, and with the control judged against the measurement's own noise: *supported*, both claims. The further apart the two readings, the less they shared, in the same order in every run: about half for the next word and the word after (0.49), a third for words four apart (0.31), under a quarter for words eight apart (0.23). Two heads reading the same word converged (0.96). The numbers matched N6's to within 0.02.[^14]

10. **N7. Is it "more different", or just "harder"?** Words further ahead are also harder to predict, so N6r could not tell which mattered. N7 added a reading that is the next word exactly, made harder by scrambling 80% of its answers. If sharing tracked how different readings are, this head should have shared almost everything with the clean next-word head. It shared about a third (0.31), as little as the head reading four words ahead (0.30). *Refuted*, both claims.[^15] So the ladder N6r found is real, and it replicated again, but it does not measure how different two readings are, not on its own. The reason suggested at the time was that the measure picks up where a reading's mistakes fall as well as which features it uses. N8 refuted that.

11. **N8. A measure that leaves the mistakes out.** N8 measured the same trained models again, asking which directions change each head's *predictions*, with the right answers left out entirely. The scrambled next-word head still shared only about a third with the clean one (0.31), and still did not converge (agreement ratio 0.33): *refuted*, the two claims about it. The ladder survived: about 0.95, 0.44, 0.25 and 0.19 for words 1, 2, 4 and 8 ahead, *supported*.[^16] The new measure agreed with the old one to within 0.05 everywhere. So the old measure was not being fooled by mistakes. The ladder shows which directions each reading's predictions depend on. Why a scrambled copy of the next-word reading ends up using different directions is still open. One possibility is that it learned a weaker, different version of the reading. Another is that this kind of measure cannot recognise a noisy copy as the same reading at all. N8 did not check that. N9 did.

12. **N9. Can the measure recognise a noisy copy?** N9 built, from each clean next-word head, the perfect scrambled-answer version of it: a head that predicts exactly 20% of what the clean head predicts plus 80% word frequencies, with no training. The measure called this copy the same reading as the clean head (about 0.98): *supported*. The scrambled head N7 actually trained was nowhere near the copy (about 0.32): *refuted*.[^17] So the measure is not the problem. Trained on scrambled answers, the head learned a different and weaker reading of the context, not a noisy version of the right one. It still predicts almost the same words as the perfect copy overall, because both are mostly word frequencies, but the part that depends on the passage differs.

13. **N10. How much scrambling does it take, and can a small "payload" protect a reading?** New runs scrambled 20%, 50% or 80% of the second head's answers. The more scrambling, the further the head drifted from its perfect noisy copy, in that order in every run: *supported*. At 20% the head predicted almost exactly what the perfect copy would (within 0.04 nats), yet the directions it read had already drifted a little, just short of the bar for "the same" (0.81 against 0.85): *intermediate*. Three more runs at 80% scrambling each got a small payload: matching the perfect copy on 64 passages, pinning the head's directions to the clean head's, or clean answers on the same 64 passages. The main payload test gave no verdict, because the matching payload was too weak to take hold. That was my design error, and the check written for exactly this case caught it. Of the other two, pinning the directions made the head *look* closer to the copy but barely improved what it predicted, and it made the clean head worse. Clean answers on 64 passages backfired: the network memorised those passages.[^18]

## What we now understand

- **The voices are separated by direction, not by tempo.** When one internal state must serve two readings, the readings occupy partly different directions within it, and squeezing them into the same directions costs accuracy. They do not get different timescales in the continuous block.
- **Two related voices share about half their space, and training finds that balance.** The next word and the word after settle at roughly half-overlap from either side (N5), and did so again on new runs (N6).
- **Readings further ahead depend on different directions (N6r, N7, N8).** The overlap falls from a half to a third to a quarter as the second reading looks further ahead. That replicated three times, and it held under a second measure that leaves the right answers out (N8). So it is about what the readings' predictions depend on, not about their mistakes. But a reading that is the same as the next word, only scrambled, shares nearly as little. So "the more different the voices, the more separate their directions", which this summary stated as a rule after N6r, did not survive its test. N9 showed why the scrambled reading separates: training on scrambled answers taught the head a different, weaker reading. The measure can tell a perfect noisy copy from a different reading.
- **Why sharing costs is not yet settled.** N5 could not tell competition for directions from a clash of learning signals; some cost may remain even when nothing is restricted.
- **The continuous dynamics are not where the readings live,** at this scale. Switching readings needs only a shift (N1). Holding them at once relies on geometry, the arrangement of directions, rather than on the flow's dynamics (N2, N4).
- **This matches the music analogy better than the original picture did.** Listeners follow several voices more easily when the voices differ in timbre, the quality that tells instruments apart (the evidence is in `CONJECTURE.md`). Here, a direction plays the part of timbre.
- **Refutations taught as much as successes.** Each "no" ruled out a mechanism and pointed to the next question. None was rescued after the fact. Calibration on untrained models caught problems in six tests (G1, N2, N4, N7, N8, N9) before any real run. In G1, the original measure would have called an untrained model a success. It missed one in N6, where a control's threshold was not re-checked for a smaller measurement.

## Limits

- Small models (about 30 million parameters, or 23 to 43 million with the two-reading heads), one dataset, 10% of the full training budget, and one learning rate.
- Only two readings, and closely related ones: the next word and the word after.
- N4 restricted the model's final readout, not the continuous block, and with only 64 of 256 directions. That is a tight squeeze.
- How much the results vary from run to run depends on the readout width. The measured cost varied about 3 to 24 times more between runs at N5's widths than at N4's width of 64 (standard deviations 0.022 to 0.194, against 0.008).[^11] So a plan's estimate of how sensitive a test will be cannot be carried from one width to another, and this is why N5 was inconclusive.

## What comes next

The candidates, in order:
- **A payload strong enough to take hold.** Repeat N10's payload test with the matching payload weighted properly, at several strengths, and judge it against a fixed yardstick. That would finally test "anchor what is predicted, not the directions".
- **Does the cost of sharing grow along the ladder?** Repeat N4's forced-sharing test at 2, 4 and 8 words ahead. This measures what sharing *costs*, which N7 does not touch.
- **Readings that differ in kind, not only in distance,** such as a topic label.
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

[^13]: **N6.** Head A predicts token i+1; head B predicts token i+k, k = 1, 2, 4 or 8, at readout width 64, 5 seeds each. Overlap as in note 9, on validation windows 1,793–1,931 (the last 139). Same-target (k = 1) overlaps: 0.9498, 0.9571, 0.9624, 0.9396, 0.9633; the precondition needed at least 0.95 in 4 of 5. Exploratory: mean overlap 0.481 at k = 2, 0.305 at k = 4, 0.254 at k = 8; o(2) − o(8) = +0.227, 95% interval [+0.184, +0.269]; ordered in 5 of 5 seeds; all arms 0.48 untrained. Split-half stability of every head at least 0.89. Head B's loss rose with k (6.23, 6.92, 7.11 nats), so distance confounds "different" with "harder". Source: `N6_NOTE.md`.

[^14]: **N6r.** As note 13, but on new seeds (3701–3705), on 256 windows of the WikiText-103 test split, with the control judged by the agreement ratio: head A's subspace from one half of the windows against head B's from the other, divided by the geometric mean of each head's own split-half stability, so that two heads reading the same directions score about 1 whatever the number of windows. Same-target ratios 0.952–0.970 (at least 0.95 needed in 4 of 5); identical heads scored exactly 1, and untrained heads 0.48–0.50. Mean overlap 0.956 (k = 1), 0.485 (k = 2), 0.308 (k = 4), 0.233 (k = 8); o(2) − o(8) = +0.252, 95% interval [+0.219, +0.286], ordered in 5 of 5 seeds. Head B's loss rose with k (6.24, 6.92, 7.11 nats). Source: `N6R_NOTE.md`.

[^15]: **N7.** Head A predicts token i+1; head B predicts token i+1 (control), i+4, i+8, or i+1 with each target replaced, with probability 0.8, by a token drawn from the training data's word frequencies (amendment A1; 0.6 was too low to make the reading hard enough). New seeds 3801–3805, test-split windows 257–512, N6r's measures. Mean overlap with the clean head: 0.951 (same), 0.297 (i+4), 0.242 (i+8), 0.307 (noisy). Noisy minus i+8: +0.066, 95% interval [+0.045, +0.087], against at least 0.30 required. Noisy head's agreement ratio 0.30–0.33, against at least 0.85 required. Losses of head B: 5.13, 6.91, 7.11, 7.35 nats; the noisy head's loss against the clean targets was 6.60, so it also learned the reading less well. Source: `N7_NOTE.md`.

[^16]: **N8.** The models were N7's and N6r's, unchanged, measured on test-split windows 513–768.
    - *The target-free measure:* as in note 9, but each gradient is taken against a word drawn from the head's own predicted probabilities, not the word that actually came next. Its expectation is the Fisher information of the head's predictions with respect to the state. The two heads' draws share one random number per word, so identical heads draw identical words.
    - *Untrained heads score about 1 under this measure.* Before training, predictions are nearly uniform, and head B's readout starts as a copy of head A's. So amendment A1 added a control a head can fail: head A reading a randomly rotated state scored 0.13–0.19.
    - *Results.* Target-free overlap with the clean head on N7's models: 0.944 (same), 0.245 (i+4), 0.183 (i+8), 0.310 (noisy). Noisy minus i+8: +0.127, 97.5% interval [+0.105, +0.150], against at least 0.30 required. Noisy agreement ratio 0.315–0.339, against at least 0.85 required. On N6r's models: 0.945, 0.436, 0.249, 0.188, and o(2) − o(8) = +0.248, interval [+0.214, +0.281], against at least 0.10 required.
    - *Old against new.* The old measure on the same windows differed by at most 0.045 on average in any arm, and by 0.001 for the noisy head. When a head's predictions match its targets' distribution, the two measures have the same expected value (Kunstner, F., Balles, L., & Hennig, P., 2019, *Limitations of the empirical Fisher approximation for natural gradient descent*, https://doi.org/10.48550/arxiv.1905.12558).

    Source: `N8_NOTE.md`.

[^17]: **N9.** The models were N7's, unchanged, measured on test-split windows 769–1280 (amendment A1 doubled them from 256, for stability).
    - *The ideal noisy copy* of a clean head predicts exactly 0.2·p + 0.8·u, where p is the clean head's prediction and u the training-set word frequencies. It is built in log space from the clean head's own weights.
    - *Clean head against its copy:* 0.955–0.995 at 80% noise, and 0.954–0.991 in N7's other 15 models.
    - *Trained noisy head against the copy:* 0.305–0.349.
    - *Controls:* the copy at zero noise matched the clean head at 0.99999 to 1.00000; the rotated control scored 0.12–0.13; the analytic check scored 0.994, where measuring the wrong head scores 0.80.
    - *In prediction space:* the KL divergence from the noisy head to the copy was 0.07 nats per position. On the clean targets the noisy head's loss was 6.62, against 6.01 for the copy; on the noisy targets, 7.36 against 7.24.

    Source: `N9_NOTE.md`.

[^18]: **N10.** New seeds 3901–3905, 35 runs, measured on test-split windows 1281–1792.
    - *Dose arms:* head B's agreement with its ideal copy was 0.976 (clean), 0.812, 0.636 and 0.305 at 0, 20, 50 and 80% noise, ordered in 5 of 5 seeds. G, head B's clean-target loss minus the copy's, was 0.00, 0.04, 0.17 and 0.43 nats.
    - *Functional payload:* KL(copy ‖ head B) on 64 fixed windows, λ = 1, but on 4 windows per step against 32 in the batch. It reduced its windows' divergence only to 0.73–0.77 of the unanchored value, against at most 0.5 required, so N10a and N10b were untestable.
    - *Geometric payload:* pinned the readouts (overlap 0.998). It raised the agreement to 0.70, improved head B's clean loss by 0.07 nats (16% of the gap), and cost head A 0.13 nats.
    - *Data payload:* head B memorised its 64 windows (loss 0.003 nats on them). Head A got worse by 0.36 nats.

    The last three results are exploratory. Sources: `N10_NOTE.md`, `N10_EXPLORATORY.md`.
