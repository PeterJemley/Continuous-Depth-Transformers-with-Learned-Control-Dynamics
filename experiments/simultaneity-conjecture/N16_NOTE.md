# Result note: N16

Date: 2026-10-22    Preregistration: `N16_PREREGISTRATION.md`    Generated report: `N16_RESULTS.md`    Calibration run: `N16_CALIBRATION.md`    Exploratory report, added after the results: `N16_EXPLORATORY.md`

## In brief

A standard head trained on next-word answers, 80% of them scrambled, with each answer weighted by the head's own estimate that it is genuine, learns as N12's mixture head does (N13), and it ends 0.17 nats, a unit of prediction loss, further from the clean head than a head given only the genuine answers: the shortfall. N16 asked whether the shortfall comes from the head judging its own answers rather than from the weight the scrambled answers keep, by letting the clean head, head A, make that estimate instead. Head A's judgement recovered all of the shortfall and more, 108% of it (N16a, supported), while removing a tenth of the scrambled answers' share of the weight (N16b, supported), and on genuine answers alone it cost 0.45 nats less than the head's own judgement (N16c, supported); whether the recovered head's directions sit where its accuracy predicts was inconclusive (N16d). As a judge, head A differs from head B in at least three ways: its judgement does not depend on what head B learns, it predicts better, and its judgement is sharper; the test does not separate which of these matters.

## The arms and measures

Every model has two heads that read the next word from the same internal state, through a shared network, the trunk. **Head A**, the clean head, is trained on the genuine next word. **Head B** is the second head. In the arms with scrambled targets, 80% of head B's targets were replaced by words drawn at random in proportion to their frequency, u; the note calls these random targets, and genuine ones clean targets. The other 20%, the kept positions, keep the genuine word. All arms use N12's five seeds, 4101 to 4105, paired with the saved models of the same seed. A seed is one training run from its own random start. Everything is measured on N12's 512 windows held out from the training text.

**The judge** is the head whose prediction p sets each target's weight, the discount: w(y) = 0.2 p(y) / (0.2 p(y) + 0.8 u(y)), an estimate that target y is genuine. The note writes it r(y) when head B judges itself. In a self-judged arm, p is head B's own prediction. With head A as the judge, the oracle, p is head A's prediction, detached so that head A's own loss and gradient do not depend on it. The oracle needs a clean head, so it is a diagnostic, not a method.

| Arm | Genuine targets at | Random targets at | Weight on each target | Source |
|---|---|---|---|---|
| `rw-b1` | 20% of positions | The other 80% | Head B judges itself | N13's `n80-rw`, saved. By N13, the mixture head's inner softmax under another name |
| `rw-oracle` | 20% | 80% | Head A judges | New: the oracle head |
| `sub20` | 20% | None | Full weight | N12's genuine-only head, saved: the oracle mask, which drops the random targets exactly |
| `sub20-rw` | 20% | None | Head B judges itself | N14's control, saved |
| `sub20-oracle` | 20% | None | Head A judges | New |
| `clean` | Every position | None | Full weight | N12's, saved: the top rung of the ladder, and the reference for head A's cost |
| `sub10`, `sub05`, `sub025`, `sub0125` | 10%, 5%, 2.5% and 1.25%, each set inside the larger | None | Full weight | N14's, saved: the lower rungs of the ladder |

Two exploratory arms, with a sharper self-judged weight after a 300-step warm-up, were preregistered and dropped before the real run by the calibration's rule, because their heads did not train in the calibration seed.

| Term | Meaning |
|---|---|
| Nats | The unit of prediction loss |
| Inner reading | Head B's softmax: inside the mixture, for a mixture head; the head itself, for a single softmax |
| Δ_A | The inner reading's loss on the genuine targets minus head A's, within a model, in nats. Lower is closer to the clean head |
| G, the shortfall | Δ_A(`rw-b1`) − Δ_A(`sub20`), within a seed |
| F | (Δ_A(`rw-b1`) − Δ_A(`rw-oracle`)) / G: the fraction of the shortfall that head A's judgement recovers. 1 is all of it; above 1 is more |
| s, the random share | The random targets' share of head B's summed weight at the end of training |
| P | 1 − s(`rw-oracle`) / s(`rw-b1`): the share of that weight that head A's judgement removed |
| C | Δ_A(`sub20-oracle`) − Δ_A(`sub20-rw`): the two judges compared on genuine targets alone. Negative when head A's judgement costs less than head B's own |
| a | The agreement of the inner reading with head A: N9's agreement ratio, how far two predictions depend on the same directions of the internal state, corrected for measurement noise. It is 1 for the same directions |
| The ladder | N14's yardstick: for each seed, the points (Δ_A, a) of six heads trained on fewer and fewer genuine targets with no random ones, `clean`, `sub20`, `sub10`, `sub05`, `sub025` and `sub0125` |
| â, b | â is the ladder's agreement at a model's own Δ_A, by straight-line interpolation between the two rungs either side. b = â − a; positive means the head reads head A's directions worse than a genuine-only head with the same Δ_A |
| Kept, replaced | In the weights over training: positions with a genuine target, and positions with a random one |
| Ignored, neglect | A genuine target is ignored when its weight is below 0.01 |
| The copy and the 0.85 bar | The copy is the perfect noisy copy, 0.2 × head A's prediction + 0.8 × u. An agreement of 0.85 is taken to mean "the same reading" |
| S1, S3 | Two of the preregistration's secondary measures: S1 is the neglect measurement, and S3 is Δ_A(`sub20-oracle`) − Δ_A(`sub20`) |
| Head A's cost | Head A's loss on the genuine targets minus that of the same seed's `clean` arm, in nats. Positive is worse |

**The rules.** Each claim uses a Bonferroni-corrected 98.75% interval over the five seeds: four rules, so the level is 1 − 0.05 / 4. A claim is *supported* if its interval lies wholly on the claim's side of its margin: F above 0.75, P below 0.5, C below −0.30 nats, b within ±0.04. It is *refuted* if its interval lies wholly past the refuting margin with a capped standard deviation: F below 0.25 (at most 0.15), P above 0.75 (at most 0.15), C above −0.10 nats (at most 0.10), b outside ±0.04 (at most 0.03). It is *some effect, size unclear* if its interval excludes 0 but meets neither rule, and *inconclusive* otherwise. The exploratory intervals are 95%.

*The summary, this key and the records section at the end were added on 2026-10-25 so that the note can be read on its own. The record details, including the account of how the run went, were moved to the end, except its bullets on changes after approval and on the withdrawn N15, which stay below. Nothing else was changed.*

- **Changes after approval:** two readings of the preconditions fixed at build time, before the calibration (clarifications recorded in the deviation log: preconditions 1–3 apply to the claim arms, and precondition 4 is split per head); put to the author with the report that queued the calibration, with no objection raised before the real run; and the calibration's one permitted decision, by its rule, to drop both exploratory warm-up arms, whose heads did not train in seed 3001. Nothing after the calibration.
- **What came before.** N15, "sharpening the discount", was approved, built and calibrated, and withdrawn by the author before any real run: its sharpened discounts did not train at all, and its head-A-judged arm recovered the whole shortfall while removing too little of the random share for its rule to apply. N16 was written from that calibration. The calibration is N15's record; it motivated N16 and is not evidence for it.

## What we expected

**The question.** The mixture head, as N13 showed, trains its inner softmax by weighting each target with the discount r(y) = 0.2 p(y) / (0.2 p(y) + 0.8 u(y)), computed from head B's own prediction p. Trained that way as a single softmax (N13's `n80-rw`, called `rw-b1` here), it ends 0.337 nats from head A on clean targets, where the oracle mask `sub20`, which drops the random targets exactly, ends at 0.163. N15 asked whether the shortfall between them, G = 0.17 nats, is the random weight the discount leaves behind. Its calibration answered, in one seed, that a discount judged by head A instead recovered all of G while leaving most of the random weight in place. N14's control had found the self-judged weighting of clean targets alone, with no random targets (`sub20-rw`), collapsing to 0.74 nats from head A.

**The conjecture.** The shortfall is the discount's self-referential judgement: computed from head B's own prediction, it neglects the clean targets head B predicts poorly, and so head B never learns them. The random weight is not the cause. A judge that does not share head B's errors recovers the shortfall with most of the random weight still there, and its directions follow its function along the clean-target ladder.

**The arms,** on N12's five seeds, with the saved models of the same seeds re-measured:
- **`rw-oracle`:** `rw-b1` with the weight computed from head A's prediction, detached, so that head A's own loss and gradient do not depend on it (head A still shares the trunk with head B).
- **`sub20-oracle`:** `sub20`'s clean targets, each weighted by head A's judgement, with no random targets: the counterpart of N14's `sub20-rw`, which weights the same targets by head B's own.
- **References:** `rw-b1` (N13), `sub20` and `clean` (N12), `sub20-rw` and the clean-target ladder's rungs (N14), six rungs from `clean` to `sub0125`: N14's five with N12's `clean` added at the top, which is the only rung below `rw-oracle`'s loss.

**The claims,** each with a 98.75% interval (four rules, Bonferroni):
- **N16a (primary):** the recovered fraction F = (Δ_A(`rw-b1`) − Δ_A(`rw-oracle`)) / G wholly above 0.75. Refuted wholly below 0.25.
- **N16b:** the removed share P = 1 − s(`rw-oracle`) / s(`rw-b1`), with s the random targets' share of the summed weight at the end of training, wholly below 0.5. Refuted wholly above 0.75.
- **N16c:** C = Δ_A(`sub20-oracle`) − Δ_A(`sub20-rw`) wholly below −0.30 nats. Refuted wholly above −0.10.
- **N16d:** the residual b(`rw-oracle`) against the ladder wholly within ±0.04. Refuted wholly outside.

The preregistration said that N16a and N16c would be decisive if the calibration's values held within about 0.1, that N16b would be decisive in any case, and that N16d would most likely be inconclusive unless the residual was near zero or beyond 0.04.

## What happened

| Claim | Rule | Result (5 seeds, 98.75% intervals) | Verdict |
|---|---|---|---|
| N16a | F wholly above 0.75 | 1.081 [1.043, 1.119], standard deviation 0.020; 1.055–1.104 by seed | **Supported** |
| N16b | P wholly below 0.5 | 0.103 [0.065, 0.140], standard deviation 0.020; 0.072–0.120 by seed | **Supported** |
| N16c | C wholly below −0.30 | −0.448 [−0.474, −0.423], standard deviation 0.013; −0.471 to −0.437 by seed | **Supported** |
| N16d | b wholly within ±0.04 | +0.011 [−0.022, +0.043], standard deviation 0.017; −0.003 to +0.039 by seed | **Inconclusive** |

All six preconditions held: the setup checks (the weight identity at 4.9 × 10⁻⁷, the detachment, the warm-up switch); the 40 exact reproductions, the constructed check and the zero-noise copy in every model; the rotated control at 0.146–0.161 by arm; both new heads trained in every seed; G = 0.165–0.179; the ladder ordered and bracketing `rw-oracle` in every seed.

**By arm** (means over seeds):

| Arm | Judge | Random targets | Δ_A (nats) | Agreement with head A, a | The ladder at the same Δ_A, â | b = â − a | Random share at the end, s |
|---|---|---|---|---|---|---|---|
| `rw-b1` | head B's own | 80% | 0.337 | 0.804 | 0.814 | +0.010 | 0.422 |
| `rw-oracle` | head A | 80% | 0.150 | 0.892 | 0.903 | +0.011 | 0.378 |
| `sub20-rw` | head B's own | none | 0.736 | 0.772 | 0.668 | −0.104 | — |
| `sub20-oracle` | head A | none | 0.288 | 0.853 | 0.836 | −0.017 | — |
| `sub20` | none (full weight) | none | 0.163 | 0.896 | on the ladder | 0 | — |
| `clean` | none (full weight) | none, every target clean | −0.001 | 0.976 | on the ladder | 0 | — |

**The values against the calibration's.** F and P against N15's calibration seed, which motivated those two margins: F 1.08 against 1.02, P 0.10 against 0.15. C and b against N16's own calibration, run after the margins were fixed: C −0.45 against −0.48, b +0.011 against +0.022. C's margin had been set against the −0.57 implied by N14's control, and ±0.04 was carried over from N15. Each calibration value lies just outside the real run's 98.75% interval, and far from the margins. *[Corrected 2026-10-25: F, P and C lie just outside their real intervals; b's calibration value, +0.022, lies inside its real interval [−0.022, +0.043].]*

## Which kind of result this is

- **Three claims supported, with room.** Each interval sits far inside its supporting margin: F's lower end at 1.04 against 0.75, P's upper end at 0.14 against 0.5, C's upper end at −0.42 against −0.30. No seed came near a margin. The design predicted this if the calibration's values held, and they held.
- **The fourth inconclusive, at the design's stated limit.** The residual's spread across seeds (0.017, with one seed at +0.039) gives a half-width of 0.033, so support needed a mean within about 0.007 of zero; the mean was 0.011. The preregistration stated this limit in advance. The point estimate is on the conjecture's side, within the margin, and it is no more than that.
- **What N16c's support does and does not separate.** The rule compares the same clean targets weighted by two judges, and head A's judgement costs 0.45 nats less than head B's own. That is far from what the rival the claim was built against predicted, that any uneven weighting of clean targets costs alike whoever judges (within 0.10 nats). But head A differs from head B in at least three ways: it does not learn from its own weights, it predicts better, and its judgement is sharper (at the end of training it tells clean from random targets 6.7 times apart, against 5.6 for head B's own). The conjecture names the first, a self-confirming loop; the audit at drafting flagged the third; the design separates none of them. The claim's label, "self-reference is the mechanism", says more than its comparison separates. The verdict stands by its rule; the attribution is open, and next conjecture 1 proposes a test of the first two.
- **The critic's objection, that the claims were fitted to one seed,** was answered as the audit said it would be: the margins sat far from the values seen before drafting, and the five real runs, on N12's seeds reused as declared, came out far from the margins and near those values.

## What the results say together

**The shortfall is in who judges, not in how much random weight is left.** Head A's judgement removed a tenth of the random targets' share of the weight (P = 0.10) and recovered all of the shortfall and more (F = 1.08). On this evidence, and with N15's withdrawn conjecture as the rival, the random weight the discount leaves behind is not what holds the self-judged head back.

**A two-by-two, exploratory.** The four reweighted heads share the same clean targets at the same positions and differ in two things: who judges, and whether the other 80% of positions carry random targets. Their Δ_A, the inner reading's clean-target loss minus head A's, with 95% intervals from `N16_EXPLORATORY.md` (the judge's effect without random targets is N16c's C, whose preregistered 98.75% interval is [−0.474, −0.423]):

| | No random targets | Random targets | Effect of the random targets |
|---|---|---|---|
| **Head B judges itself** | 0.736 (`sub20-rw`) | 0.337 (`rw-b1`) | −0.399 [−0.422, −0.375] |
| **Head A judges** | 0.288 (`sub20-oracle`) | 0.150 (`rw-oracle`) | −0.138 [−0.145, −0.131] |
| **Effect of the judge** | −0.448 [−0.465, −0.432] | −0.188 [−0.194, −0.182] | interaction +0.261 [+0.243, +0.278] |

Three things follow from the table, all exploratory.
- **The weighting of clean targets is what costs.** Against `sub20`, the same clean targets at full weight, head B's own weighting costs 0.57 nats and head A's 0.12. Both judges cost something; the self-judged one costs nearly five times as much.
- **The random targets help, under either judge.** Adding them improves the reweighted head by 0.40 nats when it judges itself and by 0.14 when head A judges. This is the opposite of the share mechanism's premise. For the self-judged head it fits N14's exploratory conjecture, that the random targets, drawn from the word frequencies, keep the inner softmax from neglecting what the discount down-weights; under head A's judgement the weights do not depend on head B, and the gain has another candidate reading (below). The neglect measurement (secondary) agrees as far as it goes: on the clean targets head A finds implausible (about 0.6% of them, a set defined by head A's prediction, not head B's), the heads without random targets have losses of 14.2 (`sub20-oracle`) and 15.6 nats (`sub20-rw`), and the heads with them 11.9 (`rw-oracle`) and 11.4 (`rw-b1`), where `sub20` has 11.6.
- **The two factors interact.** The judge matters less when the random targets are present (0.19 against 0.45 nats), and the random targets matter more when the head judges itself (0.40 against 0.14). One reading is that the random targets compensate for part of the self-judged discount's damage, and that the judge's effect at 80% noise is what is left after that compensation.

**The oracle head beats the oracle mask, by a little, in every seed** (exploratory). `rw-oracle` ends 0.014 nats closer to head A than `sub20` does (95% interval [0.010, 0.018]; 5 of 5 seeds), which is why F exceeds 1. The calibration entry recorded this possibility before the real run. The preregistered "head A teaches" rival predicted a gain beyond the oracle mask, which happened, and located it in the clean targets, as `sub20-oracle` beating `sub20` (S3), which did not: S3 was +0.124 (98.75% interval [+0.110, +0.138]), so head A's weighting of the clean targets, on its own, gives no net gain over full weighting, and any teaching through the clean targets is outweighed by the down-weighting's cost. By the two-by-two, the excess comes with the random targets weighted by head A (exploratory). *A reading, by derivation and not tested:* at a random position the target y is drawn from u, so the expected weighted target is proportional to u(y) · w(y) = u(y) p(y) / (p(y) + 4 u(y)), with p the judge's prediction. That is about a quarter of p(y) where the judge finds y much less likely than four times its frequency, a fifth of u(y) where p(y) = u(y), and close to u(y) only where p(y) is many times u(y). With head A as the judge, the random positions therefore carry part of head A's prediction into head B's training, a form of distillation through the noise. The design did not isolate it.

**Neglect is graded, not exclusion.** At the preregistered threshold (weight below 0.01), most heads ignore 0.6–1.4% of the clean targets under their own discount, the hardest ones; the two reweighted heads without random targets ignore far more: `sub20-oracle` 3.1% and `sub20-rw` 8.5%, against `sub20`'s 0.6%. `rw-b1`'s own discount ignores 1.0%, 0.3 points more than `sub20`'s, and head A's judgement 0.6%. The signature the conjecture names is therefore strong only where the random targets are absent; with them, the self-judged head's neglect is a lower weight on targets it predicts poorly, not their exclusion. Read as the audit prescribed, to say how much of C is the blind spot (secondary): on the clean targets outside its own ignored set, `sub20-rw`'s loss (5.22 nats) is no worse than `sub20-oracle`'s (5.26), so its excess is concentrated in the 8.5% its own discount ignores, against 3.1%, rather than spread as a uniform shift. Because each head's ignored set is defined by its own prediction, this split partly sorts targets by loss, and it is a description, not a decomposition. A note on the measure: the loss on a head's own ignored set is high by construction (a weight below 0.01 requires p(y) below about 4% of u(y)), so the informative quantities are the fractions and the losses on the set head A ignores.

**The weights over training show the judges parting early.** The two discounts start equal, as they must (kept 0.092, replaced 0.089), and head A's judgement pulls ahead by step 100 (kept 0.247 against 0.140): head A learns from every clean target at full weight, so its judgement improves without waiting for head B. At the end it tells clean from random targets 6.7 times apart, against 5.6 for head B's own.

**Geometry.** The oracle head reads head A's directions at 0.892, as `sub20` does (0.896), and is not distinguishable from the ladder (N16d inconclusive, +0.011). Read against its own copy, as N12b read the mixture head, it reaches the 0.85 bar in 5 of 5 seeds (0.87–0.92), where `rw-b1` did in 1 of 5 and N12's mixture head in none. This is consistent with N14's reading that the bar is a functional bar. Of the two heads reweighted without random targets, `sub20-rw` reads head A's directions better than its loss predicts (0.10 above the ladder), and `sub20-oracle` is 0.02 above with a 95% interval that includes 0; the two with random targets have residuals of +0.010 and +0.011. Exploratory.

**Head A's cost** was within noise in every reweighted arm: against `clean`, from −0.007 to +0.009, every 95% interval including 0 (`sub20` itself costs +0.020, N12's finding).

## What this teaches

- **A diagnostic arm can answer a mechanism question that a method arm cannot.** The oracle needs a clean head, which a real noisy-label problem does not have, so it is no method. But by changing one thing, the judge, it showed where the shortfall lives. The result is consistent with co-teaching's reason for using a peer network to select samples (Han et al., 2018), a peer's independent errors, though N16 does not separate independence from accuracy.
- **A calibration can be a source of conjecture, if it is kept from being evidence.** N15's calibration overturned N15's design; the author withdrew N15 rather than amend it, and N16 was preregistered from what the calibration showed, with the calibration recorded as motivation, its values beside the margins, and new runs for the test, on N12's seeds as declared. The margins held at a distance from the calibration values, and the real runs came out near them.
- **A claim's label can attribute more than its comparison separates.** N16c compared two judges that differ in self-reference, in accuracy and in sharpness, and its label named only the first. The audit had asked whether the comparison isolates the judge, and flagged sharpness, but not accuracy; it had also prescribed reading a support with S1, which this note does above. The verdict stands by its rule; the attribution is reported as open.
- **The random targets were never only noise to be removed.** Earlier designs treated them as a cost: N12 took the mask that removes them as the limit, N14 looked for their trace, and N15 conjectured that their remaining weight was the shortfall. The two-by-two (exploratory) says that, given a reweighting, they are also part of what makes the reweighted head work.

## What this does not change

- Every earlier verdict stands. N13's identity stands: the self-judged reweighted head is the mixture head's inner softmax, and its shortfall is the mixture head's. N14's reading stands: along the axis of random targets, the directions a head reads follow its clean-target loss within that test's noise.
- N15 stays withdrawn; its calibration is its record.

## Limitations

- **Self-reference and the judge's accuracy are not separated** (above).
- **The oracle is a diagnostic,** and the oracle weight changes the random positions' targets as well as the clean ones' weights; the two-by-two separates them only for the clean targets.
- **The exploratory warm-up arms were dropped** by the calibration's rule, so whether a sharper self-judged discount can work after a warm-up was seen in one seed only (it did not: after the switch its ignored fraction grew as it trained).
- **Five seeds, reused with the saved models,** by necessity and declared; N16d's design limit, stated in advance, was reached.
- **One noise rate, one readout width, one scale, 10% of the budget;** the discounts are told the noise.

## Next conjectures

1. **Separate self-reference from the judge's accuracy.** Add a third head, C, trained exactly as `rw-b1` is, with its own discount, and weight head B's targets by head C's judgement, as co-teaching weights each network by its peer. The judge then has `rw-b1`'s accuracy, but head B's weights no longer depend on head B's own prediction. If head B recovers the shortfall as the oracle head did, the self-confirming loop is the mechanism; if it ends near `rw-b1`, the judge's accuracy is. A caveat for the design: head C shares the trunk and the targets with head B, so its errors may be correlated with head B's, and its own self-confirming loop runs as head B's did; a partial recovery would need its own reading.
2. **A floor on the self-judged discount,** no clean target below a minimum weight: the preregistration's next step for a supported N16a and N16b, and a test of whether the loop can be broken without a second head.
3. **Separate the two roles of the oracle weight:** head A judging only the random positions, with head B's own discount on the clean ones, and the reverse. The two-by-two's interaction and the excess over `sub20` both point at the random positions; this would test the distillation reading directly.
4. **Return to N10's question with mixture heads** (carried from N12, N13 and N14).

## References

Han, B., Yao, Q., Yu, X., Niu, G., Xu, M., Hu, W., Tsang, I., & Sugiyama, M. (2018). *Co-teaching: Robust training of deep neural networks with extremely noisy labels* [Preprint]. arXiv. https://doi.org/10.48550/arxiv.1804.06872

## Records

- **Raw results:** `results/n16/`, from Colab run `n16_judge/20261022-204649`.
- **Code commit:** 6b9501c.
- **Analysis:** `n16_analyze.py` writes `N16_RESULTS.md` from those records, with N12's and N13's, and `n16_exploratory.py` writes `N16_EXPLORATORY.md`; the job is `n16_judge.py`.

**How the run went.**
- **One session:** an NVIDIA A100 with 80 GB and PyTorch 2.11. It trained the 10 models, re-measured the 40 saved ones and measured everything in 30 minutes, about 2 minutes per training run.
- **No problems:** no resumes, no Drive failures, every training draw before the held-out boundary, every fingerprint equal to N12's `n80`'s.
- **The re-measurements** reproduced their records exactly: every loss and every ratio differed by 0, for all 40 saved models of N12, N13 and N14.
