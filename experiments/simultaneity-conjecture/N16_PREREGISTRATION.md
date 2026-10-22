# Preregistration N16: the judge, not the share

Date: 2026-10-22    Status: **draft, awaiting the author's approval.** Nothing has been built, calibrated or run.

Type: confirmatory for N16a (primary), N16b, N16c and N16d, each with its own decision rule. N16a, N16b and N16d use one new arm trained for this test; N16c uses a second new arm and N14's saved control. Two further arms are exploratory, with no claim. Everything else is secondary or exploratory and labelled so.

Parent documents: `N15_PREREGISTRATION.md` (withdrawn before any real run; its calibration entry is this test's motivating observation), `N14_NOTE.md` ("The reweighting without random targets is a different system"), `N13_NOTE.md` (the discount as the mixture head's learning signal).

**Relation to earlier results.**
- **The shortfall.** The mixture head's discount, computed from head B's own prediction, weights each target by r(y) = 0.2 p(y) / (0.2 p(y) + 0.8 u(y)). As a reweighted single softmax (N13's `n80-rw`, called `rw-b1` here) it ends 0.337 nats from head A on clean targets, reading its directions at 0.804, with the random targets holding 0.42 of the summed weight at the end. The oracle mask `sub20`, which drops the random targets exactly, ends at 0.163 and 0.896. The shortfall between them is 0.174 nats (0.165–0.179 by seed).
- **What N15's calibration found, in one seed (3001, the calibration windows).** A discount judged by head A's prediction instead of head B's, with nothing else changed, ended 0.163 nats from head A, where `sub20` was at 0.167: it recovered the whole shortfall. Its random share at the end was 0.35 against 0.41 for the self-judged discount, so it removed 15% of the random weight; it nearly ignored no clean target (0.01 of them). On N14's clean-target ladder at its own loss it sat 0.022 above the prediction, within the ladder's noise. The sharpened self-judged discounts, by contrast, did not train at all (10 nats from head A): a sharper rule admits so little at the start that the head's prediction never sharpens enough to admit the rest.
- **What N14's control found.** The self-judged discount applied to clean targets alone, with no random targets (`sub20-rw`), collapsed function to 0.736 nats from head A, while `sub20`, the same targets at full weight, stayed at 0.163: the self-judged weighting of clean targets costs 0.57 nats on its own.
- **What these two observations say together.** The mixture head's shortfall is not the random weight its discount leaves behind, since a judge that left most of it recovered all of the shortfall; and the self-judged weighting of clean targets damages a head even with no random targets at all. The explanation that fits both: the discount computed from head B's own prediction gives little weight to the clean targets head B predicts poorly, which are the ones it most needs to learn, and so it never learns them; a judge that does not share head B's errors has no such blind spot. This is co-teaching's reason for using a peer network to select samples (Han et al., 2018), met here from the other side.
- **What stands regardless.** Every earlier verdict stands. N15 was withdrawn before any real run and decided nothing; its calibration is one seed, on other windows, and motivates this test without testing it.

## The conjecture

**The mixture head's shortfall against the oracle mask is its self-referential judgement: the discount, computed from head B's own prediction, neglects the clean targets head B predicts poorly, and so fails to learn them. The random weight the discount leaves behind is not the cause. A discount judged by head A, which does not share head B's errors, recovers the shortfall while leaving most of the random weight in place, and its directions follow its function along the clean-target ladder.**

- **The mechanism.** A clean target that head B predicts poorly has a small p(y), so under head B's own judgement its posterior of being clean is low and its weight small; head B learns little from it, predicts it no better, and the weight stays small: a self-confirming neglect. The random targets are also down-weighted, which is the discount's purpose, but the harm is in what it does to the clean ones. Head A predicts the same targets from the same state without head B's errors, so a target head B predicts poorly but head A predicts well keeps its weight, and head B learns it. The random targets head A finds plausible keep weight too, which is why the share falls little; by N14, their cost is small while the clean targets are learned.
- **What it forbids.**
  1. The head-A-judged discount recovering less than three quarters of the shortfall on the real seeds.
  2. Its recovery coming with half or more of the random share removed, which would put the share mechanism back in play.
  3. The head-A-judged weighting of clean targets alone costing as much as the self-judged weighting of the same targets.
  4. Its directions off the clean-target ladder by more than the ladder's own spread.
- **The rivals it is built to exclude.**
  - *The share mechanism* (N15's conjecture): the shortfall is the random weight remaining; the oracle's recovery should then be about its removed share, 15% in the calibration. It predicts N16a refuted.
  - *The weighting's unevenness, whoever judges:* any discount that weights clean targets unevenly costs function; the self-judged and the head-A-judged weightings of clean targets alone would then cost alike. It predicts N16c refuted.
  - *Head A teaches head B:* the oracle weighting transmits what head A knows (a form of distillation), so the head-A-judged head gains beyond the oracle mask. It predicts `sub20-oracle` below `sub20`'s loss, which S3 reports; the conjecture predicts it at or slightly above.
  - *A trace beyond function:* the recovered head sits below the ladder. N16d, with its stated power.

## Competing hypotheses

| Hypothesis | N16a: the oracle's recovery | N16b: its removed share | N16c: self-judged against head-A-judged, clean targets only | N16d: the oracle head against the ladder |
|---|---|---|---|---|
| **The conjecture:** self-referential neglect | At least three quarters | Less than half | The head-A-judged weighting costs far less | On the ladder within 0.04 |
| The share mechanism | About its removed share, under a quarter | Any | Any | Any |
| The weighting's unevenness, whoever judges | Any | Any | Alike within 0.1 nats | Any |
| Head A teaches head B | At least three quarters, and `sub20-oracle` beats `sub20` (S3) | Any | Far less | Any |
| A trace beyond function | Any | Any | Any | Below the ladder by more than 0.04 |

## The arms

All use N12's model, training and batches at the 80% noise rate, on **N12's seeds 4101–4105**, paired with the saved models of the same seed: N13's `n80-rw`, N12's `sub20` and `clean`, N14's `sub20-rw` and ladder rungs. The reuse of seeds and models is declared, as in N13–N15; no earlier verdict is re-decided.

| Arm | Clean targets at | Random targets at | Head B's weight w(y) | Trained for N16? | Role |
|---|---|---|---|---|---|
| `rw-b1` | K (20%) | R (80%) | r(y) from head B's own prediction | No: N13's saved `n80-rw` | The self-judged reference |
| `rw-oracle` | K | R | 0.2 p_A(y) / (0.2 p_A(y) + 0.8 u(y)), from head A's prediction at the same forward pass, detached | Yes | N16a, N16b, N16d |
| `sub20` | K | none | 1 | No: N12's saved model | The oracle mask: the limit |
| `sub20-rw` | K | none | r(y) at ρ = 0.8 from head B's own prediction | No: N14's saved control | The self-judged weighting of clean targets alone |
| `sub20-oracle` | K | none | the head-A-judged weight at ρ = 0.8 | Yes | The head-A-judged weighting of clean targets alone; N16c |
| `rw-b2w`, `rw-b4w` | K | R | r(y) for the first 300 steps, then r_β(y) = 1 / (1 + 4 (u(y) / p(y))^β) at β = 2 or 4, from head B's own prediction | Yes, exploratory | Whether a sharper self-judged discount can work once the head has bootstrapped; no claim |
| `clean`, `sub10`, `sub05`, `sub025`, `sub0125` | — | — | — | No: N12's and N14's saved models | The clean-target ladder, six rungs with `clean` at the top |

**Details, fixed now:**
- **Weights** are computed in float32 without gradient at the same forward pass, as in N13 and N15; `rw-oracle` and the exploratory arms use N13's convention (the mean over all positions of w · CE), `sub20-oracle` uses N14's for clean-target-only arms (the sum over the kept positions divided by all positions), as `sub20-rw` did, so that N16c compares like with like.
- **The oracle weight** uses head A's softmax, detached; head A's own loss and gradient do not depend on it, and the setup check confirms that head A's readout receives no gradient from head B's loss. The arm is a diagnostic, not a method: it needs a clean head.
- **The warm-up** is 300 steps at β = 1, the point at which N13's discount had begun to discriminate (kept 0.31 against replaced 0.15); the switch is a step change. The exploratory arms carry no claim, and the calibration decides only whether they run.
- **Every run records,** every 50 steps, the mean weight on kept and on replaced positions, the random targets' share of the summed weight, and the fraction of kept positions whose weight is below 0.01.
- **Neglect at the end, measured.** For every model, on the measurement windows: the fraction of clean targets whose weight under head B's own discount (at ρ = 0.8) is below 0.01, the same under head A's judgement, and the model's clean-target loss on each ignored set against the rest. This is the mechanism's signature, recorded for every arm, as a secondary measure.
- **The measurement windows** are N12's: the held-out tail's windows 1–512.

## The measures

- **Δ_A and a,** as in N13–N15.
- **The shortfall G** = Δ_A(`rw-b1`) − Δ_A(`sub20`), within seed; **the recovered fraction F** = (Δ_A(`rw-b1`) − Δ_A(`rw-oracle`)) / G; **the removed share P** = 1 − s(`rw-oracle`) / s(`rw-b1`), with s the random share of the summed weight at the end of training.
- **The clean-target cost of each weighting, C** = Δ_A(`sub20-oracle`) − Δ_A(`sub20-rw`), within seed: negative when head A's judgement costs less than head B's own.
- **The ladder and the residual,** as in N14 and N15: six rungs from `clean` to `sub0125`; â the interpolation at a model's Δ_A; b = â − a.

## The claims

**N16a (primary): the head-A-judged discount recovers the shortfall.** The interval for F lies wholly above 0.75.

**N16b: the share is not the mechanism.** The interval for P lies wholly below 0.5.

**N16c: self-reference is the mechanism.** On clean targets alone, head A's judgement costs far less than head B's own: the interval for C lies wholly below −0.30 nats.

**N16d: geometry follows function.** The interval for b(`rw-oracle`) lies wholly within [−0.04, +0.04].

## Refuting results

- **N16a** is refuted if the interval for F lies wholly below 0.25, with a standard deviation of at most 0.15: the judge recovers at most a quarter.
- **N16b** is refuted if the interval for P lies wholly above 0.75, with a standard deviation of at most 0.15: the recovery came with the share removed.
- **N16c** is refuted if the interval for C lies wholly above −0.10 nats, with a standard deviation of at most 0.10: the two judges' weightings of clean targets cost alike, and the unevenness, not the self-reference, is the cause.
- **N16d** is refuted if the interval for b lies wholly outside [−0.04, +0.04], with a standard deviation of at most 0.03.

## Design

- **Runs:** 20 training runs (`rw-oracle`, `sub20-oracle`, `rw-b2w`, `rw-b4w` on seeds 4101–4105; the last two exploratory and subject to the calibration's rule); measurements of those models; re-measurements of N13's `n80-rw`, N12's `sub20` and `clean`, and N14's `sub20-rw`, `sub10`, `sub05`, `sub025` and `sub0125` (40 models), each checked against its committed record; the neglect measurement for every model.
- **Saved models** are loaded from the `n13_signal`, `n12_bottleneck` and `n14_dose` Drive folders; their committed records are the reference for precondition 2.
- **Setup checks:** the weight identity at β = 1 (N15's), the oracle's detachment (N15's), the warm-up switch (the weight at step 299 equals N13's discount and at step 300 the tempered one, on a batch), and the fingerprints equal N12's `n80`'s.

## Decision rules

**Preconditions:**
1. **Setup.** The batch, same-model, identical-heads and analytic checks; the weight identity, the detachment and the warm-up switch; the windows; no training draw in the held-out tail; each new run's fingerprints and noise fraction at step 0 equal N12's `n80`'s (`sub20-oracle`'s kept set equal to K).
2. **Must match.** Re-measuring the saved models reproduces their records (losses within 0.001, ratios within 0.01); the constructed mixture check and the zero-noise copy (≥ 0.999) in every model.
3. **Must differ.** The rotated control at most 0.60 in at least 4 of 5 models of each arm.
4. **The heads trained.** Δ_A(`rw-oracle`) ≤ 1.0 and Δ_A(`sub20-oracle`) ≤ 1.0 in at least 4 of 5 seeds.
5. **The shortfall exists.** G ≥ 0.10 nats in every seed.
6. **The ladder brackets and is ordered** for `rw-oracle` in at least 4 of 5 seeds, for N16d.

Preconditions 1–3 apply to every claim; 4 and 5 to N16a and N16b; 4 to N16c; 6 to N16d.

**Each claim has four outcomes:**

| Outcome | Condition |
|---|---|
| Supported | The interval lies wholly on the claim's side of its margin (F above 0.75; P below 0.5; C below −0.30; b within ±0.04) |
| Refuted | It lies wholly past the refuting margin (F below 0.25; P above 0.75; C above −0.10; b outside ±0.04), with the stated standard-deviation cap |
| Some effect, size unclear | It excludes 0 (for F, P and C) or meets neither rule while excluding 0 (for b) |
| Inconclusive | Everything else |

**Intervals.** Four interval rules, so each uses a Bonferroni-corrected t-interval at the 1 − 0.05 / 4 level (98.75%), with 4 degrees of freedom.

**Why these thresholds.**
- **Three quarters and a quarter of the shortfall** for N16a: the calibration seed gave 1.02; three quarters is "most of it", a quarter is "about the share removed", the rival's prediction.
- **Half and three quarters of the share** for N16b: the calibration seed gave 0.15; a recovery with less than half the share removed cannot be the share mechanism's; one with three quarters removed could be.
- **−0.30 and −0.10 nats** for N16c: N14's `sub20-rw` sits 0.57 nats above `sub20`; a head-A-judged weighting that costs at least 0.30 less than that is the self-reference's doing, and one within 0.10 of it is not.
- **±0.04** for N16d, as in N15: the ladder's own spread across seeds (0.05), above the measure's noise (0.02).
- **What the design can detect.** Δ_A varies across seeds by about 0.01, so F by about 0.06 and C by about 0.015; the 98.75% half-widths are about 0.11 for F and 0.03 for C, so N16a and N16c are decisive if the calibration's values hold within 0.1 and 0.1 nats respectively. P is exact per seed up to the share's seed spread (about 0.02): decisive. N16d's half-width is about 0.03; it is supported only if the residual is within about 0.01 of zero, as the calibration's 0.022 would not quite be; a residual beyond 0.04 would be decisive the other way. This is stated so that an inconclusive N16d is read as the design's limit.

## Secondary and exploratory outcomes

- **S1: neglect at the end,** for every model: the fraction of clean targets ignored (weight < 0.01) under head B's own discount and under head A's judgement, and the model's loss on each ignored set against the rest. The conjecture expects `rw-b1`'s own discount to ignore a larger fraction than head A's judgement does, and its loss on the targets its own discount ignores to be far above its loss on the rest.
- **S2: the exploratory warm-up arms,** if they ran: whether they trained, their F, their share, their ignored fraction, and their residual against the ladder. A sharper self-judged discount that trains after the bootstrap and recovers part of the shortfall would bear on the next test; one that still collapses would say the bootstrap is not the only problem.
- **S3: `sub20-oracle` against `sub20`,** Δ_A(`sub20-oracle`) − Δ_A(`sub20`): the head-A-judged weighting of clean targets alone against the full weighting. Below zero would mean head A teaches; at or above, the judge only removes the blind spot.
- **S4: the weights over training** in every new arm; and the oracle's share against β = 1's, step by step.
- **S5: head A's cost** in each new arm against `clean` and `rw-b1`.
- **S6: agreement with the copy** at 0.8 beside the head A measure, and the residual of `sub20-oracle` against the ladder.

## Necessity audit, at drafting (`NECESSITY_AUDIT.md`)

| Claim the design relies on | Features and ground | The check, and whether it works at the boundary |
|---|---|---|
| The oracle head trains, and recovers for the reason conjectured | Present mechanism; one calibration seed | Precondition 4; S1 shows the signature (what each judge ignores). At the boundary, a recovery with the share also removed is N16b's refutation, not N16a's |
| G, the shortfall, is fixed by construction | Definitional | Precondition 5; the records give 0.165–0.179 |
| The oracle weight does not couple the heads' training | Definitional (detached) | The gradient check; S5 shows head A's cost through the trunk, as in every arm |
| `sub20-oracle` against `sub20-rw` isolates the judge | Functional; the two arms share targets, positions and loss form | Reply 3: head A's judgement differs from head B's in sharpness as well as in what it ignores; S1's ignored fractions on the same windows say how much of C is the blind spot. A refutation holds either way; a support is read with S1 |
| "Head A teaches" is distinguishable from "the judge removes the blind spot" | Reply 3 | S3: if `sub20-oracle` beats `sub20`, head A teaches, and N16a's support would be read with that in view |
| The ladder is a null for function without random targets | As in N14 and N15 | N14 found the random-target heads on it and the self-judged control above it; N16d asks where the oracle head sits, and precondition 6 only that the ladder reaches it |
| The warm-up arms do not affect the claims | Definitional: they carry none | The calibration decides whether they run, and nothing else |
| One calibration seed motivated the conjecture | Reply 5 (level): a single value | The five real seeds are new runs with their own weights; the margins were set from the calibration value with room on both sides, and the refuting margins are live |

The critic's strongest objection, two moves deep: *"The oracle head's recovery was seen before this preregistration was written, in the calibration seed; the claims are fitted to it."* The reply: the calibration seed is on other windows and decides nothing; the five real seeds are new runs; and the claims forbid specific outcomes those runs can produce (a recovery under a quarter, a share removed by three quarters, two judges that cost alike, a residual beyond 0.04). *The critic's answer:* the margins were still chosen knowing the one value. *The next move:* accepted and recorded; the margins sit far from the calibration value on the supporting side (0.75 against 1.02; 0.5 against 0.15; −0.30 against the implied −0.57) so that an ordinary seed-to-seed spread cannot carry them, and the refuting margins are where the rivals live.

## Calibration before queuing, including a full-length run

- **A full-length calibration run on the A100,** as its own Colab job, seed 3001: `sub20-oracle`, `rw-b2w` and `rw-b4w`, with N15's calibration `rw-oracle` of seed 3001 (from the `n15_calib` Drive folder) and the saved calibration references re-measured, on the calibration windows 513–1024. `rw-oracle` is not retrained: N15's calibration model is that arm, by the same code, and its record is in `results/n15_calib/`. The run reports:
  - whether `sub20-oracle` trains, and C for seed 3001;
  - whether the warm-up arms train, and their F, share and ignored fraction;
  - the neglect measurement for every model;
  - the residuals against the five-rung calibration ladder.

  The values for seed 3001 will be seen, and recorded in the deviation log.
- **What the calibration may change:** only whether the exploratory arms run, by this rule: a warm-up arm runs on the real seeds if its head trains in seed 3001 (Δ_A ≤ 1.0); otherwise it is dropped and the reason recorded. No claim, margin, measure or rule changes in response to the calibration.
- **Also:** the untrained checks as in N15; the tiny end-to-end run and the resume test; synthetic records for every branch of the four rules.

## Sanity checks before trusting results

- Every data file's fingerprint matches the manifest; the held-out boundary is logged.
- Each new run's fingerprints match N12's `n80`'s; the weight identity, the detachment and the warm-up switch are logged.
- Every record names the GPU, the PyTorch version and the code commit. The job runs only on an A100.
- Every number in the report is computed by code from saved JSON records. `n16_analyze.py` is written and committed before any real run, and tested on synthetic records covering every branch of the rules.

## Compute, where it runs, and stopping rule

- **Where:** one Colab job, `n16_judge`, on an A100, resumable, after the calibration job `n16_calib`.
- **Cost:** 20 training runs at about 2 minutes each (10 if the exploratory arms are dropped), 60 measurements and 40 model loads: about 60 minutes. The calibration job: about 15 minutes.
- **Stopping rule:** exactly these runs and measurements, each once. Nothing is added after seeing results except as a dated, exploratory entry in the deviation log.

## What each outcome would teach

| N16a | N16b | N16c | N16d | What we would learn | Next |
|---|---|---|---|---|---|
| Supported | Supported | Supported | Supported | The mixture head's shortfall is its self-referential judgement, which neglects the clean targets it predicts poorly; a judge without its errors recovers the whole shortfall while leaving most of the random weight; geometry follows. The share mechanism is refuted. | A self-judged discount with a floor (no clean target below a minimum weight), to test whether the blind spot can be removed without a second head |
| Refuted | Any | Any | Any | The oracle's recovery in the calibration seed does not hold on new seeds; the judge is not the mechanism, or not reliably. | Measure the oracle head's course over training on every seed |
| Supported | Refuted | Any | Any | The judge recovers the shortfall, but by removing the share after all: the share mechanism survives, with a better discriminator. | Separate the two: an oracle judgement applied to random positions only, or to kept positions only |
| Supported | Supported | Refuted | Any | The judge recovers the shortfall with the random targets present, yet its weighting of clean targets alone costs as much as head B's own: the blind spot is not self-reference but the weighting's unevenness, which the random targets somehow offset. | Compare the two judges' ignored sets directly (S1), and test a floor |
| Any | Any | Any | Refuted below | The recovered head reads head A's directions worse than its function predicts: a trace beyond function, resolved where N14 could not. | Return to N14's conjecture with this design's power |
| Some effect or inconclusive | Any | Any | Any | The design's limits, as stated. | More seeds, or the floor test |

## Limitations known now

- **One calibration seed motivated the conjecture,** on other windows; the audit records how the margins were set.
- **Reused seeds and models,** by necessity and declared.
- **The oracle is a diagnostic, not a method:** it needs a clean head, which a real noisy-label setting does not have.
- **The exploratory arms carry no claim** and may be dropped by the calibration.
- **Five seeds,** with the detectable effects stated above; N16d in particular will most likely be inconclusive unless the residual is near zero or beyond 0.04.
- **One noise rate, one readout width, one scale, 10% of the budget;** the discounts are told the noise.

## Deviation log

(none yet)

## References

Han, B., Yao, Q., Yu, X., Niu, G., Xu, M., Hu, W., Tsang, I., & Sugiyama, M. (2018). *Co-teaching: Robust training of deep neural networks with extremely noisy labels* [Preprint]. arXiv. https://doi.org/10.48550/arxiv.1804.06872

Patrini, G., Rozza, A., Krishna Menon, A., Nock, R., & Qu, L. (2017). Making deep neural networks robust to label noise: A loss correction approach. In *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 2233–2241). IEEE. https://doi.org/10.1109/cvpr.2017.240
