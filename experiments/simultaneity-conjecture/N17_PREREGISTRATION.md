# Preregistration N17: the loop, not the judge's accuracy

Date: 2026-10-25    Status: **draft for the author's approval.** Not built, not run. Nothing in this file is a finding.

Type: confirmatory for N17a (primary) and N17b, each with its own decision rule. Both use new arms trained for this test, with a third head that judges head B. Everything else is secondary or exploratory and labelled so.

Parent documents: `N16_NOTE.md` ("What N16c's support does and does not separate"; next conjecture 1), `N16_PREREGISTRATION.md` (the oracle judge), `N14_NOTE.md` (the clean-target ladder), `N13_NOTE.md` (the discount as the mixture head's learning signal).

**Relation to earlier results.**
- **The shortfall.** A single softmax trained on 80%-scrambled next-word targets, each weighted by the discount r(y) = 0.2 p(y) / (0.2 p(y) + 0.8 u(y)) computed from its own prediction p, where u(y) is the word's frequency, from which the random targets are drawn (N13's `n80-rw`, called `rw-b1` here), ends 0.337 nats from head A on clean targets. The oracle mask `sub20`, which drops the random targets exactly, ends at 0.163. The shortfall between them, G, is 0.165–0.179 by seed.
- **What N16 found.** With head A, the clean head, as the judge (`rw-oracle`), head B recovered 108% of the shortfall (N16a, supported) while removing only a tenth of the random targets' share of the weight (N16b, supported). On clean targets alone, head A's judgement cost 0.45 nats less than head B's own (N16c, supported).
- **What N16 left open.** As a judge, head A differs from head B in at least three ways: it does not learn from the weights it sets, it predicts better, and its judgement is sharper. N16c's label, "self-reference is the mechanism", named only the first, and the comparison did not separate them. The note said so and proposed a peer judge to separate the first two. A peer alone cannot tell a loop from accuracy, since a self-judged peer is about as accurate as head B. So N17 keeps the peer and adds loop-free judges matched to head B's accuracy.
- **What stands regardless.** Every earlier verdict stands. N17 asks which of head A's differences carried N16's result. It does not re-decide N16.

## The conjecture

**The shortfall is a self-confirming loop in the judge. The discount is computed from a prediction that the same discount trains, so the clean targets the judge predicts poorly get little weight and stay poorly predicted. Remove the loop, and a judge no more accurate than head B recovers most of the shortfall. Keep the loop, in a judge that is not head B, and that judge does far worse than a loop-free judge as accurate as it is.**

- **The mechanism.** When head B judges itself, a clean target it predicts poorly gets a low weight, so head B learns little from it and goes on predicting it poorly. This would be analogous to the confirmation bias described in self-training, where a network fits its own wrong predictions and resists correcting them (Arazo et al., 2020). A judge trained at full weight on clean targets has no such loop in its own training. Its errors come from having less data, not from neglecting what it predicts poorly. A judge that weights its own targets by its own discount does have the loop, whether or not it is head B.
- **What it forbids.**
  1. A loop-free judge as accurate as head B, judging head B, recovering a quarter of the shortfall or less.
  2. A judge with its own loop doing about as well as a loop-free judge of the same accuracy.
- **The rivals it is built to exclude.**
  - *The judge's accuracy (R1).* What matters is how well the judge predicts the clean targets. Any judge as accurate as head B's own gives head B's result, loop or not. It predicts N17a refuted and N17b refuted.
  - *The judge being the learner (R2).* What matters is that head B judges itself. Any other judge of equal accuracy recovers, even one with its own loop. It predicts N17a supported and N17b refuted.
  - *Sharpness (R3).* Judges of equal accuracy can differ in how far apart they put clean and random targets. N17 does not control this. S2 reports it.
  - *The course of training (R4).* A loop-free judge matched to head B's accuracy at the end may have been more accurate earlier, when the weights shaped head B. The self-judged peer starts slowly, as head B does: N13 and N16 found the discount equal on kept and replaced targets at the start. So R4 predicts N17a supported, and d below 0, if the loop-free judges led early. A self-judged head's slow start is partly the loop's own doing, so N17 does not separate R4 from the conjecture. S3 reports the course.
  - *What the judge is trained on (R5).* The loop-free judges train on clean targets only, chosen with the true mask. The peer and the self-judge train on head B's targets, 80% of them random, and N16's two-by-two found that random targets change a self-judged head by 0.40 nats. At equal accuracy on clean targets, the two kinds of judge can still differ in what they predict for random words, and in how far their errors overlap head B's. R5 predicts what the conjecture predicts on both claims. S2 and S4 describe the differences.

**What N17 can separate.** N17 can separate the conjecture from R1 and from R2, each of which predicts a refutation. It cannot separate the conjecture from R3, R4 or R5, which predict the same outcomes. A support is therefore read as consistent with the loop, with those three named, and not as showing that the loop alone is the cause.

## Competing hypotheses

| Hypothesis | N17a: F*, a loop-free judge at head B's own accuracy | N17b: d, the peer against a loop-free judge of the peer's accuracy | S1: F of the peer itself |
|---|---|---|---|
| **The conjecture:** the loop in the judge | Above three quarters | Below −0.5: the loop costs at least half the shortfall | Low |
| R1: the judge's accuracy | About 0 | About 0 | About what its accuracy gives |
| R2: the judge being the learner | Above three quarters | About 0 | About what its accuracy gives |
| R3: sharpness | Follows the judges' sharpness (S2) | Follows the judges' sharpness (S2) | Any |
| R4: the course of training | High if the loop-free judges were ahead early (S3) | Below 0 if the loop-free judges led the peer early (S3) | Low |
| R5: what the judge is trained on | Above three quarters | Below 0 | Low |

## The arms

Every model is N12's: the trunk with head A and head B, each reading the trunk's final state through its own 64-direction readout. The training, batches and windows are also N12's, at the 80% noise rate. The arms use **N12's seeds 4101–4105**, paired with the saved models of the same seed. Seeds and models are reused, as declared in N13–N16. No earlier verdict is re-decided.

Each new arm adds **head C**, a third head, which is the judge. Its readout is also 64 directions wide, and it has its own initial values, the same in every arm of a seed. **Head C reads the trunk's state detached.** Its loss trains only its own readout, so the trunk is trained by head A and head B alone, as in `rw-b1` and `rw-oracle`. Head B's loss is N13's: the mean over all positions of w(y) · CE, with w(y) = 0.2 p_C(y) / (0.2 p_C(y) + 0.8 u(y)) computed from head C's prediction, without gradient. Head A is trained as in every earlier arm.

| Arm | Head C's targets | Head C's loss | Head C has a loop? | Trained for N17? | Role |
|---|---|---|---|---|---|
| `rw-peer` | Head B's own targets: clean at K (20%), random at R (80%) | Weighted by its own discount, as `rw-b1`'s head B is | Yes | Yes | The peer: a judge with a loop that is not head B. N17b, S1 |
| `rw-j10`, `rw-j05` | Clean targets at a nested 10% and 5% of positions, inside K: N14's `sub10` and `sub05` positions | Full weight, summed over its positions and divided by all positions (N14's convention) | No | Yes | Loop-free judges of two accuracies. N17a and N17b |
| `rw-b1` | — (head B judges itself) | — | Yes, in head B | No: N13's saved `n80-rw` | The self-judged reference. Its head B's accuracy is the target for N17a |
| `rw-oracle` | — (head A judges) | — | No | No: N16's saved model | The most accurate loop-free judge, for S7 |
| `sub20`, `clean`, `sub10`, `sub05`, `sub025`, `sub0125` | — | — | — | No: N12's and N14's saved models | G's reference (`sub20`), head A's cost (`clean`), and the clean-target ladder for S6 |

The loop-free judges are told which targets are clean, as head A is. Like N16's oracle, they are diagnostics, not methods.

**Details, fixed now:**
- **Which rungs.** The default loop-free rungs are 10% and 5%. In N14 the heads trained on those fractions ended 0.28 and 0.43 nats from head A, either side of `rw-b1`'s 0.34. A detached head C may land elsewhere. The calibration may replace the pair, by the rule given below, with adjacent rungs from 20%, 10%, 5%, 2.5% and 1.25% (N14's nested positions).
- **Head C's optimisation is tied to the main group's, one way.** The trunk, head A and head B are trained as before: one optimiser, one gradient scaler, clipped together by norm 1.0. Head C has its own optimiser, but its loss is scaled by the main group's current scale and its gradient clipped by the main group's coefficient, and it steps only when the main group steps. An overflow in head C alone skips only head C's step. Nothing about head C feeds back into the main group. So head C changes nothing in the trunk, head A or head B, and a peer started as an exact copy of head B stays identical to it. The setup checks test both.
- **Head C's initial values** are drawn from their own generator, so the main random state is untouched.
- **Every run records,** every 50 steps:
  - the mean weight head B's targets get, at kept and at replaced positions;
  - the random targets' share of the summed weight;
  - the fraction of kept positions with weight below 0.01;
  - head C's and head A's losses on the batch's clean targets, which give the judge's accuracy over training.
- **The measurement windows** are N12's: windows 1–512 of the held-out tail.

## The measures

- **Δ_A and a,** as in N13–N16. Δ_A is a head's loss on the clean targets minus head A's, in nats. a is its agreement with head A.
- **The judge's accuracy, x.**
  - For a new arm, x is head C's Δ_A at the end, on the measurement windows.
  - For `rw-b1`, the judge is head B itself, so x_self = Δ_A(`rw-b1`).
  - For `rw-oracle`, the judge is head A, so x = 0.
- **The recovered fraction,** F(arm) = (Δ_A(`rw-b1`) − Δ_A(head B in the arm)) / G, with G = Δ_A(`rw-b1`) − Δ_A(`sub20`), within seed. This is N16's F.
- **The loop-free line,** within seed: the points (x, F) of the loop-free rungs, sorted by x and joined by straight lines. Inside the rungs' range it gives F_lf(x). Outside it gives nothing.
- **F\* = F_lf(x_self):** what a loop-free judge exactly as accurate as `rw-b1`'s self-judge recovers.
- **d = F(`rw-peer`) − F_lf(x_peer),** with x_peer the peer's own accuracy: how far the peer falls short of a loop-free judge of its accuracy.

## The claims

**N17a (primary): without the loop, a judge as accurate as head B recovers the shortfall.** The interval for F* lies wholly above 0.75.

**N17b: with the loop, a judge that is not head B does not.** The interval for d lies wholly below −0.5.

## Refuting results

- **N17a** is refuted if the interval for F* lies wholly below 0.25, with a standard deviation of at most 0.15. A loop-free judge as accurate as head B then recovers at most a quarter of the shortfall, and the judge's accuracy, not the loop, carries N16's result.
- **N17b** is refuted if the interval for d lies wholly above −0.25, with a standard deviation of at most 0.15. The peer's loop then costs at most a quarter of the shortfall beyond what its accuracy explains.

## Design

- **Runs:**
  - 15 training runs: `rw-peer`, `rw-j10` and `rw-j05` on seeds 4101–4105. If the calibration names three rungs, the most it may, there are 20.
  - The measurements of those models.
  - Re-measurements of 40 saved models, each checked against its committed record. These are, for the five seeds, N13's `n80-rw`, N16's `rw-oracle`, N12's `sub20` and `clean`, and N14's `sub10`, `sub05`, `sub025` and `sub0125`.
- **Saved models** are loaded from the `n13_signal`, `n16_judge`, `n12_bottleneck` and `n14_dose` Drive folders.
- **Setup checks, at full size on a real batch:**
  - the weight identity (N15's);
  - head C's detachment;
  - the judge-source check;
  - the passenger check;
  - the copied-peer check;
  - the overflow test;
  - the nesting of head C's positions;
  - the fingerprints.

  Precondition 1 says what each must show.

## Decision rules

**Preconditions:**
1. **Setup.**
   - N16's checks: batch, same-model, identical-heads, analytic, the weight identity, the windows, and no training draw in the held-out tail.
   - Each new run's fingerprints and noise fraction at step 0 equal N12's `n80`'s.
   - Head C's positions in each loop-free arm equal N14's for that fraction, inside K.
   - Three new checks:
     - **Detachment:** head B's loss sends no gradient to head C, and head C's loss sends none to the trunk, head A or head B.
     - **The judge-source check:** with head C at its own initial values, the weight in head B's loss equals, bit for bit, the weight computed from head C's logits, and differs from head B's own discount. Without it, a bug that weighted head B by its own prediction would pass every other check, turn `rw-peer` into `rw-b1`, and favour N17b.
     - **The passenger check:** 20 steps of `rw-b1` with a head C trained as the peer but not judging leave the trunk, head A and head B bit-identical to 20 steps without it.
     - **The copied-peer check:** over 20 steps, a peer started as an exact copy of head B, judging head B, gives a trunk and heads bit-identical to `rw-b1`'s, and head C stays bit-identical to head B. AdamW is elementwise, so a separate optimiser adds no roundoff. Bit-identity requires that head C's loss carries the same factor of 0.5 as head B's and runs in the same autocast region, that it is unscaled by the same reciprocal scale and clipped by the same coefficient tensor, and that its AdamW settings are identical.
     - **The overflow test:** an injected inf, once in head C's gradient alone and once in the main group's. In the first case only head C's step is skipped and the main scale is unchanged; in the second both steps are skipped. Overflows are rare (at most one skipped step per run in N12–N16), so the 20-step checks would not meet one.
2. **Must match.** Re-measuring the saved models reproduces their records: losses within 0.001 and ratios within 0.01. The constructed mixture check passes, and the zero-noise copy scores at least 0.999, in every model.
3. **Must differ.** The rotated control scores at most 0.60 in at least 4 of 5 models of each arm.
4. **The heads trained.** In each new arm, Δ_A of head B and of head C is at most 1.0 in at least 4 of 5 seeds.
5. **The shortfall exists.** G is at least 0.10 nats in every seed.
6. **The loop-free line reaches the judge.**
   - *(a)* In at least 4 of 5 seeds, the rungs' accuracies x are ordered by fraction, at least 0.05 apart, and bracket x_self.
   - *(b)* In at least 4 of 5 seeds, they bracket x_peer, and F_lf(x_peer) is at least 0.75.

Preconditions 1–3 and 5 apply to both claims. Precondition 4 applies to the arms each claim uses: the loop-free rungs for N17a, and those plus `rw-peer` for N17b. 6a applies to N17a and 6b to N17b. Each claim pools the seeds its precondition 6 brackets, as N16d did.

**Each claim has four outcomes:**

| Outcome | Condition |
|---|---|
| Supported | The interval lies wholly on the claim's side of its margin: F* above 0.75; d below −0.5 |
| Refuted | It lies wholly past the refuting margin, with the stated standard-deviation cap: F* below 0.25; d above −0.25 |
| Some effect, size unclear | It excludes 0 but meets neither rule |
| Inconclusive | Everything else |

**Intervals.** There are two interval rules, so each uses a Bonferroni-corrected t-interval at the 1 − 0.05 / 2 level (97.5%), with 4 degrees of freedom, or 3 if one seed is not bracketed.

**Why these thresholds.**
- **Three quarters and a quarter** for N17a, as for N16a. Three quarters is "most of the shortfall". A quarter is close to what R1 predicts: under R1, the loop-free line passes through F = 0 at x_self, the self-judge's own point.
- **−0.5 and −0.25** for N17b. If the loop costs at least half the shortfall beyond what accuracy explains, the loop is a main cause. If it costs at most a quarter, it is not. R1 and R2 both predict d near 0.
- **Why N17b needs F_lf(x_peer) of at least 0.75** (precondition 6b). The peer's F is expected near or above 0, since a self-judged peer judges about as well as head B judges itself. So d can fall below −0.5 only where F_lf(x_peer) is well above 0.5. Where the loop-free line is low at the peer's accuracy, the conjecture and R2 both predict d near 0, and the comparison cannot tell them apart. N17b is then untestable, not refuted.
- **What the design can detect.**
  - Δ_A varies across seeds by about 0.01, so a single arm's F varies by about 0.06; N16 saw 0.02 for `rw-oracle`.
  - F* and d also carry the rungs' scatter, so their standard deviations are likely 0.05–0.08. The 97.5% half-width with five seeds is about 1.56 standard deviations, so 0.08–0.12. With four bracketed seeds it is 2.09 standard deviations, so 0.10–0.17.
  - N17a is then decisive if F* lies above about 0.87 or below about 0.13.
  - N17b is decisive if d lies below about −0.62 or above about −0.13.
  - Between those values the verdicts will most likely be intermediate. That is the design's limit, stated now.

## Secondary and exploratory outcomes

- **S1: the peer's own recovery,** F(`rw-peer`), beside its accuracy x_peer. This asks whether a judge that needs no clean labels, a second self-judged head, helps head B: the co-teaching question (Han et al., 2018; Malach & Shalev-Shwartz, 2017).
- **S2: each judge's sharpness and share** at the end: the mean weight on kept targets over the mean on replaced ones, and the random targets' share of the summed weight, s. Both come from the training-batch records at the last logged step (2850), averaged over seeds. That is the source of N16's 5.6 for head B's own discount and 6.7 for head A's. S2 also gives F against sharpness across all the judges.
- **S3: each judge's accuracy over training,** from the batch records: the loop-free judges against the peer, a self-judged head, in the same runs. N13 recorded head B's validation loss on its noisy targets, not on clean ones, so `rw-b1`'s own course is not available, and the peer stands in for it. If the loop-free judges were ahead early, a support of either claim is read with that in view (R4).
- **S4: how far the judge's blind spot is head B's.** At the end, on the measurement windows:
  - the clean targets ignored (weight below 0.01) under the judge, and under head B's own discount;
  - the share of head B's own ignored set that the judge also ignores, which is 1 for `rw-b1` by construction;
  - head B's loss on each set.
- **S5: head A's cost** in each new arm against `clean`.
- **S6: geometry.** Head B's agreement with head A against the clean-target ladder, for each new arm, as in N16d, with no rule. â is the ladder's agreement at the head's Δ_A, by straight-line interpolation between the rungs either side, and b = â − a.
- **S7: the whole loop-free line,** with `rw-oracle` (head A, x = 0) as its most accurate point. Description only.

## Necessity audit, at drafting (`NECESSITY_AUDIT.md`)

| Claim the design relies on | Features and ground | The check, and whether it works at the boundary |
|---|---|---|
| The loop-free judges have no loop | Definitional for head C's own loss: it is unweighted, so its targets' weights never depend on its prediction. A weak second-order path remains: head B's loss, weighted by head C, trains the trunk that head C reads. That path runs in every arm with a judge other than head B, `rw-oracle` included, and is not separated | The code and precondition 1. At the boundary, a rung's head C that does not train fails precondition 4, and is not read as a judge |
| Head B is weighted by head C's prediction | Definitional, but a bug here would turn the peer into `rw-b1` and favour N17b | The judge-source check (precondition 1), with head C at its own initial values, where the copied-peer check cannot see the difference |
| The judges do not train the trunk, so the arms differ only in the judge | Definitional: the detach, and head C's one-way tie to the main group's scale, clipping and steps | The detachment and passenger checks (precondition 1), both exact. Overflow steps are covered by the overflow test, since the 20-step checks would not meet one |
| The peer is not head B, so its result is not self-judging under another name | Reply 1: only its initial values differ | The copied-peer check shows the boundary: with head B's initial values, the peer *is* head B. With its own, it starts as another head and may converge towards head B, since both read the same features and targets. S4 measures how far the blind spots overlap |
| "As accurate as head B" means matched at the end of training | Reply 5: a condition fixed to one time | S3 reports the course (R4). A support of N17a is read with S3. A refutation holds whatever the course, since a loop-free judge ahead early that still fails is stronger evidence for R1 |
| Matching accuracy also matches the judge in every respect that matters | Reply 3: another mechanism | No. Sharpness (R3) is not matched, and S2 reports it. What the judge is trained on (R5), clean targets only against head B's noisy ones, is not matched, and S2 and S4 describe it. The judge's prediction entering head B's training at the random positions (N16's distillation reading) is also not separated. A loop-free judge no better than head B has little to pass on, but this is stated as a limit, not ruled out |
| N17b's support margin is reachable | Reply 8, at the boundary: with the peer's F near or above 0, d is unlikely to fall far below −F_lf(x_peer) | Precondition 6b requires F_lf(x_peer) of at least 0.75. Below that, N17b is untestable, not refuted |
| The loop-free line is straight between rungs | Reply 4: an interpolation, two or three points | The calibration chooses rungs that bracket the target values with room. Each rung's F is reported, so a bent line is visible |
| G is fixed by construction | Definitional | Precondition 5. The records give 0.165–0.179 |
| One calibration seed chooses the rungs | Reply 5 (level): a single value | The rule chooses only the rungs. No claim, margin, measure or rule changes, and the real seeds are new runs |
| The tests can fail | Reply 8: every test | N17a fails if the loop-free judges at head B's accuracy recover little, which R1 predicts. N17b fails if the peer does as well as its accuracy allows, which R1 and R2 predict. The refuting margins are where those rivals live |

**The critic's strongest objection, two moves deep:** *"A loop-free judge trained at full weight is ahead early, when the weights shape head B, even if it ends level with the self-judge. N17a matches the wrong thing."*
- *The reply:* the conjecture names the loop as what holds the self-judge back. A loop-free judge being ahead early is then the loop's absence at work, not a confound. S3 shows whether it happened.
- *The critic's answer:* then N17a cannot tell the loop from early accuracy, whatever caused it.
- *The next move:* accepted and recorded. A support of N17a is reported as "a loop-free judge, matched at the end, recovers the shortfall", and if S3 shows the judges ahead early, the note says the course is not separated. N17b carries the same caveat. The peer, self-judged, starts slowly as head B does, so the loop-free judges may lead it early even when the two are compared at the peer's end accuracy. A support of N17b is read with S3 in the same way.

## Calibration before queuing, including a full-length run

- **A full-length calibration run on the A100,** as its own Colab job, on seed 3001 and the calibration windows 513–1024. It trains:
  - `rw-peer`;
  - all five loop-free rungs, `rw-j20`, `rw-j10`, `rw-j05`, `rw-j025` and `rw-j0125`;
  - **the full-length passenger pair:** `rw-b1` retrained twice in N17's code, once plain and once with a head C trained as the peer but not judging.

  It re-measures the saved seed-3001 references: N13's `n80-rw`, N12's `sub20` and `clean`, N15's calibration `rw-oracle`, and N14's `sub10`, `sub05` and `sub025`. It reports:
  - x_self, x_peer and every rung's x and F for seed 3001;
  - whether the two retrains are bit-identical in the trunk, head A and head B at the end;
  - how far they differ from N13's saved seed-3001 record, for the record only. N17's code computes the discount with N15's function, which differs from N13's by up to 4 × 10⁻⁷, and N13 found that such differences in arithmetic can grow over a run;
  - the secondary measures.

  The values for seed 3001 will be seen and recorded in the deviation log.
- **What the calibration may change:** only which rungs the real run trains, by this rule.
  - The real run uses the shortest run of adjacent rungs, at most three, whose accuracies in seed 3001 are ordered by fraction and span both x_self and x_peer with at least 0.03 nats to spare at each end. If 10% and 5% qualify, they are used. A tie goes to the run nearer 10% and 5%.
  - If no such run spans both, the real run uses the shortest run, at most three, that spans x_self by the same criteria. N17b's likely untestability (precondition 6b) is recorded before the real run.
  - If no such run spans x_self, the real run is not queued. The default is to report the calibration's values to the author. Any change to the arms would be an amendment, needing approval.
- **If the two retrains are not bit-identical,** the real run is not queued. The default is to find the bug, fix it and repeat the calibration. Nothing in the design changes.
- **Nothing else changes in response to the calibration:** no claim, margin, measure or rule.
- **Also:**
  - the untrained checks, as in N16;
  - the tiny end-to-end run and the resume test;
  - synthetic records for every branch of the two rules and every precondition.

## Sanity checks before trusting results

- Every data file's fingerprint matches the manifest. The held-out boundary is logged.
- Each new run's fingerprints match N12's `n80`'s. The setup checks are logged.
- Every record names the GPU, the PyTorch version and the code commit. The job runs only on an A100.
- Every number in the report is computed by code from saved JSON records. `n17_analyze.py` is written and committed before any real run, and it is tested on synthetic records covering every branch of the rules.

## Compute, where it runs, and stopping rule

- **Where:** one Colab job, `n17_loop`, on an A100, resumable. It runs after the calibration job, `n17_calib`.
- **Cost:**
  - Training: 15 runs at about 2.5 minutes each (20 if three rungs).
  - 55–60 measurements and 40 model loads.
  - In all, about 60 minutes. The calibration job trains 8 runs and takes about 30 minutes.
- **Stopping rule:** exactly these runs and measurements, each once. Nothing is added after seeing results except as a dated, exploratory entry in the deviation log.

## What each outcome would teach

| N17a | N17b | What we would learn | Next |
|---|---|---|---|
| Supported | Supported | Consistent with the loop in the judge. A loop-free judge as poor as head B recovers most of the shortfall, and a self-judged judge that is not head B does far worse than a loop-free judge of its accuracy. End accuracy (R1) and the judge being head B (R2) are excluded as explanations. What the judge is trained on (R5), its course (R4) and its sharpness (R3) are not separated, so this does not show that the loop alone is the cause. | Separate the loop from what the judge is trained on, which needs a loop-free judge trained on head B's noisy targets: for example a standard head on those targets, its prediction unmixed from u. Or break the loop without a clean head: a floor on the self-judged discount (N16's next conjecture 2) |
| Supported | Intermediate, some effect, inconclusive or untestable | Without the loop, a judge as accurate as head B recovers the shortfall, so end accuracy did not carry N16's result. Whether a self-judged peer does worse than a loop-free judge of its accuracy is unsettled, so R2 is not excluded. | More seeds for the peer, or rungs nearer x_peer |
| Supported | Refuted | What matters is that head B judges itself. Any other judge of its accuracy recovers, loop or not. | Co-teaching proper: two self-judged heads, each judging the other, as a method that needs no clean head |
| Refuted | Refuted | The judge's accuracy carried N16's result. A loop-free judge as poor as head B does no better than head B's own judgement. N16c's verdict stands by its rule, but the attribution N16's note left open goes to accuracy, not self-reference. | Find which part of accuracy matters: the judge's sharpness (S2) or its course (S3) |
| Refuted | Supported | Possible only where the peer is more accurate than head B, since precondition 6b needs the loop-free line high at x_peer. At head B's accuracy no judge helps, loop or not. At the peer's better accuracy, a loop-free judge helps and the peer does not. The loop matters, but only once the judge is accurate enough. Read with S2 and S3. | More rungs between x_peer and x_self |
| Refuted | Intermediate, some effect, inconclusive or untestable | The judge's accuracy carried N16's result, as far as N17a can say. The peer's comparison adds nothing either way. | As for Refuted, Refuted |
| Intermediate, some effect, or inconclusive | Any | Both the loop and accuracy contribute, or the design's limits, as stated. | More seeds, or rungs closer to x_self |
| Untestable (precondition 6a) | Any | Head C's accuracy did not reach head B's, so the matching failed. | Recalibrate the rungs |

## Limitations known now

- **Matched at the end, not over training.** The loop-free judges' accuracy course differs from the self-judge's. S3 reports it, and the audit records how a support is read.
- **Sharpness is not matched, nor what the judge is trained on.** The judge's prediction also enters head B's training at the random positions, and head B's weighted loss trains the trunk the judge reads. None of these is separated, so a support is read as consistent with the loop, not as showing that the loop alone is the cause.
- **The loop-free judges and the oracle need clean labels.** They are diagnostics. Only the peer is a method, and it is secondary (S1).
- **The peer's independence from head B comes from its initial values alone.** It reads the same features and the same targets.
- **Reused seeds and models,** by necessity and declared.
- **Five seeds,** with the detectable effects stated above.
- **One noise rate, one readout width, one scale, 10% of the budget.** The discounts are told the noise.

## Deviation log

*Empty: the preregistration has not been approved.*

## References

Arazo, E., Ortego, D., O'Connor, N. E., & McGuinness, K. (2020). Pseudo-labeling and confirmation bias in deep semi-supervised learning. In *2020 International Joint Conference on Neural Networks (IJCNN)* (pp. 1–8). IEEE. https://doi.org/10.1109/ijcnn48605.2020.9207304

Han, B., Yao, Q., Yu, X., Niu, G., Xu, M., Hu, W., Tsang, I., & Sugiyama, M. (2018). *Co-teaching: Robust training of deep neural networks with extremely noisy labels* [Preprint]. arXiv. https://doi.org/10.48550/arxiv.1804.06872

Malach, E., & Shalev-Shwartz, S. (2017). *Decoupling "when to update" from "how to update"* [Preprint]. arXiv. https://doi.org/10.48550/arxiv.1706.02613
