# Preregistration N15: sharpening the discount

Date: 2026-10-22    Status: **draft, awaiting the author's approval.** Nothing has been built, calibrated or run.

Type: confirmatory for N15a (primary), N15b and N15c, each with its own decision rule. N15a and N15c compare new models with N13's and N12's saved models; N15b compares new models with N14's clean-target ladder. Everything else is secondary or exploratory and labelled so.

Parent documents: `N14_NOTE.md` (next conjecture 1, "Sharpen the discount and see whether geometry follows", and "The reweighting without random targets is a different system"), `N13_NOTE.md` (the discount as the mixture head's learning signal), and `CRITICAL_REVIEW.md`, open problem 2 (what N14 left open).

**Relation to earlier results.**
- **What the discount is, and what it leaves.** N13 showed that the mixture head's advantage is its learning signal: each target y is weighted by r(y) = 0.2 p(y) / (0.2 p(y) + 0.8 u(y)), the head's own posterior that y is clean rather than a draw from the word frequencies u. By the end of training the discount weights a kept target 5.6 times a replaced one, but the replaced targets still hold 0.42 of the summed weight. The discount as a reweighted single softmax (N13's `n80-rw`) ends 0.337 nats from head A on clean targets, reading head A's directions at 0.804. The oracle mask, `sub20`, which drops the replaced targets exactly, ends at 0.163 and 0.896. The shortfall between them, 0.174 nats, is what N15 is about.
- **What N14 found about cost and share.** Along a dose of random targets, the inner reading's cost over `sub20` tracked the random targets' share of the summed weight at the end of training: 0.30 to 0.41 nats per unit of share, 0.41 at the 80% noise rate. Geometry followed function: at matched clean-target loss, the heads with random targets read head A's directions as the clean-target ladder does, within 0.02 (N14a inconclusive by its rule, the point estimates on that side). The ladder, from `sub20` down to `sub0125`, crosses the 0.85 bar at Δ_A = 0.255.
- **What N14's control warned.** The discount's weighting of the kept targets, applied with no random targets, collapsed function (Δ_A 0.736) while keeping geometry (0.772). So the discount has a cost of its own on the kept targets, which the mixture head's random targets may hide; a sharper discount sharpens that weighting too.
- **What is already in hand, and what it predicts.** If the shortfall is the random targets' remaining weight, removing a fraction P of the share removes about the same fraction of the shortfall, as it did along the dose. If the shortfall is the discount's own cost on the kept targets, sharpening removes share without removing shortfall, and may add to it. The two predictions differ by the whole effect, and the clean-target loss is measured to about 0.01 nats, so this part of the test is decisive.
- **What stands regardless.** Every earlier verdict stands. N15 decides nothing about N12's, N13's or N14's claims.

## The conjecture

**The mixture head's shortfall against the oracle mask is the random targets it fails to discount. A sharper discount removes it in proportion to the random weight it removes, and the inner reading's directions follow its function along the clean-target ladder.**

- **The mechanism.** The discount is a posterior under the head's own prediction. Raising the likelihood ratio to a power β > 1 sharpens it: r_β(y) = 1 / (1 + 4 (u(y) / p(y))^β), which is r at β = 1, and tends to a hard decision (p(y) > 4 u(y)) as β grows. A sharper discount gives the random targets less of the gradient at every step after the head's prediction has begun to discriminate, so less of the noise is fitted, and the inner reading moves toward `sub20`'s. The directions it reads move with its function, as N14 found along the dose. A discount computed from head A's prediction, the best posterior the shared state affords, removes share for a different reason: not a sharper rule but a better judge.
- **What it forbids.**
  1. A sharper discount that removes a material part of the random share recovering less than three quarters of the proportional part of the shortfall.
  2. A sharpened head reading head A's directions off the clean-target ladder, by more than the ladder's own spread.
  3. The discount from the clean head's own prediction recovering less than three quarters of its proportional part.
- **The rivals it is built to exclude.**
  - *The discount's own cost.* The uneven weighting of the kept targets costs function (N14's control); sharpening sharpens it, so the shortfall does not fall with the share, or rises. This rival predicts N15a refuted and N15c, where the weighting is not sharper but better judged, supported.
  - *A trace beyond function.* The sharpened heads sit below the ladder. N14 could not resolve a residual under 0.06; three arms pooled are the best this design can do.
  - *The judge, not the rule.* The head's own early predictions are a poor judge, and sharpening a poor judge sharpens its errors as well as its judgements; the oracle, judged by head A, recovers what sharpening cannot. This rival predicts N15c supported and N15a short of it.

## Competing hypotheses

| Hypothesis | N15a: β = 4 against its removed share | N15b: the sharpened heads against the ladder | N15c: the oracle against its removed share |
|---|---|---|---|
| **The conjecture:** the shortfall is the undiscounted random weight; geometry follows function | At least three quarters of the proportional part | On the ladder within 0.04 | At least three quarters |
| The discount's own cost on the kept targets | Less than a quarter, or negative | Any | At least three quarters |
| A trace beyond function | Any | Below the ladder by more than 0.04 | Any |
| The judge, not the rule: sharpening the head's own posterior sharpens its errors | Between a quarter and three quarters | On the ladder | At least three quarters |
| The sharpened head does not train (the discount admits too little early) | Untestable by precondition 5, with the weights' time course recorded | Any | Any |

## The arms

All use N12's model, training and batches at the 80% noise rate, on **N12's seeds 4101–4105**, so that every comparison is paired with the saved models of the same seed: N13's `n80-rw` (the discount at β = 1), N12's `sub20` and `clean`, and N14's ladder. The reuse of seeds and models is declared, as in N13 and N14; no earlier verdict is re-decided.

| Arm | Head B's weight w(y) on each target | Trained for N15? | Role |
|---|---|---|---|
| `rw-b1` | r(y) = 0.2 p(y) / (0.2 p(y) + 0.8 u(y)), the head's own posterior | No: N13's saved `n80-rw` | β = 1, the reference |
| `rw-b2` | r₂(y) = 1 / (1 + 4 (u(y) / p(y))²) | Yes | β = 2 |
| `rw-b4` | r₄(y) = 1 / (1 + 4 (u(y) / p(y))⁴) | Yes | β = 4: the sharpest; N15a's arm |
| `rw-oracle` | 0.2 p_A(y) / (0.2 p_A(y) + 0.8 u(y)), from head A's prediction at the same forward pass, detached | Yes | The best judge the shared state affords; N15c's arm |
| `sub20` | 1 on the kept positions, 0 on the replaced: the oracle mask | No: N12's saved model | The limit of discounting: Δ_A 0.163 |
| `clean`, `sub10`, `sub05`, `sub025`, `sub0125` | — | No: N12's and N14's saved models | The clean-target ladder, with `clean` (every position a clean target, Δ_A 0.000, a 0.976) as a sixth rung at the top |

**Details, fixed now:**
- **The weights** are computed in float32 from the head's current logits without gradient, at the same step, as in N13; the loss is the mean over all positions of w(y) · CE(l, y), N13's convention (every position is targeted at the 80% noise rate). Nothing else differs from `n80-rw`. At β = 1 the formula equals N13's discount exactly, which the setup check confirms on a batch.
- **The oracle** uses head A's softmax at the same forward pass, detached. Head A's loss and gradient do not depend on head B's weights, so head A's own training is unchanged; the trunk receives head B's weighted gradient as in every arm. The setup check confirms that head A's readout parameters receive no gradient from head B's loss. This arm is a diagnostic of the self-referential posterior, not a method: it uses a clean head that exists only because the training has clean targets for head A. It is the peer-judged selection of co-teaching (Han et al., 2018) in the form this project's heads allow.
- **Every run records,** every 50 steps, the mean weight on kept and on replaced positions, the random targets' share of the summed weight, and the fraction of kept positions whose weight is below 0.01 (the targets the discount nearly ignores). The share at the last record (step 2,850) is "the share at the end", as N14 used; the schedule-average share is recorded beside it.
- **The measurement windows** are N12's: the held-out tail's windows 1–512.

## The measures

- **Δ_A and a,** as in N13 and N14: the inner reading's clean-target loss minus head A's, and its agreement with head A, within each model.
- **The shortfall, G** = Δ_A(`rw-b1`) − Δ_A(`sub20`), within seed, from the re-measured models (0.174 on the committed records).
- **The recovered fraction, F(arm)** = (Δ_A(`rw-b1`) − Δ_A(arm)) / G.
- **The removed share, P(arm)** = 1 − s(arm) / s(`rw-b1`), where s is the random targets' share of the summed weight at the end of training (0.42 for `rw-b1` on N13's record).
- **The ladder and the residual,** as in N14: â is the linear interpolation of a at the model's Δ_A on the seed's clean-target ladder, now six rungs from `clean` (Δ_A ≈ 0) to `sub0125`; b = â − a.

## The claims

**N15a (primary): the sharper discount recovers the shortfall in proportion to the share it removes.** For `rw-b4`, the interval for F − 0.75 · P lies wholly above 0.

**N15b: geometry follows function along the sharpness ladder.** The per-seed mean of b over `rw-b2`, `rw-b4` and `rw-oracle` (each where its Δ_A lies within the ladder's range) has its interval wholly within [−0.04, +0.04].

**N15c: a better judge recovers the shortfall in proportion to the share it removes.** For `rw-oracle`, the interval for F − 0.75 · P lies wholly above 0.

## Refuting results

- **N15a** is refuted if the interval for F − 0.25 · P lies wholly below 0, with a standard deviation of at most 0.15: the sharper discount removes share and recovers less than a quarter of the proportional part, or loses ground.
- **N15b** is refuted if the interval for the pooled residual lies wholly outside [−0.04, +0.04], with a standard deviation of at most 0.03.
- **N15c** is refuted as N15a is, for `rw-oracle`.

## Design

- **Runs:** 15 training runs (`rw-b2`, `rw-b4`, `rw-oracle` on seeds 4101–4105); measurements of those 15 models; re-measurements of N13's `n80-rw` (5), N12's `sub20` and `clean` (10) and N14's four lower rungs (20) on the same windows, each checked against its committed record.
- **Saved models** are loaded from the `n13_signal`, `n12_bottleneck` and `n14_dose` Drive folders; their committed records in `results/n13/`, `results/n12/` and `results/n14/` are the reference for precondition 2.
- **Setup checks, recorded in the summary:** the weight identity at β = 1 (r₁ against N13's discount on a batch, relative difference at most 10⁻⁶); the oracle's detachment (the gradient of head B's loss with respect to head A's readout parameters is exactly zero on a batch); the batch and replaced-position fingerprints equal N12's `n80`'s.
- **Measured per model:** Δ_A, a, the rotated control, stability, the zero-noise copy and the constructed mixture check at 0.8, as before; for every new run, the weights' time course.

## Decision rules

**Preconditions:**
1. **Setup.** The batch, same-model, identical-heads and analytic checks pass; the weight identity at β = 1 and the oracle's detachment pass; the windows are shifted correctly; no training draw reaches the held-out tail; each new run's batch and replaced-position fingerprints at step 0 equal N12's `n80`'s; each new run's noise fraction at step 0 is within 0.02 of 0.8.
2. **Must match.** Re-measuring the saved models reproduces their committed records (every loss within 0.001 nats, every agreement ratio within 0.01); the constructed mixture check and the zero-noise copy (≥ 0.999) in every model.
3. **Must differ.** The rotated control's ratio is at most 0.60 in at least 4 of 5 models of each arm.
4. **The manipulation took.** s(`rw-b4`) ≤ 0.75 · s(`rw-b1`) in at least 4 of 5 seeds, for N15a; s(`rw-oracle`) ≤ 0.75 · s(`rw-b1`) in at least 4 of 5, for N15c. A discount that does not remove a quarter of the share has not been sharpened, or has not judged better, and the claim that depends on it is untestable.
5. **The head trained.** Δ_A(`rw-b4`) ≤ 1.0 in at least 4 of 5 seeds, for N15a; the same for `rw-oracle` and N15c. A sharp discount can admit too little early, so that the head never learns; such a head is not a sharpened discount at work, and the claim is untestable, with the weights' time course recorded.
6. **The shortfall exists.** G ≥ 0.10 nats in every seed (the records give 0.155–0.186).
7. **The ladder brackets and is ordered,** as in N14, for each arm entering N15b; N15b is computed over the arm-seed pairs it brackets, and needs at least 4 seeds with at least two arms bracketed each.

Preconditions 1–3 apply to every claim; 4, 5 and 6 to N15a and N15c; 7 to N15b.

**N15a and N15c** each have four outcomes:

| Outcome | Condition |
|---|---|
| Supported | The interval for F − 0.75 · P lies wholly above 0 |
| Refuted | The interval for F − 0.25 · P lies wholly below 0, with a standard deviation of at most 0.15 |
| Some effect, size unclear | The interval for F excludes 0 but neither rule holds |
| Inconclusive | Everything else |

**N15b:**

| Outcome | Condition |
|---|---|
| Supported | The interval for the pooled residual lies wholly within [−0.04, +0.04] |
| Refuted | It lies wholly outside that range, with a standard deviation of at most 0.03 |
| Some effect, size unclear | It excludes 0 but neither rule holds |
| Inconclusive | Everything else |

**Intervals.** All three are interval rules, so each uses a Bonferroni-corrected t-interval at the 1 − 0.05 / 3 level (98.3%), with 4 degrees of freedom (3 if a precondition excludes a seed).

**Why these thresholds.**
- **Three quarters and a quarter of P** bracket proportionality with a quarter's tolerance each way: a discount that recovers at least three quarters of what its removed share predicts is working as the conjecture says; one that recovers less than a quarter is not working through the share.
- **0.04 for the ladder** is the ladder's own spread across seeds (its rungs vary by up to 0.05) and above the measure's noise between implementations (0.02); N14's margin of 0.03 sat at the noise floor, and 0.04 keeps the test possible.
- **The standard-deviation caps** (0.15 for F − 0.25 · P, 0.03 for the residual) are, as before, the condition under which "wholly below" means the effect is absent rather than the seeds scattered.
- **What the design can detect.** Δ_A varies across seeds by about 0.01 nats, so F varies by about 0.06, and a 98.3% interval's half-width is about 0.10. N15a resolves proportionality when P is at least 0.4, where a quarter of P exceeds that half-width; between the precondition's 0.25 and 0.4 a proportional recovery will most likely read "some effect, size unclear". For N15b, the pooled residual's half-width is about 0.03, so "within ±0.04" is reached only if the residual is within about 0.01 of zero; N14's pooled residual (0.013) would read inconclusive here. N15b is kept because a residual beyond 0.04 would be decisive the other way, and because three arms pooled are the best five seeds allow. This is stated so that an inconclusive N15b is read as the design's limit.

## Secondary and exploratory outcomes

- **S1: the cost per unit of share** for each arm, Δ_A(arm) − Δ_A(`sub20`) divided by s(arm), against N14's 0.30–0.41 along the dose.
- **S2: β = 2,** its F and P, and whether F and P fall in the same order across β = 1, 2, 4.
- **S3: the weights over training** in every arm: the kept and replaced means, the share, and the fraction of kept targets nearly ignored, every 50 steps. The conjecture expects the share to fall faster at higher β and the ignored fraction to rise; the rival expects the ignored fraction to carry the cost.
- **S4: the schedule-average share** beside the end share, and F against the removed average share, as an alternative P.
- **S5: the oracle's share against the self-discount's,** step by step: how much better head A judges, and from when.
- **S6: head A's cost** in each arm against `clean` and `rw-b1`, following N13's and N14's exploratory readings.
- **S7: agreement with the copy** at 0.8, N12b's measure, beside the head A measure.

## Necessity audit, at drafting (`NECESSITY_AUDIT.md`)

| Claim the design relies on | Features and ground | The check, and whether it works at the boundary |
|---|---|---|
| A sharper discount removes random share | Functional; present mechanism | Precondition 4 at the end of training. If the share falls early and recovers late, the end share misses it; S4's average share is beside it (reply 5, time) |
| The sharpened head trains at all | Present mechanism (the cold start) | Precondition 5, with S3 showing the admitted weight over the first steps. At the boundary, a head that trains late but fully would pass with a high early ignored fraction, which S3 records |
| Proportionality in P is the right yardstick | Borrowed from N14's dose series, where the discount was the same at every dose | Reply 3: sharpening changes the kept weights as well as the share. The refuting margin (a quarter) is reserved for the strong rival; between the margins the verdict is "some effect", and S3's ignored fraction says which part is the discount's own cost |
| G, the shortfall, is fixed by construction | Definitional: the same seeds' re-measured models | Precondition 6; the records give 0.155–0.186 |
| The oracle does not couple the heads' training | Definitional (detached) | The gradient check in the setup; S6 shows head A's cost, which the trunk's share of head B's gradient can still change, as in every arm |
| The ladder is a null for function without random targets | As in N14, with its control's caveat: the ladder is not a law | N14 found the random-target heads on it and the reweighted control above it. A sharpened head is a reweighted head with random targets; where it sits is the question, and precondition 7 only asks that the ladder reach its loss |
| Five seeds can tell three quarters from a quarter of P | Reply 5 (level); a measured noise | Stated under "What the design can detect": decisive for P ≥ 0.4, likely "some effect" below it |
| Re-measuring the saved models reproduces their records | Present mechanism: deterministic forward passes | Precondition 2; N13 and N14 found the reproduction exact on both A100 models |

The critic's strongest objection, two moves deep: *"Proportionality in the share is borrowed from a dose series where the discount's rule never changed. Sharpening changes the kept weights too, so a recovery short of three quarters shows only that share is not the whole story, not that the discount costs anything."* The reply: that is the rival named, and the design separates the strong form (a recovery under a quarter, refuting) from the rest ("some effect"), with S3's record of the ignored kept targets to say what the rest is. *The critic's answer:* then an intermediate result attributes nothing. *The next move:* accepted; the intermediate outcome is listed as teaching that the share is part of the story and the kept-target weighting the rest, in a proportion the next test must set, and the next test is named: a floor on the discount, so that no kept target is nearly ignored, at the same β.

## Calibration before queuing, including a full-length run

- **A full-length calibration run on the A100,** as its own Colab job, seed 3001: the three new arms, with N13's calibration `n80-rw`, N12's calibration `sub20` and `clean`, and N14's calibration rungs `sub10`, `sub05` and `sub025` re-measured (the calibration ladder has no `sub0125`), on the calibration windows 513–1024. It reports:
  - the shares and the removed fractions P for β = 2, β = 4 and the oracle, and whether each head trained;
  - F for each arm, and F against P;
  - the residuals against the five-rung calibration ladder;
  - the weights' time course, including the ignored fraction.

  The values for seed 3001 will be seen, and recorded in the deviation log.
- **What the calibration may change:** only the sharpest β, by this rule: N15a's arm is the largest β in {2, 4, 8} whose head trains (Δ_A ≤ 1.0) and whose share is at most 0.75 of β = 1's in seed 3001; the other tempered arm is the next lower β (or β = 2). If β = 4 meets the rule, nothing changes. No claim, margin, measure or rule changes in response to the calibration.
- **Also:** the untrained checks as in N12–N14; the tiny end-to-end run and the resume test; synthetic records for every branch of the rules, including an arm that did not train and an arm whose share did not fall.

## Sanity checks before trusting results

- Every data file's fingerprint matches the manifest; the held-out boundary is logged.
- Each new run's fingerprints match N12's `n80`'s; the weight identity at β = 1 and the oracle's detachment are logged.
- Every record names the GPU, the PyTorch version and the code commit. The job runs only on an A100.
- Every number in the report is computed by code from saved JSON records. `n15_analyze.py` is written and committed before any real run, and tested on synthetic records covering every branch of the rules.

## Compute, where it runs, and stopping rule

- **Where:** one Colab job, `n15_sharpen`, on an A100, resumable, after the calibration job `n15_calib`.
- **Cost:** 15 training runs at about 2 minutes each, 50 measurements and 35 model loads: about 45 minutes. The calibration job: about 15 minutes.
- **Stopping rule:** exactly these runs and measurements, each once. Nothing is added after seeing results except as a dated, exploratory entry in the deviation log.

## What each outcome would teach

| N15a | N15b | N15c | What we would learn | Next |
|---|---|---|---|---|
| Supported | Supported | Supported | The mixture head's shortfall is the random weight it fails to discount, and both a sharper rule and a better judge remove it in proportion; geometry follows function here too. The discount is improvable, and the 0.85 bar is reachable by function. | Push the sharpest discount that still trains to the bar, and measure whether agreement crosses 0.85 where the ladder says it should |
| Refuted | Any | Supported | The discount's own weighting of the kept targets costs function; a better judge helps, a sharper rule does not. The mixture head is limited by what it ignores, not only by what it admits. | A floor on the discount at the same β, so that no kept target is nearly ignored |
| Refuted | Any | Refuted | Neither sharpening nor a better judge recovers the shortfall: the shortfall is not the random targets' weight at the end. | Measure the cost over training: where in the schedule the inner reading falls behind `sub20`'s |
| Some effect | Any | Supported | Sharpening recovers part; the rest is the discount's own cost, in a proportion S3 sizes. | The floor on the discount, as above |
| Any | Refuted below the ladder | Any | A sharpened head reads head A's directions worse than its function predicts: a trace that N14 could not resolve, now resolved. | Return to N14's conjecture with this design's power |
| Any | Refuted above the ladder | Any | Sharpened heads read better than their function predicts, as N14's control did: the discount's weighting preserves directions it does not preserve the loss on. | Compare what the ignored kept targets have in common |
| Untestable (4 or 5) | Any | Any | The sharpened discount did not take hold, or did not train; the calibration's choice of β was wrong. | Amend β, calibrate again |

## Limitations known now

- **Reused seeds and models,** by necessity and declared: the reference, the limit and the ladder are saved models.
- **Proportionality in the share is borrowed** from the dose series; the audit says what an intermediate result can and cannot attribute.
- **The oracle is a diagnostic, not a method:** it needs a clean head, which a real noisy-label setting does not have; the question it answers is about the self-referential posterior.
- **Five seeds,** with the detectable effects stated above; N15b in particular will most likely be inconclusive unless the residual is near zero or beyond 0.04.
- **One noise rate, one readout width, one scale, 10% of the budget;** the discount is told the noise.

## Deviation log

(none yet)

## References

Han, B., Yao, Q., Yu, X., Niu, G., Xu, M., Hu, W., Tsang, I., & Sugiyama, M. (2018). *Co-teaching: Robust training of deep neural networks with extremely noisy labels* [Preprint]. arXiv. https://doi.org/10.48550/arxiv.1804.06872

Patrini, G., Rozza, A., Krishna Menon, A., Nock, R., & Qu, L. (2017). Making deep neural networks robust to label noise: A loss correction approach. In *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 2233–2241). IEEE. https://doi.org/10.1109/cvpr.2017.240
