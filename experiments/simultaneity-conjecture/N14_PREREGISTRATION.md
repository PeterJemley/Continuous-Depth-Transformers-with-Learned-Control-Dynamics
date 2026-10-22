# Preregistration N14: a dose of random targets at a fixed number of informative ones

Date: 2026-10-22    Status: **Run** (2026-10-22). Approved by the author, calibrated at full length and amended (A1, a fifth rung) the same day. The results are in `N14_RESULTS.md` and `N14_NOTE.md`; the deviation log records everything that happened after approval.

Type: confirmatory for N14a (primary), N14b and N14c, each with its own decision rule. N14a and N14c compare new models with N12's saved models; N14b uses new models and N12's. Everything else is secondary or exploratory and labelled so.

Parent documents: `N12_NOTE.md` (next conjecture 2, and "What this teaches": "The random targets leave a geometric trace that function does not show"), `N13_NOTE.md` (next conjecture 1), and `CRITICAL_REVIEW.md`, open problem 2.

**Relation to earlier results.**
- **The observation.** In N12, `sub20` (clean targets at exactly the 20% of positions where `n80`'s targets were left alone, nothing elsewhere) and `n80-mix` (the same clean targets, plus random targets at the other 80% of positions, with the mixture head) are functionally close: 80% and 71% of the gap to the ideal copy closed. But `sub20`'s inner softmax reads head A's directions at 0.90 and `n80-mix`'s at 0.81. Within seed, the difference is 0.091, 95% interval [0.052, 0.131], positive in every seed (0.047–0.127). N12's note read this as a geometric trace of the random targets, and conjectured it as the cause of two near misses at the 0.85 bar: N12b (0.79–0.85) and N10c (0.79–0.84, a standard head at 20% noise).
- **What is already in hand, and what it predicts.** Function also moves between the two arms: the inner softmax's clean-target loss minus head A's, Δ_A, is 0.163 for `sub20` and 0.338 for `n80-mix`, a difference of 0.175 [0.166, 0.184]. So the drop in agreement per nat of Δ_A is 0.52. Along a different axis, the number of clean targets, the same ratio appears: from N12's `clean` arm (every position a clean target: agreement 0.976, Δ_A 0.000) to `sub20` (0.896, 0.163), agreement falls 0.080 [0.054, 0.106], which is 0.49 per nat. If the clean-target ladder continued at that slope below 20%, a clean-target head at Δ_A = 0.338 would read head A's directions at about 0.81, which is where `n80-mix` is. On that reading there is no geometric trace of the random targets beyond their functional cost, and "a trace that function does not show" is wrong. N14 measures the ladder below 20% instead of extrapolating it.
- **What N13 settled.** The mixture head's advantage is its learning signal, a discount r(y) on each target by the head's posterior that it is clean; the form matters only at readout. So the random targets act on the mixture head's inner softmax only through the discounted gradient at the random positions, and through the discount's reweighting of the kept positions. N14's arms are built on that.
- **What stands regardless.** Every earlier verdict stands. N14 decides nothing about N12's or N13's claims.

## The conjecture

**Random targets move the directions a head reads away from the clean head's, by more than their cost to the head's clean-target prediction accounts for, and by an amount that grows with their number.** The mechanism conjectured: a random target, drawn from the word frequencies with no regard to context, is fitted by the head *in context*, as an association between that state and that word; the clean-target loss sees only the flattening this causes, while the directions the head reads take up the spurious associations. Fewer clean targets, by contrast, leave the head under-trained but do not point it at anything else.

- **What it forbids.**
  1. A head trained with random targets reading head A's directions as well as a clean-target head with the same clean-target loss does.
  2. The trace of the random targets being complete at the smallest dose, as it would be if it were set in the first steps of training, before the discount discriminates, and did not grow after.
  3. The discount's reweighting of the kept targets, without any random targets, producing the trace.
- **The rival it is built to exclude: geometry follows function.** The agreement ratio measures how head B's prediction depends on the state; a head whose prediction is worse depends on the state less sharply, whatever made it worse. On this view, `n80-mix` reads head A's directions at 0.81 because its inner reading is 0.34 nats from head A, and a clean-target head that is 0.34 nats from head A would read them at 0.81 too. The numbers in hand fit this rival as well as they fit the conjecture. It predicts a residual of zero in N14a.

## Competing hypotheses

| Hypothesis | N14a: `n80-mix` against the ladder at its own Δ_A | N14b: the trace at the smallest dose | N14c: the reweighting alone |
|---|---|---|---|
| **The conjecture:** random targets redirect the reading in context, in proportion to their number | Below the ladder by a material part of the 0.09 drop | Less than half the trace at the largest dose | Within noise of `sub20` |
| Geometry follows function: agreement is set by the clean-target loss, whatever degrades it | On the ladder | Any | On the ladder at its own Δ_A |
| The trace is set early: random targets enter before the discount discriminates, and the dose matters little after | Below the ladder | As large as at the largest dose | Within noise of `sub20` |
| The discount itself is the trace: its uneven weighting of the kept targets moves the directions, and the random targets' own gradient does not | Below the ladder | Any | Far below `sub20` |
| Fewer effective clean targets: the discount lowers the kept targets' weight, which trains the readout less | On the ladder, once Δ_A is matched | Any | Below `sub20` |

The last rival is already weakened by N13: a constant scale on head B's loss changed head B's own readout by nothing (AdamW's normalisation), so a lower weight on the kept targets trains the readout no less. What it leaves is the *uneven* weighting, which N14c isolates.

## The arms

All use N12's model, training and batches, on **N12's seeds 4101–4105**, so that every comparison is paired with N12's saved `sub20` and `n80-mix` of the same seed, which are the two ends of the dose. The reuse of seeds and models is declared, as in N13; no verdict of N12 is re-decided.

Positions in a training batch are of three kinds: **kept** (a clean target), **random** (a target drawn from the word frequencies u) and **masked** (no target). `n80`'s per-step draw fixes which 20% of positions are kept, K, and which 80% are replaced, R, and the random words at R; every N14 arm uses that same draw.

| Arm | Clean targets at | Random targets at | Head B | Trained for N14? | Role |
|---|---|---|---|---|---|
| `sub20` | K (20%) | none | Single softmax | No: N12's saved model | Dose 0; the top of the clean-target ladder |
| `r10` | K | a nested 10% of positions, drawn from R | Mixture at ρ = 1/3 | Yes | Dose 0.1 |
| `r20` | K | a nested 20%, from R | Mixture at ρ = 1/2 | Yes | Dose 0.2 |
| `r40` | K | a nested 40%, from R | Mixture at ρ = 2/3 | Yes | Dose 0.4 |
| `n80-mix` | K | all of R (80%) | Mixture at ρ = 4/5 | No: N12's saved model | Dose 0.8 |
| `sub10` | a nested 10%, drawn from K | none | Single softmax | Yes | The clean-target ladder |
| `sub05` | a nested 5%, from K | none | Single softmax | Yes | The ladder |
| `sub025` | a nested 2.5%, from K | none | Single softmax | Yes | The ladder |
| `sub0125` | a nested 1.25%, from K | none | Single softmax | Yes | The ladder's fifth rung, added by amendment A1 after the calibration run |
| `sub20-rw` | K | none | Single softmax, each kept target's cross-entropy multiplied by stopgrad(r(y)) at ρ = 4/5 | Yes | The reweighting control: the mixture head's weighting of the kept targets, without the random targets |
| `clean` | every position | none | Single softmax | No: N12's records | Head A's reference for the cost measure only |

**Details, fixed now:**
- **Nesting.** For each step, one seeded uniform draw per position (seeded by the seed and the step, independent of `n80`'s draw) orders the positions of R; dose d takes the positions of R whose draw is below d / 0.8, so the random sets are nested: `r10` ⊂ `r20` ⊂ `r40` ⊂ `n80-mix`. A second draw orders K; `sub` at fraction f takes the positions of K below f / 0.2, so `sub0125` ⊂ `sub025` ⊂ `sub05` ⊂ `sub10` ⊂ `sub20` (`sub0125` by amendment A1). The random words at the random positions are `n80`'s.
- **Each arm's ρ is its true noise rate among its targeted positions,** d / (0.2 + d): the head is told the noise, as in N12 and N13. Its mixture output is (1 − ρ) softmax + ρ u, its loss −log q(y), and its discount r(y) = (1 − ρ) p(y) / q(y).
- **The loss at every arm is the sum over the targeted positions divided by the number of all positions,** which is N12's convention for `sub20` and, since every position is targeted there, for `n80-mix`. So head B's share of the trunk's gradient scales with the number of targets, as it did between those two arms. N13 showed that a constant scale changes nothing in head B's own readout.
- **`sub20-rw`:** at each kept position, r(y) is computed at ρ = 4/5 from the head's own logits in float32 without gradient, as in N13's `n80-rw`, and multiplies that position's cross-entropy. There are no random targets, so this is the mixture head's weighting of the kept targets applied to `sub20`'s training. Its r(y) is computed from a head trained without random targets, so its weighting is sharper than `n80-mix`'s on the same positions; the recorded discount means (below) will show by how much, and this is a stated limitation.
- **Every run records,** every 50 steps, the mean of its discount r(y) over the batch, over the kept positions and over the random positions, and from them the random targets' share of the summed weight, d · r̄_random / (d · r̄_random + 0.2 · r̄_kept). For the ladder arms, r(y) at ρ = 4/5 is computed and recorded only.
- **The measurement windows** are N12's: the held-out tail's windows 1–512. N14 must use them, because its comparisons are with N12's models measured there.

## The measures

- **Agreement with head A, a:** N12's geometric measure within each model, the agreement ratio of head B's inner softmax (for a single softmax, head B itself) with head A, on the measurement windows. N12b used the agreement with the ideal copy instead; the copy's directions are head A's (N12 found their agreement 0.99), and the two measures differ by at most 0.02 on N12's models (`n80-mix` 0.826 against 0.805, `sub20` 0.911 against 0.896). N14 uses the head A measure, as N13 did, and records the copy measure beside it.
- **The inner reading against head A, Δ_A:** the inner softmax's clean-target loss minus head A's, within the model, as in N13.
- **The ladder prediction, â:** for each seed, the clean-target ladder is the points (Δ_A, a) of `sub20`, `sub10`, `sub05`, `sub025` and, by amendment A1, `sub0125`, in order of Δ_A. For a model with Δ_A inside the ladder's range, â is the linear interpolation of a between the two ladder points that bracket its Δ_A. This asks: how well does a clean-target head with the same clean-target loss read head A's directions?
- **The residual, b (for "beyond"):** b = â − a, how far a model's agreement falls below a clean-target head's at the same Δ_A. Positive means the random targets moved the directions more than their functional cost accounts for.
- **The drop from dose 0, D(d) = a(`sub20`) − a(d),** within seed.

## The claims

**N14a (primary): the trace is beyond function.** At dose 0.8, the residual is a material part of the 0.09 drop: the interval for b(`n80-mix`) lies wholly above 0.03.

**N14b: the trace grows with the dose.** At the smallest dose it is less than half of what it is at the largest: D(`r10`) < D(`n80-mix`) / 2 in at least 4 of 5 seeds.

**N14c: the discount's reweighting of the kept targets is not the trace.** The interval for a(`sub20`) − a(`sub20-rw`) lies wholly below 0.03.

## Refuting results

- **N14a** is refuted if the interval for b(`n80-mix`) lies wholly below 0.01, with a standard deviation of at most 0.03: a clean-target head at the same clean-target loss reads head A's directions no better, and geometry followed function.
- **N14b** is refuted if D(`r10`) ≥ 0.75 · D(`n80-mix`) in at least 4 of 5 seeds: the trace is complete at the smallest dose.
- **N14c** is refuted if the interval for a(`sub20`) − a(`sub20-rw`) lies wholly above 0.06, two thirds of the known trace, with a standard deviation of at most 0.03: the reweighting alone produces most of it.

## Design

- **Runs:** 40 training runs (`r10`, `r20`, `r40`, `sub10`, `sub05`, `sub025`, `sub0125` by amendment A1, `sub20-rw` on seeds 4101–4105), measurements of those 40 models, and re-measurements of N12's `sub20` and `n80-mix` (10 models) on the same windows. `clean`'s head A loss is taken from N12's records.
- **N12's models** are loaded from the `n12_bottleneck` Drive folder; N12's measurement records in `results/n12/` are the reference for precondition 2.
- **Measured per model:** head A's and the inner softmax's clean-target losses and Δ_A; the agreement with head A and with the copy at the arm's ρ; the rotated control; stability; the constructed mixture check at the arm's ρ; for every new model, the fingerprints of its kept and random position sets at step 0 and a nesting check against `n80`'s sets.

## Decision rules

**Preconditions:**
1. **Setup.** The batch, same-model, identical-heads and analytic checks pass; the windows are shifted correctly; no training draw reaches the held-out tail; each new run's batch fingerprint at step 0 equals N12's `n80`'s of the same seed; each arm's kept set is a subset of `n80`'s K (equal to K for the dose arms and `sub20-rw`), its random set a subset of R, and the nested sets are nested, all checked at step 0; each arm's fractions of kept and random positions at step 0 are within 0.02 of their design values.
2. **Must match.**
   - Re-measuring N12's saved `sub20` and `n80-mix` reproduces N12's records: every loss within 0.001 nats, every agreement ratio within 0.01.
   - The constructed mixture check passes in every model, at the model's own ρ.
   - The zero-noise copy matches head A at a ratio of at least 0.999 in every model.
3. **Must differ.** The rotated control's ratio is at most 0.60 in at least 4 of 5 models of each arm.
4. **The ladder brackets.** In at least 4 of 5 seeds, Δ_A(`sub025`) ≥ Δ_A(`n80-mix`), so that â(`n80-mix`) is an interpolation, never an extrapolation. N14a is computed over the seeds where this holds; if fewer than 4, N14a is untestable.
5. **The ladder is ordered.** In at least 4 of 5 seeds, Δ_A increases along `sub20`, `sub10`, `sub05`, `sub025` and, by amendment A1, `sub0125`. If it does not, the ladder is not a function of the clean-target loss, and N14a is untestable.
6. **The trace exists.** D(`n80-mix`) ≥ 0.02 in at least 4 of 5 seeds, on the re-measured models. N12 recorded 0.047–0.127. If it does not hold, N14b's ratios are undefined, and N14b is untestable.

Preconditions 1–3 apply to every claim; 4 and 5 to N14a; 6 to N14b.

**N14a and N14c** each have four outcomes:

| Outcome | Condition |
|---|---|
| Supported | The 97.5% interval lies wholly on the claim's side of its margin: above 0.03 for N14a; below 0.03 for N14c |
| Refuted | It lies wholly past the refuting margin (below 0.01 for N14a; above 0.06 for N14c), with a standard deviation of at most 0.03 |
| Some effect, size unclear | It excludes 0 but meets neither rule |
| Inconclusive | Everything else |

**N14b:**

| Outcome | Condition |
|---|---|
| Supported | D(`r10`) < D(`n80-mix`) / 2 in at least 4 of 5 seeds |
| Refuted | D(`r10`) ≥ 0.75 · D(`n80-mix`) in at least 4 of 5 seeds |
| Intermediate | Anything else |

**Intervals.** N14a and N14c are interval rules, so each uses a Bonferroni-corrected 97.5% t-interval, with 4 degrees of freedom (3 if precondition 4 excludes a seed). N14b is a count rule.

**Why these thresholds.**
- **0.03 in agreement** is the project's noise scale for the measure: N13 found the agreement differs by up to 0.02 between two implementations of the same mathematics, and N12's arms had standard deviations across seeds of 0.02–0.03. A residual of 0.03 is a third of the known 0.09 drop. **0.01** is "none, within the measure's noise".
- **Half and three quarters** of the trace for N14b: with nested doses of 0.1 and 0.8, a trace that grows with the dose puts the smallest dose well under half; one set early puts it near the whole.
- **0.06** for N14c is two thirds of the known trace: a reweighting that produces that much is the trace's main part.
- **What the design can detect.** With five seeds and a standard deviation of 0.02–0.03 in the residual, the 97.5% interval's half-width is 0.03–0.05, so N14a is supported only if the residual is about 0.06–0.08 or more, most of the known drop. A residual between 0.03 and 0.06 will most likely come out "some effect, size unclear". This is stated now so that such a result is read as the design's limit, not as evidence against the conjecture. More seeds would need their own `sub20` and `n80-mix`; the choice to stay with N12's five is a cost decision.

## Secondary and exploratory outcomes

- **S1: the residuals at the other doses,** b(`r10`), b(`r20`), b(`r40`), and b(`sub20-rw`), each with its interval. The conjecture predicts them positive and growing with the dose; geometry-follows-function predicts them near zero.
- **S2: the ladder's own noise.** For each inner ladder point (`sub10`, `sub05`, and `sub025` by amendment A1), its leave-one-out residual against the line through its neighbours. This is the scale of the interpolation's error, and the test's noise floor; if it is near 0.03, N14a's margin is at the floor, and the note will say so.
- **S3: the ordering of agreement with the dose,** `sub20` ≥ `r10` ≥ `r20` ≥ `r40` ≥ `n80-mix` within seed: the number of seeds in which it holds in full, and D(d) at every dose with intervals.
- **S4: Δ_A by dose,** and by ladder fraction: the functional cost of the random targets and of the missing clean targets.
- **S5: the cost to head A** in every arm against N12's `clean`, following N13's finding that the cost tracks head B's pull. Here the pull grows with the number of targets.
- **S6: the discount over training** in every arm: the kept and random means and the random targets' share of the summed weight, every 50 steps. The conjecture's mechanism expects the trace to track that share, which the dose arms vary.
- **S7: agreement with the copy** at each arm's ρ, N12b's measure, beside the head A measure.

## Necessity audit, at drafting (`NECESSITY_AUDIT.md`)

| Claim the design relies on | Features and ground | The check, and whether it works at the boundary |
|---|---|---|
| "Beyond function" presupposes a clean-target head with the same clean-target loss to compare with | Functional; definitional (the ladder) | Preconditions 4 and 5. At the boundary, a ladder that brackets `n80-mix` only in its last segment makes â an interpolation over a wide span; S2 measures the error that costs |
| The ladder is a fair null: fewer clean targets degrade function without pointing the head elsewhere | Reply 3 (another mechanism); a present mechanism | This is the test's premise, not its finding. The modest conclusion (reply 4): N14a compares two ways of raising the clean-target loss, and "a trace that function does not show" is a gloss on that contrast. The note will state the contrast |
| Interpolating a on Δ_A presupposes a is a function of Δ_A along the ladder | Definitional; a present mechanism | Precondition 5 (ordered), S2 (noise). A ladder whose agreement is not monotone in Δ_A would be its own finding, and N14a would be untestable |
| The dose arms' random sets are nested and are `n80`'s positions and words | Definitional (construction) | Fingerprints and the nesting check at step 0; the fractions within 0.02 of design |
| Each arm is told its true noise rate | Definitional | ρ = d / (0.2 + d) by construction; the noise fraction at step 0 is recorded |
| A trace that grows with the dose shows itself between doses 0.1 and 0.8 | Reply 5 (level) | If the trace saturates below 0.1, N14b reads as "complete at the smallest dose" and the conjecture's growth clause is refuted at the only levels tested; the note will say which |
| `sub20-rw` isolates the discount's reweighting of the kept targets | Functional; present mechanism | Its r(y) comes from a head trained without random targets, so its weighting is sharper than `n80-mix`'s; S6 records both. A refutation of N14c holds for the sharper weighting; a support holds only for it, and is a weaker exclusion |
| Five seeds can tell "most of the drop" from "none of it" | Reply 5 (level); a measured noise | Stated under "What the design can detect": residuals between 0.03 and 0.06 fall in the gap, and the rule names that outcome |
| Re-measuring N12's models reproduces N12 | Present mechanism: deterministic forward passes on the same GPU type | Precondition 2; the job runs only on an A100, and N13 found the reproduction exact |

The critic's strongest objection, two moves deep: *"The ladder is not a null. Removing clean targets and adding random ones are different manipulations, so equal clean-target loss gives no licence to expect equal agreement, and a residual proves nothing about 'function'."* The reply: the claim is stated as that contrast, and it is the contrast that the two near misses call for: if a head with random targets reads head A's directions worse than a clean-target head at the same loss, then the 0.85 bar asks for something the loss does not supply, and the near misses have a cause; if not, the bar is a functional bar in disguise. *The critic's answer:* then say "worse than a clean-target head at the same loss", and drop "function does not show". *The next move:* accepted; the note will use the contrast, and the gloss only with it.

## Calibration before queuing, including a full-length run

- **A full-length calibration run on the A100,** as its own Colab job, seed 3001: all seven new arms, with N12's calibration models `sub20` and `n80-mix` of seed 3001 (from the `n12_calib` Drive folder) re-measured, on the calibration windows 513–1024. It reports:
  - whether the ladder brackets `n80-mix`'s Δ_A, and the ladder's shape;
  - the residual b(`n80-mix`) for one seed, and the residuals at the other doses;
  - D at every dose, and a(`sub20`) − a(`sub20-rw`);
  - the discount's means and the random targets' share of the weight in every arm.

  The values for seed 3001 will be seen, and recorded in the deviation log.
- **What the calibration may change:** only the ladder, by adding a lower fraction (1.25%) if `sub025` does not bracket `n80-mix` in seed 3001, through an amendment that comes back for approval first. No claim, margin, dose, window or measure changes in response to it.
- **Also:** the untrained checks as in N12 and N13; the tiny end-to-end run and the resume test; synthetic records for every branch of the rules, including unbracketed and unordered ladders.

## Sanity checks before trusting results

- Every data file's fingerprint matches the manifest; the held-out boundary is logged.
- Each new run's batch fingerprint matches N12's `n80`'s; the position sets' fingerprints, fractions and nesting are logged at step 0.
- Every record names the GPU, the PyTorch version and the code commit. The job runs only on an A100.
- Every number in the report is computed by code from saved JSON records. `n14_analyze.py` is written and committed before any real run, and tested on synthetic records covering every branch of the rules.

## Compute, where it runs, and stopping rule

- **Where:** one Colab job, `n14_dose`, on an A100, resumable, after the calibration job `n14_calib`.
- **Cost:** 40 training runs at about 2 minutes each (35 before amendment A1), 50 measurements: about 90 minutes. The calibration job: about 20 minutes.
- **Stopping rule:** exactly these runs and measurements, each once. Nothing is added after seeing results except as a dated, exploratory entry in the deviation log.

## What each outcome would teach

| N14a | N14b | N14c | What we would learn | Next |
|---|---|---|---|---|
| Supported | Supported | Supported | Random targets point the head at spurious associations, in proportion to their number, beyond their cost to the clean-target prediction and not through the discount's weighting of the kept targets. The 0.85 bar asks for something the loss does not supply, and the near misses of N10c and N12b have a cause. | Whether the associations form early: mask the random positions until the discount discriminates (about step 300), and see whether the trace shrinks |
| Refuted | Any | Any | Geometry follows function here: a clean-target head at the same clean-target loss reads head A's directions no better. The 0.85 bar is a functional bar in disguise, and N12b's shortfall is the discount's incomplete suppression of the random targets (Δ_A 0.34 against 0.16). | Sharpen the discount (a tempered posterior, or a floor on u) and see whether function and geometry improve together |
| Supported | Refuted | Any | The trace is set early, before the discount discriminates, and the dose matters little after. | The masking schedule above, as the direct test |
| Supported | Any | Refuted | The discount's own uneven weighting of the kept targets moves the directions; the random targets' gradient is not the trace, or not alone. The mixture head's geometric shortfall is the price of its discount. | A discount with a floor, so that no kept target is nearly ignored, against the same dose |
| Some effect | Any | Any | A residual between the noise and the margin: the design's stated limit. | More seeds with their own `sub20` and `n80-mix`, if the question is worth them |
| Untestable (4 or 5) | Any | Any | The ladder does not reach or does not order `n80-mix`'s clean-target loss; the comparison was not available at these fractions. | Amend the ladder and rerun its arms only |

## Limitations known now

- **Reused seeds and models,** by necessity and declared: the two ends of the dose are N12's models.
- **The ladder is one way of raising the clean-target loss without random targets.** Another (a weaker readout, a shorter schedule) might give a different curve; N14a is relative to this ladder.
- **The agreement measure is symmetric** between the heads: it cannot say whether head B's directions moved or head A's. Head A's cost (S5) bears on that only functionally.
- **`sub20-rw`'s weighting is sharper than the mixture head's,** as stated under the arms.
- **One noise distribution (u), one readout width, one scale, 10% of the budget;** the discounting head is told the noise.
- **Five seeds,** with the detectable residual stated above.

## Deviation log

### Calibration before any real run, 2026-10-22 (no amendment)

Results seen before this entry:
- the setup checks on an untrained full-size model with real training batches;
- test-mode runs on tiny models: an end-to-end run with its rerun, and the resume test;
- synthetic records.

The full-length calibration run on the A100 has not started, and no real N14 run has started.

- **The loss identity at the dose's ends, at full size.** On an untrained full-size model with a real training batch, the job's loss for the arm with every replaced position random (dose 0.8, clean 20%) equalled N12's `n80-mix` loss to a relative 1.2 × 10⁻⁷, and its loss for the arm with no random targets (dose 0, clean 20%) equalled N12's `sub20` loss exactly. The check's threshold is 10⁻⁶. It runs at the start of every job and is recorded in its summary.
- **The position sets, at full size.** For seeds 4101 and 3001 at step 0: every arm's kept set is a subset of `n80`'s K and its random set a subset of its R; the dose sets and the ladder sets are nested; the ends reproduce K and R exactly; the fractions are within 0.005 of design (for seed 4101: kept 0.196, random 0.096, 0.195 and 0.390 at doses 0.1, 0.2 and 0.4; clean 0.101, 0.052 and 0.024 on the ladder). The same check runs for every seed at the start of every job, and each run records its own sets' fingerprints, which the analysis compares with the check's.
- **The discount at initialisation,** as the N13 correction says: equal on kept and random targets (0.248 against 0.265 at ρ = 1/3; 0.187 against 0.198 at ρ = 1/2; 0.134 against 0.137 at ρ = 2/3), so the random targets' share of the summed weight at the start is the arm's ρ (0.34, 0.51, 0.67). S6 will show how the share falls as the discount learns to discriminate.
- **Tiny test-mode runs** (2 seeds, 60 steps, 8 measurement windows, on a CPU in float32):
  - *End to end:* the 7 new arms trained and were measured, and N12's two ends were re-measured: 33 records and the summary. A rerun skipped all finished work.
  - *Re-measuring N12's models reproduced N12's records exactly:* every loss and every ratio differed by 0.
  - *The analysis* produced every section from these records: the ladder, the interpolation and residuals, the six preconditions, the by-seed and by-arm tables, S1–S7, and the verdicts read "not reached" with two seeds, as they must.
  - *The resume test:* the calibration mode was interrupted at step 28 of 60 of its first training run. A second session, with its local folder deleted as on a new Colab machine, resumed from the Drive copy at step 28, and was itself cut off at step 44 by a restart of the test machine. A third session, its local folder deleted again, resumed at step 44 and finished all seven runs and the measurements. The calibration-mode analysis of its records produced every table, with "n/a" where a tiny arm's Δ_A fell outside its ladder's range, as designed.
  - *The tiny models' numbers mean nothing* (their ladders are not even ordered in one seed), and none is carried forward.
- **Synthetic records.** 24 cases covering every branch of the three rules and every precondition gave the intended verdicts, including an unbracketed ladder, an unordered ladder, a trace absent at dose 0.8, and mismatched set fingerprints.
- **The Colab wrappers** run only on an A100, tested with simulated GPU names: a T4 and an L4 are refused before anything is trained, both A100 variants pass, and the calibration wrapper adds `--calibration`.
- **Implementation details, not changes:**
  - The nesting draw is one uniform number per position per step, seeded by the seed and the step, independent of `n80`'s replacement draw; an arm takes the positions of K (or R) whose number is below its fraction's share of K (or R).
  - Each arm's validation curve uses noisy targets at the arm's own ρ (0.8 for the arms without random targets, as N12's `sub20` did).
  - `n12.measure` records every model in N12's form (its copy at 0.8), so that N12's two ends reproduce their records; the arm's own ρ enters through the output form (`mix:ρ`) and the `at_rho` block, which holds the copy, the constructed check and the agreement with the copy at that ρ.

The values for seed 3001 from the calibration run are recorded in the next entry.

### The calibration run, before any real run, 2026-10-22, and amendment A1 (proposed, awaiting the author's approval)

**The session:** `colab-results` branch `n14_calib/20261022-113912`, code commit fae35f5, on an NVIDIA A100-SXM4-80GB with PyTorch 2.11. It ran in 16 minutes, about 2 minutes per training run, with no resumes and no Drive failures.

**What it was:** seed 3001: the seven new arms trained on the full schedule, and N12's `sub20` and `n80-mix` of seed 3001 re-measured, all on the tail's windows 513–1024.

**The report:** `N14_CALIBRATION.md`, generated by `n14_analyze.py` from `results/n14_calib/` and `results/n12_calib/`.

**What it showed, recorded as the preregistration said it would be. The values decide nothing.**
- **Preconditions 1–6 held in this seed:** the loss identity at the ends (6.3 × 10⁻⁸ and 0), the nested sets, every fingerprint; the re-measurements reproduced N12's calibration records exactly (every loss and ratio differed by 0); the constructed checks at 0.8 and at each arm's ρ; the zero-noise copy; the rotated control 0.13–0.15; the ladder brackets and orders `n80-mix`; D(`n80-mix`) = 0.101.
- **The ladder** (Δ_A, a): `sub20` (0.167, 0.891), `sub10` (0.276, 0.846), `sub05` (0.426, 0.767), `sub025` (0.611, 0.724). Its slope is −0.41, −0.53 and −0.23 per nat along the three segments; its leave-one-out residuals are −0.008 and +0.025, so the interpolation's noise is about 0.02–0.03, at N14a's margin, as S2 was written to show.
- **The dose arms** (Δ_A, a, b): `r10` (0.211, 0.838, +0.035), `r20` (0.239, 0.854, +0.008), `r40` (0.277, 0.814, +0.032), `n80-mix` (0.352, 0.790, +0.016). `n80-mix` sits 0.016 below the ladder at its own Δ_A, a sixth of its 0.101 drop from `sub20`; the rest of the drop is where a clean-target head with the same clean-target loss would be. The dose is not monotone in a (`r20` above `r10`), and D(`r10`) / D(`n80-mix`) = 0.52, at N14b's half.
- **The control `sub20-rw`** (Δ_A 0.805, a 0.771): the discount at ρ = 0.8 on a head trained with no random targets made its inner reading far worse than `sub20`'s (0.805 against 0.167 nats from head A) and worse than any dose arm's; a(`sub20`) − a(`sub20-rw`) = 0.120, more than the whole trace. Its Δ_A lies beyond the ladder's top (0.611), so its residual cannot be computed. The preregistration's limitation said this control's weighting would be sharper than the mixture head's; the calibration shows the difference is larger than that: without random targets to pull the head toward u, the discount lets the head neglect the targets it predicts poorly, and the head's function collapses. The preregistered N14c compares agreements directly, so it will read this control's geometric drop as the reweighting's, whatever its function. That rule stands as approved; the residual against the ladder, which asks whether the control's drop is beyond its function, is the reading the ladder was built for, and it needs a rung below 0.805.
- **The discount over training** (kept / random / the random share of the weight): `r10` 0.246 / 0.274 / 0.36 at step 0 to 0.854 / 0.272 / 0.14 at the end; `r20` 0.185 / 0.200 / 0.52 to 0.777 / 0.204 / 0.20; `r40` 0.133 / 0.142 / 0.68 to 0.690 / 0.145 / 0.28. The share starts at the arm's ρ and falls as the discount discriminates, to about a third of it.
- **Head A against N12's `clean` arm** (S5, exploratory): `sub20` +0.031, `r10` +0.021, `r20` +0.023, `r40` +0.014, `n80-mix` +0.001; `sub10` +0.021, `sub05` +0.006, `sub025` +0.002; `sub20-rw` −0.004. The cost falls with the dose of random targets and with fewer clean targets alike; it is the `sub20` puzzle of N13 again, with more points, and it is left for the note.

**Amendment A1, proposed: a fifth rung on the ladder, `sub0125`** (clean targets at a nested 1.25% of positions, drawn from K as the other rungs are; single softmax; 5 runs, about 10 minutes). Its purpose is to bracket `sub20-rw` (Δ_A 0.805) and any arm whose Δ_A exceeds `sub025`'s, so that the residual against the ladder (S1) can be computed for the control, as a secondary measure. Nothing else changes: no claim, margin, dose, window, measure or rule; precondition 5 (the ladder is ordered) covers the new rung, and N14a's interpolation at `n80-mix`'s Δ_A, which lies between `sub10` and `sub05`, is unaffected. The preregistration's calibration clause allowed a rung to be added only if `sub025` failed to bracket `n80-mix`, which did not happen; so this is an amendment, and it waits for the author.

**A1 approved by the author, 2026-10-22,** with the real run queued. The rung is added to the job, the analysis and the synthetic cases; the arms table, the ladder's definition, precondition 5, S2 and the cost above carry it. The new rung was not calibrated at full length: its clean-target loss is expected near 0.8 nats from head A, close to the control's 0.805, so it may bracket the control in some seeds and not others; the control's residual (S1) is reported for the seeds it brackets, and nothing else depends on it. Tests after the change: the 25 synthetic cases (one added, a collapsed control bracketed by the new rung) gave the intended verdicts; the full-size nesting check passed with the rung (fractions 0.0129 and 0.0127 in seeds 4101 and 3001); the tiny end-to-end run, resumed, trained and measured the new rung alone (every other run skipped as finished), and the analysis read the five-rung ladder with three leave-one-out residuals.

### The run, 2026-10-22

- **Session:** Colab run `n14_dose/20261022-122325`, code commit 8b4fb43 (the code of A1, c0619da, plus the deviation log), on an NVIDIA A100-SXM4-40GB with PyTorch 2.11. It trained the 40 models, re-measured N12's `sub20` and `n80-mix` and measured everything in 93 minutes, with no resumes, no Drive failures and no 16-bit steps skipped. The records, log and run information are committed under `results/n14/`.
- **What changed after approval:** amendment A1, before the run. Nothing after it.
- **Results, under the rules:**
  - All six preconditions held. The re-measurements reproduced N12's records exactly (every loss and ratio differed by 0), on the 40 GB model. The fifth rung bracketed `sub20-rw` in every seed.
  - N14a inconclusive: b(`n80-mix`) = +0.009, 97.5% interval [−0.035, +0.052], standard deviation 0.028.
  - N14b intermediate: D(`r10`) < D(`n80-mix`) / 2 in 2 of 5 seeds; at least three quarters in 0.
  - N14c refuted: a(`sub20`) − a(`sub20-rw`) = +0.125 [+0.089, +0.160], standard deviation 0.028.
  - The conjecture did not survive.

### Exploratory analysis added after the results, 2026-10-22

- **What was added:** `n14_exploratory.py` (report `N14_EXPLORATORY.md`): the ladder's slopes and the clean-target loss at which it crosses 0.85; the residuals pooled over the four doses; the trace and the functional cost at each dose against the random targets' share of the gradient weight; the control taken apart; head A's cost against the `clean` arm by arm.
- **Why:** to read N14a's inconclusive verdict (the residuals at the other doses and their pooled value), to give N12b's near miss its functional measure, and to read N14c's refuted control against the ladder.
- **Its status:** written after the results were seen. It decides nothing, and the preregistered analysis is unchanged.

## References

Patrini, G., Rozza, A., Krishna Menon, A., Nock, R., & Qu, L. (2017). Making deep neural networks robust to label noise: A loss correction approach. In *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 2233–2241). IEEE. https://doi.org/10.1109/cvpr.2017.240
