# Preregistration N5: is N4's cost a competition for directions? A readout-width sweep

Date: 2026-10-06    Status: **approved by the author, 2026-10-06.** Queued; no real N5 run has started.

Type: confirmatory for two hypotheses, N5a and N5b, each with its own decision rule and a Bonferroni-corrected interval. Everything else is secondary or exploratory and labelled so.

Parent documents: `N4_NOTE.md` (next conjecture 1), `N4_PREREGISTRATION.md`, and `PATHS_AND_UNDERSTANDING.md`.

**Relation to earlier results.**
- **What N4 found.** Forcing two different readings (the next token, and the token after next) through one shared set of 64 readout directions cost 0.088 nats more than the same restriction cost two heads reading the same target. 95% interval [+0.078, +0.097].
- **The explanation N4 offered.** The two readings compete for directions. A shared 64-direction readout cannot hold what both readings need, while two separate readouts can.
- **A rival N4 could not exclude.** Tying also makes both heads' gradients meet in one shared matrix, where different targets can conflict (Yu et al., 2020). That would cost something even if directions were plentiful. N4's note named this as an inferred, unmeasured mechanism.
- **What N5 does.** It tests N4's explanation by changing only the readout width. That is the one variable the two explanations disagree about.
- **What stands regardless.** N4's result as stated, that sharing 64 readout directions costs two readings more than one, stands whatever N5 shows. So do all earlier results.

## The claims

The explanation: **the cost of tying is a competition for directions.** It has two consequences, tested separately.

**N5a (primary): remove the competition, and the cost vanishes.**
- At width 256, the full width of the state, the tied readout is a full-rank 256 × 256 matrix. Each head's own output matrix can undo it, so tied and separate readouts can represent exactly the same functions.
- If the cost is a competition for directions, it should be gone: D at width 256 near zero.
- If it is gradient conflict in a shared matrix, it should persist.

**N5b: tighten the competition, and the cost grows.** At width 32, D is at least 0.05 nats larger than at width 128.

### The explanation behind it

- **Each reading needs some set of directions in the final state, and the two sets only partly overlap.** N2 and N4 both found the readings' sensitivity subspaces partly distinct, while two heads on the same target converge.
- **A tied readout of width w must hold both readings' needs in w directions.** Separate readouts give each reading its own w.
- **Diminishing returns.** Each extra direction helps a reading less than the one before. So squeezing two readings into w directions costs more the smaller w is, and costs nothing once w covers what both need.
- **At w = 256 nothing is squeezed.** The readout spans the whole state.
- **Prior work.** Restricting a language model's readout to fewer directions limits its accuracy (Yang et al., 2017). Many features can share a space as directions, at the price of interference (Elhage et al., 2022).

### Why this is hard to vary

The explanation fixes the sign of both predictions. It forbids a cost at full width, and it forbids the cost shrinking as the readout tightens. The rival, gradient conflict, predicts a cost at every width, including 256. A third possibility is a cost that peaks at an intermediate width. At very tight widths, both readings might fall back on the same few dominant directions, such as those for frequent tokens, so that their differences no longer matter. That would refute N5b while leaving N5a open.

## Competing hypotheses

| Hypothesis | D at 256 (N5a) | D at 32 − D at 128 (N5b) |
|---|---|---|
| The claim: a competition for directions | Near zero | ≥ 0.05 |
| Gradient conflict in the shared matrix | Persists, about N4's size | Roughly flat |
| Both mechanisms | Smaller than at 64, not zero | Positive |
| The cost peaks at intermediate width | Near zero | Below 0.05, or negative |
| Noise, or a bug | Inconsistent across seeds | Inconsistent |

## Refuting results

- **N5a** is refuted if the 97.5% interval for D at width 256 lies wholly above +0.05 nats.
- **N5b** is refuted if the 97.5% interval for D(32) − D(128) lies wholly below +0.05 nats, with a standard deviation of at most 0.04.

## Design

### What changes from N4

Only the readout width w, the number of directions each projection P_k reads. Everything else is N4's design:
- the paper's `HybridODETransformer`, exact gradients, no control, two heads reading the same final state;
- logits_k = W_k (P_k · ln_f(z)), with P_k a w × 256 matrix and W_k a 50,257 × w matrix;
- the four arms below, with P_B initialised as a copy of P_A, so separate and tied are bit-identical at step 0;
- the same training, 2,879 steps at peak learning rate 2.4e-3, batch 32 × 128, on the pinned WikiText-103 token files.

| Arm | Head A target | Head B target | Readout |
|---|---|---|---|
| dual-sep | i + 1 | i + 2 | separate |
| dual-tied | i + 1 | i + 2 | tied |
| same-sep | i + 1 | i + 1 | separate |
| same-tied | i + 1 | i + 1 | tied |

### Widths and seeds

- **New widths:** 32, 128 and 256, all four arms each.
- **Seeds:** 3501 to 3505, N4's seeds, so that every width is paired with N4's width-64 runs on the same seed, the same data order and the same trunk initialisation.
- **N4's width-64 runs** are reused for the exploratory description of the whole curve only. They enter neither decision rule, since their result is already known.
- **Runs:** 60 new.

### Why these widths

- **256** is the full width of the state, where the restriction disappears. This is the arm that separates the two explanations.
- **32 and 128** sit at half and double N4's width, so the trend is tested on widths that no run has seen.
- **The choices** were made before any run at these widths. The result applies to them.

## Primary and secondary outcomes

For each seed and width w:

D(w) = (J_dual-tied − J_dual-sep) − (J_same-tied − J_same-sep), with J = L_A + L_B,

the combined validation loss on the whole validation split, each head against its own target. This is N4's measure.

- **N5a (primary):** D(256).
- **N5b:** D(32) − D(128), paired by seed.
- **Intervals:** both form one family. Each uses a Bonferroni-corrected t-interval, 97.5% with 4 degrees of freedom (critical value 3.495), over the 5 seeds.

## Decision rules

**Preconditions:**
1. **Must match, by construction.** At initialisation, for every seed and width, the separate and tied versions give bit-identical logits on one batch.
2. **The batch, data and same-model checks** from G1 pass, and all four arms of a seed and width see the same batches.
3. **The full-width readout really is full rank.** At width 256, the initial P has full rank (smallest singular value above 1e-4 times the largest), for every seed. Otherwise tied and separate would not be equivalent at width 256 and N5a would not test what it claims.

**N5a (primary):**

| Outcome | Condition |
|---|---|
| Supported: without competition, the cost vanishes | The 97.5% interval for D(256) lies wholly below +0.02 |
| Refuted: the cost persists without any restriction | The interval lies wholly above +0.05 |
| Partly: a smaller cost remains | The interval lies wholly above 0 and wholly below +0.05 |
| Inconclusive | Everything else |

The rows are checked in order, and the first that applies is the verdict. So an interval wholly between 0 and +0.02 is scored supported.

**N5b:**

| Outcome | Condition |
|---|---|
| Supported: tightening the competition raises the cost | The 97.5% interval for D(32) − D(128) lies wholly above +0.05 |
| Refuted | The interval lies wholly below +0.05, with a standard deviation of at most 0.04 |
| Some effect, size unclear | The interval excludes 0 but meets neither rule |
| Inconclusive | Everything else |

**Why these thresholds.**
- **0.05 nats** is the margin used throughout the programme.
- **0.02 for "vanishes"** is less than a quarter of N4's D (0.088), so "supported" requires the cost to fall by more than three quarters, not merely to fall. A cost between 0.02 and 0.05 that excludes zero is scored "partly", the outcome the "both mechanisms" hypothesis predicts.
- **Sensitivity.** N4's D had a standard deviation of 0.008 across seeds. A difference of two Ds should have a spread near 0.011, which resolves the 0.05 margin comfortably, even at 97.5%.

## Secondary and exploratory outcomes

- **S1, the whole curve.** D at widths 32, 64 (N4's runs), 128 and 256, with intervals. Is it monotone?
- **S2, how many directions one reading needs.** The same-sep loss at each width. Where it stops improving is a measure of the directions a single reading uses.
- **S3, which reading pays.** Each reading's cost of tying, by width. N4 found the next-token reading paid more.
- **S4, the readings' overlap at each width.** N4a's measure (16-dimensional sensitivity subspaces at the ODE block's output, on N4's windows 1,537–1,792) for the dual-sep and same-sep models. At width 256, does the different-readings-do-not-converge pattern hold without any bottleneck? Untrained values at each width are measured too, so that change from initialisation can be reported, which N4's note found to matter.

## Sanity checks before trusting results

- Every data file's fingerprint matches the manifest.
- One batch is decoded, and both heads' targets are checked.
- Every number in the report is computed by code from saved JSON records. `n5_analyze.py` is written and committed before any real run, and tested on synthetic records covering every branch of both rules.
- **Calibration before queuing.** On untrained full-size models at each new width: the must-match control, the rank of the width-256 readout, and the untrained overlap. If a threshold or control turns out not to work as written, an amendment comes back for approval first.

## Compute, where it runs, and stopping rule

- **Where:** one Colab job, `n5_widths`, on a T4, resumable.
- **Cost:** 60 runs at about 0.10 to 0.13 hours each (wider readouts cost slightly more), plus the measurements. About 7 hours, likely more than one Colab session. A session that ends is resumed with Run all, as before.
- **Order:** width 256 first, then 32, then 128, so the primary hypothesis is answered first if the run is interrupted for long.
- **Stopping rule:** exactly these runs and measurements, each once. A crashed run resumes from its checkpoint. Nothing is added after seeing results except as a dated, exploratory entry in the deviation log.

## What each outcome would teach

| N5a | N5b | What we would learn | Next |
|---|---|---|---|
| Supported | Supported | N4's explanation survives a test it could have failed: the readings compete for directions, and the cost grows as directions grow scarce and vanishes when they are plentiful. | More voices at a fixed width: the explanation predicts the cost grows with the number of readings |
| Supported | Refuted | The cost needs a restriction, but does not grow when the restriction tightens. Competition matters at intermediate widths only. At tight widths both readings fall back on the same dominant directions. | Measure which directions each reading uses at width 32 |
| Refuted | — | N4's cost is not about directions. Sharing a matrix costs two different readings even when it restricts nothing. Gradient conflict is the better explanation. N4's result stands as stated, but its explanation is wrong. | Test gradient conflict directly, for example by projecting out conflicting gradient components in the tied arm (Yu et al., 2020) |
| Partly | Any | Both mechanisms contribute: some cost is competition for directions, some is conflict in the shared matrix. | Separate the two at width 64 by removing gradient conflict |
| Inconclusive | Inconclusive | The design was too noisy at these widths | More seeds, as a new test |

## Limitations known now

- **Same limits as before:**
  - 10% of the budget, one learning rate (not re-tuned for each width), one size and one dataset;
  - two related readings, with farther and harder confounded.
- **The test still acts at the readout.** It does not locate the readings' separation in the ODE block.
- **Equivalence at width 256 is representational, not optimisational.** Tied and separate can represent the same functions, but the extra matrix changes how training proceeds in both arms. That is what N5a tests, and it is why a cost there would point to optimisation.
- **Four widths.** The curve between them is not measured.
- **Reuse of N4's seeds.** This pairs the widths but means N5 shares N4's trunk initialisations and data order. The new widths are new runs, and the reused width-64 runs enter no decision rule.

## Deviation log

### Calibration and implementation details, before any real run, 2026-10-06

Results seen before this entry:
- the promised calibration on **untrained** full-size models: the full-rank check on N5's seeds and on G1's seeds 3001–3005, and the must-match check and untrained overlap at each new width on seeds 3001 and 3002, using the first 128 validation windows;
- test-mode runs on tiny models;
- synthetic records.

No real N5 run had started. No rule, threshold or control was changed, so no amendment was needed.

**What the calibration found.**
- **Must match.** Separate and tied readouts gave bit-identical logits at initialisation at every width.
- **The full-rank precondition passes for N5's seeds,** with the smallest singular value of the initial width-256 readout between 2.0e-4 and 2.5e-3 of the largest.
  - The threshold of 1e-4 sits inside the natural spread for random square matrices: one of the five calibration seeds not used by N5 (3001) came out at 9.9e-5.
  - Mathematically, any invertible P makes tied and separate equivalent. The threshold only guards against numerically singular readouts, and in fp32 every value seen is far from that.
  - Since N5's own seeds pass, the precondition is left as written.
- **The untrained overlap depends on width.** At initialisation the two heads' subspaces overlap about 0.66–0.73 at width 32, 0.48 at 64 (N4), 0.27–0.30 at 128 and 0.20 at 256, identically for dual and same. A narrower shared readout confines both heads' gradients to fewer common directions. This matters only for S4, which reports change from initialisation.

**Implementation details.**
- **Code.**
  - `n5_widths.py` runs the checks, the untrained overlaps, the 60 runs (width 256 first, then 32, then 128) and their measurements. It sets N4's readout width and calls N4's own code (`n4_directions.py`) for the models, training and measurement, so nothing but the width differs from N4.
  - `n5_analyze.py` applies both rules and writes `N5_RESULTS.md`.
  - The Colab job is `colab/jobs/n5_widths.py`.
- **Test mode** scales the widths with the model: 32, 4 and 16 for its 32-dimensional state.
- **An exploratory measure added before the run:** the conditioning of each trained readout P (smallest over largest singular value). At width 256, tied and separate are equivalent only while P stays invertible, so this reports whether training kept it so. It enters no decision rule.
- **Tests:**
  - The job ran end to end on tiny models: all checks, the untrained overlaps, 24 runs at the three test widths, the measurements and the result upload. A run interrupted at step 30 resumed from its checkpoint. `n5_analyze.py` read the job's real records without error.
  - `n5_analyze.py` gave the expected verdicts on synthetic records for 14 cases. These covered both supported; N5a refuted, partly, inconclusive, and supported with a small positive cost; N5b refuted (including the non-monotone case), not sensitive enough, some effect, and inconclusive; untestable through each precondition; and not reached.

## References

Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., et al. (2022). *Toy models of superposition*. arXiv. https://doi.org/10.48550/arxiv.2209.10652

Yang, Z., Dai, Z., Salakhutdinov, R., & Cohen, W. W. (2017). *Breaking the softmax bottleneck: A high-rank RNN language model*. arXiv. https://doi.org/10.48550/arxiv.1711.03953

Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. (2020). *Gradient surgery for multi-task learning*. arXiv. https://doi.org/10.48550/arxiv.2001.06782
