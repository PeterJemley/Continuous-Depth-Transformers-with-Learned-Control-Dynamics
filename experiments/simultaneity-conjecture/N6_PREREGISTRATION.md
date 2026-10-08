# Preregistration N6: does the half-overlap balance depend on how different the readings are?

Date: 2026-10-08    Status: **approved by the author, 2026-10-08.** No real N6 run has started.

Type: confirmatory for N6a (primary) and N6b (secondary hypothesis), each with its own decision rule. Everything else is secondary or exploratory and labelled so.

Parent documents: `N5_NOTE.md` (next conjecture 1), `N4_NOTE.md` (with its dated correction), and `PLAIN_SUMMARY.md`.

**Relation to earlier results.**
- **What N5 observed, without a decision rule.** In all 20 models that read two different targets (the next token, and the token after next), the two readings' 16-dimensional sensitivity subspaces ended at an overlap of 0.42 to 0.53. That held at every readout width, whether they started above that level (about 0.7 at width 32) or below it (about 0.2 at widths 128 and 256). Two heads reading the same target converged instead, to 0.87–0.99.
- **The conjecture N5 proposed.** The balance point is set by how different the two readings are, not by the readout width or the starting point.
- **What N6 does.** It changes only how different the second reading is, and asks whether the balance point moves in the predicted direction.
- **What stands regardless.** All earlier results stand, including N5's inconclusive verdicts. N6 is a new test on new seeds and new windows, not a re-scoring of N5.

## The claims

**N6a (primary): the more the readings differ, the less of their space they share.** Head A always predicts the next token (i + 1). Head B predicts token i + k, for k = 2, 4 or 8. The overlap between the two readings' sensitivity subspaces falls as k grows. Between k = 2 and k = 8 it falls by at least 0.10.

**N6b (secondary hypothesis): the half-overlap replicates.** At k = 2, the reading pair N5 measured, the overlap lies between 0.40 and 0.60 in at least 4 of 5 models, on new seeds and new windows.

### The explanation behind it

- **A reading's sensitive directions are the features it relies on.** Two readings share the directions for features both need, and each keeps its own directions for the rest.
- **Readings that differ more need fewer common features.** The next token and the one after it depend largely on the same local context: the current phrase, its syntax, the words just read. The token eight places ahead depends much less on the exact local context and more on what the passage is about.
- **That ordering is fixed in advance.** How much two words tell you about each other falls with the distance between them (Lin & Tegmark, 2017). So k = 2, 4, 8 is a ladder of readings that differ more and more from the next-token reading, set before any run.
- **The prediction.** The balance point, about 0.5 at k = 2, falls as k grows, because fewer of the features each reading needs are shared.
- **What it forbids.** An overlap that stays flat across k, near 0.5 whatever the second reading is. That would mean the half-overlap reflects the model's structure (its width, its shared trunk, the readout design), not the readings.

## Competing hypotheses

| Hypothesis | o(2) − o(8) (N6a) | o(2) (N6b) |
|---|---|---|
| The claim: the balance point is set by how different the readings are | ≥ 0.10 | About 0.5 |
| Half-overlap is structural, the same for any two different readings | Near 0 | About 0.5 |
| Farther readings are just noisier, so their subspaces are poorly estimated | Falls, but the farther head's subspace is unstable | About 0.5 |
| The farther reading collapses onto word frequency, which both readings share | Rises (negative difference) | About 0.5 |
| N5's 0.5 was specific to its seeds or windows | Any | Outside 0.40–0.60 |

The noisier-subspace rival is the main threat to N6a, because poorly estimated subspaces overlap less by chance. It is handled by a precondition on each head's split-half stability (below).

## Refuting result

N6a is refuted if the 95% interval for o(2) − o(8) lies wholly below +0.10, with a standard deviation of at most 0.05.

## Design

### Models and arms

All models are N4's: the paper's `HybridODETransformer`, exact gradients, no control, and two heads reading the final state through separate rank-64 readouts. P_B starts as a copy of P_A, and W_B is seeded separately, as in N4. Width 64 is used because N5 showed that training is consistent there and erratic at narrower widths.

| Arm | Head A target | Head B target | Role |
|---|---|---|---|
| same | i + 1 | i + 1 | Must-fail control |
| k2 | i + 1 | i + 2 | N5's pair, replicated |
| k4 | i + 1 | i + 4 | |
| k8 | i + 1 | i + 8 | |

- **Training:** as in N4, on the mean of the two heads' losses, 2,879 steps at peak learning rate 2.4e-3, batch 32 × 128, on the pinned WikiText-103 token files.
- **Same inputs in every arm.** Every training window is drawn with 8 extra tokens, so all four arms of a seed see exactly the same inputs in the same order and differ only in head B's target.
- **Seeds:** 3601 to 3605, not used before.
- **Runs:** 20.

### The measure

N4a's measure, unchanged:
- each head's top-16 sensitivity subspace at the ODE block's output, from the per-token gradient covariance of its loss;
- the overlap of the two subspaces, the mean squared cosine of their principal angles (1 for identical subspaces, about 0.06 for random ones).

**Windows.** Validation windows 1,793 to 1,931, the last 139. No earlier measurement used them: G1 used windows 1–256, G1c 257–512, G1d 513–768, N1 769–1,024, N2 1,025–1,536, and N4 and N5 1,537–1,792. Each window carries 8 extra tokens, so every position has all four targets.

## Primary outcome (N6a)

For each seed, the overlap difference o(2) − o(8), with o(k) the overlap in arm k. The interval is a t-interval at 95% with 4 degrees of freedom, over the 5 seeds.

## Decision rules

**Preconditions:**
1. **The data and setup checks** from G1 pass for the new seeds, and all four arms of a seed see the same batches, confirmed by fingerprinting the first batch of each run.
2. **Must fail:** the same arm's overlap is at least 0.95 in at least 4 of 5 models. Two heads reading the same target must converge on the same directions, or the measure cannot attribute a lower overlap to the readings.
3. **Stable subspaces:** in every model of arms k2, k4 and k8, each head's subspace estimated from the odd windows overlaps the one from the even windows by at least 0.80. This rules out the noisier-subspace rival: a farther head whose subspace cannot be estimated reliably would show a low overlap for that reason alone.

**N6a:**

| Outcome | Condition |
|---|---|
| Supported: the balance point falls as the readings differ more | The 95% interval for o(2) − o(8) lies wholly above +0.10 |
| Refuted | The interval lies wholly below +0.10, with a standard deviation of at most 0.05 |
| Some effect, size unclear | The interval excludes 0 but meets neither rule |
| Inconclusive | Everything else |

A negative difference that excludes 0 is scored as refuted if the standard-deviation condition holds. The table above reports it, and S1 names its direction.

**N6b:**

| Outcome | Condition |
|---|---|
| Supported: the half-overlap replicates | o(2) lies between 0.40 and 0.60 in at least 4 of 5 models |
| Refuted | o(2) lies outside 0.40–0.60 in at least 4 of 5 |
| Intermediate | Anything else |

N6b also requires preconditions 1 to 3.

**Why these thresholds.**
- **0.10 for N6a.** In N5, changing the readout width eightfold, and starting the readings anywhere from 0.2 to 0.7 apart, moved their end point by no more than about 0.1 (0.42 to 0.53). The claim is that how different the readings are matters more than all of that did. A fall of at least 0.10 is the smallest that would show it.
- **Sensitivity.** At width 64, the two-reading overlap varied across N4's seeds with a standard deviation of about 0.013. The difference of two overlaps should then vary by about 0.02 to 0.05, so the design resolves 0.10 comfortably. The rule requires a standard deviation of at most 0.05 before it will refute, so if the spread is larger than expected, the result is scored inconclusive, not refuted. This is N5's lesson, applied: the sensitivity is checked in the result itself, not assumed.
- **0.40 to 0.60 for N6b** is a band of ±0.10 around half-overlap, slightly wider than N5's range (0.42–0.53), to allow for new seeds and fewer windows.
- **0.80 for stability.** N4's subspaces had a split-half stability of 0.92 to 0.98 on 256 windows. N6 has 139 windows, so each half has about half as many, and stability will be somewhat lower. The calibration below checks that 0.80 is attainable.

## Secondary and exploratory outcomes

- **S1, the whole ladder.** o(1), o(2), o(4) and o(8) for every seed, and whether o(4) lies between o(2) and o(8). The claim predicts o(2) > o(4) > o(8). Only the endpoints enter N6a.
- **S2, the losses.** Each head's loss by arm. Head B's loss rises with k: the farther ahead, the harder to predict. This is reported because distance confounds "different" with "harder" (see Limitations).
- **S3, stability-corrected overlap.** o(k) divided by the geometric mean of the two heads' split-half stabilities, as a check that any fall with k is not an estimation artefact. Descriptive only.
- **S4, change from initialisation.** The untrained overlap in each arm, measured by the job. All arms should start alike, because the next token and the token k places ahead have the same overall word frequencies.

## Sanity checks before trusting results

- Every data file's fingerprint matches the manifest.
- One batch is decoded, and all four targets are checked against the input tokens.
- Every number in the report is computed by code from saved JSON records. `n6_analyze.py` is written and committed before any real run, and tested on synthetic records covering every branch of both rules.
- **Calibration before queuing.** On untrained full-size models:
  - the overlap in each arm;
  - each head's split-half stability on the 139 windows, to check that the 0.80 threshold is attainable;
  - the must-match and same-input checks.

  If a threshold or control turns out not to work as written, an amendment comes back for approval first.

## Compute, where it runs, and stopping rule

- **Where:** one Colab job, `n6_halfoverlap`, on a T4, resumable.
- **Cost:** 20 runs at about 0.10 hours each, plus the measurements. About 2.2 hours.
- **Stopping rule:** exactly these runs and measurements, each once. A crashed run resumes from its checkpoint. Nothing is added after seeing results except as a dated, exploratory entry in the deviation log.

## What each outcome would teach

| N6a | N6b | What we would learn | Next |
|---|---|---|---|
| Supported | Supported | The balance point is a property of the readings: the more two readings differ, the less of the state they share. Half-overlap is what the next token and the one after it happen to need. | Readings that differ in kind, not only distance; and whether the cost of sharing (N4) grows as the overlap falls |
| Refuted | Supported | Half-overlap is structural: any two different readings settle at about 0.5 in this model, however different they are. | Find what sets it: vary the subspace dimension, the trunk, or the number of readings |
| Supported | Refuted | The overlap does fall with distance, but N5's level of 0.5 did not replicate on new seeds and windows. | Treat the level as variable, and the ordering as the finding |
| Negative difference | Any | The farther reading shares *more* with the next token, probably because both fall back on word frequency. | Measure what the shared directions encode |
| Inconclusive, or a precondition failed | — | The design could not tell, for example because the farther head's subspace was unstable. | More windows, or a lower subspace dimension, as a new test |

## Limitations known now

- **Distance confounds "different" with "harder".** The token eight places ahead is both less related to the next token and harder to predict. A fall in overlap could come from either. S2 and S3 bear on this, but cannot separate them fully.
- **One kind of difference.** All the readings are future tokens. Readings that differ in kind, such as a topic or sentiment label, are not tested.
- **Same limits as before:** 10% of the budget, one learning rate, one size, one dataset, one readout width.
- **Fewer windows.** Only 139 validation windows remain unused, about half N4's 256, so the subspaces are estimated less precisely. Precondition 3 guards against that.
- **N6b's band comes from N5's data,** so N6b is a replication, not a fresh prediction.

## Deviation log

None yet.

## References

Lin, H. W., & Tegmark, M. (2017). Critical behavior in physics and probabilistic formal languages. *Entropy, 19*(7), 299. https://doi.org/10.3390/e19070299
