# Preregistration N6r: the half-overlap test, with its control set for its measurement

Date: 2026-10-08    Status: **approved by the author, 2026-10-08.** Queued; no real N6r run has started.

Type: confirmatory for N6a (primary) and N6b (secondary hypothesis), the same claims and thresholds as N6. Everything else is secondary or exploratory and labelled so.

Parent documents: `N6_PREREGISTRATION.md` and `N6_NOTE.md`.

**Relation to earlier results.**
- **What N6 found.** Both verdicts were untestable. The must-fail control, two heads reading the same target, needed an overlap of at least 0.95 in at least 4 of 5 models, and reached it in 3. Seed 3601 missed by 0.0002.
- **Why.** The threshold came from N4's measurement on 256 windows. N6 had only 139, and with fewer windows noise caps how close even identical readings can measure. The two seeds that missed had the least stable subspaces. The calibration should have caught this and did not.
- **What N6 showed, as exploratory data.** The overlap fell from 0.48 at k = 2 to 0.31 at k = 4 and 0.25 at k = 8, in that order in all 5 seeds: o(2) − o(8) = +0.227, 95% interval [+0.184, +0.269].
- **What N6r does.** It repeats N6 on new seeds and new data, with two changes, both fixed here before any run: the control is judged against the measurement's own noise, and the measurement uses twice as many windows.
- **What stands regardless.** N6 stays untestable; N6r does not re-score it. N6's exploratory pattern is what N6r tests. Because that pattern has been seen, N6r is a replication of it, under thresholds that were themselves set before N6 ran and are unchanged.

## The claims

Unchanged from N6:

**N6a (primary): the more the readings differ, the less of their space they share.** Head A predicts the next token (i + 1); head B predicts token i + k, for k = 2, 4 or 8. The overlap between the two readings' sensitivity subspaces falls as k grows, by at least 0.10 between k = 2 and k = 8.

**N6b (secondary hypothesis): the half-overlap replicates.** At k = 2 the overlap lies between 0.40 and 0.60 in at least 4 of 5 models.

The explanation, the competing hypotheses and the reasons for the ordering by distance are as in `N6_PREREGISTRATION.md` (Lin & Tegmark, 2017).

## Refuting result

As in N6: N6a is refuted if the 95% interval for o(2) − o(8) lies wholly below +0.10, with a standard deviation of at most 0.05.

## What changes from N6

### 1. The must-fail control, judged like with like

**The problem with a raw threshold.** A raw overlap is limited by estimation noise, and how much depends on the number of windows. N6 showed that a threshold set at one window count can fail at another for that reason alone.

**The problem with the simplest correction.** Dividing the overlap by the heads' split-half stabilities, as N6's note suggested, mixes estimates of different quality. The overlap uses all the windows, while each stability compares two half-sized estimates and so is lower. The ratio is biased upward: N6's same-target heads came out at 1.02. A control with that bias could pass heads that had not fully converged.

**The rule used here.** Every quantity in the comparison is built from half-sized estimates on independent halves of the windows:
- the **cross-head, cross-half overlap**: head A's subspace from the odd windows against head B's from the even windows, and head A's from the even windows against head B's from the odd, averaged;
- each head's **split-half stability**: its subspace from the odd windows against its own from the even windows, as before.

The **agreement ratio** is the cross-head, cross-half overlap divided by the geometric mean of the two heads' stabilities.
- If the two heads read the same directions, comparing A with B across halves is no different from comparing A with itself across halves, and the ratio is about 1, whatever the window count.
- If they read different directions, the ratio is lower.

**Must fail:** the same arm's agreement ratio is at least 0.95 in at least 4 of 5 models.

**Both ends checked before any run** (see Calibration):
- two heads with *identical* weights must give an agreement ratio of exactly 1;
- two independently initialised, untrained heads must fail the control.

### 2. Fresh data, with more windows

- The overlap is measured on **256 windows of the WikiText-103 test split**: windows 1 to 256 in the same definition as N6, each with 8 extra tokens.
- No overlap or geometry measurement in this programme has used the test split, and no model was trained, tuned or selected on it. It was used before only to evaluate the paper's model: for test losses and solver checks in the WikiText-103 study (`../wikitext103-training/`), and in T0 (`../too-linear-conjecture/`). Nothing in the simultaneity programme was chosen on it.
- 256 windows is N4's count, twice N6's.
- The validation split has no unused windows left, which is why the test split is used.

### 3. New seeds

3701 to 3705, not used before.

## Design, otherwise unchanged

- **Models:** N4's model with separate rank-64 readouts; P_B starts as a copy of P_A, and W_B is seeded separately.
- **Arms:** same (k = 1, the must-fail control), k2, k4, k8.
- **Training:** 2,879 steps at peak learning rate 2.4e-3, batch 32 × 128, on the pinned WikiText-103 token files. Every training window carries 8 extra tokens, so all four arms of a seed see the same inputs.
- **The measure:** each head's top-16 sensitivity subspace at the ODE block's output, from the per-token gradient covariance of its loss; the overlap is the mean squared cosine of the principal angles.
- **Runs:** 20.

## Decision rules

**Preconditions:**
1. **The data and setup checks** from G1 pass, and all four arms of a seed see the same batches (first-batch fingerprint).
2. **Must fail:** the same arm's agreement ratio is at least 0.95 in at least 4 of 5 models.
3. **Stable subspaces:** in every model of arms k2, k4 and k8, each head's split-half stability is at least 0.80.

**N6a and N6b:** exactly as in N6.

| N6a outcome | Condition |
|---|---|
| Supported: the balance point falls as the readings differ more | The 95% interval for o(2) − o(8) lies wholly above +0.10 |
| Refuted | The interval lies wholly below +0.10, with a standard deviation of at most 0.05 |
| Some effect, size unclear | The interval excludes 0 but meets neither rule |
| Inconclusive | Everything else |

| N6b outcome | Condition |
|---|---|
| Supported: the half-overlap replicates | o(2) lies between 0.40 and 0.60 in at least 4 of 5 models |
| Refuted | o(2) lies outside 0.40–0.60 in at least 4 of 5 |
| Intermediate | Anything else |

The primary measure o(k) is the full-window overlap, as in N6. The agreement ratio is used only for the must-fail control.

**Why the thresholds are unchanged.** They were set before N6 ran, for reasons that still hold (`N6_PREREGISTRATION.md`). Changing them after seeing N6's data would fit them to it. The one change, to the control, follows from a defect in the measurement identified in `N6_NOTE.md`, not from N6's results for the claims.

## Secondary and exploratory outcomes

As in N6:
- **S1:** the whole ladder, o(1), o(2), o(4) and o(8), and whether o(2) > o(4) > o(8) in each seed.
- **S2:** each head's loss.
- **S3:** stabilities, and the agreement ratio for every arm, not only the same arm.
- **S4:** change from initialisation.

Added:
- **S5, N6r against N6.** The ladder in N6r beside N6's, descriptive only, since the seeds and data differ.

## Calibration before queuing

On untrained full-size models (G1's seeds, not N6r's), on N6r's test-split windows:
- **The control's two ends.**
  - A model whose two heads have identical weights must give an agreement ratio of exactly 1.
  - The same arm with independently seeded heads, as in the real runs, must give a ratio well below 0.95. Untrained heads have not converged, so the control must fail there.
- **Stability** of each head with 256 windows, against the 0.80 threshold.
- **The data:** the test-split windows decode correctly, with every target shifted by the right amount.

If any check shows a threshold or control not working as written, an amendment comes back for approval first.

## Sanity checks before trusting results

- Every data file's fingerprint matches the manifest.
- Every number in the report is computed by code from saved JSON records. `n6r_analyze.py` is written and committed before any real run, and tested on synthetic records covering every branch of the rules, including the new control.

## Compute, where it runs, and stopping rule

- **Where:** one Colab job, `n6r_halfoverlap`, on a T4, resumable.
- **Cost:** 20 runs at about 0.10 hours each, plus the measurements. About 2.3 hours.
- **Stopping rule:** exactly these runs and measurements, each once. A crashed run resumes from its checkpoint. Nothing is added after seeing results except as a dated, exploratory entry in the deviation log.

## What each outcome would teach

| N6a | N6b | What we would learn | Next |
|---|---|---|---|
| Supported | Supported | The balance point is set by how different the readings are. N6's pattern is real, and the half-overlap is what two adjacent tokens need. | Readings that differ in kind, such as a topic label; and whether the cost of sharing (N4) grows as the overlap falls |
| Refuted | Supported | N6's ladder did not replicate: half-overlap is structural, the same for any two different readings. | Find what sets it: subspace dimension, trunk, number of readings |
| Supported | Refuted | The ordering holds, but the level at k = 2 varies with seeds or data | Treat the ordering as the finding |
| Negative difference | Any | The farther reading shares more with the next token | Measure what the shared directions encode |
| Untestable again | — | The control or the stability still does not work at this scale | More windows, or a lower subspace dimension |

## Limitations known now

- **Distance confounds "different" with "harder",** as in N6.
- **The test split is a different text from the validation split.** If the overlap depends on the text, N6r and N6 can differ for that reason. S5 reports both.
- **N6r is a replication of a pattern already seen in N6.** Its confirmatory weight comes from new seeds, new data and thresholds set before N6, not from a fresh prediction.
- **Same limits as before:** 10% of the budget, one learning rate, one size, one dataset, one readout width.

## Deviation log

### Calibration and implementation details, before any real run, 2026-10-08

Results seen before this entry:
- the promised calibration on **untrained** full-size models (G1's seeds 3001 and 3002, not N6r's), on N6r's 256 test-split windows;
- test-mode runs on tiny models;
- synthetic records.

No real N6r run had started. No rule, threshold or control was changed, so no amendment was needed.

**What the calibration found.**
- **The control's upper end works.** A model whose two heads have identical weights gave an agreement ratio of exactly 1.0.
- **The control's lower end works.** Untrained, independently seeded same-target heads gave agreement ratios of 0.49 and 0.53, far below 0.95. The control fails where it must.
- **Stability is high with 256 windows:** 0.978–0.986 for every head at initialisation, against 0.92–0.97 for N6's 139 windows.
- **All arms start alike:** overlaps of 0.475–0.516, identical across arms within each seed.
- **The test-split windows decode correctly,** with every target shifted by exactly k.

**Implementation details.**
- **Code.**
  - `n6r_halfoverlap.py` reuses N6's code (`n6_halfoverlap.py`) for the model, arms and training. It adds the test-split windows, the identical-heads check and the agreement ratio.
  - `n6r_analyze.py` applies the rules and writes `N6R_RESULTS.md`.
  - The Colab job is `colab/jobs/n6r_halfoverlap.py`.
- **The identical-heads check** runs inside the job, on N6r's own first seed, as a precondition alongside the data checks.
- **Tests:**
  - The job ran end to end on tiny models: all checks, the untrained measurements, all 8 runs, the measurements and the result upload. A run interrupted at step 14 resumed from its checkpoint.
  - `n6r_analyze.py` gave the expected verdicts on synthetic records for 15 cases: N6's 13, plus a same-target control with a raw overlap of only 0.93 that correctly passes on an agreement ratio of 1.0, and an identical-heads check that correctly makes the run untestable when it fails.

## References

Lin, H. W., & Tegmark, M. (2017). Critical behavior in physics and probabilistic formal languages. *Entropy, 19*(7), 299. https://doi.org/10.3390/e19070299
