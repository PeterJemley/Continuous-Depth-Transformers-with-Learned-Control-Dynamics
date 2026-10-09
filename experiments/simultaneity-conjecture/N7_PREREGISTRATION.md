# Preregistration N7: does sharing track how different the readings are, or how hard?

Date: 2026-10-09    Status: **approved by the author, 2026-10-09, with amendment A1 (noise rate 0.8, approved).** Queued; no real N7 run has started.

Type: confirmatory for N7a (primary) and N7b (secondary hypothesis), each with its own decision rule. Everything else is secondary or exploratory and labelled so.

Parent documents: `N6R_NOTE.md` (next conjecture 1), `N6R_PREREGISTRATION.md`, and `PLAIN_SUMMARY.md`.

**Relation to earlier results.**
- **What N6r found.** Two readings share less of the model's internal space the further apart they are. Head A predicts the next token; head B predicts the token k places ahead. The overlap of their sensitivity subspaces was 0.96 for k = 1 (the same target), 0.49 for k = 2, 0.31 for k = 4 and 0.23 for k = 8, in that order in every model.
- **The confound N6r named.** Tokens further ahead are also harder to predict: head B's loss was 6.24, 6.92 and 7.11 nats for k = 2, 4 and 8, against 5.14 for the next token. So the overlap may track how *hard* head B's task is, not how *different* it is.
- **What N7 does.** It adds a reading that is as hard as the farthest one but no different from the nearest: the next token, with most of its training targets replaced at random.
- **What stands regardless.** N6r's result stands as stated, whatever N7 shows. N7 tests its explanation.

## The claims

**N7a (primary): sharing tracks how different the readings are, not how hard.** Head B predicts the next token, but 60% of its targets are replaced by tokens drawn at random from the training data's word frequencies. Its task becomes about as hard as predicting eight tokens ahead, but it is still the next-token reading. Its overlap with the clean next-token head exceeds the eight-ahead head's overlap by at least 0.30: o(noisy) − o(8) ≥ 0.30.

**N7b (secondary hypothesis): a harder copy of the same reading converges like the same reading.** The noisy head's agreement ratio with the clean head is at least 0.85 in at least 4 of 5 models.

### The explanation behind it

- **Why the noisy head is the same reading.** With random replacement at a fixed rate, the best prediction a model can make for the noisy target is a fixed blend: 40% of the true next-token prediction plus 60% of the word frequencies. That blend is a fixed transformation of the clean next-token prediction, the standard model of label noise (Patrini et al., 2017). It contains no information beyond the next-token reading. So the noisy head needs the same features as the clean one and nothing else.
- **Why it is as hard as k = 8.** Mixing in 60% random targets raises the loss of even a perfect predictor. Using N6r's next-token loss (5.14) and the word-frequency entropy of the training data, a rough estimate puts the noisy head's loss near 7.0 nats, about that of the k = 8 head (7.11). The real value is checked during the run (precondition 4).
- **What the two explanations predict.**
  - If sharing tracks **difference**, the noisy head shares nearly everything with the clean head, like the same-target control, and much more than the k = 8 head does.
  - If sharing tracks **difficulty**, the noisy head shares about as little as the k = 8 head, since its task is about as hard.

## Competing hypotheses

| Hypothesis | o(noisy) − o(8) (N7a) | Noisy head's agreement ratio (N7b) |
|---|---|---|
| The claim: sharing tracks how different the readings are | ≥ 0.30 | ≥ 0.85 |
| Sharing tracks how hard the readings are | Near 0 | About the k = 8 level (0.24) |
| Both: difficulty lowers sharing somewhat, difference more | Between 0 and 0.30 | Between |
| The random targets inject a "word-frequency" reading of their own | Positive but below 0.30 | Below 0.85 |
| The noisy head's subspace is just poorly estimated | Lowered by noise | Stability fails (precondition 3) |

The "own reading" rival is serious. Random targets drawn from word frequencies push the noisy head toward predicting common words, which could occupy directions of its own. N7b exists to tell this apart from the claim: the claim says the noisy head converges with the clean head; the rival says it doesn't, fully.

## Refuting result

N7a is refuted if the 95% interval for o(noisy) − o(8) lies wholly below +0.30, with a standard deviation of at most 0.05.

## Design

### Models and arms

N6r's model and training: the paper's `HybridODETransformer`, exact gradients, no control, two heads reading the final state through separate rank-64 readouts, P_B starting as a copy of P_A.

| Arm | Head A target | Head B target | Role |
|---|---|---|---|
| same | i + 1 | i + 1 | Must-fail control |
| k4 | i + 1 | i + 4 | Difficulty reference (N6r: loss 6.92, overlap 0.31) |
| k8 | i + 1 | i + 8 | The comparison (N6r: loss 7.11, overlap 0.23) |
| noisy | i + 1 | i + 1, with 60% of targets replaced | The new reading |

- **The noise.** In each training batch, each of head B's targets is replaced, independently with probability 0.6, by a token drawn from the unigram distribution of the training token file. The draws come from their own random generator, seeded per run, so the inputs and data order are identical to the other arms. For measurement and validation, the same corruption is applied once, with a fixed seed, so every model sees the same noisy targets.
- **Same inputs in every arm,** as in N6 and N6r.
- **Training:** 2,879 steps at peak learning rate 2.4e-3, batch 32 × 128, on the pinned WikiText-103 token files.
- **Seeds:** 3801 to 3805, not used before.
- **Runs:** 20.

### The measure

N6r's, unchanged:
- each head's top-16 sensitivity subspace at the ODE block's output, from the per-token gradient covariance of its loss against its own targets (the noisy targets, for the noisy head);
- o, the overlap: the mean squared cosine of the principal angles;
- the agreement ratio: head A's subspace from one half of the windows against head B's from the other half, divided by the geometric mean of the heads' own split-half stabilities.

**Windows:** WikiText-103 test-split windows 257 to 512, which no measurement has used. N6r used windows 1 to 256.

## Primary outcome (N7a)

For each seed, o(noisy) − o(8). The interval is a t-interval at 95% with 4 degrees of freedom, over the 5 seeds.

## Decision rules

**Preconditions:**
1. **Data and setup:** the G1 checks pass, the targets are shifted correctly, the identical-heads check gives an agreement ratio of 1, and all four arms of a seed see the same batches.
2. **Must fail:** the same arm's agreement ratio is at least 0.95 in at least 4 of 5 models.
3. **Stable subspaces:** every head in arms k4, k8 and noisy has a split-half stability of at least 0.80.
4. **The noisy reading really is hard:** the noisy head's validation loss, against its noisy targets, is at least the k4 head B's loss for the same seed, in at least 4 of 5 seeds. Otherwise the noisy head is not as hard as a reading known to share little, and N7 cannot separate difference from difficulty.

**N7a:**

| Outcome | Condition |
|---|---|
| Supported: sharing tracks how different the readings are | The 95% interval for o(noisy) − o(8) lies wholly above +0.30 |
| Refuted | The interval lies wholly below +0.30, with a standard deviation of at most 0.05 |
| Some effect, size unclear | The interval excludes 0 but meets neither rule |
| Inconclusive | Everything else |

**N7b:**

| Outcome | Condition |
|---|---|
| Supported: a harder copy of the same reading converges | The noisy head's agreement ratio is at least 0.85 in at least 4 of 5 models |
| Refuted | It is at most 0.60 in at least 4 of 5 models |
| Intermediate | Anything else |

**Why these thresholds.**
- **0.30 for N7a.** The two explanations predict very different values. If sharing tracks difference, o(noisy) should be near the same-target level (about 0.9) and the difference near 0.6. If it tracks difficulty, the difference should be near 0. A threshold of 0.30 sits between them, so supporting the claim requires the noisy head to share far more than the k = 8 head, and refuting it requires them to be close.
- **Sensitivity.** In N6r, the overlap at each rung varied across seeds with a standard deviation of about 0.01 to 0.04. A difference of two should vary by about 0.05 at most, which resolves a margin of 0.30 easily. The refutation rule still checks the spread in the result itself.
- **0.85 and 0.60 for N7b.** Same-target heads reached agreement ratios of 0.95 to 0.97 in N6r, and the nearest different reading (k = 2) about 0.50. 0.85 is close to the first; 0.60 is close to the second.
- **The raw overlap in N7a is conservative for the claim.** The noisy head's subspace may be estimated a little less precisely, which lowers the raw overlap. That works against the claim, not for it. N7b uses the agreement ratio, which corrects for it.
- **60% noise.** Chosen in advance from the rough loss estimate above, to make the noisy head about as hard as k = 8. Precondition 4 checks that it is at least as hard as k = 4.

## Secondary and exploratory outcomes

- **S1:** every arm's overlap and agreement ratio, by seed.
- **S2:** every head's loss. The noisy head's loss is also reported against the *clean* next-token targets, to show how much of the next-token reading it still holds.
- **S3:** each head's stability.
- **S4:** change from initialisation.
- **S5:** k4 and k8 against N6r's values, as a replication on new seeds and windows (descriptive only).

## Sanity checks before trusting results

- Every data file's fingerprint matches the manifest.
- One batch is decoded: the noisy targets match the clean ones at about 40% of positions, and the replaced tokens follow the word frequencies.
- Every number in the report is computed by code from saved JSON records. `n7_analyze.py` is written and committed before any real run, and tested on synthetic records covering every branch of the rules.
- **Calibration before queuing,** on untrained full-size models (G1's seeds), on N7's windows:
  - the identical-heads check, and the same arm failing the control;
  - each head's stability, including the noisy head's;
  - the fraction of replaced targets, and the noisy targets' loss for an untrained model.

  If a threshold or control does not work as written, an amendment comes back for approval first.

## Compute, where it runs, and stopping rule

- **Where:** one Colab job, `n7_harder`, on a T4, resumable.
- **Cost:** 20 runs at about 0.10 hours each, plus the measurements. About 2.3 hours.
- **Stopping rule:** exactly these runs and measurements, each once. A crashed run resumes from its checkpoint. Nothing is added after seeing results except as a dated, exploratory entry in the deviation log.

## What each outcome would teach

| N7a | N7b | What we would learn | Next |
|---|---|---|---|
| Supported | Supported | Sharing tracks how different the readings are, not how hard. A harder copy of a reading still converges with it. N6r's ladder measures relatedness. | Readings that differ in kind; and whether the cost of sharing (N4) grows along the ladder |
| Supported | Intermediate or refuted | The noisy head shares far more than k = 8 but does not fully converge: the random targets add a reading of their own. Relatedness dominates, with a smaller effect of the noise. | Lower the noise rate and see whether convergence returns |
| Refuted | Refuted | Sharing tracks difficulty: a harder reading shares less, even when it is the same reading. N6r's ladder measured difficulty, not relatedness. | Rethink the overlap as a measure of readings |
| Some effect | Any | Both play a part | Separate them with several noise rates |
| Untestable | — | A precondition failed, for example the noisy head was not hard enough | A higher noise rate, as a new test |

## Limitations known now

- **One way of making a reading harder.** Random replacement adds a word-frequency component; other ways of adding difficulty might behave differently.
- **One noise rate.** 60% was chosen to match difficulty; the effect of the rate itself is not measured.
- **Same limits as before:** 10% of the budget, one learning rate, one size, one dataset, one readout width.

## Deviation log

### Amendment A1, before any real run, 2026-10-09: approved by the author

Results seen before this entry:
- the promised calibration on **untrained** full-size models (G1's seeds 3001 and 3002) on N7's test-split windows 257–512;
- test-mode runs on tiny models;
- synthetic records;
- a rough estimate of the noisy head's loss from a bigram model built on the test-mode token files.

No real N7 run had started.

**What the calibration found.**
- **The controls and measures work.**
  - Untrained same-target heads gave agreement ratios of 0.47–0.50, so the control fails where it must.
  - Each head's stability was 0.97–0.99 at initialisation, including the noisy head's.
  - All four arms started at the same overlap (0.46–0.50).
  - The noise replaced 60.2% of one training batch's targets and 59.7% of the measurement targets, with the replacements following the word frequencies.
- **But precondition 4 would very likely fail at 60% noise.**
  - The estimate of about 7.0 nats in the preregistration was an upper limit, not a central estimate. A trained model can partly anticipate the random replacements, because they are common words it already gives weight to. That lowers the noisy head's loss below the limit.
  - Bounding each part of the loss separately gives a likely range of about 6.6–6.9 nats at 60% noise. The two parts are the 40% of targets that are the true next token, and the 60% that are drawn from word frequencies.
  - The k4 head's loss in N6r was 6.92. So the noisy head would probably not be as hard as k4, precondition 4 would fail, and N7 would be untestable.
  - Test mode could not show this: tiny models after 60 steps barely predict anything.

**The amendment.**
- **The noise rate becomes 0.8 instead of 0.6.** The same bounds put the noisy head's loss at about 7.0–7.3 nats at 80%, around the k8 head's 7.11. That is clearly above k4's.
- **Why the reading is still the same reading.** At 80% noise, the best prediction for the noisy target is 20% of the next-token prediction plus 80% of the word frequencies. That is still a fixed transformation of the next-token reading (Patrini et al., 2017), and contains nothing else.
- **What it costs.** The noisy head sees the true next token in only 20% of its targets, so it may learn the reading less precisely. That would lower its overlap with the clean head, which works *against* the claim, not for it.
- **Everything else is unchanged:** precondition 4 (still against k4), every threshold, the arms, seeds and windows.

### Implementation details fixed before any real run, 2026-10-09

- **Code.**
  - `n7_harder.py` reuses N6r's code for the model, windows, identical-heads check and agreement ratio. It adds the noisy arm and records each head's loss against both its own and the clean targets.
  - `n7_analyze.py` applies the rules and writes `N7_RESULTS.md`.
  - The Colab job is `colab/jobs/n7_harder.py`.
- **The training noise** is drawn per step from a generator seeded by the run's seed and the step number, so a resumed run draws exactly the same noise without storing any state. The validation and measurement targets are corrupted once, with fixed seeds.
- **The unigram distribution** is counted from the full training token file, in chunks of 10 million tokens to keep memory small.
- **A bug caught in testing:** the first version referred to N6's module before importing it. The calibration run failed on import and the bug was fixed. No result was affected.
- **Tests:**
  - The job ran end to end on tiny models at 60% noise: all checks, the untrained measurements, all 8 runs, the measurements and the result upload. A run interrupted at step 23 resumed from its checkpoint.
  - After amendment A1, the noise check was rerun at 80%: 80.2% of one batch's targets were replaced.
  - `n7_analyze.py` gave the expected verdicts on synthetic records for 14 cases. These covered both supported; N7a supported with N7b intermediate or refuted; N7a refuted, some effect, not sensitive enough, and inconclusive; untestable through each precondition (must-fail, stability, hardness, the identical-heads check, the noise check, data order); and not reached.

## References

Patrini, G., Rozza, A., Menon, A. K., et al. (2017). Making deep neural networks robust to label noise: A loss correction approach. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2233–2241. https://doi.org/10.1109/cvpr.2017.240
