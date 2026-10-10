# Preregistration N8: a measure of what a reading uses, free of its mistakes

Date: 2026-10-10    Status: **approved by the author, 2026-10-10, with amendment A1 (a must-differ control, approved).** Queued; no real N8 run has started.

Type: confirmatory for N8a (primary), N8b and N8c, each with its own decision rule, on models already trained in N6r and N7, with a measure not yet computed on them. Everything else is secondary or exploratory and labelled so.

Parent documents: `N7_NOTE.md` (next conjecture 1), `N6R_NOTE.md` (with its dated correction), and `PLAIN_SUMMARY.md`.

**Relation to earlier results.**
- **What N7 found.** A head reading exactly the next token, with 80% of its targets randomised, shared no more with the clean next-token head (overlap 0.31) than a head reading four tokens ahead (0.30). It did not converge with it (agreement ratio 0.30–0.33).
- **N7's conjecture.** The overlap measure used from N4 to N7 is built from the gradient of each head's loss against its *actual* targets. So it sees where a reading's errors fall as well as which features it uses. Readings with unpredictable targets, whether far ahead or randomised, then look separate partly because their errors differ.
- **What N8 does.** It measures again with a target-free version of the same quantity, on the same trained models and on fresh windows. That separates the two explanations.
- **What stands regardless.** Every earlier verdict stands, including N6r's ladder and N7's refutations, which were made with the measure as preregistered. N8 tests an explanation of them with a different measure.

## The measure

**The measure used so far.** For each head, the covariance of the gradient of its loss at the ODE block's output, with the gradient taken against the actual target token at each position. In the literature on second-order optimisation this is the *empirical Fisher*. It is known to differ from the Fisher information, and to mislead when used as if it were the Fisher (Kunstner et al., 2019; Martens, 2014).

**The target-free measure (new).** The same covariance, with each gradient taken against a token *drawn from the head's own predicted distribution* rather than the actual token. Its expectation is the Fisher information of the head's predictions with respect to the state. That is a property of what the head computes: the directions in the state that most change its predictions, weighted by how much they change them. It does not depend on what the targets turned out to be, and so not on where the head's errors fall.

**Everything else is N6r's:** each head's top-16 eigenvectors, the overlap (the mean squared cosine of the principal angles), split-half stability, and the agreement ratio built from independent halves of the windows.

**Common random numbers.** Each position's sampled token comes from a single uniform draw shared by the two heads, through the inverse of each head's cumulative distribution. Two heads with identical predictions therefore sample identical tokens, and score exactly 1. Different heads still sample from their own distributions.

## The claims

**N8a (primary): with errors removed, a harder copy of the same reading shares what it uses.** On N7's models, under the target-free measure, the noisy head's overlap with the clean next-token head exceeds the eight-ahead head's by at least 0.30: o_F(noisy) − o_F(8) ≥ 0.30.

**N8b: and it converges.** On N7's models, the noisy head's target-free agreement ratio with the clean head is at least 0.85 in at least 4 of 5 models.

**N8c: the readings further ahead really do use different directions.** On N6r's models, under the target-free measure, the overlap still falls with distance: o_F(2) − o_F(8) ≥ 0.10.

### The explanation behind it

- **Why the noisy head should match the clean head under the target-free measure.** The best prediction for a target randomised at 80% is 20% of the next-token prediction plus 80% of the word frequencies (Patrini et al., 2017). The word frequencies do not depend on the state, so the noisy head's predictions change with the state only through its next-token part. Its sensitive directions are therefore those of a next-token reading.
- **The caveat this design must face.** N7's noisy head learned the next-token reading less well: its loss against the clean targets was 6.60, against 5.13 for the clean head. If it reads a weaker version of the same features, the overlap may still fall short of the clean head's. N8b tells convergence from partial sharing.
- **Why the ladder might persist.** Predicting a word eight places ahead rests more on what a passage is about than on the words just read. If that is so, the far reading's predictions are sensitive to different directions, whatever its errors. N8c tests this: the ladder persisting under a target-free measure would show that it reflects what the readings use, not only their errors.

## Competing hypotheses

| Hypothesis | o_F(noisy) − o_F(8) (N8a) | Noisy head's ratio (N8b) | o_F(2) − o_F(8) (N8c) |
|---|---|---|---|
| The claim: the old measure mixed in errors, and the readings also differ in what they use | ≥ 0.30 | ≥ 0.85 | ≥ 0.10 |
| The old measure mixed in errors, and that was all | ≥ 0.30 | ≥ 0.85 | Near 0: the ladder flattens |
| The noisy head is a weaker copy: shares more, but not all | Between 0 and 0.30 | Between 0.60 and 0.85 | Any |
| Errors were not the problem: the target-free measure agrees with the old one | Near 0 | ≤ 0.60 | About 0.25, as before |

## Refuting results

- **N8a** is refuted if the 97.5% interval for o_F(noisy) − o_F(8) lies wholly below +0.30, with a standard deviation of at most 0.05.
- **N8c** is refuted if the 97.5% interval for o_F(2) − o_F(8) lies wholly below +0.10, with a standard deviation of at most 0.05.

## Design

### Models

No training. The job loads the final weights saved to Google Drive by two earlier jobs:
- **N7's 20 models** (seeds 3801–3805; arms same, k4, k8, noisy), for N8a and N8b;
- **N6r's 20 models** (seeds 3701–3705; arms same, k2, k4, k8), for N8c.

### Windows

WikiText-103 test-split windows 513 to 768, each with 8 extra tokens. No measurement has used them: N6r used 1–256, and N7 used 257–512.

### Measures, per model

1. **The target-free measure:** subspaces, overlap, stabilities and agreement ratio, as above.
2. **The old measure on the same windows,** with the actual targets (the noisy head's targets corrupted once, with N7's fixed measurement seed). This serves as a loading check, and as a direct comparison.

## Decision rules

**Preconditions:**
1. **The estimator works.**
   - On a small linear-softmax model with an exactly computable Fisher, the sampled estimate's top subspace overlaps the exact one by at least 0.98.
   - Two identical heads give a target-free agreement ratio of exactly 1.
2. **The right models are loaded.** For every model, the old measure's overlap on N8's windows lies within 0.05 of the value recorded for it in N6r or N7. That would show the weights are the ones measured before, while allowing for the change of windows.
3. **Must fail, under the new measure.** In each set of models, the same-target arm's target-free agreement ratio is at least 0.95 in at least 4 of 5.
4. **Stable subspaces under the new measure.** Every head used in a decision rule has a target-free split-half stability of at least 0.80.

**N8a:**

| Outcome | Condition |
|---|---|
| Supported | The 97.5% interval for o_F(noisy) − o_F(8) lies wholly above +0.30 |
| Refuted | It lies wholly below +0.30, with a standard deviation of at most 0.05 |
| Some effect, size unclear | It excludes 0 but meets neither rule |
| Inconclusive | Everything else |

**N8b:**

| Outcome | Condition |
|---|---|
| Supported: the noisy head converges | Its target-free agreement ratio is at least 0.85 in at least 4 of 5 models |
| Refuted | It is at most 0.60 in at least 4 of 5 models |
| Intermediate | Anything else |

**N8c:**

| Outcome | Condition |
|---|---|
| Supported: the ladder persists | The 97.5% interval for o_F(2) − o_F(8) lies wholly above +0.10 |
| Refuted: it flattens | It lies wholly below +0.10, with a standard deviation of at most 0.05 |
| Some effect, size unclear | It excludes 0 but meets neither rule |
| Inconclusive | Everything else |

**Intervals.** N8a and N8c are both interval rules, so each uses a Bonferroni-corrected 97.5% t-interval with 4 degrees of freedom. N8b is a count rule.

**Why these thresholds.** They are carried over from N7 (0.30, 0.85, 0.60) and N6 / N6r (0.10), where they were set for the same comparisons, so that only the measure changes. This time they are checked against the new measure before any model is measured (see Calibration).

## Secondary and exploratory outcomes

- **S1:** both measures side by side for every arm and model: overlap, agreement ratio and stability.
- **S2:** how much the old measure's overlap exceeds or falls short of the new one, by arm. If errors drove the old measure, the gap should be largest for the noisy head and the far readings.
- **S3:** the target-free ladder for all of N6r's arms (k = 1, 2, 4, 8) and N7's (same, 4, 8, noisy).

## Calibration before queuing

On untrained full-size models (G1's seeds), on N8's windows:
- the estimator's analytic check and the identical-heads check;
- the target-free overlap and stability in each arm, against the stability threshold;
- the untrained same arm's target-free agreement ratio, which must fall well below 0.95, as before.

If a threshold or control does not work as written, an amendment comes back for approval first.

## Sanity checks before trusting results

- Every data file's fingerprint matches the manifest.
- Every model's weights load, and the loading check passes.
- Every number in the report is computed by code from saved JSON records. `n8_analyze.py` is written and committed before any real run, and tested on synthetic records covering every branch of the rules.

## Compute, where it runs, and stopping rule

- **Where:** one Colab job, `n8_fisher`, on a T4, resumable. It needs N6r's and N7's saved weights in the same Drive folder as before (`cdt-colab/n6r_halfoverlap/final/` and `cdt-colab/n7_harder/final/`).
- **Cost:** no training. 40 models, about one to two minutes each for both measures: under an hour in all.
- **Stopping rule:** exactly these measurements, each once. Nothing is added after seeing results except as a dated, exploratory entry in the deviation log.

## What each outcome would teach

| N8a | N8b | N8c | What we would learn | Next |
|---|---|---|---|---|
| Supported | Supported | Supported | The old measure mixed in errors, and the readings still differ in what they use. A cleaner form of N6r's finding survives. | Rerun the ladder on new seeds with the new measure; then ask whether the cost of sharing (N4) follows it |
| Supported | Supported | Refuted | The old measure's ladder came from errors. With them removed, readings further ahead use the same directions. | Drop the distance ladder as evidence of separate voices |
| Supported or some effect | Intermediate | Any | The noisy head is a weaker copy: it shares much of what the clean head uses, but not all | Train noisy heads longer or at lower noise |
| Refuted | Refuted | Any | Removing errors changes nothing: the noisy head really does use different directions | Rethink what "the same reading" means for a trained network |

## Limitations known now

- **The models are reused.** They are the same trained networks N7's conjecture came from. The measure is new and the predictions are fixed before it is computed, but the training randomness is shared. A supported result here would need a fresh-seed replication to carry full weight.
- **The target-free measure has its own limits.** It weights directions by how much they change the predictions, not by how much they matter to accuracy. A direction that shifts probability among unlikely words can count for as much as one that changes the likely ones.
- **The noisy head's weaker learning** may limit how far it can converge, whatever the measure.
- **Same limits as before:** 10% of the budget, one learning rate, one size, one dataset, one readout width.

## Deviation log

### Amendment A1, before any real run, 2026-10-10: approved by the author

Results seen before this entry:
- the promised calibration on **untrained** full-size models (G1's seeds 3001 and 3002) on N8's test-split windows 513–768;
- the analytic check;
- synthetic records;
- test-mode runs on tiny models.

No trained full-size model has been measured with the new measure.

**What the calibration found.**
- **The estimator works.** The analytic check passed: the sampled estimate's top subspace overlapped the exact Fisher's at 0.99998. Two identical heads gave an agreement ratio of 1 to within rounding.
- **Sharing the random draws doesn't inflate the overlap.** Drawing the two heads' tokens independently changed the untrained overlaps by at most 0.011 (0.973 against 0.977, and 0.956 against 0.967).
- **The new measure is stable enough.** Untrained split-half stabilities were 0.92 to 0.97, above the threshold of 0.80.
- **The must-fail control's calibration fails as written.** The calibration section required the untrained same-target arm's target-free agreement ratio to "fall well below 0.95, as before". It was 1.02 and 0.98. The old measure gave 0.50 on the same models.

**Why.** An untrained head predicts almost uniformly. Its Fisher at the state is then close to Jᵀ Pᵀ (WᵀW / V) P J, where J is the Jacobian from the ODE block's output to the final state, P is the readout projection and W is the output matrix. A random W gives WᵀW close to a multiple of the identity, so W drops out. Head B's projection starts as a copy of head A's, so every pair of heads starts with the same target-free subspace. The old measure differed at the start only because its gradients point toward the actual target's row of W, which differs between the heads.

**What this means for the design.**
- Under this measure every arm starts at about 1, so the must-fail control passing on trained models would no longer show that training made the heads converge.
- More seriously, the measure has to be able to call two readings separate. Otherwise a supported N8a or N8b, or a refuted N8c, could come from the measure rather than the models.

**The amendment.**
1. **Replace the low-end calibration with a must-differ control, applied to the trained models as precondition 3b.**
   - *The control:* for each same-target model, a copy whose head B is head A reading a randomly rotated final state: logits_B = W_A P_A Q ln_f(z), with Q a fixed random orthogonal matrix. It has head A's own weights but reads different directions of the state by construction.
   - *Precondition 3b:* in each set of models, the control's target-free agreement ratio is at most 0.60 in at least 4 of 5. That is the level N8b calls separate, so the measure must be able to reach the verdict N8b would call refuted.
   - *Calibration:* on the untrained models the control gave 0.078 and 0.078. A head with an independently drawn random projection gave 0.074 and 0.073. Random 16-dimensional subspaces give 0.06.
   - Precondition 3 (same-target ratio of at least 0.95) is unchanged.
2. **Add an exploratory measure, S4: how far each readout moved from where it started.**
   - It records the row-space overlap of each trained projection, P_A and P_B, with the initial projection both heads started from, and of P_A with P_B.
   - The reason: every pair of heads starts at about 1 under this measure. A noisy head that learned little could score high only by staying where it began, if head A also moved little. If head A moved far and the noisy head followed it, that is convergence. S4 lets the reader tell these apart. It enters no decision rule.

**Clarifications of the written design, not changes:**
- Preconditions 2 to 4 are applied to each set of models separately: N7's models for N8a and N8b, N6r's for N8c. Precondition 1 applies to all three claims.
- *The analytic check's toy model:* logits = hWᵀ + b, with b putting most of the probability on 17 tokens.
  - A first version planted the top subspace in W instead. It passed, but a sampler that drew tokens uniformly passed it too (0.996).
  - In the version used, the uniform sampler scores 0.72, so the check can fail.
  - Its threshold of 0.98 is unchanged.

**A risk that can't be calibrated without trained models.** The trained same-target heads might fall below 0.95 under the new measure, which would make N8 untestable. In N7's trained same-target models, the two readout projections overlapped at only 0.71.

## References

Kunstner, F., Balles, L., & Hennig, P. (2019). *Limitations of the empirical Fisher approximation for natural gradient descent*. arXiv. https://doi.org/10.48550/arxiv.1905.12558

Martens, J. (2014). *New insights and perspectives on the natural gradient method*. arXiv. https://doi.org/10.48550/arxiv.1412.1193

Patrini, G., Rozza, A., Menon, A. K., et al. (2017). Making deep neural networks robust to label noise: A loss correction approach. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2233–2241. https://doi.org/10.1109/cvpr.2017.240
