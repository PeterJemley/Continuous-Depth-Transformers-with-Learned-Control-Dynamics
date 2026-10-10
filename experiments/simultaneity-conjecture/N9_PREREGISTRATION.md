# Preregistration N9: can the target-free measure recognise an ideal noisy copy of a reading?

Date: 2026-10-10    Status: **approved by the author, 2026-10-10, with amendment A1 (512 windows instead of 256, approved).** Queued; no real N9 run has started.

Type: confirmatory for N9a (primary) and N9b, each with its own decision rule, on models already trained in N7, with heads constructed from them and not yet measured. Everything else is secondary or exploratory and labelled so.

Parent documents: `N8_NOTE.md` (its "Which kind of result this is" and next conjecture 1), `N7_NOTE.md`, and `PLAIN_SUMMARY.md`.

**Relation to earlier results.**
- **What N8 found.** Under a target-free measure, the Fisher information of each head's predictions, N7's noisy next-token head still shared about a third of its sensitive directions with the clean head (agreement ratio 0.32–0.34). It did not converge (N8a and N8b refuted). The distance ladder persisted (N8c supported).
- **The gap N8 left.** N8's controls checked that the measure calls two heads with the same targets the same, and a rotated head separate. They did not check that it would call an *ideal* noisy copy of a reading the same. So N8 cannot tell apart two explanations:
  - the noisy head learned a different, weaker reading;
  - the measure cannot recognise a noisy copy as the same reading, however perfect.
- **What N9 does.** It builds the ideal noisy copy of each clean head exactly, with no training, and asks two questions of the measure:
  - does it call the ideal copy the same as the clean head?
  - is the trained noisy head close to the ideal copy?
- **What stands regardless.** Every earlier verdict stands, including N8a's and N8b's refutations. N9 asks what those refutations mean. It does not re-score them.

## The ideal noisy copy

N7's noisy targets were the next token, replaced with probability ρ = 0.8 by a draw from the training data's word frequencies u. If head A predicts p_A, the best possible prediction of such targets is

q = (1 − ρ) · p_A + ρ · u   (Patrini et al., 2017).

The *ideal noisy copy* of head A is a head whose prediction is exactly q. It is built from head A's own weights, in log space for stability: its logits are logaddexp(log(1 − ρ) + log p_A, log ρ + log u). It computes the same reading as head A in every sense except the noise. Every input that changes p_A changes q, and nothing else does.

**Why it is not obvious that the measure will recognise it.** Write G for the Jacobian of head A's logits with respect to the state, and C_p = diag(p) − p pᵀ.
- Head A's Fisher is Gᵀ C_p G.
- The copy's is (1 − ρ)² Gᵀ C_p diag(1/q) C_p G.
- Up to an overall constant, each vocabulary item's contribution is reweighted by p/q relative to head A's.
  - For words head A expects (p well above u), that factor is about 1/(1 − ρ) = 5, the same for all of them.
  - For frequent words head A thinks unlikely here (p well below u), it falls towards 0.
- So the copy's sensitive directions are head A's, with the contribution of unexpected frequent words removed. Whether that changes the top 16 directions much depends on how much of head A's Fisher comes from such words. That is an empirical question, and this test answers it.

## The claims

**N9a (primary): the measure recognises an ideal noisy copy.** On each N7 noisy-arm model, the target-free agreement ratio between head A and its ideal noisy copy (ρ = 0.8) is at least 0.85 in at least 4 of 5 models.

**N9b: the trained noisy head is close to the ideal copy.** On the same models, the target-free agreement ratio between the trained noisy head B and head A's ideal noisy copy is at least 0.85 in at least 4 of 5 models.

### Competing hypotheses

| Hypothesis | Ratio of head A with the copy (N9a) | Ratio of the trained noisy head with the copy (N9b) |
|---|---|---|
| The measure is fine. The noisy head learned a different reading, so N8's conclusion holds as written | ≥ 0.85 | ≤ 0.60 |
| The measure cannot recognise a noisy copy, and the noisy head is near the ideal copy. N7's and N8's "separation" was the measure's weighting | ≤ 0.60 | ≥ 0.85 |
| The measure cannot recognise a noisy copy, and the noisy head is not near the ideal copy either | ≤ 0.60 | ≤ 0.60 |
| Partial on either count | between | between |

**Both ratios at least 0.85 is not expected.**
- The overlap's distance, √(1 − overlap), obeys the triangle inequality, and all three overlaps are computed on the same windows.
- So if head A and the trained noisy head each overlapped the copy by 0.85 or more, they would have to overlap each other by at least about 0.40 on N9's windows. N8 measured about 0.31, and N9 measures that pair again (S3).
- The ratios differ from the overlaps only by the heads' stabilities, which are near 1. So the analysis reports this outcome as a likely bug, not as a finding.

## Refuting results

- **N9a** is refuted if head A's ratio with its ideal copy is at most 0.60 in at least 4 of 5 models.
- **N9b** is refuted if the trained noisy head's ratio with the ideal copy is at most 0.60 in at least 4 of 5 models.

## Design

### Models

No training. The job loads the final weights of N7's noisy-arm models (seeds 3801–3805) from Google Drive (`cdt-colab/n7_harder/final/noisy_seed380*.pt`). For S4 it also loads N7's other 15 models, whose head A is a clean next-token head too.

### Constructed heads

All are built from a trained model's head A, without training:
- **The ideal noisy copy at ρ = 0.8,** as above, for N9a and N9b.
- **Copies at ρ = 0, 0.2, 0.5 and 0.95,** for S1. The copy at ρ = 0 is head A itself, routed through the same log-space code (the must-match control).
- **The rotated-readout control from N8,** head A reading a randomly rotated state (the must-differ control).

u is the unigram distribution of the WikiText-103 training tokens, computed as in N7 (`n7_harder.unigram`). It is the distribution N7's noise was drawn from.

### Measure

N8's target-free measure, unchanged:
- each head's target is drawn from its own predicted distribution, with one uniform per position shared by all heads (common random numbers);
- the covariance of the gradient of −log prediction at the ODE block's output;
- top-16 eigenvectors, overlap, split-half stability and the agreement ratio.

For the noisy head and the copies, "its own predicted distribution" is q, and the gradient flows through q into head A's logits and the state.

### Windows

WikiText-103 test-split windows 769 to 1024, each with 8 extra tokens. No measurement has used them: N6r used 1–256, N7 257–512, and N8 513–768.

### Measures, per noisy-arm model

1. Target-free ratio, overlap and stabilities for each of these pairs:
   - head A with the copy at ρ = 0.8 (N9a);
   - the trained noisy head with the copy at ρ = 0.8 (N9b);
   - head A with the trained noisy head (S3, N8's comparison on new windows);
   - head A with the copies at ρ = 0, 0.2, 0.5 and 0.95 (S1);
   - head A with the rotated control.
2. Comparisons in prediction space, which do not depend on how the Fisher weights directions (S2). Each is averaged over every position of the windows:
   - the KL divergence from the trained noisy head's prediction to the ideal copy's, and to u alone;
   - each of the two heads' losses against the clean next-token targets and against noisy targets (corrupted once with N7's fixed measurement seed).

## Decision rules

**Preconditions:**
1. **The estimator works on mixtures.**
   - On N8's small linear-softmax model, with the head replaced by its ideal noisy copy (a random unigram u, ρ = 0.8), the sampled estimate's top subspace overlaps the exact Fisher (1 − ρ)² Gᵀ C_p diag(1/q) C_p G by at least 0.98.
   - Two identical heads give a ratio of exactly 1.
2. **Must match.** In every model, the copy at ρ = 0 gives a ratio with head A of at least 0.999. It is head A by construction, and anything less means the log-space code changes the head.
3. **Must differ.** In at least 4 of 5 models, the rotated control's ratio is at most 0.60, as in N8.
4. **The right models are loaded.** In every model, head A's target-free overlap with the trained noisy head on N9's windows lies within 0.05 of the value N8 recorded for it.
5. **Stable subspaces.** Head A, the trained noisy head and the copy at ρ = 0.8 each have a target-free split-half stability of at least 0.80 in every model.

**N9a:**

| Outcome | Condition |
|---|---|
| Supported: the measure recognises an ideal noisy copy | Head A's ratio with its copy is at least 0.85 in at least 4 of 5 models |
| Refuted: it does not | It is at most 0.60 in at least 4 of 5 models |
| Intermediate | Anything else |

**N9b:**

| Outcome | Condition |
|---|---|
| Supported: the trained noisy head is close to the ideal copy | Its ratio with the copy is at least 0.85 in at least 4 of 5 models |
| Refuted | It is at most 0.60 in at least 4 of 5 models |
| Intermediate | Anything else |

**Why these thresholds.** 0.85 and 0.60 are N7's and N8's thresholds for "converged" and "separate", carried over so that the copy is judged by the standard the noisy head was judged by. Both rules are count rules, so no interval correction is needed.

## Secondary and exploratory outcomes

- **S1: dose-response.** Head A's ratio with its copy at ρ = 0, 0.2, 0.5, 0.8 and 0.95. It shows how fast the measure's recognition falls as the noise grows, and whether N7's original rate of 0.6 would have mattered.
- **S2: prediction space.** How close the trained noisy head is to the ideal copy in what it predicts, measured in KL and in loss on clean and noisy targets. This answers "did the noisy head learn the ideal noisy reading?" without the Fisher's weighting.
- **S3: N8b on fresh windows.** Head A with the trained noisy head, a direct replication of N8's 0.32–0.34 on new windows.
- **S4: other clean heads.** N9a's comparison for the clean head A of N7's same-target, four-ahead and eight-ahead models, to see whether the answer depends on which model the clean head comes from.

## Calibration before queuing

On untrained full-size models (G1's seeds 3001 and 3002), on N9's windows:
- the analytic check on mixtures, and the identical-heads check;
- the must-match control (ρ = 0) and the must-differ control;
- the copy's ratio and stability at each ρ, recorded and not used for any threshold. An untrained head's predictions are close to uniform, so its copy weights rare words heavily, unlike a trained head's. The untrained values say little about the trained ones.

If a threshold or control does not work as written, an amendment comes back for approval first.

## Sanity checks before trusting results

- Every data file's fingerprint matches the manifest.
- Every model's weights load, and the loading check passes.
- The unigram u sums to 1. It matches N7's, which is recorded by N7's noise check: its five most frequent tokens.
- Every number in the report is computed by code from saved JSON records. `n9_analyze.py` is written and committed before any real run, and tested on synthetic records covering every branch of the rules.

## Compute, where it runs, and stopping rule

- **Where:** one Colab job, `n9_noisycopy`, on a T4, resumable. It needs N7's saved weights in the same Drive folder as before (`cdt-colab/n7_harder/final/`).
- **Cost:** no training. Five noisy-arm models with about eight measurements each, plus 15 models for S4 with one each: about 20 minutes.
- **Stopping rule:** exactly these measurements, each once. Nothing is added after seeing results except as a dated, exploratory entry in the deviation log.

## What each outcome would teach

| N9a | N9b | What we would learn | Next |
|---|---|---|---|
| Supported | Refuted | The measure recognises an ideal noisy copy, and the trained noisy head is not one. So N7's noisy head really learned a different reading, and N8's conclusion holds as written. Noisy training changes which directions a head comes to read, beyond adding noise. | Ask why: train noisy heads at lower ρ, or for longer, and see whether they move toward the copy (S1 and S2 suggest where) |
| Refuted | Supported | The measure cannot recognise a noisy copy, and the trained noisy head is close to the ideal copy. N7's and N8's "separation" of the noisy head came from how the Fisher weights directions, not from the head reading something different. | Compare heads by what they compute (prediction-space measures, S2) rather than by sensitivity, and revisit what the ladder means under such a measure |
| Refuted | Refuted | The measure cannot recognise a noisy copy, and the noisy head is not near one either. Both explanations contribute, and the measure cannot separate them. | Move to prediction-space comparisons |
| Supported or intermediate | Intermediate | The noisy head is partway to the ideal copy | Use S2 to tell how far, then decide between longer training and a different measure |
| Refuted or intermediate | Intermediate | The measure is limited, and how close the noisy head is to the copy cannot be settled by this measure | Prediction-space comparisons (S2) carry the question |
| Supported | Supported | Inconsistent with N8's own measurement (see the triangle inequality above) | Look for a bug before interpreting anything |

## Limitations known now

- **The models are reused,** for the third time: N7 trained them, and N8 measured them. The constructed heads are new, and the predictions are fixed before they are measured. But the training randomness is shared with every earlier test on these models.
- **The ideal copy uses the exact ρ and u** that generated N7's noise. A trained noisy head could only approximate them. S2 measures how closely.
- **The copy is built from head A's parameters,** so it shares head A's readout projection exactly. That is the point of the comparison, but it means N9a tests the measure's weighting, not anything a trained network did.
- **Same limits as before:** 10% of the budget, one learning rate, one size, one dataset, one readout width.

## Deviation log

### Amendment A1, before any real run, 2026-10-10: approved by the author

Results seen before this entry:
- the analytic check;
- calibration on **untrained** full-size models (G1's seeds 3001 and 3002) on N9's windows;
- synthetic records;
- a test-mode run on tiny models.

No trained full-size model has been measured.

**What the calibration found.**
- **The estimator works on mixtures.** The analytic check passed at 0.994.
- **The controls work.** Identical heads gave a ratio of 1 to within rounding. The must-match copy (ρ = 0) gave exactly 1.000000. The must-differ control gave 0.077 and 0.079.
- **The ideal copy's subspace is noisy at 256 windows.** At ρ = 0.8, with the full training-set unigram, its split-half stability was 0.849 and 0.818, within 0.05 of the 0.80 threshold of precondition 5. At ρ = 0.95 it was 0.65–0.66.
  - The reason: 80% of the copy's sampled targets come from the unigram part of its prediction. For those tokens the copy's gradient is small, because head A thinks most of them unlikely. So fewer positions carry information than for head A, whose stabilities were 0.91–0.94.
  - With 512 windows (769–1280), the copy's stability rose to 0.884 and 0.874, and head A's to 0.94–0.99.
- **The tiny test-mode run** failed precondition 5, mostly through the copy. That is not decisive, since its models are tiny and its windows few, but it points the same way.
- How stable the copy will be in *trained* models cannot be calibrated in advance. N8's trained noisy head, a learned mixture of the same kind, had stabilities of 0.93–0.97.

**The amendment.** Measure on WikiText-103 test-split windows 769–1280, 512 windows instead of 256.
- No earlier test has used windows 1025–1280.
- Nothing else changes: the measure, the heads, every threshold and every rule are as written.
- The cost roughly doubles, to about 40 minutes.

**Clarifications of the written design, not changes:**
- **S2's "KL divergence to u alone" is infinite.** u is zero for 3,369 tokens that never occur in the training data, and the noisy head's softmax gives every token some probability. S2 instead reports two things:
  - the KL divergence from the noisy head to the copy at each ρ of S1 (0 to 0.95), which shows which noise rate the noisy head's predictions are closest to;
  - the noisy head's total probability on the tokens where u is zero.
- **The analytic check's toy.** A first version, N8's toy with a random unigram, passed, but a wrong estimator also passed it: measuring the copy as if it were head A scored 0.999. In the version used, head A gives some probability to tokens the unigram favours much more. The wrong estimator then scores 0.66–0.80, and the real one 0.994–0.995. The threshold of 0.98 is unchanged.
- **Precondition 2 (must match) is applied to every measured model,** the S4 models included.

## References

Kunstner, F., Balles, L., & Hennig, P. (2019). *Limitations of the empirical Fisher approximation for natural gradient descent*. arXiv. https://doi.org/10.48550/arxiv.1905.12558

Patrini, G., Rozza, A., Menon, A. K., et al. (2017). Making deep neural networks robust to label noise: A loss correction approach. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2233–2241. https://doi.org/10.1109/cvpr.2017.240
