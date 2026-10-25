# Result note: N8

Date: 2026-10-10    Preregistration: `N8_PREREGISTRATION.md`, with amendment A1 (a must-differ control)    Generated report: `N8_RESULTS.md`

## In brief

N7's note guessed that the overlap measure used so far sees where a reading's errors fall, and not only which directions it uses, because it is built from gradients against the real answers. N8 measured the same trained models again with a target-free version, in which each answer is drawn from the head's own prediction, so that errors play no part. The head trained on scrambled answers still shared little with the clean head: 0.127 more than the head reading eight words ahead, where 0.30 was needed, and an agreement ratio of 0.315 to 0.339, where 0.85 was needed, so N8a and N8b are refuted. The overlap still fell the further ahead a head reads (0.95 for the same word, then 0.44, 0.25 and 0.19 at 2, 4 and 8 words ahead), so N8c is supported, and since the two measures agreed to within 0.045 on average, N7's explanation is refuted.

## The arms and measures

N8 trained nothing. It measured 40 models already trained in two earlier tests: N7's 20 (seeds 3801 to 3805) for N8a and N8b, and N6r's 20 (seeds 3701 to 3705) for N8c. In every model, two heads read the same final internal state, each through its own readout of 64 directions. Head A, *the clean head*, predicts the next word from the real answers. The arms differ in what head B predicts.

| Arm | Also called in this note | Head B predicts | Models |
|---|---|---|---|
| `same` | Same target | The next word, as head A does. This is the must-fail control: the measure must not call these two heads different | N7's and N6r's |
| `k2` | 2 ahead, k = 2 | The word 2 places ahead | N6r's |
| `k4` | 4 ahead, k = 4 | The word 4 places ahead | N7's and N6r's |
| `k8` | 8 ahead, k = 8 | The word 8 places ahead | N7's and N6r's |
| `noisy` | Noisy next token, the noisy head | The next word from scrambled answers: in training, 80% of its answers were replaced by words drawn at random in proportion to how common they are | N7's |

- **The rotated control,** or must-differ control, was added by amendment A1 and built in each `same` model. It is a second copy of head A that reads a randomly rotated copy of the state. It has head A's own weights but reads different directions by construction, so the measure must be able to call it separate.
- **Measurement windows:** 256 fixed passages of the WikiText-103 test text (windows 513 to 768), which no earlier measurement had used.
- **The old measure,** used from N4 to N7: for each head, the 16 directions of the internal state, at the continuous block's output, along which the gradient of its loss against its actual answers is largest on average (the top eigenvectors of the gradient covariance). The literature calls this covariance the *empirical Fisher*.
- **The target-free measure, o_F,** new in N8: the same, except that each answer is drawn from the head's own predicted distribution instead of taken from the text. Its expected value is the *Fisher information* of the head's predictions: it picks out the directions of the state that most change what the head predicts, whatever the answers turned out to be. The two heads share one random draw per position, so two identical heads score exactly 1.
- **The overlap:** how far two heads' 16 directions coincide, as the mean squared cosine of the angles between them. It is 1 for the same directions, and random directions score about 0.06. o_F(noisy), o_F(2) and o_F(8) are the target-free overlaps of the noisy, two-ahead and eight-ahead heads with head A.
- **The agreement ratio:** the overlap corrected for measurement noise. Head A's directions from one half of the passages are compared with head B's from the other half. The result is divided by the geometric mean of the two heads' *split-half stabilities*, which say how well each head's directions agree with themselves across the two halves.
- **Readout drift (S4),** added by amendment A1 and exploratory: how far each trained readout's 64 directions still overlap the readout both heads started from (1 means it did not move, and random directions would give 0.25), and how far head A's readout overlaps head B's.
- **The ideal noisy copy,** in "Which kind of result this is": a head whose prediction is exactly 0.2·p + 0.8·(word frequencies), where p is the clean head's prediction. It is what a perfect head trained on 80% scrambled answers would predict. The plain summary calls it the perfect noisy copy.
- **N8a (primary)** claims that, with errors removed, the noisy head shares what the clean head uses: o_F(noisy) − o_F(8) ≥ 0.30. **N8c** claims that the readings further ahead still use different directions: o_F(2) − o_F(8) ≥ 0.10. Each is judged on a Bonferroni-corrected 97.5% interval over the five seeds (a t-interval with 4 degrees of freedom). A claim is *supported* if its interval lies wholly above its threshold, *refuted* if it lies wholly below with a standard deviation of at most 0.05, *some effect, size unclear* if it excludes 0 but meets neither rule, and *inconclusive* otherwise.
- **N8b** claims that the noisy head converges with the clean head. It is *supported* if its target-free agreement ratio is at least 0.85 in at least 4 of 5 models, *refuted* if it is at most 0.60 in at least 4 of 5, and *intermediate* otherwise.
- **Preconditions,** without which the claims would be *untestable*. Each set of models must pass them separately:
  - *the estimator works:* on a small model whose Fisher can be computed exactly (the *analytic check*), the sampled estimate's top directions overlap the exact ones by at least 0.98, and two identical heads score exactly 1;
  - *the right models are loaded:* the old measure on N8's windows lies within 0.05 of the value recorded for every model;
  - *must fail:* the `same` arm's target-free agreement ratio is at least 0.95 in at least 4 of 5 models;
  - *must differ* (amendment A1): the rotated control's ratio is at most 0.60 in at least 4 of 5;
  - *stability:* every head used in a rule has a split-half stability of at least 0.80.
- **Amendment A1,** approved before any real run, added the must-differ control and S4. Calibration had shown that under the target-free measure untrained heads already score about 1, so the must-fail control alone could not show that the measure can call two heads separate.
- **Other names.** N4, N6r, N7 and N9 are other tests in this folder, each with its own note. *N6r's ladder* is the finding that the overlap with the next-word head falls the further ahead head B reads. Runs were trained on Google Colab and saved to Google Drive; a *session* is one sitting of the Colab job.

*The summary, this key and the records section at the end were added on 2026-10-25 so that the note can be read on its own. Nothing else was changed.*

**How the run went.**
- Two earlier sessions stopped with an error before measuring any model. Both were caused by a bug in the new control: a matrix left on the CPU. The second session had started 25 seconds before the fix was pushed. Both are recorded in the deviation log.
- The third session measured all 40 models with no Drive failures.
- Nothing was changed after amendment A1 was approved, apart from the one-line device fix.

## What we expected

N7's note proposed that the overlap measure used from N4 to N7 sees where each reading's errors fall as well as which features it uses, because it is built from gradients against the actual targets. N8 measured the same trained models again with a target-free version: the same gradient covariance, with each target drawn from the head's own predictions. Its expectation is the Fisher information of the head's predictions.

- **N8a (primary).** With errors removed, the noisy next-token head shares what the clean head uses: o_F(noisy) − o_F(8) ≥ 0.30.
- **N8b.** And it converges: its target-free agreement ratio is at least 0.85 in at least 4 of 5 models.
- **N8c.** The readings further ahead still use different directions: o_F(2) − o_F(8) ≥ 0.10 on N6r's models.

## What happened

**N8a and N8b are refuted. N8c is supported.**

| Head B | N7's models: target-free overlap / ratio | Old measure, same windows | N6r's models: target-free overlap / ratio | Old measure, same windows |
|---|---|---|---|---|
| Same target | 0.944 / 0.969 | 0.962 | 0.945 / 0.964 | 0.951 |
| 2 ahead | | | 0.436 / 0.451 | 0.471 |
| 4 ahead | 0.245 / 0.251 | 0.290 | 0.249 / 0.254 | 0.294 |
| 8 ahead | 0.183 / 0.189 | 0.225 | 0.188 / 0.193 | 0.229 |
| Noisy next token | 0.310 / 0.326 | 0.309 | | |

- **N8a refuted.** o_F(noisy) − o_F(8) = +0.127, 97.5% interval [+0.105, +0.150], standard deviation 0.015. That lies wholly below the 0.30 required. Every seed was between +0.108 and +0.142.
- **N8b refuted.** The noisy head's target-free agreement ratio was 0.315 to 0.339 in all 5 models, far below 0.60.
- **N8c supported.** o_F(2) − o_F(8) = +0.248, 97.5% interval [+0.214, +0.281], standard deviation 0.021. Every seed was between +0.211 and +0.266.

All preconditions held, in both sets of models:
- **The estimator works.** The analytic check scored 1.0000 against the exact Fisher, where a sampler drawing tokens uniformly scores 0.72. Two identical heads gave a ratio of exactly 1.
- **The right models were loaded.** The old measure on N8's windows was within 0.035 of the value recorded for every model.
- **Must fail.** The same-target heads' target-free ratio was 0.948 to 0.989 in N7's models (4 of 5 at least 0.95) and 0.951 to 0.976 in N6r's (5 of 5). Seed 3801 missed by 0.002.
- **Must differ (amendment A1).** Head A reading a randomly rotated state scored 0.134 to 0.193, far below 0.60. So the measure can call a head separate.
- **Stability.** Every head was at least 0.92.

## What this teaches

- **N7's explanation is refuted.** The old measure did not differ from the target-free one by mixing in errors. In every arm the two agree to within 0.045 on average. For the noisy head, where errors should have mattered most, they agree to 0.001 (0.309 against 0.310). A dated correction has been added to `N7_NOTE.md`.
- **In hindsight this was predictable, and I should have seen it.** The empirical Fisher and the Fisher have the same expectation when a model's predicted distribution matches the distribution of its targets (Kunstner et al., 2019, cited in the preregistration). A trained head is roughly calibrated to its own targets, the noisy head included: it learns to predict mostly word frequencies, and that is mostly what its targets are. So, for trained heads, the old measure was already close to the target-free one. The preregistration did not consider this, and the explanation N8 tested was weaker than it looked.
- **The ladder reflects what the readings' predictions depend on.** Under a measure with no targets in it, the overlap still falls with distance: 0.95, 0.44, 0.25, 0.19. N6r's ladder therefore measures which directions of the state each reading's predictions are sensitive to, not where its errors fall.
- **The noisy head really did move away from the clean head.** It did not just stay where it started; this is the alternative S4 was added to rule out.
  - Its readout moved from the projection both heads started with about as far as the four-ahead head's did (overlap with the start 0.57, against 0.56).
  - It also moved away from head A's: 0.48, between the two-ahead head (0.51) and the four-ahead head (0.42).
  - Head A moved too (0.69).
- **So, measured by what a head's predictions are sensitive to, a noisy copy of the next-token reading is not the same reading.** Neither "how different are the targets" (N7) nor "where do the errors fall" (N8) explains why.

## Which kind of result this is

- **A refutation of N8a and N8b as stated,** under a sound design. Every control held, including the new must-differ control, and the run was sensitive enough (standard deviation 0.015 against a margin of 0.30).
- **One gap in the design limits what the refutation says about the noisy head.** This is a criticism of the design, not a rescue: the claim as stated and tested stays refuted.
  - The must-fail control checked that the measure calls two heads with the *same targets* the same. It did not check that the measure would call an *ideal noisy copy* the same.
  - Even a head computing exactly 0.2·p + 0.8·(word frequencies), where p is the clean head's prediction, weights the directions of p differently in its Fisher. It largely drops the contribution of words the clean head finds unlikely.
  - So it is not known whether this measure could ever have scored a perfect noisy copy near 0.85. Until that is checked, the result does not separate two explanations:
    - the noisy head learned a different, weaker reading (its loss against the clean targets was 6.60, against 5.13);
    - the measure cannot recognise a noisy copy as the same reading.
  - **Update, 2026-10-10, from N9.** N9 separated the two. The measure recognises an ideal noisy copy of the clean head as the same reading (0.955–0.995). The trained noisy head is not close to that copy (0.305–0.349). So the noisy head learned a different, weaker reading, and the conclusion above holds as written. See `N9_NOTE.md`.

## What this does not change

- Every earlier verdict stands, including N6r's ladder and N7's refutations.
- N4's result (sharing 64 readout directions costs two readings 0.088 nats) stands. It is measured in loss, not overlap.
- The paper's original claims stay refuted.

## Exploratory observations, not confirmatory

- **The target-free measure gives slightly lower overlaps for the far readings than the old one.** It was 0.035 to 0.045 lower for k = 2, 4 and 8, and 0.006 to 0.018 lower for the same-target heads. The old measure's actual targets add a little shared structure, probably frequent words, which every head's targets contain. The effect is small, and it does not change the order.
- **The noisy head's target-free ratio (0.33) is above the four-ahead head's (0.25),** as under the old measure (0.32 against 0.30). The noisy head still shares more than the far readings, but nowhere near the same-target level.
- **Untrained models confirmed the calibration.** All ten untrained models scored 0.97 to 1.01 between their heads, and 0.07 to 0.09 for the rotated control.

## Limitations

- **The models are reused** from N6r and N7. A fresh-seed replication would be needed for the supported N8c to carry full weight. N8c also agrees with the old measure on the same models, so it is less a new finding than a check that N6r's ladder survives a change of measure.
- **One target-free measure.** It weights directions by how much they move the head's predictions, not by how much they matter to accuracy.
- **The same limits as before:** 10% of the budget, one learning rate, one size, one dataset, one readout width (64).

## Next conjectures

1. **Can this measure recognise an ideal noisy copy?**
   - *The test:* on each trained clean head, construct the head whose prediction is exactly 0.2·p_A + 0.8·(word frequencies), and measure its target-free overlap and agreement ratio with head A.
   - *If it scores near 1,* the measure could have supported N8a and N8b, and N7's noisy head really did learn a different reading. Training on noisy targets, not the task, would then be what separates it.
   - *If it scores low,* this measure cannot tell "the same reading, made noisy" from "a different reading". The noisy-head question would then need a measure that compares what heads compute rather than how sensitive their predictions are.
   - *Cost:* no training, but the weights are on Drive, so it needs a short Colab job.
2. **Does the cost of sharing grow along the ladder?** Repeat N4's forced-sharing test at k = 2, 4 and 8. The ladder is now established under two measures. Whether it matters for what sharing *costs* is the open functional question.

## References

Kunstner, F., Balles, L., & Hennig, P. (2019). *Limitations of the empirical Fisher approximation for natural gradient descent*. arXiv. https://doi.org/10.48550/arxiv.1905.12558

## Records

- **Raw results:** `results/n8/` and `results/n8_log.txt`, from the `colab-results` branch, run `n8_fisher/20261010-154616`. This was the third session. The two before it, `n8_fisher/20261010-143934` and `n8_fisher/20261010-144219`, stopped before measuring any model.
- **Code:** commit b30f6ec. `n8_analyze.py` writes `N8_RESULTS.md` from those records; the job is `n8_fisher.py`.
