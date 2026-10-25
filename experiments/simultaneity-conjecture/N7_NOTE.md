# Result note: N7

Date: 2026-10-09    Preregistration: `N7_PREREGISTRATION.md`, with amendment A1 (noise rate 0.8)    Generated report: `N7_RESULTS.md`

## In brief

An earlier test, N6r, found that a head reading further ahead shares fewer directions with the next-word head, and N7 asked whether that sharing tracks how different two readings are or how hard they are. It added a head trained on scrambled answers: the next word, with 80% of its training answers replaced, which is the same reading made harder. That head shared only 0.066 more with the clean head than a head reading eight words ahead did, where the primary claim needed 0.30, and its agreement ratio was 0.30 to 0.33, where the second claim needed 0.85. Both claims are refuted, and because this head was also the hardest yet shared more than the eight-ahead head, neither difference nor difficulty orders the results on its own.

## The arms and measures

Every model is the paper's continuous-depth transformer, trained with exact gradients for 10% of the WikiText-103 training budget. Two heads read the same final internal state, each through its own readout of 64 directions, and head B's readout starts as a copy of head A's. Head A always predicts the next word from the real answers: it is *the clean head*. The arms differ only in what head B predicts. Each arm was trained from five seeds (3801 to 3805), 20 runs in all, and all four arms of a seed saw the same inputs in the same order.

| Arm | Also called in this note | Head B predicts | Role |
|---|---|---|---|
| `same` | Same target, the same arm | The next word, as head A does | The must-fail control: two heads reading the same word, which the measure must not call different |
| `k4` | 4 ahead, k = 4 | The word 4 places ahead | A reading known from N6r to share little with the next word, and the bar for how hard the noisy head must be |
| `k8` | 8 ahead, k = 8 | The word 8 places ahead | The comparison for N7a |
| `noisy` | Noisy next token, the noisy head | The next word from scrambled answers: in training, each answer is replaced, with probability 0.8, by a word drawn at random in proportion to how common it is | The new reading: the same reading as head A, made harder |

- **Measurement windows:** 256 fixed passages of the WikiText-103 test text (windows 257 to 512), which no earlier measurement had used.
- **The sensitivity subspace:** for each head, the 16 directions of the internal state, at the continuous block's output, along which the gradient of the head's loss against its own answers is largest on average over the passages (the top eigenvectors of the gradient covariance). For the noisy head, its own answers are the scrambled ones.
- **The overlap, o:** how far two heads' 16 directions coincide, as the mean squared cosine of the angles between them. It is 1 for the same directions, and random directions score about 0.06. o(noisy) and o(8) are the overlaps of the noisy head and the eight-ahead head with head A.
- **The agreement ratio:** the overlap corrected for measurement noise. Head A's directions from one half of the passages are compared with head B's from the other half. The result is divided by the geometric mean of the two heads' *split-half stabilities*, which say how well each head's directions agree with themselves across the two halves. It is 1 for the same reading.
- **Losses** are in nats, on the validation text. A head's *own targets* are the answers it was trained on, scrambled for the noisy head. The *clean targets* are the real next words.
- **N7a (primary)** claims that sharing tracks how different the readings are, not how hard: o(noisy) − o(8) ≥ 0.30. It is judged on a 95% interval over the five seeds (a t-interval with 4 degrees of freedom). It is *supported* if the interval lies wholly above +0.30, *refuted* if it lies wholly below +0.30 with a standard deviation of at most 0.05, *some effect, size unclear* if it excludes 0 but meets neither rule, and *inconclusive* otherwise.
- **N7b** claims that a harder copy of the same reading converges with it. It is *supported* if the noisy head's agreement ratio is at least 0.85 in at least 4 of 5 models, *refuted* if it is at most 0.60 in at least 4 of 5, and *intermediate* otherwise.
- **Preconditions,** without which the claims would be *untestable*:
  - *data and setup:* among these checks, the *noise check* counts the share of answers actually replaced, and the *identical-heads check* requires two copies of one head to score exactly 1;
  - *must fail:* the `same` arm's agreement ratio is at least 0.95 in at least 4 of 5 models;
  - *stability:* every head in `k4`, `k8` and `noisy` has a split-half stability of at least 0.80;
  - *hardness:* the noisy head's loss on its own targets is at least the `k4` head's, in at least 4 of 5 seeds.
- **Amendment A1,** approved before any real run, raised the share of scrambled answers from the planned 60% to 80%. Calibration had shown that at 60% the noisy head would probably not be as hard as `k4`.
- **Other names.** N4, N6r and N8 are other tests in this folder, each with its own note. *N6r's ladder* is its finding that the overlap with the next-word head falls the further ahead head B reads (0.49, 0.31 and 0.23 at 2, 4 and 8 words ahead). The *target-free measure*, in the dated correction, is N8's version of the overlap: it draws each answer from the head's own prediction instead of using the real one. Runs were trained on Google Colab and saved to Google Drive. A *resume* restarts a run from its saved copy, and a *skipped fp16 step* is a training step dropped because its 16-bit arithmetic overflowed.

*The summary, this key and the records section at the end were added on 2026-10-25 so that the note can be read on its own. Nothing else was changed.*

**How the run went.** One Colab session ran all 20 models and their measurements, with no resumes, no skipped fp16 steps and no Drive failures. Nothing was changed after approval.

## What we expected

Head A predicts the next token. Head B predicts the next token (the must-fail control), the token 4 or 8 ahead, or the next token with 80% of its targets replaced by words drawn at their normal frequencies (the noisy head). The noisy head is the same reading as the next token, made harder.

- **N7a (primary).** Sharing tracks how different the readings are, not how hard: the noisy head's overlap with the clean head exceeds the k = 8 head's by at least 0.30.
- **N7b.** A harder copy of the same reading converges: the noisy head's agreement ratio is at least 0.85 in at least 4 of 5 models.

## What happened

**Both claims are refuted.**

| Head B | Its loss (own targets) | Overlap with the clean next-token head | Agreement ratio |
|---|---|---|---|
| Same target | 5.13 | 0.951 | 0.961 |
| 4 ahead | 6.91 | 0.297 | 0.305 |
| 8 ahead | 7.11 | 0.242 | 0.249 |
| Noisy next token | 7.35 | 0.307 | 0.315 |

- **N7a refuted.** o(noisy) − o(8) = +0.066, 95% interval [+0.045, +0.087], standard deviation 0.017: wholly below the 0.30 required. Every seed was between +0.045 and +0.086.
- **N7b refuted.** The noisy head's agreement ratio was 0.30 to 0.33 in all 5 models, far below 0.60.

All four preconditions held:
- **Data and setup:** the noise check (80.2% replaced), the identical-heads check, the target shifts and the data order all passed.
- **Must fail:** same-target agreement ratios were 0.947 to 0.973, at least 0.95 in 4 of 5.
- **Stability:** every head was at least 0.80 (means 0.96 to 0.98).
- **The noisy reading was hard:** its loss (7.35) exceeded the k = 4 head's in all 5 seeds. That was the condition amendment A1 was written to secure. At the original 60% noise the run would probably have been untestable.

The k = 4 and k = 8 overlaps replicated N6r's on new seeds and windows: 0.297 against 0.308, and 0.242 against 0.233.

## What this teaches

- **The overlap measure does not track relatedness alone.** A head reading exactly the same thing as the clean head, only with most of its targets randomised, shared no more with it (0.31) than a head reading four words ahead (0.30). If the overlap measured how related two readings are, the noisy head should have scored near 0.95.
- **N6r's result stands; its explanation does not.**
  - N6r found, and N7 replicated, that the overlap falls the further ahead head B reads. That stands as stated.
  - N6r's note explained this as "how much two readings share depends on how different they are". N7 refutes that explanation as the whole story. The ladder is at least as consistent with difficulty, or with something difficulty brings with it.
  - A dated correction has been added to `N6R_NOTE.md`, and the plain summary has been corrected.
- **Difficulty alone does not fit either.** The noisy head was the hardest (7.35 nats) but shared *more* than the k = 8 head (0.31 against 0.24). So neither "more different" nor "harder" orders the results on its own.
- **A likely reason: the measure sees where a reading's errors fall, not only what it uses.** The sensitivity subspace comes from the gradient of each head's loss against its actual targets. When targets are unpredictable, whether because they are far ahead or because they are randomised, much of that gradient comes from the errors. The errors point toward whichever words the targets turned out to be, frequent words above all. Two heads then "share" directions in proportion to how alike their errors are, not only how alike their features are. This is a conjecture, prompted by the result, and it needs its own test.
- **Correction, 2026-10-10, from N8.** This conjecture is refuted. On the same models and on fresh windows, a target-free version of the measure gave the same overlaps to within 0.045 on average, and 0.001 for the noisy head (0.310 against 0.309). Errors were not what made the noisy head look separate. In hindsight this was predictable: a trained head is roughly calibrated to its own targets, and then the two measures have the same expectation. N7's verdicts are unchanged. See `N8_NOTE.md`.

## Which kind of result this is

- **A refutation of the claim as stated,** under a sound design: all the controls held, and the run was sensitive enough, with a standard deviation of 0.017 against a margin of 0.30.
- **One alternative reading is worth recording, and it is not a rescue.** The noisy head learned the next-token reading less well: its loss against the *clean* next-token targets was 6.60, against 5.13 for the clean head. So it is "the same reading, learned less well" as much as "the same reading, made harder". The preregistration anticipated this and judged that it worked against the claim. It did. The claim, as stated and tested, stays refuted.

## What this does not change

- N6r's measured ladder (the overlap falls with distance) stands, and it replicated here.
- N4's result (sharing 64 readout directions costs two readings 0.088 nats more than one) stands. It was measured in loss, not with this overlap.
- All earlier results stand.

## Exploratory observations, not confirmatory

- **The noisy head moved away from the clean head during training,** just as the far readings did. All arms started at about 0.47. The same arm rose to 0.95; the noisy head fell to 0.31, like k = 4 (0.30) and k = 8 (0.24).
- **Head A was unaffected by head B's task.** Its loss was 5.13 to 5.19 in every arm.

## Limitations

- **One way of making a reading harder,** random replacement at one rate. It both adds difficulty and degrades how well the reading is learned.
- **The same limits as the earlier tests:** 10% of the budget, one learning rate, one size, one dataset, one readout width.

## Next conjectures

1. **The overlap measures shared error structure, not shared features.** If so, a target-free measure of which directions each head reads should behave differently. Such a measure would use the sensitivity of each head's *output distribution* to the state, the Jacobian of its predictions, rather than the gradient of its loss against actual targets. The prediction that could fail: under a target-free measure, the noisy head overlaps the clean head almost completely (because it computes the same thing), while the distance ladder either persists, if the readings really use different features, or flattens, if it was error structure all along. This can be computed on the models already trained, so it would be cheap, but the weights are on Drive, so it needs a Colab job.
2. **Return to loss, where N4 stood firm.** N4's finding is measured in what sharing *costs*, not in this overlap. Repeating N4's forced-sharing test along the distance ladder (k = 2, 4, 8) would ask whether readings that score as more separate also pay more to share. That is a test of function, not of a measure.

## References

Patrini, G., Rozza, A., Menon, A. K., et al. (2017). Making deep neural networks robust to label noise: A loss correction approach. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2233–2241. https://doi.org/10.1109/cvpr.2017.240

## Records

- **Raw results:** `results/n7/` and `results/n7_log.txt`, from the `colab-results` branch, run `n7_harder/20261009-162206`.
- **Code:** commit 554679f. `n7_analyze.py` writes `N7_RESULTS.md` from those records; the job is `n7_harder.py`.
