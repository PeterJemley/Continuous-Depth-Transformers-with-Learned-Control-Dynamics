# Result note: N7

Date: 2026-10-09    Preregistration: `N7_PREREGISTRATION.md`, with amendment A1 (noise rate 0.8)    Generated report: `N7_RESULTS.md`    Raw results: `results/n7/` and `results/n7_log.txt`, from `colab-results` branch `n7_harder/20261009-162206`. Code commit 554679f.

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
