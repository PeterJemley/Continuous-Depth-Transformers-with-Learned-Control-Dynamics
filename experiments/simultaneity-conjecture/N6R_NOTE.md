# Result note: N6r

Date: 2026-10-08    Preregistration: `N6R_PREREGISTRATION.md`    Generated report: `N6R_RESULTS.md`    Raw results: `results/n6r/` and `results/n6r_log.txt`, from `colab-results` branch `n6r_halfoverlap/20261008-182116`. Code commit 3ebe497.

**How the run went.** The first Colab session did not upload, and evidently its progress did not reach Drive, since the second session found nothing to resume. A second session ran all 20 models and their measurements from the start, with no resumes, no skipped fp16 steps and no Drive failures. Nothing was changed after approval.

## What we expected

Head A predicts the next token (i + 1); head B predicts token i + k, for k = 1 (the must-fail control), 2, 4 or 8. The claims and thresholds were N6's, unchanged:
- **N6a (primary).** The overlap of the two readings' sensitivity subspaces falls as the readings differ more: o(2) − o(8) at least 0.10.
- **N6b.** At k = 2 the overlap lies between 0.40 and 0.60 in at least 4 of 5 models.

Two things differed from N6. The must-fail control was judged by the agreement ratio, built from independent halves of the windows. And the overlap was measured on 256 windows of the WikiText-103 test split, on new seeds 3701–3705.

## What happened

**N6a is supported.** o(2) − o(8) = +0.252, 95% interval [+0.219, +0.286], standard deviation 0.027. Every seed was between +0.21 and +0.28.

**N6b is supported.** o(2) lay between 0.445 and 0.506 in all 5 models.

| Seed | Same-arm agreement ratio | o(1), same | o(2) | o(4) | o(8) | o(2) − o(8) |
|---|---|---|---|---|---|---|
| 3701 | 0.952 | 0.942 | 0.500 | 0.302 | 0.236 | +0.263 |
| 3702 | 0.970 | 0.967 | 0.494 | 0.281 | 0.238 | +0.256 |
| 3703 | 0.970 | 0.961 | 0.482 | 0.274 | 0.228 | +0.255 |
| 3704 | 0.965 | 0.957 | 0.506 | 0.343 | 0.225 | +0.281 |
| 3705 | 0.965 | 0.954 | 0.445 | 0.342 | 0.238 | +0.207 |

All three preconditions held:
- **Data and setup:** the batch and same-model checks passed, the test-split windows decoded correctly with every target shifted by k, and all four arms of each seed saw the same batches.
- **Must fail:** the same arm's agreement ratio was 0.952 to 0.970, at least 0.95 in all 5 models.
  - On the untrained models of the same seeds it was 0.48 to 0.50, so the control would have failed them, as it should.
  - The identical-heads check gave a ratio of 1 to within rounding.
  - Seed 3701 passed with the least room, at 0.952. Its raw overlap, 0.942, would have failed N6's rule. That is the case the like-with-like ratio was designed for: its heads agree as closely as the measurement can show.
- **Stability:** every head in arms k2, k4 and k8 had a split-half stability of at least 0.92.

## What this teaches

- **How much two readings share depends on how different they are.** Two heads reading the same target converge (0.96). The next token and the one after it share about half of their most sensitive directions (0.49). Readings four and eight tokens apart share a third (0.31) and under a quarter (0.23). The order held in every model.
- **Half-overlap is not structural.** It is what the next token and the token after it happen to need. That refutes the rival that any two different readings settle at about 0.5 in this model.
- **Correction, 2026-10-09, from N7.** The first bullet above explains the ladder by how *different* the readings are. N7 refuted that as the whole story. A head reading exactly the next token, made harder by randomising 80% of its targets, shared no more with the clean head (0.31) than the four-ahead head did (0.30). The ladder itself, with overlap falling as head B reads further ahead, stands and replicated in N7. But it cannot be read as a measure of relatedness alone. The overlap probably reflects where each reading's errors fall as well as which features it uses. See `N7_NOTE.md`. *Further correction, 2026-10-10, from N8:* that explanation is refuted. A target-free measure gives the same ladder (0.95, 0.44, 0.25, 0.19 on these models), so the ladder reflects what each reading's predictions depend on, not where its errors fall. See `N8_NOTE.md`. N6r's verdicts are unchanged: they were about the ladder, not its explanation.
- **N6's exploratory pattern replicated on new seeds and new text.** The two tests' mean overlaps agree to within 0.02 at every rung (S5): 0.954 and 0.956, 0.481 and 0.485, 0.305 and 0.308, 0.254 and 0.233.
- **Training sets each balance point.** All arms started at an overlap of about 0.47. Training raised the same arm by 0.49, left k = 2 almost unmoved (+0.01), and lowered k = 4 by 0.16 and k = 8 by 0.24. This fits N5: the end point is set by the readings, not by where they start.
- **For the programme.** This is the first confirmed, quantitative regularity about how one state holds several readings. The polyphony picture now has a measured form: voices are kept in partly separate directions, and the more different the voices, the more separate their directions.

## What this does not change

- N6 stays untestable; N6r does not re-score it.
- All earlier results stand, including N4's (sharing directions costs two readings) and N5's inconclusive verdicts.
- The paper's original claims stay refuted.

## Exploratory observations, not confirmatory

- **The confound with difficulty remains, and is the main open question.** Head B's loss rose with k: 6.24, 6.92 and 7.11 nats, against 5.14 for the next token. So "more different" and "harder to predict" moved together, as both preregistrations warned. They do not move in step: from k = 4 to k = 8 the loss rose by only 0.18, while the overlap fell by another 0.08. That hints the overlap tracks relatedness more than difficulty, but this design cannot separate them.
- **Even the most different reading shares something.** At k = 8 the overlap, 0.23, is still nearly four times what two random 16-dimensional subspaces would share (0.06).
- **The agreement ratio tracks the raw overlap closely** in every arm (S3: 0.965, 0.504, 0.317, 0.240 against 0.956, 0.485, 0.308, 0.233). The control's correction matters only near the top of the scale, where noise caps the raw measure.

## Limitations

- **Distance confounds "different" with "harder",** as above.
- **One kind of difference.** All the readings are future tokens. Readings that differ in kind are untested.
- **The same limits as the earlier tests:** 10% of the budget, one learning rate, one size, one dataset, one readout width (64).
- **N6r replicates a pattern already seen in N6.** Its confirmatory weight comes from new seeds, new data and thresholds fixed before N6 ran, not from a fresh prediction.

## Next conjectures

1. **Separate "different" from "harder".** Add a reading that is harder but no more different, for example the next token with half its targets replaced at random. If the overlap tracks relatedness, the noisy next-token head should still converge with the clean one, at an overlap near 0.95, however much harder it is. If it tracks difficulty, the overlap should fall to the level of a reading of similar loss.
2. **Does the cost of sharing grow as the overlap falls?** Combine N4's test with N6r's ladder: force the two readings through one shared 64-direction readout at k = 2, 4 and 8. If voices need separate directions in proportion to how different they are, the cost of tying should grow with k. This would link N6r's measure to N4's cost.
3. **Readings that differ in kind,** such as a topic label against the next token, to see whether the ladder extends beyond distance.

## References

Lin, H. W., & Tegmark, M. (2017). Critical behavior in physics and probabilistic formal languages. *Entropy, 19*(7), 299. https://doi.org/10.3390/e19070299
