# Result note: N9

Date: 2026-10-10    Preregistration: `N9_PREREGISTRATION.md`, with amendment A1 (512 windows)    Generated report: `N9_RESULTS.md`

## In brief

N8 left two explanations for why the head trained on scrambled answers reads different directions from the clean head: it learned a different reading, or the measure cannot recognise a noisy copy of a reading as the same reading. N9 built, from each clean head and without training, its ideal noisy copy: the head whose prediction is exactly 0.2 of the clean prediction plus 0.8 of the word frequencies. The measure recognised that copy as the same reading in all 5 models (agreement ratios of 0.955 to 0.995), so N9a is supported. The trained head was not close to the copy (0.305 to 0.349), so N9b is refuted, and the note concludes that training on scrambled answers made the head learn a different, weaker reading.

## The heads and measures

N9 trained nothing. It measured N7's five `noisy` models (seeds 3801 to 3805). In each, two heads read the same final internal state through their own readouts of 64 directions. Head A, *the clean head*, predicts the next word from the real answers. Head B, *the trained noisy head*, predicts it from scrambled answers: in training, 80% of its answers were replaced by words drawn at random in proportion to how common they are. N9 also built new heads from each head A, without training, and compared them all with N8's target-free measure.

| Head | What it is | Role |
|---|---|---|
| Head A, the clean head | Trained on the real next words | The reading to be matched |
| Head B, the trained noisy head | Trained on 80% scrambled answers, in N7's `noisy` arm | Tested by N9b |
| The ideal noisy copy | Built from head A: its prediction is exactly 0.2 of head A's plus 0.8 of the word frequencies. It is the best possible prediction of 80% scrambled answers, the plain summary's perfect noisy copy | Tested by N9a, and the yardstick for N9b |
| Copies at other noise rates | The same construction with ρ = 0, 0.2, 0.5 and 0.95. At ρ = 0 the copy is head A itself, routed through the same code | S1; the copy at ρ = 0 is the must-match control |
| The rotated control | Head A reading a randomly rotated copy of the state, so it reads different directions by construction | The must-differ control |

- **Symbols.** ρ is the share of answers replaced, and "noise" or "the noise rate" means ρ. A copy's prediction is q = (1 − ρ)·p_A + ρ·u, where p_A is head A's prediction and u is the word frequencies of the training text, the distribution N7's scrambled answers were drawn from.
- **Measurement windows:** 512 fixed passages of the WikiText-103 test text (windows 769 to 1280), which no earlier measurement had used.
- **The measure:** N8's target-free agreement ratio. For each head, it finds the 16 directions of the internal state that most change the head's predictions. Each answer is drawn from the head's own predicted distribution, so the real answers play no part. Two heads' directions are compared across independent halves of the passages, and the result is corrected for measurement noise by each head's *split-half stability*, which says how well its directions agree with themselves across the halves. It is 1 for the same directions, and random directions score about 0.06. The *overlap* is the same comparison without the correction.
- **The secondary measures,** which decide nothing:
  - *S1:* head A against its copy at each noise rate;
  - *S2:* comparisons of what the heads predict, not which directions they read. These are the *KL divergence* from the trained noisy head's prediction to each copy's, which says how far apart two predicted distributions are, in nats per position, and is 0 when they are the same; and losses against the real next words (*clean targets*) and against scrambled answers (*noisy targets*);
  - *S3:* head A against the trained noisy head, N8's comparison on new windows;
  - *S4:* head A against its copy in N7's other 15 models, whose head A is also a clean head.
- **N9a (primary)** claims that the measure recognises an ideal noisy copy. **N9b** claims that the trained noisy head is close to the ideal copy. Each is judged by the agreement ratio with the copy at ρ = 0.8. A claim is *supported* if that ratio is at least 0.85 in at least 4 of 5 models, *refuted* if it is at most 0.60 in at least 4 of 5, and *intermediate* otherwise. Both are count rules, so no interval is used.
- **Preconditions,** without which the claims would be *untestable*:
  - *the estimator works on mixtures:* on a small model whose Fisher can be computed exactly (the *analytic check*), the estimate's top directions overlap the exact ones by at least 0.98, and identical heads score exactly 1;
  - *must match:* the copy at ρ = 0 scores at least 0.999 with head A in every model;
  - *must differ:* the rotated control scores at most 0.60 in at least 4 of 5 models;
  - *the right models are loaded:* head A's overlap with the trained noisy head lies within 0.05 of the value N8 recorded;
  - *stability:* head A, the trained noisy head and the copy at ρ = 0.8 each have a split-half stability of at least 0.80.
- **Amendment A1,** approved before any real run, doubled the measurement windows from 256 to 512. Calibration had put the copy's stability within 0.05 of the 0.80 threshold.
- **Other names.** N7 and N8 are earlier tests in this folder, each with its own note. N7 trained these models, and N8 first measured them with the target-free measure. Runs were saved to Google Drive and measured in a Google Colab session.

*The summary, this key and the records section at the end were added on 2026-10-25 so that the note can be read on its own. Nothing else was changed.*

**How the run went.** One Colab session measured all 20 models, with no resumes and no Drive failures. Nothing was changed after amendment A1 was approved.

## What we expected

N8 found that N7's noisy next-token head, trained on targets 80% replaced by words drawn at their normal frequencies, reads different directions from the clean head, even under a measure that leaves mistakes out. N8 could not say why. Either the noisy head learned a different reading, or the measure cannot recognise a noisy copy of a reading as the same reading.

N9 built, from each clean head, its *ideal noisy copy*: the head whose prediction is exactly 0.2 of the clean prediction plus 0.8 of the word frequencies. That is the best possible prediction of N7's noisy targets, and the same reading in every sense except the noise.

- **N9a (primary).** The measure recognises the ideal copy: its agreement ratio with the clean head is at least 0.85 in at least 4 of 5 models.
- **N9b.** The trained noisy head is close to the ideal copy, by the same rule.

## What happened

**N9a is supported. N9b is refuted.**

| Seed | Clean head vs its ideal copy (N9a) | Trained noisy head vs the ideal copy (N9b) | Clean head vs trained noisy head (S3) |
|---|---|---|---|
| 3801 | 0.977 | 0.305 | 0.320 |
| 3802 | 0.975 | 0.319 | 0.326 |
| 3803 | 0.955 | 0.330 | 0.341 |
| 3804 | 0.989 | 0.349 | 0.352 |
| 3805 | 0.995 | 0.320 | 0.327 |

- **N9a supported.** The clean head and its ideal noisy copy scored 0.955 to 0.995, at least 0.85 in all 5 models. The clean heads of N7's other 15 models gave the same answer (S4: 0.954 to 0.991).
- **N9b refuted.** The trained noisy head and the ideal copy scored 0.305 to 0.349, at most 0.60 in all 5 models.

All five preconditions held:
- **The estimator works on mixtures.** The analytic check scored 0.994. Measuring the clean head in place of its copy scores 0.80 on the same check, so it can fail. Identical heads scored exactly 1.
- **Must match.** The copy at zero noise matched the clean head at 0.99999 to 1.00000 in all 20 models.
- **Must differ.** The rotated control scored 0.119 to 0.133.
- **The right models were loaded.** The clean and noisy heads' overlap reproduced N8's to within 0.023 on N9's new windows.
- **Stability.** Every head in the comparison scored at least 0.91. Amendment A1's extra windows turned out not to be needed: the copies in trained models were more stable than calibration on untrained ones suggested.

## What this teaches

- **The measure is not the problem.** It recognises a noisy copy of a reading as the same reading, at 0.98 on average, with 80% of the copy's prediction replaced by word frequencies. Even at 95% noise it scores 0.94 (S1). So N7's and N8's finding that the noisy head looks separate is not an artefact of how the measure weights directions.
- **The trained noisy head learned a different reading.** It shares no more with the ideal noisy copy (0.32) than with the clean head (0.33). Training on noisy targets did not produce a noisy copy of the next-token reading. It produced a head whose predictions depend on different directions of the state. N8's conclusion holds as written, and the open anomaly from N7 and N8 is a property of the model.
- **Close in what it predicts, far in what it reads (S2).**
  - The noisy head's predictions are close to the ideal copy's: 0.07 nats of KL divergence per position.
  - But most of both predictions is the same fixed word-frequency component, and the measure looks only at the part that changes with the text. In that part, the two differ.
  - The noisy head's predictions against the clean targets (6.62 nats) are much worse than the ideal copy's (6.01). Its loss on its own noisy targets is 7.36, against 7.24 for the ideal copy.
  - So the noisy head learned a weaker reading of the context than the ideal copy has, as well as a different one.
  - By KL divergence, the noisy head's predictions are about as close to the copy at 95% noise as to the copy at 80% (0.071 and 0.070). Its context-dependent part is weaker than the noise rate alone requires.
- **The order of explanations so far.**
  - N7 refuted "sharing tracks how different readings are".
  - N8 refuted "the measure mixes in mistakes".
  - N9 refutes "the measure cannot recognise a noisy copy".
  - What remains is the plainest reading of N7: training a head on noisy targets makes it read different directions, and read them less well.

## Which kind of result this is

- **A confirmation and a refutation,** under a sound design. All five preconditions held with room, the must-match control was exact to five decimal places, and every seed fell the same way.
- **N9b's refutation is the informative part.** It rules out the last explanation of N7's anomaly that blamed the instrument.

## What this does not change

- N7's and N8's verdicts stand: N8a and N8b refuted, N8c supported.
- N4's cost of sharing, measured in loss, stands.
- The distance ladder stands, and its measure is now validated on a case where the right answer was known in advance.

## Exploratory observations, not confirmatory

- **The measure loses recognition slowly as noise grows** (S1): 1.000, 0.998, 1.002, 0.978 and 0.941 at 0, 20, 50, 80 and 95% noise. The copy's stability falls faster: 0.96, 0.97, 0.95, 0.93, 0.82.
- **The noisy head puts very little probability on words the training data never contains** (1.2 × 10⁻⁴ per position), so it learned the frequency component closely. Its shortfall is in the part that depends on the context.

## Limitations

- **Reused models.** These are N7's models, measured for the third time. A fresh-seed replication would be needed for the result to carry full weight. N9a's constructed copies are new, but the clean heads are not.
- **One noise rate in training.** N7 trained at 80% noise only, so whether lower noise or longer training would bring the noisy head toward the ideal copy is untested.
- **The same limits as before:** 10% of the budget, one learning rate, one size, one dataset, one readout width.

## Next conjectures

1. **Noisy training changes what a head learns to read, more as the noise grows.**
   - *The test:* train noisy heads at 20%, 50% and 80% noise, on new seeds. Measure each one's agreement with its own clean head's ideal copy, which N9 showed the measure can recognise.
   - *The prediction:* agreement falls as the noise rises, and stays high at low noise. A low score even at 20% would refute it.
   - *Why it matters:* whether any learning signal weaker than the clean one leads a head to different directions, or only a very weak one does.
2. **The cost of sharing along the ladder** (from N8's note). It is still the open functional question.

## References

Patrini, G., Rozza, A., Menon, A. K., et al. (2017). Making deep neural networks robust to label noise: A loss correction approach. *Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition*, 2233–2241. https://doi.org/10.1109/cvpr.2017.240

## Records

- **Raw results:** `results/n9/` and `results/n9_log.txt`, from the `colab-results` branch, run `n9_noisycopy/20261010-185851`.
- **Code:** commit 5c83254. `n9_analyze.py` writes `N9_RESULTS.md` from those records; the job is `n9_noisycopy.py`.
