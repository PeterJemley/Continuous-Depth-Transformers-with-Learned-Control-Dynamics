# Result note: N5

Date: 2026-10-07    Preregistration: `N5_PREREGISTRATION.md`    Generated report: `N5_RESULTS.md`    Raw results: `results/n5/` and `results/n5_log.txt`, from `colab-results` branch `n5_widths/20261006-233331`. Code commit 0e60457, whose experiment code is identical to the queued commit 92382e6 (only the plain-language summary changed between them).

**How the run went.** One Colab session ran all 60 models and their measurements, with no resumes and no Drive failures. One fp16 step in 172,740 was skipped. A second session the next morning (`n5_widths/20261007-091612`) found every run finished and skipped them all; its 121 result files are byte-for-byte identical to the first session's. Nothing was changed after approval.

## What we expected

N4 found that forcing two different readings (the next token, and the token after next) through one shared 64-direction readout cost 0.088 nats more than it cost two heads reading the same target. N4 explained this as a **competition for directions**. N5 tested that explanation by changing only the readout width:

- **N5a (primary).** At width 256, the full width of the state, tied and separate readouts can represent the same functions. If the cost is a competition for directions, it vanishes there: the 97.5% interval for D(256) lies below +0.02. If it is a clash of learning signals in the shared matrix, it persists: the interval lies above +0.05.
- **N5b.** Tightening the competition raises the cost: D(32) − D(128) at least 0.05.

## What happened

**Both verdicts are inconclusive.** All three preconditions held: separate and tied were bit-identical at initialisation at every width and seed; the batch, data and same-model checks passed; and the width-256 readout was full rank at initialisation for every seed.

**N5a: inconclusive.** D(256) = +0.030, 97.5% interval [−0.004, +0.065], standard deviation 0.022.

| Seed | D(256) |
|---|---|
| 3501 | +0.017 |
| 3502 | +0.013 |
| 3503 | +0.045 |
| 3504 | +0.063 |
| 3505 | +0.014 |

The interval reaches both below zero and above the 0.05 margin, so it meets neither the "vanishes" rule nor the "persists" rule. D was positive in all five seeds, though.

**N5b: inconclusive.** D(32) − D(128) = −0.063, 97.5% interval [−0.323, +0.196], standard deviation 0.166.

| Seed | D(32) | D(64), from N4 | D(128) | D(256) |
|---|---|---|---|---|
| 3501 | −0.073 | +0.101 | +0.117 | +0.017 |
| 3502 | +0.102 | +0.089 | +0.047 | +0.013 |
| 3503 | +0.177 | +0.082 | +0.060 | +0.045 |
| 3504 | +0.001 | +0.084 | +0.023 | +0.063 |
| 3505 | −0.327 | +0.083 | −0.049 | +0.014 |

## Which kind of result this is

**The test was not sensitive enough, and the cause is identified. The explanation was neither confirmed nor refuted.**

- **The preregistration's sensitivity estimate was wrong.** It assumed the seed-to-seed spread at the new widths would be like N4's at width 64, where D varied with a standard deviation of 0.008, giving about 0.011 for a difference of two Ds. The actual spread was 0.022 at width 256, 0.060 at width 128, 0.194 at width 32, and 0.166 for the N5b difference: 15 times the assumption.
- **The noise is training variability at narrow widths, not a failure.**
  - At width 32, two of the five separate-readout dual models trained markedly worse than the rest: seed 3505 ended at 5.59 / 6.56 nats and seed 3501 at 5.46 / 6.46, against 5.34–5.38 / 6.38–6.41 for the other three.
  - There were no loss spikes. These runs learned more slowly from about step 800 and never caught up. Seed 3505's run also grew an unusually large ODE scale (α 0.34, against 0.14–0.21).
  - Because D subtracts the separate arm, a poor separate run makes D strongly negative. Width 128 shows the same effect on a smaller scale, in both arms.
  - Width 64 (N4) happened to sit where training was consistent. That is luck of the width, and the design did not anticipate it.
- **Nothing here is a rescue.** N5a and N5b stay inconclusive as tested. Any sharper test of the same question is a new test on new seeds, with its sensitivity set from N5's spread.

## What this teaches

- **N4's result stands as stated:** at width 64, sharing directions costs two readings 0.088 nats more than one. N5 does not weaken it.
- **N4's explanation is neither confirmed nor refuted.** If anything, the full-width data lean against a pure competition for directions:
  - D(256) was positive in all five seeds, with a mean of +0.030, about a third of the width-64 cost.
  - With 5 seeds, an exact sign test cannot reach significance (its smallest p is 2/32 = 0.0625), and the 95% interval, [+0.003, +0.058], is exploratory only.
  - So some cost may remain even when nothing is restricted. That would fit the "both mechanisms" row of the outcome table: part competition for directions, part a clash of learning signals in the shared matrix. It is a lead, not a finding.
- **The cost is not a smooth function of width at this budget.** Narrow readouts make training itself erratic, so "the cost grows as directions grow scarce" cannot be measured with this design below width 64.

## Exploratory observations, not confirmatory

These had no decision rule.

- **Two different readings settle at about half-overlap, whatever their starting point.** This is the most striking result of the run.

  | Width | Untrained overlap | Trained, two readings (5 seeds) | Trained, same reading |
  |---|---|---|---|
  | 32 | 0.67–0.74 | 0.42–0.48 | 0.87–0.91 |
  | 64 (N4) | 0.45–0.50 | 0.46–0.49 | 0.95–0.97 |
  | 128 | 0.28–0.31 | 0.44–0.53 | 0.98–0.99 |
  | 256 | 0.18–0.20 | 0.47–0.53 | 0.96–0.99 |

  - At width 32, training pushed the two readings *apart*, from 0.7 to 0.45. At widths 128 and 256 it pulled them *together*, from 0.2–0.3 to 0.5. At width 64 they started near 0.5 and stayed.
  - All 20 two-reading models ended between 0.42 and 0.53. N2's full-width heads, a slightly different architecture, ended at 0.57–0.63.
  - Heads reading the same target converged instead: 0.87 to 0.99.
  - So, for these two readings, about half of each reading's most sensitive directions are shared and half are its own, and training finds that balance from either side.
- **This corrects N4's note.** N4's note said the overlap level "depends on where they start", and that training "did not push them apart". That generalised from one width, where the start happened to be near 0.5. A dated correction has been added to `N4_NOTE.md`.
- **Each reading needs many directions.** The same-target models' next-token loss improved with every doubling of width: 5.34 at 32, 5.14 at 64, 5.00 at 128, 4.88 at 256. Even width 256, with its extra readout matrix, was a little behind N2's plain full-width heads (4.83; different seeds, descriptive only).
- **The full-width readouts stayed invertible.** After training, the smallest singular value of P was at least 9.1e-5 of the largest (median 7.3e-4), so tied and separate remained equivalent in what they could represent.
- **Which reading pays.** Wherever the cost of tying was clear (widths 64 and 256), the next-token reading paid about twice what the farther reading paid: at 256, +0.031 against +0.015.

## Limitations

- **The same limits as the earlier tests:** 10% of the budget, one learning rate (not re-tuned for each width), one size and one dataset; two related readings.
- **The sensitivity estimate came from one width.** The design's noise assumption did not transfer, and that is what made both verdicts inconclusive.
- **Reused seeds.** All widths shared N4's five seeds, so the width-64 point and the new widths share trunk initialisations and data order.

## Next conjectures

1. **Half-overlap as a regularity.** For two related readings, the trained overlap of their sensitive directions is about 0.5, independent of readout width and starting overlap. A sharper form: readings that differ more (the next token against a topic label) settle at a *lower* overlap, and readings that differ less (the next token against the next token at a slightly later layer) at a higher one. This would turn N5's strongest observation into a test that can fail, on new seeds and new readings.
2. **Settle N5a with the right power.** Repeat width 256 alone on new seeds, with the number of seeds set from N5's spread (standard deviation 0.022): about 10 seeds would resolve a cost of 0.03 from zero. If the cost is real, the next step is to remove the clash of learning signals directly in the shared matrix, for example by projecting out conflicting gradient components (Yu et al., 2020), and see whether the cost goes.
3. **More voices.** Three or four readings at once, at width 64 where training is consistent. The question is whether the cost of sharing grows with the number of voices, and whether each pair still settles at about half-overlap.

## References

Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. (2020). *Gradient surgery for multi-task learning*. arXiv. https://doi.org/10.48550/arxiv.2001.06782
