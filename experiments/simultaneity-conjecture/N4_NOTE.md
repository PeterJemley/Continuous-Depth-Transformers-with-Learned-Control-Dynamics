# Result note: N4

Date: 2026-10-06    Preregistration: `N4_PREREGISTRATION.md`, with amendment A1    Generated report: `N4_RESULTS.md`    Raw results: `results/n4/` and `results/n4_log.txt`, from `colab-results` branch `n4_directions/20261006-200755`. Code commit aebb830.

**How the run went.** One Colab session ran all 20 models and their measurements, with no resumes and no Drive failures. One fp16 step in 57,580 was skipped. Nothing was changed after approval.

## What we expected

**N4b (primary).** Forcing two different readings, the next token (A) and the token after next (B), to be read out through one shared set of 64 directions costs at least 0.05 nats of combined loss *more* than the same restriction costs two heads that read the same thing.

**N4a (secondary).** On new seeds and windows, the two readings' 16-dimensional sensitivity subspaces at the ODE block's output overlap at most 0.8 in at least 4 of 5 dual models. Two heads reading the same target overlap at least 0.95 (the must-fail control).

## What happened

**N4b is supported.** D = +0.088, 95% interval [+0.078, +0.097], standard deviation 0.008. Every seed gave between +0.082 and +0.101.

| Seed | Cost of tying, two readings | Cost of tying, same reading | D |
|---|---|---|---|
| 3501 | +0.096 | −0.004 | +0.101 |
| 3502 | +0.074 | −0.015 | +0.089 |
| 3503 | +0.136 | +0.054 | +0.082 |
| 3504 | +0.090 | +0.006 | +0.084 |
| 3505 | +0.096 | +0.013 | +0.083 |

- Sharing 64 directions cost two different readings 0.099 nats, 95% interval [+0.070, +0.127].
- It cost two heads reading the same target 0.011 nats, interval [−0.022, +0.044], which is consistent with nothing.
- The restriction bound by amendment A1's rule, so the verdict stands as written.

**N4a is supported.** The two readings' subspaces overlapped 0.455 to 0.488 in all five dual-sep models. Two heads reading the same target overlapped 0.952 to 0.967, so the must-fail control held. In both separate arms, overlaps from the odd and even halves of the windows agreed within 0.03.

The controls held:
- **Must match:** separate and tied readouts gave bit-identical logits at initialisation, for every seed.
- **Same data:** all four arms of each seed saw the same batches.
- **The batch and same-model checks** passed.

## What this teaches

- **Coexisting readings need separate directions.** This is the first test in the programme where a property of how the state holds two readings turned out to be *needed*.
  - G1d showed the contraction regimes are not needed for switching, and N1 that an offset suffices.
  - N2 showed coexistence does not produce distinct timescales.
  - N4 shows that two readings forced through the same directions pay about 0.09 nats, while the identical restriction costs one reading nothing measurable.
- **The polyphony idea, in the form that survives.** The voices are not separated in depth or time; they are separated in direction, and the separation is load-bearing. This is the form N2's exploratory finding pointed to, now tested and passed.
- **Which reading pays.** When directions were shared, the next-token reading lost more (+0.062) than the farther reading (+0.036). N2 found the same asymmetry when the two readings coexisted at full width: the nearer reading paid 0.05 nats.

## What this does not change

- All earlier results stand: G1 refuted, G1c supported, G1d refuted, N1 supported, N2a refuted.
- The paper's original claims stay refuted.
- Nothing here bears on depth dynamics. The test acts at the readout.

## Exploratory observations, not confirmatory

These had no decision rule.

- **The readings did not move apart; they failed to come together.**
  - After the run, N4a's overlap was measured on the *untrained* models of N4's own seeds and windows (`n4_init_baseline.py`, `results/n4_exploratory/untrained_overlap.json`). It was 0.455 to 0.501, identical for dual and same, since they differ only in head B's target.
  - Training changed the dual-sep overlap by −0.046 to +0.012, essentially nothing. It raised the same-sep overlap by 0.45 to 0.51, to above 0.95.
  - In N2, with full-width heads, the two readings started at 0.07 and ended at 0.57 to 0.63.
  - So the overlap level that two different readings end at depends on where they start; it is not a fixed property of the readings. What replicates across N2 and N4 is the contrast: two heads on the same target converge on the same directions, and two different readings do not.
  - N4a's verdict is unaffected, since its must-fail control was built to catch exactly this. But "the separation replicates" should be read as "different readings do not converge", not as "training pushes them apart".
  - **Correction, 2026-10-07, from N5.** The two sentences above generalised from one width and are wrong. N5 measured the same overlap at readout widths 32, 128 and 256 on the same seeds. Two different readings started at 0.67–0.74, 0.28–0.31 and 0.18–0.20, and ended at 0.42–0.53 in every case. At width 32 training pushed them apart; at 128 and 256 it pulled them together. So the level they end at does *not* depend on where they start: for these two readings it is about 0.5. Width 64 merely started near that level. See `N5_NOTE.md`. N4a's verdict is unaffected.
- **Tying pulls the readings together.** In the tied dual models the overlap rose to 0.65 to 0.71, from the same starting point of about 0.48. The model paid 0.099 nats for that.
- **The readouts drift apart even for the same target.** The row spaces of P_A and P_B, identical at the start, ended at an overlap of about 0.51 for two readings and about 0.71 for one. The second number shows the readout matrices are not unique: different 64-dimensional readouts can serve the same reading. The upstream sensitivity subspaces converged regardless.
- **The 64-direction readout is a severe bottleneck.** The rank-64 models' next-token loss was about 0.3 nats worse than N2's full-width models (5.14 against 4.83; different seeds, descriptive only). The result was shown where directions are scarce. Whether separate directions matter at full width is untested; N2's 0.05-nat cost of reading twice is suggestive.

## Limitations

- **The same limits as the earlier tests:** 10% of the budget, one learning rate (not re-tuned for the new readout), one size and one dataset; two related readings, with farther and harder confounded.
- **One width, and a tight one.** Rank 64 costs a single reading about 0.3 nats, so the test sits in a regime of scarce directions.
- **The test acts at the readout.** It shows the readings need separate directions in what is read from the final state. It does not locate the separation in the ODE block, where the programme's dynamics live.
- **The mechanism is inferred, not shown.** Tying makes the two heads' gradients meet in one matrix, where different targets can conflict (Yu et al., 2020). Same-target heads also share that matrix and pay nothing, so tying as such is not the cost. Gradient conflict was not measured directly.

## Next conjectures

1. **How many directions does a reading need?** Sweep the readout width (for example 32, 64, 128). The conjecture: D shrinks as the width grows and reaches zero once two separate readouts no longer compete for the state's directions. Where it reaches zero measures how much room each voice needs.
2. **More voices.** Read 3 or 4 targets at once (tokens i+1 to i+4). If voices need separate directions, the cost of tying should grow with the number of voices.
3. **Readings that differ more.** Pair the next token with a topic or sentiment label. The conjecture predicts a larger D for less related readings.
4. **Locate the need.** Apply the shared-directions restriction at the ODE block's output instead of the readout. This would tell whether the depth dynamics themselves must keep the voices apart, which links N4 back to the original simultaneity conjecture.

## References

Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., et al. (2022). *Toy models of superposition*. arXiv. https://doi.org/10.48550/arxiv.2209.10652

Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. (2020). *Gradient surgery for multi-task learning*. arXiv. https://doi.org/10.48550/arxiv.2001.06782
