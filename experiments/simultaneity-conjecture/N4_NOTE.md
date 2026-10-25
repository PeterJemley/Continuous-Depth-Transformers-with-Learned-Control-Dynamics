# Result note: N4

Date: 2026-10-06    Preregistration: `N4_PREREGISTRATION.md`, with amendment A1    Generated report: `N4_RESULTS.md`

## In brief

N4 asked whether two readings held at once, the next token and the token after next, need separate directions in what their heads read from the model's state. Forcing both heads to read through one shared set of 64 directions cost 0.088 nats more than the same restriction cost two heads reading the same target, 95% interval [+0.078, +0.097], so N4b, the primary claim, is supported. N4a is supported too: on new seeds and windows, the two readings' most sensitive directions overlapped 0.455 to 0.488, while two heads reading the same target overlapped 0.952 to 0.967. The test acts at the readout, so it bears on nothing in the continuous block's depth dynamics.

## The arms and measures

Every arm is N2's two-head model: Model A, the paper's model, trained with exact gradients and no control signal for 2,879 steps (10% of the WikiText-103 budget), on seeds 3501 to 3505. Head A reads the next token (i + 1). Each head now reads the final state z through a narrow readout, logits_k = W_k (P_k · LayerNorm(z)). P_k, a 64 × 256 matrix, keeps 64 directions of the 256-number state, and W_k turns those 64 numbers into logits, a raw score for every word.

| Arm | Head B reads | Readout |
|---|---|---|
| dual-sep | the token after next (i + 2) | Separate: P_A and P_B are independent, so the two heads can read up to 128 directions between them |
| dual-tied | i + 2 | Tied: P_A = P_B, one shared matrix, so both heads read the same 64 directions |
| same-sep | the next token (i + 1), like head A | Separate |
| same-tied | i + 1 | Tied |

P_B starts as a copy of P_A, so separate and tied readouts are the same function at the start and differ only through training. All four arms of a seed start from the same weights and see the same batches.

- **J** is the combined validation loss, L_A + L_B, on the whole validation split, each head against its own target.
- **The cost of tying** is J_tied − J_sep for one pair of arms. "Two readings" is the dual pair, and "same reading" the same pair.
- **D**, N4b's measure, is (J_dual-tied − J_dual-sep) − (J_same-tied − J_same-sep): the extra cost that two different readings pay for sharing directions, beyond what the same restriction costs two heads reading one target.
- **Overlap**, N4a's measure, is how much the two readings rely on the same directions. A reading's directions are the 16 directions of the ODE block's output that its loss is most sensitive to, found from the gradient covariance on validation windows 1,537 to 1,792. The ODE block is Model A's continuous block, and a window is one fixed-length passage of text. The overlap is the mean squared cosine of the angles between the two sets of directions: 1 when they are identical, about 0.06 for random directions. It is measured in the separate arms; the odd and even halves of the windows give a stability check.
- **The row space** of P_A or P_B is the 64 directions that readout reads. Two row spaces are compared by the same overlap.
- **Must match:** separate and tied readouts must give bit-identical logits at initialisation, for every seed. **Must fail:** the same-sep overlap must be at least 0.95 in at least 4 of 5 models, since two heads reading the same target must converge on the same directions.
- **Binding** (amendment A1): the restriction counts as not binding if the 95% intervals for the cost of tying lie wholly within ±0.01 nats for both pairs. N4b would then be untestable at this width.
- **fp16:** training ran in 16-bit arithmetic, and a step whose gradients overflow is skipped.
- **G1, G1c, G1d, N1 and N2** are earlier tests, and **N5** a later one, each with a note in this folder.

Each interval is a 95% t-interval over the five seeds. N4b's margin of practical importance is 0.05 nats. It is *supported* if D's interval lies wholly above +0.05, *refuted* if wholly below +0.05 with a standard deviation of at most 0.04, *some effect, size unclear* if it excludes 0 but meets neither rule, and *inconclusive* otherwise. N4a is *supported* if the dual-sep overlap is at most 0.8 in at least 4 of 5 models, *refuted* if it is at least 0.95 in at least 4 of 5, and *intermediate* otherwise.

*The summary, this key and the records section at the end were added on 2026-10-25 so that the note can be read on its own. Apart from the corrections dated 2026-10-25, nothing else was changed.*

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
- **Which reading pays.** When directions were shared, the next-token reading lost more (+0.062) than the farther reading (+0.036). N2 found the same asymmetry when the two readings coexisted at full width: the nearer reading paid 0.05 nats. *[Corrected 2026-10-25: N2 measured only the nearer reading's cost (its S6, +0.052); it had no measure of what the farther reading paid, so it cannot show an asymmetry.]*

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
  - **Correction, 2026-10-07, from N5.** The two sentences above generalised from one width and are wrong. N5 measured the same overlap at readout widths 32, 128 and 256 on the same seeds. Two different readings started at 0.67–0.74, 0.28–0.31 and 0.18–0.20, and ended at 0.42–0.53 in every case. At width 32 training pushed them apart; at 128 and 256 it pulled them together. So the level they end at does *not* depend on where they start: for these two readings it is about 0.5. Width 64 merely started near that level. See `N5_NOTE.md`. N4a's verdict is unaffected. *[Corrected 2026-10-25: this holds for separate readouts (dual-sep), the arm the sentences above describe. With one tied readout the two readings ended higher, 0.58 to 0.75, and higher the narrower the readout (0.65–0.71 here, at width 64); see `N5_NOTE.md`.]*
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

## Records

- **Raw results:** `results/n4/` and `results/n4_log.txt`, from the `colab-results` branch, run `n4_directions/20261006-200755`. The untrained baseline, measured after the run: `results/n4_exploratory/untrained_overlap.json`, from `n4_init_baseline.py`.
- **Code:** commit aebb830. `n4_directions.py` trains the four arms and measures them, and `n4_analyze.py` writes `N4_RESULTS.md` from the records; the Colab job is `colab/jobs/n4_directions.py`.
