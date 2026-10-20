# Result note: N10

Date: 2026-10-18    Preregistration: `N10_PREREGISTRATION.md`    Generated report: `N10_RESULTS.md`    Exploratory report, added after the results: `N10_EXPLORATORY.md`    Raw results: `results/n10/` and `results/n10_log.txt`, from `colab-results` branch `n10_payload/20261018-191435`. Code commit e2f9f13.

**How the run went.** One Colab session trained and measured all 35 models, with no resumes and no Drive failures. Nothing was changed after approval.

## What we expected

Head A predicts the next token. Head B predicts the next token with a fraction ρ of its targets replaced by words drawn at their normal frequencies.
- **The dose arms:** ρ = 0 (`clean`), 0.2, 0.5 and 0.8, on new seeds.
- **The payload arms:** three more at ρ = 0.8, each with a small payload:
  - *functional:* match head A's ideal noisy copy on 64 fixed windows;
  - *geometric:* pin head B's readout to head A's;
  - *data:* clean answers on the same 64 windows.

Two measures:
- **The agreement ratio:** how far head B reads the same directions as its ideal copy.
- **G:** head B's loss on the clean targets minus the ideal copy's.

The claims:
- **N10a (primary):** the functional payload brings G below 0.20 nats.
- **N10b:** it beats the geometric payload by at least 0.20.
- **N10c:** at 20% noise, the ratio is at least 0.85.
- **N10d:** the ratio falls in the order 20% > 50% > 80%.

## What happened

| Arm | Ratio, head B vs its ideal copy | G (nats) |
|---|---|---|
| `clean` | 0.976 | −0.000 |
| `n20` | 0.812 | +0.041 |
| `n50` | 0.636 | +0.167 |
| `n80` | 0.305 | +0.427 |
| `n80-func` | 0.320 | +0.320 |
| `n80-geom` | 0.703 | +0.250 |
| `n80-data` | 0.159 | +0.358 |

- **N10a and N10b are untestable.** The functional payload failed its own manipulation check. It had to cut the divergence between head B and the copy on its 64 windows to at most half of the unanchored arm's. It cut it to 0.73–0.77 of it (0.047–0.050, against 0.064–0.065). The rules say a payload that did not take hold cannot decide the claim, and they were followed.
- **N10c is intermediate.** At 20% noise, head B's ratio with its ideal copy was 0.79–0.84, just short of 0.85 in every seed and far above 0.60. Functionally the head was nearly the ideal copy: G = 0.04 nats.
- **N10d is supported.** The ratio fell 0.81, 0.64, 0.31 from 20% to 50% to 80% noise, in all 5 seeds.

The other preconditions held:
- **Must match:** the clean arm's head B matched head A (ratio 0.97–0.99, |G| ≤ 0.003), and the zero-noise copy matched exactly.
- **Recognising copies:** the measure recognised every ideal copy (0.95–1.00).
- **Must differ:** the rotated control scored 0.08–0.23.
- **The geometric payload took hold:** P_A and P_B overlapped at 0.998.
- **Stability:** every head was stable, at 0.90 or more.

**Replication.** At 80% noise the ratio, 0.305, replicates N9's 0.31–0.35 on new seeds. G was smaller than N9's (0.43 against 0.61). Part of that is the windows: head A's loss was 5.37 on N10's windows against 5.06 on N9's, so the comparison is not like for like.

## Why the functional payload did not take hold

The payload term fell from 2.5 to about 0.06 by the middle of training and stayed there, barely below the unanchored arm's level. The likely reason is a design error of mine.
- The payload used 4 windows per step, against 32 in the training batch. At λ = 1 it therefore carried about an eighth of the weight of head B's own noisy objective.
- The short calibration run (150 steps, ratio 0.64) suggested it was taking hold, and I judged it "likely" to pass the check by the end. It did not: the ratio barely changed after that.
- The preregistration's limitation says a null result is "about this size and strength only". That turned out to be the whole story.

## What this teaches

- **The noise dose matters, and the change is gradual.**
  - Functionally, the cost grows steeply but smoothly: G is 0.04, 0.17 and 0.43 nats at 20, 50 and 80% noise.
  - Geometrically, the head's directions drift further than its function suggests. At 20% noise it is almost the ideal copy in what it predicts (G = 0.04), but its directions already differ (ratio 0.81).
  - So the geometric measure is more sensitive to training differences than the functional one. That fits N9 and Son (2026): the two can come apart.
- **Pinning the geometry partly preserves the geometric measure, and costs the other reading.** Exploratory, from S2 and `N10_EXPLORATORY.md`.
  - The pin raised head B's ratio with its copy from 0.31 to 0.70, but not to the 0.85 of a converged head.
  - Against a fixed yardstick it improved head B's clean-target loss by only 0.07 nats, 16% of the gap.
  - It made head A worse by 0.13 nats. That is N4's finding again: forcing two different readings through the same directions costs accuracy.
- **G, as defined, has a moving yardstick.**
  - G compares head B with the ideal copy *of the same model's head A*, and every payload changed head A.
  - So the preregistered G flatters the geometric payload (0.25): most of its drop comes from the copy getting worse, not head B getting better.
  - Against a fixed yardstick, the copy from the `n80` model of the same seed, the payloads closed 20% (functional), 16% (geometric) and −45% (data) of the gap. This is exploratory and decides nothing.
- **The data payload backfired.** Exploratory.
  - Head B memorised the clean answers of its 64 windows: its loss on them fell to 0.003 nats, and its divergence from the copy there rose to 6.4.
  - Head A got worse by 0.36 nats, and head B worse on clean text by 0.19.
  - Repeated clean answers on a few passages taught the shared network those passages, not the reading.

## Which kind of result this is

- **Two untestable claims, caused by a design error:** a payload too weak to take hold. The rule built to catch exactly this case worked. It stopped a weak payload from being read as evidence against the principle.
- **One supported claim (N10d)** and **one intermediate claim (N10c)**, under sound controls.
- **The substrate-independence question is still open.**
  - The geometric payload closed a little of the gap, the functional one a little more, and neither preserved the reading.
  - Nothing here supports or refutes "anchor the function, not the geometry", because the functional payload never took hold.

## What this does not change

- Every earlier verdict stands.
- N9's finding replicated: at 80% noise the head reads different directions from its ideal copy (0.31).

## Exploratory observations, not confirmatory

- **Head A was unaffected by noise in head B alone.** Its validation loss was 5.13 to 5.19 in the clean and dose arms. Only the payloads changed it.
- **The clean arm's readouts overlap at 0.71,** as in N7's same-target models. Two heads computing the same thing still use partly different readout directions, which is consistent with the geometry being a property of the substrate.

## Limitations

- **One strength for each payload,** and for the functional one, too weak by design.
- **The measurement windows differ from N9's,** so G's size is not comparable across the two tests.
- **The same limits as before:** 10% of the budget, one learning rate, one size, one dataset, one readout width.

## Next conjectures

1. **A functional payload strong enough to take hold.**
   - *The test:* repeat the functional and geometric arms with the payload weighted to match the main batch. That means λ = 8 for 4 windows against 32, or the payload windows spread through every batch.
   - *Payload strengths:* a dose of λ = 8, 32 and 128.
   - *The functional measure:* against a fixed yardstick, the ideal copy from the unanchored model of the same seed, so that a payload cannot move it.
   - *The original question:* whether anchoring the function preserves the reading better than pinning the geometry.
   - *Update, 2026-10-19:* preregistered as N11 (`N11_PREREGISTRATION.md`), with λ = 8, 32 and 128 and the fixed yardstick. Its dose check found that at λ = 128 a payload acting through the shared trunk damaged head A, so under amendment A1 the payload trains head B's readout only. Run 2026-10-20: the payload again failed its manipulation check; see `N11_NOTE.md`.
2. **At light noise, function survives but geometry drifts.** Whether that holds generally decides which measure future tests should rely on. N10's dose arms suggest the functional measure should lead.

## References

Son, C. (2026). *J-space forecasts model collapse, but only anchoring the function prevents it* [Preprint]. Research Square. https://doi.org/10.21203/rs.3.rs-10713541/v1
