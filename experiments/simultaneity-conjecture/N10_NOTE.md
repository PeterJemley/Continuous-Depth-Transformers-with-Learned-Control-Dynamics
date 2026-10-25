# Result note: N10

Date: 2026-10-18    Preregistration: `N10_PREREGISTRATION.md`    Generated report: `N10_RESULTS.md`    Exploratory report, added after the results: `N10_EXPLORATORY.md`

## In brief

N10 trained new models to ask how much scrambling it takes to change what a head reads, and whether a small payload, an extra training signal, can protect the reading when 80% of its answers are scrambled. The head's directions moved further from its ideal noisy copy as the scrambling grew, with agreement ratios of 0.81, 0.64 and 0.31 at 20, 50 and 80%, so N10d is supported; at 20% they fell just short of the 0.85 required (0.79 to 0.84), so N10c is intermediate. The two payload claims, N10a and N10b, are untestable, because the functional payload failed its own check that it took hold. In exploratory analyses, pinning head B's readout to head A's raised its agreement ratio to 0.70 but closed only 16% of the gap in what it predicts and made head A worse by 0.13 nats, and the data payload backfired.

## The arms and measures

Every model is the paper's continuous-depth transformer, trained with exact gradients for 10% of the WikiText-103 training budget. Two heads read the same final internal state, each through its own readout of 64 directions. Head A always predicts the next word from the real answers. Head B predicts it from scrambled answers: a share ρ of its training answers is replaced by words drawn at random in proportion to how common they are. The arms differ only in head B's training. Each arm was trained from five seeds (3901 to 3905), 35 runs in all, and all seven arms of a seed shared their starting weights and the order of the training text.

| Arm | Head B's answers | Payload |
|---|---|---|
| `clean` | The real next words (ρ = 0) | None. The must-match control |
| `n20` | 20% scrambled | None |
| `n50` | 50% scrambled | None |
| `n80` | 80% scrambled | None. The unanchored arm, which repeats N7 on new seeds |
| `n80-func` | 80% scrambled | *Functional:* on 64 fixed training passages, a penalty on how far head B's prediction is from head A's ideal noisy copy (λ = 1) |
| `n80-geom` | 80% scrambled | *Geometric,* also called the pin: a penalty that pins head B's readout directions to head A's (λ = 10) |
| `n80-data` | 80% scrambled | *Data:* on the same 64 passages, head B is also trained on the real answers (λ = 1) |

- **A payload** is an extra training signal meant to protect head B's reading. Its weight is λ: the training loss is half the sum of the two heads' losses, plus ½λ times the payload term. Each step used 4 of the 64 payload passages, against 32 passages in the training batch.
- **The ideal noisy copy:** a head built from head A, without training, whose prediction is exactly (1 − ρ)·p_A + ρ·u. Here p_A is head A's prediction and u is the word frequencies of the training text. It is the best possible prediction of answers scrambled at rate ρ, the plain summary's perfect noisy copy.
- **Measurement windows:** 512 fixed passages of the WikiText-103 test text (windows 1281 to 1792), which no earlier measurement had used and the payload never saw.
- **The agreement ratio,** the geometric measure: N9's target-free agreement ratio between head B and its ideal copy. It compares the 16 directions of the internal state that most change each head's predictions, corrected for measurement noise by each head's split-half stability. It is 1 for the same directions, and random directions score about 0.06.
- **G,** the functional measure: head B's loss on the real next words minus the ideal copy's loss on the same words, in nats, on the measurement windows. 0 means head B carries as much next-word information as the copy.
- **The fixed yardstick,** in the exploratory analysis (`N10_EXPLORATORY.md`): the ideal copy built from the same seed's `n80` model, which no payload could change. *The gap* is head B's loss on the real next words minus that copy's, and *the share of the gap closed* is measured against `n80`'s gap.
- **P_A and P_B** are the two heads' readout projections, the 64 directions each reads. Their *readout overlap* is 1 when they are the same. The *KL divergence* says how far one predicted distribution is from another, in nats per position.
- **N10a (primary)** claims that the functional payload preserves the reading: G(`n80-func`) below 0.20 nats. **N10b** claims that it beats the geometric payload: G(`n80-geom`) − G(`n80-func`) of at least 0.20, paired by seed. Each is judged on a Bonferroni-corrected 97.5% interval over the five seeds (a t-interval with 4 degrees of freedom). N10a is *supported* if its interval lies wholly below 0.20, and N10b if its interval lies wholly above 0.20. Each is *refuted* if its interval lies wholly on the other side of 0.20 with a standard deviation of at most 0.05. A claim that meets neither rule is *some effect, size unclear* if an interval excludes 0, and *inconclusive* otherwise. For N10b that is its own interval; for N10a it is the paired interval for G(`n80`) − G(`n80-func`).
- **N10c** claims that light noise does not change what the head reads. It is *supported* if head B's agreement ratio with its copy in `n20` is at least 0.85 in at least 4 of 5 models, *refuted* if it is at most 0.60 in at least 4 of 5, and *intermediate* otherwise.
- **N10d** claims that the change grows with the noise. It is *supported* if the ratio falls in the order `n20` > `n50` > `n80` in at least 4 of 5 seeds, *refuted* if `n20` is at or below `n80` in at least 4 of 5, and *partial* otherwise.
- **Preconditions,** without which a claim is *untestable*:
  - *setup:* the data, batch and identical-heads checks pass;
  - *must match:* in `clean`, head B's ratio with head A is at least 0.95 and |G| is at most 0.05, each in at least 4 of 5 models, and the copy at zero noise matches head A in every model;
  - *recognising copies,* for N10c and N10d: head A's ratio with its own copy is at least 0.85 in at least 4 of 5 models of every arm;
  - *must differ:* the rotated control, head A reading a randomly rotated copy of the state, scores at most 0.60 in at least 4 of 5 models of every arm;
  - *the payloads took hold,* the manipulation checks for N10a and N10b: P_A and P_B overlap by at least 0.90 in every `n80-geom` model; and, for the functional payload, head B's KL divergence from the copy on the 64 payload passages is at most half of the same seed's `n80`, in at least 4 of 5 seeds;
  - *stability:* head B and the copies have a split-half stability of at least 0.80.
- **S1 to S5** are the preregistered secondary measures, which decide nothing. The note cites S2: head B's ratio with its copy in `n80-geom`, which asks whether pinning the directions satisfies the geometric measure without preserving the function.
- **Other names.** N4, N7, N9, N11 and N12 are other tests in this folder, each with its own note. *Substrate independence* is the principle, argued by Deutsch, that information is independent of the physical form that holds it. Here it predicts that anchoring what a head predicts (its function) should protect its reading better than pinning the directions it reads (its geometry, the substrate). Runs were trained in a Google Colab session and saved to Google Drive.

*The summary, this key and the records section at the end were added on 2026-10-25 so that the note can be read on its own. Nothing else was changed.*

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

- **Head A was unaffected by noise in head B alone.** Its validation loss was 5.13 to 5.19 in the clean and dose arms. Only the payloads changed it. *Correction, 2026-10-21, after N12:* this was wrong. The range across arms hid a consistent shift. In N10's own records, `n80`'s head A was worse than `clean`'s by 0.026–0.046 nats on the measurement windows in every seed (0.028–0.047 on validation), and in N11's by 0.027–0.061. N12 measured the cost at +0.040 [+0.033, +0.047] on new seeds, and found that a head whose form can represent the noisy target imposes little or none (−0.009 [−0.024, +0.007]). See `N12_NOTE.md`.
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

## Records

- **Raw results:** `results/n10/` and `results/n10_log.txt`, from the `colab-results` branch, run `n10_payload/20261018-191435`.
- **Code:** commit e2f9f13. `n10_analyze.py` writes `N10_RESULTS.md` and `n10_exploratory.py` writes `N10_EXPLORATORY.md` from those records; the job is `n10_payload.py`.
