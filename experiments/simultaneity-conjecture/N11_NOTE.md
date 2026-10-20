# Result note: N11

Date: 2026-10-20    Preregistration: `N11_PREREGISTRATION.md`, with amendment A1 and deviation D1    Generated report: `N11_RESULTS.md`    Exploratory report, added after the results: `N11_EXPLORATORY.md`    Raw results: `results/n11/` and `results/n11_log.txt`, from `colab-results` branch `n11_strength_a100/20261020-190036`. Code commit c4f8d0a.

**How the run went.**
- **One session:** an NVIDIA A100 (80 GB), PyTorch 2.11, the same version as N10. It trained and measured all 30 models in 61 minutes, at 1.6–2.2 minutes of training per run.
- **No problems:** no resumes, no Drive failures and no skipped 16-bit steps.
- **The only change after approval** was deviation D1. A first session on a T4 ended at a usage limit after 14 runs, and all 30 were redone on one kind of GPU. Those T4 runs played no part in what follows.

## What we expected

**The test.** At 80% noise, can anchoring what head B *predicts* preserve its reading better than pinning the *directions* it reads? Five arms answered it, plus a clean control:
- **`n80`:** no payload. It is also the yardstick.
- **Three functional payloads (λ = 8, 32 and 128):** each trained head B's readout to match its own model's ideal noisy copy on 64 fixed passages.
- **`n80-geom`:** pinned head B's readout directions to head A's.

**The measure.** S is the share of the gap closed. The gap is measured against a yardstick no payload can move: the ideal copy built from the same seed's `n80` model.

**The claims.**
- **N11a (primary):** S at λ = 128 above 0.67.
- **N11b:** S rising with the strength.
- **N11c:** the λ = 128 payload beating the geometric pin by at least 0.30.
- **A manipulation check (precondition 5f)** came first, for N11a and N11c. At λ = 128, head B's divergence from the copy on the payload's own passages had to fall to at most half the unanchored arm's.

## What happened

| Arm | S, the share of the gap closed | Payload-window divergence ÷ `n80`'s | Head A's cost (nats) | Head B vs its copy (geometry) |
|---|---|---|---|---|
| `n80` | 0 (gap 0.48–0.50 nats) | 1 | — | 0.32 |
| `n80-f8` | 0.40 (0.39–0.41) | 0.67 | +0.004 | 0.37 |
| `n80-f32` | 0.45 (0.44–0.46) | 0.61 | +0.003 | 0.36 |
| `n80-f128` | 0.47 (0.46–0.47) | **0.59** | +0.005 | 0.35 |
| `n80-geom` | 0.19 (0.17–0.21) | 1.22 | +0.139 | 0.73 |

Ranges are across the five seeds.

- **N11a and N11c are untestable.** The functional payload failed its manipulation check. At λ = 128 it brought the divergence down only to 0.585–0.601 of the unanchored arm's, against at most 0.5 required. The rules say a payload that did not take hold cannot decide these claims, and they were followed.
- **N11b is supported.** S rose with the strength, f8 < f32 < f128, in all 5 seeds. The rise flattens, though: 0.40, 0.45, 0.47. Quadrupling the strength twice added only 0.07.

The other preconditions held:
- **Must match:** the clean arm's head B matched head A (ratio 0.955–0.980, losses within 0.008 nats).
- **Must differ:** the rotated control scored 0.07–0.20.
- **The yardstick:** it left a gap of at least 0.48 nats in every seed.
- **The geometric pin took hold:** overlap 0.997.
- **Stability:** every subspace was stable.

**Replication of N10's unanchored arm.** On new seeds and windows, head B's agreement with its copy was 0.32 (N10: 0.31). The gap was 0.49 (N10: 0.43).

## Why the payload failed its check: the dose check misled again

This is a design error of mine, of the kind the preregistration itself named as the only way this outcome could arise ("possible only if the dose check misled again").
- **The dose check stopped too early.** It ran 300 steps of the 2,879, at the peak learning rate. There, λ = 128 held its passages at 0.27 of the unanchored arm's divergence, and I recorded that it "holds at every checkpoint".
  - In fact the ratio was rising at each checkpoint: 0.15, 0.23, 0.27. I read a trend as a level.
  - Over the full run the two arms moved towards each other. The payload arm's divergence rose from about 0.021 at step 300 to 0.038 at the end; the unanchored arm's fell from about 0.078 to 0.063.
  - The training records show the same thing: the payload term rose steadily through training, from 0.027 at step 300 to 0.047 at the end (`N11_EXPLORATORY.md`).
- **The lesson.** This is lesson 7 of `CRITICAL_REVIEW.md` failing a second time in a new form. A dose check must cover the whole training schedule, not its start.
  - On the A100 a full run takes about two minutes, so a full-length dose check now costs almost nothing.

## What this teaches

These are preregistered secondary measures or exploratory analyses. None decides a claim.

- **The functional payload did a lot, and cost nothing.** On 421 passages it never saw, it closed 40–47% of the gap, at no cost to head A (at most +0.005 nats).
  - It carried over well: 82% of its effect on its own 64 passages appeared on new text. So the effect was not memorisation.
  - Compare N10's data payload, which memorised its passages and made head B worse on new text.
- **The geometric pin did the opposite.** It restored the *directions* head B reads, raising its agreement with the copy from 0.32 to 0.73. But it closed only 19% of the functional gap, and it made head A worse by 0.14 nats. That is N10's exploratory finding (16% and 0.13) again, now preregistered as a secondary measure and on new seeds.
- **Function and geometry came apart, in both directions.**
  - The functional payload restored a large part of what head B predicts while leaving its directions almost where they were (0.35, against 0.32 unanchored).
  - The pin restored the directions while leaving most of the function lost.
  - Preregistered secondary measure S1 asked whether the geometry would follow the function. It did not.
  - This is the pattern Son (2026) reports for model collapse, that only anchoring the function prevents it. It also fits substrate independence: the information head B carries could be partly restored without restoring the substrate it was read from. But the claim that would test this, N11c, is untestable, so this is an observation, not a verdict.
- **The divergence the check asked for may be harder to reach than it looked.** Two heads trained on *identical* clean targets differ from each other by 0.046 on these passages. The λ = 128 head was closer to its copy than that (0.038). The check asked for at most 0.032.
  - This is said to explain the failure, not to excuse it. The threshold was preregistered, and it stands.

## Two explanations for why the payload saturated, and the test between them

Both are conjectures, formed after the results.
1. **The ideal copy may be out of reach of head B's readout.** The ideal noisy copy is a mixture: 20% of head A's prediction plus 80% word frequencies. That makes it a mixture of two softmaxes. Yang et al. (2017) showed that a single softmax on a linear readout cannot in general represent such distributions: the *softmax bottleneck*. Head B is a single softmax on a rank-64 readout.
   - As head A sharpens during training, the mixture moves further from anything one softmax can represent. That would explain both the rising divergence and why more strength stops helping.
2. **The payload's weight may have been diluted.** Under amendment A1, the payload's gradient is added after the main loss's gradient is clipped. Early in training the clip binds hard, so the main loss is scaled down and the payload dominates. Later the clip stops binding, the main loss's share grows, and the payload's relative weight falls.
   - That would explain the rising divergence, but less easily the saturation across strengths.

**The test between them.** Freeze a trained `n80` model and train head B's readout alone on the payload to convergence. Explanation 1 predicts a floor near the 0.038 seen here. Explanation 2 predicts the divergence falls well below it. On the A100 this takes minutes.

## Which kind of result this is

- **Two untestable claims (N11a, N11c), caused by a design error:** a dose check too short to see where the payload would end up. The manipulation check worked as built and kept a payload that missed its target from being read as evidence either way.
- **One supported claim (N11b):** strength helps, but with sharply diminishing returns.
- **The substrate-independence question is still open, but narrower.** The functional payload closed more than twice the gap the geometric pin did (0.47 against 0.19), at no cost to the other reading against 0.14. Whether that difference counts was the job of N11c, which could not decide it.

## What this does not change

- Every earlier verdict stands, including N10's untestable ones.
- N9's finding replicated again: at 80% noise the trained head reads different directions from its ideal copy (0.32). The functional payload barely changed that (0.35).

## Exploratory observations, not confirmatory

- **The hardware.** D1 set aside the 14 interrupted T4 runs, to be compared with these runs after the verdict. That comparison needs their records from Drive. Two of them, seed 4003's control arms, were visible in the screenshot shared when D1 was decided.
  - `clean`: head B's clean loss 5.121 on the T4 against 5.142 on the A100; agreement with the copy 0.977 against 0.980.
  - `n80`: gap 0.503 against 0.500; agreement 0.319 against 0.320.
  - So the hardware moved losses by up to about 0.02 nats, but the gap and the agreement barely at all. Two runs are not a test; the 14-run comparison is still to do.

## Limitations

- **One noise rate, one payload size (64 passages), one geometric strength,** and a functional payload confined to head B's readout by amendment A1. That makes the test narrower than "anchoring function", as A1 recorded.
- **The manipulation check failed,** so the primary question is unanswered at any strength tested.
- **The same limits as before:** 10% of the budget, one learning rate, one size, one dataset, one readout width.

## Next conjectures

1. **Where the payload's floor comes from.** Freeze trained models and fit head B's readout alone to the copy, as above. If the floor is the softmax bottleneck, a functional anchor needs a readout that can represent mixtures, or a different target.
2. **Anchor the clean reading, not the noisy copy.** The noisy copy is 80% word frequencies, so matching it mostly teaches head B the noise. A payload that matches head A's own prediction, the clean reading, is a single softmax that head B's readout *can* represent. Whether it preserves more of the reading is a sharper version of N11's question.
3. **Rerun N11's question with a full-length dose check,** before queuing. The A100 makes it cheap.
4. **Finish D1's hardware comparison** on the 14 matched T4 runs. Every test from N1 to N10 ran on a T4 and later tests run on an A100, so the size of the hardware effect matters.

## References

Son, C. (2026). *J-space forecasts model collapse, but only anchoring the function prevents it* [Preprint]. Research Square. https://doi.org/10.21203/rs.3.rs-10713541/v1

Yang, Z., Dai, Z., Salakhutdinov, R., & Cohen, W. W. (2017). *Breaking the softmax bottleneck: A high-rank RNN language model* [Preprint]. arXiv. https://doi.org/10.48550/arxiv.1711.03953
