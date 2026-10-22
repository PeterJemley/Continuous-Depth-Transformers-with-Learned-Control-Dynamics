# Result note: N13

Date: 2026-10-22    Preregistration: `N13_PREREGISTRATION.md` (no amendments)    Generated report: `N13_RESULTS.md`    Calibration run: `N13_CALIBRATION.md`    Exploratory report, added after the results: `N13_EXPLORATORY.md`    Raw results: `results/n13/`, from Colab run `n13_signal/20261022-004948`, code commit fa14e3b, whose job code is identical to the calibrated commit 122c955.

**How the run went.**
- **One session:** an NVIDIA A100 (80 GB) with PyTorch 2.11, the GPU N12 ran on. It trained the 15 models, re-measured N12's 15 saved models and ran the 15 cross-model comparisons in 35 minutes, about 2 minutes per training run.
- **No problems:** no resumes, no Drive failures, no 16-bit steps skipped, every training draw before the held-out boundary, and every fingerprint equal to N12's `n80`'s.
- **The records arrived by the usual route,** uploaded to the results branch and committed under `results/n13/`.
- **Changes after approval:** none. The calibration corrected the stated mechanism's claim about the start of training, before any run, with no claim or rule changed.

## What we expected

**The question.** N12 found that a head whose output is the noisy mixture, trained on 80%-noisy targets, closes 71% of the gap to the ideal copy, and its necessity audit found that the design mixed two effects: the head's form can represent the mixture, and its training signal discounts targets it judges to be random. N12's note proposed to separate them with two arms. The preregistration showed those arms are the wrong design. By the chain rule, the gradient of the mixture loss −log q(y), with q = 0.2 p + 0.8 u, is r(y)·(p − e_y): the single softmax's gradient scaled by r(y) = 0.2 p(y) / q(y), the posterior that the target is clean. So "reweighting without representation" *is* the mixture head's inner softmax, and "representation without reweighting" is the standard head read out in mixture form. The questions that remain are whether the pipeline follows the identity, whether the form at readout does anything on its own, and whether the discount works by telling targets apart or only by making head B pull less on the shared network.

**The conjecture.** The mixture head's advantage is its learning signal, a discount on each target by the estimated probability that it is clean, in context. The output form matters only at readout.

**The arms,** on N12's five seeds and the same 512 held-out windows, with N12's saved models re-measured beside them:
- **`n80-rw`:** a single softmax trained with the discounted cross-entropy, r(y) computed from its own logits with the gradient stopped. By the identity, the mixture head's inner softmax under another name.
- **`n80-mix-rep`:** `n80-mix` trained again, for the replicate floor.
- **`n80-w02`:** a single softmax with loss 0.2 · CE, the scale-only control. AdamW removes a constant scale from head B's own update, so this changes only head B's share of the trunk's gradient.
- **`n80` in mixture form:** N12's standard head, its softmax placed in 0.2 softmax + 0.8 u at readout, after training.

**The claims.**
- **N13a (primary):** |Δ_A(`n80-rw`) − Δ_A(`n80-mix`)| ≤ 0.03 nats in at least 4 of 5 seeds, where Δ_A is the inner softmax's clean-target loss minus head A's. Refuted if above 0.10 in 4 of 5, with the replicate within 0.03.
- **N13b:** S(`n80`, mixture readout) has its 97.5% interval wholly below 0.10. Refuted if wholly above 0.33.
- **N13c:** D = Δ_A(`n80-w02`) − Δ_A(`n80`) has its interval wholly above −0.27, a quarter of N12's gain of 1.062 nats. Refuted if wholly below −0.53.

## What happened

| Claim | Rule | Result (5 seeds) | Verdict |
|---|---|---|---|
| N13a | within 0.03 in at least 4 of 5 | −0.005 to +0.003, within 0.03 in 5 of 5 | **Supported** |
| N13b | interval wholly below 0.10 | S = −1.014, 97.5% interval [−1.045, −0.983] | **Supported** |
| N13c | interval wholly above −0.27 | D = +0.050 [+0.036, +0.064] | **Supported** |

All five preconditions held. The re-measurements reproduced N12's records exactly (every loss and ratio differed by 0); the constructed mixture check and the zero-noise copy were exact in every model; the rotated control's arm means were 0.12–0.16; gap(`n80`) was 0.48–0.50 nats; and the replicate was bit-identical to the original in every seed, with a per-position divergence of 0.0000, so the replicate floor is 0.

**By arm:**

| Arm | Δ_A: inner reading minus head A (nats) | Inner softmax's agreement with head A | S, in mixture form | Head A, vs N12's `clean` (nats) |
|---|---|---|---|---|
| `n80` (N12) | +1.400 | 0.31 | −1.01 (its softmax placed in the mixture form) | +0.040 [+0.033, +0.047] |
| `n80-mix` (N12) | +0.338 | 0.81 | +0.71 | −0.009 [−0.024, +0.007] |
| `sub20` (N12) | +0.163 | 0.90 | +0.80 | +0.020 [+0.002, +0.039] |
| `n80-rw` | +0.337 | 0.80 | +0.71 | −0.006 [−0.024, +0.012] |
| `n80-mix-rep` | +0.338 | 0.81 | +0.71 | −0.009 [−0.024, +0.007] |
| `n80-w02` | +1.451 | 0.33 | −1.02 (its softmax placed in the mixture form) | −0.011 [−0.026, +0.004] |

The last column's intervals are 95% and exploratory.

**The identity, seed by seed.** `n80-rw` against `n80-mix`: Δ_A differed by −0.0052, +0.0009, +0.0028, −0.0017 and −0.0010; S in mixture form by at most 0.013; the agreement with head A by at most 0.02. Per position, the two models differ by 0.097 nats of symmetrised KL at the inner softmax and 0.095 at head A, where the replicate differs by 0.0000.

**The scale control, taken apart** (`n80-w02` against `n80`, paired by seed): head A's clean-target loss fell by 0.051 [0.036, 0.066]; head B's own softmax changed by −0.001 [−0.011, +0.009]. D is the second minus the first.

**The discount over training** (`n80-rw`, means over seeds, kept / replaced): step 0: 0.092 / 0.089; step 100: 0.140 / 0.142; step 300: 0.302 / 0.156; step 1,500: 0.516 / 0.113; step 2,850: 0.581 / 0.103. Its batch mean was 0.189 averaged over the schedule and 0.197 at the end. The same quantity computed from `n80-w02`'s logits, where no discount was in use, ended at 0.366 / 0.193.

## Which kind of result this is

- **Each claim could have failed in a specific way, and none did.** N13a would have failed if the pipeline's mixed precision, its clipping or a bug broke the identity: the differences were a sixth of the tolerance. N13b would have failed if the standard head's softmax were a clean estimate that the form could complete into the copy: instead the form doubled its gap. N13c would have failed if the mixture head's gain came from head B pulling less on the shared network: the constant discount lowered the pull as much as the real discount does on average (0.2 against 0.19) and gave head B's reading nothing.
- **The conjecture survived as a whole,** where N12's had fallen short on geometry. Its claims are modest: they fix what the form does in training (nothing beyond the discount), what it does at readout on a head trained without the discount (harm), and what the discount does not work through (the magnitude of the pull).
- **The critic's objection stands as the preregistration stated it:** N13a tests whether the pipeline implements the mathematics, and its support is knowledge about the pipeline, not the world. Its value is in what N13c means given it, and in a measured fact: two implementations of the same mathematics, trained on the same data in the same order, differ in Δ_A by up to 0.005 nats, in S by up to 0.013 and in agreement by up to 0.02, while a replicate of the same code differs by nothing. Every later paired comparison has that scale of noise to respect.

## What the three results say together

N12 named two effects in the mixture head, representation and a cleaner learning signal, and left them mixed. N13 shows they are not two causes of the gain but two roles of one form:
- **In training, the form is the discount and nothing else** (N13a). The mixture head's inner softmax is trained toward a clean target: each noisy target counts by the head's own posterior that it is clean, which by the end of training weighted a kept target 5.6 times a replaced one (0.58 against 0.10). A single softmax can represent that target, and does: `n80-rw`'s softmax is 0.34 nats from head A on clean targets, where the standard head's is 1.40.
- **At readout, the form expresses the noisy-target prediction from the clean estimate.** That is the role N12d measured a single softmax cannot play well (62% of the way to the copy by direct fitting), and it is why the mixture head reaches 71% from a clean inner estimate. On a head whose softmax is not a clean estimate, the standard head's, the form at readout does harm: the gap doubles (N13b), by 0.49–0.50 nats in every seed.
- **The discount works by telling targets apart, not by pulling less** (N13c). The constant discount, which lowers head B's pull on the trunk as the real discount does on average, left head B's own softmax where the standard head's was (−0.001 [−0.011, +0.009]).

So the explanation now reads: *the standard noisy head drifts because it is trained to fit every target, random ones included, through a form that cannot represent what that training aims at; the mixture head does not drift because its training discounts the random targets in context, which leaves its inner softmax a clean target that a single softmax can represent, and its form supplies the noisy prediction at readout.* N12's bound stands: the single-softmax form accounts for at most 38% of the gap, as a measure of what the readout can represent of the copy. N13 adds that the mixture head's gain is not won by representing more, but by learning something else.

## What this teaches

- **Where the cost to head A comes from.** N12 found that the standard noisy head costs head A 0.040 nats and the mixture head costs it nothing. The constant discount removed the whole cost (−0.011 [−0.026, +0.004] against `clean`, beside the mixture head's −0.009 and `n80-rw`'s −0.006), with the same random targets at a fifth of the weight. So the cost is a matter of how hard head B pulls on the shared network, not of the random targets as such. This narrows N12's exploratory conjecture that "the random targets themselves cost something more": at a fifth of the weight they cost nothing measurable. A puzzle remains, exploratory: `sub20`, with clean targets on a fifth of the positions, cost head A 0.020 [0.002, 0.039], more than `n80-w02` did with random targets at a fifth of the weight; the paired difference is positive in every seed, +0.018 to +0.048. The cost does not track how random head B's targets are.
- **Head B's gain and head A's relief come apart.** Under the constant discount, head A improved by 0.051 and head B's own reading by nothing; under the real discount, head A improved by about the same (0.046–0.049) and head B's reading by 1.06 nats. The mixture head's two effects on the two heads have different causes: the pull's magnitude for head A, the discrimination for head B.
- **The discount sharpens itself.** Computed from the standard head's logits, where it was not in use, the discount would weight a kept target 1.9 times a replaced one at the end of training; in the head trained with it, 5.6 times. A head trained with the discount becomes better at the discrimination the discount depends on. Descriptive.
- **The discount's time course corrected a stated mechanism.** The preregistration said the discount tells kept from replaced targets "from the start, by frequency". The calibration found the two means equal at the start, and they must be: both kinds of target have the unigram distribution. The discount separates them only through context, from about step 200 on; the replaced mean, after rising with the kept one to 0.14 at step 100, falls back to 0.10 as the kept mean rises to 0.58. The mechanism's claim was fixed to the wrong time, the same error as N11's dose check, caught this time before the run.
- **Geometry followed function exactly, where function was the same.** `n80-rw`'s inner softmax reads head A's directions at 0.80 and `n80-mix`'s at 0.81; `n80-w02`'s at 0.33 and `n80`'s at 0.31. The same training signal gave the same directions, to within the noise between implementations.
- **A replicate floor of zero, and what it does not measure.** The pipeline is deterministic: the replicate differed from the original by nothing. So the replicate floor measured determinism, not the noise between two implementations of the same mathematics, which was 0.005 nats in Δ_A and 0.09–0.10 nats per position. That per-position divergence between `n80-rw` and `n80-mix`, two models with the same aggregate behaviour, is comparable to the divergence between `n80-w02` and `n80`, two models trained with different losses (0.016 at the inner softmax and 0.138 at head A). *A conjecture about why:* the two losses are computed by different arithmetic, their gradients differ at the eighth significant figure, and training amplifies that into different parameters with the same statistics. Comparisons across models must be made on aggregates; a per-position divergence between models cannot tell a different learning signal from a different rounding.

## What this does not change

- Every earlier verdict stands. N12b stays intermediate, and N9b refuted.
- N12's bound on the form's share of the gap, at most 38%, stands as a bound on what a single softmax can represent of the copy.
- The agreement measure was not changed; its must-match and must-differ controls held as before.

## Limitations

- **Reused seeds and models.** Every N13 comparison is within N12's seeds, against N12's saved models, by necessity: the identity and the replicate need the same training. New seeds would add nothing to N13a or N13b and could only widen N13c's interval.
- **The mixture head is told the noise:** its 0.2, 0.8 and u are the ones that generated the targets, in every arm. Whether a head that must estimate them keeps the advantage is untested.
- **N13a tests the pipeline, not the world.** Its support is a property of this code on this GPU.
- **The scale control uses one constant,** 0.2, the prior clean-ness. The real discount's mean over training was 0.19, so the match is close, but a dose of constants was not run.
- **One noise rate, one readout width, one scale, 10% of the budget.**

## Next conjectures

1. **Where the random targets leave their geometric trace** (N12's conjecture 2), now with the discounting head: a dose of random targets at a fixed number of informative ones. If the trace grows with the dose while the discount keeps the function, N10c's and N12b's near misses at 0.85 have a cause.
2. **The cost of sharing as a function of the pull.** A dose of the constant (0.05, 0.2, 0.5, 1.0) on the standard head would show whether head A's cost rises with head B's weight. The `sub20` puzzle, clean targets on a fifth of the positions costing more than random ones at a fifth of the weight, says the content of the targets matters too, in a way not yet understood.
3. **Return to N10's question with mixture heads** (N12's conjecture 3). The functional payload's target, the mixture, is reachable by a mixture head; N13 adds that the head's clean inner estimate is what a single softmax can hold.
4. **Unknown noise.** The discount is a posterior under known noise parameters. A head that estimates ρ and u from its own predictions is the method version of this result, outside this project's question but the natural test of whether the advantage survives not being told.

## References

Patrini, G., Rozza, A., Krishna Menon, A., Nock, R., & Qu, L. (2017). Making deep neural networks robust to label noise: A loss correction approach. In *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 2233–2241). IEEE. https://doi.org/10.1109/cvpr.2017.240

Yang, Z., Dai, Z., Salakhutdinov, R., & Cohen, W. W. (2017). *Breaking the softmax bottleneck: A high-rank RNN language model* [Preprint]. arXiv. https://doi.org/10.48550/arxiv.1711.03953
