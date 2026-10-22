# Result note: N12

Date: 2026-10-21    Preregistration: `N12_PREREGISTRATION.md`, with amendment A1 and the necessity audit's additions    Generated report: `N12_RESULTS.md`    Calibration run: `N12_CALIBRATION.md`    Exploratory report, added after the results: `N12_EXPLORATORY.md`    Raw results: `results/n12/` and `results/n12_log.txt`, from Colab run `n12_bottleneck/20261021-155235`. Code commit 12f2957, whose job code is identical to the queued commit 9126974.

**How the run went.**
- **One session:** an NVIDIA A100 (80 GB) with PyTorch 2.11, as N11 used; the calibration had run on the 40 GB model. It trained the 20 models, ran the 10 fits and measured everything in 55 minutes.
- **No problems in the job:** no resumes, no Drive failures, every training draw before the held-out boundary. One 16-bit step was skipped by the gradient scaler across the 20 runs, which is routine.
- **The records came by a different route.** The notebook's upload step found no GitHub token in that session and downloaded a zip instead, which the author attached. The records, log and run information are committed under `results/n12/`. The results branch holds only this job's three earlier sessions, each stopped by the GPU check on an L4 before any training.
- **Changes after approval:** amendment A1 (the convergence test at zero loss) and the necessity audit's additions, both before the run. Nothing changed after it.

## What we expected

**The question.** Since N7, a head trained on targets with 80% replaced by random words has read different directions from the clean head, and N9 found it is not the ideal noisy copy of the clean head either. N12 tested one explanation: that the head's readout cannot represent what it is trained to predict. The best prediction of a noisy target is a mixture, 0.2 of the true next-word distribution plus 0.8 of the word frequencies, and a single softmax on a linear readout cannot in general produce a mixture (the softmax bottleneck, Yang et al., 2017).

**The arms,** on five new seeds, measured on 512 windows held out from the training text:
- **`n80`:** the standard head, trained on the noisy targets. It is also the yardstick: S is the share of its gap to the ideal copy that an arm closes.
- **`n80-mix`:** a head whose output *is* the mixture, trained on the same noisy targets (forward correction, Patrini et al., 2017).
- **`sub20`:** a standard head given the clean target at exactly the positions `n80`'s targets were left alone, and nothing elsewhere. It has the same informative targets as `n80`, without the random ones.
- **Two frozen fits** of head B's readout on each `n80` model, to that model's own ideal copy: `fit-single` with the usual single softmax, and `fit-mix` with the mixture form, which can reach the copy exactly.

**The claims.**
- **N12a (primary):** the mixture head closes at least two thirds of the gap.
- **N12b:** and reads the same directions as its ideal copy (agreement at least 0.85).
- **N12c:** `sub20` also closes two thirds, so scarcity alone does not cause the drift.
- **N12d:** a single softmax fitted directly to the copy closes less than two thirds, so the bottleneck is large.

The preregistration said the conjecture survives as a whole only if all four are supported.

## What happened

| Claim | Rule | Result (5 seeds) | Verdict |
|---|---|---|---|
| N12a | S(`n80-mix`) above 0.67 | 0.713, 97.5% interval [0.690, 0.736] | **Supported** |
| N12b | agreement of `n80-mix` with its copy ≥ 0.85 in 4 of 5 | 0.79–0.85, in 0 of 5 (and above 0.60 in 5 of 5) | **Intermediate** |
| N12c | S(`sub20`) above 0.67 | 0.796 [0.772, 0.820] | **Supported** |
| N12d | S(`fit-single`) below 0.67 | 0.616 [0.605, 0.626] | **Supported** |

All six preconditions held. The must-match controls were exact (the constructed mixture check scored 1.0000 in every model; `fit-mix` reached the copy at S = 0.997 and agreement 0.999), the rotated control scored 0.12–0.18 per model, and every subspace was stable (at least 0.86, against a floor of 0.80).

**By arm:**

| Arm | Gap to the copy (nats) | S | Agreement with its copy | Inner reading, minus head A (nats) | Cost to head A, vs `clean` (nats) |
|---|---|---|---|---|---|
| `n80` | +0.491 | 0 | 0.31 | +1.40 | +0.040 [+0.033, +0.047] |
| `n80-mix` | +0.141 | 0.71 | 0.83 (0.79–0.85) | +0.34 | −0.009 [−0.024, +0.007] |
| `sub20` | +0.100 | 0.80 | 0.91 (0.90–0.93) | +0.16 | +0.020 [+0.002, +0.039] |
| `fit-single` | +0.189 | 0.62 | 0.14 (0.13–0.15) | — | — (head A frozen) |
| `fit-mix` | +0.001 | 1.00 | 1.00 | — | — |

The "inner reading" is the head's own softmax, before the mixture; for `n80` that is the head itself. The last two columns are secondary measures with 95% intervals.

**Replication.** `n80`'s agreement with its copy was 0.308 (N11 0.323, N10 0.305, N9 0.31–0.35), and its gap 0.491 (N11 0.492), on new seeds and new windows.

## Which kind of result this is

- **The conjecture's functional claims survived a test they could have failed.** Changing the head's output form, with the same noisy targets and no payload, closed 71% of the gap. The data-matched clean head closed 80%, so too few informative targets is not, on its own, the cause. A single softmax fitted directly to the copy stopped at 62%, and none fitted did better, so by the best fit found the form's ceiling is about there, and the standard head's training reached none of what the form allows.
- **Its geometric claim fell just short.** The mixture head's directions moved most of the way to the copy's, from 0.31 to 0.83, but no seed reached 0.85. N12b is intermediate, so the conjecture as stated did not fully survive. This is the second time a head functionally close to its copy has stopped just under 0.85: N10's head at 20% noise scored 0.79–0.84.
- **The explanation's wording was too strong, as the necessity audit said before the run.** The preregistration's outcome table said that all four claims supported would mean the drift "was the readout's form, not noisy training". The numbers divide it:
  - A single softmax of the `n80` state reached 62% of the gap when fitted directly to the copy. So the form accounts for at most 38% of the gap, 0.384 [0.376, 0.393], The bound holds whatever the procedure, since a better fit could only lower it; how tight it is depends on this procedure, which reached the copy when the form allowed it.
  - The standard head, trained on the noisy targets, reached none of the 62% its form allowed. Training through a single softmax on noisy targets fails first, and the form's ceiling comes after.
  - The mixture head beat the fitted single softmax by 0.097 [0.079, 0.115]. So it did more than escape the ceiling.

## Why the mixture head does more than escape the ceiling

The audit found this before the run, and the results fit it. The mixture head's loss changes the learning signal as well as the output's form: for each target, its gradient is the single softmax's gradient scaled by the posterior probability that the target is clean, 0.2 p(y) / q(y). It therefore discounts targets it judges to be noise, as `sub20` ignores them outright. Two effects are mixed in N12a:
1. **Representation:** the mixture form can produce the copy, and a single softmax cannot. N12d bounds this effect: at most the 38% of the gap no fitted single softmax reached.
2. **A cleaner learning signal:** the discounting of random targets. Two differences bear on it without isolating it. The mixture head beat the fitted single softmax by 0.10, which could as well be the fitting procedure's shortfall as the cleaner signal; and `sub20`, which drops the random targets altogether, beat the mixture head by 0.083 [0.059, 0.107].

N12 did not separate the two effects, and neither verdict needs them separated. A test that would is in the next conjectures.

## What this teaches

- **Function and geometry came apart again, in the strongest form yet.** The single softmax fitted directly to the copy was functionally the closest a single softmax got, 62% of the gap, yet it read directions almost unrelated to the copy's: agreement 0.13–0.15, the rotated control's level, from stable subspaces (its own 0.97–0.98, the copy's 0.91–0.96). Fitting the function directly made the geometry *worse* than training on noisy targets had (0.31). The mixture head's geometry, by contrast, followed its function most of the way (0.83). *A conjecture about why:* a single softmax can only imitate a mixture by becoming nearly flat with small modulations, and the directions those modulations depend on are its own; the mixture form inherits the copy's directions through its inner softmax. Exploratory.
- **The random targets leave a geometric trace that function does not show.** `sub20` and `n80-mix` are functionally close (80% and 71% of the gap), but `sub20` reads the copy's directions at 0.91 and `n80-mix` at 0.83. The random targets, even when the head's form discounts them, moved the directions. This may be why N12b and N10c both stopped just under 0.85. [N14, 2026-10-22: function moved too, and by the amount that accounts for the directions within 0.02 in the point estimates (N14a inconclusive by its rule); see `N14_NOTE.md`.]
- **The standard noisy head costs the shared network, and the mixture head costs it little or nothing.** Head A, trained on clean targets, was 0.040 nats worse [0.033, 0.047] when head B was the standard noisy head, and −0.009 [−0.024, +0.007] when head B was the mixture head, an interval that allows a cost of up to 0.007. `sub20` cost 0.020 [0.002, 0.039], so a representable target removes only part of the cost. This looks like N4's cost of sharing in a new form. *A conjecture about the mechanism:* a head whose target it cannot represent pulls the shared representation towards what it can, and the random targets themselves cost something more. Exploratory. [N13, 2026-10-22: a constant discount of 0.2 on the standard head's loss, with the same random targets, left no measurable cost (−0.011 [−0.026, +0.004] against `clean`, 95%). On this evidence the cost tracks how hard head B pulls rather than the random targets as such, though `sub20` cost more than that scaled head (paired differences +0.018 to +0.048), which is not understood. See `N13_NOTE.md`.]
  - **This corrects N10's note,** which said head A was "unaffected by noise in head B alone", from a range of validation losses (5.13 to 5.19) across arms. N10's own records show `n80`'s head A worse than `clean`'s by 0.026–0.046 nats in every seed, and N11's by 0.027–0.061. The range hid a consistent shift. A dated correction has been added to `N10_NOTE.md`.
- **The inner reading recovered most of head A, not all of it.** The mixture head's own softmax was 0.34 nats worse than head A on clean targets (against 1.40 for the standard head and 0.16 for `sub20`). "Keeps the reading" in N12a's sense, the noisy-target prediction, is true at 71%; the clean reading inside it is clearly better than the standard head's, and clearly short of head A's.
- **What it means for N9's refutation.** N9b stays refuted: the trained noisy head is not the ideal copy. N12 gives that refutation an explanation with two parts, each now bounded: the single-softmax form accounts for at most 38% of the gap (the best fit found reached 62%), and training on noisy targets through that form reaches none of what the form allows. The explanation forbade something specific, and the forbidden result did not occur: a head given a form that can represent the mixture, trained on the same targets, could have stayed as far from the copy as the standard head, and it did not. The audit noted before the run that a rival, a cleaner learning signal, predicts the same for N12a; N12d, which the rival does not need, is what separates them, and it too was supported, in the modest form the audit prescribed.
- **What it means for N10 and N11.** N11's functional payload saturated at 47% of the gap; the best single softmax found here by direct fitting reached 62%, above that, so the payload stopped short of what a single softmax of this state can do. Whether N11a's two thirds was within reach of the head's form at all, N12 cannot say: that would need a fit that reaches it, or a proof that none can. What N12 does show is that the payload's target, the mixture, is one the single-softmax fit did not reach (a divergence of 0.034, against 0.0001 for the mixture-form fit).

## What this does not change

- Every earlier verdict stands, including N9b's refutation and N11's untestable claims.
- The agreement measure was not changed by this test; its must-match and must-differ controls held as before.

## Limitations

- **The mixture head is told the noise:** its 0.2, 0.8 and u are the ones that generated the targets. N12 tests an explanation, not a method for unknown noise.
- **N12d's support is bounded by one fitting procedure.** Another procedure might fit a single softmax closer. The must-match fit shows the procedure reaches the copy when the form allows, which is the available check.
- **The held-out tail is training text,** never trained on by any N12 model or fit, but not a designed test set.
- **One noise rate, one readout width, one scale, 10% of the budget.**

## Next conjectures

1. **Separate the two effects.** Train a single-softmax head with the mixture head's gradient weighting but a single-softmax output (reweighting without representation), and a mixture head whose gradient is not discounted (representation without reweighting). If the first closes about 62% of the gap and the second about 38%, the two effects add; if either alone closes most of it, the explanation simplifies. [Tested by N13, 2026-10-22, with a different design: the chain rule shows that the first arm is the mixture head's own training, and the second a readout rather than a training. The discounting does the work, and the form applied at readout to the standard head doubles its gap. See `N13_NOTE.md`.]
2. **Where the random targets leave their geometric trace.** `sub20` reads the copy's directions at 0.91 and `n80-mix` at 0.83. A dose of random targets at a fixed number of informative ones, with the mixture head, would show whether the trace grows with the noise. If it does, the 0.85 bar measures something that function does not, and N10c's and N12b's near misses have a cause. [Tested by N14, 2026-10-22, with a clean-target ladder as the comparison: in the point estimates, within 0.02 (N14a inconclusive by its rule), the heads with random targets read head A's directions as a clean-target head with the same clean-target loss does, and the bar corresponds to a clean-target loss about 0.26 nats from head A's. The conjecture did not survive as stated; see `N14_NOTE.md`.]
3. **Return to N10's question with mixture heads.** N10 and N11 asked whether anchoring function preserves a reading better than pinning geometry, and both payloads failed their checks, and both were aimed at the mixture, which N12's fits show a single softmax of this state does not reach. N12 did not test whether that is why the checks failed. With a mixture head the target is reachable, so a functional payload has at least that obstacle removed; whether its check then passes is for the test to show.
4. **The cost to the shared network.** Whether the standard noisy head's 0.04-nat cost to head A grows with the noise rate, and whether the mixture head's absence of cost holds at higher rates, would tie this to N4's cost of sharing.

## References

Patrini, G., Rozza, A., Krishna Menon, A., Nock, R., & Qu, L. (2017). Making deep neural networks robust to label noise: A loss correction approach. In *2017 IEEE Conference on Computer Vision and Pattern Recognition (CVPR)* (pp. 2233–2241). IEEE. https://doi.org/10.1109/cvpr.2017.240

Yang, Z., Dai, Z., Salakhutdinov, R., & Cohen, W. W. (2017). *Breaking the softmax bottleneck: A high-rank RNN language model* [Preprint]. arXiv. https://doi.org/10.48550/arxiv.1711.03953
