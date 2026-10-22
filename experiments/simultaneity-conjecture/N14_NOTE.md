# Result note: N14

Date: 2026-10-22    Preregistration: `N14_PREREGISTRATION.md`, with amendment A1    Generated report: `N14_RESULTS.md`    Calibration run: `N14_CALIBRATION.md`    Exploratory report, added after the results: `N14_EXPLORATORY.md`    Raw results: `results/n14/`, from Colab run `n14_dose/20261022-122325`, code commit 8b4fb43.

**How the run went.**
- **One session:** an NVIDIA A100 with 40 GB and PyTorch 2.11 (N12's calibration had run on this model; its real run and N13 on the 80 GB one). It trained the 40 models, re-measured N12's `sub20` and `n80-mix` and measured everything in 93 minutes, about 2 minutes per training run.
- **No problems:** no resumes, no Drive failures, no 16-bit steps skipped, every training draw before the held-out boundary, every position set nested and at its design fraction, every fingerprint equal to the setup check's.
- **The re-measurements** reproduced N12's records exactly (every loss and ratio differed by 0), on the 40 GB model.
- **Changes after approval:** amendment A1, the fifth rung of the ladder, after the calibration run and before the real run. Nothing after it.

## What we expected

**The question.** In N12, `sub20` (clean targets at the 20% of positions `n80` leaves unreplaced, nothing elsewhere) and `n80-mix` (the same clean targets, random targets at the other 80%, with the mixture head) were functionally close, yet read head A's directions at 0.90 and 0.81. N12's note read the difference as a geometric trace of the random targets that function does not show, and offered it as a possible cause of two near misses at the 0.85 bar, N12b and N10c. N14 asked whether a head with random targets reads head A's directions worse than a clean-target head with the *same clean-target loss* does, and whether the trace grows with the number of random targets.

**What the preregistration said was already in hand.** Function also moves between the two arms: the inner reading's clean-target loss minus head A's, Δ_A, is 0.16 for `sub20` and 0.34 for `n80-mix`, and the drop in agreement per nat along that axis (0.52) equals the drop per nat from N12's `clean` arm to `sub20` (0.49), where only the number of clean targets changes. If a clean-target ladder continued at that slope, the rival, that geometry follows function, would predict `n80-mix`'s agreement at about 0.81, where it is. N14 measured the ladder instead of extrapolating it.

**The conjecture.** Random targets move the directions a head reads away from the clean head's by more than their cost to the head's clean-target prediction accounts for, and by an amount that grows with their number.

**The arms,** on N12's five seeds, with N12's `sub20` and `n80-mix` re-measured as the ends of the dose:
- **The dose:** `r10`, `r20`, `r40`: random targets at a nested 10, 20 and 40% of positions, drawn from `n80`'s replaced positions, on top of the fixed 20% clean ones, with the mixture head told each arm's true noise rate (1/3, 1/2, 2/3). `sub20` is dose 0 and `n80-mix` dose 0.8.
- **The ladder:** `sub10`, `sub05`, `sub025` and, by amendment A1, `sub0125`: clean targets at a nested 10, 5, 2.5 and 1.25% of positions, no random ones. With `sub20` it gives, for each seed, the agreement a clean-target head reads at any clean-target loss in its range, by linear interpolation: â.
- **The control:** `sub20-rw`, `sub20`'s training with each kept target weighted by the discount at ρ = 0.8 and no random targets, the mixture head's reweighting of the kept targets on its own.

**The claims.**
- **N14a (primary):** at dose 0.8, the residual b = â − a is a material part of the 0.09 drop: its 97.5% interval wholly above 0.03. Refuted if wholly below 0.01, with a standard deviation of at most 0.03.
- **N14b:** the trace at the smallest dose is less than half the trace at the largest, D(`r10`) < D(`n80-mix`) / 2, in at least 4 of 5 seeds. Refuted if at least three quarters in 4 of 5.
- **N14c:** the reweighting alone costs less than 0.03 of agreement: a(`sub20`) − a(`sub20-rw`) wholly below 0.03. Refuted if wholly above 0.06.

The preregistration also said what the design could detect: a residual of about 0.06 or more; and that the calibration run had shown the control's function collapsing, so that N14c's direct comparison would read that collapse as the reweighting's effect. The author kept the rule and added the rung.

## What happened

| Claim | Rule | Result (5 seeds) | Verdict |
|---|---|---|---|
| N14a | b(`n80-mix`) wholly above 0.03 | +0.009, 97.5% interval [−0.035, +0.052], standard deviation 0.028 | **Inconclusive** |
| N14b | D(`r10`) < D(`n80-mix`) / 2 in 4 of 5 | in 2 of 5; the ratio 0.03–0.74, mean 0.45 | **Intermediate** |
| N14c | a(`sub20`) − a(`sub20-rw`) wholly below 0.03 | +0.125, 97.5% interval [+0.080, +0.169], standard deviation 0.028 | **Refuted** |

All six preconditions held: the re-measurements reproduced N12; the constructed checks at 0.8 and at each arm's ρ, the zero-noise copy and the rotated control (0.15–0.16 by arm) as before; the ladder bracketed and ordered `n80-mix` in every seed, and its fifth rung bracketed the control in every seed; the trace existed in every seed (0.047–0.127).

**The ladder** is smooth in every seed, and its rungs vary by up to 0.05 across seeds. Its five rungs, as (Δ_A, agreement), averaged over seeds: (0.16, 0.90), (0.28, 0.84), (0.43, 0.78), (0.61, 0.71), (0.82, 0.64); its slope falls from about −0.5 to −0.3 per nat along the way; its leave-one-out residuals average 0.004 to 0.005, the largest 0.021.

**The dose arms sit on it.** Their residuals, dose by dose: +0.019 [−0.006, +0.044], +0.008 [−0.023, +0.040], +0.016 [−0.019, +0.051], +0.009 [−0.026, +0.043]; pooled over the four doses, +0.013 [−0.011, +0.037] (95% intervals, exploratory). Every interval includes 0, and no point estimate reaches the 0.03 margin.

**By arm** (means over seeds):

| Arm | Clean targets | Random targets | Δ_A (nats) | Agreement with head A, a | The ladder at the same Δ_A, â | b = â − a | Head A, vs N12's `clean` |
|---|---|---|---|---|---|---|---|
| `sub20` | 20% | 0 | 0.163 | 0.896 | on the ladder | 0 | +0.020 [+0.002, +0.039] |
| `r10` | 20% | 10% | 0.206 | 0.856 | 0.875 | +0.019 | +0.009 [−0.012, +0.030] |
| `r20` | 20% | 20% | 0.232 | 0.854 | 0.862 | +0.008 | −0.003 [−0.018, +0.011] |
| `r40` | 20% | 40% | 0.271 | 0.828 | 0.843 | +0.016 | +0.000 [−0.026, +0.027] |
| `n80-mix` | 20% | 80% | 0.338 | 0.805 | 0.814 | +0.009 | −0.009 [−0.024, +0.007] |
| `sub10` | 10% | 0 | 0.282 | 0.838 | on the ladder | 0 | +0.004 [−0.019, +0.026] |
| `sub05` | 5% | 0 | 0.428 | 0.775 | on the ladder | 0 | −0.007 [−0.030, +0.015] |
| `sub025` | 2.5% | 0 | 0.610 | 0.708 | on the ladder | 0 | −0.018 [−0.034, −0.002] |
| `sub0125` | 1.25% | 0 | 0.824 | 0.638 | on the ladder | 0 | −0.020 [−0.037, −0.003] |
| `sub20-rw` | 20%, reweighted | 0 | 0.736 | 0.772 | 0.668 | −0.104 | −0.007 [−0.024, +0.009] |

The last column's intervals are 95% and exploratory.

**The bar.** The ladder crosses 0.85 at Δ_A = 0.255 [0.225, 0.284] nats from head A; `n80-mix`'s inner reading is at 0.338, which is 0.084 [0.055, 0.112] nats beyond it.

**The dose.** The trace D(d) = a(`sub20`) − a(d) was 0.040, 0.042, 0.069 and 0.091 at doses 0.1, 0.2, 0.4 and 0.8; the functional cost Δ_A(d) − Δ_A(`sub20`) was 0.042, 0.069, 0.107 and 0.175. The random targets' share of head B's summed gradient weight at the end of training was 0.14, 0.22, 0.31 and 0.42 (it starts at each arm's ρ and falls as the discount learns to discriminate). The trace per unit of share was 0.20–0.29 at every dose, and the functional cost per unit of share 0.30–0.41.

**The control.** `sub20-rw`'s inner reading was 0.736 nats from head A, near the clean-target loss of a 1.25% clean head (0.09 nats better than it), while it read head A's directions at 0.772, as a 5% clean head does. Against the ladder at its own loss it sits 0.104 [0.070, 0.138] *above*, not below.

## Which kind of result this is

- **The conjecture did not survive.** Its primary claim is undecided by its rule and the numbers lie on the rival's side; its growth clause is intermediate; its third clause is refuted as stated. The preregistration had recorded, before the run, that the numbers already in hand predicted this if the ladder was straight. The ladder was measured, and the random-target heads lie on it.
- **Why "inconclusive" rather than "refuted".** Refutation needed the interval wholly below 0.01 with a standard deviation of at most 0.03. The point estimate is 0.009, a tenth of the drop, but the residual's spread across seeds (0.028) is as large as the margin, so the interval reaches 0.052. The preregistration said that five seeds could resolve a residual of about 0.06 or more, and that a smaller one would most likely read "some effect, size unclear"; this one, smaller still, read "inconclusive", the rule's outcome below both margins. The spread comes mostly from `n80-mix`'s own agreement across seeds (0.78–0.85, N12's values), not from the ladder, whose leave-one-out error is 0.02 at most.
- **What the rule's "refuted" for N14c means, and what it does not.** The reweighting alone did move the directions, by more than the whole trace, so the clause as written is false. But the clause meant a trace beyond function, and the control is the opposite case: its function collapsed to near a 1.25% clean head's and its directions stayed a 5% clean head's. The calibration run had shown this before the real run, and the preregistration said so; the rule was kept, and this reading is exploratory.

## What the results say together

**Along the axis of random targets, geometry followed function.** Four heads trained with 10 to 80% random targets at a fixed 20% of clean ones read head A's directions as a clean-target head with the same clean-target loss does, within about 0.02. Within the design's noise there is no trace that function does not show; the small positive residual at every dose (0.008–0.019) is what a trace smaller than the design could resolve would look like, and also what nothing would look like.

**The 0.85 bar is, on this evidence, a functional bar.** A head reaches it when its inner reading is within about a quarter of a nat of head A's on clean targets. The mixture head's inner reading in N12 was a third of a nat away, and 0.084 nats past the bar's equivalent, so N12b's near miss needs no geometric explanation. Whether N10c's does cannot be said from here: N10 did not record its head's inner reading against head A in this form.

**The trace and the functional cost both track the random targets' share of the gradient.** The share falls from each arm's noise rate to about half of it as the discount learns to tell targets apart, and what is left of the random targets in the gradient at the end rises with both the trace and the functional cost. This is consistent with the mechanism the conjecture named, working through function; it is exploratory, and since the share is itself monotone in the dose, it shows no more than that both track the dose.

**The reweighting without random targets is a different system.** The discount weights each kept target by the head's own posterior that it is clean, computed against the word frequencies. Its clean-target loss collapsed while the directions it reads stayed those of a far better head. *A conjecture about why, exploratory:* with no random targets in the loss, nothing pulls the head toward those frequencies, so the discount lets the head neglect every target it predicts poorly; the random targets' gradient keeps the inner softmax honest about the frequencies that the discount compares each target against, and the mixture head is self-consistent in a way the reweighting alone is not. On this evidence, one control at one noise rate, the discount as a method needs the noise it was built for.

## What this teaches

- **The ladder is a yardstick the project did not have.** For a head trained on clean targets, agreement with head A is a smooth function of its clean-target loss, to within 0.02 in every seed by its leave-one-out residuals, with the rung values varying by up to 0.05 across seeds. Any future head's agreement can be read against it, which turns "how far has geometry moved" into "has it moved more than function says".
- **The relation is not a law.** The reweighted control reads 0.10 better than its loss predicts. So the agreement ratio is not simply a function of the clean-target loss; the heads trained with random targets happen to follow the clean-target curve, and that is the finding, not a general rule.
- **Head A's cost is not monotone in head B's clean targets** (exploratory, S5). Against the `clean` arm, whose head B trains on every clean target, head A was worse under `sub20` (+0.020 [+0.002, +0.039]) and better under `sub025` and `sub0125` (−0.018 and −0.020; these are 95% intervals excluding 0, and at the preregistered 97.5% level every interval includes 0): a nearly absent head B leaves head A better off than a full one, N4's cost of sharing in a new form, while a head B with a fifth of the clean targets costs head A more than one with all of them. Adding random targets lowered the cost (+0.020 to −0.009 across the dose). N13's reading, that the cost tracks how hard head B pulls, does not cover the non-monotone part: `sub20` pulls a fifth as hard as `clean` and costs more. Not understood.
- **The N14c rule measured the wrong thing, and the calibration said so.** A control must be matched on the axis the claim is about, here the clean-target loss, or the rule must correct for it. The direct comparison was approved, the calibration showed what it would read, and it was kept; the ladder reading is beside it. The lesson is for the design stage: when a calibration shows that a rule will answer a different question from the one asked, the rule needs the amendment, not only the measurement that makes a better reading possible.

## What this does not change

- Every earlier verdict stands. N12b stays intermediate; N13's three claims stay supported.
- The observation behind the conjecture stands as a fact: `sub20` and `n80-mix` read head A's directions at 0.90 and 0.81, with the difference positive in every seed. What changed is its reading.

## Limitations

- **Five seeds,** with the residual's spread equal to N14a's margin; the design's limit was stated in advance, and it was reached.
- **Reused seeds and models:** the dose's ends are N12's models, by necessity.
- **The ladder is one way of lowering function without random targets.** The control shows the agreement at a given loss depends on how the loss was reached; another ladder (a weaker readout, a shorter schedule) might sit elsewhere.
- **The agreement measure is symmetric** between the heads; head A's cost bears on which head moved only functionally.
- **One noise distribution (u), one readout width, one scale, 10% of the budget;** the discounting head is told the noise.

## Next conjectures

1. **Sharpen the discount and see whether geometry follows.** If a sharper posterior (a tempered discount, or a floor on u) raises the inner reading's function and its agreement together along the ladder, the functional account holds further; if agreement lags, there is a trace after all. [N15 and N16, 2026-10-22: in N15's calibration the tempered self-judged discounts did not train at all, and N15 was withdrawn before any real run; N16 found the remaining shortfall in who judges, not in how sharply (`N16_NOTE.md`).]
2. **Why head A's cost is not monotone in head B's clean targets.** A dose of the constant on the standard head (N13's next conjecture 2) and a ladder of clean fractions with the standard loss scale would separate the pull's magnitude from what the pull is toward.
3. **What the discount needs.** A floor on the discount (no target below 0.2) for the reweighted head, with and without random targets, would test the conjecture that the random targets keep the inner softmax honest about the frequencies. [N16, 2026-10-22, exploratory: adding the random targets improved the reweighted head by 0.40 nats when it judged itself and by 0.14 when head A judged, consistent with this conjecture; the floor itself is untested (`N16_EXPLORATORY.md`).]
4. **Return to N10's question with mixture heads** (carried from N12 and N13).
