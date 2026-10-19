# Preregistration N11: a payload strong enough to test, and a yardstick it cannot move

Date: 2026-10-18    Status: **approved by the author, 2026-10-18, with amendment A1 (the functional payload trains head B's readout only), approved 2026-10-19.** Queued; no real N11 run has started.

Type: confirmatory for N11a (primary), N11b and N11c, each with its own decision rule, on new models trained for this test. Everything else is secondary or exploratory and labelled so.

Parent documents: `N10_NOTE.md` (its explanation of why the payload did not take hold, and next conjecture 1), `N10_EXPLORATORY.md`, and `CRITICAL_REVIEW.md` section 7.

**Relation to earlier results.**
- **What N10 left open.** N10 asked whether anchoring what a head *predicts* preserves its reading under heavy label noise better than pinning the *directions* it reads. A principle Deutsch argues for, that information is independent of its substrate, predicts that it should. N10 could not answer, for two reasons.
  1. **Its functional payload did not take hold.** It was applied to 4 windows per step against 32 in the training batch, at λ = 1. It therefore carried about an eighth of the weight of the noisy objective it was meant to counter. It reduced its own windows' divergence only to 0.73–0.77 of the unanchored value, and the preregistered check correctly made the claims untestable.
  2. **Its function measure had a moving yardstick.** G compared head B with the ideal noisy copy of the *same model's* head A, and the payloads changed head A. Pinning the geometry made head A worse by 0.13 nats, which made G look better than head B was.
- **What N10 found, exploratory.** Against a fixed yardstick, the weak functional payload closed 20% of the gap and the geometric pin 16%.
- **What N11 does.** It repeats the payload test with two changes:
  - the functional payload is given real weight, at three strengths, and its strength is checked at full size before queuing;
  - function is measured against a yardstick no payload can move.
- **What stands regardless.** Every earlier verdict stands, including N10's untestable ones. N11 trains new models on new seeds, and measures them on new windows.

## The measure: the share of the gap closed, against a fixed yardstick

For each seed, the *unanchored* model (arm `n80`, 80% noise, no payload) defines the yardstick: the ideal noisy copy of its head A, q = 0.2·p_A + 0.8·u, where u is the training-set word frequencies.

- **Gap.** gap(arm) = head B's loss on the clean next-token targets in that arm, minus the yardstick copy's loss on the same targets. Both are measured on the same windows; the yardstick comes from the same seed's `n80` model.
- **Share closed.** S(arm) = 1 − gap(arm) / gap(`n80`), per seed.
  - S = 0: the payload did nothing.
  - S = 1: head B carries as much next-token information as the ideal copy of an unanchored head A.
  - Below 0: the payload made head B worse.

No payload can change the yardstick, because it comes from a different model, trained identically except for the payload.

The geometric measure, N9's agreement ratio of head B with its own model's ideal copy, is still recorded, as a secondary outcome. A geometric comparison cannot use another model's copy, because two models' state coordinates are not comparable.

## The arms

All are N7's and N10's model and training, at 10% of the budget. Head B's targets are 80% replaced by draws from u in every arm except `clean`.

| Arm | Head B's targets | Payload | Payload weight relative to the training batch |
|---|---|---|---|
| `clean` | The next token (ρ = 0) | None | — (must-match control) |
| `n80` | 80% replaced | None | — (the yardstick, and the unanchored comparison) |
| `n80-f8` | 80% replaced | Functional, λ = 8 | 1× |
| `n80-f32` | 80% replaced | Functional, λ = 32 | 4× |
| `n80-f128` | 80% replaced | Functional, λ = 128 | 16× |
| `n80-geom` | 80% replaced | Geometric, λ = 10 | (pins the readout, as in N10) |

**Details, fixed now, and as in N10 unless stated:**
- **The payload windows:** the same 64 training windows (seed 4242), 4 per step.
- **The functional payload:** KL(copy_ρ(A) ‖ head B) over those windows' positions, with the copy computed from the same model's head A and not differentiated through.
- **The geometric payload:** ‖P_B − P_A‖² / ‖P_A‖², with P_A not differentiated through.
- **The loss:** ½ (head A's loss + head B's loss) + ½ λ · payload term.
- **The weights:** λ = 8 makes the payload's weight per window equal to the batch's (4 windows against 32). λ = 32 and 128 give it 4 and 16 times that.

## The claims

**N11a (primary): a functional payload of sufficient strength preserves the reading's function.** In arm `n80-f128`, the 97.5% interval for S lies wholly above 0.67. That means two thirds of the gap closed, against 0.20 for N10's weak payload.

**N11b: the effect grows with the strength.** S(`n80-f8`) < S(`n80-f32`) < S(`n80-f128`) in at least 4 of 5 seeds.

**N11c: function, not geometry.** Each payload at a strength where it fully takes hold of its own target, the functional one preserves function better: S(`n80-f128`) − S(`n80-geom`) ≥ 0.30, paired by seed. The geometric pin takes hold fully at λ = 10: its readout overlap was 0.998 in N10.

### The explanation behind it

- **Why a strong functional payload should work.** The trunk is shared, and head A is trained on clean targets, so the state already carries the next-token information. Head B fails only to read it. A functional payload gives head B full distributions to match, not single labels, on 8,192 positions. That teaches it how to read the information, not which answers to give. If it does, the effect should carry over to text the payload never saw.
- **Why the geometric pin should not do as well.** It fixes which 64 directions head B reads, but not what head B does with them. In N10 it raised the geometric agreement from 0.31 to 0.70 but closed only 16% of the functional gap.
- **What this tests.** If information is independent of its substrate, anchoring the information (what is predicted) preserves it, and pinning the substrate (the directions) does not.
- **The rival that could make N11a succeed for the wrong reason: memorisation.** In N10 the data payload made the network memorise its 64 windows. A strong functional payload could do the same, matching the copy on those windows without reading the information in general. S is measured on 421 test windows the payload never sees, so memorisation alone would satisfy the manipulation check but not N11a.

## Competing hypotheses

| Hypothesis | S at λ = 128 (N11a) | Dose (N11b) | S(f128) − S(geom) (N11c) |
|---|---|---|---|
| The claims: anchoring function preserves it; pinning geometry does not | ≥ 0.67 | rising | ≥ 0.30 |
| Memorisation: a strong payload fits its windows but does not carry over | low, despite the manipulation check passing | flat | ≈ 0 |
| Strength alone: any strong anchor helps, of either kind | high | rising | ≈ 0, if a geometric pin at full strength helps as much |
| No small payload can counter 80% noise at this size | low | flat | ≈ 0 |

## Refuting results

- **N11a** is refuted if the 97.5% interval for S(`n80-f128`) lies wholly below 0.67, with a standard deviation of at most 0.05.
- **N11b** is refuted if S(`n80-f128`) ≤ S(`n80-f8`) in at least 4 of 5 seeds.
- **N11c** is refuted if the 97.5% interval for S(`n80-f128`) − S(`n80-geom`) lies wholly below 0.30, with a standard deviation of at most 0.05.

## Design

- **Seeds:** 4001–4005, new. All six arms of a seed share the initial weights and the order of the training data.
- **Runs:** 30.
- **Measurement windows:** WikiText-103 test-split windows 1793 to 2213, the last 421, with 8 extra tokens each. No measurement has used them: earlier tests used windows 1–1792.

**Measured per model, as in N10, plus:**
1. Head B's loss on the clean targets, for S.
2. For the `n80` model of each seed, the yardstick: its head A's ideal copy's loss on the clean targets.
3. For every model, head A's loss, to record what each payload costs head A.

## Decision rules

**Preconditions:**
1. **Setup.**
   - The batch, same-model, identical-heads and analytic checks pass.
   - The windows are shifted correctly.
   - Each noisy arm's replaced fraction at step 0 is within 0.05 of 0.8.
   - All six arms of a seed see the same batches.
   - The payload windows' targets are shifted correctly.
2. **Must match.**
   - In arm `clean`, head B's ratio with head A is at least 0.95, and |head B's clean loss − head A's| ≤ 0.05, each in at least 4 of 5 models.
   - The zero-noise copy matches head A at a ratio of at least 0.999 in every model.
3. **Must differ.** The rotated-readout control's ratio is at most 0.60 in at least 4 of 5 models of each arm.
4. **The yardstick leaves a gap to close.** gap(`n80`) ≥ 0.30 nats in every seed. In N10 it was 0.42–0.43.
5. **The payloads took hold of their own targets.**
   - *Functional:* in `n80-f128`, the payload-window KL(head B ‖ copy) is at most half of the same seed's `n80`, in at least 4 of 5 seeds.
   - *Geometric:* in every `n80-geom` model, the readout overlap of P_A and P_B is at least 0.90.
6. **Stable subspaces, for the secondary geometric outcomes only.** Head B and the copies have a split-half stability of at least 0.80.

Preconditions 1–4 apply to every claim. The functional part of 5 applies to N11a and N11c, and the geometric part to N11c. Precondition 6 applies to no claim.

**N11a:**

| Outcome | Condition |
|---|---|
| Supported | The 97.5% interval for S(`n80-f128`) lies wholly above 0.67 |
| Refuted | It lies wholly below 0.67, with a standard deviation of at most 0.05 |
| Some effect, size unclear | It excludes 0 but meets neither rule |
| Inconclusive | Everything else |

**N11b:**

| Outcome | Condition |
|---|---|
| Supported | S(f8) < S(f32) < S(f128) in at least 4 of 5 seeds |
| Refuted | S(f128) ≤ S(f8) in at least 4 of 5 seeds |
| Partial | Anything else |

**N11c:**

| Outcome | Condition |
|---|---|
| Supported | The 97.5% interval for S(f128) − S(geom) lies wholly above 0.30 |
| Refuted | It lies wholly below 0.30, with a standard deviation of at most 0.05 |
| Some effect, size unclear | It excludes 0 but meets neither rule |
| Inconclusive | Everything else |

**Intervals.** N11a and N11c are interval rules, so each uses a Bonferroni-corrected 97.5% t-interval with 4 degrees of freedom. N11b is a count rule.

**Why these thresholds.**
- **0.67** asks the payload to close two thirds of the gap. N10's weak payload closed 0.20.
- **0.30** for N11c is about twice the share N10's geometric pin closed (0.16).
- **The standard-deviation rule of 0.05:** in N10, S varied across seeds with a standard deviation of about 0.02.

## Secondary and exploratory outcomes

- **S1: does the geometry follow the function?** Head B's agreement ratio with its own model's copy, by arm. The target-free Fisher depends only on how head B's predictions change with the state. So if a functional payload makes head B compute the ideal copy, the geometric agreement should follow, without the readout being pinned.
- **S2: the cost to head A** of each payload, as head A's loss minus the same seed's `n80` head A.
- **S3: memorisation.**
  - Head B's gap on the 64 payload windows against its gap on the measurement windows, at each strength.
  - A payload that fits its own windows far better than new text is memorising.
- **S4: the dose curve,** S at λ = 8, 32 and 128, and the payload-window divergence at each.
- **S5: replication,** `n80` against N10's `n80`: the agreement ratio, gap(`n80`), and head A's loss.

## Calibration before queuing: a dose check at full size

This is the step N10 lacked.

- **Before queuing,** each functional strength, and the unanchored arm, is trained at full size for 300 steps at the peak learning rate, on the full training text, for one seed (G1's seed 3001). The payload-window divergence is recorded.
- **Check:** at λ = 128, the payload-window divergence must be at most 0.5 of the unanchored arm's by step 300.
  - If it is, the strengths stand.
  - If it is not, an amendment comes back for approval first, with higher strengths or the payload windows spread through the batch.
  - The geometric pin was checked at full size before N10, and held at 0.999.
- **Also:**
  - the untrained-model checks as in N10: the analytic check, identical heads, must-match and must-differ controls;
  - the tiny end-to-end run, and the resume test;
  - synthetic records for every branch of the rules.

## Sanity checks before trusting results

- Every data file's fingerprint matches the manifest.
- Each arm's noise fraction, its payload weight and the yardstick for each seed are logged.
- Every number in the report is computed by code from saved JSON records. `n11_analyze.py` is written and committed before any real run, and tested on synthetic records covering every branch of the rules.

## Compute, where it runs, and stopping rule

- **Where:** one Colab job, `n11_strength`, on a T4, resumable.
- **Cost:** 30 training runs at about 7 to 8 minutes each, plus measurement: about 4 hours.
- **Stopping rule:** exactly these runs and measurements, each once. Nothing is added after seeing results except as a dated, exploratory entry in the deviation log.

## What each outcome would teach

| N11a | N11b | N11c | What we would learn | Next |
|---|---|---|---|---|
| Supported | Supported | Supported | Anchoring what a head predicts, on a handful of passages, preserves its reading against heavy noise, and pinning the directions does not. The substrate-independence prediction survives a test it could have failed. | Shrink the payload to see how few windows suffice. Check whether the geometry follows (S1) |
| Supported | Any | Refuted | A strong functional payload preserves function, but a geometric pin preserves it about as well. Anchoring the substrate is as good as anchoring the information here | Look for a case where many geometries carry the same function, such as N5's full width |
| Refuted, manipulation check passed | Any | Any | The payload fits its own windows but does not carry over: memorisation (S3 should show it). Anchoring function on a few passages does not preserve a reading at this size | Spread the payload over many more passages at lower strength |
| Untestable (manipulation failed) | Any | Any | Even 16× weight cannot fit 64 windows against 80% noise. Possible only if the dose check misled again | Rethink the payload's form |

## Limitations known now

- **One noise rate (80%), one payload size (64 windows), one geometric strength.**
- **The functional payload's reference is the co-trained head A,** not detached from training altogether. The yardstick avoids the measurement problem this caused, but not any effect on training itself.
- **The yardstick is a sibling model:** the same seed, data and training, differing only by the payload. Head A in that model is not identical to head A in the payload models, so S compares head B with a reference, not with its own head A.
- **421 measurement windows,** fewer than N10's 512, because the test split is nearly used up. Precondition 6 checks stability for the secondary geometric outcomes.
- **The same limits as before:** 10% of the budget, one learning rate, one size, one dataset, one readout width.

## Deviation log

### Other calibration, before any real run, 2026-10-19 (no amendment)

- **Untrained full-size models** (G1's seeds 3001 and 3002), on N11's 421 windows:
  - the analytic check scored 0.994, and identical heads 1.0;
  - the rotated control scored 0.078 and 0.077;
  - head A against its own ideal copy scored 0.990 and 1.009;
  - every head and copy had a stability of at least 0.86, and the copy at ρ = 0.8 had 0.86–0.88.

  The smaller window count therefore still gives stable subspaces, at least before training.
- **Tiny test-mode run with amendment A1's code.**
  - All 12 models trained and were measured, and a rerun skipped all finished work.
  - The payload-window divergence fell with strength: 1.78 unanchored, then 1.39 (λ = 8), 1.36 (λ = 32) and 1.35 (λ = 128).
  - Head A's loss was the same in every arm, 9.73 to 9.74, so the payload does not reach head A.
  - The tiny models are too small to pass the must-match control, as in N10.
- **Synthetic records.** 17 cases covering every branch of the three rules and every precondition gave the intended verdicts.

### Amendment A1, before any real run, 2026-10-19: approved by the author

Results seen before this entry:
- the preregistered dose check at full size (seed 3001, 300 steps at the peak learning rate, the full training text);
- two runs of a candidate fix under the same conditions;
- synthetic records.

No real N11 run has started.

**What the dose check found.** The table gives the payload-window KL(head B ‖ copy) as a ratio of the unanchored arm's at the same step, at steps 100 / 200 / 300. The last column is the clean-target loss, on the payload windows, of head A's ideal copy at step 300. It shows what the payload did to head A; unanchored, it is 6.89.

| Arm, as written | Ratio at 100 / 200 / 300 | Head A's copy at 300 |
|---|---|---|
| λ = 8 | 0.22 / 0.37 / 0.42 | 7.13 |
| λ = 32 | 0.33 / 0.13 / 0.37 | 7.29 |
| λ = 128 | 0.02 / 1.03 / **0.98** | **7.44** |

- **λ = 128 fails the check as written.** It held its windows at step 100, then lost them. Worse, it damaged head A, the reference it was teaching head B to copy.
- **The cause is in the implementation, not the idea.**
  - The payload's gradient ran through the trunk that both heads share.
  - All gradients were clipped together to a total norm of 1.
  - So a heavy payload crowded out head A's learning, and the copy it was matching degraded.
- **A first candidate fix failed in a different way.** It detached the trunk from the payload and clipped the payload's gradient separately. That kept head A intact, but it erased the strengths: at step 100, λ = 32 and λ = 128 gave the same divergence (0.0219 and 0.0218), because a clipped gradient no longer scales with λ.

**The amendment.**
- *What it changes.* The functional payload trains only head B's readout: P_B and W_B, its 64 × 256 projection and output matrix.
  - The shared state and head A's copy are computed on the payload windows without gradient, so the payload cannot reach the trunk or head A.
  - The payload's gradient is added to head B's readout after the main loss's gradient has been clipped, and is not clipped itself, so λ still sets its weight. Adam bounds the size of each update.
  - Code: `n10_payload.readout_payload_grads`, used by the arms of kind `func-readout`.
- *Its result in the same dose check:*

  | Arm, amended | Ratio at 100 / 200 / 300 | Head A's copy at 300 |
  |---|---|---|
  | λ = 8 | 0.23 / 0.39 / 0.40 | 6.89 |
  | λ = 128 | 0.15 / 0.23 / **0.27** | 6.90 |

  - λ = 128 passes the check (0.27 ≤ 0.5) and holds at every checkpoint.
  - The strengths now differ: 0.40 against 0.27.
  - Head A is undamaged at both.
- *What it does not change:* the claims, thresholds, arms, strengths, yardstick and every other rule. The geometric payload already acted only on head B's readout projection, so it is unchanged.

**What it means for the claims.**
- **It matches the claim's own explanation better than the original did.** The preregistration's explanation says that the state already carries the next-token information, and head B must only learn to read it. The amended payload tests exactly that.
- **It narrows the claim.** N11 now tests whether anchoring what head B's *readout* computes preserves the reading. It no longer tests a payload that could also reshape the shared network.
- **N10's functional payload acted through the trunk as well.** So N11 is not a pure repeat of N10's payload at higher strength. That is recorded here so that the two are not read as one design.

## References

Deutsch, D. (1985). Quantum theory, the Church–Turing principle and the universal quantum computer. *Proceedings of the Royal Society of London A, 400*(1818), 97–117. https://doi.org/10.1098/rspa.1985.0070

Son, C. (2026). *J-space forecasts model collapse, but only anchoring the function prevents it* [Preprint]. Research Square. https://doi.org/10.21203/rs.3.rs-10713541/v1
