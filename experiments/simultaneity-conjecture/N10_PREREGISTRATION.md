# Preregistration N10: how much noise changes what a head learns, and which payload preserves it

Date: 2026-10-15    Status: **approved by the author, 2026-10-18.** Run; see the deviation log, `N10_RESULTS.md`, `N10_EXPLORATORY.md` and `N10_NOTE.md`.

Type: confirmatory for N10a (primary), N10b, N10c and N10d, each with its own decision rule, on new models trained for this test. Everything else is secondary or exploratory and labelled so.

Parent documents: `N9_NOTE.md` (next conjecture 1), `CRITICAL_REVIEW.md`, and the discussion of 2026-10-14 and 2026-10-15 on payloads and substrate independence.

**Relation to earlier results.**
- **What N9 found.** A head trained on next-token targets with 80% replaced by words drawn at their normal frequencies did not learn the ideal noisy copy of the clean reading.
  - Its sensitive directions overlapped the copy's at an agreement ratio of only 0.31–0.35. The measure recognises the copy itself at 0.96–1.00.
  - Its predictions were worse on the clean targets than the copy's, by 0.61 nats on average (0.58 to 0.62 across seeds, standard deviation 0.018).
  - So noisy training changed what the head reads, and weakened it.
- **Two questions follow.**
  1. *How much noise does that take?* Only 80% has been tested.
  2. *Can a small "payload" preserve the reading under heavy noise, and of what kind must the payload be?*
- **The second question tests a principle.** Deutsch argues that information is independent of its substrate: the same knowledge can be held in many physical forms, and a system works only as well as it preserves the information as it moves between forms. If so, a payload that preserves the *function*, what the head predicts, should protect the reading. A payload that pins a particular *geometry*, the directions it reads, should not, because many geometries can carry the same function and the rest of the network can change around a pinned one.
- **One study points the same way.** It is a 2026 preprint on a single 70-million-parameter model trained on WikiText-103, so it is suggestive at most. When a model collapsed under training on its own outputs, pinning its representation geometry prevented about 12% of the damage. A penalty keeping its predictions close to the old model's, on 64 real passages, prevented 99.6% (Son, 2026).
- **What stands regardless.** Every earlier verdict stands. N10 trains new models on new seeds and measures them on new windows.

## The measures

Two measures of how close the trained head B is to the *ideal noisy copy* of head A, the head whose prediction is exactly (1 − ρ)·p_A + ρ·u (N9). Here p_A is head A's prediction, u the training-set word frequencies, and ρ the arm's noise rate.

- **Geometric: the agreement ratio.** N9's target-free measure, unchanged. It compares the directions of the state that each head's predictions are sensitive to. 1 means the same directions; random directions give about 0.06.
- **Functional: the gap G.** G = head B's loss on the clean next-token targets minus the ideal copy's loss on the same targets, in nats, on the measurement windows.
  - 0 means head B carries as much next-token information as the ideal copy.
  - In N9, at 80% noise, G was 0.61.
  - G looks only at what head B predicts, not at the directions it reads, so it does not depend on the substrate.

## The arms

All are N7's model and training: the paper's HybridODETransformer with exact gradients, head A predicting the next token, two rank-64 readouts, 10% of the WikiText-103 budget. Only head B's training changes.

| Arm | Head B's targets | Payload |
|---|---|---|
| `clean` | The next token (ρ = 0) | None (must-match control) |
| `n20` | The next token, 20% replaced | None |
| `n50` | The next token, 50% replaced | None |
| `n80` | The next token, 80% replaced | None (replicates N7 on new seeds) |
| `n80-func` | As `n80` | **Functional payload:** on 64 fixed training windows, a penalty on the divergence between head B's prediction and head A's ideal noisy copy |
| `n80-geom` | As `n80` | **Geometric payload:** a penalty pinning head B's readout projection to head A's |
| `n80-data` | As `n80` | **Data payload:** on the same 64 windows, head B is also trained on the clean targets |

**Details, fixed now:**
- **Replaced targets:** each replaced target is drawn from u, the training data's unigram distribution, as in N7. The noise for each step comes from a generator seeded by the run's seed and the step.
- **The payload windows:** 64 windows of the training split, chosen once with a fixed seed (4242), the same for every seed and arm. Each training step uses 4 of them in turn, so all 64 are seen every 16 steps.
- **The loss:** total = ½ (head A's loss + head B's loss) + ½ λ · payload term.
  - *Functional:* the payload term is the mean KL divergence KL(copy_ρ(A) ‖ head B) over the positions of the step's 4 payload windows, with the copy computed from head A and not differentiated through (head A is not pulled towards head B). λ = 1.
  - *Data:* the payload term is head B's cross-entropy against the clean targets of the 4 payload windows. λ = 1.
  - *Geometric:* the payload term is ‖P_B − P_A‖² / ‖P_A‖², with P_A not differentiated through, applied every step. λ = 10. P_B and P_A are the two heads' 64 × 256 readout projections.
- **Why the functional payload is not circular.** It is applied only on 64 training windows, about 0.03% of the training text. G and the ratio are measured on 512 test windows the payload never sees. And the ratio measures sensitive directions, which the payload does not target.

## The claims

**N10a (primary): a small functional payload preserves the reading under heavy noise.** In arm `n80-func`, the 97.5% interval for G lies wholly below 0.20 nats, a third of N9's gap.

**N10b: function, not geometry.** The geometric payload preserves function less well than the functional one: G(`n80-geom`) − G(`n80-func`) ≥ 0.20 nats, paired by seed.

**N10c: light noise does not change what the head reads.** In arm `n20`, head B's agreement ratio with its ideal copy is at least 0.85 in at least 4 of 5 models.

**N10d: the change grows with the noise.** Head B's agreement ratio with its ideal copy falls in the order `n20` > `n50` > `n80` in at least 4 of 5 seeds.

### The explanation behind it

- **Why light noise should not matter.**
  - At ρ = 0.2, 80% of head B's targets are the clean next token, so its learning signal is mostly the clean one.
  - The ideal copy is then a mild blend, and the best way to fit the targets is to learn the clean reading and blend.
  - At ρ = 0.8, only a fifth of the signal carries next-token information, and the head can lower its loss a long way just by learning the word frequencies. That is consistent with N9, where it learned a weaker reading in different directions.
- **Why a functional payload should work and a geometric one should not.**
  - The functional payload tells head B what to predict, in full distributions, on a few passages. Because the trunk is shared, the next-token information is already in the state; head B must only learn to read it.
  - The geometric payload fixes which 64 directions head B reads, but not what it does with them. Head B could read head A's directions and still predict mostly frequencies.
  - If information is independent of its substrate, fixing the substrate leaves the information free to be lost.
- **Why the data payload is recorded but makes no claim.**
  - It is the analogue of mixing real data into training, which slows collapse in other work (Gerstgrasser et al., 2024). In Son's study it left at least 23 times more damage than the functional anchor.
  - Here it conflicts with head B's own task: it is trained towards the clean prediction on 64 windows and the noisy one elsewhere. Its result is exploratory.

## Competing hypotheses

| Hypothesis | G, functional payload (N10a) | G, geometric minus functional (N10b) | Ratio at 20% noise (N10c) |
|---|---|---|---|
| The claims: function must be anchored, and only heavy noise changes the reading | < 0.20 | ≥ 0.20 | ≥ 0.85 |
| Geometry is function here: pinning the directions also preserves what is read | < 0.20 | ≈ 0 | any |
| No small payload is enough: the noisy signal dominates whatever is added | ≈ 0.6 | ≈ 0 | any |
| Any noise changes the reading: the trouble is not the dose | any | any | ≤ 0.60 |

## Refuting results

- **N10a** is refuted if the 97.5% interval for G(`n80-func`) lies wholly above 0.20, with a standard deviation of at most 0.05.
- **N10b** is refuted if the 97.5% interval for G(`n80-geom`) − G(`n80-func`) lies wholly below 0.20, with a standard deviation of at most 0.05.
- **N10c** is refuted if head B's ratio with its ideal copy in arm `n20` is at most 0.60 in at least 4 of 5 models.
- **N10d** is refuted if the ratio in `n20` is at or below that in `n80` in at least 4 of 5 seeds.

## Design

- **Seeds:** 3901–3905, new. All seven arms of a seed share the initial weights and the order of the training data, as in N7.
- **Runs:** 35, each at N7's settings.
- **Measurement windows:** WikiText-103 test-split windows 1281 to 1792, with 8 extra tokens each. No measurement has used them: N6r used 1–256, N7 257–512, N8 513–768, N9 769–1280.

**Measured per model:**
1. The ideal copy of head A at the arm's own ρ (ρ = 0 for `clean`, 0.8 for the three payload arms).
2. The agreement ratio of head B with that copy, and of head A with it, using N9's code.
3. G, and the KL divergence from head B's prediction to the copy's, per position.
4. Head B's loss on its own noisy targets, corrupted once with a fixed measurement seed.
5. The controls below.

## Decision rules

**Preconditions:**
1. **Setup.**
   - The batch, same-model and identical-heads checks pass.
   - The test windows are shifted correctly.
   - Each noisy arm's replaced fraction at step 0 is within 0.05 of its ρ.
   - All seven arms of a seed see the same batches.
   - The 64 payload windows decode as text, and none of them is a measurement window.
2. **Must match.**
   - In arm `clean`, head B's ratio with its copy, which is head A itself, is at least 0.95 in at least 4 of 5 models. In the same arm, |G| ≤ 0.05 in at least 4 of 5 models.
   - In every model, the zero-noise copy matches head A at a ratio of at least 0.999.
3. **The measure recognises copies.** In every arm, head A's ratio with its own ideal copy is at least 0.85 in at least 4 of 5 models. This is N9a, replicated, and N10c and N10d rest on it.
4. **Must differ.** The rotated-readout control's ratio is at most 0.60 in at least 4 of 5 models of each arm.
5. **The payloads did what they were built to do.**
   - *Geometric:* in every `n80-geom` model, the row spaces of P_A and P_B overlap by at least 0.90.
   - *Functional:* in at least 4 of 5 seeds, the mean KL divergence from head B to the copy on the 64 payload windows is at most half of the same quantity in that seed's `n80` model.

   Without these, a null result would say only that the payload was too weak.
6. **Stable subspaces.** Head B and the copies have a split-half stability of at least 0.80 in every model used by N10c and N10d.

Preconditions 1, 2 and 4 apply to every claim. Precondition 3 applies to N10c and N10d. Precondition 5 applies to N10a and N10b: its geometric part to N10b, its functional part to both.

**N10a:**

| Outcome | Condition |
|---|---|
| Supported: the functional payload preserves the reading | The 97.5% interval for G(`n80-func`) lies wholly below 0.20 |
| Refuted | It lies wholly above 0.20, with a standard deviation of at most 0.05 |
| Some effect, size unclear | It meets neither rule, but the paired 97.5% interval for G(`n80`) − G(`n80-func`) excludes 0 |
| Inconclusive | Everything else |

**N10b:**

| Outcome | Condition |
|---|---|
| Supported: function, not geometry | The 97.5% interval for G(`n80-geom`) − G(`n80-func`) lies wholly above 0.20 |
| Refuted | It lies wholly below 0.20, with a standard deviation of at most 0.05 |
| Some effect, size unclear | It excludes 0 but meets neither rule |
| Inconclusive | Everything else |

**N10c:**

| Outcome | Condition |
|---|---|
| Supported | Ratio with the copy in `n20` ≥ 0.85 in at least 4 of 5 models |
| Refuted | ≤ 0.60 in at least 4 of 5 |
| Intermediate | Anything else |

**N10d:**

| Outcome | Condition |
|---|---|
| Supported | `n20` > `n50` > `n80` in at least 4 of 5 seeds |
| Refuted | `n20` ≤ `n80` in at least 4 of 5 seeds |
| Partial | Anything else |

**Intervals.** N10a and N10b are interval rules, so each uses a Bonferroni-corrected 97.5% t-interval with 4 degrees of freedom. N10c and N10d are count rules.

**Why these thresholds.**
- **0.20 nats** is a third of N9's gap. "Preserves" therefore means removing at least two thirds of the damage.
- **The standard-deviation rule of 0.05** is the one used in N7 and N8. N9's G varied across seeds with a standard deviation of 0.018.
- **0.85 and 0.60** are N7's and N9's thresholds for "converged" and "separate".
- **The manipulation checks** (0.90, and half the KL divergence) are set so that a payload counts as applied only if it clearly moved its own target.

## Secondary and exploratory outcomes

- **S1: the data payload.** G and the ratio in `n80-data`, against `n80-func` and `n80`. Son's study would predict data < unanchored, but data much worse than functional.
- **S2: does the geometric payload fool the geometric measure?** Head B's ratio with the copy in `n80-geom`. If it is high while G stays near 0.6, the agreement ratio can be satisfied by pinning the substrate without preserving the function. That would mean our geometric measure should not be used alone to judge whether a reading is preserved.
- **S3: the dose curves.** G and the ratio for `clean`, `n20`, `n50` and `n80`.
- **S4: replication.** `n80` against N7's noisy arm and N9's measurements: ratio with the copy (0.31–0.35) and G (0.61).
- **S5:** losses during training, readout drift as in N8, and head A's loss in every arm. The last checks that the payloads did not harm head A.

## Calibration before queuing

On untrained full-size models (G1's seeds 3001 and 3002), on N10's windows:
- the analytic check on mixtures, the identical-heads check, the must-match and must-differ controls;
- G and the ratios at each ρ, recorded and not used for any threshold.

From N9's records, which are already in hand: the seed-to-seed spread of G at 80% noise, 0.018, against the 0.05 rule.

On tiny test-mode runs:
- that the geometric payload pins P_B at the chosen λ;
- that the functional payload lowers the payload-window KL divergence;
- that the job resumes.

If a threshold, λ or control does not work as written, an amendment comes back for approval first.

## Sanity checks before trusting results

- Every data file's fingerprint matches the manifest.
- Each arm's noise fraction and the payload windows are logged.
- Every number in the report is computed by code from saved JSON records. `n10_analyze.py` is written and committed before any real run, and tested on synthetic records covering every branch of the rules.

## Compute, where it runs, and stopping rule

- **Where:** one Colab job, `n10_payload`, on a T4, resumable.
- **Cost:** 35 training runs at about 7 minutes each, with the payload arms about 12% longer, plus measurement: about 4.5 hours in all. If a session ends, choose Run all again, and finished runs are skipped.
- **Stopping rule:** exactly these runs and measurements, each once. Nothing is added after seeing results except as a dated, exploratory entry in the deviation log.

## What each outcome would teach

| N10a | N10b | N10c, N10d | What we would learn | Next |
|---|---|---|---|---|
| Supported | Supported | Supported | Heavy noise changes what a head reads, and a handful of functional examples prevents it while pinning the geometry does not. The substrate-independence prediction survives a test it could have failed. | Vary the payload's size, down to a few windows. Then test whether the same holds for readings that differ in kind |
| Supported | Refuted | Any | Pinning the geometry preserves the function as well as anchoring the function does. In this model the geometry *is* the function, and the principle gives no guidance at this scale | Test at a larger readout width, where many geometries can carry the same function (N5's width 256) |
| Refuted | Any | Any | Even a functional payload cannot hold the reading against 80% noise at this size | Larger payloads, or a lower noise rate |
| Any | Any | N10c refuted | Even light noise changes what is read. The trouble is noise itself, not its dose | Ask whether any objective that differs from the clean one does the same |
| Any | Any | N10c supported, N10d partial | Something other than the dose matters | Use S3 to locate where it changes |

## Limitations known now

- **One payload size and one strength for each payload:** 64 windows, λ = 1 or 10. A null result for a payload is about this size and strength only.
- **The functional payload uses the co-trained head A** as its reference, not a frozen earlier model as in collapse studies. That is closer to distillation from a sibling than to anchoring to a past self.
- **The noise is random replacement by word frequencies.** It is not model-generated text, so this is an analogue of collapse, not collapse.
- **Same limits as before:** 10% of the budget, one learning rate, one size, one dataset, one readout width.

## Deviation log

### The run, 2026-10-18

- **Session:** `colab-results` branch `n10_payload/20261018-191435`, code commit e2f9f13. It trained and measured all 35 models in one session, with no resumes and no Drive failures.
- **What changed after approval:** nothing.
- **Results, under the rules:**
  - N10a and N10b are untestable: precondition 5b failed. The functional payload's window KL divergence was 0.73–0.77 of the unanchored arm's, against at most 0.5 required.
  - N10c is intermediate.
  - N10d is supported.

  See `N10_RESULTS.md` and `N10_NOTE.md`.

### Exploratory analysis added after the results, 2026-10-18

- **What was added:** `n10_exploratory.py` (report `N10_EXPLORATORY.md`). It compares head B against a fixed yardstick, the ideal copy from the same seed's `n80` model, and reports what each payload cost head A.
- **Why:** the results showed that the payloads changed head A, and with it the copy G is measured against.
- **Its status:** written after the results were seen. It decides nothing, and the preregistered analysis is unchanged.

### Calibration and implementation details, before any real run, 2026-10-18 (no amendment)

Results seen before this entry:
- calibration on **untrained** full-size models (G1's seeds 3001 and 3002) on N10's windows;
- a short full-size stress check of the payloads;
- test-mode runs on tiny models;
- synthetic records.

No real N10 run has started.

- **Untrained models.**
  - *The estimator:* the analytic check scored 0.994, identical heads 1.0, and the rotated control 0.074 and 0.076.
  - *Recognising copies:* head A against its own ideal copy scored 0.98 to 1.01 at every ρ.
  - *Stability:* at least 0.89 for every head and copy, 0.89 to 0.90 for the copy at ρ = 0.8.
  - *Head B against the copies:* untrained heads score near 1 (0.96 to 1.00), as N8 found, because head B's readout starts as a copy of head A's. The must-match control therefore means something only in trained models.
- **The payloads at full size.** Each arm was trained for 150 steps at the peak learning rate, which is harder on the geometric pin than the real warm-up schedule, on the full training text, seed 3001.
  - *Geometric:* with λ = 10, P_A and P_B overlapped at 0.999, against 0.805 in the unanchored arm. Precondition 5a should be met.
  - *Functional:* with λ = 1, the payload-window KL divergence was 0.054, against 0.084 unanchored, a ratio of 0.64. Precondition 5b asks for at most 0.5 by the end of training, which is 19 times longer. The payload windows will then each have been seen about 180 times, so this looks likely to be met, but it is not certain. If it is not met, N10a and N10b are untestable, as the rules say.
  - *An early-training observation, no more:* G on held-out windows was 0.223 unanchored, 0.171 functional and 0.146 geometric. This is far from the end of training and decides nothing.
- **Tiny test-mode run.** All 14 models trained and were measured, and a rerun skipped all finished work. The geometric payload pinned P_B exactly (1.000) and raised head B's ratio with the copy to 0.99–1.00 while G stayed at 2.0–2.1 nats, a small-scale instance of what S2 asks about. The tiny models are too small to pass the must-match control.
- **Synthetic records.** 21 cases covering every branch of the four rules and every precondition gave the intended verdicts.
- **Clarifications of the written design, not changes:**
  - Precondition 5b's "KL divergence from head B to the copy" is computed as KL(head B ‖ copy). The training penalty is KL(copy ‖ head B), as written. Both are recorded.
  - The 64 payload windows are drawn from the training split, so they cannot be measurement windows, which come from the test split.

## References

Deutsch, D. (1985). Quantum theory, the Church–Turing principle and the universal quantum computer. *Proceedings of the Royal Society of London A, 400*(1818), 97–117. https://doi.org/10.1098/rspa.1985.0070

Gerstgrasser, M., Schaeffer, R., Dey, A., et al. (2024). *Is model collapse inevitable? Breaking the curse of recursion by accumulating real and synthetic data*. arXiv. https://doi.org/10.48550/arxiv.2404.01413

Son, C. (2026). *J-space forecasts model collapse, but only anchoring the function prevents it* [Preprint]. Research Square. https://doi.org/10.21203/rs.3.rs-10713541/v1
