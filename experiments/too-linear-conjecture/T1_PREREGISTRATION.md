# Preregistration T1: where does Model A's loss gap come from?

Date: 2026-10-04    Status: **approved by the author on 2026-10-04, final.** Nothing below had been run when this version was committed. Written while T0 was queued and before any T0 result existed; approved while T0's second run was in progress, before any T0 result existed. From here on, the only changes are dated entries in the deviation log.

Type: confirmatory for the four contrasts below and their decision rules. Everything else is secondary or exploratory and labelled so.

Parent documents: `CONJECTURE.md` (test T1) and `T0_PREREGISTRATION.md`, both in this folder. Whatever T1 shows, the refutations in `experiments/wikitext2-replication/RESULTS.md` and `experiments/wikitext103-training/RESULTS.md` stand. Those studies tested the paper's model as built and trained. T1 builds new models in order to explain a result, not to revisit it.

## The question

On WikiText-103, Model A ended 1.10 nats worse than the 6-layer baseline on test loss, and about 0.73 nats worse at 10% of the training budget (stage 1). Model A differs from the baseline in four ways at once:

- it has fewer separate weights;
- it applies one set of weights several times;
- it uses the ODE form (α scaling, time embedding, steps of size α/4);
- it was trained with adjoint gradients that may not be true gradients.

T1 changes one of these at a time, in a ladder of five models, to find which differences carry the gap.

## The ladder

All five models share the paper's embedding, width 256, 4 heads, sequence length 128, dropout 0.1, untied output head and layers 0–1 and 4–5. Only what replaces layers 2–3 differs.

| Arm | Layers 2–3 replaced by | What it adds relative to the arm above |
|---|---|---|
| B6 | Two standard blocks, separate weights (the baseline) | — |
| B5 | One standard block, applied once | Removes one block's weights and its compute |
| T4 | One standard block, applied 4 times, with ordinary residual steps | Weight-tied depth: B5's weights with four applications |
| AX | Model A's ODE block, trained with exact gradients | The ODE form: α, the time embedding, the parallel attention-and-MLP update, steps of size α/4 |
| AA | Model A's ODE block, trained with adjoint gradients, exactly as in the WikiText-103 study | The adjoint gradient |

A "standard block" is the baseline's own pre-norm `TransformerBlock`. B5 and T4 have the same number of separate weights. AX and AA are the same model, with the same weights at every step for a given seed and the same forward pass. Only the backward pass differs: exact backpropagation through the four Euler steps (`torchdiffeq.odeint`) for AX, `odeint_adjoint` for AA.

## The four contrasts and what each tests

Each contrast is a paired difference in validation loss, the second arm minus the first, by seed. Positive means the second arm is worse.

| Contrast | Tests | From |
|---|---|---|
| **G: AA − AX** (primary) | Does the adjoint gradient cost loss? | Rival G; T0 tests the mechanism, this tests the consequence |
| **P: B5 − B6** | Does losing one block's weights cost about as much as Model A loses? | Rival P |
| **C2: T4 − B6** | Does weight-tied depth with the baseline's block already match the baseline? | Conjecture C2's forbidden case |
| **F: AX − T4** | Does the ODE form cost loss beyond weight tying, with exact gradients? | Separates the ODE form from weight tying |

## Predictions

These are the predictions of the conjecture note's explanations, stated in advance. They are not hopes, and every outcome is reported.

- **G:** if T0's mechanism matters for training, AA is worse than AX by more than 0.05 nats.
- **P:** rival P predicts B5 − B6 is at least half of AA − B6. The conjecture note expects much less.
- **C2:** conjecture C2 predicts T4 is worse than B6 by more than 0.05 nats. If T4 matches B6 within 0.05, C2's forbidden case has happened, and C2 is refuted at this scale.
- **F:** no directional prediction. This contrast measures.

## Decision rules

The four contrasts form one confirmatory family. Each uses a Bonferroni-corrected t-interval of the mean paired difference: a 98.75% interval with 4 degrees of freedom, critical value 4.315.

The margin of practical importance is 0.05 nats. It was chosen before any T1 run, as about 7% of the 0.73-nat gap at this budget, and well above the 0.008-nat spread seen in stage 1.

| Outcome for each contrast | Condition |
|---|---|
| Material: the second arm is worse | The interval lies wholly above +0.05 |
| Material: the second arm is better | The interval lies wholly below −0.05 |
| Negligible | The interval lies wholly within ±0.05 |
| Some effect, size unclear | The interval excludes 0 but meets none of the rules above |
| Inconclusive | Everything else |

**Rival P has an extra rule.** P is supported if the lower end of the interval for B5 − B6 is at least half the mean of AA − B6. It is refuted if the upper end is below half that mean. Otherwise it is inconclusive.

**Sensitivity.** If a contrast's standard deviation exceeds 0.04 nats, the design can't detect 0.05 reliably (the detectable effect is about 0.094 at that spread). The report then says so, and "negligible" is not claimed for that contrast even if its interval happens to fit.

## Design

### Budget and schedule

Each run is 2,879 steps, 10% of the full WikiText-103 budget and the same length as stage 1's sweep runs. Everything else follows the WikiText-103 study unchanged: batch 32 × 128 tokens, AdamW, warm-up over 500 steps then cosine decay to 10% of the peak, gradient clipping at 1.0, fp16 mixed precision with loss scaling, and the same token files and fingerprints.

### Tuning, the same procedure for every arm

Each arm's peak learning rate is chosen from the stage 1 grid, 3e-4, 6e-4, 1.2e-3 and 2.4e-3, by validation loss at seed 100. B6 and AA reuse their stage 1 sweep results, which used exactly this procedure: 2.4e-3 and 6e-4. B5, T4 and AX each get four new sweep runs. As in stage 1, a choice at the edge of the grid is kept, the grid is not extended, and the edge is reported as a limitation.

### Main runs and seeds

Each arm gets 5 seeds, 2001 to 2005, at its chosen rate. These seeds have not been used before. A seed fixes the data order, so arms with the same seed see the same batches. The initial weights also come from the seed, but differ between architectures. That pairs the contrasts by data order and leaves initialisation as noise, which the controls below measure.

### Primary outcome

Validation loss after 2,879 steps on the whole WikiText-103 validation split, in fp32. The test split is not used. It was used once, for the study's final evaluation, and T1 is diagnostic.

### Count of runs and cost

12 sweep runs, 25 main runs and 10 control runs, 47 short runs in all. At stage 1's measured 0.07 hours per baseline-type run and 0.11 hours per Model A run, that is about 4 hours on a T4. One Colab job, resumable across sessions like the study, with its own Drive folder.

## Controls

### Must match

1. **Same model, two backward passes.** For each seed, AX and AA start with identical weights. On the first batch they give the same loss, relative difference at most 1e-6, and identical gradients for every parameter after the ODE block, cosine at least 0.99999. This is checked before training.
2. **Construction.** The parameter counts follow from the definitions: B5 equals B6 minus one block, T4 equals B5, and AX equals AA. This is checked before training.
3. **A null contrast.** B6 is run again with seeds 2001 to 2005, with a different initialisation (seed + 50) but the same data order. The paired difference between the two B6 runs must be negligible by the rule above. If it isn't, initialisation noise is too large for the 0.05 margin, and every contrast is reported as inconclusive.

### Injected effect

B6 is trained with seeds 2001 to 2005 for 80% of the budget, 2,303 steps, with the same schedule compressed to that length. Fewer steps must make it worse, and the paired analysis must detect that with an interval wholly above 0. If it doesn't, the analysis can't see a known effect, and the confirmatory results are reported as unreliable.

## Secondary and exploratory outcomes

- **S1:** each arm's validation-loss curve, recorded at step 500, then every 400 steps, then at the end.
- **S2:** the paired gap AA − AX over training, to see when the two runs diverge.
- **S3:** α during training for AX and AA, every 50 steps.
- **S4:** training time per run per arm.
- **S5, exploratory:** the remaining contrasts, B5 − T4 and AA − B6, with unadjusted 95% intervals.

## Sanity checks before trusting results

- Every data file's fingerprint matches the manifest.
- The first batch of seed 2001 is decoded to text and inspected.
- AX's ODE block matches the paper's model in code: AX and AA build the same `HybridODETransformer`, and AX only swaps `odeint_adjoint` for `odeint` inside `ODEFlowBlock.forward`, through a flag. Must-match 1 checks the result.
- Every number in the report is computed by code from the runs' saved JSON records, and `t1_analyze.py` is written and committed before any T1 main run.

## Stopping rule

Exactly these 47 runs, each at most once. A crashed run resumes from its checkpoint, using the study's resumable training code. No arm, seed or contrast is added after seeing results except as a dated, exploratory entry in the deviation log.

## Relation to T0

T1's design does not depend on T0's result. AX is run whatever T0 shows. This supersedes the line in T0's outcome table that would have dropped the exact-gradient rung if T0 refuted a material defect. Running it anyway costs about 0.9 hours and checks T0's inference directly.

| T0 result | What T1's contrast G then means |
|---|---|
| T0 supported (biased gradients) and G material | The bias cost loss. Model A trained with exact gradients is the version worth studying further. |
| T0 supported, G negligible | The gradients were biased, but training tolerated it at this budget. |
| T0 refuted, G material | The gradients matched at the end of training but not along the way, or some other difference between the two backward passes matters. The contrast is reported as is, and the T0 limitation about measuring only the start and end becomes central. |
| T0 refuted, G negligible | Rival G is ruled out on both counts. |

## What each outcome would teach

| Outcome | What we would learn | Next |
|---|---|---|
| G material | Training Model A with adjoint gradients costs loss. Part of the WikiText-103 gap was the training method. | Retrain Model A with exact gradients at the full budget as a new model under its own preregistration, and compare it with the baseline. |
| P supported | Model A's gap is mostly missing weights. Its structure costs little. | Compare at matched separate weights before any redesign. |
| C2: T4 within ±0.05 of B6 | Weight-tied depth is not the problem: C2's forbidden case happened, and C2 is refuted at this scale. | Look at the ODE form (contrast F) and the gradient (G). |
| C2: T4 materially worse than B6 | Repeating one block costs loss even without any ODE machinery. C2 survives its test. | T3, depth-dependent computation such as a router or per-step modulation. |
| F material | The ODE parametrisation itself costs loss beyond weight tying. | Find which part: α, the time embedding, or the parallel update. Each is a single-change test. |
| F negligible | The ODE form costs nothing beyond weight tying here. | The gap is accounted for by P, C2 and G. |
| The null contrast fails | Initialisation noise is too large for a 0.05 margin with 5 seeds. | More seeds, preregistered as a new run, before any conclusion. |

## Limitations known now

- 10% of the training budget, not the full budget. The gap at 10% (about 0.73 nats) is smaller than at the full budget (1.10 nats), so the ladder explains the early gap, which may differ from the late one.
- Learning rates come from one seed per grid point, and the baselines may sit at the edge of the grid.
- One model size and one dataset.
- The ladder's order matters. Other orders, such as weight tying before removing a block, are not tested.

## Deviation log

### Implementation details fixed before any real run, 2026-10-04

Results seen before this entry: only test-mode runs of `t1_ladder.py` on tiny models (width 32, 60 steps), built to test the code. No real T1 run had started, and no T0 result existed.

- **How AX differs from AA in code.** AX replaces `ODEFlowBlock.forward` on the model instance with a version that calls `torchdiffeq.odeint`, not `odeint_adjoint`. The plan described this as a flag. The model class, its weights and the forward computation are unchanged, and must-match check 1 confirms it before training.
- **The study's training function gained two options,** with its defaults unchanged. `train_run` accepts `init_seed`, used only by the null contrast, and logs α for any model with an ODE block, not only the one named "hybrid". The finished WikiText-103 runs are unaffected.
- **Validation schedule.** The study's rule records validation loss at step 500 and at every multiple of the interval. With an interval of 400, that is steps 400, 500, 800, 1,200 and so on, then the end.
- **Learning rates for B6 and AA** are recomputed from stage 1's sweep records in `experiments/wikitext103-training/results/`, by the same lowest-validation-loss rule. They come to 2.4e-3 and 6e-4, as stage 1 recorded.
- **Storage.** T1 reads token files from the study's Drive folder without writing to it, and keeps its own checkpoints and results in `cdt-colab/t1_ladder`.
