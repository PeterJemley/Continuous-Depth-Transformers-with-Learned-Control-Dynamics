# Preregistration N1: is the reading a direction, not a regime?

Date: 2026-10-06    Status: **approved by the author on 2026-10-06, final.** Nothing below had been run when this version was committed. From here on, the only changes are dated entries in the deviation log.

Type: confirmatory for the primary outcome (N1b) and for the secondary hypothesis N1a, each with its own decision rule. Everything else is secondary or exploratory and labelled so.

Parent document: `PATHS_AND_UNDERSTANDING.md`, section "After G1d", conjecture N1.

**Relation to earlier results.**
- **What came before.** G1 found the trained control barely changes how trajectories turn. G1c found it changes how strongly they contract, by a factor of 4 to 10, replicated twice. G1d found that removing over 90% of that contraction difference costs nothing, while the models go on using the control fully.
- **What N1 asks.** Is there anything in the ODE block's *dynamics* that the reading needs at all? Or is an offset, a fixed vector per control value, enough?
- **What stands regardless.** All earlier results stand whatever N1 shows.

## The claims

**N1b (primary): an offset is enough.** A model in which the control value u never enters the vector field, and is only added as a learned vector to the ODE block's output, does both tasks as well as the model with u inside the field. Neither task's validation loss is materially worse, by 0.05 nats or more.

**N1a (secondary hypothesis): in the trained model, the control's effect is an offset.** In the model with u inside the field, the difference the control makes to the block's output, h_out(u = −1) − h_out(u = +1), is at least 80% one constant vector across all tokens and windows.

### The explanation behind it
- **The easy solution.** The control enters the vector field additively, through `control_embed(u)`. The easiest solution for training to find is a shift of the state along one direction per control value, which the later layers then read. The curvature and contraction differences found in G1 and G1c would then be side effects of where the shift moves the state relative to the LayerNorms, not computations the model relies on. That would fit G1d's result.
- **Prior work.** Adding a constant vector to a language model's internal state is known to steer its behaviour (Turner et al., 2023), and many concepts appear to be represented as directions (Park et al., 2023). So N1 is a live possibility, not a strawman.
- **The rival.** Conditioning that only shifts features is weaker than conditioning that also scales them (Perez et al., 2018). If the dynamics do something an offset cannot, the offset-only model will show it.

### Calibration: why N1a is the weaker test
Before this draft, the N1a measure was run on three **untrained** full-size models: G1's seeds 3001–3003 at initialisation, on the first 64 windows of the test-mode token file, which N1 will not use.
- **Already an offset.** At initialisation, the control's effect on the block's output is already 92% a single constant vector (0.916 to 0.923).
- **The measure is sensitive.** An injected control that *scales* the block's output, rather than shifting it, reads 0.015 to 0.020.
- **What follows.** A high N1a score in trained models would mean training *kept* the offset structure, not that it learned it. N1a can still fail, if training makes the effect depend on the state, but it cannot tell an inherited offset from a learned one. That is why the decisive test is the intervention, N1b.

## Competing hypotheses

| Hypothesis | Prediction |
|---|---|
| N1: the reading is an offset | The offset-after model matches the full model on both tasks; N1a at least 0.8 |
| The dynamics do work an offset cannot | The offset-after model is worse on at least one task by 0.05 nats or more; N1a may fall well below 0.8 |
| The tasks don't need the control at all | The no-control arm matches the full model too. The positive control fails, and N1 is untestable |
| An offset suffices, but training finds it less easily without the dynamics (a learnability difference) | Offset-after worse, *and* N1a high in the full model. The outcome table separates this case |

## Refuting result

N1b is refuted if the 97.5% interval for either task's loss difference (offset-after minus full) lies wholly above +0.05 nats.

## Design

### Arms

All arms share G1c's training exactly:
- the paper's `HybridODETransformer` with exact gradients;
- u = (±1, 0, 0, 0) telling the task per sequence;
- 2,879 steps at peak learning rate 2.4e-3, batch 32 × 128;
- the pinned WikiText-103 token files.

They differ only in *where* `control_embed(u)` enters. Every arm has the same modules and the same number of parameters.

| Arm | Where the control enters |
|---|---|
| **full** | Inside the vector field at every evaluation, as in G1, G1c and G1d |
| **post** | Nowhere inside the block. `control_embed(u)` is added once to the block's *output* |
| **pre** (secondary) | Nowhere inside the field. `control_embed(u)` is added once to the block's *input*, so the dynamics carry it but do not see u directly |
| **none** (positive control) | Nowhere. `control_embed` is never used |

- **Seeds:** 3301 to 3305 for every arm. These have not been used before. A seed fixes the initial weights, the data order and the tasks, so a seed's arms start from identical weights.
- **Runs:** 20 in all.

### Measuring

- **Losses.** Both tasks' validation losses on the whole validation split, in fp32, each with its right control value: the next-token task with u = +1 and the farther task with u = −1. The test split is not used.
- **N1a.** The offset fraction of the full models, on validation windows 769 to 1,024, which no earlier measurement used.
  - Δ is the difference h_out(u = −1) − h_out(u = +1), taken per token at the block's output. That is 256 windows × 128 tokens of 256-dimensional vectors.
  - The offset fraction is N‖mean Δ‖² / Σ‖Δ‖², over all N token positions. It is 1 if Δ is the same vector everywhere, and near 0 if Δ varies freely.
  - It is also measured on the same seeds' untrained initial weights, for reference.

## Primary outcome (N1b)

For each seed and each task, the loss of the post arm minus the loss of the full arm, on the whole validation split. These two contrasts form the family. Each uses a Bonferroni-corrected t-interval, 97.5% with 4 degrees of freedom (critical value 3.495), over the 5 seeds.

## Decision rules

**Preconditions.** If any fails, the result is "untestable", whatever the contrasts show.

1. **Must match, by construction.** For every seed, with `control_embed`'s weights and bias set to zero, all four arms produce bit-identical logits on one batch at initialisation. The arms then differ only in where the control's vector enters.
2. **Positive control: the tasks need the control.** For the farther task, the none arm's loss minus the full arm's loss has a 97.5% interval wholly above +0.05 nats. G1 and G1c's shuffled-control models suggest about 0.3. If the tasks do not need u, "post matches full" would mean nothing.
3. **The full models use u.** The wrong control costs at least 0.5 nats on the next-token task in at least 4 of 5 full models, as before.

**N1b, primary:**

| Outcome | Condition |
|---|---|
| Supported: an offset is enough | Both upper bounds below +0.05, and both standard deviations at most 0.04 |
| Refuted: the dynamics do work an offset cannot | At least one interval wholly above +0.05 |
| Some effect, size unclear | At least one interval excludes 0 on the positive side, and neither rule is met |
| Inconclusive | Everything else, including a standard deviation above 0.04 |

**N1a, secondary hypothesis:**
- **Precondition, the measure can fail.** On each trained full model, an injected multiplicative control, scaling the block's output by 0.9 against 1.1, reads at most 0.1. If not, N1a is untestable.

| Outcome | Condition |
|---|---|
| Supported: the trained effect is an offset | The offset fraction is at least 0.8 in at least 4 of 5 full models |
| Refuted: training made the effect depend on the state | At most 0.5 in at least 4 of 5 |
| Intermediate | Anything else |

**Why these thresholds.**
- **0.05 nats** is the margin T1 and G1d used. G1 and G1c's seed-to-seed spread in task loss was about 0.005, so the margin is well resolved.
- **0.8 and 0.5** bracket the untrained value of 0.92. A drop to 0.5 or below means at least half the control's effect varies with the state.
- These were set after the untrained calibration only.

## Secondary and exploratory outcomes

- **S1, the pre arm.** Pre minus full, per task, with 95% intervals. Can the dynamics turn an input offset into what the later layers need, when the field never sees u?
- **S2, other diagnostics of the control's effect:**
  - the share of Δ's squared norm along its top singular direction;
  - Δ's size relative to the state's size;
  - the offset fraction at the untrained initial weights.
- **S3, contraction.** G1c's contraction ratio in the full and pre arms. In the full arm this is a third replication. In the post and none arms the field does not see u, so their ratio is exactly 1 by construction and is not measured.
- **S4, the remaining checks.** Wrong-control costs in every arm that sees u; α; training time.

## Sanity checks before trusting results

- Every data file's fingerprint matches the manifest.
- G1's batch check and same-model check pass for the new seeds.
- Every number in the report is computed by code from saved JSON records. `n1_analyze.py` is written and committed before any real run, and tested on synthetic records covering every branch of both decision rules.

## Compute, where it runs, and stopping rule

- **Where:** one Colab job, `n1_offset`, on a T4, resumable.
- **Cost:** 20 runs at about 0.09 hours, plus the measurements; about 2 hours.
- **Stopping rule:** exactly these runs and measurements, each once. A crashed run resumes from its checkpoint. Nothing is added after seeing results except as a dated, exploratory entry in the deviation log.

## What each outcome would teach

| N1b | N1a | What we would learn | Next |
|---|---|---|---|
| Supported | Supported | Switching between readings needs only a direction, and the trained model uses one. The ODE dynamics play no needed part in switching; G1c's contraction regimes were side effects of the offset. | N2: whether dynamics matter when readings must coexist, the setting where an offset cannot keep two readings apart |
| Supported | Refuted | An offset is enough, yet the full model's control effect varies with the state. The state-dependent part is not needed, like G1d's contraction. | N2, and a probe of what the state-dependent part encodes |
| Refuted | Supported | The full model's effect is mostly an offset, yet the post model cannot match it. Most likely a learnability difference: the later layers alone find the offset harder to use than dynamics that carry it. S1, the pre arm, helps tell. | Test learnability directly, for example post with a longer budget, as a new test |
| Refuted | Refuted | The dynamics do state-dependent work for the reading that a fixed offset cannot. This is the first evidence that the ODE block's dynamics carry the reading. | Find the mechanism: the Jacobian spectrum, and how Δ varies with the context |
| Untestable: positive control fails | | The tasks don't need the control, which would contradict G1, G1c and G1d. | Look for a bug before anything else |

## Limitations known now

- **Same limits as G1–G1d:**
  - 10% of the budget, one learning rate (not re-tuned per arm), one size and one dataset;
  - two tasks, with farther and harder confounded.
- **Capacity versus learnability.** The post arm can represent the full arm's function exactly only if the full arm's effect is exactly an offset. N1a measures how close it is, and the outcome table separates the cases. But a single test cannot fully separate capacity from learnability.
- **Two arms reported as secondary.** The pre arm and the contraction measures inform the reading of the result, but do not decide it.

## Deviation log

### Implementation details fixed before any real run, 2026-10-06

Results seen before this entry: only test-mode runs of `n1_offset.py` on tiny models (width 32, 60 steps), unit tests of the arms and measures, and synthetic records for the analysis. No real N1 run had started.

- **Code.**
  - `n1_offset.py` trains the four arms (through G1's `train_g1`) and measures them.
  - `n1_analyze.py` applies both decision rules and writes `N1_RESULTS.md`.
  - The Colab job is `colab/jobs/n1_offset.py`.
- **How the arms are built.** Each arm replaces the ODE block's forward method on the model instance. Every arm keeps the same modules, so `control_embed` exists in all four, unused in the none arm, and all start from identical weights for a seed.
- **The must-match check** compares logits in evaluation mode, so dropout is off, on one training batch per seed. It also confirms that, with the control *not* zeroed, the post, pre and none arms each differ from the full arm, so the construction is not vacuous. In the test runs: identical with the control zeroed, different otherwise.
- **N1a's measurement.**
  - The block's output is taken in evaluation mode for u = +1 and u = −1. The offset fraction and top singular share are computed in float64 over all 32,768 token positions.
  - The injected check scales the u = +1 output by 0.9 against 1.1.
  - Δ's relative size is the mean ‖Δ‖ over the mean ‖output‖.
- **S3 in the pre arm.** The field never sees u there, so each control value's trajectory starts from h0 + `control_embed(u)`, with the field run with no control. The full arm uses G1c's measurement unchanged.
- **A risk noticed in testing, recorded and not acted on.**
  - In tiny *untrained* test models, the injected multiplicative check read 0.04 to 0.10, against 0.02 in full-size untrained models. In tiny models trained for only 60 steps, it read 0.95 to 0.96.
  - The reason: when the block's output carries a large component shared by all tokens, scaling the output also looks like adding a constant vector. Trained transformers often develop such a component.
  - So N1a will very likely come out "untestable" through its own precondition in the real run. The rule is unchanged, because it does what it was written to do. N1b does not depend on this measure, and the offset fraction and top singular share are still reported, as description.
- **Tests.**
  - The job ran end to end on tiny models, and a second run skipped all finished work. Its preconditions failed there, as expected after 60 steps, because the tasks are not yet learned.
  - `n1_analyze.py` gave the expected verdicts on synthetic records for 14 cases:
    - N1b supported (including the post arm doing *better*, and an interval wholly below the margin), refuted through either task, some effect, inconclusive through noise, and untestable through each of its three preconditions;
    - N1a supported, refuted, intermediate, and untestable through the injected check.

## References

Park, K., Choe, Y. J., & Veitch, V. (2023). *The linear representation hypothesis and the geometry of large language models*. arXiv. https://doi.org/10.48550/arxiv.2311.03658

Perez, E., Strub, F., de Vries, H., Dumoulin, V., & Courville, A. (2018). FiLM: Visual reasoning with a general conditioning layer. *Proceedings of the AAAI Conference on Artificial Intelligence, 32*(1). https://doi.org/10.1609/aaai.v32i1.11671

Turner, A. M., Thiergart, L., Leech, G., Udell, D., Vazquez, J. J., Mini, U., & MacDiarmid, M. (2023). *Steering language models with activation engineering*. arXiv. https://doi.org/10.48550/arxiv.2308.10248
