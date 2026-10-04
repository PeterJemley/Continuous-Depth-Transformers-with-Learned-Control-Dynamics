# Preregistration T0: are Model A's training gradients the gradients of its own forward pass?

Date: 2026-10-04    Status: **approved by the author on 2026-10-04, final.** Nothing below had been run when this version was committed. From here on, the only changes are dated entries in the deviation log.

Type: confirmatory for the primary outcome and its decision rule. Everything else is secondary or exploratory and labelled so.

Parent document: `CONJECTURE.md` in this folder, test T0, rival G. Whatever T0 shows, the refutations in `experiments/wikitext2-replication/RESULTS.md` and `experiments/wikitext103-training/RESULTS.md` stand. Those studies tested the paper's model as built and trained. T0 asks a new question about how it was trained. It cannot restore any of the paper's claims.

## The claim

**At the end of WikiText-103 training, the gradient Model A was trained with differs materially from the true gradient of the forward computation it actually ran. For the ODE block's parameters, the direction of the expected training gradient is more than about 26° away from the direction of the expected true gradient (cosine below 0.90) in at least 5 of 6 trained models.**

### The explanation behind it

`ODEFlowBlock` integrates with `torchdiffeq.odeint_adjoint`, using Euler steps between five saved times, τ = 0, 0.25, 0.5, 0.75 and 1. Reading the backward pass in torchdiffeq 0.2.5 (`OdeintAdjointMethod.backward`) shows two departures from the gradient of the forward computation.

- **G1, mismatched dropout.** The backward pass calls the vector field again, in training mode, so dropout draws fresh masks. The forward pass used other masks. The gradient therefore belongs to a different random function than the one that produced the loss.
- **G2, derivatives taken at the wrong point.** The forward Euler step from τᵢ₋₁ to τᵢ is evaluated at the state and time where the step starts. The backward pass integrates the adjoint equation backwards with one Euler step per interval, starting from the step's end. It therefore evaluates the vector field's derivatives at τᵢ and at the later state, where the step finished, not at the state where it started. For example, the first forward step was taken at τ = 0, and its derivatives are taken at τ = 0.25. The state is reset to the saved forward value at each saved time, so the error does not accumulate across steps. Within each step it grows with the step's size, which is α/4 times the vector field. α ended between 1.0 and 2.1, compared with 0.1 at the start.

Both can bias the expected gradient, a systematic error that averaging over dropout draws does not remove. G2's bias grows with α and is present with dropout off. G1 adds noise, and also a bias of its own: the true gradient depends on the same masks being used going forward and coming back, and independent masks lose that dependence. The primary outcome targets the bias, because a bias steers training somewhere else, while noise only slows it. Secondary outcome S2, with dropout off, separates G2 from G1.

## Competing hypotheses

| Hypothesis | What it predicts for the primary outcome |
|---|---|
| The claim: a material defect, mainly G2 | Cosine below 0.90 at the trained models; falls further as α is scaled up; near 1 at small α |
| No material defect: the adjoint is accurate enough here | Cosine at or above 0.98, close to the noise reference |
| A defect from dropout only (G1) | The primary cosine is low, but with dropout off (S2) the adjoint and true gradients agree, and S3 shows no fall with α |
| The test is broken (wrong "true" gradient, mismatched weights or batches) | Must-match controls fail |
| Noise, too few draws | The reference cosine between two independent averages of true gradients is itself well below 1 |

## Refuting result

The claim is refuted if, in at least 5 of 6 trained models, the primary cosine is at least 0.98 or within 0.01 of that model's noise reference, whichever bound is lower.

## Design

### Models

- **Primary:** the six WikiText-103 Model A final weights, `main_hybrid_seed1001` to `main_hybrid_seed1006`. They are on the author's Google Drive (`cdt-colab/wt103_study/final/`), and stage 3 loaded them.
- **Weight check:** before use, each model's test loss on the whole test split is recomputed and must match the value recorded in `experiments/wikitext103-training/results/test_main_hybrid_seed*.json` to within 1e-4.
- **Secondary:** the same six models freshly initialised, α = 0.1, which is the state training started from. They are rebuilt exactly with the study's `new_model('hybrid', seed)`.

### Data

Eight batches per model, each 32 windows of 128 tokens, the training shape. They are drawn from the WikiText-103 training tokens with `torch.Generator().manual_seed(12345)`, so the same eight batches are used for every model and every gradient type. One batch is decoded to text and inspected before any gradient is computed.

### Gradients

All are computed in fp32, with the model in training mode and gradient clipping off. Clipping rescales the total gradient but leaves its direction unchanged; norms are reported separately. Each gradient is the gradient of mean next-token cross-entropy on one batch.

- **Training gradient (a):** the model exactly as trained, with `odeint_adjoint`, Euler steps and dropout on.
- **True gradient (c):** the same forward computation, the same weights, the same batch and the same dropout masks, differentiated exactly by ordinary backpropagation through the four Euler steps (`torchdiffeq.odeint`, not the adjoint version). The masks are made identical by setting the random seed to the same value before the forward pass.
- **Dropout-off versions (b) and (e):** the adjoint and exact versions with dropout switched off in every module.

**Averaging out dropout noise.** For each batch, ḡₐ is the mean of 16 training gradients and ḡ_c the mean of 16 true gradients. The seeds for those 32 draws are fixed: seeds 0 to 15 are shared between the two lists, so each pair shares its forward masks. A further 16 true gradients, seeds 16 to 31, give ḡ_c′, the noise reference.

### Parameter groups

- **ODE block, primary:** every parameter of `ode_block.func`.
- **Upstream, secondary:** the token and position embeddings and the two early blocks. Their gradients pass back through the adjoint.
- **Downstream, a must-match control:** the two late blocks, `ln_f` and `head`. Their gradients are computed before the adjoint runs, so they must be identical for (a) and (c).

## Primary outcome

For each model, the cosine similarity between ḡₐ and ḡ_c over the ODE block's parameters, averaged over the 8 batches. The noise reference is the cosine between ḡ_c and ḡ_c′, averaged the same way.

The unit of analysis is the trained model, 6 in total. The 8 batches within a model are averaged and are not counted as independent.

## Decision rule

**Precondition.** The noise reference must be at least 0.95 in every model. If it isn't, the number of draws is doubled, up to 64, and the rule is applied to the first count that meets the precondition. If even 64 draws don't meet it, the result is "too noisy to decide", and no verdict is given.

| Outcome | Condition |
|---|---|
| Supported: material defect | In at least 5 of 6 models, the primary cosine is below 0.90 and below the noise reference by more than 0.05 |
| Refuted: no material defect | In at least 5 of 6 models, the primary cosine is at least min(0.98, reference − 0.01) |
| Intermediate | Anything else. All six values are reported with the reference. It is called a measurable but moderate defect if the cosine is below the reference by more than 0.01 in at least 5 models, and mixed otherwise. |
| Untestable | Any must-match control fails and its cause cannot be fixed without changing this plan. A fix that changes the plan goes into the deviation log, and the results are then exploratory. |

The 0.90 and 0.98 thresholds were chosen before any gradient was computed. A cosine of 0.90 is an angle of about 26° between the expected gradients, and 0.98 about 11°.

## Must-match controls

All are run before the primary comparison, on seed 1001 and batch 1.

1. **Repeatability:** two true gradients with the same random seed agree, cosine at least 0.99999. This bounds GPU nondeterminism.
2. **Same forward pass:** training and true gradients with the same seed give the same loss, relative difference at most 1e-6, and identical downstream gradients, cosine at least 0.99999. This confirms the masks really are shared.
3. **The true gradient is true:** with dropout off, the true gradient from `torchdiffeq.odeint` equals the gradient from a hand-written loop of four Euler steps, cosine at least 0.99999.
4. **Agreement where theory requires it:** with dropout off and α scaled down to 0.001, the adjoint and true gradients agree, cosine at least 0.999. G2's error is proportional to step size, so it must vanish here. This is checked on every model.

## Must-fail and injected-effect controls

1. **Must fail:** the true gradient with each element's sign flipped at random has a cosine with the true gradient within 0.05 of 0.
2. **Injected effect:** the true gradient plus Gaussian noise of equal norm has a cosine with the true gradient within 0.02 of 0.707. This confirms the measure detects a known degradation of a known size.

## Secondary and exploratory outcomes

Each is reported for every model, labelled as secondary.

- **S1, the error of a single training step:** the cosine between one training gradient (a) and the true gradient with the same masks (c), against the cosine between two true gradients with different masks. Their ratio, (1 − cos(a,c)) / (1 − cos(c,d)), says how the per-step error compares with dropout's own variation.
- **S2, G2 alone:** dropout off, the cosine between (b) and (e).
- **S3, dose-response in α, exploratory:** dropout off, on the trained models, with α scaled to 0.001, 0.1, 0.5, 1 and 2 times its learned value. G2 predicts the cosine falls as α grows.
- **S4, at initialisation:** the primary measure on the six freshly initialised models. This shows whether the defect was present from the start of training.
- **S5, norm ratio:** ‖ḡₐ‖ / ‖ḡ_c‖ for the ODE block. Training clipped the total gradient norm at 1.0, so an inflated ODE gradient would shrink every other parameter's step too.
- **S6, the upstream group:** the primary measure for the embeddings and early blocks.
- **S7, mixed precision, exploratory:** the training gradient under fp16 autocast, as in training, against the fp32 true gradient.

## Sanity checks before trusting results

- Each model's recomputed test loss matches its recorded value (Models above).
- One training batch is decoded to text and inspected.
- The torchdiffeq and PyTorch versions are recorded. The G2 explanation was read from torchdiffeq 0.2.5. If a different version is installed, its `OdeintAdjointMethod.backward` is checked for the same per-interval state reset before running, and any difference is logged.
- Every number in the report is computed by code from the saved JSON outputs.

## Compute, where it runs, and stopping rule

- **Where:** one Colab job, `t0_gradients`, through `colab/runner.ipynb` on a T4, reading the final weights and token files from the author's Drive folder read-only.
- **Cost:** about 2,500 backward passes of a 30M-parameter model. Expected under 30 minutes, without counting the possible doubling of draws.
- **Stopping rule:** exactly these computations, run once. If the job crashes, it is rerun unchanged. Nothing is added after seeing results except as logged, exploratory analyses.

## What each outcome would teach

| Outcome | What we would learn | Next |
|---|---|---|
| Supported, and S2 and S3 show G2 grows with α | Model A was trained on systematically biased gradients that worsened as α grew. The WikiText-103 comparison measured the architecture together with a training defect. The paper's refuted claims stay refuted: they were about the model as built and trained. | T1 trains Model A with exact gradients as a new model under its own preregistration, and measures how much of the 1.1-nat gap a correct gradient closes. |
| Supported, and S2 shows the gap comes mainly from dropout (G1) | The defect is the dropout mismatch, not the adjoint's evaluation point. | T1 includes a run with the adjoint and dropout removed only inside the ODE block, to separate the two causes. |
| Supported at the trained models but not at initialisation | The defect grew as α grew, so early training was mostly sound. | T1 is still needed; its curves show when the paths diverge. |
| Refuted | Rival G is ruled out as a material bias at the end of training. The 1.1-nat gap belongs to the architecture or its other training settings. | Drop G. C2 and rival P become the main candidates, so go to T1 without the exact-gradient rung. |
| Intermediate | There is a measurable but moderate defect. Whether it matters for the loss is a training question. | T1 with the exact-gradient rung, as planned. |
| Untestable | The instrument failed, and that is reported as such. | Fix the instrument under a logged deviation, and rerun as exploratory. |

## Limitations known now

- Only two states of training are measured: the start and the end. The gradient defect along the way is inferred, not measured.
- A biased gradient does not prove worse training. T0 tests the mechanism; only T1 can test the consequence for the loss.
- One model size, one dataset, one solver configuration.

## Deviation log

### Amendment before any real result, 2026-10-04: awaiting the author's approval

Results seen before this entry: only test-mode runs of `t0_gradients.py`, on two tiny models (width 32, trained for 60 steps) built to test the code. No gradient of any real WikiText-103 model had been computed.

- **A1. The primary vector leaves out α.** In the test runs, the must-fail control failed for the vector of every ODE block parameter. Flipping signs at random gave a cosine of 0.90, not about 0, because α's single gradient value held 64% of the vector's squared length. A cosine over that vector mostly measures one number. The primary outcome and the decision rule therefore use the vector field's parameters: every ODE block parameter except `output_scale` (α). α's own gradient is reported separately as S8: the training mean, the exact mean, and a second exact mean. The all-parameter version is still computed and reported for comparison. At the real model size and at initialisation, α held under 1% of the squared length, so the amendment matters only if α's gradient dominates at the trained state, which is unknown.
- **A2. Must-match control 4 uses a time-independent field.** The test runs refined the explanation: the adjoint evaluates the time embedding at the end of each step too, and that error does not shrink as α shrinks. So agreement at small α is guaranteed only for a time-independent field. Control 4 now zeroes the time embedding's output, sets α to 0.001 and switches dropout off, and requires a cosine of at least 0.999. In the test runs it gave 0.99995 and 0.99990. The original version, time embedding intact, is kept as secondary S9, which measures the time-mismatch floor.
- **A3. Seeds for more draws.** With K draws, the training and exact gradients share seeds 0 to K − 1, and the noise reference uses seeds K to 2K − 1. This generalises the plan's 0–15 and 16–31. If the draws are doubled, all models are recomputed at the new K.
- **A4. Two clarifications of wording.** Must-match control 4 sets α to the absolute value 0.001. S3 scales each model's learned α by 0.001, 0.1, 0.5, 1 and 2. The precondition applies to the noise reference of the primary vector.
- **A5. Controls are recorded, not acted on, during the run.** The job computes every measure and `t0_analyze.py` applies the decision rule, so a failed control never loses the other measurements. The rule itself is unchanged: any failed control makes the result untestable.
- **Updated cost:** about 6,800 backward passes in fp32, including the secondary measures. That is about 45 to 60 minutes on a T4, not under 30.
