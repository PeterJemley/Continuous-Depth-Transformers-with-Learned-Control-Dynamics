# Preregistration N2: do dynamical regimes become necessary when two readings must coexist?

Date: 2026-10-06    Status: **approved by the author on 2026-10-06, final; amendment A1 (the per-dimension floor) approved the same day, before any real run.** Nothing below had been run when this version was committed. From here on, the only changes are dated entries in the deviation log.

Type: confirmatory for two hypotheses, N2a (existence) and N2b (necessity), each with its own decision rule. Everything else is secondary or exploratory and labelled so.

Parent documents: `PATHS_AND_UNDERSTANDING.md` ("After G1d", conjecture N2) and `N1_NOTE.md` (next conjecture 1).

**Relation to earlier results.**
- **What switching showed.** In every test so far, the model *switched* between two readings, one per sequence:
  - the control produces contraction regimes (G1c);
  - those regimes are not needed (G1d);
  - a fixed offset does the switching on its own (N1).
- **What N2 asks.** Switching was never polyphony. N2 asks the founding question directly: when one state must serve both readings *at the same time*, does the ODE block develop distinct dynamical regimes for them, and does the model need those regimes?
- **What stands regardless.** All earlier results stand whatever N2 shows.

## The claims

**N2a (existence).** In a model that predicts both the next token and the token after next, from one state in one forward pass, the directions of the ODE block's state that each prediction relies on contract at different rates. The farther reading's directions contract at least twice as fast as the nearer reading's, in at least 4 of 5 models.

**N2b (necessity).** Forcing those two contraction rates to be equal, during the second half of training, costs the model at least 0.05 nats of combined loss.

### The explanation behind it

- **Why switching didn't need regimes.** When the model switches, the whole state serves one reading at a time, so nothing has to be kept apart. N1 showed an offset is enough.
- **Why coexistence might.** When both readings must be read from one state at once, they compete for it. Information useful to one may interfere with the other.
- **The proposed mechanism.** One way to keep them apart is for different directions of the state to evolve at different rates: different timescales, like registers in music.
  - G1c found that the farther reading goes with stronger contraction.
  - Networks with units at several timescales develop functional hierarchy (Yamashita & Tani, 2008).
  - State-space sequence models run many decay rates side by side (Gu et al., 2021), but over time, not depth.
- **Prior work on the setup.** Predicting several future tokens with separate heads from a shared trunk improves language models (Gloeckle et al., 2024). So the dual-reading setup is established, not invented here.

## Competing hypotheses

| Hypothesis | N2a predicts | N2b predicts |
|---|---|---|
| The claim: per-reading timescales, needed | Ratio ≥ 2 | Equalising costs ≥ 0.05 nats |
| Regimes exist but are side effects, as in G1c and G1d | Ratio ≥ 2 | No material cost |
| The later layers separate the readings, and the ODE block treats both alike | Ratio < 1.25 | Nothing to equalise; N2b untestable |
| Any two heads, even identical ones, produce a split (an artefact) | The dual-same arm also shows a ratio ≥ 1.25; must-fail fails | — |
| The subspace estimate is noise | The split-half check fails | — |

## Design

### Models

All models are the paper's `HybridODETransformer` with exact gradients and no control signal. They have **two output heads** (two untied 256 × 50,257 linear maps), both reading the same final state, and train on the mean of the two heads' cross-entropy losses. Data, batch, optimiser, schedule and the 2,879-step budget are as in G1–N1.

| Arm | Head A target | Head B target | Training |
|---|---|---|---|
| **dual** | token i + 1 | token i + 2 | Full budget. A checkpoint is kept at step 1,440. |
| **dual-same** (must-fail control) | token i + 1 | token i + 1 | Full budget. Same parameters as dual; two heads, one reading. |
| **equalised** | i + 1 | i + 2 | Resumes from the dual model's step-1,440 checkpoint, with the equalising penalty. |
| **sham** (null contrast) | i + 1 | i + 2 | Resumes from the same checkpoint, with the same penalty computation at zero weight. |

- **Seeds:** 3401 to 3405, not used before.
- **Pairing.** A seed's equalised and sham arms resume from that seed's own checkpoint, with the same data order, so they pair with the dual model's own second half.
- **Runs:** 10 full-length and 10 half-length, plus the pilot below.

### Each reading's subspace

- **How it is found.** For each reading k (A or B), the directions it relies on are estimated at the ODE block's output, h_out. They are the top 16 eigenvectors of the per-token gradient covariance, E[g_k g_kᵀ], where g_k = ∂L_k / ∂h_out for one token's 256-dimensional state.
- **Data.** Estimated on validation windows 1,025 to 1,280, in evaluation mode.
- **The result.** U_A and U_B, each a 256 × 16 orthonormal basis, applied at every token position.
- **Why 16.** It is small enough that two subspaces of 256 dimensions can be distinct, and large enough to estimate stably. 32 is reported as a sensitivity check (S3).

### Contraction within a subspace

- **The measure.** c_k = −tr(U_kᵀ (∂f/∂h) U_k) / 16, per token: the mean contraction rate per dimension within subspace k.
- **The estimator.** Hutchinson's estimator with probes confined to the subspace: z = U_k ε, with ε a random ±1 vector in 16 dimensions at each token, so that E[zᵀ J z] = tr(U_kᵀ J U_k). It uses 8 probes per batch, at depths 0, 0.25, 0.5, 0.75 and 1, along dopri5 trajectories at tolerance 1e-5, as in G1c.
- **Data.** Measured on validation windows 1,281 to 1,536, separate from the windows used to find the subspaces.
- **The ratio.** R_sub = c̄_B / c̄_A, scored by G1c's table: rates below 0.1 are treated as near zero, not divided.

### The equalising penalty (N2b)

- **When.** From step 1,440 to 2,879.
- **The subspaces.** U_A and U_B are fixed, estimated from the step-1,440 checkpoint on windows 1,025 to 1,280.
- **The penalty.** At every step, λ · (c_A − c_B)², with c_A and c_B estimated for each sequence at the four Euler states, by one projected probe per sequence and state. The penalty is trained through, with the graph kept, as in G1d.
- **Choosing λ.** By the G1d procedure, on a pilot of seed 100: a dual run to step 1,440, then continuations at λ = 0.01, 0.1, 1 and 10, in order. The chosen λ is the first at which the measured |log R_sub| at the end is at most log 1.25. The rule looks only at that ratio, never at a loss. If no λ qualifies, N2b is untestable.

### Calibration still to be done

Unlike G1c and N1, the subspace measure has not yet been run on untrained models.
- **What we already know.** In earlier tests, whole-state contraction at initialisation was close to zero, about ±0.01 per token. G1c's scoring table handles such rates, so ratios of near-zero numbers are not taken at face value.
- **What will be checked.** During the build, before any real run, the measure will be run on untrained dual and dual-same models, and its split-half noise will be measured.
- **If that shows a threshold or a control cannot work as written,** an amendment will be proposed for approval before the job is queued, as amendment A1 was for G1.

## Primary outcomes and decision rules

### Preconditions for both

1. **Analytic check.** On linear fields f(h) = A h, the projected estimator returns tr(UᵀAU) for a known U within 2% when averaged, and exactly for diagonal A with U aligned to the axes. The training penalty's gradient is checked as in G1d.
2. **The batch, data and same-model checks** from G1 pass for the new seeds.

### N2a: do the two readings' directions contract at different rates?

**Preconditions:**
- **Must fail.** The dual-same models give R_sub from 0.8 to below 1.25 in at least 4 of 5. Two heads reading the *same* thing must not produce a split.
- **Must match.** Subspaces estimated from the odd and the even estimation windows give log R_sub values within 0.1 of each other, in every dual model.

| Outcome | Condition |
|---|---|
| Supported: per-reading timescales exist | R_sub ≥ 2 in at least 4 of 5 dual models |
| Refuted | R_sub < 1.25 in at least 4 of 5 |
| Intermediate | Anything else |

### N2b: does equalising them cost the model?

**Primary outcome.** D_joint = (L_A + L_B), equalised − dual, per seed, on the whole validation split at step 2,879. The interval is a t-interval at 95% with 4 degrees of freedom.

**Preconditions:**
- **There is something to equalise.** R_sub ≥ 2 in at least 4 of 5 dual models, which is N2a supported. If N2a is not supported, N2b is reported as untestable.
- **The equalising took hold.** |log R_sub| ≤ log 1.25 in at least 4 of 5 equalised models, measured as above.
- **Null contrast.** The sham arm's D_joint interval lies within ±0.05.

| Outcome | Condition |
|---|---|
| Supported: the regimes are needed when readings coexist | The interval lies wholly above +0.05 |
| Refuted: not needed | The interval lies wholly below +0.05, with a standard deviation of at most 0.04 |
| Some effect, size unclear | The interval excludes 0 but meets neither rule |
| Inconclusive | Everything else |

**Why these thresholds.** They are the same as before: 2 and 1.25 from amendment A1, and the 0.05-nat margin from T1, G1d and N1. These models have two heads and a combined loss, so the margin applies to the sum.

## Secondary and exploratory outcomes

- **S1, overlap between the two subspaces.** The mean squared cosine of the principal angles between U_A and U_B. If they largely coincide, distinct rates are impossible, and this shows it.
- **S2, each reading's loss separately in N2b.** Which reading pays for equalising?
- **S3, the same analysis at 32 dimensions per subspace.**
- **S4, whole-state contraction** (G1c's measure without projection) in the dual and dual-same arms.
- **S5, how far the subspaces drift.** Overlap between the step-1,440 and step-2,879 subspaces in the dual arm, since N2b fixes them at step 1,440.
- **S6, the cost of reading twice.** Head A's loss in dual against dual-same. Does reading the farther target cost the nearer one?

## Sanity checks before trusting results

- Every data file's fingerprint matches the manifest.
- One batch is decoded, with both heads' targets checked against the input.
- The resumed arms start from bit-identical weights and optimiser state at step 1,440, and see the same batches as the dual model's second half.
- Every number in the report is computed by code from saved JSON records. `n2_analyze.py` is written and committed before any real run, and tested on synthetic records covering every branch of both rules.

## Compute, where it runs, and stopping rule

- **Where:** one Colab job, `n2_coexist`, on a T4, resumable.
- **Cost:**
  - 10 full-length runs (two heads cost more per step, about 0.12 hours each);
  - 10 half-length runs with the penalty, about 0.1 hours each;
  - the pilot, up to 0.6 hours;
  - the measurements;
  - about 3 to 3.5 hours in all. Two Colab sessions may be needed.
- **Stopping rule:** exactly these runs and measurements, each once. A crashed run resumes from its checkpoint. Nothing is added after seeing results except as a dated, exploratory entry in the deviation log.

## What each outcome would teach

| N2a | N2b | What we would learn | Next |
|---|---|---|---|
| Supported | Supported | When two readings must coexist in one state, the dynamics give them different timescales, and the model needs that. This is the founding polyphony idea, in a measurable and necessary form. | Polyphonic ODE with more readings (path D); scale (H); does the farther reading's register generalise to other "far" readings? |
| Supported | Refuted | Distinct timescales appear under coexistence but are not needed, as with switching (G1c, G1d). Separation happens elsewhere. | Whole-stack test; probe where the readings are actually kept apart |
| Refuted | — | Even when readings coexist, the ODE block treats them alike. Coexistence alone does not create registers in depth. | Represent readings in subspaces without depth dynamics; or G3, multiplicative structure that makes registers cheap |
| Intermediate | — | Some separation, short of distinct registers | S1 and S3 show whether the subspaces overlap too much to tell |
| Untestable: dual-same also splits | — | Two heads alone create a split; the measure cannot attribute it to two readings | Redesign the measure |

## Limitations known now

- **Same limits as before:**
  - 10% of the budget, one learning rate (not re-tuned for two heads), one size and one dataset;
  - two readings that are related (adjacent future tokens), and farther and harder still confounded.
- **The subspaces are estimates.** Gradient covariance identifies the directions each head is *sensitive* to, which is not necessarily where the information is stored. S1 and S5 report how well defined and how stable they are.
- **N2b fixes the subspaces at mid-training,** and so constrains the second half only. A model could move its readings to new directions to escape the penalty; S5 would show it.
- **The register picture is one mechanism among several.** The readings could be kept apart by attention patterns, or by the later layers, without any difference in timescale. N2 tests only the timescale mechanism.

## Deviation log

### Amendment A1, before any real run, 2026-10-06: approved by the author

Results seen before this entry:
- the calibration this preregistration promised, on five **untrained** full-size models (G1's seeds 3001–3005, not N2's), using the first 128 windows of the test-mode token file, which N2 does not use;
- test-mode runs on tiny models;
- synthetic records.

No real N2 run had started.

**What the calibration found.**
- **The near-zero floor is in the wrong units.**
  - N2 measures contraction *per dimension* inside each 16-dimensional subspace. G1c's floor of 0.1, and its "clearly contracts" level of 0.2, were set for a different quantity: the per-token trace summed over all 256 dimensions.
  - Untrained models gave per-dimension subspace rates of at most 0.0005 in absolute value. Trained per-dimension rates will be of the order of a whole-state trace divided by 256, a few thousandths to a few hundredths.
  - A floor of 0.1 would therefore classify almost every trained model as "not contracting", and N2a could not come out supported whatever the dynamics did.
- **Dual and dual-same models are identical at initialisation, as expected.**
  - An untrained model's gradient covariance depends on how often each token appears as a target, and that is the same for the next token and the token after next.
  - The two heads' subspaces barely overlap at initialisation (0.066 to 0.075), only because the heads start with different random weights. That is why the must-fail control has two heads too. It can only bite after training, which is where it is applied.
- **The subspaces are stable.** Odd and even estimation windows gave subspaces overlapping 0.83 to 0.86 at initialisation.

**The amendment.** In the scoring table for R_sub, the near-zero floor becomes **0.002 per dimension**, and "clearly contracts" becomes **0.004 per dimension**, in place of 0.1 and 0.2. The floor is 4 times the largest untrained value seen (0.0005). G1c's floor was about 5.6 times its untrained maximum. Every other rule, threshold and control is unchanged, including the ratios 2 and 1.25 and the 0.05-nat margin.

### Implementation details fixed before any real run, 2026-10-06

- **Code.**
  - `n2_coexist.py` trains, checkpoints at step 1,440, runs the pilot and the penalised continuations, and measures.
  - `n2_analyze.py` applies both rules, with amendment A1, and writes `N2_RESULTS.md`.
  - The Colab job is `colab/jobs/n2_coexist.py`.
- **The second head** is initialised from its own seed (the model's seed + 99,991), so dual and dual-same start identically for a seed.
- **The penalty uses batch means.** It is applied to the *batch means* of the per-sequence, per-state contraction estimates, C_A and C_B. Averaging over 128 estimates (32 sequences × 4 states) before squaring reduces the upward bias that squaring a single noisy estimate would add. A small bias remains.
- **The manipulation check uses fresh subspaces.** The check that the equalising took hold, and the pilot rule, measure R_sub with subspaces re-estimated on the trained model at step 2,879, as in N2a. This way, a model that escaped the penalty by moving its readings to new directions shows up as not equalised. The penalty itself uses the subspaces fixed at step 1,440.
- **Same batches.** The batch drawn at step 1,440 is fingerprinted in the dual, equalised and sham runs, to confirm the resumed arms see the same batches.
- **The sham arm** computes the same penalty at weight 0, so its gradient is exactly zero.
- **Tests:**
  - The analytic checks passed: dense error 0.12%, diagonal exact, gradient 0.41%.
  - The job ran end to end on tiny models, including the pilot, the mid checkpoint and both resumed arms. The batch fingerprint at step 1,440 matched across the dual, equalised and sham runs.
  - `n2_analyze.py` gave the expected verdicts on synthetic records for 13 cases. These covered both supported; N2a supported and N2b refuted; N2b with some effect, and inconclusive; N2a refuted, and intermediate; the case where reading A is near zero; and untestable through each precondition of both rules.

## References

Finlay, C., Jacobsen, J.-H., Nurbekyan, L., & Oberman, A. M. (2020). *How to train your neural ODE: The world of Jacobian and kinetic regularization*. arXiv. https://doi.org/10.48550/arxiv.2002.02798

Gloeckle, F., Youbi Idrissi, B., Rozière, B., Lopez-Paz, D., & Synnaeve, G. (2024). *Better & faster large language models via multi-token prediction*. arXiv. https://doi.org/10.48550/arxiv.2404.19737

Gu, A., Goel, K., & Ré, C. (2021). *Efficiently modeling long sequences with structured state spaces*. arXiv. https://doi.org/10.48550/arxiv.2111.00396

Hutchinson, M. F. (1990). A stochastic estimator of the trace of the influence matrix for Laplacian smoothing splines. *Communications in Statistics: Simulation and Computation, 19*(2), 433–450. https://doi.org/10.1080/03610919008812866

Yamashita, Y., & Tani, J. (2008). Emergence of functional hierarchy in a multiple timescale neural network model: A humanoid robot experiment. *PLoS Computational Biology, 4*(11), e1000220. https://doi.org/10.1371/journal.pcbi.1000220 (A correction was published in 2010.)

### The run, 2026-10-06

The job needed more than one Colab session, the second after a restart. The earlier sessions saved their finished runs and checkpoints to Drive but did not reach the upload step, so their logs are not in the repository. The final session (`colab-results` branch `n2_coexist/20261006-164621`, code commit a7724b7) resumed from Drive, finished the remaining runs, and uploaded all 50 result records. No Drive failures were recorded, and nothing was changed after approval. The results are in `N2_RESULTS.md` and `N2_NOTE.md`.
