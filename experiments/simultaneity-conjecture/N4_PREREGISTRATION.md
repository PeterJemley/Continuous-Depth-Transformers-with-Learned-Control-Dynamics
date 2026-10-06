# Preregistration N4: are coexisting readings kept apart by direction, and does the model need that?

Date: 2026-10-06    Status: **draft, awaiting the author's approval.** Nothing below has been run.

Type: confirmatory for the primary outcome (N4b) and for the secondary hypothesis N4a, each with its own decision rule. Everything else is secondary or exploratory and labelled so.

Parent documents: `N2_NOTE.md` (exploratory observations and next conjecture 1), and `PATHS_AND_UNDERSTANDING.md`.

**Relation to earlier results.**
- **What N2 found.** When one state had to serve two readings at once, the ODE block did *not* give them distinct timescales (N2a refuted).
- **What N2 noticed by looking.**
  - The two readings relied on partly different directions in the state: their 16-dimensional sensitivity subspaces overlapped 0.57 to 0.63.
  - Two heads reading the *same* target, from different random starting weights, overlapped 0.98 to 0.99.
- **What N4 asks.** Is that separation real on new data, and does the model need it?
- **What stands regardless.** N2's refutation, and every earlier result, stand whatever N4 shows.

## The claims

**N4b (primary): the readings need separate directions.** Forcing two different readings, the next token and the token after next, to be read out through one shared set of directions costs the model at least 0.05 nats of combined loss *more* than the same restriction costs two heads that read the same thing.

**N4a (secondary hypothesis): the separation replicates.** On new seeds and windows, the two readings' sensitivity subspaces overlap at most 0.8 in at least 4 of 5 dual models. Two heads reading the same target overlap at least 0.95 in at least 4 of 5.

### The explanation behind it

- **Separate directions in one space.** A single state can carry several things at once if they occupy different directions, as several notes share one sound by occupying different frequencies.
- **Prior work.** Networks are known to represent many features as directions in one space, even more features than dimensions, at the price of interference (Elhage et al., 2022).
- **What the idea predicts.** If two readings must be held at once, giving them separate directions should limit interference. Squeezing them into the same directions should cost accuracy, and cost more than squeezing a single reading the same way.
- **Prior work on the cost.** Tasks that share parameters are known to interfere through conflicting gradients (Yu et al., 2020).
- **Why this is the polyphony idea.** It is the founding idea in the form N2 pointed to: voices as separable directions in one state, rather than as separate timescales.

### Why the comparison is a difference of differences

- **The naive comparison is confounded.** Restricting any language model's readout to fewer directions costs accuracy by itself (Yang et al., 2017).
- **The same-target arms remove that.** The same restriction is applied to two heads that read the *same* target. Those heads need no separate directions, so their cost is the cost of the restriction alone.
- **What remains.** The extra cost borne by two *different* readings is what the conjecture is about.

## Competing hypotheses

| Hypothesis | Prediction for D (N4b) | N4a |
|---|---|---|
| The claim: readings need separate directions | D ≥ 0.05 | Dual overlap ≤ 0.8 |
| Readings occupy partly separate directions, but can share them at no cost | D < 0.05 | Dual overlap ≤ 0.8 |
| N2's separation was luck of its seeds | D < 0.05 | Dual overlap > 0.8 |
| Restricting the readout costs the same whatever is read | D near 0 | — |
| The restriction is too loose to bind | Tied and separate arms barely differ for either reading | Reported, not a precondition |

## Refuting result

N4b is refuted if the 95% interval for D lies wholly below +0.05 nats, with a standard deviation of at most 0.04.

## Design

### The readout

- **The base model.** All models are N2's dual-head model: the paper's `HybridODETransformer`, exact gradients, no control, two heads reading the same final state.
- **The change.** Each head now reads the final state z through a rank-64 projection: logits_k = W_k (P_k · LayerNorm(z)), with P_k a 64 × 256 matrix and W_k a 50,257 × 64 matrix.
- **Separate:** P_A and P_B are independent, so the two heads can read up to 128 directions between them.
- **Tied:** P_A = P_B, one shared matrix, so both heads read the same 64 directions. Within those 64 they can still use different combinations; what is shared is the space they read from.
- **Initialisation.** Both versions start from the same values: in the separate version, P_B is initialised as a copy of P_A. So separate and tied are the same function at step 0 and diverge only in training. This is an exact must-match control, by construction.

| Arm | Head A target | Head B target | Readout |
|---|---|---|---|
| dual-sep | i + 1 | i + 2 | separate |
| dual-tied | i + 1 | i + 2 | tied |
| same-sep | i + 1 | i + 1 | separate |
| same-tied | i + 1 | i + 1 | tied |

- **Training:** as in N2, on the mean of the two heads' cross-entropy losses, 2,879 steps at peak learning rate 2.4e-3, batch 32 × 128, on the pinned WikiText-103 token files.
- **Seeds:** 3501 to 3505, not used before. A seed fixes the initial weights and the data order for all four arms.
- **Runs:** 20.

### Why 64

- **The intended regime.** 64 is the width at which the restriction should bind on two readings without crippling one. The separate arms read up to 128 of the state's 256 directions, the tied arms 64.
- **What N2 suggests.** The readings' top-16 sensitivity subspaces were about 40% distinct, so two readings plausibly need more than 64 shared dimensions, while one fits in 64.
- **A judgement.** The choice is a judgement, made before any run, and the result is specific to it.

### N4a's measurement

As in N2, on the dual-sep and same-sep models:
- each head's top-16 sensitivity subspace at the ODE block's output, from the per-token gradient covariance on validation windows 1,537 to 1,792, which no earlier test used;
- the overlap of the two subspaces: the mean squared cosine of their principal angles, from 0 (orthogonal) to 1 (identical).

Two random 16-dimensional subspaces of 256 dimensions overlap about 0.06.

## Primary outcome (N4b)

For each seed:

D = (J_dual-tied − J_dual-sep) − (J_same-tied − J_same-sep)

J is the combined validation loss, L_A + L_B, on the whole validation split, each head against its own target. The interval is a t-interval at 95% with 4 degrees of freedom.

## Decision rules

**Preconditions:**
1. **Must match, by construction.** At initialisation, for every seed, the separate and tied versions give bit-identical logits on one batch.
2. **The batch, data and same-model checks** from G1 pass for the new seeds.

**N4b:**

| Outcome | Condition |
|---|---|
| Supported: the readings need separate directions | The interval for D lies wholly above +0.05 |
| Refuted | The interval lies wholly below +0.05, with a standard deviation of at most 0.04 |
| Some effect, size unclear | The interval excludes 0 but meets neither rule |
| Inconclusive | Everything else |

**N4a:**
- **Must fail.** Same-sep overlap is at least 0.95 in at least 4 of 5 models. Two heads reading the same thing must converge on the same directions, or the measure cannot attribute separation to the readings.

| Outcome | Condition |
|---|---|
| Supported: the separation replicates | Dual-sep overlap at most 0.8 in at least 4 of 5 |
| Refuted | Dual-sep overlap at least 0.95 in at least 4 of 5 |
| Intermediate | Anything else |

**Why these thresholds.**
- **0.05 nats** is the margin used throughout: T1, G1d, N1 and N2. The seed-to-seed spread of losses in N2 was about 0.005, so D, a combination of four runs, should have a spread near 0.01. That resolves the margin.
- **0.8 and 0.95** were set from N2's observed values (dual 0.57 to 0.63; same 0.98 to 0.99), leaving room on both sides. N4a is a replication on new seeds and new windows, under a slightly different readout.

## Secondary and exploratory outcomes

- **S1, each arm's losses by reading.** Which reading pays when directions are shared?
- **S2, overlap at the readout itself.** The overlap of P_A's and P_B's row spaces in the separate arms, dual against same. This is the separation at the readout, as against N4a's at the ODE block's output.
- **S3, does the restriction bind?** J_tied − J_sep for each pair of arms separately, with intervals. If both are near zero, the restriction did not bind and D can only be small. The report will then say that N4b could not test the claim at this width.
- **S4, comparison with N2's unrestricted dual models.** Descriptive only, since the seeds differ.

## Sanity checks before trusting results

- Every data file's fingerprint matches the manifest.
- One batch is decoded, and both heads' targets are checked.
- Every number in the report is computed by code from saved JSON records. `n4_analyze.py` is written and committed before any real run, and tested on synthetic records covering every branch of both rules.
- **The calibration this project now does before every run.** N4a's overlap is measured on untrained models of all four arms, and the restriction's cost is checked in test mode, before the job is queued. If a threshold or control turns out not to work as written, an amendment comes back for approval first.

## Compute, where it runs, and stopping rule

- **Where:** one Colab job, `n4_directions`, on a T4, resumable.
- **Cost:** 20 runs at about 0.12 hours each, since the rank-64 heads are cheaper than N2's full heads, plus the measurements. About 2.5 hours in all.
- **Stopping rule:** exactly these runs and measurements, each once. A crashed run resumes from its checkpoint. Nothing is added after seeing results except as a dated, exploratory entry in the deviation log.

## What each outcome would teach

| N4b | N4a | What we would learn | Next |
|---|---|---|---|
| Supported | Supported | Coexisting readings occupy separate directions, and the model needs that separation: forcing them through shared directions costs more than it costs one reading. This is polyphony as separable directions in one state, in a measured and necessary form. | Scale it: more readings, and less related ones (a topic or sentiment label alongside the next token); measure how separation grows with the number of voices |
| Refuted | Supported | The readings occupy partly separate directions but can share them at no cost. The separation, like G1c's contraction, is real but not load-bearing. | Ask what *is* load-bearing under coexistence, for example attention patterns |
| Refuted | Refuted | N2's separation does not replicate, and sharing costs nothing. At this scale the two readings are, in effect, one. | Readings that differ more (path 3 in N2's note), before concluding |
| Supported | Refuted | Sharing costs, though the sensitivity subspaces do not separate as N2 suggested. The need lies in directions N4a's measure does not capture. | Compare S2's readout-level overlap with N4a's |
| Inconclusive, or S3 shows no binding | — | The restriction at 64 dimensions was too loose or too noisy to tell | A tighter width, as a new test |

## Limitations known now

- **Same limits as before:**
  - 10% of the budget, one learning rate (not re-tuned for the new readout), one size and one dataset;
  - two related readings, with farther and harder confounded.
- **The test acts at the readout.** It restricts the final state's readout, not the ODE block's output, where N2 observed the separation. A cost at the readout shows the readings need separate directions *somewhere* in what is read; it does not locate them in the ODE block.
- **One width.** The result is specific to rank 64.
- **N4a's thresholds come from N2's data.** That makes N4a a replication, not a fresh prediction.

## Deviation log

None yet.

## References

Elhage, N., Hume, T., Olsson, C., Schiefer, N., Henighan, T., Kravec, S., et al. (2022). *Toy models of superposition*. arXiv. https://doi.org/10.48550/arxiv.2209.10652

Gloeckle, F., Youbi Idrissi, B., Rozière, B., Lopez-Paz, D., & Synnaeve, G. (2024). *Better & faster large language models via multi-token prediction*. arXiv. https://doi.org/10.48550/arxiv.2404.19737

Yang, Z., Dai, Z., Salakhutdinov, R., & Cohen, W. W. (2017). *Breaking the softmax bottleneck: A high-rank RNN language model*. arXiv. https://doi.org/10.48550/arxiv.1711.03953

Yu, T., Kumar, S., Gupta, A., Levine, S., Hausman, K., & Finn, C. (2020). *Gradient surgery for multi-task learning*. arXiv. https://doi.org/10.48550/arxiv.2001.06782
