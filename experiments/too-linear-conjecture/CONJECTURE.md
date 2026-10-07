# Conjecture note: is the hybrid model "too linear"?

Date: 2026-10-04    Status: conjecture, partly tested (updated 2026-10-07). T0 and T1 have run; T2–T4 have not.

**T0 result, 2026-10-04: untestable.** A must-match control fell short in five of six models, and the dropout noise reference was too low, so the gradient comparison could give no verdict under its own rules. See `T0_NOTE.md`.

**T1 result, 2026-10-04: rival G confirmed from the training side; P and C2 refuted at this scale.** Training Model A with adjoint gradients cost +0.79 nats (95% interval [+0.66, +0.92]), about 95% of the gap to the baseline at 10% of the budget. Losing one block's separate weights cost only +0.02 (rival P refuted). A weight-tied repeated block matched the two separate blocks it replaced, +0.007 (C2's forbidden case, so C2 is refuted at this scale). Trained with exact gradients, Model A was 0.04 nats behind the baseline. See `T1_NOTE.md`. C1 and C3 remain untested; the work moved on to the simultaneity conjecture (`../simultaneity-conjecture/`).

Written with the scientific-experimentation skill (conjecture-note template). Nothing here is a finding. The refutations recorded in `experiments/wikitext2-replication/RESULTS.md` and `experiments/wikitext103-training/RESULTS.md` stand whatever the tests below show. These conjectures are new explanations to be criticised, not ways to restore the paper's claims.

## The question

As asked: is this project "too linear", and would it benefit from a modular reconfiguration in which its modules interact through a design that can itself be modulated?

Restated against the code: does the way information and control move through Model A (`experiments/wikitext2-replication/paper_model.py`, the paper's model, restored verbatim) limit what the model can learn or how well it can be steered, and would input- or control-dependent interaction between its parts remove that limit?

## Readings

"Too linear" has at least three different readings in this code. They make different predictions, so they are kept apart.

| Reading | Where it shows in the code | What it predicts |
|---|---|---|
| R1. The control enters linearly | `ODEFunc.forward`: `c_emb = control_embed(u)`, one linear map, added to the state as `h = x + t_emb + c_emb`, with the same value at every depth step and every position. It then passes through LayerNorm before attention and the MLP, which removes its mean and shrinks it relative to the state. | u acts only by shifting what the vector field sees. Through the nonlinear layers a shift can in principle change any computation, so this is a claim about what training finds easily, not about what the model can represent. If shifting is a hard lever to learn with, steering should be weak, inconsistent across seeds, and costly to language-model loss. |
| R2. Depth is one fixed, repeated function | The ODE block applies one set of attention and MLP weights four times (Euler steps of size α/4), and the only difference between steps is another additive vector, `t_emb`. It replaces two blocks with separate weights. | The block cannot represent the different functions two separate layers learn, so language-model loss suffers even at matched compute. Making each step's computation depth-dependent in a stronger way than an added vector should help. |
| R3. Modules interact in a fixed order | Embedding, then 2 blocks, then the ODE block, then 2 blocks, the same path for every token and every control value. No part's behaviour depends on another's output except through the shared residual stream. | Input- or control-dependent routing among parts would let different inputs, or different control values, use different computations. |

A fourth reading, that the research process was too linear, is about method rather than the model. It is answered by the two preregistered studies already done and is not pursued here.

## What the evidence already says

These results are criticism already in hand.

- **The control is present but weak in effect** (bears on R1). In the WikiText-2 replication, the sign of u is decodable from the hidden state almost perfectly at every depth inside the block (probe accuracy 0.93–1.00 at τ ≥ 0.5). Yet steering toward " good" was correct on only 211 of 320 held-out prompts (66%), varied from 20/40 to 40/40 across seeds, and raised validation loss by 0.03–0.10 in every seed. Information about u reaches the state, but redirects the computation poorly. That fits R1, but it does not single R1 out: see the rivals below.
- **The block works hard when properly trained** (bears on R2). On WikiText-103, α grew from 0.1 to between 1.0 and 2.1, and Model A finished 1.10 nats worse than the baseline on test loss (95% interval 0.97 to 1.23, all 6 seeds). It also became measurably dependent on its 4-step solver (1.1–1.8% change in test loss with dopri5). A single repeated function being pushed to do the work of two layers is one explanation. It is not the only one.
- **Model A has fewer separate weights.** Its block parameters equal 5 standard blocks plus small embeddings, against the baseline's 6 (29.78 M against 30.50 M parameters in total). Losing one block is unlikely to cost 1.1 nats at this size, but that is itself a conjecture, and it is cheap to test.
- **The control plays no part in language-model training.** During training `control` is `None`, so `c_emb` is zero. The control is attached afterwards, by 500 steps of steering training on a single prompt, "The movie was" (`run_steering.py`). That alone could explain weak, inconsistent steering whatever the injection form.

## Conjectures

**C1. Additive control is too weak a lever.** Because u only shifts the vector field's input, partly removed by LayerNorm, training finds it hard to make u reconfigure the computation, even though a large enough shift could in principle do so through the nonlinear layers. A different, nonlinear embedding of u would not help either: steering uses only u = ±(1, 0, 0, 0), so any embedding of u reduces to two fixed vectors. Consequence: replacing the additive shift with multiplicative, depth-dependent modulation (feature-wise scale and shift of the normalised state, γ(u, τ) and β(u, τ), as in FiLM) gives better held-out steering at an equal or lower cost in language-model loss, with less variation across seeds. It forbids this: additive conditioning applied at every depth step, at matched parameters, doing as well as the multiplicative version. If that happens, the lever was where u enters, not the fact that it adds.

**C2. One repeated function is too rigid for the block's job.** Because every Euler step applies the same weights, differing only by an added time vector, the block can't represent what two separate layers do. Consequence: letting the computation vary with depth and state, through τ-dependent modulation or a router choosing among K smaller vector fields at matched parameters, closes a substantial share of the language-model gap. It forbids this: a plain weight-tied discrete block (the same 4 steps, no ODE machinery, exact gradients) already matching a two-layer baseline. If that happens, rigidity of the repeated function is not what costs the loss.

**C3. Interaction designed to be modulated: the full conjecture.** If the parts interact through a routing or coupling that is itself a function of u and τ, for example router weights produced by a small network from u, then different control values select different computations rather than shifted inputs. Consequence: steering transfers to held-out prompts and is consistent across seeds, without the language-model cost that C1's variant pays. It forbids this: C3 doing no better than the best of C1 and C2 at matched parameters. If that happens, modulating the interaction adds nothing beyond its parts. C3 is the boldest conjecture and is tested last, because without C1 and C2 its result couldn't say which part did the work.

## Rivals

| Rival explanation | What it predicts that differs | Cheapest test that tells them apart |
|---|---|---|
| G. The gradients were wrong. `odeint_adjoint` recomputes the vector field in the backward pass with fresh dropout masks, and the Euler adjoint is only approximate, more so as α grows (Gholami et al., 2019). | The gap is mostly an optimisation defect. Exact backpropagation, with no architectural change, closes much of it. | T0 below: measure the gradient error directly on the saved checkpoints, with no training. |
| P. Fewer separate weights. | A 5-layer baseline loses about as much as Model A. | A short 5-layer baseline run in T1. |
| S. Steering is weak because of the steering procedure: one training prompt, 500 steps, control attached after training. | The weak steering persists unchanged with multiplicative modulation unless the procedure changes, and improves with a better procedure even with additive control. | In T2, fix the procedure first, with many prompts and the control used in training, and only then compare injection forms. |
| N. Too few runs or prompts. | Differences between variants vanish with more seeds or prompts. | Preregister at least 5 seeds per variant and enough held-out prompts for a narrow chance range (`chance_check.py`). |

## Prior art

- FiLM, feature-wise modulation by a conditioning input (Perez et al., 2018). The direct form of C1.
- Neural controlled differential equations, in which the control drives the dynamics multiplicatively, not additively (Kidger et al., 2020).
- Sparsely gated mixture-of-experts, routing among sub-networks per input (Shazeer et al., 2017). One form of C2 and C3.
- Hypernetworks, one network generating another's weights (Ha et al., 2016). One form of C3's "design that can itself be modulated".
- Recurrent Independent Mechanisms, modules that interact sparingly through attention and activate selectively (Goyal et al., 2019).
- Gated Recurrent Transformers (Hegazy et al., 2026, preprint): fixed prelude and coda blocks around one shared core iterated several times, the same layout as Model A, with an elementwise gate modulating each recurrent update. The authors report that plain depth-sharing "enforce[s] uniform transformations that collapse representational diversity and degrade modeling quality". Closest prior art to C2. Not yet peer reviewed.
- Solver dependence of trained neural ODEs (Ott et al., 2020), already cited by the studies.

None of these combines a continuous-depth block with control-modulated routing between modules, as far as this search found. That is a statement about the search, not a novelty claim.

## Proposed tests, in order

Every test uses exact backpropagation for every variant, so rival G cannot confound C1 to C3. Budgets, tuning and seeds are matched within each test, and each gets its own preregistration before it runs.

**T0. The gradient diagnostic, with no training (rival G).** On the six WikiText-103 Model A checkpoints, compute the gradient on fixed batches four ways: (a) the training configuration, adjoint with dropout; (b) adjoint with dropout off; (c) exact backpropagation through the 4 Euler steps with the same dropout mask as the forward pass; (d) exact backpropagation with a second, independent mask. Primary outcome: the cosine similarity of (a) with (c), compared with the dropout-only baseline, (c) with (d).
- Must match: with dropout off and α scaled down to 0.1, the adjoint and exact gradients agree (cosine above 0.99).
- Must fail: a deliberately corrupted gradient (random sign flips) gives a cosine near 0.
- Cost: minutes on one GPU, no training.

**T1. The short ablation ladder (rivals G and P, conjecture C2's forbidden case).** At 10% of the WikiText-103 budget (2,879 steps), with 5 seeds each:
- the 6-layer baseline;
- a 5-layer baseline;
- a weight-tied 4-step discrete block in place of layers 2–3;
- Model A trained with exact gradients;
- Model A as trained before.

Primary outcome: validation loss. The gap between neighbouring rungs says where the 1.1 nats lives. About 2.5 GPU-hours on a T4.

**T2. Control injection (C1, rival S).** First fix the steering procedure: many labelled prompts from a pinned sentiment dataset, and the control used during training. Then compare three variants at matched parameters:
- additive once (the current design);
- additive at every Euler step;
- FiLM at every Euler step.

Primary outcome: held-out steering accuracy on a preregistered prompt set large enough for a narrow chance range. Secondary outcomes: language-model loss cost, and seed-to-seed spread.
- Must match: FiLM with u = 0 reproduces the base model exactly.
- Must fail: training with shuffled u labels gives chance-level steering.

**T3. Depth-dependent computation (C2).** A router over K smaller vector fields, depending on τ and the state, at matched parameters and compute, against the single vector field. Primary outcome: validation loss, short budget first and full budget only if the short run passes a preregistered bar.
- Must match: K = 1 reproduces the single field.
- Must fail: a fixed random router does no better than K = 1.

**T4. Modulated interaction (C3).** Router weights or couplings generated from u and τ, against the best of T2 and T3 at matched parameters. Primary outcome: held-out steering accuracy together with language-model loss, under a decision rule fixed in advance.
- Must match: with the u pathway switched off, the model reproduces the T3 model.
- Must fail: random u gives chance-level steering.

## What each outcome would teach

| Outcome | What we would learn | Next conjecture |
|---|---|---|
| T0: the training gradients are far from exact, well beyond dropout noise | The WikiText-103 comparison measured a mis-trained Model A, not the architecture alone. The refutation of the paper's claims stands, since the claims were about the model as built. | Model A trained with exact gradients performs differently. Test it directly, as a new model with its own preregistration. |
| T0: the gradients are close to exact | Rival G is ruled out. The 1.1-nat gap belongs to the architecture or its training setup. | C2 and P become the main candidates; go to T1. |
| T1: most of the gap appears at the weight-tied rung | Repeating one function costs the loss, not the ODE machinery. Supports pursuing C2. | T3. |
| T1: most of the gap appears between exact and adjoint Model A | Confirms G from the training side as well. | Stop treating the adjoint as free; any continuous-depth design here needs exact or checkpointed gradients. |
| T1: the 5-layer baseline loses as much as Model A | Capacity, not structure, explains the gap (rival P). | Compare at matched separate weights before any modular design. |
| T2: FiLM beats additive-every-step | Multiplicativity is the lever (C1 survives its test). | T4 builds on FiLM-style modulation. |
| T2: additive-every-step matches FiLM | Where u enters matters, not how (C1's forbidden case). | Revise C1 around the injection point. |
| T2: nothing beats additive once once the procedure is fixed | The steering weakness came from the procedure (rival S), not linearity. | R1 is not a limit at this scale. Drop C1. |
| T3: the router closes a substantial share of the gap | Depth-dependent computation matters (C2 survives its test). | T4 with u added to the router. |
| T3: the router does no better than K = 1 | C2 is refuted at this scale. | Look elsewhere for the gap, such as the time embedding or the step size. |
| T4: modulated interaction beats both parts | The full conjecture survives its test, and becomes the explanation to criticise next, for example at larger scale. | Test transfer to a second control attribute. |
| T4: modulated interaction does no better than its parts | Modulating the interaction adds nothing here, and C3 is refuted. | Keep whichever of C1 and C2 survived, and drop the routing. |

## What this note does not claim

- That additive conditioning is less expressive. Through the nonlinear layers it is not; C1 and C3 are claims about what training can find, so their tests measure learned behaviour, not capacity.

- That the model is too linear. These are conjectures. C2 has since been refuted at this scale by T1; C1 and C3 remain untested.
- That any outcome would rescue the paper's claims. They were tested as stated and refuted, and those records stand. A better model found by these tests would be a new model with new evidence.
- Novelty. The prior art above covers each component. Only the combination appears untested in what was searched.

## References

Gholami, A., Keutzer, K., & Biros, G. (2019). *ANODE: Unconditionally accurate memory-efficient gradients for neural ODEs*. arXiv. https://doi.org/10.48550/arxiv.1902.10298

Goyal, A., Lamb, A., Hoffmann, J., et al. (2019). *Recurrent independent mechanisms*. arXiv. https://doi.org/10.48550/arxiv.1909.10893

Ha, D., Dai, A., & Le, Q. V. (2016). *HyperNetworks*. arXiv. https://doi.org/10.48550/arxiv.1609.09106

Hegazy, A., Alanwar, A., Elhoushi, M., et al. (2026). *Gated recurrent transformers: Expressive depth through recurrent modulation*. arXiv. https://doi.org/10.48550/arxiv.2608.15062

Kidger, P., Morrill, J., Foster, J., et al. (2020). *Neural controlled differential equations for irregular time series*. arXiv. https://doi.org/10.48550/arxiv.2005.08926

Ott, K., Katiyar, P., Hennig, P., et al. (2020). *ResNet after all? Neural ODEs and their numerical solution*. arXiv. https://doi.org/10.48550/arxiv.2007.15386

Perez, E., Strub, F., de Vries, H., et al. (2018). FiLM: Visual reasoning with a general conditioning layer. *Proceedings of the AAAI Conference on Artificial Intelligence, 32*(1). https://doi.org/10.1609/aaai.v32i1.11671

Shazeer, N., Mirhoseini, A., Maziarz, K., et al. (2017). *Outrageously large neural networks: The sparsely-gated mixture-of-experts layer*. arXiv. https://doi.org/10.48550/arxiv.1701.06538

Author lists after the third author were not verified.
