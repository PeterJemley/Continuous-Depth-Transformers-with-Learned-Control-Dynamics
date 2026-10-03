# Preregistration: WikiText-2 replication of the paper's experiments

Status: **approved by the author on 2026-10-03, final**. Nothing below had been run when this version was committed, and its git timestamp records that it came first. From here on, the only changes are dated entries in the deviation log at the end.

Type: confirmatory for hypotheses H1 to H8 exactly as stated. Anything else is exploratory and is labelled so.

## Why this replication

An audit of the repository found that the paper's main results come from one run of a notebook that was later deleted (git commit `6b65d64`). That run used one seed, trained loss only, and a steering target token that is a space rather than " good". Other headline numbers come from a different model trained on random tokens. This replication reruns the paper's own model on the paper's own data, with enough seeds and held-out evaluation, and states in advance what would count against each claim. A claim that fails here is a result, and is reported as plainly as one that holds.

## The object under test

**Model A**, the model behind the paper's training, steering, and solver-probe results. The code is restored verbatim from commit `6b65d64`, notebook cells 4 and 6. It has an ODE block with attention plus an MLP, the control signal added to the hidden state inside the vector field, learned output scale α initialised to 0.1, 4 fixed Euler steps, and the adjoint method.

| Setting | Value |
|---|---|
| Layers, width, heads | 6, 256, 4; the ODE block replaces layers 2-3 |
| Vocabulary | GPT-2, 50,257 tokens, from the pinned files in `data/gpt2` |
| Context and dropout | 128 maximum, 0.1 |
| Training | 500 steps, batch 32, sequence length 64, AdamW learning rate 3e-4, gradient clipping at 1.0 |
| Baseline | The same code's 6-layer discrete transformer, same settings |

The model in today's `model.py` is a different architecture, and it is out of scope here. Its claims could be a second round.

## Data

WikiText-2 raw, version 1, loaded only through `pinned_data.py`. Every file is checked against `data/MANIFEST.json`. Training text is joined exactly as the original notebook joined it. The check is that the train split must give 2,391,884 GPT-2 tokens.

- **Training batches** are sampled from the train split, as in the original code.
- **Validation loss** uses the whole validation split, cut into consecutive length-64 windows. It is computed once, at the end of training.
- **The test split** is not used in this round.

**Prerequisite:** WikiText-2 must be pinned first. Its download hosts are blocked in the current cloud environment.

## Runs and seeds

8 seeds for each model: 42, as in the original run, then 0 to 6. Each seed trains the baseline and Model A from the same seed. With 8 pairs, an exact paired sign-flip test can reach p = 0.0078. With 5 pairs it could never go below 0.0625.

| Block | Runs | Estimated CPU time |
|---|---|---|
| Training, both models | 16 runs of 500 steps | about 4 hours |
| Validation loss | 16 evaluations | about 30 minutes |
| Steering, two target tokens | 16 runs of 500 small steps | about 15 minutes |
| Solver tests, probes, latency | single passes | about 30 minutes |

**Stopping rule.** Exactly these runs. A crashed run is rerun once with the same seed and logged. No seeds are added after results are seen, unless this file is amended first.

## Hypotheses, predictions, and what would refute them

Each hypothesis states the paper's claim as a prediction. The decision rule is fixed now.

### H1. Training is stable

- **Paper's claim:** zero vanishing or exploding gradient events.
- **Prediction:** in all 8 Model A runs, the ODE block's gradient norm stays between 1e-6 and 100 at every step.
- **Must-fail control:** the same detector must flag a deliberately unstable run, with α initialised at 10 and learning rate 1e-1 for 50 steps. If it flags nothing there, H1 is reported as untestable with this detector.
- **Refuted if:** any run has an event.

### H2. Loss: comparable to, or slightly better than, the baseline

- **Paper's claim:** final loss 6.449 against 6.471.
- **Primary outcome:** validation loss after 500 steps, paired by seed. The difference is Model A minus baseline.
- **"Slightly better" is supported if** the mean difference is below 0 and the two-sided exact sign-flip p-value is below 0.05.
- **"Comparable" is supported if** the t-based 95% interval of the mean paired difference lies within ±0.05 nats. A bootstrap interval is not used, because with 8 runs it comes out too narrow.
- **Refuted if:** neither holds. If the interval lies wholly above 0, Model A is reported as worse.
- **Inconclusive** covers every other outcome, for example an interval that includes 0 but is wider than ±0.05. It is reported as "not enough runs to tell", not as support.
- **Assumption:** the paired analysis assumes each seed's two runs share initialisation draws and batch order. That holds by construction here, and the unpaired result is reported alongside it.
- **Secondary:** mean training loss over the last 50 steps, the paper's own measure, for comparison.
- **Detectable effect, stated in advance:** the paper's gap is 0.02 nats. With 8 pairs, the smallest effect detectable with 80% power is about 3.3 × the paired standard deviation ÷ √8, which is about 1.15 times the paired standard deviation. If that standard deviation is near 0.05 nats, a true gap of 0.02 cannot be detected, and an inconclusive "slightly better" result is expected. That outcome would not be a failed replication. The measured standard deviation and detectable effect are reported with the result.

### H3. α converges to about 0.065

- **Paper's claim:** α moves from 0.1 to 0.065. No saved output contains this value.
- **Prediction:** the mean final α across 8 seeds lies within [0.055, 0.075], and every seed lies within [0.045, 0.085].
- **Refuted if:** either condition fails. All 8 values are reported, along with their spread over training.

### H4. The trained model behaves as a continuous ODE, not a fixed-step network

- **Paper's claim:** solver divergence of 0.068%. That figure came from a different model trained on random tokens.
- **Test:** the solver-dependence test of Ott et al. (2020), cited as prior work. Validation loss is computed with the training solver, 4 Euler steps, and again with 8 and 16 Euler steps, 4 steps of RK4, and dopri5 at a tolerance of 1e-5.
- **Prediction:** the relative change in validation loss stays below 1% for every more accurate solver, in every seed.
- **Second measure:** hidden-state divergence relative to how far the block moves the state.
- **Must-fail controls:**
  - The same test on each untrained model.
  - The test on trained models with α multiplied by 2, 4, 8 and 16, to find where it starts failing.
  - If the test never fails even at 16 times the learned α, it is reported as uninformative instead of passed.
- **Refuted if:** any seed shows a change of 1% or more at the learned α.

### H5. The control signal steers sentiment

The paper's three-phase procedure runs on each trained Model A, with `" good"` as token 922 and `" bad"` as token 2089. It is also rerun with the paper's token 220 to reproduce its numbers.

- **H5a, on the training prompt.** Prediction: P(" good") > P(" bad") at u = +1, and the reverse at u = −1, in at least 7 of 8 seeds. The paper's figures were 98% and 88%. Refuted if fewer than 7 seeds succeed.
- **H5b, on held-out prompts.** Prediction: semantic steering transfers to 20 prompts of the form "The X was" that steering never saw. X is film, book, meal, show, concert, game, hotel, flight, album, play, story, class, trip, party, lecture, restaurant, service, product, series, or performance. That gives 40 prompt-and-sign pairs per seed.
  - **Supported if** the mean share of correctly ordered pairs is at least 75%, and the pooled count is above 50% by an exact binomial test at p < 0.05.
  - **Refuted if** the 95% Wilson interval for the pooled share includes 50%.
  - **Inconclusive** if the share is above chance but below 75%. That would show partial transfer, which is reported as such.
- **H5c, the neutral bias.** Prediction, from the paper's "dynamics of resistance" argument: P(" good") > P(" bad") at u = 0 in at least 7 of 8 seeds. Refuted otherwise.
- **H5d, the path check.** With the control input removed, the steering effect on the training prompt must disappear. If it doesn't, the effect is not acting through u, and the run is investigated as a bug.
- **Secondary:** validation loss before and after steering, which measures what steering costs the language model.

### H6. Negative steering needs more solver effort

- **Paper's claim:** 20 against 14 function evaluations, which the paper reports as 43%. The notebook itself computed 36%.
- **Test:** on each steered model, sweep u over 21 values from −2 to 2, with u set on all four dimensions as in the paper. Count dopri5 evaluations at a tolerance of 1e-3, and report them as solver steps, since each step is 6 evaluations.
- **Prediction:** mean evaluations for u < 0.3 exceed those for u > 0.4 in at least 7 of 8 seeds.
- **Refuted if:** fewer than 7 seeds show it.

### H7. Latency parity

- **Paper's claim:** 0.98 times the baseline's time. That came from the other model, measured on a CPU.
- **Test:** inference with batch 32 and length 128. 5 warm-up passes, then 30 timed passes, repeated in 3 separate processes on this machine's CPU. No GPU is available, so the result covers this CPU only.
- **Prediction:** the median ratio of Model A's time to the baseline's is at most 1.05 in all 3 processes.
- **Refuted if:** the ratio exceeds 1.05 in any process.

### H8. Linear probes show flat separability across depth

- **Paper's claim:** no rise in separability at τ ≈ 0.67.
- **Test:** on each steered model, logistic-regression probes predict the sign of u from the mean hidden state at τ = 0, 0.5, 0.67, 0.8 and 1.0. The inputs are 1,024 validation windows, half for training and half for testing, with u = ±1 assigned at random.
- **Controls:** the same probe trained on shuffled labels, and the same probe on untrained models.
- **Built-in must-fail check:** H(0) is computed before the control signal enters, so accuracy at τ = 0 must be at chance. If it isn't, there is a leak or a bug.
- **Prediction:** probe accuracy above the control varies by less than 5 points across τ > 0.
- **Refuted if:** it varies by 5 points or more.

## Reproduction check of the original run

Seed 42 reproduces the original run's model initialisation and batch order, since both use the CPU random number generator. Dropout masks differ, because the original ran on a GPU.

- **Prediction:** the step-0 training loss matches the original's 11.0107 for the baseline and 10.9800 for Model A, within 0.05.
- **Prediction:** the last-50-step mean matches its 6.4707 and 6.4496 within 0.1.
- **If it misses,** the data pipeline differs from the original, and that is investigated before H1 to H8 are interpreted.

## Sanity checks before trusting any result

- Decode every hard-coded token ID and print it: " good", " bad", and each held-out prompt.
- Confirm the train split gives 2,391,884 tokens.
- Inspect one training batch as text.
- Run `chance_check.py` from the experimentation skill for every probe and every steering test set size, before reading the results.

## Analysis and reporting

- **Every number comes from code.** Each run writes a JSON manifest with the commit, configuration, seed, data checksums, package versions, and hardware, plus its metrics. A single analysis script reads only those files and writes the results report, including a provenance table. No number is typed by hand.
- **Primary outcomes are reported first** for every hypothesis, whatever they show, with every seed's value.
- **Multiple comparisons:** the eight hypotheses are reported separately. Any claim combining several of them uses a Holm correction.

## References

Ott, K., Katiyar, P., Hennig, P., et al. (2020). *ResNet after all? Neural ODEs and their numerical solution*. arXiv. https://doi.org/10.48550/arxiv.2007.15386. Also published at ICLR 2021.

Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2016). Pointer sentinel mixture models. arXiv:1609.07843.

## Deviation log

- (none yet)
