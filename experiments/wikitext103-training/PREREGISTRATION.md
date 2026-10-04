# Preregistration: the paper's model, properly trained on WikiText-103

Status: **approved by the author on 2026-10-03, final**. Nothing below had been run when this version was committed. Runs use free Colab, so jobs save checkpoints to Google Drive and resume across sessions. From here on, the only changes are dated entries in the deviation log at the end.

Type: confirmatory for S1 to S3 exactly as stated. Everything else is exploratory and labelled so.

## Why this study

The paper's experiments, and the replication that reruns them, train for 500 steps on WikiText-2. That is about 43% of a small dataset, seen once, and the loss is still falling when training stops (see `experiments/wikitext2-replication/KNOWN_LIMITATIONS.md`, limitations 1 and 2). So their comparisons describe early training, not the architecture.

This study asks the paper's main architectural questions of models that are actually trained. It keeps the paper's model size, so model scale stays a separate, later question.

## The objects under test

**Model A and its baseline**, from the same verbatim code as the replication (`experiments/wikitext2-replication/paper_model.py`, restored from commit `6b65d64`).

| Setting | Value | Change from the paper |
|---|---|---|
| Layers, width, heads | 6, 256, 4; the ODE block replaces layers 2-3 | none |
| Vocabulary | GPT-2, 50,257 tokens, pinned files in `data/gpt2` | none |
| Sequence length | 128 | was 64; 128 is the longest the model supports, and trains faster per token |
| Batch | 32 sequences, 4,096 tokens per step | none |
| Optimiser | AdamW, default betas and weight decay, gradient clipping at 1.0 | none |
| Learning rate | warm-up over 500 steps, then cosine decay to 10% of the peak; the peak is chosen per model by the sweep below | was a constant 3e-4 |
| Precision | fp16 mixed precision with loss scaling | was fp32; the connection test showed finite losses and 1.5 to 2 times the speed |
| Training length | one full pass over WikiText-103's training tokens, about 28,800 steps | was 500 steps on WikiText-2 |

## Data

WikiText-103 raw, the Hugging Face revision and file fingerprints recorded in `data/MANIFEST.json`. Every job checks the four files' SHA-256 before use and stops on any mismatch.

- **Training text:** all non-empty rows joined with newlines, exactly as the replication joined WikiText-2, then encoded with the pinned GPT-2 tokenizer. The first job records the exact token count; every later job must reproduce it. The connection test estimated about 118 million tokens.
- **Validation:** consecutive length-128 windows over the whole validation split. It is byte-identical to WikiText-2's validation split. It is used for learning-rate selection and for monitoring.
- **Test:** consecutive length-128 windows over the whole test split, which nothing in this repository has used yet. It is evaluated once per final model, after all main runs finish, and gives the primary outcome.

## Runs

All runs go through `colab/runner.ipynb` on a Colab T4 GPU. Each job works through a queue of runs, saves checkpoints to Google Drive, and resumes after a disconnect.

**Stage 1, learning-rate sweep (selection, not a test).** For each model separately, three peak learning rates, 3e-4, 6e-4 and 1.2e-3, each trained for 10% of the main budget with seed 100. The peak with the lowest validation loss is chosen for that model. Both models get exactly this procedure. One more short run per model, at its chosen rate with seed 101, gives a first estimate of seed-to-seed spread. Estimated GPU time: about 1.2 hours.

**Stage 2, main runs.** 6 seeds, 1001 to 1006, each training both models at their chosen rates for the full budget. With 6 pairs, the exact paired sign-flip test can reach p = 0.031; with 5 it could not go below 0.0625. Estimated GPU time: about 10 hours, at the measured 0.7 hours per baseline run and 1.0 hour per Model A run.

**Stopping rule.** Exactly these runs. A crashed run resumes from its last checkpoint, or reruns with the same seed if no checkpoint exists, and is logged. Seeds are added only through an amendment made before any Stage 2 test loss is seen.

**Before Stage 2.** The Stage 1 spread estimate gives the smallest effect 6 seeds can detect. If that exceeds the 0.02-nat margin in S1, this is recorded in the deviation log, together with any decision to add seeds, before Stage 2 starts.

## Hypotheses, predictions, and what would refute them

### S1. A properly trained Model A matches or beats the baseline

- **Paper's claim:** comparable loss, slightly better, with fewer parameters.
- **Primary outcome:** test loss after the full budget, paired by seed, as Model A minus baseline, in nats.
- **Supported, better:** the t-based 95% interval of the mean difference lies wholly below 0.
- **Supported, comparable:** otherwise, the interval lies within ±0.02 nats. 0.02 is the gap the paper itself reported.
- **Refuted, worse:** otherwise, the interval lies wholly above 0.
- **Inconclusive:** every other outcome. It is reported as not enough runs to tell.
- **Reported alongside:** the exact sign-flip p-value, the unpaired Welch interval, every seed's value, and the smallest effect detectable with 80% power at the observed spread.
- **Assumption:** pairing assumes each seed's two runs share data order. They do by construction, since batches come from a generator seeded by the run seed alone.

### S2. α settles in a conservative range

- **Paper's claim:** α moves from 0.1 to about 0.065. No saved output contains that value.
- **Prediction:** the mean final α across the 6 seeds lies within [0.045, 0.085], and every seed's lies within [0.035, 0.095].
- **Refuted if:** either condition fails. All values are reported, with their path over training.

### S3. The trained Model A behaves as a continuous ODE

- **Test:** the solver-dependence test of Ott et al. (2020), on each final Model A. Test loss is computed with the training solver, 4 Euler steps, and with 8 and 16 Euler steps, 4 steps of RK4, and dopri5 at tolerance 1e-5.
- **Prediction:** the relative change in test loss stays below 1% for every more accurate solver, in every seed.
- **Must-fail controls:** the same test with α multiplied by 2, 4, 8 and 16, on the first 512 test windows. If the test never fails even at 16 times the learned α, S3 is reported as uninformative instead of supported.
- **Refuted if:** any seed shows a change of 1% or more at the learned α.

## Secondary measures, labelled exploratory

- **Convergence.** Validation loss every 2,000 steps. Training counts as converged if validation loss improved by less than 0.01 nats over the last 10% of steps. Runs that don't converge are reported, and limit the conclusions.
- **Training cost.** The ratio of Model A's to the baseline's training time per run on the T4. The paper claims latency parity, but that claim concerned inference.
- **Comparison with the replication.** Validation loss at 500 steps, logged so the WikiText-2 replication's early-training comparison can be set beside the trained one.

## Sanity checks before trusting any result

- Every data file's fingerprint matches the manifest, and the training token count matches the first job's.
- One training batch and one test window are decoded and inspected as text.
- Model A's forward pass through the solver-switching code equals its normal forward pass exactly. The replication's `hybrid_forward` check is reused.
- No training batch is drawn from the validation or test splits. They come from separate files.

## Analysis and reporting

Every number comes from code. Each run writes a JSON record with the code commit, configuration, seed, data fingerprints, package versions and GPU. One analysis script reads only those files and writes the report, including a provenance table. The primary outcome is reported first, whatever it shows. The three hypotheses are reported separately, so there is no multiple-comparison correction.

**Limitations known now.** One model size; one dataset; one pass over the data; and tuning of the learning rate only, not of other settings. The results apply to a properly trained model of the paper's size, not to larger models.

## References

Ott, K., Katiyar, P., Hennig, P., et al. (2020). *ResNet after all? Neural ODEs and their numerical solution*. arXiv. https://doi.org/10.48550/arxiv.2007.15386

Merity, S., Xiong, C., Bradbury, J., & Socher, R. (2016). Pointer sentinel mixture models. arXiv:1609.07843.

## Deviation log

All entries below were made on 2026-10-03, while building the code and before any real run. Results seen before the change: no, for every entry. None changes a prediction, threshold, or decision rule.

- **Encoding in pieces.** WikiText-103's training text is too large to encode as one string within free Colab's memory. It is encoded in pieces, cut only before rows that start with one space followed by a non-space character, where the cut cannot change the tokens. On WikiText-2 this reproduced the whole-string encoding exactly for all three splits. The validation and test splits are small and are encoded whole.
- **fp16 gradient check.** Before any training, both models' gradients on one batch are computed in fp32 and in fp16 mixed precision. Training stops if their cosine similarity is 0.99 or lower, so the precision choice can't silently distort training.
- **Evaluation precision.** Validation and test losses are computed in fp32, even though training uses fp16.
- **Resuming.** Each run saves its model, optimiser, loss scaler and random states to Google Drive about every 10 minutes, and resumes from there after a disconnect. In a local test, a run crashed at its midpoint and resumed finished with exactly the same weights and validation loss as an uninterrupted run.
- **Validation schedule.** Validation loss is recorded at step 500, every 2,000 steps, and at the end of every run.

### Stage 1 record and amendment, 2026-10-03

Results seen before this entry: the stage 1 learning-rate sweep and spread runs only, which are selection steps. No stage 2 run had started, and no test-set loss had been computed.

- **Stage 1 record.** WikiText-103's training text is 117,920,140 tokens, so a main run is 28,789 steps. The fp16 gradient check passed, with cosine similarity 0.9989 for both models. Sweep validation losses after 2,879 steps:

  | Learning rate | Baseline | Model A |
  |---|---|---|
  | 0.0003 | 5.7034 | 5.7730 |
  | 0.0006 | 5.2752 | 5.4990 |
  | 0.0012 | 4.9498 | 5.6293 |

  The baseline chose 0.0012 and Model A chose 0.0006. Source: `colab-results` branch, `wt103_study/20261003-201639`.
- **Spread estimate.** Model A minus baseline was 0.5492 at seed 100 and 0.5570 at seed 101. That gives a paired standard deviation of about 0.0055 nats, and an estimated smallest detectable effect for 6 pairs of about (2.571 + 0.920) × 0.0055 / √6 ≈ 0.008 nats. That is below the 0.02 margin, so no seeds are added. The estimate has one degree of freedom and comes from short runs, so it is very rough.
- **Amendment, approved by the author.** The baseline's best learning rate was the largest in the grid, so its true optimum may lie beyond it. That would leave the baseline under-tuned and favour Model A. Both models' grids are therefore extended with 0.0024, using the same procedure: one more sweep run each, at seed 100 and 2,879 steps. The best of the four rates is then chosen per model, and a seed-101 spread run is added for any model whose choice changes. If 0.0024 is chosen and is again the largest rate tried, it is kept, the grid is not extended further, and the boundary is reported as a limitation. This was decided before any stage 2 run. The decision rules for S1 to S3 are unchanged.

### Stage 1 rerun with the wider grid, 2026-10-03

Results seen before this entry: the stage 1 sweep and spread runs only. No stage 2 run had started, and no test-set loss had been computed.

- **Sweep results with 0.0024 added.** The baseline scored 4.7720 at 0.0024, and its choice moved there. Model A scored 6.2437 at 0.0024, and its choice stayed at 0.0006. Source: `colab-results` branch, `wt103_study/20261003-202835`.
- **Boundary rule applied.** 0.0024 is again the largest rate tried for the baseline. As fixed in the amendment above, it is kept, the grid is not extended further, and this is reported as a limitation: the baseline's optimum may lie above 0.0024. If so, the baseline is still under-tuned, which favours Model A in S1.
- **Final rates for stage 2.** Baseline 0.0024; Model A 0.0006.
- **Updated spread estimate.** Model A minus baseline was 0.7270 at seed 100 and 0.7388 at seed 101. That gives a paired standard deviation of about 0.0083 nats, and an estimated smallest detectable effect for 6 pairs of about 0.012 nats. That is below the 0.02 margin, so no seeds are added.


### Stage 2 resume fix, 2026-10-03

A code fix only. The procedure, the decision rules and the results seen are unchanged.

- **What happened.** The first stage 2 session finished `main_baseline_seed1001` (final validation loss 3.8518) and was stopped at about step 28,200 of `main_hybrid_seed1001`. The next session failed while resuming that run, with `TypeError: RNG state must be a torch.ByteTensor`. Source: `colab-results` branch, `wt103_study/20261003-205057` and `wt103_study/20261003-223727`.
- **Cause.** Checkpoints were loaded with `map_location` set to the GPU, which moved the saved random-number states onto the GPU, and PyTorch accepts them only on the CPU. Earlier resume tests ran on a CPU, so they never hit this.
- **Fix.** Checkpoints are now loaded to the CPU; the model and optimiser move their own tensors to the GPU when loaded. A test that interrupts a run part-way and resumes it gives exactly the same validation curve and training losses as an uninterrupted run. The interrupted run resumes from its last checkpoint, so the resumed run is the same run that would have completed without the interruption.
- **Upload message.** The runner now reports when the job stopped with an error, instead of printing "Done" after any upload.

### Stage 2 record, 2026-10-04

Results seen before this entry: training outputs only, meaning validation losses, α and training logs. No test-set loss had been computed.

- **Completion.** All 12 main runs finished the full 28,789 steps on a Tesla T4, with no Drive failures. `main_hybrid_seed1001` resumed once, after the fix above; every other run ran without interruption. fp16 skipped between 7 and 17 steps per run. Source: `colab-results` branch, `wt103_study/20261003-231315`, code `b304104`.
- **Final validation losses, monitoring only.** These are not the primary outcome, which is test loss in stage 3.

  | Seed | Baseline | Model A |
  |---|---|---|
  | 1001 | 3.8518 | 4.8792 |
  | 1002 | 3.8549 | 4.8457 |
  | 1003 | 3.8514 | 4.8942 |
  | 1004 | 3.8541 | 5.0832 |
  | 1005 | 3.8501 | 5.1396 |
  | 1006 | 3.8491 | 4.8750 |

- **Final α, S2's data.** 2.1402, 1.8146, 1.3254, 1.0033, 1.3318 and 1.0439 for seeds 1001 to 1006. S2 is decided on these by its fixed rule.
- **Convergence, exploratory.** Validation loss improved by 0.034 to 0.050 nats over the last 10% of steps in every run of both models, so by the rule above no run counts as converged. Both models were still improving when the budget ended. This is reported as a limitation.
- **Training cost, exploratory.** About 1.11 hours per Model A run against 0.75 hours per baseline run, a ratio of about 1.48.

### Stage 3 safeguard, 2026-10-04

Decided before any test loss was computed. The decision rules are unchanged.

- **Why.** The learned α is 1.0 to 2.1, far above the 0.1 it started at. The must-fail controls multiply it by up to 16, so dopri5 could need an unbounded number of steps and stall the job.
- **Change.** Each dopri5 solve may take at most 1,000 adaptive steps. A solve that needs more stops with an error, which is recorded for that solver and setting, and the job moves on. The other solvers are unaffected. If the cap is hit at the learned α in any seed, S3 is reported as untestable for that seed rather than supported. Stage 3 also logs each solver's progress and time.
- **Checked.** In test mode, a cap of 1 step produced recorded errors and the job completed; a cap of 1,000 let every solve finish.
