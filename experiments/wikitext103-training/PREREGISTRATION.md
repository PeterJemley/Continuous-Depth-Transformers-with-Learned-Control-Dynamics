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

- (none yet)
