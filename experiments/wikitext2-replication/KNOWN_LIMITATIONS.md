# Known limitations of the original experiments

Written on 2026-10-03, while the replication was running and before any result was analysed. These limitations describe how the paper's experiments were run, not how they turned out, so they don't depend on the replication's results. They are fixed here so the results can't shape what gets called a limitation.

What had been seen of the replication when this was written: training-progress lines for seed 42, and the seed-42 baseline's validation loss. No hypothesis had been evaluated, and `analyze.py` had not been run on real results.

The paper's own section 5.4 lists model size, the fixed Euler steps, and the single control dimension. It does not mention any of the five below.

## 1. One small dataset

Every language-model result comes from WikiText-2 alone: 2,391,884 GPT-2 tokens of training text. No second dataset or domain was tested.

## 2. Training stopped well before convergence

- **Less than one pass.** Training ran for 500 steps of 32 sequences of 64 tokens, which is 1,024,000 tokens, or about 43% of the training text, seen once.
- **Still improving.** The original run's loss was still falling when it stopped. The baseline logged 6.5176 at step 400 and 6.4504 at step 450 (commit `6b65d64`, notebook cell 16).
- **Mostly vocabulary.** About 84% of the "30M-parameter" model is its vocabulary tables: 25.7M of 30.5M parameters. The transformer layers under comparison hold about 4.7M.
- **Consequence.** The loss comparison measures early training speed, not what either architecture can reach. The learned α is a value from an unfinished run, not a converged one. Steering, solver-effort, and probe results describe a barely trained model.

Training longer on WikiText-2 would not fix this, because a model this size would start to memorise so small a dataset. Testing the architecture itself needs either careful multi-pass training with validation-based stopping, or a larger corpus.

## 3. A single seed

Each model was trained once, with seed 42 (commit `6b65d64`). Run-to-run variation was never measured, so no difference between models can be told apart from seed noise.

## 4. Steering trained and tested on one prompt

- **One prompt, two examples.** Steering was trained on "The movie was", with two targets: one for u = +1 and one for u = −1.
- **Same prompt for testing.** It was evaluated on that same prompt, so the result shows the two examples were learned, not that steering generalises.
- **Wrong target.** The "good" target was token 220, which is a bare space. " good" is token 922.
- **Unconstrained neutral.** The neutral setting, u = 0, was never trained, so its output carries no meaning. The paper's "dynamics of resistance" argument rests on it.

## 5. Results drawn from two different architectures

- **Model A.** The paper's training, steering, and solver-probe results come from a model whose ODE block uses attention and adds the control signal to the hidden state (commit `6b65d64`). This model was trained on WikiText-2.
- **The other model.** The 0.068% solver divergence, the α of about 0.068, the linear probes, and the latency parity come from a different model, whose ODE block is a small network without attention that concatenates the control signal. It was trained on random tokens: 50 steps in the falsification notebook, and untrained in the benchmark notebook.
- **What the paper says.** It describes the second model's design in its architecture section and presents all results as one model's.

## What the replication can and cannot address

| Limitation | Addressed by this replication? |
|---|---|
| 1. One small dataset | No. It uses the paper's dataset by design. |
| 2. Training stopped early | No. It uses the paper's 500 steps by design. |
| 3. A single seed | Yes, for every claim it retests: 8 seeds per model. |
| 4. Steering on one prompt | Partly. It adds 20 held-out prompts and the correct token, but steering still trains on the one prompt. |
| 5. Two architectures | Partly. It retests Model A's claims, including the solver test, latency, and probes. The paper's numbers that came from the other model are not reproduced. They would need to be removed or replaced. |

Limitations 1 and 2 bound every conclusion of this replication. Its results apply to the paper's setup as run, not to the architecture at a properly trained scale.
