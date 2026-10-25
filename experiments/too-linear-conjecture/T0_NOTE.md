# Result note: T0

Date: 2026-10-04    Preregistration: `T0_PREREGISTRATION.md`    Generated report: `T0_RESULTS.md`

## In brief

Model A, the paper's model, was trained with the adjoint method, a memory-saving shortcut for computing its gradients. T0 asked, without training anything, whether those gradients point more than about 26° away from the exact gradients of the same computation, in at least 5 of the 6 models trained on WikiText-103. The preregistered verdict is untestable: a required agreement check fell short in five of the six models, and the measure was too noisy in four, even after averaging over 64 random dropout patterns. Every check of the pipeline itself passed, so T0 neither supports nor refutes the claim, and its two notable findings are exploratory: with dropout off, the two gradients diverge as α, a learned scale on the block's update, grows, and α's own adjoint gradient has the opposite sign to the exact one.

## The gradients and the measures

Model A replaces layers 2–3 of an ordinary six-layer transformer with a continuous block, the ODE block. Inside it, a learned function, the vector field, says how each token's state changes with depth. The block applies it in four Euler steps, each of size α/4 times the field. α is a learned scale that starts at 0.1. Dropout switches off random parts of the network during training; each random pattern is a mask. The six models are the WikiText-103 study's final Model A runs, seeds 1001 to 1006. Every gradient is computed on the same eight batches of training text.

| Gradient | How it is computed |
|---|---|
| Adjoint, or training, gradient | As in training: the adjoint method (`torchdiffeq.odeint_adjoint`) runs the block backwards to get the gradient, instead of storing its steps. On the way back it draws fresh dropout masks. |
| Exact gradient | Ordinary backpropagation through the same four Euler steps, with the same weights, batch and dropout masks as the forward pass |

- **Cosine:** how closely two gradients point the same way, from 1 (the same direction) to 0 (unrelated). A cosine of 0.90 is an angle of about 26°, and 0.98 about 11°.
- **Draws:** each gradient is averaged over many dropout masks. The plan set 16 draws, to be doubled up to 64 if the noise reference fell short.
- **Primary cosine:** the cosine between the averaged adjoint gradient and the averaged exact gradient, over the vector field's parameters (every ODE block parameter except α), averaged over the 8 batches.
- **Noise reference:** the cosine between two independent averages of exact gradients, drawn with different masks. It shows how far dropout noise alone pulls a cosine below 1.
- **Must-match control 4:** with dropout off, the time embedding (the block's signal for depth) switched off and α set to 0.001, the two gradients must agree at a cosine of at least 0.999, since the adjoint's error should vanish there.
- **The other controls:**
  - the weight check: each model's recomputed test loss matches its record;
  - repeatability: two exact gradients with the same random seed agree;
  - the shared forward pass: the adjoint and exact versions give the same loss, and the same gradients for the later layers;
  - the hand-written Euler loop: the exact gradient equals one from a four-step loop written by hand;
  - the sign-flip control: an exact gradient with signs flipped at random must give a cosine near 0, which shows the measure can come out low;
  - the injected effect: an exact gradient plus random noise of equal size must give a cosine of about 0.707.
- **Secondary measures:** S1 compares single gradients rather than averages: one adjoint gradient against one exact gradient with the same masks, and two exact gradients with different masks. S3 turns dropout off and scales α from 0.001 to 2 times its learned value. S4 repeats the primary measure on the six models freshly initialised, at α = 0.1. S8 is α's own gradient: the training (adjoint) mean, the exact mean, and a second exact mean.
- **Rival G**, in `CONJECTURE.md`, is the guess that Model A's gap with an ordinary transformer comes from wrong gradients, not from its design. **T1** is the test that followed, which trained models: AX is Model A trained with exact gradients, and AA is Model A trained with adjoint gradients. **T0b** is a possible follow-up, described at the end of this note.

**The decision rule.** The unit is the trained model, and there is no interval. First, the noise reference had to reach 0.95 in every model, with the draws doubled up to 64; if it never did, the result is "too noisy to decide". The claim is *supported* if, in at least 5 of 6 models, the primary cosine is below 0.90 and more than 0.05 below the noise reference. It is *refuted* if, in at least 5 of 6, the primary cosine is at least the lower of 0.98 and the noise reference minus 0.01. Anything else is *intermediate*. The result is *untestable* if any must-match control fails.

*The summary, this key and the records section at the end were added on 2026-10-25 so that the note can be read on its own. Nothing else was changed.*

## What we expected

The claim: at the end of WikiText-103 training, the expected adjoint gradient for the ODE block's vector field points more than about 26° away from the expected exact gradient (cosine below 0.90) in at least 5 of 6 models.

## What happened

**The preregistered verdict is untestable.** Two conditions in the plan were not met. Either one alone blocks a verdict.

- **Must-match control 4 failed.** With dropout off, the time embedding switched off and α set to 0.001, the adjoint and exact gradients were required to agree with a cosine of at least 0.999. Five of the six models fell short: 0.9982, 0.9990, 0.9988, 0.9987 and 0.9990, for seeds 1001, 1002, 1003, 1004 and 1006. Seed 1005 passed at 0.9992.
- **The noise precondition was not met.** Even after the preregistered doubling to 64 draws, the noise reference stayed below 0.95 in four models: 0.839, 0.356, 0.820 and 0.814, for seeds 1001, 1004, 1005 and 1006. At the trained state, two exact gradients with different dropout masks are close to orthogonal. Single-draw cosines run from 0.07 to 0.38 (S1).

The other controls passed:
- the weight check, exact for all six models;
- repeatability, the shared forward pass and the hand-written Euler loop, all to 1.0000000;
- the sign-flip control (−0.0018) and the injected effect (0.7072, against an expected 0.7071).

So the pipeline computes what it was designed to compute. The measure was too noisy, and one threshold was too tight for these models.

The primary cosines are reported in `T0_RESULTS.md` (mean 0.64), but under the plan they decide nothing.

## Which kind of failure

- **A bug or pipeline error:** evidence against. Every check of the pipeline itself passed exactly.
- **An insensitive test, or too little data:** yes, for the primary measure. Dropout noise at the trained state is far larger than the plan assumed, and 64 draws were not enough.
- **Must-match control 4:** its threshold assumed the adjoint's remaining error at α = 0.001 would be negligible. In the trained models it was about 0.001 in cosine. One untested explanation: the error grows with α at a rate set by the trained model, and that rate is large here. If so, the threshold, not the pipeline, was wrong. This explanation was formed after seeing the result, so it is a conjecture, not a finding.
- **The hypothesis false:** not assessed. T0 gives no verdict on the claim.

## What this rules out

Nothing about the claim. T0 neither supports nor refutes rival G.

## Exploratory observations, not confirmatory

These were measured once, on the planned batches, as secondary measures. They are recorded because they bear on the next test, not as findings.

- **Dropout off, the gradients diverge as α grows** (S3). The cosine between the adjoint and exact gradients is 0.98 to 0.997 at 0.001 times the learned α. It falls to 0.10 to 0.33 at the learned α, in every model. Without dropout these gradients have no random noise, so these numbers are not affected by the noise problem above.
- **The adjoint gradient of α has the opposite sign to the exact one** in all six trained models (S8). The training mean is negative (−0.017 to −0.047) and the exact mean positive (+0.075 to +0.14). Two independent exact means agree closely, so this is not noise. Gradient descent on the adjoint gradient pushes α up where the exact gradient would push it down. That is one candidate explanation for α growing from 0.1 to between 1.0 and 2.1, which T1 can check: AX and AA log α throughout training.
- **At initialisation the two gradients agree** (S4): cosine 0.994 to 0.996, noise reference 0.9999, and α gradients of matching size. So the disagreement develops during training.

## Next conjectures

- **T1, already approved and queued.** It tests the consequence directly: does training with exact gradients (AX) end with lower loss than training with adjoint gradients (AA)? It also records α throughout training for both.
- **A possible T0b, not yet preregistered.** Confirm the two exploratory observations above, the dropout-off divergence at learned α and the reversed α gradient, on new batches with dropout off, where a single deterministic gradient per batch avoids the noise problem. This would need its own preregistration before it runs. The prediction and thresholds should be set from what S3 and S8 would imply, and stated before any new batch is drawn.

## Records

- **Raw results:** `results/t0_results.json`, from the `colab-results` branch, run `t0_gradients/20261004-155450`.
- **Analysis:** `t0_analyze.py` writes `T0_RESULTS.md` from that file; the job is `t0_gradients.py`.
