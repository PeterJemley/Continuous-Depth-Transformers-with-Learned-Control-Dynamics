# Result note: T0

Date: 2026-10-04    Preregistration: `T0_PREREGISTRATION.md`    Generated report: `T0_RESULTS.md`    Raw results: `results/t0_results.json`, from `colab-results` branch `t0_gradients/20261004-155450`

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
