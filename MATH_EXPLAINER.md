# Mathematics of Continuous-Depth Transformers: A Plain Language Guide

This document explains the mathematical ideas behind the ODE flow experiment notebook. It's written for readers who want to understand *why* the code works, not just *what* it computes.

Mathematical expressions are shown in both LaTeX (for rendered viewing) and plain text (for raw markdown viewing).

---

## Table of Contents

1. [The Core Idea: Depth as a Continuous Variable](#1-the-core-idea-depth-as-a-continuous-variable)
2. [Attention: How Tokens Talk to Each Other](#2-attention-how-tokens-talk-to-each-other)
3. [Residual Connections: The Hidden ODE](#3-residual-connections-the-hidden-ode)
4. [The ODE Flow Block: Making Continuity Explicit](#4-the-ode-flow-block-making-continuity-explicit)
5. [Stability: Why the Output Scale Matters](#5-stability-why-the-output-scale-matters)
6. [Gradients: How Learning Flows Backward](#6-gradients-how-learning-flows-backward)
7. [Control Signals: Steering the Dynamics](#7-control-signals-steering-the-dynamics)
8. [Solver Probing: Using NFE as a Geometry Probe](#8-solver-probing-using-nfe-as-a-geometry-probe)
9. [The Training Objective: Cross-Entropy Loss](#9-the-training-objective-cross-entropy-loss)
10. [Putting It Together: The Hybrid Architecture](#10-putting-it-together-the-hybrid-architecture)

---

## 1. The Core Idea: Depth as a Continuous Variable

### The Standard View

A typical transformer stacks discrete layers:

```
Input → Layer 1 → Layer 2 → Layer 3 → ... → Layer L → Output
```

Each layer transforms the representation, and we count layers as integers: 1, 2, 3, and so on.

Think of this as an **assembly line**: Station 1 adds a part, Station 2 adds a part, Station 3 adds a part. The token moves through discrete, fixed processing stages.

### The Continuous View

What if depth weren't an integer but a continuous number? Instead of "layer 3," we could ask: "what's the representation at depth 2.7?"

Think of this as a **particle flowing through a fluid**. Instead of jumping between stations, the token smoothly evolves as it moves through the network. At any moment, the fluid's current (the vector field) tells the particle which direction to move.

This isn't just mathematical abstraction. It changes how we think about the network:

- **Assembly line (discrete)**: Fixed stations, fixed order, no in-between states
- **Fluid flow (continuous)**: Smooth trajectory, can be sampled anywhere, dynamics can be adjusted

The ODE flow block implements this continuous view. Instead of discrete updates, the representation *flows* through a learned transformation, and we can query it at any depth.

---

## 2. Attention: How Tokens Talk to Each Other

### The Problem

Language is relational. The meaning of "it" in "The cat sat on the mat because it was tired" depends on connecting "it" back to "cat." A neural network needs a mechanism for tokens to exchange information.

### Queries, Keys, and Values

Attention solves this with three learned projections:

- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "What can I contribute?"

### The Filing System Analogy

Imagine a filing system:

1. You have a **Query**: "I need files on cats"
2. Every folder has a **Key**: "This folder is about dogs", "This folder is about cats", etc.
3. You calculate the match score (dot product) between your Query and every Key
4. You scale this score down for numerical stability
5. You mask out folders that haven't been created yet (causality—can't look at future tokens)
6. You convert scores to probabilities (softmax)
7. You take a weighted sum of the contents (**Values**) of the relevant folders

### The Math

Given a sequence of token representations X ∈ ℝ^{T×D} (shape: sequence_length × dimension):

**LaTeX:**
$$Q = XW_Q, \quad K = XW_K, \quad V = XW_V$$

$$\text{Attention}(X) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_h}}\right) V$$

**Plain text:**
```
Q = X · W_Q    (queries)
K = X · W_K    (keys)  
V = X · W_V    (values)

Attention(X) = softmax(Q · K^T / sqrt(d_h)) · V
```

The division by √d_h (`sqrt(d_h)`) prevents the dot products from growing too large as dimension increases. Without this scaling, the softmax would saturate—putting nearly all attention on one token and making gradients vanishingly small.

### Causal Masking

The equation above is general attention. For language modeling, we need an additional constraint: token 5 shouldn't see tokens 6, 7, 8... (that would be cheating—using the future to predict the future).

We enforce this by adding a causal mask M to the attention scores before softmax:

**LaTeX:**
$$\text{Attention}(X) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_h}} + M\right) V$$

**Plain text:**
```
Attention(X) = softmax(Q · K^T / sqrt(d_h) + M) · V
```

The mask M is defined as:

**LaTeX:**
$$M_{ij} = \begin{cases} 0 & \text{if } i \geq j \\ -\infty & \text{otherwise} \end{cases}$$

**Plain text:**
```
M_{ij} = 0 if i >= j, else -infinity
```

How does this work? When we add -∞ to an attention score and then apply softmax, we get e^{-∞} = 0. This means future tokens receive exactly zero attention weight—it's mathematically impossible for a token to attend to positions that come after it.

### Multiple Heads

One attention pattern might focus on syntax ("the" relates to "cat"), another on semantics ("tired" relates to "cat"), another on position. Multiple heads run in parallel, each learning different patterns, then combine their outputs.

For H heads, each head h operates on d_h = D/H dimensions:

**LaTeX:**
$$\text{head}_h = \text{Attention}(XW_Q^{(h)}, XW_K^{(h)}, XW_V^{(h)})$$

$$\text{MultiHead}(X) = \text{Concat}(\text{head}_1, \ldots, \text{head}_H) W_O$$

**Plain text:**
```
head_h = Attention(X·W_Q^h, X·W_K^h, X·W_V^h)
MultiHead(X) = Concat(head_1, ..., head_H) · W_O
```

---

## 3. Residual Connections: The Hidden ODE

### The Residual Block

A transformer layer doesn't replace the representation—it *refines* it:

**LaTeX:**
$$H_{\text{new}} = H_{\text{old}} + \text{Sublayer}(H_{\text{old}})$$

**Plain text:**
```
H_new = H_old + Sublayer(H_old)
```

This "+H_old" is the residual connection. It says: "keep what you had, plus whatever refinement the sublayer computed."

### Why This Matters for Training

Without residuals, gradients must flow through every sublayer to reach early parameters. If any sublayer shrinks gradients (multiplies by numbers < 1), the signal vanishes exponentially with depth.

With residuals, gradients have a "skip path" directly backward:

**LaTeX:**
$$\frac{\partial \mathcal{L}}{\partial H_{\text{old}}} = \frac{\partial \mathcal{L}}{\partial H_{\text{new}}} \cdot \left(I + \frac{\partial \text{Sublayer}}{\partial H_{\text{old}}}\right)$$

**Plain text:**
```
dLoss/dH_old = dLoss/dH_new · (I + dSublayer/dH_old)
```

The identity matrix I ensures gradients flow even if the sublayer gradient is small.

### The Hidden ODE

Here's the key insight: the residual update

**LaTeX:**
$$H_{n+1} = H_n + F(H_n)$$

**Plain text:**
```
H_{n+1} = H_n + F(H_n)
```

is exactly **forward Euler integration** of the differential equation

**LaTeX:**
$$\frac{dH}{dt} = F(H)$$

**Plain text:**
```
dH/dt = F(H)
```

with step size Δt = 1.

Forward Euler is the simplest numerical method for solving ODEs: "to find where you'll be, take your current velocity and step forward." Each residual layer is one Euler step.

This means a transformer with L residual layers is implicitly approximating a continuous dynamical system with L discrete steps.

---

## 4. The ODE Flow Block: Making Continuity Explicit

### From Implicit to Explicit

If residual layers are secretly Euler steps, why not make this explicit? Define:

**LaTeX:**
$$\frac{dH}{d\tau} = F_\theta(H, \tau, u)$$

**Plain text:**
```
dH/dtau = F_theta(H, tau, u)
```

where:
- τ ∈ [0, 1] (`tau in [0, 1]`) is "continuous depth" (τ=0 is the input, τ=1 is the output)
- F_θ (`F_theta`) is a learned vector field (a neural network)
- u is an optional control signal

Then solve this ODE numerically to get the output:

**LaTeX:**
$$H(1) = H(0) + \int_0^1 F_\theta(H(\tau), \tau, u) \, d\tau$$

**Plain text:**
```
H(1) = H(0) + integral_0^1 F_theta(H(tau), tau, u) dtau
```

### The Particle-in-Fluid Picture

In a standard Transformer, a token is processed like an **assembly line**: Station 1 adds a part, Station 2 adds a part.

In the ODE Transformer, the token is like a **particle flowing in a fluid**:
- The **Vector Field** (F_θ) tells the particle which direction to move at any given moment
- The **Control** (u) is like a wind that can slightly alter the current
- The **Output Scale** (α, see Section 5) controls the viscosity of the fluid—too low and the particle barely moves; too high and it shoots off unpredictably

### What Does the Vector Field Compute?

In our implementation, F_θ looks like a transformer sublayer: attention plus MLP. But it also takes:

- **Time τ**: The vector field can behave differently at different depths. Early processing (τ ≈ 0) might differ from late processing (τ ≈ 1). The time embedding projects the scalar τ into the model dimension D, allowing the network to know "how deep" it is in the flow.
- **Control u**: An external signal that biases the dynamics (more on this in Section 7).

### Solving the ODE

Since computers cannot handle true continuous time, we approximate the flow by taking small steps. We use Euler's method with m steps:

**LaTeX:**
$$\tau_k = \frac{k}{m}, \quad k = 0, 1, \ldots, m$$

$$H(\tau_{k+1}) = H(\tau_k) + \Delta\tau \cdot F_\theta(H(\tau_k), \tau_k, u)$$

**Plain text:**
```
tau_0 = 0, tau_1 = 0.25, tau_2 = 0.5, tau_3 = 0.75, tau_4 = 1.0  (for m=4)

H(tau_{k+1}) = H(tau_k) + delta_tau · F_theta(H(tau_k), tau_k, u)
```

where Δτ = 1/m (`delta_tau = 1/m`).

The Euler method is the simplest approach: look at the current direction (velocity) and take a small step that way. Then re-evaluate the direction and repeat. More steps = better approximation to the true continuous dynamics, but more compute.

### Why Bother?

Several benefits emerge from the continuous view:

1. **Parameter sharing**: One vector field F_θ is reused across all integration steps. Replacing 2 transformer layers with 4 Euler steps uses fewer parameters.

2. **Adaptive computation**: We could use more steps for harder inputs (not implemented here, but possible with adaptive solvers).

3. **Continuous control**: The control signal u can smoothly steer the dynamics, not just switch between discrete modes.

4. **Memory efficiency**: The adjoint method computes gradients in O(1) memory regardless of step count (see Section 6).

---

## 5. Stability: Why the Output Scale Matters

### The Problem with Large Updates

Consider what happens if F_θ(H) is large. The Euler update

**LaTeX:**
$$H_{k+1} = H_k + \Delta\tau \cdot F_\theta(H_k)$$

**Plain text:**
```
H_{k+1} = H_k + delta_tau · F_theta(H_k)
```

takes a big step. If we're approximating a smooth curve, big steps overshoot and oscillate. Worse, errors compound: each step's error affects the next step's input.

For gradient-based learning, this is catastrophic. Large updates mean large Jacobians, which amplify gradients exponentially during backpropagation.

### The Output Scale α

Our vector field includes a learned scalar α (`alpha`):

**LaTeX:**
$$\frac{dH}{d\tau} = \alpha \cdot F_\theta(H, \tau, u)$$

**Plain text:**
```
dH/dtau = alpha · F_theta(H, tau, u)
```

initialized to α = 0.1 (`alpha = 0.1`). This scales down the entire vector field.

In the fluid analogy, α controls the **viscosity**. If viscosity is too low (large α), the particle shoots off unpredictably. If too high (small α), the particle barely moves. We initialize with high viscosity (small α) for safety.

With m = 4 steps over [0, 1], each step has Δτ = 0.25 (`delta_tau = 0.25`). The effective step size is:

**LaTeX:**
$$\alpha \cdot \Delta\tau = 0.1 \times 0.25 = 0.025$$

**Plain text:**
```
alpha · delta_tau = 0.1 × 0.25 = 0.025
```

This is conservative: small steps stay close to the true continuous trajectory.

### The Jacobian Argument

For an ODE dH/dτ = f(H) (`dH/dtau = f(H)`), how sensitive is the final state H(1) to the initial state H(0)? This sensitivity satisfies its own ODE:

**LaTeX:**
$$\frac{d}{d\tau} \left[\frac{\partial H(\tau)}{\partial H(0)}\right] = \frac{\partial f}{\partial H} \cdot \frac{\partial H(\tau)}{\partial H(0)}$$

**Plain text:**
```
d/dtau [dH(tau)/dH(0)] = (df/dH) · (dH(tau)/dH(0))
```

This is a matrix differential equation. If the Jacobian ∂f/∂H (`df/dH`) has eigenvalues with positive real parts, the sensitivity grows exponentially. In backpropagation terms: gradients explode.

Our parameterization bounds this:

**LaTeX:**
$$\frac{\partial (\alpha F)}{\partial H} = \alpha \cdot \frac{\partial F}{\partial H}$$

**Plain text:**
```
d(alpha·F)/dH = alpha · dF/dH
```

By setting α = 0.1 (`alpha = 0.1`), we reduce the Jacobian's eigenvalues by a factor of 10. What might have been an unstable system becomes stable.

### Learned, Not Fixed

Crucially, α is a learnable **scalar** parameter, not a fixed constant. During training, if the model benefits from stronger dynamics, it can increase α. But it starts safe—in a regime where Euler is accurate and gradients are well-behaved.

**Experimental Result:** The scalar α converged from 0.1 to **0.065**—a 35% reduction. The model preferred a conservative, damped update regime rather than learning to "speed up" the dynamics. This confirms the stability mechanism works as intended.

---

## 6. Gradients: How Learning Flows Backward

### The Naive Approach

To train the ODE block, we need ∂Loss/∂θ (`dLoss/dtheta`) where θ are the vector field's parameters. The obvious approach: backpropagate through every Euler step, storing intermediate states.

Problem: memory scales with the number of steps. For adaptive solvers that might take hundreds of steps, this is prohibitive.

### The Adjoint Method

Chen et al. (2018) showed a better way. Define the "adjoint" a(τ) = ∂Loss/∂H(τ) (`a(tau) = dLoss/dH(tau)`). This satisfies its own ODE running *backward* in time:

**LaTeX:**
$$\frac{da}{d\tau} = -a^\top \cdot \frac{\partial F}{\partial H}$$

**Plain text:**
```
da/dtau = -a^T · (dF/dH)
```

Starting from a(1) = ∂Loss/∂H(1) (`a(1) = dLoss/dH(1)`, which we get from the downstream loss), we integrate backward to get a(0), then use this to compute parameter gradients.

The parameter gradients are accumulated during backward integration:

**LaTeX:**
$$\frac{\partial \mathcal{L}}{\partial \theta} = -\int_1^0 a(\tau)^\top \frac{\partial F_\theta}{\partial \theta} \, d\tau$$

**Plain text:**
```
dLoss/dtheta = -integral_1^0 a(tau)^T · (dF_theta/dtheta) dtau
```

The key insight: we don't need to store intermediate H(τ). We can recompute them during the backward pass, or use checkpointing. Memory cost is O(1) in the number of steps.

### In Practice

The `torchdiffeq` library handles this automatically. When we call `odeint(...)`, backward passes use the adjoint method. We get the memory benefits without implementing the math ourselves.

**Plain language**: The adjoint method ensures that calculating the error (gradients) doesn't run out of memory, no matter how many small steps we take.

---

## 7. Control Signals: Steering the Dynamics

### The Idea

The control signal u ∈ ℝ^c (`u in R^c`) is a low-dimensional vector that biases the dynamics:

**LaTeX:**
$$\frac{dH}{d\tau} = \alpha \cdot F_\theta(H, \tau, u)$$

**Plain text:**
```
dH/dtau = alpha · F_theta(H, tau, u)
```

Different values of u produce different trajectories through representation space, and thus different outputs.

In the fluid analogy, the control signal u is like a **wind** that can slightly alter the current the particle flows through.

### How Control Enters the Vector Field

In our implementation, u is embedded into the model dimension and added to the hidden state before computing the vector field:

**LaTeX:**
$$H_{\text{cond}} = H + \text{embed}(\tau) + \text{embed}(u)$$

$$\text{output} = \text{Attention}(\text{LN}(H_{\text{cond}})) + \text{MLP}(\text{LN}(H_{\text{cond}}))$$

**Plain text:**
```
H_conditioned = H + embed(tau) + embed(u)
output = Attention(LayerNorm(H_conditioned)) + MLP(LayerNorm(H_conditioned))
```

This is additive conditioning: u shifts the "starting point" of the attention and MLP computations.

### The "Global Bias" Mechanism

A key finding from our experiments: the control signal u affects the dynamics **globally from τ = 0**, not at a discrete "decision point."

Since u is an input to F_θ at every depth τ, it mathematically must influence the entire trajectory. We verified this empirically by training linear probes on hidden states at intermediate depths (τ = 0, 0.5, 0.67, 1.0). Sentiment separability was flat across all depths—no spike at any particular "decision point."

**Analogy:** The control signal is like tilting a table before releasing a ball, not like pressing a button halfway through the ball's journey. The decision is made at the moment of tilting; the rest is just physics playing out.

### The "Dynamics of Resistance"

Control signals interact with the model's intrinsic biases:

- **Cooperative Dynamics (u > 0):** When steering toward the model's natural bias (e.g., positive sentiment when the model already prefers "good"), the control works *with* the flow. Easy trajectory, low curvature.

- **Adversarial Dynamics (u < 0):** When steering against the natural bias, the control must *fight* the existing basin of attraction. The vector field must exhibit greater curvature to divert trajectories. Harder trajectory, high curvature.

This explains the observed steering asymmetry (98% positive vs 88% negative accuracy) and the NFE asymmetry (14 vs 20 solver steps).

### Why ODE Control Differs from Activation Steering

Activation steering (adding vectors to residual streams) operates at fixed injection points. ODE control is *continuous*: the control signal influences the dynamics throughout the integration, not just at one layer.

This potentially allows smoother, more predictable control—small changes in u produce proportionally small changes in output, rather than discontinuous jumps.

---

## 8. Solver Probing: Using NFE as a Geometry Probe

### The Mathematical Insight

Adaptive ODE solvers (like `dopri5`) adjust their step size based on local error estimates. Regions where the vector field F_θ has high curvature require smaller steps to maintain accuracy. This means the **Number of Function Evaluations (NFE)** serves as a proxy for the geometric complexity of the learned dynamics.

### Local Truncation Error

For a Runge-Kutta method of order p, the local truncation error scales as:

**LaTeX:**
$$\epsilon_{\text{local}} \propto \left\| \frac{d^{p+1}H}{d\tau^{p+1}} \right\| \cdot (\Delta\tau)^{p+1}$$

**Plain text:**
```
epsilon_local proportional_to ||d^{p+1}H/dtau^{p+1}|| · (delta_tau)^{p+1}
```

The higher derivative d^{p+1}H/dτ^{p+1} measures how rapidly the dynamics change—essentially, the **curvature** of the trajectory through state space.

### Adaptive Step Size Control

An adaptive solver maintains error below tolerance ε_tol by adjusting step size:

**LaTeX:**
$$\Delta\tau_{\text{new}} = \Delta\tau_{\text{old}} \cdot \left( \frac{\epsilon_{\text{tol}}}{\epsilon_{\text{local}}} \right)^{1/(p+1)}$$

**Plain text:**
```
delta_tau_new = delta_tau_old · (epsilon_tol / epsilon_local)^{1/(p+1)}
```

High-curvature regions (large ε_local) force smaller steps. Integrating from τ = 0 to τ = 1 thus requires more steps, increasing NFE.

### NFE as a Curvature Measure

The total NFE over the integration interval reflects the integrated curvature:

**LaTeX:**
$$\text{NFE} \propto \int_0^1 \left\| \frac{\partial F_\theta}{\partial H} \right\| d\tau$$

**Plain text:**
```
NFE proportional_to integral_0^1 ||dF_theta/dH|| dtau
```

This is approximate—the actual relationship depends on the solver's specific error estimation—but captures the key insight: **NFE measures geometry without requiring explicit Jacobian computation**.

### Experimental Findings

Sweeping the control signal u from -2 to +2 revealed two distinct dynamical regimes:

| Control Region | NFE | Interpretation |
|----------------|-----|----------------|
| u < 0.3 (negative sentiment) | 20 | High-curvature dynamics |
| u > 0.4 (positive sentiment) | 14 | Low-curvature dynamics |

The 43% difference in NFE indicates that negative sentiment requires traversing a region of the vector field with systematically higher curvature. This aligns with the "Dynamics of Resistance" framework: negative steering must fight the model's natural positive bias.

### Validation: Solver Agreement

If NFE reflects intrinsic geometry rather than solver artifacts, different adaptive solvers should agree. We tested:

| Solver | Order | NFE |
|--------|-------|-----|
| dopri5 | 5th | 20 |
| adaptive_heun | 2nd | 17 |

Agreement within 15% confirms NFE measures the dynamics, not the solver.

### Validation: Tolerance Scaling

For smooth dynamics, NFE should scale sublinearly with tolerance:

| Tolerance | NFE |
|-----------|-----|
| 10^{-2} | 14 |
| 10^{-3} | 20 |
| 10^{-4} | 20 |
| 10^{-5} | 26 |

A 1000× tighter tolerance increased NFE by only 86% (not 1000×), confirming the learned vector field is smooth rather than fractal.

### The Solver Invariance Test

We also used solver comparison as a **diagnostic for true continuity**. By comparing trajectories from the fixed Euler solver (used in training) against a high-precision adaptive solver (Dopri5), we can detect whether the model has overfitted to discretization artifacts.

**Result:** 0.068% trajectory divergence—proving the model learned an intrinsic continuous vector field, not a "ResNet in disguise."

### Interpretability Implication

This reveals a form of interpretability unavailable in discrete transformers: the solver acts as a probe into the geometry of learned representations without requiring additional trained classifiers.

---

## 9. The Training Objective: Cross-Entropy Loss

The model is trained to predict the next token in the sequence.

### The Equation

**LaTeX:**
$$\mathcal{L} = -\frac{1}{BT} \sum_{b,t} \log P(y_{b,t} \mid x_{b,1:t})$$

**Plain text:**
```
L = -(1/BT) sum_{b,t} log P(y_{b,t} | x_{b,1:t})
```

### Variable Breakdown

- y_{b,t} (`y_{b,t}`): The actual next token (target) at batch index b and position t
- x_{b,1:t} (`x_{b,1:t}`): The context (history) of tokens up to position t
- P(·): The probability distribution output by the model (via the final softmax layer)
- log: The natural logarithm. Since probabilities are in (0,1], the log is negative or zero; the negative sign at the front makes the loss positive

### Plain Language

The model assigns a probability to every word in the vocabulary being the "next word." We look at the probability it assigned to the *correct* word.

- If the model assigns **high probability** to the correct word → log(high) ≈ 0 → loss is **low** (good)
- If the model assigns **low probability** to the correct word → log(low) << 0 → loss is **high** (bad, punishment)

We want the model to assign probability 1 to the correct word (maximizing likelihood). Mathematically, maximizing likelihood is equivalent to minimizing the negative log-likelihood, which is what cross-entropy loss does.

---

## 10. Putting It Together: The Hybrid Architecture

### Why Hybrid?

We don't replace the entire transformer with ODEs. The architecture is:

```
[Discrete Layers 0-1] → [ODE Flow Block] → [Discrete Layers 4-5]
```

Intuition for this design:

- **Early layers** (discrete): Low-level feature extraction. These layers learn token-level patterns that don't need continuous-depth refinement.

- **Middle layers** (ODE): Representation refinement. This is where the model builds abstract understanding—a good candidate for continuous dynamics and control.

- **Late layers** (discrete): Task-specific readout. These layers convert refined representations into predictions.

### Parameter Count

The ODE block replaces k discrete layers with a single vector field reused across m integration steps. If the vector field has similar structure to one transformer layer:

- **Discrete**: k × (parameters per layer)
- **ODE**: 1 × (parameters per layer) + small overhead for time/control embeddings

**Plain text:**
```
Discrete: k × (parameters per layer)
ODE: 1 × (parameters per layer) + small overhead
```

The experiment shows: replacing 2 layers with the ODE block yields 97.6% of baseline parameters—a slight reduction.

### The Experiment

The notebook trains two models:

1. **Baseline**: 6 standard transformer layers
2. **Hybrid**: Layers 0-1 discrete, layers 2-3 replaced by ODE (4 Euler steps), layers 4-5 discrete

Both train on WikiText-2 for 500 steps. The results:

| Metric | Baseline | Hybrid ODE |
|--------|----------|------------|
| Parameters | 30,503,424 | 29,781,249 (97.6%) |
| Final loss (last 50 steps) | 6.471 | 6.449 |
| Gradient norm (mean ± std) | 0.521 ± 0.140 | 0.509 ± 0.142 |
| ODE block gradient norm | — | 0.033 ± 0.021 |
| Learned scale α (Scalar) | N/A | 0.1 → 0.065 |
| Vanishing gradient steps | 0 | 0 |
| Exploding gradient steps | 0 | 0 |

This validates the core architectural assumption: ODE blocks are a drop-in replacement for transformer layers, with stable training dynamics.

---

## Summary: The Mathematical Story

1. **Transformers stack discrete layers**, each refining the representation via residual connections. Think: assembly line.

2. **Residual updates are Euler steps** of an implicit ODE: H_{n+1} = H_n + F(H_n) (`H_{n+1} = H_n + F(H_n)`).

3. **We make the ODE explicit**, solving dH/dτ = α·F(H,τ,u) (`dH/dtau = alpha·F(H,tau,u)`) with a learned vector field. Think: particle in fluid.

4. **The output scale α is the stability mechanism**: initialized to 0.1, it converged to 0.065. This bounds the Jacobian and prevents gradient explosion, while remaining learnable. Think: viscosity.

5. **Control signals u steer the dynamics** continuously throughout integration, enabling smooth, interpretable control over generation. The control tilts the landscape globally from τ = 0. Think: wind altering the current.

6. **Adaptive solvers reveal geometry**: NFE (Number of Function Evaluations) measures the curvature of the learned dynamics without explicit Jacobian computation. Different control values create distinct dynamical regimes.

7. **The adjoint method** (`odeint_adjoint`) provides memory-efficient gradients, enabling deep continuous dynamics without O(depth) memory cost.

8. **Cross-entropy loss** trains the model to assign high probability to the correct next token.

9. **The hybrid architecture** keeps discrete layers for early/late processing, using continuous dynamics only where they add value.

---

## Validated Experimental Results

The architecture has been validated through four experiments (see README.md for full details):

### Gradient Flow & Stability
Zero exploding or vanishing gradient steps across 500 training iterations. The hybrid model achieved slightly better convergence (loss 6.449) than the baseline (loss 6.471) with 97.6% of the parameters. The scalar α converged from 0.1 to 0.065, confirming the model prefers a conservative update regime.

### Semantic Steering ("Hybrid Unfreeze")
By freezing embeddings and output head and training only the ODE block, the control signal learned meaningful semantics:

| Control Signal | Target | P(Good) | P(Bad) | Result |
|----------------|--------|---------|--------|--------|
| +1.0 (Positive) | "Good" | **98.0%** | 0.2% | ✅ Success |
| -1.0 (Negative) | "Bad" | 0.2% | **88.1%** | ✅ Success |
| 0.0 (Neutral) | — | 91.4% | 3.3% | (Natural Bias) |

The asymmetry (98% vs 88%) reflects "Dynamics of Resistance": positive steering is cooperative (working with the 91.4% natural bias), while negative steering is adversarial (fighting against it).

### Continuous Interpolation & Solver Invariance
The probability curves formed smooth sigmoids—proving the model learned a continuous manifold, not binary switching. The Solver Invariance Test showed only **0.068% trajectory divergence** between fixed Euler and adaptive Dopri5 solvers, proving true continuous dynamics.

### Efficiency
The hybrid model achieves **latency parity** with the baseline (0.98×, actually 2.4% faster)—far better than the theoretical 1.33× overhead. This is attributed to architectural simplicity (MLP vs attention) and cache locality from weight reuse.

---

## Code-to-Math Variable Mapping

| Code Variable | Math Symbol | Plain Text | Meaning |
|---------------|-------------|------------|---------|
| `d_model` | D | D | Dimensionality of hidden states (e.g., 256) |
| `n_heads` | H | H | Number of attention heads (e.g., 4) |
| `head_dim` | d_h | d_h | Dimension per head (D/H) |
| `output_scale` | α | alpha | Learned scalar scaling the ODE vector field magnitude |
| `integration_times` | τ_k | tau_k | Tensor `[0, 0.25, 0.5, 0.75, 1.0]` defining Euler steps |
| `control` | u | u | External signal vector for steering the ODE flow |
| `n_steps` | m | m | Number of Euler integration steps |
| `qkv` | W_Q, W_K, W_V | W_Q, W_K, W_V | Query, Key, Value projection weights (fused) |
| `mask` | M | M | Causal attention mask |
| `time_embed` | embed(τ) | embed(tau) | Projects scalar depth to model dimension |
| `control_embed` | embed(u) | embed(u) | Projects control signal to model dimension |

---

## Notation Reference

| Symbol | LaTeX | Plain Text | Meaning |
|--------|-------|------------|---------|
| H | `H` | H | Hidden state (sequence of token representations) |
| τ | `\tau` | tau | Continuous depth parameter, τ ∈ [0, 1] |
| F_θ | `F_\theta` | F_theta | Vector field (neural network with parameters θ) |
| α | `\alpha` | alpha | Learned output scale (initialized to 0.1, converged to 0.065) |
| u | `u` | u | Control signal (low-dimensional vector) |
| Q, K, V | `Q, K, V` | Q, K, V | Query, Key, Value projections in attention |
| d | `d` | d | Model dimension |
| m | `m` | m | Number of Euler integration steps |
| Δτ | `\Delta\tau` | delta_tau | Step size = 1/m |
| a(τ) | `a(\tau)` | a(tau) | Adjoint state = ∂L/∂H(τ) |
| L | `\mathcal{L}` | L or Loss | Loss function |
| ℝ | `\mathbb{R}` | R | Real numbers |
| NFE | — | NFE | Number of Function Evaluations (solver step count) |

---

## Further Reading

- Chen et al. (2018). "Neural Ordinary Differential Equations." NeurIPS. *The foundational paper on neural ODEs and the adjoint method.*

- Vaswani et al. (2017). "Attention Is All You Need." NeurIPS. *The original transformer architecture.*

- He et al. (2016). "Deep Residual Learning." CVPR. *Residual connections and their impact on trainability.*

- Turner et al. (2023). "Steering Language Models with Activation Engineering." *Activation steering for controllable generation.*
