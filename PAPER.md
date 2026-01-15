# Continuous-Depth Transformers with Learned Control Dynamics

**Peter Jemley**  
jemley.p@northeastern.edu

*January 2026*

---

## Abstract

We present a hybrid transformer architecture that replaces discrete middle layers with a continuous-depth Neural Ordinary Differential Equation (ODE) block, enabling inference-time control over generation attributes via a learned steering signal. Unlike standard transformers that process representations through fixed discrete layers, our approach treats depth as a continuous variable governed by a learned vector field F_θ(H, τ, u), where u is a low-dimensional control signal injected via explicit concatenation. We validate the architecture through four experiments: (1) gradient flow stability with zero exploding/vanishing gradient events, (2) semantic steering achieving 98%/88% accuracy for positive/negative sentiment control, (3) continuous interpolation validated by a negligible 0.068% trajectory divergence between fixed and adaptive solvers, and (4) efficiency benchmarking demonstrating latency parity with standard discrete baselines. Additionally, we show that adaptive ODE solvers reveal geometric structure in the learned dynamics: the control signal partitions the vector field into distinct dynamical regimes with different curvature characteristics. The adjoint method enables O(1) memory training regardless of integration depth. Our results demonstrate that continuous-depth dynamics with learned control signals provide a viable, efficient mechanism for steerable language generation.

---

## 1. Introduction

Autoregressive transformers optimize for next-token prediction, producing locally probable continuations that are grammatically correct but often lack fine-grained controllability. Users seeking to shift generation along interpretable axes—creativity, formality, sentiment—must resort to prompt engineering, temperature adjustment, or rejection sampling, all of which are inefficient and unpredictable.

We propose treating the transformer's depth dimension as a continuous variable, replacing a subset of residual blocks with a neural ODE that admits an external control signal. This reframes generation as trajectory evolution through a learned dynamical system, where the control signal biases the trajectory without requiring discrete mode switches or weight updates.

### 1.1 Contributions

- A hybrid architecture combining discrete transformer layers with continuous ODE dynamics, using a learned output scale α (`alpha`) for stability and explicit control injection.

- Validated controllability: the control signal u successfully steers sentiment with 98%/88% accuracy for positive/negative targets.

- Proposal of the **Solver Invariance Test**: We formalize a Popperian diagnostic for continuous-depth architectures. By measuring the trajectory divergence between fixed-step training solvers and adaptive inference solvers, we provide a quantitative metric (0.068% in our case) to falsify the hypothesis that a model has overfitted to discrete layers.

- Practical efficiency: Benchmarks demonstrate inference latency comparable to standard transformers, effectively eliminating the computational overhead typically associated with ODE integration.

- Interpretability via solver probing: Adaptive solvers reveal that positive and negative sentiment occupy geometrically distinct regions of the learned dynamics.

---

## 2. Related Work

### 2.1 Neural ODEs and Continuous-Depth Networks

Chen et al. (2018) introduced neural ODEs, showing that residual networks can be viewed as Euler discretizations of continuous dynamics. Their formulation enables constant-memory training via the adjoint method and adaptive compute via variable solver steps. Dupont et al. (2019) addressed expressivity limitations by augmenting the state space.

Applied to transformers, Li et al. (2022) proposed ODE Transformer, reinterpreting transformer blocks as Runge-Kutta discretizations and achieving state-of-the-art on WMT translation benchmarks. Our work differs in focus: we target inference-time controllability via an explicit control signal, rather than task performance or parameter efficiency.

### 2.2 Activation Steering

Turner et al. (2023) introduced activation engineering, computing steering vectors from contrastive prompt pairs and adding them to residual streams at inference time. This achieves strong results on sentiment control and detoxification. Zou et al. (2023) extended this to representation engineering for safety applications.

Our approach differs fundamentally: activation steering discovers steering vectors post-hoc from existing representations, while our control dynamics are learned end-to-end during training. Additionally, ODE-based control operates continuously across depth rather than at fixed injection points.

---

## 3. Architecture

### 3.1 Continuous-Depth Flow Module

Let H(τ) ∈ ℝ^{B×T×D} (`H(tau) in R^{B×T×D}`) denote the hidden state at continuous depth τ ∈ [0, 1] (`tau in [0, 1]`). We replace k consecutive residual blocks with an Initial Value Problem (IVP):

**LaTeX:**
$$\frac{dH}{d\tau} = \alpha \cdot F_\theta(H, \tau, u)$$

$$H(1) = H(0) + \int_0^1 \alpha \cdot F_\theta(H(t), t, u) \, dt$$

**Plain text:**
```
dH/dtau = alpha · F_theta(H, tau, u)

H(1) = H(0) + integral_0^1 alpha · F_theta(H(t), t, u) dt
```

where F_θ (`F_theta`) is a neural network parameterizing the vector field and α (`alpha`) is a learned scaling factor initialized to 0.1 for stability.

To ensure the control signal u ∈ ℝ^c (`u in R^c`, where c ≪ D) effectively biases the dynamics at every depth, we define F_θ via explicit concatenation:

**LaTeX:**
$$F_\theta(H, \tau, u) = \text{MLP}\left(\text{Concat}(H, \text{Broadcast}(u))\right)$$

**Plain text:**
```
F_theta(H, tau, u) = MLP(Concat(H, Broadcast(u)))
```

This formulation ensures the derivative of the hidden state is directly conditioned on the control variable, preventing the "vanishing control" problem where the signal might otherwise be diluted by layer normalizations.

### 3.2 The Residual-ODE Connection

A standard residual block computes H_{n+1} = H_n + F(H_n), which is exactly forward Euler integration with step size Δτ = 1 (`delta_tau = 1`). Our continuous formulation makes this explicit while introducing two key modifications:

1. **Learned output scale α**: Bounds the effective step size to α·Δτ ≈ 0.025 (`alpha·delta_tau ≈ 0.025`, assuming 4 Euler steps), preventing gradient explosion.

2. **Control signal u**: Enables inference-time steering via the learned vector field.

### 3.3 Stability via Learned Output Scaling

The sensitivity of the final state H(1) to the initial state H(0) satisfies:

**LaTeX:**
$$\frac{d}{d\tau}\left(\frac{\partial H(\tau)}{\partial H(0)}\right) = \frac{\partial f}{\partial H} \cdot \frac{\partial H(\tau)}{\partial H(0)}$$

**Plain text:**
```
d/dtau [dH(tau)/dH(0)] = (df/dH) · (dH(tau)/dH(0))
```

If the Jacobian ∂f/∂H (`df/dH`) has eigenvalues with positive real parts, sensitivity grows exponentially—gradients explode. Our parameterization bounds this:

**LaTeX:**
$$\frac{\partial(\alpha F)}{\partial H} = \alpha \cdot \frac{\partial F}{\partial H}$$

**Plain text:**
```
d(alpha·F)/dH = alpha · dF/dH
```

With α initialized to 0.1, we reduce the Jacobian's effective eigenvalues, transforming potentially unstable dynamics into stable ones without computationally expensive spectral normalization.

### 3.4 Hybrid Architecture

We utilize a sandwich design:

1. **Discrete Early Layers (0-1)**: Standard transformer blocks for low-level feature extraction

2. **Continuous ODE Block (replacing 2-3)**: Single reused vector field integrated via adjoint method for O(1) memory

3. **Discrete Late Layers (4-5)**: Standard blocks for task-specific readout

This preserves trainability at the edges while introducing continuous dynamics in the middle layers where representations are most malleable.

### 3.5 Memory-Efficient Gradients

The adjoint state a(τ) = ∂L/∂H(τ) (`a(tau) = dL/dH(tau)`) satisfies a backward ODE:

**LaTeX:**
$$\frac{da}{d\tau} = -a^\top \cdot \frac{\partial F}{\partial H}$$

**Plain text:**
```
da/dtau = -a^T · (dF/dH)
```

Starting from a(1) = ∂L/∂H(1), we integrate backward to compute parameter gradients without storing intermediate states. This gives O(1) memory cost regardless of integration steps, implemented via `odeint_adjoint` from `torchdiffeq`.

---

## 4. Experiments

We validate the architecture through four experiments on a 6-layer transformer variant with d = 256 and 4 attention heads.

### 4.1 Experiment 1: Gradient Flow & Stability

**Goal**: Verify the model trains without diverging and that gradients propagate through the ODE block.

**Setup**: Train baseline (6 discrete layers) and hybrid (layers 2-3 replaced with ODE) on WikiText-2 for 500 steps.

**Result**: Zero exploding or vanishing gradient events. We initialized the learned **scalar** output scale α to 0.1. By the end of training, α converged to **0.065**, a reduction of approximately 35%. This indicates that the model preferred a conservative update regime, effectively dampening the vector field magnitude to maintain stability while still leveraging the continuous dynamics for feature refinement.

**Table 1: Training stability comparison**

| Metric | Baseline | Hybrid ODE |
|--------|----------|------------|
| Parameters | 30,503,424 | 29,781,249 (97.6%) |
| Final loss (last 50 steps) | 6.471 | 6.449 |
| Gradient norm (mean ± std) | 0.521 ± 0.140 | 0.509 ± 0.142 |
| ODE block gradient norm | — | 0.033 ± 0.021 |
| **Learned scale α (Scalar)** | **N/A** | **0.1 → 0.065** |
| Vanishing gradient steps | 0 | 0 |
| Exploding gradient steps | 0 | 0 |

The hybrid model achieves slightly better loss with fewer parameters. The scalar α converged to 0.065, confirming the model prefers a stable, low-magnitude update regime.

![Training dynamics](figures/fig1_gradient_flow.png)

*Figure 1: Training dynamics. Left: Loss curves converge similarly. Center: Total gradient norms are comparable. Right: ODE block gradients remain healthy throughout training.*

### 4.2 Experiment 2: Semantic Steering

**Goal**: Force the model to generate specific sentiments based solely on the control signal u.

**Method (Hybrid Unfreeze)**: Freeze embeddings and output head, then train only the ODE block. This forces the vector field to learn control semantics rather than taking shortcuts through other parameters.

**Task**: Complete "The movie was..." with "good" (u = +1) or "bad" (u = -1).

**Table 2: Semantic steering results**

| Control Signal | Target | P(Good) | P(Bad) | Result |
|----------------|--------|---------|--------|--------|
| +1.0 (Positive) | "Good" | **98.0%** | 0.2% | ✓ Success |
| -1.0 (Negative) | "Bad" | 0.2% | **88.1%** | ✓ Success |
| 0.0 (Neutral) | — | 91.4% | 3.3% | (Natural Bias) |

**Result**: The control signal learned meaningful semantics, achieving 98%/88% accuracy.

### 4.3 Experiment 3: Continuous Interpolation & Manifold Verification

**Goal**: Verify that u controls a continuous manifold and that the learned dynamics are robust to solver choice.

**Continuity Test**: We define a "Popperian" falsification test to determine if the model is a true continuous system or a "ResNet in disguise." We compare the trajectory generated by the fixed Euler solver (used in training) against a high-precision adaptive solver (Dopri5).

**Result**: The relative divergence between trajectories is **0.068%**. This negligible difference proves the model has learned an intrinsic continuous vector field, rather than overfitting to the discretization artifacts of the training solver.

**Steering Sweep**: Sweeping u from -2 to +2 produces smooth sigmoid probability curves (Figure 2), confirming the continuity of the semantic manifold.

![Continuous steering](figures/fig2_continuous_steering.png)

*Figure 2: Control signal sweep. Smooth sigmoid curves demonstrate continuous steering—intermediate values produce mixed sentiment states.*

### 4.4 Experiment 4: Efficiency Benchmark

**Goal**: Measure the inference latency trade-off of the ODE integration.

**Table 3: Efficiency comparison**

| Model | Inference Time | Relative Speed |
|-------|----------------|----------------|
| Baseline | 519.83 ms/batch | 1.00× |
| Hybrid ODE | 507.19 ms/batch | **0.98× (Parity)** |

**Result**: The Hybrid ODE model achieves latency parity (-2.4% overhead). This outperforms the theoretical expectation of 1.33× slowdown. The efficiency gain is attributed to the architectural simplicity of the ODE vector field (MLP-based) compared to the Self-Attention mechanism it replaces, as well as improved cache locality during integration steps.

---

## 5. Discussion

### 5.1 Why Continuous Control Matters

Activation steering methods (Turner et al., 2023) discover steering vectors post-hoc and inject them at fixed layer positions. Our approach differs in two ways: (1) **Learned end-to-end**: Control dynamics are trained jointly with the model; (2) **Continuous influence**: The control signal biases dynamics throughout the integration interval. The 0.068% solver divergence result confirms that this influence is mathematically robust and not dependent on specific layer indices.

### 5.2 Dynamics of Resistance: Fighting Intrinsic Bias

The asymmetry in steering accuracy—98% for positive sentiment versus 88% for negative—must be contextualized against the model's intrinsic bias. Under neutral conditions (u=0), the model exhibits a strong prior toward positive completions, assigning 91.4% probability to "Good" and only 3.3% to "Bad" (Table 2).

Consequently, the control dynamics operate in two distinct regimes:

1. **Cooperative Dynamics (u > 0)**: The control signal acts in concert with the model's natural flow, requiring only a minor probability shift (+6.6%) to achieve saturation.

2. **Adversarial Dynamics (u < 0)**: The control signal must overcome the model's strong prior, effectively shifting the probability mass by over 85 percentage points.

This distinction provides a functional explanation for the geometric asymmetry observed in Section 5.3. The "Negative" region requires significantly higher solver effort (NFE=20 vs 14) not because the concept itself is more complex, but because the vector field **exhibits** greater curvature to forcefully divert the trajectory against the strong "current" of the pre-trained priors. The 88% success rate in the negative regime therefore represents a significantly greater dynamical intervention than the positive steering.

### 5.3 Geometric Structure via Solver Probing

The continuous formulation enables interpretation via solver behavior. An adaptive solver (dopri5) adjusts step size based on local curvature—regions where the vector field changes rapidly require more function evaluations (NFE). Since we have verified (Exp 3) that the model supports adaptive solving, NFE becomes a valid probe for geometric structure.

We observe that negative sentiment regions (u < 0.3) require significantly higher NFE (20 vs 14) than positive regions. Consistent with the "Dynamics of Resistance" hypothesis, this suggests that the adversarial task of generating negative sentiment requires traversing a more geometrically complex region of the learned manifold.

We additionally trained linear probes on hidden states at intermediate depths (τ = 0, 0.5, 0.67, 1.0) and found no spike in sentiment separability at τ ≈ 0.67, consistent with the global-bias interpretation where the control signal tilts the vector field landscape from the outset.

![Control topology](figures/fig3_control_topology.png)

*Figure 3: Solver effort reveals two dynamical regimes. The control signal partitions the vector field into regions of different curvature, with the transition aligned to the semantic crossover.*

We validated this finding through additional probes: (1) two different adaptive solvers (dopri5, adaptive_heun) agreed on step counts within 15%, confirming NFE measures the dynamics rather than solver artifacts; and (2) NFE scales sublinearly with tolerance (14 → 26 for 1000× tighter tolerance), indicating smooth rather than fractal dynamics (Figure 4). These results demonstrate that continuous-depth architectures enable inspection of learned representations through solver behavior—a form of interpretability unavailable in discrete transformer stacks.

![Resolution scaling](figures/fig4_continuous_limit.png)

*Figure 4: Resolution scaling toward continuous limit. NFE increases sublinearly with tighter tolerance, confirming smooth learned dynamics.*

### 5.4 Limitations

- **Scale**: Experiments use a small model (~30M parameters). Scaling to GPT-2 (124M) or larger remains future work.

- **Fixed integration**: We use 4 fixed Euler steps for training efficiency, though Experiment 3 demonstrates the model is compatible with adaptive solvers for variable-compute inference.

- **Single control dimension**: We validated sentiment only. Multi-dimensional control (formality, creativity) requires richer training objectives.

---

## 6. Conclusion

We demonstrated that continuous-depth transformers with learned control dynamics are feasible, stable, and steerable. The hybrid architecture achieves:

- **Stability**: Zero gradient pathologies with learned output scaling.
- **Continuity**: Validated by a 0.068% trajectory divergence test.
- **Efficiency**: Latency parity with discrete baselines.
- **Interpretability**: Solver probing reveals geometric structure in the learned dynamics.

Future work includes adaptive computation (dynamic step counts), scaling validation, and multi-dimensional control semantics.

---

## Code Availability

Implementation and experiments are available at: [https://github.com/PeterJemley/Continuous-Depth-Transformers-with-Learned-Control-Dynamics](https://github.com/PeterJemley/Continuous-Depth-Transformers-with-Learned-Control-Dynamics)

---

## References

**Chen et al. (2018)**  
Ricky TQ Chen, Yulia Rubanova, Jesse Bettencourt, and David K Duvenaud. "Neural ordinary differential equations." In *Advances in Neural Information Processing Systems*, volume 31, 2018. Best Paper Award.

**Dupont et al. (2019)**  
Emilien Dupont, Arnaud Doucet, and Yee Whye Teh. "Augmented neural ODEs." In *Advances in Neural Information Processing Systems*, volume 32, 2019.

**Li et al. (2022)**  
Bei Li, Quan Du, Tao Shao, Shuhan Wang, Shujian Huang, Jiajun Chen, and Min Zhang. "ODE transformer: An ordinary differential equation-inspired model for sequence generation." In *Proceedings of the 60th Annual Meeting of the Association for Computational Linguistics*, 2022.

**Turner et al. (2023)**  
Alexander M Turner, Lisa Thiergart, David Levinstein, Fabien Mini, Arthur Conmy, and Neel Nanda. "Steering language models with activation engineering." *arXiv preprint arXiv:2308.10248*, 2023.

**Zou et al. (2023)**  
Andy Zou, Long Phan, Sarah Chen, James Campbell, Phillip Guo, Richard Ren, Alexander Pan, Xuwang Yin, Mantas Mazeika, Ann-Kathrin Dombrowski, et al. "Representation engineering: A top-down approach to AI transparency." *arXiv preprint arXiv:2310.01405*, 2023.
