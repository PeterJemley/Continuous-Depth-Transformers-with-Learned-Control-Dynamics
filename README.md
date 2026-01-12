# Continuous-Depth Transformers with Learned Control Dynamics

This repository contains the implementation and experiments for:

> **Continuous-Depth Transformers with Learned Control Dynamics**  
> Peter Jemley  
> January 2026

📄 **[Read the full paper](PAPER.md)** | 📓 **[Run the experiments](continuous_depth_transformers_experiments.ipynb)**

## Abstract

We present a hybrid transformer architecture that replaces discrete middle layers with a continuous-depth Neural Ordinary Differential Equation (ODE) block, enabling inference-time control over generation attributes via a learned steering signal. The control signal `u` steers the model's output through a continuous manifold of semantic states, achieving 98%/88% accuracy for positive/negative sentiment control with only 1.27× inference overhead.

## Key Results

| Experiment | Result |
|------------|--------|
| Gradient Stability | Zero exploding/vanishing gradients |
| Semantic Steering | 98%/88% accuracy (positive/negative) |
| Control Continuity | Smooth sigmoid interpolation |
| Efficiency | 1.27× overhead (vs 1.33× theoretical) |
| Solver Probing | Two distinct dynamical regimes revealed |

## Architecture

The hybrid architecture uses a "sandwich" design:

```
Input → [Discrete Layers 0-1] → [ODE Block] → [Discrete Layers 4-5] → Output
                                     ↑
                              Control Signal u
```

The ODE block replaces layers 2-3 with continuous dynamics:

```
dH/dτ = α · F_θ(H, τ, u)
```

where:
- `H(τ)` is the hidden state at continuous depth τ ∈ [0, 1]
- `F_θ` is a learned vector field (attention + MLP)
- `u` is a low-dimensional control signal
- `α` is a learned output scale (initialized to 0.1) for stability

## Installation

```bash
pip install torch torchdiffeq datasets tiktoken
```

## Quick Start

```python
from model import HybridODETransformer

# Initialize model
model = HybridODETransformer(
    vocab_size=50257,
    d_model=256,
    n_heads=4,
    n_layers=6,
    ode_start=2,
    ode_end=4,
    control_dim=4
)

# Forward pass with control signal
control = torch.zeros(batch_size, 4)
control[:, 0] = 1.0  # Positive sentiment
logits = model(input_ids, control=control)
```

## Experiments

Run the complete experiment suite in Google Colab:

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PeterJemley/continuous-depth-transformers/blob/main/continuous_depth_transformers_experiments.ipynb)

### Experiment 1: Gradient Flow Validation

Verifies stable gradient propagation through the ODE block.

![Gradient Flow](figures/fig1_gradient_flow.png)

### Experiment 2: Semantic Steering

The "Hybrid Unfreeze" method trains only the ODE block to learn control semantics:

| Control Signal | P(good) | P(bad) | Result |
|----------------|---------|--------|--------|
| +1.0 (Positive) | **98.0%** | 0.2% | ✓ |
| -1.0 (Negative) | 0.2% | **88.1%** | ✓ |
| 0.0 (Neutral) | 91.4% | 3.3% | Natural bias |

### Experiment 3: Continuous Interpolation

Smooth sigmoid curves demonstrate the model learns a continuous manifold:

![Continuous Steering](figures/fig2_continuous_steering.png)

### Solver Probing: Geometric Structure

The adaptive ODE solver reveals that positive and negative sentiment occupy geometrically distinct regions of the learned dynamics:

![Control Topology](figures/fig3_control_topology.png)

**Key finding**: The negative sentiment regime requires 43% more solver steps (NFE=20 vs NFE=14), indicating higher curvature in that region of the vector field.

## Repository Structure

```
├── continuous_depth_transformers_experiments.ipynb  # Complete Colab notebook
├── main.tex                                         # Paper source
├── figures/
│   ├── fig1_gradient_flow.png
│   ├── fig2_continuous_steering.png
│   ├── fig3_control_topology.png
│   └── fig4_continuous_limit.png
└── README.md
```

## Core Components

### ODEFunc

The vector field parameterization:

```python
class ODEFunc(nn.Module):
    def forward(self, t, x):
        # Time embedding
        t_emb = self.time_embed(t)
        # Control embedding  
        c_emb = self.control_embed(self.control)
        # Dynamics
        h = x + t_emb + c_emb
        dh = self.attn(self.ln1(h)) + self.mlp(self.ln2(h))
        return self.output_scale * dh  # α · F_θ
```

### Hybrid Unfreeze Training

The key to learning meaningful control:

```python
# Phase 1: Warmup (all parameters)
# Phase 2: Intermediate (control_embed only)
# Phase 3: Hybrid Unfreeze (ODE block only)
for p in model.parameters():
    p.requires_grad = False
for p in model.ode_block.parameters():
    p.requires_grad = True
```

### Solver Probing

Use adaptive solvers as interpretability instruments:

```python
class InstrumentedODEFunc(ODEFunc):
    def forward(self, t, x):
        self.nfe += 1  # Count function evaluations
        return super().forward(t, x)
```

## Citation

```bibtex
@article{jemley2026continuous,
  title={Continuous-Depth Transformers with Learned Control Dynamics},
  author={Jemley, Peter},
  year={2026}
}
```

## License

MIT

## Acknowledgments

This work builds on Neural ODEs (Chen et al., 2018) and activation steering (Turner et al., 2023).
