"""Hybrid continuous-depth transformer with a learned control signal.

This module is the single shared definition of the architecture used by the
falsification test and efficiency benchmark notebooks. A six-layer GPT-style
transformer keeps layers 0-1 and 4-5 as ordinary discrete blocks and replaces
layers 2-3 with a Neural ODE block:

    dH/dtau = alpha * F_theta(H, tau, u),    tau in [0, 1]

where u is a low-dimensional control vector concatenated onto the hidden state
inside the vector field, and alpha is a learned output scale initialised to 0.1.

Example:

    import torch
    from model import HybridTransformer, DEFAULT_CONFIG

    model = HybridTransformer(DEFAULT_CONFIG)
    idx = torch.randint(0, DEFAULT_CONFIG['vocab_size'], (2, DEFAULT_CONFIG['seq_len']))
    u = torch.zeros(2, DEFAULT_CONFIG['control_dim'])
    u[:, 0] = 1.0
    logits = model(idx, u)                              # fixed 4-step Euler solver
    logits_adaptive = model(idx, u, use_adaptive=True)  # adaptive Dopri5 solver
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint as odeint_direct
from torchdiffeq import odeint_adjoint

__all__ = [
    'DEFAULT_CONFIG',
    'CausalSelfAttention',
    'MLP',
    'StandardBlock',
    'ControlledVectorField',
    'ContinuousDepthBlock',
    'BaselineTransformer',
    'HybridTransformer',
]

# Configuration from Section 4 of the paper.
DEFAULT_CONFIG = {
    'd_model': 256,
    'n_heads': 4,
    'n_layers': 6,          # Total effective layers (the baseline has all six)
    'ode_layer_start': 2,   # The hybrid replaces layers 2 and 3 ...
    'ode_layer_end': 4,     # ... with one continuous block
    'vocab_size': 33278,    # WikiText-2 vocabulary size (approx.)
    'seq_len': 32,
    'control_dim': 4,       # Dimension of the control signal u
    'steps_for_ode': 4,     # Fixed Euler steps over tau in [0, 1]
    'use_adjoint': True,    # Adjoint method for O(1)-memory gradients
}


class CausalSelfAttention(nn.Module):
    def __init__(self, d_model, n_heads, max_len=512):
        super().__init__()
        self.c_attn = nn.Linear(d_model, 3 * d_model)
        self.c_proj = nn.Linear(d_model, d_model)
        self.n_heads = n_heads
        self.d_model = d_model
        self.register_buffer("bias", torch.tril(torch.ones(max_len, max_len))
                                     .view(1, 1, max_len, max_len))

    def forward(self, x):
        B, T, C = x.size()
        q, k, v = self.c_attn(x).split(self.d_model, dim=2)
        k = k.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        q = q.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)
        v = v.view(B, T, self.n_heads, C // self.n_heads).transpose(1, 2)

        att = (q @ k.transpose(-2, -1)) * (1.0 / math.sqrt(k.size(-1)))
        att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        y = att @ v
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.c_proj(y)


class MLP(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, 4 * d_model),
            nn.GELU(),
            nn.Linear(4 * d_model, d_model)
        )

    def forward(self, x):
        return self.net(x)


class StandardBlock(nn.Module):
    """A pre-norm discrete transformer block."""

    def __init__(self, d_model, n_heads):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ControlledVectorField(nn.Module):
    """The vector field F_theta(h, t, u) = MLP(concat(h, u)).

    The control vector u is injected explicitly by concatenating it onto the
    hidden state at every sequence position.
    """

    def __init__(self, d_model, control_dim):
        super().__init__()
        input_dim = d_model + control_dim
        self.net = nn.Sequential(
            nn.Linear(input_dim, d_model * 2),
            nn.Softplus(),  # Smooth activation for better ODE properties
            nn.Linear(d_model * 2, d_model)
        )

        # Stability initialisation: start with a near-flat vector field.
        nn.init.normal_(self.net[-1].weight, mean=0, std=0.01)
        nn.init.constant_(self.net[-1].bias, 0)

    def forward(self, t, h, u):
        # h: [batch, seq, d_model], u: [batch, control_dim]
        seq_len = h.shape[1]
        u_expanded = u.unsqueeze(1).expand(-1, seq_len, -1)
        state = torch.cat([h, u_expanded], dim=-1)
        return self.net(state)


class ContinuousDepthBlock(nn.Module):
    """Integrates dh/dtau = alpha * F_theta(h, tau, u) from tau = 0 to tau = 1.

    By default the block uses a fixed-step Euler solver with ``steps`` steps,
    which is what the model is trained with. Passing ``use_adaptive=True`` to
    ``forward`` switches to the adaptive Dopri5 solver, which the Solver
    Invariance Test uses as a reference.

    ``adjoint=True`` integrates with ``torchdiffeq.odeint_adjoint`` so that
    gradients use constant memory; ``adjoint=False`` uses plain
    ``torchdiffeq.odeint``.
    """

    def __init__(self, d_model, control_dim, steps=4, adjoint=True):
        super().__init__()
        self.vector_field = ControlledVectorField(d_model, control_dim)

        # Learned output scale, initialised to 0.1 (Section 3.3).
        self.alpha = nn.Parameter(torch.tensor(0.1))

        # Training defaults (fixed step).
        self.adjoint = adjoint
        self.train_method = 'euler'
        self.train_options = {'step_size': 1.0 / steps}

    def forward(self, h, u, use_adaptive=False):
        # Bind u so the solver sees f(t, h).
        def func(t, x):
            return self.alpha * self.vector_field(t, x, u)

        integration_times = torch.tensor([0, 1]).float().to(h.device)

        if use_adaptive:
            method = 'dopri5'
            options = {}
        else:
            method = self.train_method
            options = self.train_options

        if self.adjoint:
            # Every parameter that func depends on.
            adjoint_params = tuple(self.vector_field.parameters()) + (self.alpha,)
            state = odeint_adjoint(func, h, integration_times, method=method,
                                   options=options, adjoint_params=adjoint_params)
        else:
            state = odeint_direct(func, h, integration_times, method=method,
                                  options=options)
        return state[1]


class BaselineTransformer(nn.Module):
    """A standard discrete transformer with ``n_layers`` blocks."""

    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config['vocab_size'], config['d_model'])
        self.pos_embed = nn.Parameter(torch.zeros(1, config['seq_len'], config['d_model']))

        self.layers = nn.ModuleList([
            StandardBlock(config['d_model'], config['n_heads'])
            for _ in range(config['n_layers'])
        ])

        self.ln_f = nn.LayerNorm(config['d_model'])
        self.head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)

    def forward(self, idx, u=None):  # u is accepted for a shared interface and ignored
        B, T = idx.size()
        x = self.embed(idx) + self.pos_embed[:, :T, :]
        for layer in self.layers:
            x = layer(x)
        return self.head(self.ln_f(x))


class HybridTransformer(nn.Module):
    """Discrete early layers, a continuous ODE block, then discrete late layers.

    With the default configuration this is layers 0-1, the ODE block in place
    of layers 2-3, and layers 4-5.
    """

    def __init__(self, config):
        super().__init__()
        self.embed = nn.Embedding(config['vocab_size'], config['d_model'])
        self.pos_embed = nn.Parameter(torch.zeros(1, config['seq_len'], config['d_model']))

        # 1. Discrete early layers.
        self.early_layers = nn.ModuleList([
            StandardBlock(config['d_model'], config['n_heads'])
            for _ in range(config['ode_layer_start'])
        ])

        # 2. Continuous ODE block in place of layers ode_layer_start .. ode_layer_end - 1.
        self.ode_block = ContinuousDepthBlock(
            config['d_model'], config['control_dim'],
            steps=config.get('steps_for_ode', 4),
            adjoint=config.get('use_adjoint', True),
        )

        # 3. Discrete late layers.
        remaining_layers = config['n_layers'] - config['ode_layer_end']
        self.late_layers = nn.ModuleList([
            StandardBlock(config['d_model'], config['n_heads'])
            for _ in range(remaining_layers)
        ])

        self.ln_f = nn.LayerNorm(config['d_model'])
        self.head = nn.Linear(config['d_model'], config['vocab_size'], bias=False)

    def forward(self, idx, u, use_adaptive=False):
        B, T = idx.size()
        x = self.embed(idx) + self.pos_embed[:, :T, :]

        for layer in self.early_layers:
            x = layer(x)

        x = self.ode_block(x, u, use_adaptive=use_adaptive)

        for layer in self.late_layers:
            x = layer(x)

        return self.head(self.ln_f(x))
