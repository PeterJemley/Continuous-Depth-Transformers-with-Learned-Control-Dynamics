"""The paper's model, restored verbatim from the deleted full experiments notebook.

Source: git commit 6b65d64, continuous_depth_transformers_experiments.ipynb.
- Cell 4: CausalSelfAttention, MLP, TransformerBlock, ODEFunc, ODEFlowBlock,
  BaselineTransformer, HybridODETransformer.
- Cell 6: InstrumentedODEFunc, AdaptiveODEFlowBlock, InstrumentedHybridTransformer.
- Cell 10: get_batch.  Cell 12: compute_gradient_stats.

Only the imports and the module-level `device` below are added; the original
cells used notebook globals for them. The code between the markers is unchanged.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchdiffeq import odeint_adjoint as odeint

device = torch.device('cpu')

# ---- BEGIN VERBATIM ----
class CausalSelfAttention(nn.Module):
    """
    Causal (masked) scaled dot-product self-attention.

    Args:
        d_model: Model dimension (must be divisible by n_heads)
        n_heads: Number of attention heads
        max_seq_len: Maximum sequence length for causal mask
        dropout: Dropout probability
    """
    def __init__(self, d_model, n_heads, max_seq_len=512, dropout=0.1):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model)
        self.proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('mask', torch.tril(torch.ones(max_seq_len, max_seq_len)))

    def forward(self, x):
        B, T, C = x.shape
        qkv = self.qkv(x).reshape(B, T, 3, self.n_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        att = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        att = att.masked_fill(self.mask[:T, :T] == 0, float('-inf'))
        att = F.softmax(att, dim=-1)
        att = self.dropout(att)
        y = (att @ v).transpose(1, 2).reshape(B, T, C)
        return self.proj(y)


class MLP(nn.Module):
    """Feed-forward network with GELU activation."""
    def __init__(self, d_model, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(d_model, 4 * d_model)
        self.fc2 = nn.Linear(4 * d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(F.gelu(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Standard transformer decoder block with pre-norm architecture."""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class ODEFunc(nn.Module):
    """
    Vector field for continuous-depth neural ODE: dH/dτ = α · F_θ(H, τ, u)

    Args:
        d_model: Model dimension
        n_heads: Number of attention heads
        control_dim: Dimension of control signal
        dropout: Dropout probability
    """
    def __init__(self, d_model, n_heads, control_dim=4, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.control_dim = control_dim
        self.time_embed = nn.Sequential(
            nn.Linear(1, d_model),
            nn.SiLU(),
            nn.Linear(d_model, d_model)
        )
        self.control_embed = nn.Linear(control_dim, d_model)
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = CausalSelfAttention(d_model, n_heads, dropout=dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model, dropout=dropout)
        self.output_scale = nn.Parameter(torch.tensor(0.1))
        self.control = None

    def forward(self, t, x):
        B, T, D = x.shape
        t_emb = self.time_embed(t.view(1, 1).expand(B, 1)).unsqueeze(1)
        c_emb = self.control_embed(self.control).unsqueeze(1) if self.control is not None else 0
        h = x + t_emb + c_emb
        dh = self.attn(self.ln1(h)) + self.mlp(self.ln2(h))
        return self.output_scale * dh


class ODEFlowBlock(nn.Module):
    """Continuous-depth ODE block replacing discrete transformer layers."""
    def __init__(self, d_model, n_heads, control_dim=4, n_steps=4, dropout=0.1):
        super().__init__()
        self.func = ODEFunc(d_model, n_heads, control_dim, dropout)
        self.n_steps = n_steps
        self.register_buffer('integration_times', torch.linspace(0, 1, n_steps + 1))

    def forward(self, x, control=None):
        self.func.control = control
        out = odeint(self.func, x, self.integration_times, method='euler')
        return out[-1]


class BaselineTransformer(nn.Module):
    """Standard transformer language model (decoder-only)."""
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=6,
                 max_seq_len=128, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.max_seq_len = max_seq_len

    def forward(self, idx):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.blocks:
            x = block(x)
        return self.head(self.ln_f(x))


class HybridODETransformer(nn.Module):
    """
    Hybrid transformer with continuous-depth ODE middle layers.

    Architecture: Early discrete layers -> ODE block -> Late discrete layers
    """
    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=6,
                 ode_start=2, ode_end=4, control_dim=4, n_steps=4,
                 max_seq_len=128, dropout=0.1):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.early_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(ode_start)
        ])
        self.ode_block = ODEFlowBlock(d_model, n_heads, control_dim, n_steps, dropout)
        self.late_blocks = nn.ModuleList([
            TransformerBlock(d_model, n_heads, dropout)
            for _ in range(n_layers - ode_end)
        ])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.control_dim = control_dim
        self.max_seq_len = max_seq_len

    def forward(self, idx, control=None):
        B, T = idx.shape
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for block in self.early_blocks:
            x = block(x)
        x = self.ode_block(x, control)
        for block in self.late_blocks:
            x = block(x)
        return self.head(self.ln_f(x))

class InstrumentedODEFunc(ODEFunc):
    """ODEFunc wrapper that counts function evaluations (NFE) and records trajectory."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.nfe = 0
        self._record_trajectory = False
        self.trajectory = []

    def forward(self, t, x):
        self.nfe += 1
        if self._record_trajectory:
            self.trajectory.append((t.item(), x.detach().cpu()))
        return super().forward(t, x)


class AdaptiveODEFlowBlock(ODEFlowBlock):
    """ODE block using adaptive dopri5 solver instead of fixed Euler steps."""
    def __init__(self, d_model, n_heads, control_dim=4, dropout=0.1):
        super().__init__(d_model, n_heads, control_dim, n_steps=1, dropout=dropout)
        self.func = InstrumentedODEFunc(d_model, n_heads, control_dim, dropout)
        self.solver_method = 'dopri5'
        self.rtol = 1e-3
        self.atol = 1e-3

    def forward(self, x, control=None):
        self.func.control = control
        self.func.nfe = 0
        self.func.trajectory = []
        t = torch.tensor([0.0, 1.0]).to(x.device)
        out = odeint(
            self.func, x, t,
            method=self.solver_method,
            rtol=self.rtol,
            atol=self.atol
        )
        self.last_nfe = self.func.nfe
        return out[-1]


class InstrumentedHybridTransformer(HybridODETransformer):
    """HybridODETransformer with instrumented adaptive ODE block."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.ode_block = AdaptiveODEFlowBlock(
            kwargs['d_model'],
            kwargs['n_heads'],
            kwargs['control_dim'],
            kwargs['dropout']
        )

    def get_last_nfe(self):
        return self.ode_block.last_nfe

def get_batch(tokens, batch_size, seq_len, device):
    """Sample a random batch from the token sequence."""
    ix = torch.randint(len(tokens) - seq_len - 1, (batch_size,))
    x = torch.stack([tokens[i:i+seq_len] for i in ix]).to(device)
    y = torch.stack([tokens[i+1:i+seq_len+1] for i in ix]).to(device)
    return x, y

def compute_gradient_stats(model):
    """Compute gradient norm statistics for monitoring training stability."""
    stats = {}
    total_norm = 0.0
    ode_norm = 0.0

    for name, param in model.named_parameters():
        if param.grad is not None:
            pn = param.grad.data.norm(2).item()
            total_norm += pn ** 2
            if 'ode' in name.lower():
                ode_norm += pn ** 2

    stats['total_grad_norm'] = total_norm ** 0.5
    if ode_norm > 0:
        stats['ode_grad_norm'] = ode_norm ** 0.5

    return stats
# ---- END VERBATIM ----
