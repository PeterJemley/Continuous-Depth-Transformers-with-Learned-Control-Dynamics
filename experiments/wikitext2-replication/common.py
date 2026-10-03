"""Shared configuration, data access, run records, and evaluation for the replication.

Everything here follows PREREGISTRATION.md. Set REPL_TEST=1 to run the whole
pipeline on synthetic tokens with a tiny model, for checking the code only;
test results go to results_test/ and are never analysed as real results.
"""
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(HERE))

from torchdiffeq import odeint_adjoint as odeint  # noqa: E402

import paper_model as pm  # noqa: E402

TEST = os.environ.get('REPL_TEST') == '1'
RESULTS = Path(os.environ.get('REPL_RESULTS_DIR', HERE / ('results_test' if TEST else 'results')))
CKPT = Path(os.environ.get('REPL_CKPT_DIR', HERE / ('checkpoints_test' if TEST else 'checkpoints')))
SEEDS = [42, 0, 1, 2, 3, 4, 5, 6]

MODEL_CFG = dict(vocab_size=50257, d_model=256, n_heads=4, n_layers=6, max_seq_len=128, dropout=0.1)
HYBRID_KW = dict(ode_start=2, ode_end=4, control_dim=4, n_steps=4)
TRAIN_CFG = dict(n_steps=500, batch_size=32, seq_len=64, lr=3e-4)
if TEST:
    MODEL_CFG.update(d_model=32, n_heads=2)
    TRAIN_CFG.update(n_steps=6)

POS_TOKEN_CORRECT, POS_TOKEN_PAPER, NEG_TOKEN = 922, 220, 2089   # ' good', ' ' (paper), ' bad'
CONTEXT_IDS = [464, 3807, 373]                                    # 'The movie was'
HELD_OUT_NOUNS = ['film', 'book', 'meal', 'show', 'concert', 'game', 'hotel', 'flight', 'album', 'play',
                  'story', 'class', 'trip', 'party', 'lecture', 'restaurant', 'service', 'product',
                  'series', 'performance']

torch.set_num_threads(int(os.environ.get('REPL_THREADS', '4')))


# ---------------------------------------------------------------- run records

def git_state():
    def run(*args):
        return subprocess.run(['git', '-C', str(REPO), *args], capture_output=True, text=True).stdout.strip()
    return {'commit': run('rev-parse', 'HEAD'), 'dirty': bool(run('status', '--porcelain', '--untracked-files=no', '--', 'experiments', 'pinned_data.py', 'data'))}


def environment():
    import numpy
    import torchdiffeq
    try:
        import sklearn
        skv = sklearn.__version__
    except ImportError:
        skv = None
    return {'python': platform.python_version(), 'torch': torch.__version__, 'torchdiffeq': torchdiffeq.__version__,
            'numpy': numpy.__version__, 'sklearn': skv, 'platform': platform.platform(),
            'processor': platform.processor() or platform.machine(), 'cpu_count': os.cpu_count(),
            'torch_threads': torch.get_num_threads()}


def data_fingerprint():
    if TEST:
        return {'synthetic': True}
    m = json.loads((REPO / 'data' / 'MANIFEST.json').read_text())
    return {'wikitext-2-raw-v1': m['wikitext-2-raw-v1']['files'], 'gpt2': m['gpt2_tokenizer']['files']}


def write_result(name, payload):
    RESULTS.mkdir(parents=True, exist_ok=True)
    record = {'name': name, 'written_at': time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
              'git': git_state(), 'environment': environment(), 'data': data_fingerprint(),
              'test_mode': TEST, 'result': payload}
    path = RESULTS / f'{name}.json'
    path.write_text(json.dumps(record, indent=2) + '\n')
    return path


def read_result(name):
    return json.loads((RESULTS / f'{name}.json').read_text())['result']


# ---------------------------------------------------------------- data

_TOKENS = {}


def tokens(split):
    """GPT-2 tokens of a WikiText-2 split, joined as in the original notebook."""
    if split not in _TOKENS:
        if TEST:
            g = torch.Generator().manual_seed({'train': 1, 'validation': 2, 'test': 3}[split])
            _TOKENS[split] = torch.randint(0, MODEL_CFG['vocab_size'], (60000 if split == 'train' else 70000,), generator=g)
        else:
            from pinned_data import wikitext2_tokens
            _TOKENS[split] = wikitext2_tokens(split)
    return _TOKENS[split]


def val_windows(seq_len=64, limit=None):
    """Consecutive length-64 windows over the whole validation split, with next-token targets."""
    t = tokens('validation')
    n = (len(t) - 1) // seq_len
    if limit is not None:
        n = min(n, limit)
    return t[:n * seq_len].view(n, seq_len), t[1:n * seq_len + 1].view(n, seq_len)


# ---------------------------------------------------------------- models

def new_model(kind, seed):
    """Seed, then construct, exactly as the original notebook did before each model."""
    torch.manual_seed(seed)
    if kind == 'baseline':
        return pm.BaselineTransformer(**MODEL_CFG)
    if kind == 'hybrid':
        return pm.HybridODETransformer(**MODEL_CFG, **HYBRID_KW)
    raise ValueError(kind)


def ckpt_path(name):
    CKPT.mkdir(parents=True, exist_ok=True)
    return CKPT / f'{name}.pt'


def load_model(kind, name):
    m = new_model(kind, 0)
    m.load_state_dict(torch.load(ckpt_path(name), map_location='cpu'))
    m.eval()
    return m


def hybrid_forward(model, idx, control=None, method='euler', n_steps=4, rtol=1e-5, atol=1e-5, scale=None):
    """HybridODETransformer.forward with a chosen ODE solver. With method='euler' and
    n_steps=4 it is identical to model(idx, control); run_solver.py checks this."""
    B, T = idx.shape
    x = model.tok_emb(idx) + model.pos_emb(torch.arange(T, device=idx.device))
    for block in model.early_blocks:
        x = block(x)
    func = model.ode_block.func
    func.control = control
    saved = None
    if scale is not None:
        saved = func.output_scale.data.clone()
        func.output_scale.data.mul_(scale)
    try:
        if method in ('euler', 'rk4'):
            out = odeint(func, x, torch.linspace(0, 1, n_steps + 1), method=method)
        else:
            out = odeint(func, x, torch.tensor([0.0, 1.0]), method=method, rtol=rtol, atol=atol)
    finally:
        if saved is not None:
            func.output_scale.data.copy_(saved)
    x = out[-1]
    for block in model.late_blocks:
        x = block(x)
    return model.head(model.ln_f(x))


@torch.no_grad()
def val_loss(model, forward=None, limit=None, batch=32):
    """Mean next-token cross-entropy over validation windows, in nats."""
    model.eval()
    xs, ys = val_windows(limit=limit)
    total, count = 0.0, 0
    for i in range(0, len(xs), batch):
        x, y = xs[i:i + batch], ys[i:i + batch]
        logits = forward(model, x) if forward else model(x)
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction='sum').item()
        count += y.numel()
    return total / count


# ---------------------------------------------------------------- statistics

T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228}


def t_interval(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    half = T975[n - 1] * sd / math.sqrt(n)
    return m, sd, m - half, m + half


def sign_flip_p(diffs):
    import itertools
    obs = abs(sum(diffs) / len(diffs))
    hits = total = 0
    for signs in itertools.product((1, -1), repeat=len(diffs)):
        total += 1
        hits += abs(sum(s * d for s, d in zip(signs, diffs)) / len(diffs)) >= obs - 1e-12
    return hits / total


def wilson(k, n, z=1.959964):
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return c - h, c + h


def binom_two_sided(k, n, p=0.5):
    pmf = [math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    return min(1.0, sum(q for q in pmf if q <= pmf[k] * (1 + 1e-9)))
