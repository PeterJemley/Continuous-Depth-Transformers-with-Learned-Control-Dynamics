"""T1: the five-arm ladder that locates Model A's loss gap.

Implements T1_PREREGISTRATION.md in this folder, using the WikiText-103 study's resumable
training code (study.train_run) at 10% of the full budget. Token files are read from the
study's Drive folder; checkpoints and results go to this job's own Drive folder, so a
session that ends is resumed by running the job again.

    python t1_ladder.py --out OUT --work /content/drive/MyDrive/cdt-colab/t1_ladder
    STUDY_TEST=1 python t1_ladder.py --out OUT --work W --source LOCAL_STUDY_FOLDER   # small test

Arms: B6 (baseline), B5 (one block fewer), T4 (one standard block applied 4 times),
AX (Model A with exact gradients) and AA (Model A with adjoint gradients, as trained).
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'wikitext103-training'))

import study  # noqa: E402  (sets the model config, test mode and device)
import paper_model as pm  # noqa: E402
from torchdiffeq import odeint as odeint_exact  # noqa: E402

DEV, TEST, log = study.DEV, study.TEST, study.log
STUDY_RESULTS = HERE.parent / 'wikitext103-training' / 'results'

TOTAL = 60 if TEST else 2879                       # 10% of the full budget, as in stage 1
SHORT = round(0.8 * TOTAL)                         # injected effect: 80% of the budget
GRID = [3e-4, 6e-4, 1.2e-3, 2.4e-3]
SWEEP_SEED = 100
SEEDS = [2001, 2002] if TEST else [2001, 2002, 2003, 2004, 2005]
NULL_INIT_OFFSET = 50
ARMS = ['B6', 'B5', 'T4', 'AX', 'AA']
NEW_ARMS = ['B5', 'T4', 'AX']                     # B6 and AA reuse stage 1's sweep
STAGE1_KIND = {'B6': 'baseline', 'AA': 'hybrid'}


# ------------------------------------------------------------------ the five models

class TiedTransformer(nn.Module):
    """The baseline with layers 2-3 replaced by one standard block applied `repeats` times."""

    def __init__(self, vocab_size, d_model=256, n_heads=4, n_layers=6, max_seq_len=128, dropout=0.1, repeats=4):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.early_blocks = nn.ModuleList([pm.TransformerBlock(d_model, n_heads, dropout) for _ in range(2)])
        self.shared = pm.TransformerBlock(d_model, n_heads, dropout)
        self.late_blocks = nn.ModuleList([pm.TransformerBlock(d_model, n_heads, dropout) for _ in range(n_layers - 4)])
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.repeats = repeats

    def forward(self, idx):
        T = idx.shape[1]
        x = self.tok_emb(idx) + self.pos_emb(torch.arange(T, device=idx.device))
        for b in self.early_blocks:
            x = b(x)
        for _ in range(self.repeats):
            x = self.shared(x)
        for b in self.late_blocks:
            x = b(x)
        return self.head(self.ln_f(x))


def _exact_ode_forward(self, x, control=None):
    """ODEFlowBlock.forward with ordinary backpropagation through the Euler steps in place of the adjoint."""
    self.func.control = control
    return odeint_exact(self.func, x, self.integration_times, method='euler')[-1]


def make_exact(model):
    model.ode_block.forward = types.MethodType(_exact_ode_forward, model.ode_block)
    model.gradient_method = 'exact'
    return model


def new_model(kind, seed):
    """Every arm is built right after seeding, like the study's models."""
    if kind in ('baseline', 'hybrid'):
        return study_new_model(kind, seed)
    torch.manual_seed(seed)
    if kind == 'B6':
        m = pm.BaselineTransformer(**study.CFG)
    elif kind == 'B5':
        m = pm.BaselineTransformer(**{**study.CFG, 'n_layers': 5})
    elif kind == 'T4':
        m = TiedTransformer(**study.CFG)
    elif kind in ('AX', 'AA'):
        m = pm.HybridODETransformer(**study.CFG, **study.HYB)
        if kind == 'AX':
            make_exact(m)
    else:
        raise ValueError(kind)
    return m.to(DEV)


study_new_model = study.new_model
study.new_model = new_model       # train_run builds its model through this name
study.VAL_EVERY = 400             # validation at step 500, every 400 steps, and at the end


# ------------------------------------------------------------------ checks

def construction_check():
    n = {k: sum(p.numel() for p in new_model(k, 0).parameters()) for k in ARMS}
    block = sum(p.numel() for p in pm.TransformerBlock(study.CFG['d_model'], study.CFG['n_heads']).parameters())
    return {'params': n, 'block_params': block,
            'pass': n['B5'] == n['B6'] - block and n['T4'] == n['B5'] and n['AX'] == n['AA']}


def same_model_check(train, seed):
    """AX and AA with the same seed: same weights, same first-batch loss, same downstream gradients."""
    gen = torch.Generator().manual_seed(seed)
    ix = torch.randint(len(train) - study.SEQ - 1, (study.BATCH,), generator=gen).numpy()
    b = torch.from_numpy(np.stack([np.asarray(train[i:i + study.SEQ + 1], dtype=np.int64) for i in ix])).to(DEV)
    x, y = b[:, :-1], b[:, 1:]
    out = {}
    for kind in ('AX', 'AA'):
        m = new_model(kind, seed)
        m.train()
        torch.manual_seed(seed + 7)
        loss = F.cross_entropy(m(x).reshape(-1, study.CFG['vocab_size']), y.reshape(-1))
        loss.backward()
        down = torch.cat([p.grad.reshape(-1) for blk in m.late_blocks for p in blk.parameters()]
                         + [p.grad.reshape(-1) for p in m.head.parameters()])
        weights = torch.cat([p.detach().reshape(-1) for p in m.parameters()])
        out[kind] = (loss.item(), down, weights)
    rel = abs(out['AX'][0] - out['AA'][0]) / abs(out['AA'][0])
    cosd = F.cosine_similarity(out['AX'][1].double(), out['AA'][1].double(), dim=0).item()
    same_w = torch.equal(out['AX'][2], out['AA'][2])
    return {'seed': seed, 'loss_rel_diff': rel, 'downstream_cos': cosd, 'same_initial_weights': same_w,
            'pass': same_w and rel <= 1e-6 and cosd >= 0.99999}


# ------------------------------------------------------------------ main

def stage1_choice(kind):
    """B6 and AA: the learning rate stage 1 chose, recomputed from its sweep records."""
    vals = {}
    for lr in GRID:
        p = STUDY_RESULTS / f'sweep_{kind}_lr{lr:g}.json'
        if p.exists():
            vals[lr] = json.loads(p.read_text())['final_val_loss']
    return min(vals, key=vals.get), vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--work', required=True, help="this job's Drive folder, for checkpoints and results")
    ap.add_argument('--source', default=None, help='the WikiText-103 study folder (default: sibling wt103_study)')
    ap.add_argument('--local', default=None)
    args = ap.parse_args()
    work = Path(args.work)
    source = Path(args.source) if args.source else work.parent / 'wt103_study'
    if args.local:
        local = Path(args.local)
    elif Path('/content').is_dir() and not TEST:
        local = Path('/content/cdt_local') / work.name
    else:
        local = work.parent / f'_local_{work.name}'
    data_store = study.Store(source, local)       # reads token files only; never writes to the study folder
    store = study.Store(work, local)              # checkpoints and results for T1
    out = Path(args.out)
    (out / 'results').mkdir(parents=True, exist_ok=True)
    log('device:', DEV, '| test mode:', TEST, '| study folder (data):', source, '| T1 folder:', work, '| local:', local)
    store.pull_dir('results')
    try:
        meta = study.prepare_data(data_store, None)
        train = study.load_tokens(store, meta, 'train')
        first = study.gpt2_encoding().decode(np.asarray(train[:200], dtype=np.int64).tolist())
        summary_path = store.L('results/t1_summary.json')
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        if 'checks' not in summary:
            gen = torch.Generator().manual_seed(SEEDS[0])
            i0 = int(torch.randint(len(train) - study.SEQ - 1, (study.BATCH,), generator=gen)[0])
            sample = study.gpt2_encoding().decode(np.asarray(train[i0:i0 + study.SEQ], dtype=np.int64).tolist())
            log('first window of seed 2001\'s first batch, decoded:', repr(sample[:300]))
            summary['checks'] = {'construction': construction_check(),
                                 'same_model': [same_model_check(train, s) for s in SEEDS],
                                 'sample_text': sample, 'train_start_text': first}
            summary_path.write_text(json.dumps(summary, indent=2))
            store.push('results/t1_summary.json')
        c = summary['checks']
        log('construction check:', c['construction']['pass'], c['construction']['params'])
        log('same-model check:', [x['pass'] for x in c['same_model']])
        if not (c['construction']['pass'] and all(x['pass'] for x in c['same_model'])):
            raise SystemExit('A must-match check failed before training. Stopping, as preregistered.')

        # Learning rates: stage 1's choice for B6 and AA, a new sweep for the other arms.
        lrs = {}
        for arm, kind in STAGE1_KIND.items():
            lrs[arm], _ = stage1_choice(kind)
        grid = GRID[:2] if TEST else GRID
        sweeps = {}
        for arm in NEW_ARMS:
            sweeps[arm] = {}
            for lr in grid:
                r = study.train_run(store, meta, f'sweep_{arm}_lr{lr:g}', arm, SWEEP_SEED, lr, TOTAL)
                sweeps[arm][lr] = r['final_val_loss']
            lrs[arm] = min(sweeps[arm], key=sweeps[arm].get)
        summary['learning_rates'] = {a: lrs[a] for a in ARMS}
        summary['sweeps'] = {a: {str(k): v for k, v in d.items()} for a, d in sweeps.items()}
        summary['stage1_sweeps'] = {a: {str(k): v for k, v in stage1_choice(kind)[1].items()} for a, kind in STAGE1_KIND.items()}
        summary_path.write_text(json.dumps(summary, indent=2))
        store.push('results/t1_summary.json')
        log('learning rates:', summary['learning_rates'])

        # Main runs, seed by seed, so that complete pairs accumulate if a session ends early.
        for seed in SEEDS:
            for arm in ARMS:
                study.train_run(store, meta, f'main_{arm}_seed{seed}', arm, seed, lrs[arm], TOTAL)
        for seed in SEEDS:
            study.train_run(store, meta, f'null_B6_seed{seed}', 'B6', seed, lrs['B6'], TOTAL, init_seed=seed + NULL_INIT_OFFSET)
            study.train_run(store, meta, f'short_B6_seed{seed}', 'B6', seed, lrs['B6'], SHORT)
        log('T1 complete: all runs finished')
    finally:
        dest = out / 'results'
        for f in (store.local / 'results').glob('*.json'):
            if f.name.startswith(('sweep_', 'main_', 'null_', 'short_', 't1_summary')):
                (dest / f.name).write_bytes(f.read_bytes())
        log(f'copied {len(list(dest.glob("*.json")))} result files for upload')
        log('waiting for background Drive saves to finish (up to 15 minutes)')
        store.flush()
        failures = store.failures + data_store.failures
        (out / 'drive_sync.json').write_text(json.dumps({'failures': failures}, indent=2))
        log('Drive saves finished' if not failures else f'Drive problems this session: {failures}')


if __name__ == '__main__':
    main()
