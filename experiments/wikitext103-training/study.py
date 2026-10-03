"""The WikiText-103 training study, as preregistered in PREREGISTRATION.md.

Three stages, each run as a Colab job through colab/runner.ipynb:
  stage 1  data preparation, the fp16 gradient check, the learning-rate sweep, and the spread runs
  stage 2  the 12 main runs: 6 seeds x {baseline, Model A}
  stage 3  test loss for every final model, and the solver-dependence test (S3)

Every run checkpoints to the Drive working folder about every 10 minutes and
resumes from there, and a finished run is never repeated, so pressing Run all
again simply continues. Results accumulate in <work>/results and are copied to
the job's output folder for upload.

Training length: total steps = floor(training tokens / 4096), one pass's worth of
tokens. Windows are sampled at random, as in the paper's code, from a generator
seeded by the run seed alone, so both models of a seed see the same batches.
"""
import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import time
import traceback
import urllib.request
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'experiments' / 'wikitext2-replication'))

import paper_model as pm  # noqa: E402
from pinned_data import gpt2_encoding  # noqa: E402
from torchdiffeq import odeint_adjoint as odeint  # noqa: E402

TEST = os.environ.get('STUDY_TEST') == '1'
DEV = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
AMP = DEV.type == 'cuda'

CFG = dict(vocab_size=50257, d_model=256, n_heads=4, n_layers=6, max_seq_len=128, dropout=0.1)
HYB = dict(ode_start=2, ode_end=4, control_dim=4, n_steps=4)
SEQ, BATCH, WARMUP, MIN_LR_RATIO = 128, 32, 500, 0.1
SWEEP_LRS, SWEEP_FRACTION, SWEEP_SEED, SPREAD_SEED = [3e-4, 6e-4, 1.2e-3], 0.1, 100, 101
MAIN_SEEDS = [1001, 1002, 1003, 1004, 1005, 1006]
VAL_EVERY, CKPT_SECONDS = 2000, 600
SOLVERS = {'euler4': dict(method='euler', n_steps=4), 'euler8': dict(method='euler', n_steps=8),
           'euler16': dict(method='euler', n_steps=16), 'rk4_4': dict(method='rk4', n_steps=4),
           'dopri5_1e-5': dict(method='dopri5', rtol=1e-5, atol=1e-5)}
SCALES = [2, 4, 8, 16]
if TEST:
    CFG.update(d_model=32, n_heads=2)
    WARMUP, VAL_EVERY, CKPT_SECONDS = 5, 20, 5
    MAIN_SEEDS = MAIN_SEEDS[:2]


def log(*a):
    print(time.strftime('%H:%M:%S'), *a, flush=True)


def git_commit():
    return subprocess.run(['git', '-C', str(REPO), 'rev-parse', 'HEAD'], capture_output=True, text=True).stdout.strip()


def sha256_file(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def env_record():
    return {'torch': torch.__version__, 'device': str(DEV),
            'gpu': torch.cuda.get_device_name(0) if DEV.type == 'cuda' else None, 'amp_fp16': AMP,
            'code_commit': git_commit(), 'test_mode': TEST}


# ------------------------------------------------------------------ data

def safe_split(row):
    """Cutting the joined text before this row cannot change the tokens (see verification in the commit log)."""
    return len(row) > 1 and row[0] == ' ' and not row[1].isspace()


def encode_rows(enc, rows, rows_per_chunk=20000):
    """Tokens of '\\n'.join(non-empty rows), encoded in pieces cut only at safe points."""
    rows = [r for r in rows if r.strip()]
    if not all(r.endswith('\n') for r in rows):
        raise SystemExit('A non-empty row lacks its trailing newline; the safe-split argument does not hold.')
    chunks, start = [], 0
    for i in range(1, len(rows)):
        if i - start >= rows_per_chunk and safe_split(rows[i]):
            chunks.append('\n'.join(rows[start:i]) + '\n')
            start = i
    chunks.append('\n'.join(rows[start:]))
    out = []
    for toks in enc.encode_ordinary_batch(chunks, num_threads=8):
        out.append(np.asarray(toks, dtype=np.uint16))
    return np.concatenate(out)


def prepare_data(work, max_train_rows=None):
    """Download and verify the four WikiText-103 files, then build or check the token files."""
    import pyarrow.parquet as pq
    entry = json.loads((REPO / 'data' / 'MANIFEST.json').read_text())['wikitext-103-raw-v1']
    ddir = work / 'data'
    ddir.mkdir(parents=True, exist_ok=True)
    for name, meta in entry['files'].items():
        p = ddir / name
        if not p.exists() or p.stat().st_size != meta['bytes'] or sha256_file(p) != meta['sha256']:
            log('downloading', name)
            if TEST and Path(os.environ.get('STUDY_LOCAL_PARQUET', '')).is_dir():
                shutil.copy(Path(os.environ['STUDY_LOCAL_PARQUET']) / name, p)
            else:
                urllib.request.urlretrieve(meta['url'], p)
        if sha256_file(p) != meta['sha256']:
            raise SystemExit(f'{name}: SHA-256 differs from data/MANIFEST.json. Stopping.')
    log('all four files match their fingerprints')
    meta_path = ddir / ('tokens_meta_test.json' if TEST else 'tokens_meta.json')
    enc = gpt2_encoding()
    splits = {'train': ['train-00000-of-00002.parquet', 'train-00001-of-00002.parquet'],
              'validation': ['validation-00000-of-00001.parquet'], 'test': ['test-00000-of-00001.parquet']}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text())
        for split in splits:
            f = ddir / meta[split]['file']
            if sha256_file(f) != meta[split]['sha256']:
                raise SystemExit(f'{split} token file changed since it was built. Delete {f} and rerun.')
        log('token files verified:', {s: meta[s]['tokens'] for s in splits})
        return meta
    meta = {}
    for split, files in splits.items():
        rows = []
        for f in files:
            rows += pq.read_table(ddir / f).column('text').to_pylist()
        if split == 'train' and max_train_rows:
            rows = rows[:max_train_rows]
        t = time.time()
        toks = encode_rows(enc, rows)
        fname = f'{split}{"_test" if TEST else ""}.u16'
        toks.tofile(ddir / fname)
        meta[split] = {'file': fname, 'rows': len(rows), 'tokens': int(len(toks)), 'sha256': sha256_file(ddir / fname),
                       'encode_seconds': time.time() - t}
        log(f'{split}: {len(rows):,} rows -> {len(toks):,} tokens in {time.time() - t:.0f}s')
    meta['created_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
    meta_path.write_text(json.dumps(meta, indent=2))
    return meta


def load_tokens(work, meta, split):
    return np.memmap(work / 'data' / meta[split]['file'], dtype=np.uint16, mode='r')


def windows(tokens, limit=None):
    n = (len(tokens) - 1) // SEQ
    if limit:
        n = min(n, limit)
    t = torch.from_numpy(np.asarray(tokens[:n * SEQ + 1], dtype=np.int64))
    return t[:n * SEQ].view(n, SEQ), t[1:n * SEQ + 1].view(n, SEQ)


# ------------------------------------------------------------------ models and evaluation

def new_model(kind, seed):
    torch.manual_seed(seed)
    m = pm.BaselineTransformer(**CFG) if kind == 'baseline' else pm.HybridODETransformer(**CFG, **HYB)
    return m.to(DEV)


def forward(model, x, solver=None, scale=None):
    """model(x), or for Model A with a chosen ODE solver and alpha scale."""
    if solver is None and scale is None:
        return model(x)
    kw = dict(SOLVERS[solver or 'euler4'])
    B, T = x.shape
    h = model.tok_emb(x) + model.pos_emb(torch.arange(T, device=x.device))
    for b in model.early_blocks:
        h = b(h)
    func = model.ode_block.func
    func.control = None
    saved = func.output_scale.data.clone()
    if scale:
        func.output_scale.data.mul_(scale)
    try:
        if kw['method'] in ('euler', 'rk4'):
            out = odeint(func, h, torch.linspace(0, 1, kw['n_steps'] + 1, device=x.device), method=kw['method'])
        else:
            out = odeint(func, h, torch.tensor([0.0, 1.0], device=x.device), method='dopri5', rtol=kw['rtol'], atol=kw['atol'])
    finally:
        func.output_scale.data.copy_(saved)
    h = out[-1]
    for b in model.late_blocks:
        h = b(h)
    return model.head(model.ln_f(h))


@torch.no_grad()
def eval_loss(model, xs, ys, solver=None, scale=None, batch=32):
    """Mean next-token cross-entropy in nats, computed in fp32."""
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(xs), batch):
        x, y = xs[i:i + batch].to(DEV), ys[i:i + batch].to(DEV)
        logits = forward(model, x, solver, scale).float()
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction='sum').item()
        count += y.numel()
    model.train()
    return total / count


def lr_at(step, total, peak):
    if step < WARMUP:
        return peak * (step + 1) / WARMUP
    progress = min(1.0, (step - WARMUP) / max(1, total - WARMUP))
    return peak * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * progress)))


# ------------------------------------------------------------------ training

def train_run(work, meta, tag, kind, seed, peak_lr, total_steps):
    """One training run, resumable. Returns its result record."""
    res_path = work / 'results' / f'{tag}.json'
    if res_path.exists():
        log(f'{tag}: already finished, skipping')
        return json.loads(res_path.read_text())
    ck_path = work / 'ckpt' / f'{tag}.pt'
    ck_path.parent.mkdir(parents=True, exist_ok=True)
    train = load_tokens(work, meta, 'train')
    vx, vy = windows(load_tokens(work, meta, 'validation'), limit=64 if TEST else None)
    model = new_model(kind, seed)
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr)
    scaler = torch.amp.GradScaler('cuda', enabled=AMP)
    gen = torch.Generator().manual_seed(seed)
    state = {'step': 0, 'train_loss': [], 'val': [], 'alpha': [], 'skipped_steps': 0, 'seconds': 0.0, 'resumes': 0}
    if ck_path.exists():
        ck = torch.load(ck_path, map_location=DEV, weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); scaler.load_state_dict(ck['scaler'])
        gen.set_state(ck['gen']); state = ck['state']; state['resumes'] += 1
        torch.set_rng_state(ck['torch_rng'])
        if DEV.type == 'cuda' and ck.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
        log(f"{tag}: resuming at step {state['step']}")
    else:
        log(f'{tag}: starting, {kind}, seed {seed}, peak lr {peak_lr}, {total_steps} steps')

    def save_ckpt():
        tmp = ck_path.with_suffix('.tmp')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(),
                    'gen': gen.get_state(), 'state': state, 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if DEV.type == 'cuda' else None}, tmp)
        os.replace(tmp, ck_path)

    model.train()
    last_ck, t_seg, bad = time.time(), time.time(), 0
    n_windows = len(train) - SEQ - 1
    while state['step'] < total_steps:
        step = state['step']
        for g in opt.param_groups:
            g['lr'] = lr_at(step, total_steps, peak_lr)
        ix = torch.randint(n_windows, (BATCH,), generator=gen).numpy()
        batch = torch.from_numpy(np.stack([np.asarray(train[i:i + SEQ + 1], dtype=np.int64) for i in ix])).to(DEV)
        x, y = batch[:, :-1], batch[:, 1:]
        opt.zero_grad(set_to_none=True)
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=AMP):
            loss = F.cross_entropy(model(x).view(-1, CFG['vocab_size']), y.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if AMP and scaler.get_scale() < scale_before:
            state['skipped_steps'] += 1
        if not math.isfinite(loss.item()):
            bad += 1
            if bad >= 100:
                raise SystemExit(f'{tag}: 100 consecutive non-finite losses. Stopping this run.')
        else:
            bad = 0
        state['step'] = step + 1
        if step % 50 == 0:
            state['train_loss'].append([step, loss.item()])
            if kind == 'hybrid':
                state['alpha'].append([step, model.ode_block.func.output_scale.item()])
        if state['step'] in (500,) or state['step'] % VAL_EVERY == 0 or state['step'] == total_steps:
            vl = eval_loss(model, vx, vy)
            state['val'].append([state['step'], vl])
            log(f"{tag}: step {state['step']}/{total_steps}, validation loss {vl:.4f}")
        if time.time() - last_ck > CKPT_SECONDS:
            state['seconds'] += time.time() - t_seg; t_seg = time.time()
            save_ckpt(); last_ck = time.time()
    state['seconds'] += time.time() - t_seg
    save_ckpt()
    final_dir = work / 'final'
    final_dir.mkdir(exist_ok=True)
    torch.save(model.state_dict(), final_dir / f'{tag}.pt')
    vals = state['val']
    last10 = [v for s, v in vals if s >= 0.9 * total_steps]
    before = [v for s, v in vals if s < 0.9 * total_steps]
    result = {'tag': tag, 'kind': kind, 'seed': seed, 'peak_lr': peak_lr, 'total_steps': total_steps,
              'final_val_loss': vals[-1][1], 'val_curve': vals, 'val_at_500': next((v for s, v in vals if s == 500), None),
              'improvement_last_10pct': (before[-1] - last10[-1]) if before and last10 else None,
              'train_loss_every_50': state['train_loss'], 'alpha_every_50': state['alpha'],
              'alpha_final': model.ode_block.func.output_scale.item() if kind == 'hybrid' else None,
              'skipped_steps_fp16': state['skipped_steps'], 'train_seconds': state['seconds'], 'resumes': state['resumes'],
              'n_params': sum(p.numel() for p in model.parameters()),
              'config': {**CFG, 'seq': SEQ, 'batch': BATCH, 'warmup': WARMUP, 'min_lr_ratio': MIN_LR_RATIO},
              'data': {s: {k: meta[s][k] for k in ('tokens', 'sha256')} for s in ('train', 'validation', 'test')},
              'environment': env_record()}
    res_path.parent.mkdir(exist_ok=True)
    res_path.write_text(json.dumps(result, indent=2))
    log(f'{tag}: finished, final validation loss {vals[-1][1]:.4f}, {state["seconds"] / 3600:.2f} h')
    return result


def amp_gradient_check(work, meta):
    """Sanity check: fp16 mixed-precision gradients point the same way as fp32 ones, for both models."""
    out_path = work / 'results' / ('amp_check_test.json' if TEST else 'amp_check.json')
    if out_path.exists():
        return json.loads(out_path.read_text())
    train = load_tokens(work, meta, 'train')
    batch = torch.from_numpy(np.stack([np.asarray(train[i:i + SEQ + 1], dtype=np.int64) for i in range(0, BATCH * 997, 997)])).to(DEV)
    x, y = batch[:, :-1], batch[:, 1:]
    res = {}
    for kind in ('baseline', 'hybrid'):
        grads = {}
        for amp in (False, True):
            m = new_model(kind, 0)
            m.eval()  # no dropout, so both passes compute the same function
            with torch.autocast(DEV.type, dtype=torch.float16, enabled=amp and AMP):
                loss = F.cross_entropy(m(x).view(-1, CFG['vocab_size']).float(), y.reshape(-1))
            loss.backward()
            grads[amp] = torch.cat([p.grad.float().flatten() for p in m.parameters() if p.grad is not None])
        cos = F.cosine_similarity(grads[False], grads[True], dim=0).item()
        res[kind] = {'cosine_fp32_vs_fp16': cos}
    res['passed'] = all(v['cosine_fp32_vs_fp16'] > 0.99 for k, v in res.items() if isinstance(v, dict))
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    log('fp16 gradient check:', res)
    if not res['passed']:
        raise SystemExit('fp16 gradients disagree with fp32. Stopping before training; tell Claude.')
    return res


def stage1(work, meta, total_steps):
    amp_gradient_check(work, meta)
    sweep_steps = max(1, round(SWEEP_FRACTION * total_steps))
    choice = {}
    for kind in ('baseline', 'hybrid'):
        finals = {}
        for lr in SWEEP_LRS:
            r = train_run(work, meta, f'sweep_{kind}_lr{lr:g}', kind, SWEEP_SEED, lr, sweep_steps)
            finals[lr] = r['final_val_loss']
        choice[kind] = {'chosen_lr': min(finals, key=finals.get), 'sweep_final_val_loss': {f'{k:g}': v for k, v in finals.items()}}
        spread = train_run(work, meta, f'spread_{kind}_seed{SPREAD_SEED}', kind, SPREAD_SEED, choice[kind]['chosen_lr'], sweep_steps)
        choice[kind]['spread_pair'] = [finals[choice[kind]['chosen_lr']], spread['final_val_loss']]
    summary = {'sweep_steps': sweep_steps, 'total_steps_main': total_steps, 'choice': choice}
    (work / 'results' / 'stage1_summary.json').write_text(json.dumps(summary, indent=2))
    log('stage 1 complete:', json.dumps(summary))


def stage2(work, meta, total_steps):
    s1 = work / 'results' / 'stage1_summary.json'
    if not s1.exists():
        raise SystemExit('Stage 1 has not finished; run stage 1 first.')
    choice = json.loads(s1.read_text())['choice']
    for seed in MAIN_SEEDS:
        for kind in ('baseline', 'hybrid'):
            train_run(work, meta, f'main_{kind}_seed{seed}', kind, seed, choice[kind]['chosen_lr'], total_steps)
    log('stage 2 complete: all main runs finished')


def stage3(work, meta):
    tx, ty = windows(load_tokens(work, meta, 'test'), limit=64 if TEST else None)
    cx, cy = tx[:512], ty[:512]
    for seed in MAIN_SEEDS:
        for kind in ('baseline', 'hybrid'):
            tag = f'main_{kind}_seed{seed}'
            out = work / 'results' / f'test_{tag}.json'
            if out.exists():
                continue
            m = new_model(kind, seed)
            m.load_state_dict(torch.load(work / 'final' / f'{tag}.pt', map_location=DEV))
            m.eval()
            rec = {'tag': tag, 'kind': kind, 'seed': seed, 'test_loss': eval_loss(m, tx, ty), 'environment': env_record()}
            if kind == 'hybrid':
                with torch.no_grad():
                    xb = tx[:8].to(DEV)
                    rec['must_match_max_diff'] = (forward(m, xb, solver='euler4') - m(xb)).abs().max().item()
                if rec['must_match_max_diff'] > 1e-4:
                    raise SystemExit(f'{tag}: solver-switching code does not reproduce the model ({rec["must_match_max_diff"]}).')

                def table(xs, ys, scale=None):
                    t = {}
                    for name in SOLVERS:
                        try:
                            t[name] = {'loss': eval_loss(m, xs, ys, solver=name, scale=scale)}
                        except Exception as e:
                            t[name] = {'loss': None, 'error': f'{type(e).__name__}: {e}'[:300]}
                    base = t['euler4']['loss']
                    for v in t.values():
                        v['rel_change_vs_euler4'] = None if (v['loss'] is None or base is None) else abs(v['loss'] - base) / base
                    return t
                rec['solver_test'] = table(tx, ty)
                rec['solver_controls'] = {str(s): table(cx, cy, scale=s) for s in SCALES}
                rec['alpha'] = m.ode_block.func.output_scale.item()
            out.write_text(json.dumps(rec, indent=2))
            log(f"{tag}: test loss {rec['test_loss']:.4f}")
    log('stage 3 complete')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--stage', required=True, choices=['1', '2', '3'])
    ap.add_argument('--max-train-rows', type=int, default=None, help='test mode only')
    args = ap.parse_args()
    work, out = Path(args.work), Path(args.out)
    (work / 'results').mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)
    log('device:', DEV, '| fp16:', AMP, '| test mode:', TEST)
    try:
        meta = prepare_data(work, args.max_train_rows if TEST else None)
        total_steps = (meta['train']['tokens'] // (SEQ * BATCH)) if not TEST else 60
        if args.stage == '1':
            stage1(work, meta, total_steps)
        elif args.stage == '2':
            stage2(work, meta, total_steps)
        else:
            stage3(work, meta)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        # Copy every finished result to the upload folder, including ones from earlier sessions.
        dest = out / 'results'
        dest.mkdir(exist_ok=True)
        for f in (work / 'results').glob('*.json'):
            shutil.copy(f, dest / f.name)
        meta_file = work / 'data' / ('tokens_meta_test.json' if TEST else 'tokens_meta.json')
        if meta_file.exists():
            shutil.copy(meta_file, out / meta_file.name)
        log(f'copied {len(list(dest.glob("*.json")))} result files for upload')


if __name__ == '__main__':
    main()
