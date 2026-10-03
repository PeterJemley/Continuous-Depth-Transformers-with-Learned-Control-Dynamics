"""The WikiText-103 training study, as preregistered in PREREGISTRATION.md.

Three stages, each run as a Colab job through colab/runner.ipynb:
  stage 1  data preparation, the fp16 gradient check, the learning-rate sweep, and the spread runs
  stage 2  the 12 main runs: 6 seeds x {baseline, Model A}
  stage 3  test loss for every final model, and the solver-dependence test (S3)

Working files live on the Colab machine's own disk, so training never waits on
Google Drive. Checkpoints (about every 10 minutes), finished results and final
weights are copied to the Drive working folder in the background, with time
limits, so a slow or disconnected Drive cannot stall or stop training. Each new
session first pulls back from Drive whatever earlier sessions saved, then resumes;
a finished run is never repeated, so pressing Run all again simply continues.
Results are also copied to the job's output folder for upload to GitHub.

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
import threading
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
# 2.4e-3 added by the dated amendment in PREREGISTRATION.md, for both models, before stage 2.
SWEEP_LRS, SWEEP_FRACTION, SWEEP_SEED, SPREAD_SEED = [3e-4, 6e-4, 1.2e-3, 2.4e-3], 0.1, 100, 101
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


# ------------------------------------------------------------------ storage

class Store:
    """Local working files, mirrored to Google Drive in the background.

    Drive operations run in daemon threads with time limits. A failure or a hang
    is logged and never stops training; the data is still on the local disk and
    in the upload folder.
    """

    def __init__(self, drive, local):
        self.drive, self.local = Path(drive), Path(local)
        self.local.mkdir(parents=True, exist_ok=True)
        self.threads = {}
        self.failures = []

    def L(self, rel):
        p = self.local / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    def _timed(self, fn, timeout):
        box = {}

        def run():
            try:
                box['value'] = fn()
            except Exception as e:  # noqa: BLE001
                box['error'] = f'{type(e).__name__}: {e}'
        t = threading.Thread(target=run, daemon=True)
        t.start()
        t.join(timeout)
        if t.is_alive():
            return False, f'timed out after {timeout}s'
        return ('error' not in box), box.get('error', box.get('value'))

    def pull(self, rel, timeout=900):
        """Copy a file from Drive if it is there and missing locally. Returns True if the local file exists."""
        dst = self.L(rel)
        if dst.exists():
            return True

        def copy():
            src = self.drive / rel
            if not src.exists():
                return False
            tmp = dst.with_name(dst.name + '.part')
            shutil.copyfile(src, tmp)
            os.replace(tmp, dst)
            return True
        ok, value = self._timed(copy, timeout)
        if not ok:
            log(f'Drive: could not read {rel} ({value}); continuing without it')
            self.failures.append(f'pull {rel}: {value}')
        return dst.exists()

    def pull_dir(self, rel, timeout=300):
        ok, names = self._timed(lambda: sorted(p.name for p in (self.drive / rel).glob('*') if p.is_file()), timeout)
        if not ok:
            log(f'Drive: could not list {rel} ({names})')
            return
        for n in names:
            self.pull(f'{rel}/{n}')

    def push(self, rel, wait=False, timeout=900):
        """Copy a local file to Drive in the background. Skipped if the last copy of it is still running."""
        prev = self.threads.get(rel)
        if prev is not None and prev.is_alive():
            return

        def copy():
            dst = self.drive / rel
            dst.parent.mkdir(parents=True, exist_ok=True)
            tmp = dst.with_name(dst.name + '.part')
            shutil.copyfile(self.L(rel), tmp)
            os.replace(tmp, dst)

        def runner():
            try:
                copy()
            except Exception as e:  # noqa: BLE001
                log(f'Drive: could not save {rel} ({type(e).__name__}: {e}); it stays on the local disk')
                self.failures.append(f'push {rel}: {type(e).__name__}')
        t = threading.Thread(target=runner, daemon=True)
        t.start()
        self.threads[rel] = t
        if wait:
            t.join(timeout)

    def flush(self, timeout=900):
        deadline = time.time() + timeout
        for rel, t in self.threads.items():
            t.join(max(0, deadline - time.time()))
            if t.is_alive():
                log(f'Drive: saving {rel} did not finish in time')
                self.failures.append(f'push {rel}: unfinished')


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


def prepare_data(store, max_train_rows=None):
    """Use token files saved on Drive if they verify; otherwise download, verify and build them."""
    import pyarrow.parquet as pq
    entry = json.loads((REPO / 'data' / 'MANIFEST.json').read_text())['wikitext-103-raw-v1']
    ddir = store.L('data/x').parent
    meta_rel = 'data/' + ('tokens_meta_test.json' if TEST else 'tokens_meta.json')
    if store.pull(meta_rel):
        meta = json.loads(store.L(meta_rel).read_text())
        good = all(store.pull(f"data/{meta[s]['file']}") and sha256_file(ddir / meta[s]['file']) == meta[s]['sha256']
                   for s in ('train', 'validation', 'test'))
        if good:
            log('token files verified:', {s: meta[s]['tokens'] for s in ('train', 'validation', 'test')})
            return meta
        raise SystemExit('Saved token files do not match their recorded fingerprints. Delete the data folder on Drive and rerun.')
    for name, meta in entry['files'].items():
        store.pull(f'data/{name}')
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
    meta_path = store.L(meta_rel)
    enc = gpt2_encoding()
    splits = {'train': ['train-00000-of-00002.parquet', 'train-00001-of-00002.parquet'],
              'validation': ['validation-00000-of-00001.parquet'], 'test': ['test-00000-of-00001.parquet']}
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
    for split in splits:
        store.push(f"data/{meta[split]['file']}")
    store.push(meta_rel)
    return meta


def load_tokens(store, meta, split):
    return np.memmap(store.L(f"data/{meta[split]['file']}"), dtype=np.uint16, mode='r')


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
    """Mean next-token cross-entropy in nats, computed in fp32. Restores the model's previous mode."""
    was_training = model.training
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(xs), batch):
        x, y = xs[i:i + batch].to(DEV), ys[i:i + batch].to(DEV)
        logits = forward(model, x, solver, scale).float()
        total += F.cross_entropy(logits.reshape(-1, logits.size(-1)), y.reshape(-1), reduction='sum').item()
        count += y.numel()
    model.train(was_training)
    return total / count


def lr_at(step, total, peak):
    if step < WARMUP:
        return peak * (step + 1) / WARMUP
    progress = min(1.0, (step - WARMUP) / max(1, total - WARMUP))
    return peak * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * progress)))


# ------------------------------------------------------------------ training

def train_run(store, meta, tag, kind, seed, peak_lr, total_steps):
    """One training run, resumable. Returns its result record."""
    res_path = store.L(f'results/{tag}.json')
    if res_path.exists():
        log(f'{tag}: already finished, skipping')
        return json.loads(res_path.read_text())
    ck_rel = f'ckpt/{tag}.pt'
    store.pull(ck_rel)
    ck_path = store.L(ck_rel)
    train = load_tokens(store, meta, 'train')
    vx, vy = windows(load_tokens(store, meta, 'validation'), limit=64 if TEST else None)
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
        store.push(ck_rel)

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
    torch.save(model.state_dict(), store.L(f'final/{tag}.pt'))
    store.push(f'final/{tag}.pt')
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
    res_path.write_text(json.dumps(result, indent=2))
    store.push(f'results/{tag}.json')
    log(f'{tag}: finished, final validation loss {vals[-1][1]:.4f}, {state["seconds"] / 3600:.2f} h')
    return result


def amp_gradient_check(store, meta):
    """Sanity check: fp16 mixed-precision gradients point the same way as fp32 ones, for both models."""
    rel = 'results/' + ('amp_check_test.json' if TEST else 'amp_check.json')
    out_path = store.L(rel)
    if out_path.exists():
        return json.loads(out_path.read_text())
    train = load_tokens(store, meta, 'train')
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
    out_path.write_text(json.dumps(res, indent=2))
    store.push(rel)
    log('fp16 gradient check:', res)
    if not res['passed']:
        raise SystemExit('fp16 gradients disagree with fp32. Stopping before training; tell Claude.')
    return res


def stage1(store, meta, total_steps):
    amp_gradient_check(store, meta)
    sweep_steps = max(1, round(SWEEP_FRACTION * total_steps))
    choice = {}
    for kind in ('baseline', 'hybrid'):
        finals = {}
        for lr in SWEEP_LRS:
            r = train_run(store, meta, f'sweep_{kind}_lr{lr:g}', kind, SWEEP_SEED, lr, sweep_steps)
            finals[lr] = r['final_val_loss']
        choice[kind] = {'chosen_lr': min(finals, key=finals.get), 'sweep_final_val_loss': {f'{k:g}': v for k, v in finals.items()}}
        spread = train_run(store, meta, f'spread_{kind}_seed{SPREAD_SEED}', kind, SPREAD_SEED, choice[kind]['chosen_lr'], sweep_steps)
        choice[kind]['spread_pair'] = [finals[choice[kind]['chosen_lr']], spread['final_val_loss']]
    summary = {'sweep_steps': sweep_steps, 'total_steps_main': total_steps, 'choice': choice}
    store.L('results/stage1_summary.json').write_text(json.dumps(summary, indent=2))
    store.push('results/stage1_summary.json')
    log('stage 1 complete:', json.dumps(summary))


def stage2(store, meta, total_steps):
    s1 = store.L('results/stage1_summary.json')
    if not s1.exists():
        raise SystemExit('Stage 1 has not finished; run stage 1 first.')
    choice = json.loads(s1.read_text())['choice']
    for seed in MAIN_SEEDS:
        for kind in ('baseline', 'hybrid'):
            train_run(store, meta, f'main_{kind}_seed{seed}', kind, seed, choice[kind]['chosen_lr'], total_steps)
    log('stage 2 complete: all main runs finished')


def stage3(store, meta):
    tx, ty = windows(load_tokens(store, meta, 'test'), limit=64 if TEST else None)
    cx, cy = tx[:512], ty[:512]
    for seed in MAIN_SEEDS:
        for kind in ('baseline', 'hybrid'):
            tag = f'main_{kind}_seed{seed}'
            out = store.L(f'results/test_{tag}.json')
            if out.exists():
                continue
            if not store.pull(f'final/{tag}.pt'):
                raise SystemExit(f'Final weights for {tag} are not on Drive or this machine. Stage 2 must finish first.')
            m = new_model(kind, seed)
            m.load_state_dict(torch.load(store.L(f'final/{tag}.pt'), map_location=DEV))
            m.eval()
            rec = {'tag': tag, 'kind': kind, 'seed': seed, 'test_loss': eval_loss(m, tx, ty), 'environment': env_record()}
            if kind == 'hybrid':
                m.eval()
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
            store.push(f'results/test_{tag}.json')
            log(f"{tag}: test loss {rec['test_loss']:.4f}")
    log('stage 3 complete')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--stage', required=True, choices=['1', '2', '3'])
    ap.add_argument('--max-train-rows', type=int, default=None, help='test mode only')
    ap.add_argument('--local', default=None, help='local working folder; defaults to the Colab disk')
    args = ap.parse_args()
    work, out = Path(args.work), Path(args.out)
    if args.local:
        local = Path(args.local)
    elif Path('/content').is_dir() and not TEST:
        local = Path('/content/cdt_local') / work.name
    else:
        local = work.parent / f'_local_{work.name}'
    store = Store(work, local)
    out.mkdir(parents=True, exist_ok=True)
    log('device:', DEV, '| fp16:', AMP, '| test mode:', TEST, '| local folder:', local, '| Drive folder:', work)
    store.pull_dir('results')
    try:
        meta = prepare_data(store, args.max_train_rows if TEST else None)
        total_steps = (meta['train']['tokens'] // (SEQ * BATCH)) if not TEST else 60
        if args.stage == '1':
            stage1(store, meta, total_steps)
        elif args.stage == '2':
            stage2(store, meta, total_steps)
        else:
            stage3(store, meta)
    except BaseException:
        traceback.print_exc()
        raise
    finally:
        # Copy every finished result to the upload folder, including ones from earlier sessions.
        dest = out / 'results'
        dest.mkdir(exist_ok=True)
        for f in (store.local / 'results').glob('*.json'):
            shutil.copy(f, dest / f.name)
        meta_file = store.L('data/' + ('tokens_meta_test.json' if TEST else 'tokens_meta.json'))
        if meta_file.exists():
            shutil.copy(meta_file, out / meta_file.name)
        log(f'copied {len(list(dest.glob("*.json")))} result files for upload')
        log('waiting for background Drive saves to finish (up to 15 minutes)')
        store.flush()
        (out / 'drive_sync.json').write_text(json.dumps({'failures': store.failures}, indent=2))
        log('Drive saves finished' if not store.failures else f'Drive problems this session: {store.failures}')


if __name__ == '__main__':
    main()
