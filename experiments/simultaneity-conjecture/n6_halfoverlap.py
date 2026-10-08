"""N6: does the half-overlap balance depend on how different the readings are?

Implements N6_PREREGISTRATION.md in this folder. Models are N4's (the paper's HybridODETransformer, exact
gradients, no control, two heads reading the final state through separate rank-64 readouts, P_B starting as
a copy of P_A). Head A always predicts token i+1; head B predicts token i+k:
  same  k = 1 (must-fail control);   k2, k4, k8  k = 2, 4, 8.
Every window carries 8 extra tokens, so all four arms of a seed see the same inputs and differ only in head
B's target. N4a's overlap measure is taken on validation windows 1,793-1,931, which no earlier test used.

    python n6_halfoverlap.py --out OUT --work /content/drive/MyDrive/cdt-colab/n6_halfoverlap
    STUDY_TEST=1 python n6_halfoverlap.py --out OUT --work W --source LOCAL_STUDY_FOLDER   # small test
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import n4_directions as n4  # noqa: E402  (imports the study, G1 and N2 code)

g, study, DEV, TEST, AMP, log = n4.g, n4.study, n4.DEV, n4.TEST, n4.AMP, n4.log
SEQ, BATCH, VOCAB = g.SEQ, g.BATCH, g.VOCAB

SEEDS = [3601, 3602] if TEST else [3601, 3602, 3603, 3604, 3605]
g.SEEDS = SEEDS
ARMS = {'same': 1, 'k2': 2, 'k4': 4, 'k8': 8}
K_MAX = 8
TOTAL, PEAK_LR = g.TOTAL, g.PEAK_LR
R = n4.R                                           # 16 (test mode 4)
MEAS = (96, 112) if TEST else (1792, None)         # windows 1,793 to the last (1,931)
BASE = 'dual-sep'                                  # N4's separate-readout model, P_B a copy of P_A


# ------------------------------------------------------------------ data with 8 extra tokens

def windows(tokens, limit=None):
    """Non-overlapping 128-token inputs, with the targets at i+1 to i+8 (index k-1 holds token i+k)."""
    n = (len(tokens) - K_MAX) // SEQ
    if limit:
        n = min(n, limit)
    t = torch.from_numpy(np.asarray(tokens[:n * SEQ + K_MAX], dtype=np.int64))
    idx = torch.arange(n)[:, None] * SEQ + torch.arange(SEQ)[None, :]
    return t[idx], [t[idx + k] for k in range(1, K_MAX + 1)]


def draw_batch(train, gen):
    """136-token windows from the run's generator: the input and its targets at i+1 to i+8."""
    ix = torch.randint(len(train) - SEQ - K_MAX, (BATCH,), generator=gen).numpy()
    b = torch.from_numpy(np.stack([np.asarray(train[i:i + SEQ + K_MAX], dtype=np.int64) for i in ix]))
    return b[:, :SEQ], [b[:, k:SEQ + k] for k in range(1, K_MAX + 1)]


def pair(arm, ys):
    """Head A's and head B's targets for an arm."""
    return ys[0], ys[ARMS[arm] - 1]


@torch.no_grad()
def losses(model, xs, ya, yb, batch=32):
    """Each head's mean cross-entropy against its own target, fp32."""
    was = model.training
    model.eval()
    tot, cnt = [0.0, 0.0], 0
    for i in range(0, len(xs), batch):
        x = xs[i:i + batch].to(DEV)
        la, lb = n4.forward_n4(model, x)
        tot[0] += F.cross_entropy(la.float().reshape(-1, VOCAB), ya[i:i + batch].to(DEV).reshape(-1), reduction='sum').item()
        tot[1] += F.cross_entropy(lb.float().reshape(-1, VOCAB), yb[i:i + batch].to(DEV).reshape(-1), reduction='sum').item()
        cnt += x.shape[0] * x.shape[1]
    model.train(was)
    return tot[0] / cnt, tot[1] / cnt


def measure(model, mx, ya, yb):
    """N4a's overlap and split-half stabilities, with head A on ya and head B on yb (N4's dual-sep
    target mapping: its 'next' is ya and its 'skip' is yb)."""
    return n4.measure(model, BASE, mx, ya, yb)


# ------------------------------------------------------------------ training

def train_n6(store, meta, arm, seed, tag):
    """One run, resumable. The seed fixes the initial weights and the data order for all four arms."""
    res_path = store.L(f'results/{tag}.json')
    if res_path.exists():
        log(f'{tag}: already finished, skipping')
        return json.loads(res_path.read_text())
    ck_rel = f'ckpt/{tag}.pt'
    store.pull(ck_rel)
    ck_path = store.L(ck_rel)
    train = study.load_tokens(store, meta, 'train')
    vx, vys = windows(study.load_tokens(store, meta, 'validation'), limit=g.VAL_LIMIT)
    vya, vyb = pair(arm, vys)
    model = n4.new_n4(seed, BASE)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR)
    scaler = torch.amp.GradScaler('cuda', enabled=AMP)
    gen = torch.Generator().manual_seed(seed)
    state = {'step': 0, 'train_loss': [], 'val': [], 'alpha': [], 'skipped_steps': 0, 'seconds': 0.0, 'resumes': 0}
    if ck_path.exists():
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); scaler.load_state_dict(ck['scaler'])
        gen.set_state(ck['gen'])
        torch.set_rng_state(ck['torch_rng'])
        if DEV.type == 'cuda' and ck.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
        state = ck['state']; state['resumes'] += 1
        log(f"{tag}: resuming at step {state['step']}")
    else:
        log(f'{tag}: starting, head B at i+{ARMS[arm]}, seed {seed}, readout rank {n4.RANK}, peak lr {PEAK_LR}')

    def save_ckpt():
        tmp = ck_path.with_suffix('.tmp')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(),
                    'gen': gen.get_state(), 'state': state, 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if DEV.type == 'cuda' else None}, tmp)
        os.replace(tmp, ck_path)
        store.push(ck_rel)

    model.train()
    last_ck, t_seg = time.time(), time.time()
    while state['step'] < TOTAL:
        step = state['step']
        for grp in opt.param_groups:
            grp['lr'] = study.lr_at(step, TOTAL, PEAK_LR)
        x, ys = draw_batch(train, gen)
        if step == 0:
            state['batch_fingerprint_step0'] = int((x.long() * torch.arange(1, x.numel() + 1).view_as(x)).sum())
        ya, yb = pair(arm, ys)
        x, ya, yb = x.to(DEV), ya.to(DEV), yb.to(DEV)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=AMP):
            la, lb = n4.forward_n4(model, x)
            loss = 0.5 * (F.cross_entropy(la.view(-1, VOCAB), ya.reshape(-1)) + F.cross_entropy(lb.view(-1, VOCAB), yb.reshape(-1)))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if AMP and scaler.get_scale() < before:
            state['skipped_steps'] += 1
        state['step'] = step + 1
        if step % 50 == 0:
            state['train_loss'].append([step, loss.item()])
            state['alpha'].append([step, model.ode_block.func.output_scale.item()])
        if state['step'] == 500 or state['step'] % g.VAL_EVERY == 0 or state['step'] == TOTAL:
            l_a, l_b = losses(model, vx, vya, vyb)
            state['val'].append([state['step'], l_a, l_b])
            log(f"{tag}: step {state['step']}/{TOTAL}, validation loss head A {l_a:.4f}, head B {l_b:.4f}")
        if time.time() - last_ck > study.CKPT_SECONDS:
            state['seconds'] += time.time() - t_seg; t_seg = time.time()
            save_ckpt(); last_ck = time.time()
    state['seconds'] += time.time() - t_seg
    save_ckpt()
    torch.save(model.state_dict(), store.L(f'final/{tag}.pt'))
    store.push(f'final/{tag}.pt')
    fx, fys = windows(study.load_tokens(store, meta, 'validation'), limit=g.FINAL_LIMIT)
    l_a, l_b = losses(model, fx, *pair(arm, fys))
    result = {'tag': tag, 'arm': arm, 'k': ARMS[arm], 'seed': seed, 'readout_rank': n4.RANK,
              'final_loss_A': l_a, 'final_loss_B': l_b, 'final_val_windows': len(fx), 'val_curve': state['val'],
              'train_loss_every_50': state['train_loss'], 'alpha_every_50': state['alpha'],
              'alpha_final': model.ode_block.func.output_scale.item(),
              'batch_fingerprint_step0': state.get('batch_fingerprint_step0'),
              'skipped_steps_fp16': state['skipped_steps'], 'train_seconds': state['seconds'], 'resumes': state['resumes'],
              'n_params': sum(p.numel() for p in model.parameters()), 'environment': study.env_record()}
    res_path.write_text(json.dumps(result, indent=2))
    store.push(f'results/{tag}.json')
    log(f'{tag}: finished in {state["seconds"] / 3600:.2f} h; head A {l_a:.4f}, head B {l_b:.4f}')
    return result


# ------------------------------------------------------------------ checks

def batch_check(train):
    """Precondition 1: one batch decoded, and every target checked against the input tokens."""
    x, ys = draw_batch(train, torch.Generator().manual_seed(SEEDS[0]))
    ok = all(torch.equal(ys[k - 1][:, :-k], x[:, k:]) for k in range(1, K_MAX + 1))
    ok = ok and all(torch.equal(ys[k][:, :-1], ys[k - 1][:, 1:]) for k in range(1, K_MAX))
    enc = study.gpt2_encoding()
    return {'targets_are_i+1_to_i+8': bool(ok), 'input_end': enc.decode(x[0, -10:].tolist()),
            **{f'target_i+{k}_end': enc.decode(ys[k - 1][0, -10:].tolist()) for k in (1, 2, 4, 8)}}


def untrained(seed, mx, mys):
    """S4 and the calibration: overlap and stabilities on the untrained model, per arm."""
    out = {}
    for arm in ARMS:
        r = measure(n4.new_n4(seed, BASE), mx, *pair(arm, mys))
        out[arm] = {k: r[k] for k in ('overlap_AB', 'stability_A_odd_even', 'stability_B_odd_even')}
    return out


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--source', default=None)
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
    data_store = study.Store(source, local)
    store = study.Store(work, local)
    out = Path(args.out)
    (out / 'results').mkdir(parents=True, exist_ok=True)
    log('device:', DEV, '| test mode:', TEST, '| study folder (data):', source, '| N6 folder:', work, '| local:', local)
    store.pull_dir('results')
    try:
        meta = study.prepare_data(data_store, None)
        train = study.load_tokens(store, meta, 'train')
        summary_path = store.L('results/n6_summary.json')
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

        def save_json(rel, obj):
            store.L(rel).write_text(json.dumps(obj, indent=2 if rel.endswith('n6_summary.json') else None))
            store.push(rel)

        val = study.load_tokens(store, meta, 'validation')
        allx, allys = windows(val)
        lo, hi = MEAS[0], MEAS[1] if MEAS[1] else len(allx)
        mx, mys = allx[lo:hi], [y[lo:hi] for y in allys]

        if 'checks' not in summary:
            summary['checks'] = {'batch': batch_check(train), 'batch_g1': g.batch_check(train),
                                 'same_model': [g.same_model_check(train, s) for s in SEEDS]}
            summary['windows'] = {'measure': [lo + 1, hi]}
            summary['readout_rank'], summary['subspace_rank'], summary['arms'] = n4.RANK, R, ARMS
            save_json('results/n6_summary.json', summary)
        c = summary['checks']
        log('batch check, targets at i+1 to i+8:', c['batch']['targets_are_i+1_to_i+8'])
        log('same-model check:', [x['pass'] for x in c['same_model']])
        log(f'overlap measured on validation windows {lo + 1}-{hi}')
        if not (c['batch']['targets_are_i+1_to_i+8'] and all(x['pass'] for x in c['same_model'])):
            raise SystemExit('A check of the setup failed before training. Stopping.')

        if 'untrained' not in summary:
            summary['untrained'] = {f'seed{s}': untrained(s, mx, mys) for s in SEEDS}
            save_json('results/n6_summary.json', summary)

        for seed in SEEDS:
            for arm in ARMS:
                tag = f'{arm}_seed{seed}'
                train_n6(store, meta, arm, seed, tag)
                rel = f'results/sub_{tag}.json'
                if store.L(rel).exists():
                    continue
                if not store.pull(f'final/{tag}.pt'):
                    raise SystemExit(f'weights final/{tag}.pt are missing locally and on Drive.')
                m = n4.new_n4(seed, BASE)
                m.load_state_dict(torch.load(store.L(f'final/{tag}.pt'), map_location=DEV, weights_only=False))
                res = {'tag': tag, 'seed': seed, 'arm': arm, 'k': ARMS[arm], **measure(m, mx, *pair(arm, mys))}
                save_json(rel, res)
                log(f"sub_{tag}: overlap of A and B {res['overlap_AB']:.3f}, stability A {res['stability_A_odd_even']:.3f}, "
                    f"B {res['stability_B_odd_even']:.3f}")
        log('N6 complete: all runs and measurements finished')
    finally:
        dest = out / 'results'
        for f in (store.local / 'results').glob('*.json'):
            (dest / f.name).write_bytes(f.read_bytes())
        log(f'copied {len(list(dest.glob("*.json")))} result files for upload')
        log('waiting for background Drive saves to finish (up to 15 minutes)')
        store.flush()
        failures = store.failures + data_store.failures
        (out / 'drive_sync.json').write_text(json.dumps({'failures': failures}, indent=2))
        log('Drive saves finished' if not failures else f'Drive problems this session: {failures}')


if __name__ == '__main__':
    main()
