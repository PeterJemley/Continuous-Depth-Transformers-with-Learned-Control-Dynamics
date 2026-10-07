"""N2: do dynamical regimes become necessary when two readings must coexist?

Implements N2_PREREGISTRATION.md in this folder. Models are the paper's HybridODETransformer with exact
gradients, no control, and two output heads reading the same final state:
  dual       head A predicts token i+1, head B token i+2; full budget, checkpoint kept at step 1,440;
  same       both heads predict token i+1 (must-fail control); full budget;
  equalised  resumes from dual's step-1,440 checkpoint with lambda * (c_A - c_B)^2, where c_k is the
             contraction rate within reading k's subspace (fixed, estimated at step 1,440);
  sham       the same, with the penalty at zero weight (null contrast).
lambda is chosen first by a pilot on seed 100. Each reading's subspace is the top-r eigenvectors of the
per-token gradient covariance of its loss at the ODE block's output (validation windows 1,025-1,280);
contraction within it is measured on windows 1,281-1,536.

    python n2_coexist.py --out OUT --work /content/drive/MyDrive/cdt-colab/n2_coexist
    STUDY_TEST=1 python n2_coexist.py --out OUT --work W --source LOCAL_STUDY_FOLDER   # small test
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.func import jvp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import g1_regimes as g  # noqa: E402  (imports the study: model config, test mode, device)
import g1c_contraction as gc  # noqa: E402
import g1d_functional as gd  # noqa: E402

study, DEV, TEST, AMP, log = g.study, g.DEV, g.TEST, g.AMP, g.log
SEQ, BATCH, VOCAB = g.SEQ, g.BATCH, g.VOCAB

SEEDS = [3401, 3402] if TEST else [3401, 3402, 3403, 3404, 3405]
g.SEEDS = SEEDS
PILOT_SEED = 100
LAMBDAS = [0.01, 0.1, 1.0, 10.0]
TOTAL = g.TOTAL
MID = (TOTAL + 1) // 2                             # step 1,440 of 2,879
PEAK_LR = g.PEAK_LR
R_MAIN, R_ALT = (4, 8) if TEST else (16, 32)
EST = (64, 80) if TEST else (1024, 1280)           # windows 1,025-1,280 for the subspaces
MEAS = (80, 96) if TEST else (1280, 1536)          # windows 1,281-1,536 for contraction
BATCH_W = 8 if TEST else 32
EQUAL_MAX = math.log(1.25)
TARGETS = {'dual': ('next', 'skip'), 'same': ('next', 'next'), 'equalised': ('next', 'skip'), 'sham': ('next', 'skip')}


# ------------------------------------------------------------------ the dual-head model

def new_dual(seed):
    """G1's exact-gradient model with a second, untied output head, initialised from its own seed."""
    m = g.new_model(seed)
    torch.manual_seed(seed + 99991)
    m.head2 = nn.Linear(study.CFG['d_model'], VOCAB, bias=False).to(DEV)
    return m


def forward_dual(model, x, keep_out=False):
    """Both heads' logits from one forward pass, no control. Optionally also the ODE block's output."""
    T = x.shape[1]
    h = model.tok_emb(x) + model.pos_emb(torch.arange(T, device=x.device))
    for b in model.early_blocks:
        h = b(h)
    h_out = model.ode_block(h, None)
    z = h_out
    for b in model.late_blocks:
        z = b(z)
    z = model.ln_f(z)
    out = (model.head(z), model.head2(z))
    return (out, h_out) if keep_out else out


def targets(arm, y1, y2):
    a, b = TARGETS[arm]
    pick = {'next': y1, 'skip': y2}
    return pick[a], pick[b]


@torch.no_grad()
def dual_losses(model, xs, y1s, y2s, arm, batch=32):
    """Each head's mean cross-entropy against its target, fp32, whole split."""
    was = model.training
    model.eval()
    tot, cnt = [0.0, 0.0], 0
    for i in range(0, len(xs), batch):
        x = xs[i:i + batch].to(DEV)
        ya, yb = targets(arm, y1s[i:i + batch].to(DEV), y2s[i:i + batch].to(DEV))
        la, lb = forward_dual(model, x)
        tot[0] += F.cross_entropy(la.float().reshape(-1, VOCAB), ya.reshape(-1), reduction='sum').item()
        tot[1] += F.cross_entropy(lb.float().reshape(-1, VOCAB), yb.reshape(-1), reduction='sum').item()
        cnt += ya.numel()
    model.train(was)
    return tot[0] / cnt, tot[1] / cnt


# ------------------------------------------------------------------ subspaces and contraction within them

def estimate_subspaces(model, xs, y1s, y2s, arm, ranks):
    """Top-r eigenvectors of each head's per-token gradient covariance at the ODE block's output."""
    model.eval()
    D = study.CFG['d_model']
    G = [torch.zeros(D, D, dtype=torch.float64, device=DEV) for _ in range(2)]
    for i in range(0, len(xs), BATCH_W):
        x = xs[i:i + BATCH_W].to(DEV)
        ya, yb = targets(arm, y1s[i:i + BATCH_W].to(DEV), y2s[i:i + BATCH_W].to(DEV))
        (la, lb), h_out = forward_dual(model, x, keep_out=True)
        for k, (lg, y) in enumerate(((la, ya), (lb, yb))):
            loss = F.cross_entropy(lg.float().reshape(-1, VOCAB), y.reshape(-1), reduction='sum')
            (gr,) = torch.autograd.grad(loss, h_out, retain_graph=(k == 0))
            gr = gr.reshape(-1, D).double()
            G[k] += gr.T @ gr
    out = {}
    for name, Gk in zip(('A', 'B'), G):
        evals, evecs = torch.linalg.eigh(Gk)
        order = torch.argsort(evals, descending=True)
        for r in ranks:
            out[(name, r)] = evecs[:, order[:r]].float()
    return out


def subspace_probe(U, shape, gen):
    """z = U eps per token, eps a random +-1 vector in the subspace's dimension."""
    B, T, _ = shape
    eps = (torch.randint(2, (B, T, U.shape[1]), generator=gen).float() * 2 - 1).to(DEV)
    return eps @ U.T


@torch.no_grad()
def measure_subspace_contraction(model, xs, Us, probe_seed):
    """Per window and depth: contraction per dimension within each given subspace, c = -E[z^T J z]/(T r),
    along dopri5 trajectories at G1c's depths and tolerance. Also whole-state contraction (S4)."""
    model.eval()
    func = model.ode_block.func
    func.control = None
    gen = torch.Generator().manual_seed(probe_seed)
    rec = {name: [] for name in Us}
    rec['whole'] = []
    for i in range(0, len(xs), BATCH_W):
        x = xs[i:i + BATCH_W].to(DEV)
        B, T = x.shape
        h0 = model.tok_emb(x) + model.pos_emb(torch.arange(T, device=DEV))
        for blk in model.early_blocks:
            h0 = blk(h0)
        traj = g.trajectory(func, h0, gc.TIMES, tol=gc.TOL)
        probes = {name: [subspace_probe(U.to(DEV), h0.shape, gen) for _ in range(gc.PROBES)] for name, U in Us.items()}
        whole = [gc.rademacher(h0.shape, gen) for _ in range(gc.PROBES)]
        per = {name: [] for name in rec}
        for k, t in enumerate(gc.TIMES.tolist()):
            tt = torch.as_tensor(t, device=DEV)
            for name, U in Us.items():
                est = []
                for z in probes[name]:
                    _, jz = jvp(lambda hh: func(tt, hh), (traj[k],), (z,))
                    est.append((z.double() * jz.double()).flatten(1).sum(1) / (T * U.shape[1]))
                per[name].append(-torch.stack(est).mean(0))
            per['whole'].append(-gc.divergence(func, t, traj[k], whole)[0])
        for name in rec:
            rec[name] += torch.stack(per[name], 1).tolist()
    return rec


def overlap(U, V):
    """Mean squared cosine of the principal angles between two subspaces of equal dimension."""
    return ((U.T.double() @ V.double()).pow(2).sum() / U.shape[1]).item()


def projected_check():
    """Precondition 1: the projected estimator on linear fields f(h) = A h with known tr(U^T A U)."""
    gen = torch.Generator().manual_seed(4)
    D, T, B, r = 16, 8, 4096, 4
    A = torch.eye(D) + torch.randn(D, D, generator=gen) / 4
    U = torch.linalg.qr(torch.randn(D, r, generator=gen))[0]
    truth = torch.trace(U.T @ A @ U).item()
    h = torch.randn(B, T, D, generator=gen).to(DEV)
    f = lambda t, hh: hh @ A.T.to(DEV) + 0 * t  # noqa: E731
    est = []
    for _ in range(4):
        z = subspace_probe(U.to(DEV), (B, T, D), gen)
        _, jz = jvp(lambda hh: f(torch.tensor(0.0, device=DEV), hh), (h,), (z,))
        est.append(((z * jz).flatten(1).sum(1) / T).mean().item())
    dense_err = abs(np.mean(est) - truth) / abs(truth)
    # Diagonal A, axis-aligned U: exact.
    d = torch.randn(D, generator=gen)
    Ua = torch.eye(D)[:, :r]
    z = subspace_probe(Ua.to(DEV), (2, T, D), gen)
    hh = torch.randn(2, T, D, generator=gen).to(DEV)
    _, jz = jvp(lambda q: q * d.to(DEV), (hh,), (z,))
    diag_est = ((z * jz).flatten(1).sum(1) / T)
    diag_err = ((diag_est - d[:r].sum().to(DEV)).abs() / d[:r].sum().abs()).max().item()
    # The training estimator through U: its gradient with respect to A averages to U U^T.
    Ag = A.clone().to(DEV).requires_grad_()
    hq = torch.randn(65536, T, D, generator=gen).to(DEV).requires_grad_()
    zq = subspace_probe(U.to(DEV), hq.shape, gen)
    c = gd.train_contraction(lambda t, x: x @ Ag.T + 0 * t, torch.tensor(0.0, device=DEV), hq, zq)
    (gA,) = torch.autograd.grad(c.mean(), Ag)
    UUt = (U @ U.T).to(DEV)
    grad_err = ((gA + UUt).norm() / UUt.norm()).item()
    res = {'dense_truth': truth, 'dense_estimate': float(np.mean(est)), 'dense_relative_error': dense_err,
           'diagonal_worst_relative_error': diag_err, 'gradient_relative_error': grad_err}
    res['pass'] = dense_err <= 0.02 and diag_err <= 1e-5 and grad_err <= 0.02
    return res


# ------------------------------------------------------------------ training

def make_penalty(lam, Ua, Ub, probe_seed):
    """lambda * (C_A - C_B)^2, with C_k the batch mean of the per-sequence, per-Euler-state contraction per
    dimension within subspace k, one projected probe each, trained through. Probes from their own generator,
    seeded per step."""
    r = Ua.shape[1]
    Ua, Ub = Ua.to(DEV), Ub.to(DEV)

    def penalty(model, step):
        blk, func = model.ode_block, model.ode_block.func
        gen = torch.Generator().manual_seed(probe_seed * 100003 + step)
        func.control = None
        cs = {'A': [], 'B': []}
        for k in range(4):
            h = blk.states[k]
            for name, U in (('A', Ua), ('B', Ub)):
                z = subspace_probe(U, h.shape, gen)
                cs[name].append(gd.train_contraction(func, blk.integration_times[k], h, z) / r)
        ca, cb = torch.cat(cs['A']).mean(), torch.cat(cs['B']).mean()
        p = (ca - cb) ** 2
        return lam * p, p.item(), ca.item(), cb.item()
    return penalty


def train_dual(store, meta, arm, seed, tag, start=None, penalty=None, end=TOTAL, mid_save=None):
    """One run, resumable. From scratch, or from a mid checkpoint (start) with an optional penalty.
    mid_save names the checkpoint to keep at step MID."""
    res_path = store.L(f'results/{tag}.json')
    if res_path.exists():
        log(f'{tag}: already finished, skipping')
        return json.loads(res_path.read_text())
    ck_rel = f'ckpt/{tag}.pt'
    store.pull(ck_rel)
    ck_path = store.L(ck_rel)
    train = study.load_tokens(store, meta, 'train')
    vx, vy1, vy2 = g.g1_windows(study.load_tokens(store, meta, 'validation'), limit=g.VAL_LIMIT)
    model = new_dual(seed)
    model.ode_block.keep_states = penalty is not None
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR)
    scaler = torch.amp.GradScaler('cuda', enabled=AMP)
    gen = torch.Generator().manual_seed(seed)
    state = {'step': 0, 'train_loss': [], 'val': [], 'alpha': [], 'penalty': [], 'skipped_steps': 0,
             'seconds': 0.0, 'resumes': 0}

    def load(path):
        ck = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); scaler.load_state_dict(ck['scaler'])
        gen.set_state(ck['gen'])
        torch.set_rng_state(ck['torch_rng'])
        if DEV.type == 'cuda' and ck.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
        return ck['state']

    if ck_path.exists():
        state = load(ck_path); state['resumes'] += 1
        log(f"{tag}: resuming at step {state['step']}")
    elif start is not None:
        if not store.pull(start):
            raise SystemExit(f'{tag}: the mid checkpoint {start} is missing locally and on Drive.')
        s0 = load(store.L(start))
        state.update(step=s0['step'], train_loss=list(s0['train_loss']), val=list(s0['val']), alpha=list(s0['alpha']))
        log(f"{tag}: starting from {start} at step {state['step']}")
    else:
        log(f'{tag}: starting, {arm}, seed {seed}, peak lr {PEAK_LR}, to step {end}')

    def snapshot():
        return {'model': model.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(),
                'gen': gen.get_state(), 'state': state, 'torch_rng': torch.get_rng_state(),
                'cuda_rng': torch.cuda.get_rng_state_all() if DEV.type == 'cuda' else None}

    def save_ckpt():
        tmp = ck_path.with_suffix('.tmp')
        torch.save(snapshot(), tmp)
        os.replace(tmp, ck_path)
        store.push(ck_rel)

    model.train()
    last_ck, t_seg = time.time(), time.time()
    task_gen = torch.Generator().manual_seed(0)   # unused draws keep g.draw_batch's signature; tasks are not used here
    while state['step'] < end:
        step = state['step']
        for grp in opt.param_groups:
            grp['lr'] = study.lr_at(step, TOTAL, PEAK_LR)
        x, y1, y2, _ = g.draw_batch(train, gen, task_gen)
        if step == MID:
            state['batch_fingerprint_at_mid'] = int((x.long() * torch.arange(1, x.numel() + 1).view_as(x)).sum())
        ya, yb = targets(arm, y1, y2)
        x, ya, yb = x.to(DEV), ya.to(DEV), yb.to(DEV)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=AMP):
            la, lb = forward_dual(model, x)
            loss = 0.5 * (F.cross_entropy(la.view(-1, VOCAB), ya.reshape(-1)) + F.cross_entropy(lb.view(-1, VOCAB), yb.reshape(-1)))
        pen = penalty(model, step) if penalty is not None else None
        scaler.scale(loss if pen is None else loss + pen[0]).backward()
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
            if pen is not None:
                state['penalty'].append([step, pen[1], pen[2], pen[3]])
        if state['step'] == 500 or state['step'] % g.VAL_EVERY == 0 or state['step'] == end:
            l_a, l_b = dual_losses(model, vx, vy1, vy2, arm)
            state['val'].append([state['step'], l_a, l_b])
            log(f"{tag}: step {state['step']}/{end}, validation loss head A {l_a:.4f}, head B {l_b:.4f}")
        if mid_save and state['step'] == MID:
            torch.save(snapshot(), store.L(mid_save))
            store.push(mid_save)
            log(f'{tag}: mid checkpoint saved at step {MID}')
        if time.time() - last_ck > study.CKPT_SECONDS:
            state['seconds'] += time.time() - t_seg; t_seg = time.time()
            save_ckpt(); last_ck = time.time()
    state['seconds'] += time.time() - t_seg
    save_ckpt()
    torch.save(model.state_dict(), store.L(f'final/{tag}.pt'))
    store.push(f'final/{tag}.pt')
    fx, fy1, fy2 = g.g1_windows(study.load_tokens(store, meta, 'validation'), limit=g.FINAL_LIMIT)
    l_a, l_b = dual_losses(model, fx, fy1, fy2, arm)
    result = {'tag': tag, 'arm': arm, 'seed': seed, 'end_step': end, 'started_from': start,
              'final_loss_A': l_a, 'final_loss_B': l_b, 'final_val_windows': len(fx), 'val_curve': state['val'],
              'train_loss_every_50': state['train_loss'], 'alpha_every_50': state['alpha'],
              'penalty_every_50': state['penalty'], 'alpha_final': model.ode_block.func.output_scale.item(),
              'batch_fingerprint_at_mid': state.get('batch_fingerprint_at_mid'),
              'skipped_steps_fp16': state['skipped_steps'], 'train_seconds': state['seconds'], 'resumes': state['resumes'],
              'n_params': sum(p.numel() for p in model.parameters()), 'targets': TARGETS[arm],
              'environment': study.env_record()}
    res_path.write_text(json.dumps(result, indent=2))
    store.push(f'results/{tag}.json')
    log(f'{tag}: finished in {state["seconds"] / 3600:.2f} h; head A {l_a:.4f}, head B {l_b:.4f}')
    return result


# ------------------------------------------------------------------ checks

def batch_check_dual(train):
    x, y1, y2, _ = g.draw_batch(train, torch.Generator().manual_seed(SEEDS[0]), torch.Generator().manual_seed(0))
    return {'targets_are_i+1_and_i+2': bool(torch.equal(y1[:, :-1], x[:, 1:]) and torch.equal(y2[:, :-2], x[:, 2:]))}


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
    log('device:', DEV, '| test mode:', TEST, '| study folder (data):', source, '| N2 folder:', work, '| local:', local)
    store.pull_dir('results')
    try:
        meta = study.prepare_data(data_store, None)
        train = study.load_tokens(store, meta, 'train')
        summary_path = store.L('results/n2_summary.json')
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

        def save_json(rel, obj):
            store.L(rel).write_text(json.dumps(obj, indent=2 if rel.endswith('n2_summary.json') else None))
            store.push(rel)

        if 'checks' not in summary:
            summary['checks'] = {'projected': projected_check(), 'penalty': gd.penalty_check(),
                                 'batch': batch_check_dual(train), 'batch_g1': g.batch_check(train),
                                 'same_model': [g.same_model_check(train, s) for s in SEEDS]}
            summary['windows'] = {'estimate': [EST[0] + 1, EST[1]], 'measure': [MEAS[0] + 1, MEAS[1]]}
            save_json('results/n2_summary.json', summary)
        c = summary['checks']
        log('projected-estimator check:', c['projected']['pass'], {k: round(v, 6) for k, v in c['projected'].items() if k != 'pass'})
        log('penalty check:', c['penalty']['pass'])
        log('batch check, targets at i+1 and i+2:', c['batch']['targets_are_i+1_and_i+2'])
        log('same-model check:', [x['pass'] for x in c['same_model']])
        if not (c['batch']['targets_are_i+1_and_i+2'] and all(x['pass'] for x in c['same_model'])):
            raise SystemExit('A check of the training setup failed before training. Stopping.')

        val = study.load_tokens(store, meta, 'validation')
        ex, ey1, ey2 = (t[EST[0]:EST[1]] for t in g.g1_windows(val, limit=EST[1]))
        mx = g.g1_windows(val, limit=MEAS[1])[0][MEAS[0]:MEAS[1]]

        def load_model(tag, seed, mid=False):
            rel = f'final/mid_{tag}.pt' if mid else f'final/{tag}.pt'
            if not store.pull(rel):
                raise SystemExit(f'weights {rel} are missing locally and on Drive.')
            m = new_dual(seed)
            sd = torch.load(store.L(rel), map_location=DEV, weights_only=False)
            m.load_state_dict(sd['model'] if mid else sd)
            return m

        def measure(tag, seed, arm, mid=False, halves=True, ranks=(R_MAIN, R_ALT)):
            rel = f'results/sub_{tag}{"_mid" if mid else ""}.json'
            if store.L(rel).exists():
                return json.loads(store.L(rel).read_text())
            m = load_model(tag, seed, mid)
            t0 = time.time()
            Us = estimate_subspaces(m, ex, ey1, ey2, arm, ranks)
            sets = {f'{k}{r}': Us[(k, r)] for (k, r) in Us}
            if halves:
                for half, sl in (('odd', slice(0, None, 2)), ('even', slice(1, None, 2))):
                    Uh = estimate_subspaces(m, ex[sl], ey1[sl], ey2[sl], arm, (R_MAIN,))
                    sets[f'A{R_MAIN}_{half}'], sets[f'B{R_MAIN}_{half}'] = Uh[('A', R_MAIN)], Uh[('B', R_MAIN)]
            rec = measure_subspace_contraction(m, mx, sets, seed + 700)
            res = {'tag': tag, 'seed': seed, 'arm': arm, 'checkpoint': 'mid' if mid else 'end', 'ranks': list(ranks),
                   'contraction': rec, 'overlap_AB': {r: overlap(Us[('A', r)], Us[('B', r)]) for r in ranks},
                   'subspaces': {k: v.cpu().tolist() for k, v in sets.items() if not k.endswith(('_odd', '_even'))},
                   'seconds': time.time() - t0}
            save_json(rel, res)
            cA, cB = np.mean(rec[f'A{R_MAIN}']), np.mean(rec[f'B{R_MAIN}'])
            log(f"sub_{tag}{'_mid' if mid else ''}: contraction within A {cA:.4f}, within B {cB:.4f}, overlap {res['overlap_AB'][R_MAIN]:.3f}")
            return res

        def fixed_U(tag, seed):
            r = measure(tag, seed, 'dual', mid=True, halves=False, ranks=(R_MAIN,))
            return torch.tensor(r['subspaces'][f'A{R_MAIN}']), torch.tensor(r['subspaces'][f'B{R_MAIN}'])

        # Pilot: choose lambda.
        if 'lambda' not in summary:
            pilot = summary.setdefault('pilot', {})
            train_dual(store, meta, 'dual', PILOT_SEED, 'pilot_dual', end=MID, mid_save='final/mid_pilot_dual.pt')
            Ua, Ub = fixed_U('pilot_dual', PILOT_SEED)
            chosen = None
            for lam in LAMBDAS:
                tag = f'pilot_eq_lam{lam:g}'
                if str(lam) not in pilot:
                    train_dual(store, meta, 'equalised', PILOT_SEED, tag, start='final/mid_pilot_dual.pt',
                               penalty=make_penalty(lam, Ua, Ub, PILOT_SEED))
                    r = measure(tag, PILOT_SEED, 'equalised', halves=False, ranks=(R_MAIN,))
                    cA, cB = np.mean(r['contraction'][f'A{R_MAIN}']), np.mean(r['contraction'][f'B{R_MAIN}'])
                    pilot[str(lam)] = {'cA': cA, 'cB': cB}
                    save_json('results/n2_summary.json', summary)
                p = pilot[str(lam)]
                ok = p['cA'] > 0 and p['cB'] > 0 and abs(math.log(p['cB'] / p['cA'])) <= EQUAL_MAX
                log(f"pilot lambda {lam:g}: contraction A {p['cA']:.4f}, B {p['cB']:.4f}, equalised: {ok}")
                if ok:
                    chosen = lam
                    break
            summary['lambda'] = chosen
            save_json('results/n2_summary.json', summary)
        lam = summary['lambda']
        log(f'lambda chosen by the pilot rule: {lam}')

        for seed in SEEDS:
            train_dual(store, meta, 'dual', seed, f'dual_seed{seed}', mid_save=f'final/mid_dual_seed{seed}.pt')
            train_dual(store, meta, 'same', seed, f'same_seed{seed}')
            measure(f'dual_seed{seed}', seed, 'dual')
            measure(f'same_seed{seed}', seed, 'same')
            if lam is None:
                continue
            Ua, Ub = fixed_U(f'dual_seed{seed}', seed)
            for arm, weight in (('equalised', lam), ('sham', 0.0)):
                tag = f'{arm}_seed{seed}'
                train_dual(store, meta, arm, seed, tag, start=f'final/mid_dual_seed{seed}.pt',
                           penalty=make_penalty(weight, Ua, Ub, seed))
                measure(tag, seed, arm, halves=False)
        if lam is None:
            log('No lambda in the grid equalised the pilot; as preregistered, N2b is untestable. N2a runs were made.')
        log('N2 complete: all runs and measurements finished')
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
