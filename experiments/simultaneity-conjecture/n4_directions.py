"""N4: are coexisting readings kept apart by direction, and does the model need that?

Implements N4_PREREGISTRATION.md in this folder. Models are N2's dual-head model (the paper's
HybridODETransformer, exact gradients, no control), with each head reading the final state through a
rank-64 projection: logits_k = W_k (P_k . ln_f(z)). Four arms:
  dual-sep    head A predicts token i+1, head B token i+2; P_A and P_B separate;
  dual-tied   the same targets; one shared P;
  same-sep    both heads predict token i+1; separate P;
  same-tied   both heads predict token i+1; one shared P.
In the separate arms P_B starts as a copy of P_A, so separate and tied are the same function at step 0.
N4a's subspaces are the top-16 eigenvectors of each head's per-token gradient covariance at the ODE
block's output, on validation windows 1,537-1,792.

    python n4_directions.py --out OUT --work /content/drive/MyDrive/cdt-colab/n4_directions
    STUDY_TEST=1 python n4_directions.py --out OUT --work W --source LOCAL_STUDY_FOLDER   # small test
"""
import argparse
import copy
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import g1_regimes as g  # noqa: E402  (imports the study: model config, test mode, device)
import n2_coexist as n2  # noqa: E402

study, DEV, TEST, AMP, log = g.study, g.DEV, g.TEST, g.AMP, g.log
SEQ, BATCH, VOCAB = g.SEQ, g.BATCH, g.VOCAB

SEEDS = [3501, 3502] if TEST else [3501, 3502, 3503, 3504, 3505]
g.SEEDS = SEEDS
TOTAL = g.TOTAL
PEAK_LR = g.PEAK_LR
RANK = 8 if TEST else 64                           # readout width: a quarter of d_model, as 64 of 256
R = 4 if TEST else 16                              # N4a's subspace dimension
MEAS = (96, 112) if TEST else (1536, 1792)         # windows 1,537-1,792
BATCH_W = n2.BATCH_W
ARMS = {'dual-sep': (('next', 'skip'), False), 'dual-tied': (('next', 'skip'), True),
        'same-sep': (('next', 'next'), False), 'same-tied': (('next', 'next'), True)}


# ------------------------------------------------------------------ the model

def new_n4(seed, arm):
    """N2's trunk for this seed, with two rank-RANK readouts. The full head the trunk was built with is
    removed. Every arm draws the same initial values for a seed: P_A, W_A and W_B from their own seeds,
    and P_B either a copy of P_A (separate) or the same module (tied)."""
    m = g.new_model(seed)
    del m.head
    D = study.CFG['d_model']
    torch.manual_seed(seed + 77773)
    m.proj_a = nn.Linear(D, RANK, bias=False).to(DEV)
    torch.manual_seed(seed + 55547)
    m.head = nn.Linear(RANK, VOCAB, bias=False).to(DEV)
    torch.manual_seed(seed + 99991)
    m.head2 = nn.Linear(RANK, VOCAB, bias=False).to(DEV)
    m.proj_b = m.proj_a if ARMS[arm][1] else copy.deepcopy(m.proj_a)
    return m


def forward_n4(model, x, keep_out=False):
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
    out = (model.head(model.proj_a(z)), model.head2(model.proj_b(z)))
    return (out, h_out) if keep_out else out


def targets(arm, y1, y2):
    pick = {'next': y1, 'skip': y2}
    a, b = ARMS[arm][0]
    return pick[a], pick[b]


@torch.no_grad()
def n4_losses(model, xs, y1s, y2s, arm, batch=32):
    """Each head's mean cross-entropy against its own target, fp32, over the given windows."""
    was = model.training
    model.eval()
    tot, cnt = [0.0, 0.0], 0
    for i in range(0, len(xs), batch):
        x = xs[i:i + batch].to(DEV)
        ya, yb = targets(arm, y1s[i:i + batch].to(DEV), y2s[i:i + batch].to(DEV))
        la, lb = forward_n4(model, x)
        tot[0] += F.cross_entropy(la.float().reshape(-1, VOCAB), ya.reshape(-1), reduction='sum').item()
        tot[1] += F.cross_entropy(lb.float().reshape(-1, VOCAB), yb.reshape(-1), reduction='sum').item()
        cnt += ya.numel()
    model.train(was)
    return tot[0] / cnt, tot[1] / cnt


# ------------------------------------------------------------------ N4a and S2 measures

def estimate_subspaces(model, xs, y1s, y2s, arm, r):
    """Top-r eigenvectors of each head's per-token gradient covariance at the ODE block's output (as N2)."""
    model.eval()
    D = study.CFG['d_model']
    G = [torch.zeros(D, D, dtype=torch.float64, device=DEV) for _ in range(2)]
    for i in range(0, len(xs), BATCH_W):
        x = xs[i:i + BATCH_W].to(DEV)
        ya, yb = targets(arm, y1s[i:i + BATCH_W].to(DEV), y2s[i:i + BATCH_W].to(DEV))
        (la, lb), h_out = forward_n4(model, x, keep_out=True)
        for k, (lg, y) in enumerate(((la, ya), (lb, yb))):
            loss = F.cross_entropy(lg.float().reshape(-1, VOCAB), y.reshape(-1), reduction='sum')
            (gr,) = torch.autograd.grad(loss, h_out, retain_graph=(k == 0))
            gr = gr.reshape(-1, D).double()
            G[k] += gr.T @ gr
    out = []
    for Gk in G:
        evals, evecs = torch.linalg.eigh(Gk)
        out.append(evecs[:, torch.argsort(evals, descending=True)[:r]].float())
    return out


def readout_overlap(model):
    """S2: overlap of the row spaces of P_A and P_B (1 when tied or identical; RANK / d_model for random)."""
    qa = torch.linalg.qr(model.proj_a.weight.detach().double().T)[0]
    qb = torch.linalg.qr(model.proj_b.weight.detach().double().T)[0]
    return ((qa.T @ qb).pow(2).sum() / qa.shape[1]).item()


def measure(model, arm, mx, my1, my2):
    t0 = time.time()
    Ua, Ub = estimate_subspaces(model, mx, my1, my2, arm, R)
    Uao, Ubo = estimate_subspaces(model, mx[0::2], my1[0::2], my2[0::2], arm, R)
    Uae, Ube = estimate_subspaces(model, mx[1::2], my1[1::2], my2[1::2], arm, R)
    return {'overlap_AB': n2.overlap(Ua, Ub), 'overlap_AB_odd': n2.overlap(Uao, Ubo), 'overlap_AB_even': n2.overlap(Uae, Ube),
            'stability_A_odd_even': n2.overlap(Uao, Uae), 'stability_B_odd_even': n2.overlap(Ubo, Ube),
            'readout_overlap': readout_overlap(model), 'rank': R, 'readout_rank': RANK,
            'subspaces': {'A': Ua.cpu().tolist(), 'B': Ub.cpu().tolist()}, 'seconds': time.time() - t0}


# ------------------------------------------------------------------ training

def train_n4(store, meta, arm, seed, tag):
    """One run, resumable. The seed fixes the initial weights and the data order for all four arms."""
    res_path = store.L(f'results/{tag}.json')
    if res_path.exists():
        log(f'{tag}: already finished, skipping')
        return json.loads(res_path.read_text())
    ck_rel = f'ckpt/{tag}.pt'
    store.pull(ck_rel)
    ck_path = store.L(ck_rel)
    train = study.load_tokens(store, meta, 'train')
    vx, vy1, vy2 = g.g1_windows(study.load_tokens(store, meta, 'validation'), limit=g.VAL_LIMIT)
    model = new_n4(seed, arm)
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
        log(f'{tag}: starting, {arm}, seed {seed}, readout rank {RANK}, peak lr {PEAK_LR}')

    def save_ckpt():
        tmp = ck_path.with_suffix('.tmp')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(),
                    'gen': gen.get_state(), 'state': state, 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if DEV.type == 'cuda' else None}, tmp)
        os.replace(tmp, ck_path)
        store.push(ck_rel)

    model.train()
    last_ck, t_seg = time.time(), time.time()
    task_gen = torch.Generator().manual_seed(0)   # unused draws keep g.draw_batch's signature
    while state['step'] < TOTAL:
        step = state['step']
        for grp in opt.param_groups:
            grp['lr'] = study.lr_at(step, TOTAL, PEAK_LR)
        x, y1, y2, _ = g.draw_batch(train, gen, task_gen)
        if step == 0:
            state['batch_fingerprint_step0'] = int((x.long() * torch.arange(1, x.numel() + 1).view_as(x)).sum())
        ya, yb = targets(arm, y1, y2)
        x, ya, yb = x.to(DEV), ya.to(DEV), yb.to(DEV)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=AMP):
            la, lb = forward_n4(model, x)
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
            l_a, l_b = n4_losses(model, vx, vy1, vy2, arm)
            state['val'].append([state['step'], l_a, l_b])
            log(f"{tag}: step {state['step']}/{TOTAL}, validation loss head A {l_a:.4f}, head B {l_b:.4f}")
        if time.time() - last_ck > study.CKPT_SECONDS:
            state['seconds'] += time.time() - t_seg; t_seg = time.time()
            save_ckpt(); last_ck = time.time()
    state['seconds'] += time.time() - t_seg
    save_ckpt()
    torch.save(model.state_dict(), store.L(f'final/{tag}.pt'))
    store.push(f'final/{tag}.pt')
    fx, fy1, fy2 = g.g1_windows(study.load_tokens(store, meta, 'validation'), limit=g.FINAL_LIMIT)
    l_a, l_b = n4_losses(model, fx, fy1, fy2, arm)
    result = {'tag': tag, 'arm': arm, 'seed': seed, 'readout_rank': RANK, 'tied': ARMS[arm][1], 'targets': ARMS[arm][0],
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

@torch.no_grad()
def must_match_check(train, seed):
    """Precondition 1: at initialisation, separate and tied give bit-identical logits on one batch,
    for both target pairs, and the four arms share every trunk weight."""
    x, _, _, _ = g.draw_batch(train, torch.Generator().manual_seed(seed), torch.Generator().manual_seed(0))
    x = x[:4].to(DEV)
    outs, trunk = {}, {}
    for arm in ARMS:
        m = new_n4(seed, arm)
        m.eval()
        outs[arm] = forward_n4(m, x)
        trunk[arm] = torch.cat([p.detach().reshape(-1) for k, p in m.named_parameters() if not k.startswith(('proj', 'head'))])
    same_logits = all(torch.equal(outs[f'{t}-sep'][k], outs[f'{t}-tied'][k]) for t in ('dual', 'same') for k in (0, 1))
    same_trunk = all(torch.equal(trunk['dual-sep'], v) for v in trunk.values())
    return {'seed': seed, 'sep_equals_tied_logits': same_logits, 'same_trunk_all_arms': same_trunk,
            'pass': same_logits and same_trunk}


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
    log('device:', DEV, '| test mode:', TEST, '| study folder (data):', source, '| N4 folder:', work, '| local:', local)
    store.pull_dir('results')
    try:
        meta = study.prepare_data(data_store, None)
        train = study.load_tokens(store, meta, 'train')
        summary_path = store.L('results/n4_summary.json')
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

        def save_json(rel, obj):
            store.L(rel).write_text(json.dumps(obj, indent=2 if rel.endswith('n4_summary.json') else None))
            store.push(rel)

        if 'checks' not in summary:
            summary['checks'] = {'must_match': [must_match_check(train, s) for s in SEEDS],
                                 'batch': n2.batch_check_dual(train), 'batch_g1': g.batch_check(train),
                                 'same_model': [g.same_model_check(train, s) for s in SEEDS]}
            summary['windows'] = {'measure': [MEAS[0] + 1, MEAS[1]]}
            summary['readout_rank'], summary['subspace_rank'] = RANK, R
            save_json('results/n4_summary.json', summary)
        c = summary['checks']
        log('must-match check (separate = tied at step 0):', [x['pass'] for x in c['must_match']])
        log('batch check, targets at i+1 and i+2:', c['batch']['targets_are_i+1_and_i+2'])
        log('same-model check:', [x['pass'] for x in c['same_model']])
        if not (all(x['pass'] for x in c['must_match']) and c['batch']['targets_are_i+1_and_i+2']
                and all(x['pass'] for x in c['same_model'])):
            raise SystemExit('A check of the setup failed before training. Stopping.')

        val = study.load_tokens(store, meta, 'validation')
        mx, my1, my2 = (t[MEAS[0]:MEAS[1]] for t in g.g1_windows(val, limit=MEAS[1]))

        for seed in SEEDS:
            for arm in ARMS:
                tag = f'{arm}_seed{seed}'
                train_n4(store, meta, arm, seed, tag)
                rel = f'results/sub_{tag}.json'
                if store.L(rel).exists():
                    continue
                if not store.pull(f'final/{tag}.pt'):
                    raise SystemExit(f'weights final/{tag}.pt are missing locally and on Drive.')
                m = new_n4(seed, arm)
                m.load_state_dict(torch.load(store.L(f'final/{tag}.pt'), map_location=DEV, weights_only=False))
                res = {'tag': tag, 'seed': seed, 'arm': arm, **measure(m, arm, mx, my1, my2)}
                save_json(rel, res)
                log(f"sub_{tag}: overlap of A and B {res['overlap_AB']:.3f}, readout overlap {res['readout_overlap']:.3f}")
        log('N4 complete: all runs and measurements finished')
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
