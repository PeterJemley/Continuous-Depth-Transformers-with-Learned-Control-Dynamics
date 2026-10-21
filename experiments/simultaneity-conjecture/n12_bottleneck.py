"""N12: is the noisy head held back by its readout?

Implements N12_PREREGISTRATION.md in this folder. N7's to N11's model and training (n10_payload.train_n10), at
rho = 0.8, with four arms that differ only in head B's targets and output:
  clean     head B predicts the next token (must-match control)
  n80       80% of head B's targets replaced by draws from u; single softmax (the yardstick)
  n80-mix   the same targets; head B's output is the mixture 0.2 softmax + 0.8 u (forward correction)
  sub20     clean targets at exactly the positions n80 leaves unreplaced, none elsewhere; single softmax
and two frozen fits of head B's readout on each n80 model, to the model's own ideal copy:
  fit-single  head B's usual single softmax
  fit-mix     the mixture (it can represent the copy exactly: the fitting procedure's must-match control)
Training batches and fit windows come only from the first 99% of the training text, and u is computed from it.
Every model is measured on windows of the held-out last 1%: 1-512 for the real run, 513-1024 for the calibration.

    python n12_bottleneck.py --out OUT --work /content/drive/MyDrive/cdt-colab/n12_bottleneck
    python n12_bottleneck.py --calibration --out OUT --work /content/drive/MyDrive/cdt-colab/n12_calib
    STUDY_TEST=1 python n12_bottleneck.py --out OUT --work W --source LOCAL_STUDY_FOLDER   # small test
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
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import n10_payload as n10  # noqa: E402  (imports N9, N8, N7, N6r, N6, N4, N2, G1 and the study)

n9, n8, n7, n6, n4, g, study, DEV, TEST, log = n10.n9, n10.n8, n10.n7, n10.n6, n10.n4, n10.g, n10.study, n10.DEV, n10.TEST, n10.log
AMP = n6.AMP
SEQ, VOCAB, K_MAX, BATCH, BASE = g.SEQ, g.VOCAB, n6.K_MAX, n6.BATCH, n8.BASE
RHO = 0.8
CUT_FRACTION = 0.99                                  # training and fits use the first 99% of the training text
ARMS = {'clean': (0.0, 'single'), 'n80': (RHO, 'single'), 'n80-mix': (RHO, 'mix'), 'sub20': (RHO, 'sub')}
FITS = ('single', 'mix')
TOTAL, PEAK_LR = n10.TOTAL, n10.PEAK_LR
FIT_STEPS = 60 if TEST else 3000
FIT_SEED = 271829
CONVERGE_CHUNKS = 1 if TEST else 6                   # 50-step chunks: the last 300 steps against the 300 before
AT_TARGET = 1e-3                                     # candidate amendment A1: a fit whose loss is below this has reached its target
MODES = {'real': {'seeds': [4101, 4102] if TEST else [4101, 4102, 4103, 4104, 4105], 'windows': (0, 8) if TEST else (0, 512)},
         'calibration': {'seeds': [3001], 'windows': (8, 14) if TEST else (512, 1024)}}
VAL_NOISE_SEED, MEAS_NOISE_SEED = n10.VAL_NOISE_SEED, n10.MEAS_NOISE_SEED
MEASURE_BATCH = n9.BATCH


# ------------------------------------------------------------------ heads, targets and batches

def b_noisy(lb, kind, log_u):
    """Log of head B's prediction of the noisy target: its softmax, or the mixture 0.2 softmax + 0.8 u.
    sub20's head is trained as a single softmax but is measured in the mixture form, like its copy."""
    return torch.log_softmax(lb.float(), -1) if kind == 'single' else n9.copy_logits(lb, log_u, RHO)


def head_b_loss(lb, kind, yb, keep, log_u):
    """Head B's training loss. sub20: the clean targets' cross-entropy summed over the kept positions and divided
    by all positions, so that its informative signal per step matches the noisy arms'."""
    if kind == 'mix':
        return F.nll_loss(n9.copy_logits(lb, log_u, RHO).reshape(-1, VOCAB), yb.reshape(-1))
    if kind == 'sub':
        ce = F.cross_entropy(lb.reshape(-1, VOCAB).float(), yb.reshape(-1), reduction='none')
        return (ce * keep.reshape(-1).float()).sum() / ce.numel()
    return F.cross_entropy(lb.reshape(-1, VOCAB), yb.reshape(-1))


def draw_batch(train, gen):
    """As n6.draw_batch (the same draws from the same generator), also returning the largest offset drawn."""
    ix = torch.randint(len(train) - SEQ - K_MAX, (BATCH,), generator=gen).numpy()
    b = torch.from_numpy(np.stack([np.asarray(train[i:i + SEQ + K_MAX], dtype=np.int64) for i in ix]))
    return b[:, :SEQ], [b[:, k:SEQ + k] for k in range(1, K_MAX + 1)], int(ix.max())


def step_targets(ya, probs, seed, step, rho, kind):
    """Head B's targets and kept positions at this step. The replacement draw is N10's, seeded by seed and step, so
    n80, n80-mix and sub20 see the same replaced positions; sub20 keeps the clean target where the others keep it."""
    if rho == 0:
        return ya, torch.ones(ya.shape, dtype=torch.bool), None
    yb, mask = n10.corrupt(ya, probs, torch.Generator().manual_seed(seed * 100003 + step), rho)
    return (ya if kind == 'sub' else yb), ~mask, mask


def fingerprint(t):
    return int((t.long() * torch.arange(1, t.numel() + 1).view_as(t)).sum())


@torch.no_grad()
def val_losses(model, xs, ya, yb, kind, log_u, batch=32):
    """Head A's loss on the clean targets; head B's noisy-target prediction's loss on yb and on the clean targets,
    and its inner softmax's loss on the clean targets."""
    was = model.training
    model.eval()
    tot, cnt = [0.0, 0.0, 0.0, 0.0], 0
    for i in range(0, len(xs), batch):
        x, a, b = xs[i:i + batch].to(DEV), ya[i:i + batch].to(DEV), yb[i:i + batch].to(DEV)
        la, lb = n4.forward_n4(model, x)
        lq = b_noisy(lb, kind, log_u)
        tot[0] += F.cross_entropy(la.float().reshape(-1, VOCAB), a.reshape(-1), reduction='sum').item()
        tot[1] -= lq.gather(-1, b.unsqueeze(-1)).sum().item()
        tot[2] -= lq.gather(-1, a.unsqueeze(-1)).sum().item()
        tot[3] += F.cross_entropy(lb.float().reshape(-1, VOCAB), a.reshape(-1), reduction='sum').item()
        cnt += a.numel()
    model.train(was)
    return [v / cnt for v in tot]


# ------------------------------------------------------------------ training

def train_n12(store, meta, arm, seed, tag, train, probs, log_u):
    """One run, resumable. As n10.train_n10, with head B's targets and output set by the arm."""
    rho, kind = ARMS[arm]
    res_path = store.L(f'results/{tag}.json')
    if res_path.exists():
        log(f'{tag}: already finished, skipping')
        return json.loads(res_path.read_text())
    ck_rel = f'ckpt/{tag}.pt'
    store.pull(ck_rel)
    ck_path = store.L(ck_rel)
    vx, vys = n6.windows(study.load_tokens(store, meta, 'validation'), limit=g.VAL_LIMIT)
    vya, vyb = vys[0], n10.fixed_noisy(vys[0], probs, rho, VAL_NOISE_SEED)
    model = n4.new_n4(seed, BASE)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR)
    scaler = torch.amp.GradScaler('cuda', enabled=AMP)
    gen = torch.Generator().manual_seed(seed)
    state = {'step': 0, 'train_loss': [], 'val': [], 'alpha': [], 'skipped_steps': 0, 'seconds': 0.0, 'resumes': 0, 'max_offset': -1}
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
        log(f'{tag}: starting, noise {rho}, head B {kind}, seed {seed}')

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
        x, ys, top = draw_batch(train, gen)
        state['max_offset'] = max(state['max_offset'], top)
        ya = ys[0]
        yb, keep, mask = step_targets(ya, probs, seed, step, rho, kind)
        if step == 0:
            state['batch_fingerprint_step0'] = fingerprint(x)
            if mask is not None:
                state['noise_fraction_step0'] = mask.float().mean().item()
                state['replaced_positions_fingerprint_step0'] = fingerprint(mask)
            state['kept_fraction_step0'] = keep.float().mean().item()
        x, ya, yb, keep = x.to(DEV), ya.to(DEV), yb.to(DEV), keep.to(DEV)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=AMP):
            la, lb = n4.forward_n4(model, x)
            loss = 0.5 * (F.cross_entropy(la.view(-1, VOCAB), ya.reshape(-1)) + head_b_loss(lb, kind, yb, keep, log_u))
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
            l_a, l_b = val_losses(model, vx, vya, vyb, kind, log_u)[:2]
            state['val'].append([state['step'], l_a, l_b])
            log(f"{tag}: step {state['step']}/{TOTAL}, validation loss head A {l_a:.4f}, head B {l_b:.4f}")
        if time.time() - last_ck > study.CKPT_SECONDS:
            state['seconds'] += time.time() - t_seg; t_seg = time.time()
            save_ckpt(); last_ck = time.time()
    state['seconds'] += time.time() - t_seg
    save_ckpt()
    torch.save(model.state_dict(), store.L(f'final/{tag}.pt'))
    store.push(f'final/{tag}.pt')
    fx, fys = n6.windows(study.load_tokens(store, meta, 'validation'), limit=g.FINAL_LIMIT)
    l_a, l_b, l_b_clean, l_inner = val_losses(model, fx, fys[0], n10.fixed_noisy(fys[0], probs, rho, VAL_NOISE_SEED), kind, log_u)
    result = {'tag': tag, 'arm': arm, 'rho': rho, 'head_b': kind, 'seed': seed, 'readout_rank': n4.RANK,
              'final_loss_A': l_a, 'final_loss_B': l_b, 'final_loss_B_clean_targets': l_b_clean, 'final_loss_B_inner_clean': l_inner,
              'final_val_windows': len(fx), 'val_curve': state['val'], 'train_loss_every_50': state['train_loss'],
              'alpha_every_50': state['alpha'], 'alpha_final': model.ode_block.func.output_scale.item(),
              'batch_fingerprint_step0': state.get('batch_fingerprint_step0'), 'noise_fraction_step0': state.get('noise_fraction_step0'),
              'replaced_positions_fingerprint_step0': state.get('replaced_positions_fingerprint_step0'),
              'kept_fraction_step0': state.get('kept_fraction_step0'),
              'max_offset': state['max_offset'], 'training_tokens': len(train),
              'skipped_steps_fp16': state['skipped_steps'], 'train_seconds': state['seconds'], 'resumes': state['resumes'],
              'n_params': sum(p.numel() for p in model.parameters()), 'environment': study.env_record()}
    res_path.write_text(json.dumps(result, indent=2))
    store.push(f'results/{tag}.json')
    log(f'{tag}: finished in {state["seconds"] / 3600:.2f} h; head A {l_a:.4f}, head B {l_b:.4f} (clean targets {l_b_clean:.4f})')
    return result


# ------------------------------------------------------------------ the frozen fits

def fit_readout(model, form, train, log_u, seed):
    """Train head B's readout alone, everything else frozen, to match the model's own ideal copy:
    KL(copy || head B's output), on fresh windows from the training region. Returns the fitted model and the
    loss averaged over each 50 steps."""
    m = copy.deepcopy(model)
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    params = [m.proj_b.weight, m.head2.weight]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=PEAK_LR)
    gen = torch.Generator().manual_seed(seed * 10 + FIT_SEED)
    curve, chunk, top = [], [], -1
    for step in range(FIT_STEPS):
        for grp in opt.param_groups:
            grp['lr'] = study.lr_at(step, FIT_STEPS, PEAK_LR)
        x, _, t = draw_batch(train, gen)
        top = max(top, t)
        x = x.to(DEV)
        with torch.no_grad(), torch.autocast(DEV.type, dtype=torch.float16, enabled=AMP):
            z = n10.final_state(m, x)
        z = z.float()
        with torch.no_grad():
            lc = n9.copy_logits(m.head(m.proj_a(z)), log_u, RHO)
        lb = m.head2(m.proj_b(z))
        loss = n10.kl(lc, b_noisy(lb, 'single' if form == 'single' else 'mix', log_u))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        chunk.append(loss.item())
        if len(chunk) == 50 or step == FIT_STEPS - 1:
            curve.append([step + 1, sum(chunk) / len(chunk)])
            chunk = []
    for p in m.parameters():                         # the measurement takes gradients at the ODE block's output
        p.requires_grad_(True)
    return m, curve, top


def converged(curve, k=CONVERGE_CHUNKS):
    """The mean fit loss over the last k chunks, against the k before: relative change, and whether within 2%, or
    (candidate amendment A1) the loss is already below AT_TARGET, where a relative change means nothing."""
    if len(curve) < 2 * k:
        return {'relative_change': None, 'pass': False}
    last = sum(v for _, v in curve[-k:]) / k
    prev = sum(v for _, v in curve[-2 * k:-k]) / k
    rel = abs(last - prev) / prev
    return {'last': last, 'previous': prev, 'relative_change': rel, 'pass': rel <= 0.02 or last <= AT_TARGET}


# ------------------------------------------------------------------ measurement

@torch.no_grad()
def functional(model, kind, xs, y_clean, y_noisy, log_u, batch=MEASURE_BATCH):
    """Per position: head A's, head B's (noisy-target prediction), head B's inner softmax's and the copy's losses
    on the clean targets; head B's and the copy's on the noisy targets; KL both ways between head B and the copy."""
    model.eval()
    keys = ('A_clean', 'B_clean', 'B_inner_clean', 'copy_clean', 'B_noisy', 'copy_noisy', 'kl_copy_to_B', 'kl_B_to_copy')
    tot, cnt = dict.fromkeys(keys, 0.0), 0
    for i in range(0, len(xs), batch):
        x, yc, yn = xs[i:i + batch].to(DEV), y_clean[i:i + batch].to(DEV), y_noisy[i:i + batch].to(DEV)
        la, lb = n4.forward_n4(model, x)
        lpa, lpb, lpi, lpc = (torch.log_softmax(la.float(), -1), b_noisy(lb, kind, log_u), torch.log_softmax(lb.float(), -1),
                              n9.copy_logits(la, log_u, RHO))
        n = yc.numel()
        for k, lp, y in (('A_clean', lpa, yc), ('B_clean', lpb, yc), ('B_inner_clean', lpi, yc), ('copy_clean', lpc, yc),
                         ('B_noisy', lpb, yn), ('copy_noisy', lpc, yn)):
            tot[k] -= lp.gather(-1, y.unsqueeze(-1)).sum().item()
        tot['kl_copy_to_B'] += n10.kl(lpc, lpb).item() * n
        tot['kl_B_to_copy'] += n10.kl(lpb, lpc).item() * n
        cnt += n
    out = {k: v / cnt for k, v in tot.items()}
    out['G'] = out['B_clean'] - out['copy_clean']
    return out


def constructed_check(model, mx, u, y_clean, log_u):
    """Precondition 2: a mixture head given head A's readout must reproduce the ideal copy exactly, through the
    same code as the mixture arm's training loss and measurement."""
    m = copy.deepcopy(model)
    with torch.no_grad():
        m.proj_b.weight.copy_(m.proj_a.weight)
        m.head2.weight.copy_(m.head.weight)
    G = n9.head_covs(m, mx, u, {'B': lambda la, lb: b_noisy(lb, 'mix', log_u), 'copy': lambda la, lb: n9.copy_logits(la, log_u, RHO)})
    with torch.no_grad():
        m.eval()
        x, y = mx[:MEASURE_BATCH].to(DEV), y_clean[:MEASURE_BATCH].to(DEV)
        la, lb = n4.forward_n4(m, x)
        loss_mix = head_b_loss(lb, 'mix', y, None, log_u).item()
        loss_copy = F.nll_loss(n9.copy_logits(la, log_u, RHO).reshape(-1, VOCAB), y.reshape(-1)).item()
    ratio = n9.pair(G, 'B', 'copy')['agreement_ratio']
    return {'agreement_ratio': ratio, 'loss_mixture_head': loss_mix, 'loss_copy': loss_copy,
            'loss_difference': loss_mix - loss_copy, 'pass': ratio >= 0.999 and abs(loss_mix - loss_copy) <= 1e-4}


def measure(model, arm, mx, mys, u, log_u, probs):
    rho, kind = ARMS[arm]
    t0 = time.time()
    fns = {'A': lambda la, lb: la.float(), 'B': lambda la, lb: b_noisy(lb, kind, log_u),
           'copy': lambda la, lb: n9.copy_logits(la, log_u, RHO), 'copy0': lambda la, lb: n9.copy_logits(la, log_u, 0.0)}
    if kind != 'single':
        fns['B_inner'] = lambda la, lb: lb.float()
    G = n9.head_covs(model, mx, u, fns)
    y_noisy = n10.fixed_noisy(mys[0], probs, RHO, MEAS_NOISE_SEED)
    res = {'rho': rho, 'head_b': kind,
           'B_vs_copy': n9.pair(G, 'B', 'copy'), 'B_vs_A': n9.pair(G, 'B', 'A'), 'A_vs_copy': n9.pair(G, 'A', 'copy'),
           'copy0_vs_A': n9.pair(G, 'copy0', 'A'),
           'B_inner_vs_A': n9.pair(G, 'B_inner', 'A') if kind != 'single' else None,
           'stability': {k: n8.overlap(n8.top(G[k][0]), n8.top(G[k][1])) for k in G},
           'rotated_control': n8.scalars(n8.measure_f(n8.rotated_control(model), mx, u)),
           'functional': functional(model, kind, mx, mys[0], y_noisy, log_u),
           'constructed_mixture': constructed_check(model, mx, u, mys[0], log_u),
           'readout_overlap_A_B': n8.rowspace_overlap(model.proj_a.weight, model.proj_b.weight)}
    res['seconds'] = time.time() - t0
    return res


def measure_fit(m, form, curve, top, mx, mys, u, log_u, probs):
    kind = 'single' if form == 'single' else 'mix'
    G = n9.head_covs(m, mx, u, {'B': lambda la, lb: b_noisy(lb, kind, log_u), 'copy': lambda la, lb: n9.copy_logits(la, log_u, RHO)})
    return {'form': form, 'steps': FIT_STEPS, 'curve_every_50': curve, 'convergence': converged(curve), 'max_offset': top,
            'functional': functional(m, kind, mx, mys[0], n10.fixed_noisy(mys[0], probs, RHO, MEAS_NOISE_SEED), log_u),
            'B_vs_copy': n9.pair(G, 'B', 'copy'), 'stability': {k: n8.overlap(n8.top(G[k][0]), n8.top(G[k][1])) for k in G}}


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--source', default=None)
    ap.add_argument('--local', default=None)
    ap.add_argument('--calibration', action='store_true', help='the calibration run: seed 3001, the tail\'s windows 513-1024')
    args = ap.parse_args()
    mode = 'calibration' if args.calibration else 'real'
    seeds, (w0, w1) = MODES[mode]['seeds'], MODES[mode]['windows']
    n6.SEEDS = g.SEEDS = seeds
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
    log('device:', DEV, '| test mode:', TEST, '| mode:', mode, '| study folder (data):', source, '| N12 folder:', work, '| local:', local)
    store.pull_dir('results')
    try:
        meta = study.prepare_data(data_store, None)
        full = study.load_tokens(store, meta, 'train')
        cut = int(len(full) * CUT_FRACTION)
        train, tail = full[:cut], full[cut:]
        probs = n7.unigram(train)
        log_u = probs.float().log().to(DEV)
        summary_path = store.L('results/n12_summary.json')
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

        def save_json(rel, obj):
            store.L(rel).write_text(json.dumps(obj, indent=2 if rel.endswith('n12_summary.json') else None))
            store.push(rel)

        allx, allys = n6.windows(tail, limit=w1)
        mx, mys = allx[w0:w1], [y[w0:w1] for y in allys]
        u = n8.uniforms(len(mx))
        log(f'training region: tokens 0-{cut} of {len(full)}; measured on the held-out tail\'s windows {w0 + 1}-{w0 + len(mx)} '
            f'(of {(len(tail) - K_MAX) // SEQ})')

        if 'checks' not in summary:
            enc = study.gpt2_encoding()
            summary['checks'] = {
                'batch': n6.batch_check(train), 'batch_g1': g.batch_check(train),
                'same_model': [g.same_model_check(train, s) for s in seeds],
                'windows_shifted': all(torch.equal(mys[k - 1][:, :-k], mx[:, k:]) for k in range(1, 9)),
                'analytic': n9.analytic_check(),
                'identical_heads': n8.identical_heads_check(seeds[0], mx, u),
                'unigram': {'sum': probs.sum().item(), 'zeros': int((probs == 0).sum()), 'from_tokens': cut,
                            'most_frequent_tokens': [enc.decode([t]) for t in torch.topk(probs, 5).indices.tolist()]},
                'first_measurement_text': enc.decode(mx[0, :20].tolist())}
            summary['mode'] = mode
            summary['held_out'] = {'cut_fraction': CUT_FRACTION, 'cut_token': cut, 'training_tokens': len(full), 'tail_tokens': len(tail),
                                   'tail_windows': (len(tail) - K_MAX) // SEQ, 'measure': [w0 + 1, w0 + len(mx)]}
            summary['arms'] = {a: {'rho': r, 'head_b': k} for a, (r, k) in ARMS.items()}
            summary['fits'] = {'forms': list(FITS), 'steps': FIT_STEPS, 'peak_lr': PEAK_LR, 'batch': BATCH}
            summary['seeds'], summary['readout_rank'], summary['subspace_rank'] = seeds, n4.RANK, n9.R
            save_json('results/n12_summary.json', summary)
        c = summary['checks']
        log('batch check:', c['batch']['targets_are_i+1_to_i+8'], '| windows shifted:', c['windows_shifted'],
            '| analytic:', round(c['analytic']['sampled']['top_subspace_overlap'], 4), '| identical heads:', c['identical_heads']['agreement_ratio'])
        log('same-model check:', [x['pass'] for x in c['same_model']])
        if not (c['batch']['targets_are_i+1_to_i+8'] and c['windows_shifted'] and c['analytic']['pass'] and c['identical_heads']['pass']
                and all(x['pass'] for x in c['same_model']) and abs(c['unigram']['sum'] - 1) < 1e-6):
            raise SystemExit('A check of the setup failed before training. Stopping.')

        for seed in seeds:
            for arm in ARMS:
                tag = f'{arm}_seed{seed}'
                train_n12(store, meta, arm, seed, tag, train, probs, log_u)
                rel = f'results/m_{tag}.json'
                if store.L(rel).exists():
                    continue
                if not store.pull(f'final/{tag}.pt'):
                    raise SystemExit(f'weights final/{tag}.pt are missing locally and on Drive.')
                m = n4.new_n4(seed, BASE)
                m.load_state_dict(torch.load(store.L(f'final/{tag}.pt'), map_location=DEV, weights_only=False))
                res = {'tag': tag, 'seed': seed, 'arm': arm, **measure(m, arm, mx, mys, u, log_u, probs), 'environment': study.env_record()}
                save_json(rel, res)
                fn = res['functional']
                log(f"m_{tag}: head B clean {fn['B_clean']:.3f} | copy clean {fn['copy_clean']:.3f} | head A clean {fn['A_clean']:.3f} | "
                    f"B vs copy {res['B_vs_copy']['agreement_ratio']:.3f} | B noisy {fn['B_noisy']:.3f} (copy {fn['copy_noisy']:.3f}) | "
                    f"constructed {res['constructed_mixture']['agreement_ratio']:.4f} | rotated {res['rotated_control']['agreement_ratio']:.3f} | "
                    f"{res['seconds']:.0f} s")
            for form in FITS:
                rel = f'results/f_{form}_seed{seed}.json'
                if store.L(rel).exists():
                    continue
                t0 = time.time()
                if not store.pull(f'final/n80_seed{seed}.pt'):
                    raise SystemExit(f'weights final/n80_seed{seed}.pt are missing locally and on Drive.')
                m = n4.new_n4(seed, BASE)
                m.load_state_dict(torch.load(store.L(f'final/n80_seed{seed}.pt'), map_location=DEV, weights_only=False))
                fitted, curve, top = fit_readout(m, form, train, log_u, seed)
                res = {'tag': f'fit-{form}_seed{seed}', 'seed': seed, 'model': f'n80_seed{seed}',
                       **measure_fit(fitted, form, curve, top, mx, mys, u, log_u, probs), 'seconds': time.time() - t0,
                       'environment': study.env_record()}
                save_json(rel, res)
                fn = res['functional']
                log(f"f_{form}_seed{seed}: fit KL {curve[-1][1]:.4f} (change {res['convergence']['relative_change']}) | "
                    f"head B clean {fn['B_clean']:.3f} | copy clean {fn['copy_clean']:.3f} | B vs copy {res['B_vs_copy']['agreement_ratio']:.3f} | "
                    f"{res['seconds']:.0f} s")
        log(f'N12 {mode} complete: all runs, fits and measurements finished')
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
