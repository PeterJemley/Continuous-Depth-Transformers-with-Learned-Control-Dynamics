"""N13: the mixture head's learning signal, separated from its output form.

Implements N13_PREREGISTRATION.md in this folder. N12's model, training, windows and measures (n12_bottleneck),
on N12's seeds, with three new arms trained for this test and N12's saved models re-measured:
  n80-rw       single softmax, loss stopgrad(r(y)) * CE(l, y), r(y) = 0.2 p(y) / (0.2 p(y) + 0.8 u(y)): the discount alone,
               which by the chain rule is the mixture loss's gradient
  n80-mix-rep  N12's n80-mix trained again from the same seed: the pipeline's own noise floor
  n80-w02      single softmax, loss 0.2 * CE(l, y): the scale-only control
  n80, n80-mix, sub20   N12's saved models, re-measured on the same windows (precondition 2), n80 also read out in
               the mixture form (N13b) and n80-w02 also in the mixture form (S3)
Every training run records, every 50 steps, the mean discount over the batch and over the kept and replaced positions.
Each new model is also compared with its N12 partner (n80-rw and n80-mix-rep with n80-mix; n80-w02 with n80) by the
per-position divergence between their predictions on the measurement windows.

    python n13_signal.py --out OUT --work /content/drive/MyDrive/cdt-colab/n13_signal
    python n13_signal.py --calibration --out OUT --work /content/drive/MyDrive/cdt-colab/n13_calib
    STUDY_TEST=1 python n13_signal.py --out OUT --work W --source LOCAL_STUDY --n12 N12_FOLDER --n12-records N12_RESULTS
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import n12_bottleneck as n12  # noqa: E402  (imports N10, N9, N8, N7, N6r, N6, N4, N2, G1 and the study)

n10, n9, n8, n6, n4, g, study, DEV, TEST, log = n12.n10, n12.n9, n12.n8, n12.n6, n12.n4, n12.g, n12.study, n12.DEV, n12.TEST, n12.log
AMP, VOCAB, RHO, BASE, TOTAL, PEAK_LR = n12.AMP, n12.VOCAB, n12.RHO, n12.BASE, n12.TOTAL, n12.PEAK_LR
MODES = n12.MODES                                    # the same seeds and windows as N12, by design
NEW_ARMS = {'n80-rw': (RHO, 'rw'), 'n80-mix-rep': (RHO, 'mix'), 'n80-w02': (RHO, 'w02')}
PARTNER = {'n80-rw': 'n80-mix', 'n80-mix-rep': 'n80-mix', 'n80-w02': 'n80'}
N12_ARMS = ('n80', 'n80-mix', 'sub20')               # re-measured
MIXFORM = {'n80': 'n80-mixform', 'n80-w02': 'n80-w02-mixform'}      # the same model, read out in the mixture form
n12.ARMS.update(NEW_ARMS)
n12.ARMS.update({'n80-mixform': (RHO, 'mix'), 'n80-w02-mixform': (RHO, 'mix')})
LOG_RHO, LOG_1MRHO = math.log(RHO), math.log(1 - RHO)
GRAD_TOL = 1e-4                                      # the identity check: relative difference of the two losses' gradients


# ------------------------------------------------------------------ the discount, the new losses, and the output forms

def discount(lbf, y, log_u):
    """r(y) = 0.2 p(y) / (0.2 p(y) + 0.8 u(y)): the posterior, under the head's own prediction, that y is a clean target.
    Where u(y) = 0 (a token the training text never produced) the target must be clean, and r = 1."""
    lpy = torch.log_softmax(lbf, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    lq = torch.logaddexp(lpy + LOG_1MRHO, log_u[y] + LOG_RHO)
    return (lpy + LOG_1MRHO - lq).exp()


_n12_head_b_loss, _n12_b_noisy = n12.head_b_loss, n12.b_noisy


def head_b_loss(lb, kind, yb, keep, log_u):
    if kind == 'rw':
        lbf, y = lb.reshape(-1, VOCAB).float(), yb.reshape(-1)
        with torch.no_grad():
            r = discount(lbf, y, log_u)
        return (r * F.cross_entropy(lbf, y, reduction='none')).mean()
    if kind == 'w02':
        return 0.2 * F.cross_entropy(lb.reshape(-1, VOCAB), yb.reshape(-1))
    return _n12_head_b_loss(lb, kind, yb, keep, log_u)


def b_noisy(lb, kind, log_u):
    """n80-rw's head estimates the clean reading, so its noisy-target prediction is the mixture form, like n80-mix's;
    n80-w02's is a plain softmax, like n80's."""
    return _n12_b_noisy(lb, 'single' if kind == 'w02' else kind, log_u)


n12.head_b_loss, n12.b_noisy = head_b_loss, b_noisy          # n12's training, measurement and checks call these by name


def gradient_identity_check(model, x, ya, yb, keep, log_u):
    """Precondition 1: on one batch, the mixture loss and the discounted cross-entropy give the same gradient for every
    parameter (the chain rule), and 0.2 * CE gives 0.2 times the plain cross-entropy's. In float32, through the job's
    own loss code and the same forward pass."""
    out = {}
    model.eval()
    la, lb = n4.forward_n4(model, x)
    grads = {}
    for kind in ('mix', 'rw', 'single', 'w02'):
        loss = 0.5 * (F.cross_entropy(la.float().view(-1, VOCAB), ya.reshape(-1)) + head_b_loss(lb.float(), kind, yb, keep, log_u))
        grads[kind] = torch.autograd.grad(loss, list(model.parameters()), retain_graph=True, allow_unused=True)
    model.train()

    def rel(ga, gb, scale_b=1.0):
        num = den = 0.0
        for a, b in zip(ga, gb):
            if a is None and b is None:
                continue
            a = torch.zeros(1, device=DEV) if a is None else a.double()
            b = torch.zeros(1, device=DEV) if b is None else b.double()
            num += (a - scale_b * b).pow(2).sum().item(); den += a.pow(2).sum().item()
        return math.sqrt(num / den) if den > 0 else float('nan')
    # the head-A part is shared, so compare the whole gradient: identity for mix vs rw, and 0.2-scaling of head B's part
    out['mix_vs_rw_relative'] = rel(grads['mix'], grads['rw'])
    # w02 = 0.5 CE_A + 0.1 CE_B against single = 0.5 CE_A + 0.5 CE_B: check head B's own parameters, where only CE_B acts
    names = [n for n, _ in model.named_parameters()]
    idx = [i for i, n in enumerate(names) if n.startswith('proj_b') or n.startswith('head2')]
    out['w02_vs_single_headB_relative'] = rel([grads['w02'][i] for i in idx], [grads['single'][i] for i in idx], scale_b=0.2)
    out['pass'] = out['mix_vs_rw_relative'] <= GRAD_TOL and out['w02_vs_single_headB_relative'] <= GRAD_TOL
    return out


# ------------------------------------------------------------------ training: N12's loop, plus the discount's statistics

def train_n13(store, meta, arm, seed, tag, train, probs, log_u):
    """One run, resumable. n12.train_n12 with one addition: every 50 steps, the mean discount r over the batch and over
    the kept and replaced positions, computed without gradient, so the update is unchanged."""
    rho, kind = n12.ARMS[arm]
    res_path = store.L(f'results/{tag}.json')
    if res_path.exists():
        log(f'{tag}: already finished, skipping')
        return json.loads(res_path.read_text())
    ck_rel = f'ckpt/{tag}.pt'
    store.pull(ck_rel)
    ck_path = store.L(ck_rel)
    vx, vys = n6.windows(study.load_tokens(store, meta, 'validation'), limit=g.VAL_LIMIT)
    vya, vyb = vys[0], n10.fixed_noisy(vys[0], probs, rho, n12.VAL_NOISE_SEED)
    model = n4.new_n4(seed, BASE)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR)
    scaler = torch.amp.GradScaler('cuda', enabled=AMP)
    gen = torch.Generator().manual_seed(seed)
    state = {'step': 0, 'train_loss': [], 'val': [], 'alpha': [], 'r': [], 'skipped_steps': 0, 'seconds': 0.0, 'resumes': 0, 'max_offset': -1}
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
        x, ys, top = n12.draw_batch(train, gen)
        state['max_offset'] = max(state['max_offset'], top)
        ya = ys[0]
        yb, keep, mask = n12.step_targets(ya, probs, seed, step, rho, kind)
        if step == 0:
            state['batch_fingerprint_step0'] = n12.fingerprint(x)
            if mask is not None:
                state['noise_fraction_step0'] = mask.float().mean().item()
                state['replaced_positions_fingerprint_step0'] = n12.fingerprint(mask)
            state['kept_fraction_step0'] = keep.float().mean().item()
        x, ya, yb, keep = x.to(DEV), ya.to(DEV), yb.to(DEV), keep.to(DEV)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=AMP):
            la, lb = n4.forward_n4(model, x)
            loss = 0.5 * (F.cross_entropy(la.view(-1, VOCAB), ya.reshape(-1)) + head_b_loss(lb, kind, yb, keep, log_u))
        if step % 50 == 0:                                                      # the discount's statistics, no gradient
            with torch.no_grad():
                r = discount(lb.detach().reshape(-1, VOCAB).float(), yb.reshape(-1), log_u)
                k = keep.reshape(-1)
                state['r'].append([step, r.mean().item(), r[k].mean().item(), r[~k].mean().item() if (~k).any() else None])
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
            l_a, l_b = n12.val_losses(model, vx, vya, vyb, kind, log_u)[:2]
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
    l_a, l_b, l_b_clean, l_inner = n12.val_losses(model, fx, fys[0], n10.fixed_noisy(fys[0], probs, rho, n12.VAL_NOISE_SEED), kind, log_u)
    result = {'tag': tag, 'arm': arm, 'rho': rho, 'head_b': kind, 'seed': seed, 'readout_rank': n4.RANK,
              'final_loss_A': l_a, 'final_loss_B': l_b, 'final_loss_B_clean_targets': l_b_clean, 'final_loss_B_inner_clean': l_inner,
              'final_val_windows': len(fx), 'val_curve': state['val'], 'train_loss_every_50': state['train_loss'],
              'r_every_50': state['r'],                                        # [step, mean r, mean r on kept, mean r on replaced]
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


# ------------------------------------------------------------------ the cross-model comparison

@torch.no_grad()
def divergence(m_new, m_old, xs, log_u, batch=n12.MEASURE_BATCH):
    """Per-position KL divergences between two models' predictions on the same windows: head B's inner softmax both
    ways and symmetrised, the same for the mixture form of head B, and for head A."""
    m_new.eval(); m_old.eval()
    keys = ('inner_new_to_old', 'inner_old_to_new', 'mix_new_to_old', 'mix_old_to_new', 'A_new_to_old', 'A_old_to_new')
    tot, cnt = dict.fromkeys(keys, 0.0), 0
    for i in range(0, len(xs), batch):
        x = xs[i:i + batch].to(DEV)
        la1, lb1 = n4.forward_n4(m_new, x)
        la2, lb2 = n4.forward_n4(m_old, x)
        n = x.numel()
        for key, p1, p2 in (('inner', torch.log_softmax(lb1.float(), -1), torch.log_softmax(lb2.float(), -1)),
                            ('mix', n9.copy_logits(lb1, log_u, RHO), n9.copy_logits(lb2, log_u, RHO)),
                            ('A', torch.log_softmax(la1.float(), -1), torch.log_softmax(la2.float(), -1))):
            tot[f'{key}_new_to_old'] += n10.kl(p1, p2).item() * n
            tot[f'{key}_old_to_new'] += n10.kl(p2, p1).item() * n
        cnt += n
    out = {k: v / cnt for k, v in tot.items()}
    for key in ('inner', 'mix', 'A'):
        out[f'{key}_symmetrised'] = 0.5 * (out[f'{key}_new_to_old'] + out[f'{key}_old_to_new'])
    return out


def load_model(store, tag, seed):
    if not store.pull(f'final/{tag}.pt'):
        raise SystemExit(f'weights {store.drive / "final" / (tag + ".pt")} are missing locally and on Drive.')
    m = n4.new_n4(seed, BASE)
    m.load_state_dict(torch.load(store.L(f'final/{tag}.pt'), map_location=DEV, weights_only=False))
    return m


def reproduction(res, ref):
    """How far a re-measurement of an N12 model is from N12's record of it (precondition 2)."""
    f, fr = res['functional'], ref['functional']
    losses = {k: abs(f[k] - fr[k]) for k in ('A_clean', 'B_clean', 'B_inner_clean', 'copy_clean', 'B_noisy', 'copy_noisy')}
    ratios = {k: abs(res[k]['agreement_ratio'] - ref[k]['agreement_ratio']) for k in ('B_vs_copy', 'B_vs_A', 'A_vs_copy', 'copy0_vs_A')}
    ratios['rotated_control'] = abs(res['rotated_control']['agreement_ratio'] - ref['rotated_control']['agreement_ratio'])
    return {'max_loss_difference': max(losses.values()), 'max_ratio_difference': max(ratios.values()), 'losses': losses, 'ratios': ratios,
            'pass': max(losses.values()) <= 1e-3 and max(ratios.values()) <= 1e-2}


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--source', default=None)
    ap.add_argument('--local', default=None)
    ap.add_argument('--n12', default=None, help="N12's Drive folder (default: n12_bottleneck, or n12_calib, next to --work)")
    ap.add_argument('--n12-records', default=None, help="N12's committed records (default: results/n12 or results/n12_calib here)")
    ap.add_argument('--calibration', action='store_true', help='the calibration run: seed 3001, the tail\'s windows 513-1024')
    args = ap.parse_args()
    mode = 'calibration' if args.calibration else 'real'
    seeds, (w0, w1) = MODES[mode]['seeds'], MODES[mode]['windows']
    n6.SEEDS = g.SEEDS = seeds
    work = Path(args.work)
    source = Path(args.source) if args.source else work.parent / 'wt103_study'
    n12_folder = Path(args.n12) if args.n12 else work.parent / ('n12_calib' if args.calibration else 'n12_bottleneck')
    n12_records = Path(args.n12_records) if args.n12_records else HERE / 'results' / ('n12_calib' if args.calibration else 'n12')
    if args.local:
        local = Path(args.local)
    elif Path('/content').is_dir() and not TEST:
        local = Path('/content/cdt_local') / work.name
    else:
        local = work.parent / f'_local_{work.name}'
    data_store = study.Store(source, local)
    store = study.Store(work, local)
    src = study.Store(n12_folder, local.parent / f'_n13_src_{n12_folder.name}')
    out = Path(args.out)
    (out / 'results').mkdir(parents=True, exist_ok=True)
    log('device:', DEV, '| test mode:', TEST, '| mode:', mode, '| study folder (data):', source, '| N13 folder:', work, '| local:', local)
    log("N12's models from:", src.drive / 'final', "| N12's records from:", n12_records)
    store.pull_dir('results')
    try:
        meta = study.prepare_data(data_store, None)
        full = study.load_tokens(store, meta, 'train')
        cut = int(len(full) * n12.CUT_FRACTION)
        train, tail = full[:cut], full[cut:]
        probs = n12.n7.unigram(train)
        log_u = probs.float().log().to(DEV)
        summary_path = store.L('results/n13_summary.json')
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

        def save_json(rel, obj):
            store.L(rel).write_text(json.dumps(obj, indent=2 if rel.endswith('n13_summary.json') else None))
            store.push(rel)

        allx, allys = n6.windows(tail, limit=w1)
        mx, mys = allx[w0:w1], [y[w0:w1] for y in allys]
        u = n8.uniforms(len(mx))
        log(f'training region: tokens 0-{cut} of {len(full)}; measured on the held-out tail\'s windows {w0 + 1}-{w0 + len(mx)}')

        if 'checks' not in summary:
            x, ys, _ = n12.draw_batch(train, torch.Generator().manual_seed(seeds[0]))
            yb, keep, _ = n12.step_targets(ys[0], probs, seeds[0], 0, RHO, 'single')
            summary['checks'] = {
                'batch': n6.batch_check(train), 'batch_g1': g.batch_check(train),
                'same_model': [g.same_model_check(train, s) for s in seeds],
                'windows_shifted': all(torch.equal(mys[k - 1][:, :-k], mx[:, k:]) for k in range(1, 9)),
                'analytic': n9.analytic_check(),
                'identical_heads': n8.identical_heads_check(seeds[0], mx, u),
                'gradient_identity': gradient_identity_check(n4.new_n4(seeds[0], BASE), x.to(DEV), ys[0].to(DEV), yb.to(DEV), keep.to(DEV), log_u),
                'unigram': {'sum': probs.sum().item(), 'zeros': int((probs == 0).sum()), 'from_tokens': cut}}
            summary['mode'] = mode
            summary['held_out'] = {'cut_fraction': n12.CUT_FRACTION, 'cut_token': cut, 'training_tokens': len(full), 'tail_tokens': len(tail),
                                   'tail_windows': (len(tail) - n12.K_MAX) // n12.SEQ, 'measure': [w0 + 1, w0 + len(mx)]}
            summary['arms'] = {a: {'rho': r, 'head_b': k} for a, (r, k) in NEW_ARMS.items()}
            summary['n12'] = {'folder': str(n12_folder), 'records': str(n12_records), 'arms': list(N12_ARMS), 'partners': PARTNER}
            summary['seeds'], summary['readout_rank'], summary['subspace_rank'] = seeds, n4.RANK, n9.R
            save_json('results/n13_summary.json', summary)
        c = summary['checks']
        log('batch check:', c['batch']['targets_are_i+1_to_i+8'], '| windows shifted:', c['windows_shifted'],
            '| analytic:', round(c['analytic']['sampled']['top_subspace_overlap'], 4), '| identical heads:', c['identical_heads']['agreement_ratio'],
            '| gradient identity: mix vs rw', f"{c['gradient_identity']['mix_vs_rw_relative']:.2e}", ', w02 vs 0.2 single',
            f"{c['gradient_identity']['w02_vs_single_headB_relative']:.2e}", c['gradient_identity']['pass'])
        log('same-model check:', [x['pass'] for x in c['same_model']])
        if not (c['batch']['targets_are_i+1_to_i+8'] and c['windows_shifted'] and c['analytic']['pass'] and c['identical_heads']['pass']
                and c['gradient_identity']['pass'] and all(x['pass'] for x in c['same_model']) and abs(c['unigram']['sum'] - 1) < 1e-6):
            raise SystemExit('A check of the setup failed before training. Stopping.')

        for seed in seeds:
            # N12's models, re-measured (precondition 2), n80 also in the mixture form (N13b)
            n12_models = {}
            for arm in N12_ARMS:
                tag = f'{arm}_seed{seed}'
                n12_models[arm] = load_model(src, tag, seed)
                for meas_arm in (arm,) + ((MIXFORM[arm],) if arm in MIXFORM else ()):
                    rel = f'results/r_{meas_arm}_seed{seed}.json'
                    if store.L(rel).exists():
                        continue
                    res = {'tag': f'{meas_arm}_seed{seed}', 'seed': seed, 'arm': meas_arm, 'model': tag,
                           **n12.measure(n12_models[arm], meas_arm, mx, mys, u, log_u, probs), 'environment': study.env_record()}
                    if meas_arm == arm:
                        ref_path = n12_records / f'm_{tag}.json'
                        res['reproduction'] = reproduction(res, json.loads(ref_path.read_text())) if ref_path.exists() else {'pass': False, 'missing': str(ref_path)}
                    save_json(rel, res)
                    fn = res['functional']
                    rep = res.get('reproduction', {})
                    log(f"r_{meas_arm}_seed{seed}: head B clean {fn['B_clean']:.3f} | inner clean {fn['B_inner_clean']:.3f} | copy clean {fn['copy_clean']:.3f} | "
                        f"head A clean {fn['A_clean']:.3f} | B vs copy {res['B_vs_copy']['agreement_ratio']:.3f}"
                        + (f" | reproduces N12: {rep.get('pass')} (loss {rep.get('max_loss_difference', float('nan')):.1e}, ratio {rep.get('max_ratio_difference', float('nan')):.1e})" if rep else '')
                        + f" | {res['seconds']:.0f} s")
            # the new arms
            for arm in NEW_ARMS:
                tag = f'{arm}_seed{seed}'
                train_n13(store, meta, arm, seed, tag, train, probs, log_u)
                for meas_arm in (arm,) + ((MIXFORM[arm],) if arm in MIXFORM else ()):
                    rel = f'results/m_{meas_arm}_seed{seed}.json'
                    if store.L(rel).exists():
                        continue
                    m = load_model(store, tag, seed)
                    res = {'tag': f'{meas_arm}_seed{seed}', 'seed': seed, 'arm': meas_arm, 'model': tag,
                           **n12.measure(m, meas_arm, mx, mys, u, log_u, probs), 'environment': study.env_record()}
                    if meas_arm == arm:
                        res['partner'] = {'arm': PARTNER[arm], 'model': f'{PARTNER[arm]}_seed{seed}', **divergence(m, n12_models[PARTNER[arm]], mx, log_u)}
                    save_json(rel, res)
                    fn = res['functional']
                    pt = res.get('partner', {})
                    log(f"m_{meas_arm}_seed{seed}: head B clean {fn['B_clean']:.3f} | inner clean {fn['B_inner_clean']:.3f} | head A clean {fn['A_clean']:.3f} | "
                        f"B vs copy {res['B_vs_copy']['agreement_ratio']:.3f}"
                        + (f" | vs {pt['arm']}: inner KL {pt['inner_symmetrised']:.4f}, head A KL {pt['A_symmetrised']:.4f}" if pt else '')
                        + f" | {res['seconds']:.0f} s")
        log(f'N13 {mode} complete: all runs and measurements finished')
    finally:
        dest = out / 'results'
        for f in (store.local / 'results').glob('*.json'):
            (dest / f.name).write_bytes(f.read_bytes())
        log(f'copied {len(list(dest.glob("*.json")))} result files for upload')
        log('waiting for background Drive saves to finish (up to 15 minutes)')
        store.flush()
        failures = store.failures + data_store.failures + src.failures
        (out / 'drive_sync.json').write_text(json.dumps({'failures': failures}, indent=2))
        log('Drive saves finished' if not failures else f'Drive problems this session: {failures}')


if __name__ == '__main__':
    main()
