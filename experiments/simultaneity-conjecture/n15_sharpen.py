"""N15: sharpening the discount.

Implements N15_PREREGISTRATION.md in this folder. N12's model, training, batches, windows and measures (n12_bottleneck),
N13's reweighted single-softmax head and resumable loop (n13_signal), on N12's seeds at the 80% noise rate:
  rw-b2, rw-b4   head B's loss is stopgrad(w(y)) * CE(l, y) with w = r_beta(y) = 1 / (1 + 4 (u(y) / p(y))^beta), the head's own
                 posterior with its likelihood ratio raised to beta (beta = 1 is N13's discount exactly)
  rw-oracle      w = 0.2 p_A(y) / (0.2 p_A(y) + 0.8 u(y)), from head A's prediction at the same forward pass, detached
  rw-b1 (N13's n80-rw), sub20 and clean (N12), sub10, sub05, sub025, sub0125 (N14)   saved models, re-measured on the same
                 windows and checked against their committed records (precondition 2)
Every training run records, every 50 steps, the mean weight on kept and on replaced positions, the random targets' share of
the summed weight, and the fraction of kept positions whose weight is below 0.01.

    python n15_sharpen.py --out OUT --work /content/drive/MyDrive/cdt-colab/n15_sharpen
    python n15_sharpen.py --calibration --out OUT --work /content/drive/MyDrive/cdt-colab/n15_calib
    STUDY_TEST=1 python n15_sharpen.py --out OUT --work W --source LOCAL_STUDY --n12 F12 --n13 F13 --n14 F14 \
        --n12-records R12 --n13-records R13 --n14-records R14
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

import n12_bottleneck as n12  # noqa: E402
import n13_signal as n13  # noqa: E402  (its patches of n12's loss and output form stay in force: 'rw' is measured in the mixture form)

n10, n9, n8, n6, n4, g, study, DEV, TEST, log = n12.n10, n12.n9, n12.n8, n12.n6, n12.n4, n12.g, n12.study, n12.DEV, n12.TEST, n12.log
AMP, VOCAB, RHO, BASE, TOTAL, PEAK_LR = n12.AMP, n12.VOCAB, n12.RHO, n12.BASE, n12.TOTAL, n12.PEAK_LR
MODES = n12.MODES
LOG4 = math.log((1 - RHO) / RHO) * -1          # log(0.8 / 0.2) = log 4: the prior odds against a clean target
SPECS = {'rw-b2': {'beta': 2.0, 'oracle': False}, 'rw-b4': {'beta': 4.0, 'oracle': False}, 'rw-oracle': {'beta': 1.0, 'oracle': True}}
NEW_ARMS = tuple(SPECS)
REFS = {                                        # arm -> (drive folder key, model tag prefix, records key)
    'rw-b1': ('n13', 'n80-rw', 'n13'), 'sub20': ('n12', 'sub20', 'n12'), 'clean': ('n12', 'clean', 'n12'),
    'sub10': ('n14', 'sub10', 'n14'), 'sub05': ('n14', 'sub05', 'n14'), 'sub025': ('n14', 'sub025', 'n14'), 'sub0125': ('n14', 'sub0125', 'n14')}
CALIB_REFS = ('rw-b1', 'sub20', 'clean', 'sub10', 'sub05', 'sub025')     # the calibration ladder has no sub0125
for _a in NEW_ARMS:
    n12.ARMS[_a] = (RHO, 'rw')                  # measured as N13's n80-rw is: a single softmax, its noisy-target prediction in the mixture form
n12.ARMS.setdefault('sub0125', (RHO, 'sub'))
n12.ARMS.setdefault('sub10', (RHO, 'sub')); n12.ARMS.setdefault('sub05', (RHO, 'sub')); n12.ARMS.setdefault('sub025', (RHO, 'sub'))
n12.ARMS.setdefault('rw-b1', (RHO, 'rw'))
WEIGHT_TOL = 1e-6
IGNORED = 0.01


# ------------------------------------------------------------------ the weights

def weight(lf, y, log_u, beta):
    """r_beta(y) = 1 / (1 + 4 (u(y) / p(y))^beta), computed in log space from logits lf (float32) for targets y."""
    lpy = torch.log_softmax(lf, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    d = LOG4 + beta * (log_u[y] - lpy)          # log of 4 (u / p)^beta; +inf where u = 0 would give (-inf) * beta: handle below
    d = torch.where(torch.isfinite(d), d, torch.full_like(d, -float('inf')))   # u(y) = 0: the target must be clean, w = 1
    return (-torch.logaddexp(torch.zeros_like(d), d)).exp()


def head_b_weights(la, lb, yb, log_u, spec):
    """The per-position weights of an arm, without gradient: from head B's own logits at beta, or head A's at beta = 1."""
    with torch.no_grad():
        src = la if spec['oracle'] else lb
        return weight(src.detach().reshape(-1, VOCAB).float(), yb.reshape(-1), log_u, spec['beta'])


def head_b_loss(la, lb, yb, log_u, spec):
    w = head_b_weights(la, lb, yb, log_u, spec)
    return (w * F.cross_entropy(lb.reshape(-1, VOCAB).float(), yb.reshape(-1), reduction='none')).mean()


def weight_identity_check(model, x, yb, log_u):
    """Precondition 1: at beta = 1 this file's weight equals N13's discount, on a batch, through the same forward pass."""
    model.eval()
    with torch.no_grad():
        la, lb = n4.forward_n4(model, x)
        lf, y = lb.reshape(-1, VOCAB).float(), yb.reshape(-1)
        mine, theirs = weight(lf, y, log_u, 1.0), n13.discount(lf, y, log_u)
        rel = ((mine - theirs).abs().max() / theirs.abs().max()).item()
    model.train()
    return {'max_relative_difference': rel, 'pass': rel <= WEIGHT_TOL}


def oracle_detachment_check(model, x, ya, yb, log_u):
    """Precondition 1: head B's oracle-weighted loss sends no gradient to head A's readout (proj_a, head)."""
    model.eval()
    la, lb = n4.forward_n4(model, x)
    loss = head_b_loss(la, lb, yb, log_u, SPECS['rw-oracle'])
    names = [n for n, _ in model.named_parameters()]
    params = [p for _, p in model.named_parameters()]
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    head_a = {n: (0.0 if gr is None else gr.abs().max().item()) for n, gr in zip(names, grads) if n.startswith('proj_a') or n.startswith('head.')}
    trunk_max = max((gr.abs().max().item() for n, gr in zip(names, grads) if gr is not None and not (n.startswith('proj_a') or n.startswith('head.'))), default=0.0)
    model.train()
    return {'head_A_readout_max_gradient': head_a, 'trunk_and_head_B_max_gradient': trunk_max,
            'pass': all(v == 0.0 for v in head_a.values()) and trunk_max > 0}


# ------------------------------------------------------------------ training: N13's loop with the arm's weights

def train_n15(store, meta, arm, seed, tag, train, probs, log_u):
    spec = SPECS[arm]
    res_path = store.L(f'results/{tag}.json')
    if res_path.exists():
        log(f'{tag}: already finished, skipping')
        return json.loads(res_path.read_text())
    ck_rel = f'ckpt/{tag}.pt'
    store.pull(ck_rel)
    ck_path = store.L(ck_rel)
    vx, vys = n6.windows(study.load_tokens(store, meta, 'validation'), limit=g.VAL_LIMIT)
    vya, vyb = vys[0], n10.fixed_noisy(vys[0], probs, RHO, n12.VAL_NOISE_SEED)
    model = n4.new_n4(seed, BASE)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR)
    scaler = torch.amp.GradScaler('cuda', enabled=AMP)
    gen = torch.Generator().manual_seed(seed)
    state = {'step': 0, 'train_loss': [], 'val': [], 'alpha': [], 'w': [], 'skipped_steps': 0, 'seconds': 0.0, 'resumes': 0, 'max_offset': -1}
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
        label = 'oracle (head A)' if spec['oracle'] else f"beta {spec['beta']:g}"
        log(f"{tag}: starting, noise {RHO}, weight {label}, seed {seed}")

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
        yb, keep, mask = n12.step_targets(ya, probs, seed, step, RHO, 'single')
        if step == 0:
            state.update({'batch_fingerprint_step0': n12.fingerprint(x), 'replaced_positions_fingerprint_step0': n12.fingerprint(mask),
                          'noise_fraction_step0': mask.float().mean().item(), 'kept_fraction_step0': keep.float().mean().item()})
        x, ya, yb, keep = x.to(DEV), ya.to(DEV), yb.to(DEV), keep.to(DEV)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=AMP):
            la, lb = n4.forward_n4(model, x)
            loss = 0.5 * (F.cross_entropy(la.view(-1, VOCAB), ya.reshape(-1)) + head_b_loss(la, lb, yb, log_u, spec))
        if step % 50 == 0:                                                      # the weights' statistics, no gradient
            with torch.no_grad():
                w = head_b_weights(la, lb, yb, log_u, spec)
                k = keep.reshape(-1)
                sk, sr = w[k].sum().item(), w[~k].sum().item()
                state['w'].append([step, w.mean().item(), w[k].mean().item(), w[~k].mean().item(), sr / (sr + sk), (w[k] < IGNORED).float().mean().item()])
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
            l_a, l_b = n12.val_losses(model, vx, vya, vyb, 'rw', log_u)[:2]
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
    l_a, l_b, l_b_clean, l_inner = n12.val_losses(model, fx, fys[0], n10.fixed_noisy(fys[0], probs, RHO, n12.VAL_NOISE_SEED), 'rw', log_u)
    result = {'tag': tag, 'arm': arm, 'seed': seed, 'rho': RHO, 'beta': spec['beta'], 'oracle': spec['oracle'], 'head_b': 'rw', 'readout_rank': n4.RANK,
              'final_loss_A': l_a, 'final_loss_B': l_b, 'final_loss_B_clean_targets': l_b_clean, 'final_loss_B_inner_clean': l_inner,
              'final_val_windows': len(fx), 'val_curve': state['val'], 'train_loss_every_50': state['train_loss'],
              'w_every_50': state['w'],              # [step, mean w, mean w on kept, mean w on replaced, random share of the summed weight, fraction of kept with w < 0.01]
              'alpha_every_50': state['alpha'], 'alpha_final': model.ode_block.func.output_scale.item(),
              **{k: state.get(k) for k in ('batch_fingerprint_step0', 'replaced_positions_fingerprint_step0', 'noise_fraction_step0', 'kept_fraction_step0')},
              'max_offset': state['max_offset'], 'training_tokens': len(train),
              'skipped_steps_fp16': state['skipped_steps'], 'train_seconds': state['seconds'], 'resumes': state['resumes'],
              'n_params': sum(p.numel() for p in model.parameters()), 'environment': study.env_record()}
    res_path.write_text(json.dumps(result, indent=2))
    store.push(f'results/{tag}.json')
    log(f'{tag}: finished in {state["seconds"] / 3600:.2f} h; head A {l_a:.4f}, head B {l_b:.4f} (clean targets {l_b_clean:.4f}, inner {l_inner:.4f}); '
        f"final share {state['w'][-1][4]:.3f}, ignored kept {state['w'][-1][5]:.3f}")
    return result


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--source', default=None)
    ap.add_argument('--local', default=None)
    for k, real, cal in (('n12', 'n12_bottleneck', 'n12_calib'), ('n13', 'n13_signal', 'n13_calib'), ('n14', 'n14_dose', 'n14_calib')):
        ap.add_argument(f'--{k}', default=None, help=f"{k.upper()}'s Drive folder (default: {real}, or {cal}, next to --work)")
        ap.add_argument(f'--{k}-records', default=None, help=f"{k.upper()}'s committed records (default: results/{k} or results/{k}_calib here)")
    ap.add_argument('--calibration', action='store_true', help='the calibration run: seed 3001, the tail\'s windows 513-1024')
    args = ap.parse_args()
    mode = 'calibration' if args.calibration else 'real'
    seeds, (w0, w1) = MODES[mode]['seeds'], MODES[mode]['windows']
    n6.SEEDS = g.SEEDS = seeds
    work = Path(args.work)
    source = Path(args.source) if args.source else work.parent / 'wt103_study'
    folders, records = {}, {}
    for k, real, cal in (('n12', 'n12_bottleneck', 'n12_calib'), ('n13', 'n13_signal', 'n13_calib'), ('n14', 'n14_dose', 'n14_calib')):
        v = getattr(args, k); r = getattr(args, f'{k}_records')
        folders[k] = Path(v) if v else work.parent / (cal if args.calibration else real)
        records[k] = Path(r) if r else HERE / 'results' / (f'{k}_calib' if args.calibration else k)
    if args.local:
        local = Path(args.local)
    elif Path('/content').is_dir() and not TEST:
        local = Path('/content/cdt_local') / work.name
    else:
        local = work.parent / f'_local_{work.name}'
    data_store = study.Store(source, local)
    store = study.Store(work, local)
    srcs = {k: study.Store(folders[k], local.parent / f'_n15_src_{folders[k].name}') for k in folders}
    out = Path(args.out)
    (out / 'results').mkdir(parents=True, exist_ok=True)
    refs = CALIB_REFS if args.calibration else tuple(REFS)
    log('device:', DEV, '| test mode:', TEST, '| mode:', mode, '| study folder (data):', source, '| N15 folder:', work, '| local:', local)
    log('saved models from:', {k: str(f) for k, f in folders.items()}, '| records from:', {k: str(r) for k, r in records.items()})
    store.pull_dir('results')
    try:
        meta = study.prepare_data(data_store, None)
        full = study.load_tokens(store, meta, 'train')
        cut = int(len(full) * n12.CUT_FRACTION)
        train, tail = full[:cut], full[cut:]
        probs = n12.n7.unigram(train)
        log_u = probs.float().log().to(DEV)
        summary_path = store.L('results/n15_summary.json')
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

        def save_json(rel, obj):
            store.L(rel).write_text(json.dumps(obj, indent=2 if rel.endswith('n15_summary.json') else None))
            store.push(rel)

        allx, allys = n6.windows(tail, limit=w1)
        mx, mys = allx[w0:w1], [y[w0:w1] for y in allys]
        u = n8.uniforms(len(mx))
        log(f'training region: tokens 0-{cut} of {len(full)}; measured on the held-out tail\'s windows {w0 + 1}-{w0 + len(mx)}')

        if 'checks' not in summary:
            x, ys, _ = n12.draw_batch(train, torch.Generator().manual_seed(seeds[0]))
            yb, keep, _ = n12.step_targets(ys[0], probs, seeds[0], 0, RHO, 'single')
            m0 = n4.new_n4(seeds[0], BASE)
            summary['checks'] = {
                'batch': n6.batch_check(train), 'batch_g1': g.batch_check(train),
                'same_model': [g.same_model_check(train, s) for s in seeds],
                'windows_shifted': all(torch.equal(mys[k - 1][:, :-k], mx[:, k:]) for k in range(1, 9)),
                'analytic': n9.analytic_check(),
                'identical_heads': n8.identical_heads_check(seeds[0], mx, u),
                'weight_identity': weight_identity_check(m0, x.to(DEV), yb.to(DEV), log_u),
                'oracle_detachment': oracle_detachment_check(m0, x.to(DEV), ys[0].to(DEV), yb.to(DEV), log_u),
                'unigram': {'sum': probs.sum().item(), 'zeros': int((probs == 0).sum()), 'from_tokens': cut}}
            summary['mode'] = mode
            summary['held_out'] = {'cut_fraction': n12.CUT_FRACTION, 'cut_token': cut, 'training_tokens': len(full), 'tail_tokens': len(tail),
                                   'tail_windows': (len(tail) - n12.K_MAX) // n12.SEQ, 'measure': [w0 + 1, w0 + len(mx)]}
            summary['arms'] = SPECS
            summary['references'] = {a: {'folder': str(folders[REFS[a][0]]), 'tag': REFS[a][1], 'records': str(records[REFS[a][2]])} for a in refs}
            summary['seeds'], summary['readout_rank'], summary['subspace_rank'] = seeds, n4.RANK, n9.R
            save_json('results/n15_summary.json', summary)
        c = summary['checks']
        log('batch check:', c['batch']['targets_are_i+1_to_i+8'], '| windows shifted:', c['windows_shifted'],
            '| analytic:', round(c['analytic']['sampled']['top_subspace_overlap'], 4), '| identical heads:', c['identical_heads']['agreement_ratio'],
            '| weight identity at beta 1:', f"{c['weight_identity']['max_relative_difference']:.1e}", c['weight_identity']['pass'],
            '| oracle detachment:', c['oracle_detachment']['pass'])
        log('same-model check:', [x['pass'] for x in c['same_model']])
        if not (c['batch']['targets_are_i+1_to_i+8'] and c['windows_shifted'] and c['analytic']['pass'] and c['identical_heads']['pass']
                and c['weight_identity']['pass'] and c['oracle_detachment']['pass'] and all(x['pass'] for x in c['same_model'])
                and abs(c['unigram']['sum'] - 1) < 1e-6):
            raise SystemExit('A check of the setup failed before training. Stopping.')

        for seed in seeds:
            for arm in refs:                                                  # the saved models, re-measured (precondition 2)
                rel = f'results/r_{arm}_seed{seed}.json'
                if store.L(rel).exists():
                    continue
                fkey, prefix, rkey = REFS[arm]
                tag = f'{prefix}_seed{seed}'
                m = n13.load_model(srcs[fkey], tag, seed)
                res = {'tag': f'{arm}_seed{seed}', 'seed': seed, 'arm': arm, 'model': tag, **n12.measure(m, arm, mx, mys, u, log_u, probs), 'environment': study.env_record()}
                ref_path = records[rkey] / f'm_{tag}.json'
                res['reproduction'] = n13.reproduction(res, json.loads(ref_path.read_text())) if ref_path.exists() else {'pass': False, 'missing': str(ref_path)}
                save_json(rel, res)
                fn, rep = res['functional'], res['reproduction']
                a = (res.get('B_inner_vs_A') or res['B_vs_A'])['agreement_ratio']
                log(f"r_{arm}_seed{seed}: inner clean {fn['B_inner_clean']:.3f} | head A clean {fn['A_clean']:.3f} | agreement with A {a:.3f} | "
                    f"reproduces: {rep.get('pass')} (loss {rep.get('max_loss_difference', float('nan')):.1e}, ratio {rep.get('max_ratio_difference', float('nan')):.1e}) | {res['seconds']:.0f} s")
            for arm in NEW_ARMS:
                tag = f'{arm}_seed{seed}'
                train_n15(store, meta, arm, seed, tag, train, probs, log_u)
                rel = f'results/m_{tag}.json'
                if store.L(rel).exists():
                    continue
                m = n13.load_model(store, tag, seed)
                res = {'tag': tag, 'seed': seed, 'arm': arm, 'model': tag, **n12.measure(m, arm, mx, mys, u, log_u, probs), 'environment': study.env_record()}
                save_json(rel, res)
                fn = res['functional']
                a = (res.get('B_inner_vs_A') or res['B_vs_A'])['agreement_ratio']
                log(f"m_{tag}: inner clean {fn['B_inner_clean']:.3f} | head A clean {fn['A_clean']:.3f} | agreement with A {a:.3f} | rotated {res['rotated_control']['agreement_ratio']:.3f} | {res['seconds']:.0f} s")
        log(f'N15 {mode} complete: all runs and measurements finished')
    finally:
        dest = out / 'results'
        for f in (store.local / 'results').glob('*.json'):
            (dest / f.name).write_bytes(f.read_bytes())
        log(f'copied {len(list(dest.glob("*.json")))} result files for upload')
        log('waiting for background Drive saves to finish (up to 15 minutes)')
        store.flush()
        failures = store.failures + data_store.failures + sum((s.failures for s in srcs.values()), [])
        (out / 'drive_sync.json').write_text(json.dumps({'failures': failures}, indent=2))
        log('Drive saves finished' if not failures else f'Drive problems this session: {failures}')


if __name__ == '__main__':
    main()
