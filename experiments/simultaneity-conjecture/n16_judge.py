"""N16: the judge, not the share.

Implements N16_PREREGISTRATION.md in this folder. N12's model, training, batches, windows and measures (n12_bottleneck),
N13's reweighted single-softmax head and resumable loop (n13_signal), N15's weights and setup checks (n15_sharpen), on
N12's seeds at the 80% noise rate:
  rw-oracle      head B's loss is stopgrad(w(y)) * CE(l, y), w from head A's prediction (detached) at rho = 0.8, every
                 position targeted (N13's convention)
  sub20-oracle   clean targets at n80's kept positions only, each weighted by head A's judgement at rho = 0.8 (N14's convention
                 for clean-target-only arms: the sum over kept positions divided by all positions)
  rw-b2w, rw-b4w exploratory: head B's own discount for the first 300 steps, then the likelihood ratio raised to beta
  rw-b1 (N13's n80-rw), sub20 and clean (N12), sub20-rw, sub10, sub05, sub025, sub0125 (N14)   saved models, re-measured
Every model is also measured for neglect: on the measurement windows, the fraction of clean targets whose weight under
head B's own discount, and under head A's judgement, is below 0.01, and the inner reading's loss on each ignored set.
In the calibration run rw-oracle is N15's calibration model, re-measured and checked against its record.

    python n16_judge.py --out OUT --work /content/drive/MyDrive/cdt-colab/n16_judge
    python n16_judge.py --calibration --out OUT --work /content/drive/MyDrive/cdt-colab/n16_calib
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import n12_bottleneck as n12  # noqa: E402
import n13_signal as n13  # noqa: E402
import n15_sharpen as n15  # noqa: E402  (weight, weight_identity_check, oracle_detachment_check; its patches of n12 stay in force)

n10, n9, n8, n6, n4, g, study, DEV, TEST, log = n12.n10, n12.n9, n12.n8, n12.n6, n12.n4, n12.g, n12.study, n12.DEV, n12.TEST, n12.log
AMP, VOCAB, RHO, BASE, TOTAL, PEAK_LR = n12.AMP, n12.VOCAB, n12.RHO, n12.BASE, n12.TOTAL, n12.PEAK_LR
MODES = n12.MODES
WARMUP = 300 if not TEST else 20
SPECS = {
    'rw-oracle': {'judge': 'A', 'beta': 1.0, 'clean_only': False, 'warmup': 0, 'exploratory': False},
    'sub20-oracle': {'judge': 'A', 'beta': 1.0, 'clean_only': True, 'warmup': 0, 'exploratory': False},
    'rw-b2w': {'judge': 'B', 'beta': 2.0, 'clean_only': False, 'warmup': WARMUP, 'exploratory': True},
    'rw-b4w': {'judge': 'B', 'beta': 4.0, 'clean_only': False, 'warmup': WARMUP, 'exploratory': True},
}
REFS = {'rw-b1': ('n13', 'n80-rw', 'n13'), 'sub20': ('n12', 'sub20', 'n12'), 'clean': ('n12', 'clean', 'n12'),
        'sub20-rw': ('n14', 'sub20-rw', 'n14'), 'sub10': ('n14', 'sub10', 'n14'), 'sub05': ('n14', 'sub05', 'n14'),
        'sub025': ('n14', 'sub025', 'n14'), 'sub0125': ('n14', 'sub0125', 'n14')}
CALIB_REFS = {'rw-b1': ('n13', 'n80-rw', 'n13'), 'rw-oracle': ('n15', 'rw-oracle', 'n15'), 'sub20': ('n12', 'sub20', 'n12'), 'clean': ('n12', 'clean', 'n12'),
              'sub20-rw': ('n14', 'sub20-rw', 'n14'), 'sub10': ('n14', 'sub10', 'n14'), 'sub05': ('n14', 'sub05', 'n14'), 'sub025': ('n14', 'sub025', 'n14')}
for _a, _s in SPECS.items():
    n12.ARMS[_a] = (RHO, 'sub' if _s['clean_only'] else 'rw')
for _a in ('sub20-rw', 'sub10', 'sub05', 'sub025', 'sub0125'):
    n12.ARMS.setdefault(_a, (RHO, 'sub'))
n12.ARMS.setdefault('rw-b1', (RHO, 'rw'))
n12.ARMS.setdefault('rw-oracle', (RHO, 'rw'))
IGNORED = 0.01


# ------------------------------------------------------------------ the weights, the losses and the checks

def arm_weights(la, lb, yb, log_u, spec, step):
    """The per-position weights of an arm at a step, without gradient: head A's judgement, or head B's own discount at beta = 1
    during the warm-up and at the arm's beta after."""
    beta = 1.0 if step < spec['warmup'] else spec['beta']
    src = la if spec['judge'] == 'A' else lb
    with torch.no_grad():
        return n15.weight(src.detach().reshape(-1, VOCAB).float(), yb.reshape(-1), log_u, beta)


def head_b_loss(la, lb, yb, keep, log_u, spec, step):
    w = arm_weights(la, lb, yb, log_u, spec, step)
    ce = F.cross_entropy(lb.reshape(-1, VOCAB).float(), yb.reshape(-1), reduction='none')
    if spec['clean_only']:
        return (w * ce * keep.reshape(-1).float()).sum() / ce.numel()
    return (w * ce).mean()


def warmup_switch_check(model, x, yb, log_u):
    """Precondition 1: the exploratory arms' weight equals N13's discount at the last warm-up step and the tempered weight at
    the first step after it, on a batch."""
    model.eval()
    with torch.no_grad():
        la, lb = n4.forward_n4(model, x)
        lf, y = lb.reshape(-1, VOCAB).float(), yb.reshape(-1)
        out = {}
        for arm in ('rw-b2w', 'rw-b4w'):
            spec = SPECS[arm]
            before = arm_weights(la, lb, yb, log_u, spec, spec['warmup'] - 1)
            after = arm_weights(la, lb, yb, log_u, spec, spec['warmup'])
            out[arm] = {'before_vs_discount': (before - n13.discount(lf, y, log_u)).abs().max().item(),
                        'after_vs_tempered': (after - n15.weight(lf, y, log_u, spec['beta'])).abs().max().item(),
                        'warmup_steps': spec['warmup']}
    model.train()
    out['pass'] = all(v['before_vs_discount'] <= 1e-6 and v['after_vs_tempered'] == 0.0 for k, v in out.items() if k != 'pass')
    return out


@torch.no_grad()
def neglect(model, mx, mys, log_u, batch=n12.MEASURE_BATCH):
    """On the measurement windows: the fraction of clean targets whose weight under head B's own discount, and under head A's
    judgement, is below IGNORED, and the inner reading's clean-target loss on each ignored set against the rest."""
    model.eval()
    keys = ('own', 'A')
    n_ign, loss_ign, loss_rest, n_all = {k: 0 for k in keys}, {k: 0.0 for k in keys}, {k: 0.0 for k in keys}, 0
    for i in range(0, len(mx), batch):
        x, y = mx[i:i + batch].to(DEV), mys[0][i:i + batch].to(DEV)
        la, lb = n4.forward_n4(model, x)
        yf = y.reshape(-1)
        lbf = lb.reshape(-1, VOCAB).float()
        ce = F.cross_entropy(lbf, yf, reduction='none')
        w = {'own': n15.weight(lbf, yf, log_u, 1.0), 'A': n15.weight(la.reshape(-1, VOCAB).float(), yf, log_u, 1.0)}
        for k in keys:
            ign = w[k] < IGNORED
            n_ign[k] += int(ign.sum()); loss_ign[k] += ce[ign].sum().item(); loss_rest[k] += ce[~ign].sum().item()
        n_all += yf.numel()
    out = {'threshold': IGNORED, 'n_targets': n_all}
    for k in keys:
        out[f'{k}_ignored_fraction'] = n_ign[k] / n_all
        out[f'loss_on_{k}_ignored'] = loss_ign[k] / n_ign[k] if n_ign[k] else None
        out[f'loss_on_rest_{k}'] = loss_rest[k] / (n_all - n_ign[k]) if n_all > n_ign[k] else None
    return out


# ------------------------------------------------------------------ training

def train_n16(store, meta, arm, seed, tag, train, probs, log_u):
    spec = SPECS[arm]
    kind = 'sub' if spec['clean_only'] else 'rw'
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
        label = ('head A judges' if spec['judge'] == 'A' else f"own discount, beta {spec['beta']:g} after {spec['warmup']} steps") + (', clean targets only' if spec['clean_only'] else '')
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
        yb, keep, mask = n12.step_targets(ya, probs, seed, step, RHO, kind)      # 'sub': the clean targets, kept where n80 keeps
        if step == 0:
            state.update({'batch_fingerprint_step0': n12.fingerprint(x), 'replaced_positions_fingerprint_step0': n12.fingerprint(mask),
                          'noise_fraction_step0': mask.float().mean().item(), 'kept_fraction_step0': keep.float().mean().item()})
        x, ya, yb, keep = x.to(DEV), ya.to(DEV), yb.to(DEV), keep.to(DEV)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=AMP):
            la, lb = n4.forward_n4(model, x)
            loss = 0.5 * (F.cross_entropy(la.view(-1, VOCAB), ya.reshape(-1)) + head_b_loss(la, lb, yb, keep, log_u, spec, step))
        if step % 50 == 0:                                                      # the weights' statistics, no gradient
            with torch.no_grad():
                w = arm_weights(la, lb, yb, log_u, spec, step)
                k = keep.reshape(-1)
                sk = w[k].sum().item()
                if spec['clean_only']:
                    state['w'].append([step, w[k].mean().item(), w[k].mean().item(), None, None, (w[k] < IGNORED).float().mean().item()])
                else:
                    sr = w[~k].sum().item()
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
    l_a, l_b, l_b_clean, l_inner = n12.val_losses(model, fx, fys[0], n10.fixed_noisy(fys[0], probs, RHO, n12.VAL_NOISE_SEED), kind, log_u)
    last = state['w'][-1]
    result = {'tag': tag, 'arm': arm, 'seed': seed, 'rho': RHO, **{k: spec[k] for k in ('judge', 'beta', 'clean_only', 'warmup', 'exploratory')}, 'head_b': kind,
              'readout_rank': n4.RANK,
              'final_loss_A': l_a, 'final_loss_B': l_b, 'final_loss_B_clean_targets': l_b_clean, 'final_loss_B_inner_clean': l_inner,
              'final_val_windows': len(fx), 'val_curve': state['val'], 'train_loss_every_50': state['train_loss'],
              'w_every_50': state['w'],              # [step, mean w over targeted, mean w on kept, mean w on replaced (None if none), random share (None if none), fraction of kept with w < 0.01]
              'alpha_every_50': state['alpha'], 'alpha_final': model.ode_block.func.output_scale.item(),
              **{k: state.get(k) for k in ('batch_fingerprint_step0', 'replaced_positions_fingerprint_step0', 'noise_fraction_step0', 'kept_fraction_step0')},
              'max_offset': state['max_offset'], 'training_tokens': len(train),
              'skipped_steps_fp16': state['skipped_steps'], 'train_seconds': state['seconds'], 'resumes': state['resumes'],
              'n_params': sum(p.numel() for p in model.parameters()), 'environment': study.env_record()}
    res_path.write_text(json.dumps(result, indent=2))
    store.push(f'results/{tag}.json')
    log(f'{tag}: finished in {state["seconds"] / 3600:.2f} h; head A {l_a:.4f}, head B {l_b:.4f} (clean targets {l_b_clean:.4f}, inner {l_inner:.4f}); '
        + (f'final share {last[4]:.3f}, ' if last[4] is not None else '') + f'ignored kept {last[5]:.3f}')
    return result


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--source', default=None)
    ap.add_argument('--local', default=None)
    FOLDERS = (('n12', 'n12_bottleneck', 'n12_calib'), ('n13', 'n13_signal', 'n13_calib'), ('n14', 'n14_dose', 'n14_calib'), ('n15', 'n15_sharpen', 'n15_calib'))
    for k, real, cal in FOLDERS:
        ap.add_argument(f'--{k}', default=None, help=f"{k.upper()}'s Drive folder (default: {real}, or {cal}, next to --work)")
        ap.add_argument(f'--{k}-records', default=None, help=f"{k.upper()}'s committed records (default: results/{k} or results/{k}_calib here)")
    ap.add_argument('--calibration', action='store_true', help='the calibration run: seed 3001, the tail\'s windows 513-1024')
    ap.add_argument('--drop', default='', help='comma-separated exploratory arms not to run (the calibration\'s rule)')
    args = ap.parse_args()
    mode = 'calibration' if args.calibration else 'real'
    seeds, (w0, w1) = MODES[mode]['seeds'], MODES[mode]['windows']
    n6.SEEDS = g.SEEDS = seeds
    work = Path(args.work)
    source = Path(args.source) if args.source else work.parent / 'wt103_study'
    folders, records = {}, {}
    for k, real, cal in FOLDERS:
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
    refs = CALIB_REFS if args.calibration else REFS
    srcs = {k: study.Store(folders[k], local.parent / f'_n16_src_{folders[k].name}') for k in {v[0] for v in refs.values()}}
    dropped = {a for a in args.drop.split(',') if a}
    new_arms = [a for a in SPECS if not (args.calibration and a == 'rw-oracle') and a not in dropped]
    out = Path(args.out)
    (out / 'results').mkdir(parents=True, exist_ok=True)
    log('device:', DEV, '| test mode:', TEST, '| mode:', mode, '| study folder (data):', source, '| N16 folder:', work, '| local:', local)
    log('arms trained:', new_arms, '| references:', list(refs), '| saved models from:', {k: str(folders[k]) for k in srcs})
    store.pull_dir('results')
    try:
        meta = study.prepare_data(data_store, None)
        full = study.load_tokens(store, meta, 'train')
        cut = int(len(full) * n12.CUT_FRACTION)
        train, tail = full[:cut], full[cut:]
        probs = n12.n7.unigram(train)
        log_u = probs.float().log().to(DEV)
        summary_path = store.L('results/n16_summary.json')
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

        def save_json(rel, obj):
            store.L(rel).write_text(json.dumps(obj, indent=2 if rel.endswith('n16_summary.json') else None))
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
                'weight_identity': n15.weight_identity_check(m0, x.to(DEV), yb.to(DEV), log_u),
                'oracle_detachment': n15.oracle_detachment_check(m0, x.to(DEV), ys[0].to(DEV), yb.to(DEV), log_u),
                'warmup_switch': warmup_switch_check(m0, x.to(DEV), yb.to(DEV), log_u),
                'unigram': {'sum': probs.sum().item(), 'zeros': int((probs == 0).sum()), 'from_tokens': cut}}
            summary['mode'] = mode
            summary['held_out'] = {'cut_fraction': n12.CUT_FRACTION, 'cut_token': cut, 'training_tokens': len(full), 'tail_tokens': len(tail),
                                   'tail_windows': (len(tail) - n12.K_MAX) // n12.SEQ, 'measure': [w0 + 1, w0 + len(mx)]}
            summary['arms'] = {a: SPECS[a] for a in new_arms}
            summary['dropped'] = sorted(dropped)
            summary['references'] = {a: {'folder': str(folders[v[0]]), 'tag': v[1], 'records': str(records[v[2]])} for a, v in refs.items()}
            summary['seeds'], summary['readout_rank'], summary['subspace_rank'] = seeds, n4.RANK, n9.R
            save_json('results/n16_summary.json', summary)
        c = summary['checks']
        log('batch check:', c['batch']['targets_are_i+1_to_i+8'], '| windows shifted:', c['windows_shifted'],
            '| analytic:', round(c['analytic']['sampled']['top_subspace_overlap'], 4), '| identical heads:', c['identical_heads']['agreement_ratio'],
            '| weight identity:', f"{c['weight_identity']['max_relative_difference']:.1e}", c['weight_identity']['pass'],
            '| oracle detachment:', c['oracle_detachment']['pass'], '| warm-up switch:', c['warmup_switch']['pass'])
        log('same-model check:', [x['pass'] for x in c['same_model']])
        if not (c['batch']['targets_are_i+1_to_i+8'] and c['windows_shifted'] and c['analytic']['pass'] and c['identical_heads']['pass']
                and c['weight_identity']['pass'] and c['oracle_detachment']['pass'] and c['warmup_switch']['pass']
                and all(x['pass'] for x in c['same_model']) and abs(c['unigram']['sum'] - 1) < 1e-6):
            raise SystemExit('A check of the setup failed before training. Stopping.')

        def measure_all(m, arm, tag):
            return {'tag': tag, 'seed': seed, 'arm': arm, 'model': tag, **n12.measure(m, arm, mx, mys, u, log_u, probs),
                    'neglect': neglect(m, mx, mys, log_u), 'environment': study.env_record()}

        for seed in seeds:
            for arm, (fkey, prefix, rkey) in refs.items():                   # the saved models, re-measured (precondition 2)
                rel = f'results/r_{arm}_seed{seed}.json'
                if store.L(rel).exists():
                    continue
                tag = f'{prefix}_seed{seed}'
                m = n13.load_model(srcs[fkey], tag, seed)
                res = measure_all(m, arm, tag)
                res['tag'] = f'{arm}_seed{seed}'
                ref_path = records[rkey] / f'm_{tag}.json'
                res['reproduction'] = n13.reproduction(res, json.loads(ref_path.read_text())) if ref_path.exists() else {'pass': False, 'missing': str(ref_path)}
                save_json(rel, res)
                fn, rep, ng = res['functional'], res['reproduction'], res['neglect']
                a = (res.get('B_inner_vs_A') or res['B_vs_A'])['agreement_ratio']
                log(f"r_{arm}_seed{seed}: inner clean {fn['B_inner_clean']:.3f} | head A clean {fn['A_clean']:.3f} | agreement with A {a:.3f} | "
                    f"ignored by own {ng['own_ignored_fraction']:.3f} / by A {ng['A_ignored_fraction']:.3f} | "
                    f"reproduces: {rep.get('pass')} (loss {rep.get('max_loss_difference', float('nan')):.1e}, ratio {rep.get('max_ratio_difference', float('nan')):.1e}) | {res['seconds']:.0f} s")
            for arm in new_arms:
                tag = f'{arm}_seed{seed}'
                train_n16(store, meta, arm, seed, tag, train, probs, log_u)
                rel = f'results/m_{tag}.json'
                if store.L(rel).exists():
                    continue
                m = n13.load_model(store, tag, seed)
                res = measure_all(m, arm, tag)
                save_json(rel, res)
                fn, ng = res['functional'], res['neglect']
                a = (res.get('B_inner_vs_A') or res['B_vs_A'])['agreement_ratio']
                log(f"m_{tag}: inner clean {fn['B_inner_clean']:.3f} | head A clean {fn['A_clean']:.3f} | agreement with A {a:.3f} | "
                    f"ignored by own {ng['own_ignored_fraction']:.3f} / by A {ng['A_ignored_fraction']:.3f} | rotated {res['rotated_control']['agreement_ratio']:.3f} | {res['seconds']:.0f} s")
        log(f'N16 {mode} complete: all runs and measurements finished')
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
