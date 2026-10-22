"""N14: a dose of random targets at a fixed number of informative ones.

Implements N14_PREREGISTRATION.md in this folder. N12's model, training, batches, windows and measures (n12_bottleneck),
N13's resumable loop with the discount's statistics (n13_signal), on N12's seeds. Positions in a batch are kept (a clean
target), random (a target drawn from u) or masked (no target); n80's per-step draw fixes K (kept, 20%) and R (replaced,
80%) and the random words at R, and every arm uses nested subsets of K and R:
  r10, r20, r40   clean targets at K, random targets at a nested 10/20/40% of positions drawn from R; mixture head at
                  the arm's true noise rate rho = d / (0.2 + d)
  sub10, sub05, sub025, sub0125   clean targets at a nested 10/5/2.5/1.25% of positions drawn from K, nothing elsewhere; single softmax
  sub20-rw        clean targets at K, each weighted by stopgrad(r(y)) at rho = 0.8; no random targets
  sub20, n80-mix  N12's saved models, the two ends of the dose, re-measured (precondition 2)
Every loss is the sum over the targeted positions divided by the number of all positions (N12's convention).

    python n14_dose.py --out OUT --work /content/drive/MyDrive/cdt-colab/n14_dose
    python n14_dose.py --calibration --out OUT --work /content/drive/MyDrive/cdt-colab/n14_calib
    STUDY_TEST=1 python n14_dose.py --out OUT --work W --source LOCAL_STUDY --n12 N12_FOLDER --n12-records N12_RESULTS
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
import n13_signal as n13  # noqa: E402  (its patches of n12's loss and output form are replaced below)

n10, n9, n8, n6, n4, g, study, DEV, TEST, log = n12.n10, n12.n9, n12.n8, n12.n6, n12.n4, n12.g, n12.study, n12.DEV, n12.TEST, n12.log
AMP, VOCAB, RHO, BASE, TOTAL, PEAK_LR = n12.AMP, n12.VOCAB, n12.RHO, n12.BASE, n12.TOTAL, n12.PEAK_LR
MODES = n12.MODES                                    # the same seeds and windows as N12, by design
FRAC_K, FRAC_R = 0.2, 0.8                            # n80's kept and replaced fractions, which K and R are drawn at
SPECS = {                                            # dose: random fraction of positions; frac_keep: clean fraction
    'r10': {'dose': 0.1, 'frac_keep': 0.2, 'head': 'mix'},
    'r20': {'dose': 0.2, 'frac_keep': 0.2, 'head': 'mix'},
    'r40': {'dose': 0.4, 'frac_keep': 0.2, 'head': 'mix'},
    'sub10': {'dose': 0.0, 'frac_keep': 0.1, 'head': 'single'},
    'sub05': {'dose': 0.0, 'frac_keep': 0.05, 'head': 'single'},
    'sub025': {'dose': 0.0, 'frac_keep': 0.025, 'head': 'single'},
    'sub0125': {'dose': 0.0, 'frac_keep': 0.0125, 'head': 'single'},        # amendment A1: a fifth rung
    'sub20-rw': {'dose': 0.0, 'frac_keep': 0.2, 'head': 'rw'},
}
ENDS = {'sub20': {'dose': 0.0, 'frac_keep': 0.2, 'head': 'single'}, 'n80-mix': {'dose': 0.8, 'frac_keep': 0.2, 'head': 'mix'}}
for _a, _s in list(SPECS.items()) + list(ENDS.items()):
    _s['rho'] = _s['dose'] / (_s['frac_keep'] + _s['dose']) if _s['dose'] > 0 else 0.0       # the true noise rate among targets
    _s['kind'] = f"mix:{_s['rho']!r}" if _s['head'] == 'mix' else 'sub'                        # the measurement form (see b_noisy)
N12_ARMS = ('sub20', 'n80-mix')
NEW_ARMS = tuple(SPECS)
n12.ARMS.update({a: (RHO, s['kind']) for a, s in SPECS.items()})   # n12.measure reads (rho, kind) here; rho is recorded only
LOSS_TOL = 1e-6


# ------------------------------------------------------------------ position sets, losses and output forms

def step_targets(ya, probs, seed, step, spec):
    """Head B's targets and the kept and random positions at this step. n80's draw (N10's corrupt, seeded by seed and step)
    gives R, K and the random words; a second seeded uniform draw orders positions, so that every arm's kept set is a
    nested subset of K and its random set a nested subset of R."""
    yb80, mask80 = n10.corrupt(ya, probs, torch.Generator().manual_seed(seed * 100003 + step), RHO)
    v = torch.rand(ya.shape, generator=torch.Generator().manual_seed(seed * 7919 + step * 31 + 17))
    keep = (~mask80) & (v < spec['frac_keep'] / FRAC_K)
    rand = mask80 & (v < spec['dose'] / FRAC_R)
    return torch.where(rand, yb80, ya), keep, rand, mask80


def discount(lbf, y, log_u, rho):
    """r(y) = (1 - rho) p(y) / ((1 - rho) p(y) + rho u(y)), the posterior under the head's own prediction that y is clean."""
    lpy = torch.log_softmax(lbf, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    lq = torch.logaddexp(lpy + math.log(1 - rho), log_u[y] + (math.log(rho) if rho > 0 else -math.inf))
    return (lpy + math.log(1 - rho) - lq).exp()


def head_b_loss(lb, spec, yb, keep, rand, log_u):
    """The sum over the targeted positions of the arm's per-position loss, divided by the number of all positions."""
    lbf, y = lb.reshape(-1, VOCAB).float(), yb.reshape(-1)
    k, r = keep.reshape(-1), rand.reshape(-1)
    if spec['head'] == 'mix':
        nll = -n9.copy_logits(lbf, log_u, spec['rho']).gather(-1, y.unsqueeze(-1)).squeeze(-1)
        return (nll * (k | r).float()).sum() / nll.numel()
    ce = F.cross_entropy(lbf, y, reduction='none')
    if spec['head'] == 'rw':
        with torch.no_grad():
            w = discount(lbf, y, log_u, RHO)
        return (w * ce * k.float()).sum() / ce.numel()
    return (ce * k.float()).sum() / ce.numel()


_n12_b_noisy = n13._n12_b_noisy


def b_noisy(lb, kind, log_u):
    """The arm's noisy-target prediction: 'mix:rho' is the mixture (1 - rho) softmax + rho u; 'sub' and 'single' as in N12."""
    if kind.startswith('mix:'):
        return n9.copy_logits(lb, log_u, float(kind[4:]))
    return _n12_b_noisy(lb, kind, log_u)


n12.b_noisy = b_noisy                                 # n12's measurement, validation and checks call this by name
n12.head_b_loss = n13._n12_head_b_loss               # N12's own losses for its constructed check and the loss identity check


def loss_identity_check(model, x, ya, probs, seed, log_u):
    """The dose's two ends, computed by this file's loss, equal N12's losses for n80-mix and sub20 on the same batch.
    x is on the device; ya is the CPU batch of clean targets, as step_targets takes it."""
    model.eval()
    with torch.no_grad():
        la, lb = n4.forward_n4(model, x)
        out = {}
        for arm, n12_kind in (('n80-mix', 'mix'), ('sub20', 'sub')):
            yb, keep, rand, mask80 = step_targets(ya, probs, seed, 0, ENDS[arm])
            mine = head_b_loss(lb, ENDS[arm], yb.to(DEV), keep.to(DEV), rand.to(DEV), log_u).item()
            theirs = n12.head_b_loss(lb, n12_kind, yb.to(DEV), keep.to(DEV), log_u).item()
            out[arm] = {'this_file': mine, 'n12': theirs, 'relative': abs(mine - theirs) / abs(theirs)}
    model.train()
    out['pass'] = all(out[a]['relative'] <= LOSS_TOL for a in ENDS)
    return out


def nesting_check(ya, probs, seed):
    """At step 0: every arm's kept set is a subset of K, its random set a subset of R, the dose sets and the ladder sets are
    nested, the fractions are the design's, and the ends reproduce n80's sets."""
    sets = {a: step_targets(ya, probs, seed, 0, s)[1:] for a, s in list(SPECS.items()) + list(ENDS.items())}
    K, R = ~sets['sub20'][2], sets['sub20'][2]
    sub = lambda a, b: bool((a & ~b).sum() == 0)
    out = {'seed': seed,
           'kept_subset_of_K': {a: sub(k, K) for a, (k, r, _) in sets.items()},
           'random_subset_of_R': {a: sub(r, R) for a, (k, r, _) in sets.items()},
           'dose_nested': sub(sets['r10'][1], sets['r20'][1]) and sub(sets['r20'][1], sets['r40'][1]) and sub(sets['r40'][1], sets['n80-mix'][1]),
           'ladder_nested': all(sub(sets[a][0], sets[b][0]) for a, b in (('sub0125', 'sub025'), ('sub025', 'sub05'), ('sub05', 'sub10'), ('sub10', 'sub20'))),
           'ends_equal_n80': bool(torch.equal(sets['sub20'][0], K) and torch.equal(sets['n80-mix'][1], R)),
           'fractions': {a: [k.float().mean().item(), r.float().mean().item()] for a, (k, r, _) in sets.items()},
           'fingerprints': {a: [n12.fingerprint(k), n12.fingerprint(r)] for a, (k, r, _) in sets.items()}}
    specs = {**SPECS, **ENDS}
    out['fractions_ok'] = all(abs(out['fractions'][a][0] - specs[a]['frac_keep']) <= 0.02 and abs(out['fractions'][a][1] - specs[a]['dose']) <= 0.02 for a in specs)
    out['pass'] = (all(out['kept_subset_of_K'].values()) and all(out['random_subset_of_R'].values()) and out['dose_nested']
                   and out['ladder_nested'] and out['ends_equal_n80'] and out['fractions_ok'])
    return out


# ------------------------------------------------------------------ training: N13's loop with the arm's position sets

def train_n14(store, meta, arm, seed, tag, train, probs, log_u):
    """One run, resumable. n13.train_n13 with the arm's targets, kept and random sets, and its loss; the discount's statistics
    every 50 steps at the arm's rho (0.8 where the arm has no random targets), without gradient."""
    spec = SPECS[arm]
    rho_stat = spec['rho'] if spec['head'] == 'mix' else RHO
    rho_val = rho_stat
    res_path = store.L(f'results/{tag}.json')
    if res_path.exists():
        log(f'{tag}: already finished, skipping')
        return json.loads(res_path.read_text())
    ck_rel = f'ckpt/{tag}.pt'
    store.pull(ck_rel)
    ck_path = store.L(ck_rel)
    vx, vys = n6.windows(study.load_tokens(store, meta, 'validation'), limit=g.VAL_LIMIT)
    vya, vyb = vys[0], n10.fixed_noisy(vys[0], probs, rho_val, n12.VAL_NOISE_SEED)
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
        log(f"{tag}: starting, dose {spec['dose']}, clean {spec['frac_keep']}, head B {spec['head']} (rho {spec['rho']:.3f}), seed {seed}")

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
        yb, keep, rand, mask80 = step_targets(ya, probs, seed, step, spec)
        if step == 0:
            K = ~mask80
            state.update({'batch_fingerprint_step0': n12.fingerprint(x), 'replaced_positions_fingerprint_step0': n12.fingerprint(mask80),
                          'noise_fraction_step0': mask80.float().mean().item(),
                          'keep_fingerprint_step0': n12.fingerprint(keep), 'random_fingerprint_step0': n12.fingerprint(rand),
                          'kept_fraction_step0': keep.float().mean().item(), 'random_fraction_step0': rand.float().mean().item(),
                          'keep_subset_of_K_step0': bool((keep & ~K).sum() == 0), 'random_subset_of_R_step0': bool((rand & ~mask80).sum() == 0)})
        x, ya, yb, keep, rand = x.to(DEV), ya.to(DEV), yb.to(DEV), keep.to(DEV), rand.to(DEV)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=AMP):
            la, lb = n4.forward_n4(model, x)
            loss = 0.5 * (F.cross_entropy(la.view(-1, VOCAB), ya.reshape(-1)) + head_b_loss(lb, spec, yb, keep, rand, log_u))
        if step % 50 == 0:                                                      # the discount's statistics, no gradient
            with torch.no_grad():
                r = discount(lb.detach().reshape(-1, VOCAB).float(), yb.reshape(-1), log_u, rho_stat)
                k, rr = keep.reshape(-1), rand.reshape(-1)
                sk, sr = r[k].sum().item(), (r[rr].sum().item() if rr.any() else 0.0)
                state['r'].append([step, r[k].mean().item(), r[rr].mean().item() if rr.any() else None,
                                   sr / (sr + sk) if rr.any() else None, r[k | rr].mean().item()])
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
            l_a, l_b = n12.val_losses(model, vx, vya, vyb, spec['kind'], log_u)[:2]
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
    l_a, l_b, l_b_clean, l_inner = n12.val_losses(model, fx, fys[0], n10.fixed_noisy(fys[0], probs, rho_val, n12.VAL_NOISE_SEED), spec['kind'], log_u)
    result = {'tag': tag, 'arm': arm, 'seed': seed, 'dose': spec['dose'], 'frac_keep': spec['frac_keep'], 'rho': spec['rho'], 'head_b': spec['head'],
              'rho_of_discount_statistics': rho_stat, 'readout_rank': n4.RANK,
              'final_loss_A': l_a, 'final_loss_B': l_b, 'final_loss_B_clean_targets': l_b_clean, 'final_loss_B_inner_clean': l_inner,
              'final_val_windows': len(fx), 'val_curve': state['val'], 'train_loss_every_50': state['train_loss'],
              'r_every_50': state['r'],              # [step, mean r on kept, mean r on random (None if none), random share of the summed weight, mean r on targeted]
              'alpha_every_50': state['alpha'], 'alpha_final': model.ode_block.func.output_scale.item(),
              **{k: state.get(k) for k in ('batch_fingerprint_step0', 'replaced_positions_fingerprint_step0', 'noise_fraction_step0',
                                            'keep_fingerprint_step0', 'random_fingerprint_step0', 'kept_fraction_step0', 'random_fraction_step0',
                                            'keep_subset_of_K_step0', 'random_subset_of_R_step0')},
              'max_offset': state['max_offset'], 'training_tokens': len(train),
              'skipped_steps_fp16': state['skipped_steps'], 'train_seconds': state['seconds'], 'resumes': state['resumes'],
              'n_params': sum(p.numel() for p in model.parameters()), 'environment': study.env_record()}
    res_path.write_text(json.dumps(result, indent=2))
    store.push(f'results/{tag}.json')
    log(f'{tag}: finished in {state["seconds"] / 3600:.2f} h; head A {l_a:.4f}, head B {l_b:.4f} (clean targets {l_b_clean:.4f}, inner {l_inner:.4f})')
    return result


# ------------------------------------------------------------------ measurement at the arm's own rho

@torch.no_grad()
def constructed_check_at(model, mx, u, y_clean, log_u, rho):
    """A mixture head at rho given head A's readout must reproduce the copy at rho exactly, through this file's loss and form."""
    import copy as _copy
    m = _copy.deepcopy(model)
    m.proj_b.weight.copy_(m.proj_a.weight); m.head2.weight.copy_(m.head.weight)
    with torch.enable_grad():
        G = n9.head_covs(m, mx, u, {'B': lambda la, lb: n9.copy_logits(lb, log_u, rho), 'copy': lambda la, lb: n9.copy_logits(la, log_u, rho)})
    m.eval()
    x, y = mx[:n12.MEASURE_BATCH].to(DEV), y_clean[:n12.MEASURE_BATCH].to(DEV)
    la, lb = n4.forward_n4(m, x)
    spec = {'head': 'mix', 'rho': rho}
    ones = torch.ones(y.shape, dtype=torch.bool, device=DEV)
    loss_mix = head_b_loss(lb, spec, y, ones, torch.zeros_like(ones), log_u).item()
    loss_copy = F.nll_loss(n9.copy_logits(la, log_u, rho).reshape(-1, VOCAB), y.reshape(-1)).item()
    ratio = n9.pair(G, 'B', 'copy')['agreement_ratio']
    return {'rho': rho, 'agreement_ratio': ratio, 'loss_mixture_head': loss_mix, 'loss_copy': loss_copy,
            'loss_difference': loss_mix - loss_copy, 'pass': ratio >= 0.999 and abs(loss_mix - loss_copy) <= 1e-4}


def at_rho(model, spec, mx, mys, u, log_u, probs):
    """The arm's output form against the copy at the arm's own rho (0 where it has no random targets): agreement, the
    functional losses on clean and on rho-noisy targets, and for the mixture arms the constructed check at rho."""
    rho = spec['rho'] if spec['head'] == 'mix' else 0.0
    kind = f'mix:{rho!r}' if spec['head'] == 'mix' else 'single'
    G = n9.head_covs(model, mx, u, {'B': lambda la, lb: b_noisy(lb, kind, log_u), 'copy': lambda la, lb: n9.copy_logits(la, log_u, rho)})
    y_noisy = n10.fixed_noisy(mys[0], probs, rho, n12.MEAS_NOISE_SEED)
    model.eval()
    keys = ('B_clean', 'copy_clean', 'B_noisy', 'copy_noisy')
    tot, cnt = dict.fromkeys(keys, 0.0), 0
    with torch.no_grad():
        for i in range(0, len(mx), n12.MEASURE_BATCH):
            x, yc, yn = mx[i:i + n12.MEASURE_BATCH].to(DEV), mys[0][i:i + n12.MEASURE_BATCH].to(DEV), y_noisy[i:i + n12.MEASURE_BATCH].to(DEV)
            la, lb = n4.forward_n4(model, x)
            lpb, lpc = b_noisy(lb, kind, log_u), n9.copy_logits(la, log_u, rho)
            for k, lp, y in (('B_clean', lpb, yc), ('copy_clean', lpc, yc), ('B_noisy', lpb, yn), ('copy_noisy', lpc, yn)):
                tot[k] -= lp.gather(-1, y.unsqueeze(-1)).sum().item()
            cnt += yc.numel()
    out = {'rho': rho, 'B_vs_copy': n9.pair(G, 'B', 'copy'), 'functional': {k: v / cnt for k, v in tot.items()}}
    out['constructed'] = constructed_check_at(model, mx, u, mys[0], log_u, rho) if spec['head'] == 'mix' else None
    return out


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
    src = study.Store(n12_folder, local.parent / f'_n14_src_{n12_folder.name}')
    out = Path(args.out)
    (out / 'results').mkdir(parents=True, exist_ok=True)
    log('device:', DEV, '| test mode:', TEST, '| mode:', mode, '| study folder (data):', source, '| N14 folder:', work, '| local:', local)
    log("N12's models from:", src.drive / 'final', "| N12's records from:", n12_records)
    store.pull_dir('results')
    try:
        meta = study.prepare_data(data_store, None)
        full = study.load_tokens(store, meta, 'train')
        cut = int(len(full) * n12.CUT_FRACTION)
        train, tail = full[:cut], full[cut:]
        probs = n12.n7.unigram(train)
        log_u = probs.float().log().to(DEV)
        summary_path = store.L('results/n14_summary.json')
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

        def save_json(rel, obj):
            store.L(rel).write_text(json.dumps(obj, indent=2 if rel.endswith('n14_summary.json') else None))
            store.push(rel)

        allx, allys = n6.windows(tail, limit=w1)
        mx, mys = allx[w0:w1], [y[w0:w1] for y in allys]
        u = n8.uniforms(len(mx))
        log(f'training region: tokens 0-{cut} of {len(full)}; measured on the held-out tail\'s windows {w0 + 1}-{w0 + len(mx)}')

        if 'checks' not in summary:
            x, ys, _ = n12.draw_batch(train, torch.Generator().manual_seed(seeds[0]))
            summary['checks'] = {
                'batch': n6.batch_check(train), 'batch_g1': g.batch_check(train),
                'same_model': [g.same_model_check(train, s) for s in seeds],
                'windows_shifted': all(torch.equal(mys[k - 1][:, :-k], mx[:, k:]) for k in range(1, 9)),
                'analytic': n9.analytic_check(),
                'identical_heads': n8.identical_heads_check(seeds[0], mx, u),
                'loss_identity': loss_identity_check(n4.new_n4(seeds[0], BASE), x.to(DEV), ys[0], probs, seeds[0], log_u),
                'nesting': [nesting_check(n12.draw_batch(train, torch.Generator().manual_seed(s))[1][0], probs, s) for s in seeds],
                'unigram': {'sum': probs.sum().item(), 'zeros': int((probs == 0).sum()), 'from_tokens': cut}}
            summary['mode'] = mode
            summary['held_out'] = {'cut_fraction': n12.CUT_FRACTION, 'cut_token': cut, 'training_tokens': len(full), 'tail_tokens': len(tail),
                                   'tail_windows': (len(tail) - n12.K_MAX) // n12.SEQ, 'measure': [w0 + 1, w0 + len(mx)]}
            summary['arms'] = {a: {k: v for k, v in s.items() if k != 'kind'} for a, s in SPECS.items()}
            summary['ends'] = {a: {k: v for k, v in s.items() if k != 'kind'} for a, s in ENDS.items()}
            summary['n12'] = {'folder': str(n12_folder), 'records': str(n12_records), 'arms': list(N12_ARMS)}
            summary['seeds'], summary['readout_rank'], summary['subspace_rank'] = seeds, n4.RANK, n9.R
            save_json('results/n14_summary.json', summary)
        c = summary['checks']
        li = c['loss_identity']
        log('batch check:', c['batch']['targets_are_i+1_to_i+8'], '| windows shifted:', c['windows_shifted'],
            '| analytic:', round(c['analytic']['sampled']['top_subspace_overlap'], 4), '| identical heads:', c['identical_heads']['agreement_ratio'],
            '| loss identity at the ends:', f"{li['n80-mix']['relative']:.1e}", f"{li['sub20']['relative']:.1e}", li['pass'],
            '| nesting:', [n['pass'] for n in c['nesting']])
        log('same-model check:', [x['pass'] for x in c['same_model']])
        if not (c['batch']['targets_are_i+1_to_i+8'] and c['windows_shifted'] and c['analytic']['pass'] and c['identical_heads']['pass']
                and li['pass'] and all(n['pass'] for n in c['nesting']) and all(x['pass'] for x in c['same_model']) and abs(c['unigram']['sum'] - 1) < 1e-6):
            raise SystemExit('A check of the setup failed before training. Stopping.')

        for seed in seeds:
            for arm in N12_ARMS:                                          # the two ends, re-measured (precondition 2)
                tag = f'{arm}_seed{seed}'
                rel = f'results/r_{arm}_seed{seed}.json'
                if store.L(rel).exists():
                    continue
                m = n13.load_model(src, tag, seed)
                res = {'tag': tag, 'seed': seed, 'arm': arm, 'model': tag, **n12.measure(m, arm, mx, mys, u, log_u, probs),
                       'at_rho': at_rho(m, ENDS[arm], mx, mys, u, log_u, probs), 'environment': study.env_record()}
                ref_path = n12_records / f'm_{tag}.json'
                res['reproduction'] = n13.reproduction(res, json.loads(ref_path.read_text())) if ref_path.exists() else {'pass': False, 'missing': str(ref_path)}
                save_json(rel, res)
                fn, rep = res['functional'], res['reproduction']
                a = (res.get('B_inner_vs_A') or res['B_vs_A'])['agreement_ratio']
                log(f"r_{tag}: inner clean {fn['B_inner_clean']:.3f} | head A clean {fn['A_clean']:.3f} | agreement with A {a:.3f} | "
                    f"reproduces N12: {rep.get('pass')} (loss {rep.get('max_loss_difference', float('nan')):.1e}, ratio {rep.get('max_ratio_difference', float('nan')):.1e}) | {res['seconds']:.0f} s")
            for arm in NEW_ARMS:
                tag = f'{arm}_seed{seed}'
                train_n14(store, meta, arm, seed, tag, train, probs, log_u)
                rel = f'results/m_{tag}.json'
                if store.L(rel).exists():
                    continue
                m = n13.load_model(store, tag, seed)
                res = {'tag': tag, 'seed': seed, 'arm': arm, 'model': tag, **n12.measure(m, arm, mx, mys, u, log_u, probs),
                       'at_rho': at_rho(m, SPECS[arm], mx, mys, u, log_u, probs), 'environment': study.env_record()}
                save_json(rel, res)
                fn = res['functional']
                a = (res.get('B_inner_vs_A') or res['B_vs_A'])['agreement_ratio']
                log(f"m_{tag}: inner clean {fn['B_inner_clean']:.3f} | head A clean {fn['A_clean']:.3f} | agreement with A {a:.3f} | "
                    f"rotated {res['rotated_control']['agreement_ratio']:.3f} | at rho {res['at_rho']['rho']:.3f}: B vs copy {res['at_rho']['B_vs_copy']['agreement_ratio']:.3f} | {res['seconds']:.0f} s")
        log(f'N14 {mode} complete: all runs and measurements finished')
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
