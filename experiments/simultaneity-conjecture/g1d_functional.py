"""G1d: does the model need the farther reading's strong contraction?

Implements G1D_PREREGISTRATION.md in this folder. Three arms share G1c's training (g1_regimes.train_g1)
on seeds 3201-3205 and differ only in a penalty added to the loss:
  control   no penalty;
  capped    lambda * mean over u = -1 sequences and the four Euler states of max(0, c - 2.0)^2,
            where c is minus the per-token divergence of the vector field, by one Hutchinson probe;
  sham      the same computation with a cap of 1e6, which never binds.
lambda is chosen first by a pilot on seed 100: the smallest of 0.01, 0.1, 1, 10 at which the capped
model's measured contraction under u = -1 is at most 3.0. Every model is then measured as in G1c on
validation windows 513-768, and both tasks' losses are recorded on the whole validation split.

    python g1d_functional.py --out OUT --work /content/drive/MyDrive/cdt-colab/g1d_functional
    STUDY_TEST=1 python g1d_functional.py --out OUT --work W --source LOCAL_STUDY_FOLDER   # small test
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import g1_regimes as g  # noqa: E402  (imports the study: model config, test mode, device)
import g1c_contraction as gc  # noqa: E402

study, DEV, TEST, log = g.study, g.DEV, g.TEST, g.log

SEEDS = [3201, 3202] if TEST else [3201, 3202, 3203, 3204, 3205]
g.SEEDS = SEEDS                                  # G1's batch check reads its first seed from here
PILOT_SEED = 100
LAMBDAS = [0.01, 0.1, 1.0, 10.0]
CAP, SHAM_CAP = 2.0, 1e6
PILOT_MAX = 3.0                                   # the pilot rule, and the manipulation check
ARMS = ['control', 'capped', 'sham']
WIN_FROM, WIN_TO = (32, 48) if TEST else (512, 768)   # windows 513-768, numbered from 1
gc.BATCH_W = 8 if TEST else 32


# ------------------------------------------------------------------ the training penalty

def train_contraction(func, t, h, z):
    """Contraction per token, c = -z^T (df/dh) z / T, for each sequence, by one vector-Jacobian product
    kept in the graph so that the penalty can be trained through. fp32, outside autocast."""
    with torch.autocast(DEV.type, enabled=False):
        h = h.float()
        v = func(t, h)
        (zj,) = torch.autograd.grad(v, h, grad_outputs=z, create_graph=True)
        return -(zj * z).flatten(1).sum(1) / h.shape[1]


def make_penalty(lam, cap, probe_seed):
    """The capped (or sham) penalty: lambda * mean over u = -1 sequences and Euler states 0-3 of
    max(0, c - cap)^2. Probes come from their own generator, seeded per step, so they never touch the
    global random state and a resumed run draws the same probes."""
    def penalty(model, u_sign, step):
        blk, func = model.ode_block, model.ode_block.func
        idx = (u_sign == -1).nonzero().squeeze(1).to(DEV)
        if len(idx) == 0:
            return None
        gen = torch.Generator(device=DEV).manual_seed(probe_seed * 100003 + step)
        saved = func.control
        func.control = g.control(-1.0, len(idx))
        terms = []
        for k in range(4):
            h = blk.states[k][idx]
            z = torch.randint(2, h.shape, generator=gen, device=DEV).float() * 2 - 1
            c = train_contraction(func, blk.integration_times[k], h, z)
            terms.append(F.relu(c - cap) ** 2)
        func.control = saved
        p = torch.stack(terms).mean()
        return lam * p, p.item()
    return penalty


def penalty_check():
    """Precondition 6: on a linear field f(h) = A h, the training estimator averages to -trace(A) per
    token, and its gradient with respect to A averages to -I."""
    gen = torch.Generator().manual_seed(2)
    D, T, B = 16, 8, 65536                        # enough probes that sampling error in the gradient is about 0.5%
    A = (torch.eye(D) + torch.randn(D, D, generator=gen) / 4).to(DEV).requires_grad_()
    h = torch.randn(B, T, D, generator=gen).to(DEV).requires_grad_()
    z = (torch.randint(2, (B, T, D), generator=gen).float() * 2 - 1).to(DEV)
    c = train_contraction(lambda t, hh: hh @ A.T + 0 * t, torch.tensor(0.0, device=DEV), h, z)
    est = c.mean()
    (gA,) = torch.autograd.grad(est, A)
    tr = torch.trace(A).item()
    eye = torch.eye(D, device=DEV)
    res = {'estimate': est.item(), 'truth': -tr, 'relative_error': abs(est.item() + tr) / abs(tr),
           'gradient_relative_error': ((gA + eye).norm() / eye.norm()).item()}
    res['pass'] = res['relative_error'] <= 0.02 and res['gradient_relative_error'] <= 0.02
    return res


def sham_check(train):
    """Sanity check: the sham penalty is exactly zero with zero gradient on one batch; control and sham
    start from identical weights and draw identical first batches."""
    seed = SEEDS[0]
    m = g.new_model(seed)
    m.ode_block.keep_states = True
    m.train()
    x, y1, y2, task = g.draw_batch(train, torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed * 10 + 1))
    m(x.to(DEV), control=g.control(task))
    term, value = make_penalty(1.0, SHAM_CAP, seed)(m, task, 0)
    grads = torch.autograd.grad(term, [p for p in m.parameters()], allow_unused=True)
    zero = all(gr is None or torch.count_nonzero(gr).item() == 0 for gr in grads)
    w = [torch.cat([p.detach().reshape(-1) for p in g.new_model(seed).parameters()]) for _ in range(2)]
    b = [g.draw_batch(train, torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed * 10 + 1)) for _ in range(2)]
    same_batch = all(torch.equal(p, q) for p, q in zip(b[0], b[1]))
    res = {'sham_value': value, 'sham_gradient_all_zero': zero, 'same_initial_weights': torch.equal(w[0], w[1]),
           'same_first_batch': same_batch}
    res['pass'] = value == 0.0 and zero and res['same_initial_weights'] and same_batch
    return res


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--work', required=True, help="this job's Drive folder, for checkpoints and results")
    ap.add_argument('--source', default=None, help='the WikiText-103 study folder (default: sibling wt103_study)')
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
    data_store = study.Store(source, local)       # reads token files only; never writes to the study folder
    store = study.Store(work, local)              # checkpoints and results for G1d
    out = Path(args.out)
    (out / 'results').mkdir(parents=True, exist_ok=True)
    log('device:', DEV, '| test mode:', TEST, '| study folder (data):', source, '| G1d folder:', work, '| local:', local)
    store.pull_dir('results')
    try:
        meta = study.prepare_data(data_store, None)
        train = study.load_tokens(store, meta, 'train')
        summary_path = store.L('results/g1d_summary.json')
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

        def save_summary():
            summary_path.write_text(json.dumps(summary, indent=2))
            store.push('results/g1d_summary.json')

        if 'checks' not in summary:
            summary['checks'] = {'penalty': penalty_check(), 'divergence': gc.divergence_check(),
                                 'sham': sham_check(train), 'batch': g.batch_check(train),
                                 'same_model': [g.same_model_check(train, s) for s in SEEDS]}
            summary['windows'] = [WIN_FROM + 1, WIN_TO]
            save_summary()
        c = summary['checks']
        log('penalty check:', c['penalty']['pass'], f"estimate error {c['penalty']['relative_error']:.2%}, "
            f"gradient error {c['penalty']['gradient_relative_error']:.2%}")
        log('divergence check:', c['divergence']['pass'])
        log('sham check:', c['sham']['pass'], {k: v for k, v in c['sham'].items() if k != 'pass'})
        log('batch check, targets at i+1 and i+2:', c['batch']['targets_are_i+1_and_i+2'])
        log('same-model check:', [x['pass'] for x in c['same_model']])
        if not (c['sham']['pass'] and c['batch']['targets_are_i+1_and_i+2'] and all(x['pass'] for x in c['same_model'])):
            raise SystemExit('A check of the training setup failed before training. Stopping.')
        if not (c['penalty']['pass'] and c['divergence']['pass']):
            log('An analytic check failed: G1d will be untestable, as preregistered. The runs continue so that '
                'the cause can be examined.')

        xs = g.g1_windows(study.load_tokens(store, meta, 'validation'), limit=WIN_TO)[0][WIN_FROM:WIN_TO]
        log(f'measuring on validation windows {WIN_FROM + 1}-{WIN_TO} ({len(xs)} windows)')

        def run(arm, seed, lam):
            pen = None
            if arm != 'control':
                pen = make_penalty(lam, SHAM_CAP if arm == 'sham' else CAP, seed)
            orig = g.new_model

            def new_model_keeping_states(s, exact=True):
                m = orig(s, exact)
                m.ode_block.keep_states = pen is not None
                return m
            g.new_model = new_model_keeping_states
            try:
                return g.train_g1(store, meta, arm, seed, truthful=True, penalty=pen)
            finally:
                g.new_model = orig

        def measure(arm, seed, geometry=True):
            tag = f'{arm}_seed{seed}'
            ctag, gtag = f'contr_{tag}', f'geom_{tag}'
            done = store.L(f'results/{ctag}.json').exists() and (not geometry or store.L(f'results/{gtag}.json').exists())
            if done:
                return json.loads(store.L(f'results/{ctag}.json').read_text())
            if not store.pull(f'final/train_{tag}.pt'):
                raise SystemExit(f'final weights for {tag} are missing locally and on Drive.')
            model = g.new_model(seed)
            model.load_state_dict(torch.load(store.L(f'final/train_{tag}.pt'), map_location=DEV))
            alpha = model.ode_block.func.output_scale.item()
            t0 = time.time()
            rec = gc.measure_contraction(model, xs, seed + 500)
            out_rec = {'tag': ctag, 'arm': arm, 'seed': seed, 'windows': [WIN_FROM + 1, WIN_TO], 'times': gc.TIMES.tolist(),
                       'probes': gc.PROBES, 'd_model': study.CFG['d_model'], 'alpha': alpha, 'seconds': time.time() - t0,
                       'controls': rec, 'environment': study.env_record()}
            store.L(f'results/{ctag}.json').write_text(json.dumps(out_rec))
            store.push(f'results/{ctag}.json')
            if geometry:
                t1 = time.time()
                geo = g.measure_geometry(model, xs, seed)
                store.L(f'results/{gtag}.json').write_text(json.dumps(
                    {'tag': gtag, 'arm': arm, 'seed': seed, 'windows': [WIN_FROM + 1, WIN_TO], 'depths': g.DEPTHS.tolist(),
                     'alpha': alpha, 'seconds': time.time() - t1, 'controls': geo, 'environment': study.env_record()}))
                store.push(f'results/{gtag}.json')
            log(f"{ctag}: contraction u=+1 {np.mean(rec['plus']['c']):.4f}, u=-1 {np.mean(rec['minus']['c']):.4f}")
            return out_rec

        # The pilot: the smallest lambda that holds the measured contraction under u = -1 at or below 3.0.
        if 'lambda' not in summary:
            pilot = summary.setdefault('pilot', {})
            chosen = None
            for lam in LAMBDAS:
                arm = f'pilot_lam{lam:g}'
                if str(lam) not in pilot:
                    run(arm, PILOT_SEED, lam)
                    r = measure(arm, PILOT_SEED, geometry=False)
                    pilot[str(lam)] = float(np.mean(r['controls']['minus']['c']))
                    save_summary()
                log(f'pilot lambda {lam:g}: measured contraction under u = -1 {pilot[str(lam)]:.4f}')
                if pilot[str(lam)] <= PILOT_MAX:
                    chosen = lam
                    break
            summary['lambda'] = chosen
            save_summary()
        lam = summary['lambda']
        if lam is None:
            raise SystemExit('No lambda in the grid held the contraction at or below 3.0. As preregistered, the main '
                             'runs are not made and G1d is untestable.')
        log(f'lambda chosen by the pilot rule: {lam:g}')

        # Main runs, seed by seed, so that complete triples accumulate if a session ends early.
        for seed in SEEDS:
            for arm in ARMS:
                run(arm, seed, lam)
                measure(arm, seed)
        log('G1d complete: all runs and measurements finished')
    finally:
        dest = out / 'results'
        for f in (store.local / 'results').glob('*.json'):
            if f.name.startswith(('train_', 'geom_', 'contr_', 'g1d_summary')):
                (dest / f.name).write_bytes(f.read_bytes())
        log(f'copied {len(list(dest.glob("*.json")))} result files for upload')
        log('waiting for background Drive saves to finish (up to 15 minutes)')
        store.flush()
        failures = store.failures + data_store.failures
        (out / 'drive_sync.json').write_text(json.dumps({'failures': failures}, indent=2))
        log('Drive saves finished' if not failures else f'Drive problems this session: {failures}')


if __name__ == '__main__':
    main()
