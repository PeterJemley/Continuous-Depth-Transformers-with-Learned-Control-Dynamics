"""G1c: does a trained control signal set how strongly the dynamics contract?

Implements G1C_PREREGISTRATION.md in this folder. Training is G1's, unchanged (g1_regimes.train_g1),
on new seeds 3101-3105: Model A with exact gradients, u = (+1, 0, 0, 0) for the next token and
(-1, 0, 0, 0) for the token after next, plus shuffled-control models. Each trained model, and each
main seed's untrained initial weights, is then measured on validation windows 257-512: the
contraction rate of the ODE block's vector field (minus its divergence, by Hutchinson's estimator)
at u = +1, -1 and 0, with the secondary measures. G1's geometry measures are repeated on the same
windows (S5).

    python g1c_contraction.py --out OUT --work /content/drive/MyDrive/cdt-colab/g1c_contraction
    STUDY_TEST=1 python g1c_contraction.py --out OUT --work W --source LOCAL_STUDY_FOLDER   # small test
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from torch.func import jvp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import g1_regimes as g  # noqa: E402  (imports the study: model config, test mode, device)

study, DEV, TEST, log = g.study, g.DEV, g.TEST, g.log

SEEDS = [3101, 3102] if TEST else [3101, 3102, 3103, 3104, 3105]
g.SEEDS = SEEDS                                  # G1's batch check reads its first seed from here
WIN_FROM, WIN_TO = (16, 32) if TEST else (256, 512)   # windows 257-512, numbered from 1
BATCH_W = 8 if TEST else 32
TIMES = torch.tensor([0.0, 0.25, 0.5, 0.75, 1.0])     # depths for the divergence, as in G1
PROBES = 8
TOL = g.GEOM_TOL
CONTROLS = g.CONTROLS                            # plus, minus, zero
OTHER = {'plus': 'minus', 'minus': 'plus'}


# ------------------------------------------------------------------ the estimator

def divergence(func, t, h, probes):
    """Hutchinson estimates at (t, h), per window: the trace of df/dh and its squared Frobenius norm,
    each divided by the number of tokens. One forward-mode Jacobian-vector product per probe, over
    the whole sequence state."""
    t = torch.as_tensor(t, dtype=h.dtype, device=h.device)
    T = h.shape[1]
    tr, fr = [], []
    for z in probes:
        _, jz = jvp(lambda hh: func(t, hh), (h,), (z,))
        z, jz = z.double(), jz.double()
        tr.append((z * jz).flatten(1).sum(1) / T)
        fr.append((jz * jz).flatten(1).sum(1) / T)
    return torch.stack(tr).mean(0), torch.stack(fr).mean(0)


def rademacher(shape, gen):
    return (torch.randint(2, shape, generator=gen).float() * 2 - 1).to(DEV)


@torch.no_grad()
def measure_contraction(model, xs, probe_seed):
    """Per window and depth: contraction c = -divergence per token at u = +1, -1, 0 along each control
    value's own trajectory (primary); with a second probe set (must-match B); squared Frobenius norm
    (S2); and each trained control value's field at the other value's trajectory states (S3)."""
    model.eval()
    func = model.ode_block.func
    gen1 = torch.Generator().manual_seed(probe_seed)
    gen2 = torch.Generator().manual_seed(probe_seed + 1000)
    rec = {u: {'c': [], 'frob2': [], 'c_probe2': [], 'c_at_other_states': [], 'nfe': []} for u in CONTROLS}
    for i in range(0, len(xs), BATCH_W):
        x = xs[i:i + BATCH_W].to(DEV)
        B, T = x.shape
        h0 = model.tok_emb(x) + model.pos_emb(torch.arange(T, device=DEV))
        for blk in model.early_blocks:
            h0 = blk(h0)
        p1 = [rademacher(h0.shape, gen1) for _ in range(PROBES)]
        p2 = [rademacher(h0.shape, gen2) for _ in range(PROBES)]
        traj = {}
        for u, val in CONTROLS.items():
            func.control = g.control(val, B)
            counter = g.Counting(func)
            traj[u] = g.trajectory(counter, h0, TIMES, tol=TOL)        # (5, B, T, D)
            rec[u]['nfe'].append(counter.n)
            c, f, c2 = [], [], []
            for k, t in enumerate(TIMES.tolist()):
                tr, fr = divergence(func, t, traj[u][k], p1)
                c.append(-tr); f.append(fr)
                if u in OTHER:
                    c2.append(-divergence(func, t, traj[u][k], p2)[0])
            rec[u]['c'] += torch.stack(c, 1).tolist()
            rec[u]['frob2'] += torch.stack(f, 1).tolist()
            if u in OTHER:
                rec[u]['c_probe2'] += torch.stack(c2, 1).tolist()
        for u, other in OTHER.items():
            func.control = g.control(CONTROLS[u], B)
            c = [-divergence(func, t, traj[other][k], p1)[0] for k, t in enumerate(TIMES.tolist())]
            rec[u]['c_at_other_states'] += torch.stack(c, 1).tolist()
        func.control = None
    return rec


def divergence_check():
    """Precondition 2: linear fields f(h) = A h with known trace, through the same code."""
    torch.manual_seed(0)
    B, T, D = 32, 8, 16
    gen = torch.Generator().manual_seed(1)
    d = torch.randn(D, generator=gen)
    A = torch.eye(D) + torch.randn(D, D, generator=gen) / 4
    res = {}
    # Diagonal: Rademacher probes make the estimate exact.
    h = torch.randn(B, T, D, generator=gen).to(DEV)
    tr, _ = divergence(lambda t, hh: hh * d.to(DEV) + 0 * t, 0.0, h, [rademacher(h.shape, gen) for _ in range(2)])
    res['diagonal'] = {'estimate': tr.mean().item(), 'truth': d.sum().item(),
                       'worst_relative_error': ((tr - d.sum().double().to(DEV)).abs() / d.sum().abs()).max().item()}
    # Dense: averaged over 256 windows and 8 probes.
    ests, frs = [], []
    for _ in range(256 // B):
        h = torch.randn(B, T, D, generator=gen).to(DEV)
        tr, fr = divergence(lambda t, hh: hh @ A.T.to(DEV) + 0 * t, 0.0, h, [rademacher(h.shape, gen) for _ in range(PROBES)])
        ests.append(tr); frs.append(fr)
    est, fr = torch.cat(ests).mean().item(), torch.cat(frs).mean().item()
    res['dense'] = {'estimate': est, 'truth': torch.trace(A).item(),
                    'relative_error': abs(est - torch.trace(A).item()) / abs(torch.trace(A).item()),
                    'frobenius2_estimate': fr, 'frobenius2_truth': (A * A).sum().item()}
    res['pass'] = res['diagonal']['worst_relative_error'] <= 1e-5 and res['dense']['relative_error'] <= 0.02
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
    store = study.Store(work, local)              # checkpoints and results for G1c
    out = Path(args.out)
    (out / 'results').mkdir(parents=True, exist_ok=True)
    log('device:', DEV, '| test mode:', TEST, '| study folder (data):', source, '| G1c folder:', work, '| local:', local)
    store.pull_dir('results')
    try:
        meta = study.prepare_data(data_store, None)
        train = study.load_tokens(store, meta, 'train')
        summary_path = store.L('results/g1c_summary.json')
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        if 'checks' not in summary:
            summary['checks'] = {'divergence': divergence_check(), 'curvature': g.analytic_check(),
                                 'batch': g.batch_check(train),
                                 'same_model': [g.same_model_check(train, s) for s in SEEDS]}
            summary['windows'] = [WIN_FROM + 1, WIN_TO]
            summary_path.write_text(json.dumps(summary, indent=2))
            store.push('results/g1c_summary.json')
        c = summary['checks']
        log('divergence check:', c['divergence']['pass'], f"diagonal error {c['divergence']['diagonal']['worst_relative_error']:.1e}, "
            f"dense error {c['divergence']['dense']['relative_error']:.2%}")
        log('curvature check:', c['curvature']['pass'])
        log('batch check, targets at i+1 and i+2:', c['batch']['targets_are_i+1_and_i+2'])
        log('same-model check:', [x['pass'] for x in c['same_model']])
        if not (c['batch']['targets_are_i+1_and_i+2'] and all(x['pass'] for x in c['same_model'])):
            raise SystemExit('A check of the training setup failed before training. Stopping.')
        if not (c['divergence']['pass'] and c['curvature']['pass']):
            log('An analytic check failed: G1c will be untestable, as preregistered. Training and measurement '
                'continue so that the cause can be examined.')

        xs = g.g1_windows(study.load_tokens(store, meta, 'validation'), limit=WIN_TO)[0][WIN_FROM:WIN_TO]
        log(f'measuring on validation windows {WIN_FROM + 1}-{WIN_TO} ({len(xs)} windows)')

        def save(tag, rec):
            store.L(f'results/{tag}.json').write_text(json.dumps(rec))
            store.push(f'results/{tag}.json')

        # Must-fail B: the main seeds' untrained initial weights.
        for seed in SEEDS:
            tag = f'contr_init_seed{seed}'
            if not store.L(f'results/{tag}.json').exists():
                t0 = time.time()
                rec = measure_contraction(g.new_model(seed), xs, seed + 500)
                save(tag, {'tag': tag, 'seed': seed, 'variant': 'init', 'windows': [WIN_FROM + 1, WIN_TO],
                           'times': TIMES.tolist(), 'probes': PROBES, 'd_model': study.CFG['d_model'],
                           'seconds': time.time() - t0, 'controls': rec, 'environment': study.env_record()})
                log(f"{tag}: contraction u=+1 {np.mean(rec['plus']['c']):.4f}, u=-1 {np.mean(rec['minus']['c']):.4f}")

        for seed in SEEDS:
            for variant in g.VARIANTS:
                g.train_g1(store, meta, variant, seed)
            for variant in g.VARIANTS:
                tag = f'contr_{variant}_seed{seed}'
                gtag = f'geom_{variant}_seed{seed}'
                if store.L(f'results/{tag}.json').exists() and store.L(f'results/{gtag}.json').exists():
                    log(f'{tag}: already measured, skipping')
                    continue
                if not store.pull(f'final/train_{variant}_seed{seed}.pt'):
                    raise SystemExit(f'final weights for {variant} seed {seed} are missing locally and on Drive.')
                model = g.new_model(seed)
                model.load_state_dict(torch.load(store.L(f'final/train_{variant}_seed{seed}.pt'), map_location=DEV))
                alpha = model.ode_block.func.output_scale.item()
                t0 = time.time()
                rec = measure_contraction(model, xs, seed + 500)
                save(tag, {'tag': tag, 'seed': seed, 'variant': variant, 'windows': [WIN_FROM + 1, WIN_TO],
                           'times': TIMES.tolist(), 'probes': PROBES, 'd_model': study.CFG['d_model'], 'alpha': alpha,
                           'seconds': time.time() - t0, 'controls': rec, 'environment': study.env_record()})
                t1 = time.time()
                geo = g.measure_geometry(model, xs, seed)
                save(gtag, {'tag': gtag, 'variant': variant, 'seed': seed, 'windows': [WIN_FROM + 1, WIN_TO],
                            'depths': g.DEPTHS.tolist(), 'div_depths': [g.DEPTHS[k].item() for k in g.DIV_IDX],
                            'tolerance': g.GEOM_TOL, 'nfe_tolerance': g.NFE_TOL, 'seconds': time.time() - t1,
                            'alpha': alpha, 'controls': geo, 'environment': study.env_record()})
                cp, cm = np.mean(rec['plus']['c']), np.mean(rec['minus']['c'])
                log(f'{tag}: measured in {time.time() - t0:.0f}s; contraction u=+1 {cp:.4f}, u=-1 {cm:.4f}, '
                    f"u=0 {np.mean(rec['zero']['c']):.4f}")
        log('G1c complete: all runs and measurements finished')
    finally:
        dest = out / 'results'
        for f in (store.local / 'results').glob('*.json'):
            if f.name.startswith(('train_', 'geom_', 'contr_', 'g1c_summary')):
                (dest / f.name).write_bytes(f.read_bytes())
        log(f'copied {len(list(dest.glob("*.json")))} result files for upload')
        log('waiting for background Drive saves to finish (up to 15 minutes)')
        store.flush()
        failures = store.failures + data_store.failures
        (out / 'drive_sync.json').write_text(json.dumps({'failures': failures}, indent=2))
        log('Drive saves finished' if not failures else f'Drive problems this session: {failures}')


if __name__ == '__main__':
    main()
