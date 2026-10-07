"""N1: is the reading a direction, not a regime?

Implements N1_PREREGISTRATION.md in this folder. Four arms share G1c's training (g1_regimes.train_g1)
on seeds 3301-3305, with identical modules and initial weights; they differ only in where the
control's vector, control_embed(u), enters the ODE block:
  full   inside the vector field at every evaluation (as in G1-G1d);
  post   added once to the block's output; the field never sees u;
  pre    added once to the block's input; the field never sees u;
  none   nowhere.
Each model's losses on both tasks are recorded on the whole validation split. The full models (and
their untrained initial weights) are then measured on validation windows 769-1024: how much of the
control's effect on the block's output is one constant vector (N1a), with the secondary measures.

    python n1_offset.py --out OUT --work /content/drive/MyDrive/cdt-colab/n1_offset
    STUDY_TEST=1 python n1_offset.py --out OUT --work W --source LOCAL_STUDY_FOLDER   # small test
"""
import argparse
import json
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import g1_regimes as g  # noqa: E402  (imports the study: model config, test mode, device)
import g1c_contraction as gc  # noqa: E402

study, DEV, TEST, log = g.study, g.DEV, g.TEST, g.log
odeint_exact = g.odeint_exact

SEEDS = [3301, 3302] if TEST else [3301, 3302, 3303, 3304, 3305]
g.SEEDS = SEEDS                                  # G1's batch check reads its first seed from here
ARMS = ['full', 'post', 'pre', 'none']
WIN_FROM, WIN_TO = (48, 64) if TEST else (768, 1024)  # windows 769-1024, numbered from 1
BATCH_W = 8 if TEST else 32
gc.BATCH_W = BATCH_W


# ------------------------------------------------------------------ the four arms

def _integrate(block, x):
    return odeint_exact(block.func, x, block.integration_times, method='euler')[-1]


def _post_forward(self, x, control=None):
    self.func.control = None
    out = _integrate(self, x)
    return out if control is None else out + self.func.control_embed(control).unsqueeze(1)


def _pre_forward(self, x, control=None):
    self.func.control = None
    if control is not None:
        x = x + self.func.control_embed(control).unsqueeze(1)
    return _integrate(self, x)


def _none_forward(self, x, control=None):
    self.func.control = None
    return _integrate(self, x)


FORWARDS = {'full': g._exact_ode_forward, 'post': _post_forward, 'pre': _pre_forward, 'none': _none_forward}


def make_arm(model, arm):
    model.ode_block.forward = types.MethodType(FORWARDS[arm], model.ode_block)
    model.arm = arm
    return model


def arm_model(seed, arm):
    return make_arm(g.new_model(seed), arm)


# ------------------------------------------------------------------ checks

@torch.no_grad()
def must_match_check(train, seed):
    """Precondition 1: with control_embed zeroed, all four arms give bit-identical logits on one batch."""
    x, _, _, task = g.draw_batch(train, torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed * 10 + 1))
    x = x.to(DEV)
    logits = {}
    for arm in ARMS:
        m = arm_model(seed, arm)
        m.eval()
        m.ode_block.func.control_embed.weight.zero_()
        m.ode_block.func.control_embed.bias.zero_()
        logits[arm] = m(x, control=g.control(task))
    same = all(torch.equal(logits['full'], logits[a]) for a in ARMS)
    # And without zeroing, the arms really differ (the construction is not vacuous).
    differ = {}
    for arm in ('post', 'pre', 'none'):
        a, b = arm_model(seed, 'full').eval(), arm_model(seed, arm).eval()
        differ[arm] = not torch.equal(a(x, control=g.control(task)), b(x, control=g.control(task)))
    return {'seed': seed, 'identical_with_control_zeroed': same, 'differ_otherwise': differ,
            'pass': same and all(differ.values())}


# ------------------------------------------------------------------ measurements

@torch.no_grad()
def block_io(model, xs):
    """Per window batch: the block's input h0, and its output under u = +1 and u = -1."""
    model.eval()
    for i in range(0, len(xs), BATCH_W):
        x = xs[i:i + BATCH_W].to(DEV)
        B, T = x.shape
        h = model.tok_emb(x) + model.pos_emb(torch.arange(T, device=DEV))
        for blk in model.early_blocks:
            h = blk(h)
        yield h, {v: model.ode_block(h, g.control(v, B)) for v in (1.0, -1.0)}


def offset_measures(model, xs):
    """N1a and S2: the control's effect on the block's output, Delta = out(u=-1) - out(u=+1), per token."""
    D, inj, norms = [], [], []
    for h, out in block_io(model, xs):
        D.append((out[-1.0] - out[1.0]).reshape(-1, h.shape[-1]).double().cpu())
        y = out[1.0]
        inj.append((0.9 * y - 1.1 * y).reshape(-1, h.shape[-1]).double().cpu())   # an injected multiplicative effect
        norms.append(y.reshape(-1, h.shape[-1]).double().norm(dim=1).cpu())
    D, inj, norms = torch.cat(D), torch.cat(inj), torch.cat(norms)

    def fraction(M):
        mean = M.mean(0)
        return (len(M) * mean.dot(mean) / (M * M).sum()).item()

    def top_share(M):
        s = torch.linalg.svdvals(M)
        return (s[0] ** 2 / (s ** 2).sum()).item()
    return {'offset_fraction': fraction(D), 'top_singular_share': top_share(D),
            'injected_multiplicative_fraction': fraction(inj),
            'delta_norm_mean': D.norm(dim=1).mean().item(), 'output_norm_mean': norms.mean().item(),
            'relative_size': (D.norm(dim=1).mean() / norms.mean()).item(), 'tokens': len(D)}


@torch.no_grad()
def contraction_pre(model, xs, probe_seed):
    """S3 for the pre arm: the field never sees u, so each control value's trajectory starts from
    h0 + control_embed(u) and the field runs with no control. Same estimator and depths as G1c."""
    model.eval()
    func = model.ode_block.func
    gen = torch.Generator().manual_seed(probe_seed)
    rec = {u: {'c': []} for u in ('plus', 'minus')}
    for h0, _ in block_io(model, xs):
        B = h0.shape[0]
        probes = [gc.rademacher(h0.shape, gen) for _ in range(gc.PROBES)]
        func.control = None
        for u, val in (('plus', 1.0), ('minus', -1.0)):
            start = h0 + func.control_embed(g.control(val, B)).unsqueeze(1)
            traj = g.trajectory(func, start, gc.TIMES, tol=gc.TOL)
            c = [-gc.divergence(func, t, traj[k], probes)[0] for k, t in enumerate(gc.TIMES.tolist())]
            rec[u]['c'] += torch.stack(c, 1).tolist()
    return rec


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
    store = study.Store(work, local)              # checkpoints and results for N1
    out = Path(args.out)
    (out / 'results').mkdir(parents=True, exist_ok=True)
    log('device:', DEV, '| test mode:', TEST, '| study folder (data):', source, '| N1 folder:', work, '| local:', local)
    store.pull_dir('results')
    try:
        meta = study.prepare_data(data_store, None)
        train = study.load_tokens(store, meta, 'train')
        summary_path = store.L('results/n1_summary.json')
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

        def save_json(rel, obj):
            store.L(rel).write_text(json.dumps(obj, indent=None if rel != 'results/n1_summary.json' else 2))
            store.push(rel)

        if 'checks' not in summary:
            summary['checks'] = {'must_match': [must_match_check(train, s) for s in SEEDS],
                                 'divergence': gc.divergence_check(), 'batch': g.batch_check(train),
                                 'same_model': [g.same_model_check(train, s) for s in SEEDS]}
            summary['windows'] = [WIN_FROM + 1, WIN_TO]
            save_json('results/n1_summary.json', summary)
        c = summary['checks']
        log('must-match check (control zeroed, all arms identical):', [x['pass'] for x in c['must_match']])
        log('divergence check:', c['divergence']['pass'])
        log('batch check, targets at i+1 and i+2:', c['batch']['targets_are_i+1_and_i+2'])
        log('same-model check:', [x['pass'] for x in c['same_model']])
        if not (c['batch']['targets_are_i+1_and_i+2'] and all(x['pass'] for x in c['same_model'])):
            raise SystemExit('A check of the training setup failed before training. Stopping.')
        if not all(x['pass'] for x in c['must_match']):
            log('The must-match check failed: N1 will be untestable, as preregistered. The runs continue so that the '
                'cause can be examined.')

        xs = g.g1_windows(study.load_tokens(store, meta, 'validation'), limit=WIN_TO)[0][WIN_FROM:WIN_TO]
        log(f'measuring on validation windows {WIN_FROM + 1}-{WIN_TO} ({len(xs)} windows)')

        # Reference: the control's effect at the full arm's untrained initial weights.
        for seed in SEEDS:
            rel = f'results/offset_init_seed{seed}.json'
            if not store.L(rel).exists():
                r = offset_measures(arm_model(seed, 'full'), xs)
                save_json(rel, {'seed': seed, 'arm': 'full', 'state': 'init', **r})
                log(f"offset_init_seed{seed}: offset fraction {r['offset_fraction']:.4f}")

        def run(arm, seed):
            orig = g.new_model

            def new_arm_model(s, exact=True):
                return make_arm(orig(s, exact), arm)
            g.new_model = new_arm_model
            try:
                return g.train_g1(store, meta, arm, seed, truthful=True)
            finally:
                g.new_model = orig

        def trained(arm, seed):
            if not store.pull(f'final/train_{arm}_seed{seed}.pt'):
                raise SystemExit(f'final weights for {arm} seed {seed} are missing locally and on Drive.')
            m = arm_model(seed, arm)
            m.load_state_dict(torch.load(store.L(f'final/train_{arm}_seed{seed}.pt'), map_location=DEV))
            return m

        for seed in SEEDS:
            for arm in ARMS:
                run(arm, seed)
            rel = f'results/offset_full_seed{seed}.json'
            if not store.L(rel).exists():
                t0 = time.time()
                r = offset_measures(trained('full', seed), xs)
                save_json(rel, {'seed': seed, 'arm': 'full', 'state': 'trained', 'seconds': time.time() - t0, **r})
                log(f"offset_full_seed{seed}: offset fraction {r['offset_fraction']:.4f}, top singular share "
                    f"{r['top_singular_share']:.4f}, injected check {r['injected_multiplicative_fraction']:.4f}")
            for arm in ('full', 'pre'):
                rel = f'results/contr_{arm}_seed{seed}.json'
                if not store.L(rel).exists():
                    m = trained(arm, seed)
                    t0 = time.time()
                    rec = gc.measure_contraction(m, xs, seed + 500) if arm == 'full' else contraction_pre(m, xs, seed + 500)
                    save_json(rel, {'seed': seed, 'arm': arm, 'windows': [WIN_FROM + 1, WIN_TO], 'times': gc.TIMES.tolist(),
                                    'seconds': time.time() - t0, 'controls': rec})
                    log(f"contr_{arm}_seed{seed}: contraction u=+1 {np.mean(rec['plus']['c']):.4f}, "
                        f"u=-1 {np.mean(rec['minus']['c']):.4f}")
        log('N1 complete: all runs and measurements finished')
    finally:
        dest = out / 'results'
        for f in (store.local / 'results').glob('*.json'):
            if f.name.startswith(('train_', 'offset_', 'contr_', 'n1_summary')):
                (dest / f.name).write_bytes(f.read_bytes())
        log(f'copied {len(list(dest.glob("*.json")))} result files for upload')
        log('waiting for background Drive saves to finish (up to 15 minutes)')
        store.flush()
        failures = store.failures + data_store.failures
        (out / 'drive_sync.json').write_text(json.dumps({'failures': failures}, indent=2))
        log('Drive saves finished' if not failures else f'Drive problems this session: {failures}')


if __name__ == '__main__':
    main()
