"""N8: a measure of what a reading uses, free of its mistakes.

Implements N8_PREREGISTRATION.md in this folder. No training. The job loads the final weights of N7's 20
models (n7_harder) and N6r's 20 models (n6r_halfoverlap) from Drive and measures each on WikiText-103
test-split windows 513-768 in two ways:
  - the target-free measure (new): the covariance, at the ODE block's output, of the gradient of
    -log p_k(y) with y drawn from head k's own predicted distribution. Its expectation is the Fisher
    information of the head's predictions with respect to the state. Each position's draw uses one
    uniform shared by the two heads, through the inverse of each head's cumulative distribution
    (common random numbers), so two heads with identical predictions sample identical tokens;
  - the old measure (N6r's, n6r_halfoverlap.measure), with the actual targets, as a loading check and
    for comparison. The noisy head's targets are corrupted once, with N7's fixed measurement seed.
Everything else is N6r's: top-16 eigenvectors, the overlap, split-half stability and the agreement ratio.

    python n8_fisher.py --out OUT --work /content/drive/MyDrive/cdt-colab/n8_fisher
    STUDY_TEST=1 python n8_fisher.py --out OUT --work W --source LOCAL_STUDY_FOLDER   # small test
"""
import argparse
import copy
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import n7_harder as n7  # noqa: E402  (imports N6r, N6, N4, N2, G1 and the study)

n6r, n6, n4, g, study, DEV, TEST, log = n7.n6r, n7.n6, n7.n4, n7.g, n7.study, n7.DEV, n7.TEST, n7.log
SEQ, VOCAB = g.SEQ, g.VOCAB
R, BASE = n6.R, n6.BASE
SETS = {'n7': {'job': 'n7_harder', 'seeds': [3801, 3802] if TEST else [3801, 3802, 3803, 3804, 3805],
               'arms': list(n7.ARMS), 'recorded': 'n7'},
        'n6r': {'job': 'n6r_halfoverlap', 'seeds': [3701, 3702] if TEST else [3701, 3702, 3703, 3704, 3705],
                'arms': list(n6.ARMS), 'recorded': 'n6r'}}
MEAS = (32, 48) if TEST else (512, 768)            # test-split windows 513-768
FISHER_SEED = 161803                                # the shared uniforms, the same for every model
LOAD_TOL = 0.05                                     # precondition 2
TOY_MIN = 0.98                                      # precondition 1
SAMPLE_CHUNK = 4                                    # windows per inverse-CDF step, to bound memory


def overlap(U, V):
    return n4.n2.overlap(U, V)


def top(G, r=R):
    evals, evecs = torch.linalg.eigh(G)
    return evecs[:, torch.argsort(evals, descending=True)[:r]].float()


# ------------------------------------------------------------------ the target-free estimator

def uniforms(n, seed=FISHER_SEED):
    """One uniform per position of each of n windows; window i always gets the same row."""
    return torch.rand(n, SEQ, generator=torch.Generator().manual_seed(seed), dtype=torch.float64)


@torch.no_grad()
def sample(logits, u):
    """For each position, the token at which the cumulative distribution of softmax(logits) first reaches u."""
    out = torch.empty(u.shape, dtype=torch.long, device=logits.device)
    for i in range(0, len(logits), SAMPLE_CHUNK):
        cdf = torch.softmax(logits[i:i + SAMPLE_CHUNK].double(), -1).cumsum(-1)
        uu = (u[i:i + SAMPLE_CHUNK].to(cdf.device) * cdf[..., -1]).unsqueeze(-1).contiguous()
        out[i:i + SAMPLE_CHUNK] = torch.searchsorted(cdf, uu).squeeze(-1).clamp_(max=cdf.shape[-1] - 1)
    return out


def fisher_covs(model, xs, ua, ub=None):
    """Each head's target-free gradient covariance at the ODE block's output, summed separately over the
    windows 0::2 and 1::2 (N6r's halves). ua and ub are the two heads' uniforms (the same, by default)."""
    ub = ua if ub is None else ub
    model.eval()
    D = study.CFG['d_model']
    G = [[torch.zeros(D, D, dtype=torch.float64, device=DEV) for _ in range(2)] for _ in range(2)]
    for i in range(0, len(xs), n4.BATCH_W):
        x = xs[i:i + n4.BATCH_W].to(DEV)
        half = torch.arange(i, i + len(x), device=DEV) % 2
        (la, lb), h_out = n4.forward_n4(model, x, keep_out=True)
        for k, (lg, u) in enumerate(((la, ua), (lb, ub))):
            y = sample(lg.detach(), u[i:i + len(x)])
            loss = F.cross_entropy(lg.float().reshape(-1, VOCAB), y.reshape(-1), reduction='sum')
            (gr,) = torch.autograd.grad(loss, h_out, retain_graph=(k == 0))
            gr = gr.double()
            for h in (0, 1):
                gh = gr[half == h].reshape(-1, D)
                G[k][h] += gh.T @ gh
    return G


def from_covs(G):
    """N6r's quantities from the four half-covariances: the full-window subspaces are those of their sums."""
    Ua, Ub = top(G[0][0] + G[0][1]), top(G[1][0] + G[1][1])
    Uao, Uae, Ubo, Ube = top(G[0][0]), top(G[0][1]), top(G[1][0]), top(G[1][1])
    stab_a, stab_b = overlap(Uao, Uae), overlap(Ubo, Ube)
    cross = 0.5 * (overlap(Uao, Ube) + overlap(Uae, Ubo))
    return {'overlap_AB': overlap(Ua, Ub), 'stability_A_odd_even': stab_a, 'stability_B_odd_even': stab_b,
            'cross_half_overlap_AB': cross, 'agreement_ratio': cross / math.sqrt(stab_a * stab_b),
            'rank': R, 'subspaces': {'A': Ua.cpu().tolist(), 'B': Ub.cpu().tolist()}}


def measure_f(model, mx, u):
    t0 = time.time()
    return {**from_covs(fisher_covs(model, mx, u)), 'seconds': time.time() - t0}


SCALARS = ('overlap_AB', 'stability_A_odd_even', 'stability_B_odd_even', 'cross_half_overlap_AB', 'agreement_ratio')


def scalars(r):
    return {k: r[k] for k in SCALARS}


# ------------------------------------------------------------------ checks

def analytic_check(n=None, vocab=2000, seed=FISHER_SEED):
    """Precondition 1a. A linear-softmax model, logits = h W^T + b, whose Fisher with respect to h is exact:
    sum over positions of W^T (diag p - p p^T) W. The sampled estimate, through the same sampler, must share
    its top-R subspace (overlap at least TOY_MIN). The bias puts most of the probability on R + 1 tokens, so
    the exact Fisher's top-R subspace is set by the predicted distribution, not by W alone, and a sampler
    that ignored p would miss it. Two wrong estimators are recorded for contrast: tokens drawn uniformly,
    and the most likely token."""
    D = study.CFG['d_model']
    n = n or (16 if TEST else 64)
    gen = torch.Generator().manual_seed(seed)
    W = (torch.randn(vocab, D, generator=gen) / math.sqrt(D)).to(DEV)
    b = torch.zeros(vocab, device=DEV)
    b[torch.randperm(vocab, generator=gen)[:R + 1].to(DEV)] = 8.0
    h = torch.randn(n, SEQ, D, generator=gen).to(DEV)
    u = uniforms(n, seed + 1)
    Wd = W.double()
    exact = torch.zeros(D, D, dtype=torch.float64, device=DEV)
    for i in range(n):
        p = torch.softmax((h[i] @ W.T + b).double(), -1)
        exact += Wd.T @ (Wd * p.sum(0).unsqueeze(1)) - (p @ Wd).T @ (p @ Wd)
    est = {}
    for name in ('sampled', 'uniform_tokens', 'argmax_tokens'):
        G = torch.zeros(D, D, dtype=torch.float64, device=DEV)
        for i in range(0, n, 8):
            hh = h[i:i + 8].clone().requires_grad_(True)
            lg = hh @ W.T + b
            if name == 'sampled':
                y = sample(lg.detach(), u[i:i + 8])
            elif name == 'uniform_tokens':
                y = (u[i:i + 8] * vocab).long().clamp_(max=vocab - 1).to(DEV)
            else:
                y = lg.detach().argmax(-1)
            loss = F.cross_entropy(lg.reshape(-1, vocab), y.reshape(-1), reduction='sum')
            (gr,) = torch.autograd.grad(loss, hh)
            gr = gr.double().reshape(-1, D)
            G += gr.T @ gr
        est[name] = {'top_subspace_overlap': overlap(top(G), top(exact)),
                     'relative_error': ((G - exact).norm() / exact.norm()).item()}
    ev = torch.linalg.eigvalsh(exact).flip(0)
    return {'positions': n * SEQ, 'vocab': vocab, 'eigengap_R_over_R+1': (ev[R - 1] / ev[R]).item(), **est,
            'pass': est['sampled']['top_subspace_overlap'] >= TOY_MIN}


def identical_heads_check(seed, mx, u):
    """Precondition 1b. Two heads with identical weights must give a target-free agreement ratio of exactly 1."""
    m = n4.new_n4(seed, BASE)
    with torch.no_grad():
        m.head2.weight.copy_(m.head.weight)
        m.proj_b.weight.copy_(m.proj_a.weight)
    r = measure_f(m, mx, u)
    return {'seed': seed, 'agreement_ratio': r['agreement_ratio'], 'overlap_AB': r['overlap_AB'],
            'pass': abs(r['agreement_ratio'] - 1) < 1e-9}


def rotated_control(model, seed=FISHER_SEED):
    """A copy of the model whose head B is head A reading a randomly rotated final state:
    logits_B = W_A P_A Q ln_f(z), Q a fixed random orthogonal matrix. It is a head known to read different
    directions of the state, with head A's own weights, so the measure must be able to call it separate."""
    m = copy.deepcopy(model)
    D = study.CFG['d_model']
    Q = torch.linalg.qr(torch.randn(D, D, generator=torch.Generator().manual_seed(seed + 2), dtype=torch.float64))[0].to(DEV)
    with torch.no_grad():
        m.proj_b = copy.deepcopy(m.proj_a)
        m.proj_b.weight.copy_((m.proj_a.weight.double() @ Q).float())
        m.head2 = copy.deepcopy(m.head)
    return m


def rowspace_overlap(Wa, Wb):
    """Overlap of the row spaces of two readout projections (as n4.readout_overlap)."""
    qa = torch.linalg.qr(Wa.detach().double().T)[0]
    qb = torch.linalg.qr(Wb.detach().double().T)[0]
    return ((qa.T @ qb).pow(2).sum() / qa.shape[1]).item()


def readout_drift(model, seed):
    """Exploratory: how far each trained readout projection has moved from the one both started as."""
    p0 = n4.new_n4(seed, BASE).proj_a.weight
    return {'A_vs_initial': rowspace_overlap(model.proj_a.weight, p0), 'B_vs_initial': rowspace_overlap(model.proj_b.weight, p0),
            'A_vs_B': rowspace_overlap(model.proj_a.weight, model.proj_b.weight)}


def old_targets(set_name, arm, mys, probs):
    if set_name == 'n7':
        return n7.fixed_targets(arm, mys, probs, n7.MEAS_NOISE_SEED)
    return n6.pair(arm, mys)


def recorded_old(src, set_name, tag):
    """The old measure's overlap recorded for this model by N6r or N7: Drive first, the repository's copy after."""
    rel = f'results/sub_{tag}.json'
    if src.pull(rel):
        return json.loads(src.L(rel).read_text())['overlap_AB'], 'drive'
    repo = HERE / 'results' / SETS[set_name]['recorded'] / f'sub_{tag}.json'
    if repo.exists() and not TEST:
        return json.loads(repo.read_text())['overlap_AB'], 'repository'
    return None, 'missing'


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
    srcs = {s: study.Store(work.parent / SETS[s]['job'], local.parent / f'_n8_src_{SETS[s]["job"]}') for s in SETS}
    out = Path(args.out)
    (out / 'results').mkdir(parents=True, exist_ok=True)
    log('device:', DEV, '| test mode:', TEST, '| study folder (data):', source, '| N8 folder:', work, '| local:', local)
    log('weights from:', ', '.join(str(srcs[s].drive / 'final') for s in SETS))
    store.pull_dir('results')
    try:
        meta = study.prepare_data(data_store, None)
        probs = n7.unigram(study.load_tokens(store, meta, 'train'))
        summary_path = store.L('results/n8_summary.json')
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

        def save_json(rel, obj):
            store.L(rel).write_text(json.dumps(obj, indent=2 if rel.endswith('n8_summary.json') else None))
            store.push(rel)

        allx, allys = n6.windows(study.load_tokens(store, meta, 'test'), limit=MEAS[1])
        mx, mys = allx[MEAS[0]:MEAS[1]], [y[MEAS[0]:MEAS[1]] for y in allys]
        u = uniforms(len(mx))
        log(f'measured on test-split windows {MEAS[0] + 1}-{MEAS[0] + len(mx)}')

        if 'checks' not in summary:
            summary['checks'] = {'analytic': analytic_check(),
                                 'identical_heads': identical_heads_check(SETS['n7']['seeds'][0], mx, u),
                                 'test_windows_shifted': all(torch.equal(mys[k - 1][:, :-k], mx[:, k:]) for k in range(1, 9))}
            summary['windows'] = {'split': 'test', 'measure': [MEAS[0] + 1, MEAS[0] + len(mx)]}
            summary['fisher_seed'], summary['readout_rank'], summary['subspace_rank'] = FISHER_SEED, n4.RANK, R
            summary['sets'] = SETS
            save_json('results/n8_summary.json', summary)
        c = summary['checks']
        log('analytic check, top-subspace overlap:', round(c['analytic']['sampled']['top_subspace_overlap'], 4),
            '| identical heads, agreement ratio:', c['identical_heads']['agreement_ratio'], '| windows shifted:', c['test_windows_shifted'])
        if not (c['analytic']['pass'] and c['identical_heads']['pass'] and c['test_windows_shifted']):
            raise SystemExit('A check of the estimator or the windows failed before any model was measured. Stopping.')

        if 'untrained' not in summary:
            summary['untrained'] = {f'seed{s}': {**scalars(measure_f(n4.new_n4(s, BASE), mx, u)),
                                                 'rotated_control': scalars(measure_f(rotated_control(n4.new_n4(s, BASE)), mx, u))}
                                    for st in SETS for s in SETS[st]['seeds']}
            save_json('results/n8_summary.json', summary)

        for set_name, st in SETS.items():
            src = srcs[set_name]
            for seed in st['seeds']:
                for arm in st['arms']:
                    tag = f'{arm}_seed{seed}'
                    rel = f'results/f_{set_name}_{tag}.json'
                    if store.L(rel).exists():
                        continue
                    if not src.pull(f'final/{tag}.pt'):
                        raise SystemExit(f'weights {src.drive / "final" / (tag + ".pt")} are missing on Drive.')
                    m = n4.new_n4(seed, BASE)
                    m.load_state_dict(torch.load(src.L(f'final/{tag}.pt'), map_location=DEV, weights_only=False))
                    fr = measure_f(m, mx, u)
                    old = n6r.measure(m, mx, *old_targets(set_name, arm, mys, probs))
                    rec, rec_from = recorded_old(src, set_name, tag)
                    rot = scalars(measure_f(rotated_control(m), mx, u)) if arm == 'same' else None
                    res = {'tag': tag, 'set': set_name, 'seed': seed, 'arm': arm, 'fisher': fr, 'rotated_control': rot,
                           'readout_drift': readout_drift(m, seed),
                           'old': {**scalars(old), 'subspaces': old['subspaces']},
                           'loading': {'recorded_overlap_AB': rec, 'recorded_from': rec_from,
                                       'difference': None if rec is None else old['overlap_AB'] - rec,
                                       'pass': rec is not None and abs(old['overlap_AB'] - rec) <= LOAD_TOL}}
                    save_json(rel, res)
                    rot_note = '' if rot is None else f" | rotated control ratio {rot['agreement_ratio']:.3f}"
                    log(f"f_{set_name}_{tag}: target-free overlap {fr['overlap_AB']:.3f}, ratio {fr['agreement_ratio']:.3f}, "
                        f"stability A {fr['stability_A_odd_even']:.3f}, B {fr['stability_B_odd_even']:.3f} | "
                        f"old overlap {old['overlap_AB']:.3f} (recorded {rec if rec is None else round(rec, 3)})"
                        f"{rot_note} | {fr['seconds']:.0f} s")
        log('N8 complete: all measurements finished')
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
