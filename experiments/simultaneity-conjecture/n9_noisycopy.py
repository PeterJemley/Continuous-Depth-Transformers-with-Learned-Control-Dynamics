"""N9: can the target-free measure recognise an ideal noisy copy of a reading?

Implements N9_PREREGISTRATION.md in this folder. No training. The job loads N7's final weights
(n7_harder) from Drive and, for each model, builds from head A its ideal noisy copy: the head whose
prediction is exactly q = (1 - rho) p_A + rho u, with u the training data's unigram distribution, computed in
log space as logaddexp(log(1 - rho) + log p_A, log rho + log u). Every head is then measured with N8's
target-free measure (n8_fisher), on WikiText-103 test-split windows 769-1024:
  N9a  head A against its copy at rho = 0.8;
  N9b  the trained noisy head B against that copy;
  S1   head A against its copies at rho = 0, 0.2, 0.5, 0.95 (rho = 0 is the must-match control);
  S2   prediction-space comparisons of the trained noisy head with the copies;
  S3   head A against the trained noisy head (N8's comparison, on new windows);
  S4   head A against its copy at rho = 0.8 (and 0) in N7's other 15 models.
The rotated-readout control is N8's (n8_fisher.rotated_control).

    python n9_noisycopy.py --out OUT --work /content/drive/MyDrive/cdt-colab/n9_noisycopy
    STUDY_TEST=1 python n9_noisycopy.py --out OUT --work W --source LOCAL_STUDY_FOLDER   # small test
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import n8_fisher as n8  # noqa: E402  (imports N7, N6r, N6, N4, N2, G1 and the study)

n7, n6, n4, g, study, DEV, TEST, log = n8.n7, n8.n6, n8.n4, n8.g, n8.study, n8.DEV, n8.TEST, n8.log
SEQ, VOCAB, R = g.SEQ, g.VOCAB, n8.R
SEEDS = [3801, 3802] if TEST else [3801, 3802, 3803, 3804, 3805]
ARMS = list(n7.ARMS)                                 # same, k4, k8, noisy
RHO = n7.NOISE                                       # 0.8, the rate of N7's noise
RHOS = [0.0, 0.2, 0.5, RHO, 0.95]
MEAS = (48, 64) if TEST else (768, 1024)            # test-split windows 769-1024
BATCH = 4 if TEST else 16                            # windows per forward pass; seven heads share each pass
MATCH_MIN = 0.999                                    # precondition 2
LOAD_TOL = n8.LOAD_TOL                               # precondition 4
S2_BIAS, S2_MASS = 7.5, 0.9                          # the analytic check's toy (see analytic_check)


def copy_name(rho):
    return f'copy{rho:g}'


def copy_logits(la, log_u, rho):
    """Log of the ideal noisy copy's prediction, (1 - rho) softmax(la) + rho u, computed in log space."""
    a = torch.log_softmax(la.float(), -1) + (math.log(1 - rho) if rho < 1 else -math.inf)
    b = log_u + (math.log(rho) if rho > 0 else -math.inf)
    return torch.logaddexp(a, b.expand_as(a))


def head_fns(log_u, rhos, with_b=True):
    """The heads measured on one model, each a function of the two trained heads' logits."""
    fns = {'A': lambda la, lb: la.float()}
    if with_b:
        fns['B'] = lambda la, lb: lb.float()
    for rho in rhos:
        fns[copy_name(rho)] = (lambda r: lambda la, lb: copy_logits(la, log_u, r))(rho)
    return fns


def head_covs(model, xs, u, fns):
    """N8's target-free covariances, for any heads computed from one forward pass: for each head, the
    gradient covariance of -log prediction at the ODE block's output, with targets drawn from the head's own
    prediction (shared uniforms), summed separately over the windows 0::2 and 1::2."""
    model.eval()
    D = study.CFG['d_model']
    names = list(fns)
    G = {n: [torch.zeros(D, D, dtype=torch.float64, device=DEV) for _ in range(2)] for n in names}
    for i in range(0, len(xs), BATCH):
        x = xs[i:i + BATCH].to(DEV)
        half = torch.arange(i, i + len(x), device=DEV) % 2
        (la, lb), h_out = n4.forward_n4(model, x, keep_out=True)
        for j, n in enumerate(names):
            lg = fns[n](la, lb)
            y = n8.sample(lg.detach(), u[i:i + len(x)])
            loss = F.cross_entropy(lg.reshape(-1, VOCAB), y.reshape(-1), reduction='sum')
            (gr,) = torch.autograd.grad(loss, h_out, retain_graph=(j < len(names) - 1))
            gr = gr.double()
            for h in (0, 1):
                gh = gr[half == h].reshape(-1, D)
                G[n][h] += gh.T @ gh
            del lg, loss, gr
    return G


def pair(G, a, b):
    return n8.scalars(n8.from_covs([G[a], G[b]]))


@torch.no_grad()
def prediction_space(model, xs, ya_clean, y_noisy, log_u, rhos):
    """S2: per-position mean KL from the trained noisy head's prediction to each copy's, the noisy head's
    probability on tokens u never gives, and each head's loss against clean and noisy targets."""
    model.eval()
    kl = {r: 0.0 for r in rhos}
    loss = {k: [0.0, 0.0] for k in ['B'] + [copy_name(r) for r in rhos]}
    off_support, cnt = 0.0, 0
    zero_u = torch.isinf(log_u)
    for i in range(0, len(xs), BATCH):
        x = xs[i:i + BATCH].to(DEV)
        yc, yn = ya_clean[i:i + BATCH].to(DEV), y_noisy[i:i + BATCH].to(DEV)
        la, lb = n4.forward_n4(model, x)
        lq_b = torch.log_softmax(lb.float(), -1)
        q_b = lq_b.exp()
        off_support += q_b[..., zero_u].sum().item()
        heads = {'B': lq_b}
        for r in rhos:
            lc = copy_logits(la, log_u, r)
            heads[copy_name(r)] = lc
            kl[r] += (q_b * (lq_b - lc)).sum().item()
        for k, lp in heads.items():
            loss[k][0] -= lp.gather(-1, yc.unsqueeze(-1)).sum().item()
            loss[k][1] -= lp.gather(-1, yn.unsqueeze(-1)).sum().item()
        cnt += yc.numel()
    return {'kl_noisy_head_to_copy': {copy_name(r): kl[r] / cnt for r in rhos},
            'noisy_head_mass_where_u_is_zero': off_support / cnt, 'tokens_where_u_is_zero': int(zero_u.sum()),
            'loss_clean_targets': {k: v[0] / cnt for k, v in loss.items()},
            'loss_noisy_targets': {k: v[1] / cnt for k, v in loss.items()}}


# ------------------------------------------------------------------ checks

def analytic_check(n=None, vocab=2000, seed=n8.FISHER_SEED, rho=RHO):
    """Precondition 1a. N8's linear-softmax toy (logits = h W^T + b), with the head replaced by its ideal noisy
    copy. Head A puts most of its probability on R + 1 tokens (s1) and some on R + 1 others (s2); u puts
    S2_MASS on s2, far above head A's probability there, so the copy, which drops words head A thinks less likely
    than u does, has a different top subspace from head A's own Fisher. The exact Fisher of q with respect to h is
    (1 - rho)^2 sum over positions of M^T diag(1/q) M, with M = (diag p - p p^T) W. The sampled estimate, through
    the job's own code, must share its top-R subspace (overlap at least n8.TOY_MIN). For contrast: drawing the
    copy's targets from p instead of q, and head A's own Fisher."""
    D = study.CFG['d_model']
    n = n or (8 if TEST else 16)
    gen = torch.Generator().manual_seed(seed)
    W = (torch.randn(vocab, D, generator=gen) / math.sqrt(D)).to(DEV)
    perm = torch.randperm(vocab, generator=gen)
    s1, s2 = perm[:R + 1].to(DEV), perm[R + 1:2 * R + 2].to(DEV)
    b = torch.zeros(vocab, device=DEV)
    b[s1], b[s2] = 8.0, S2_BIAS                       # head A expects s1 and gives s2 some probability
    u = torch.full((vocab,), (1 - S2_MASS) / (vocab - R - 1), dtype=torch.float64, device=DEV)
    u[s2] = S2_MASS / (R + 1)                          # the frequent words: u is far above head A's p on s2
    log_u = u.float().log()
    h = torch.randn(n, SEQ, D, generator=gen).to(DEV)
    us = n8.uniforms(n, seed + 1)
    Wd = W.double()
    exact = torch.zeros(D, D, dtype=torch.float64, device=DEV)
    for i in range(n):
        p = torch.softmax((h[i] @ W.T + b).double(), -1)
        q = (1 - rho) * p + rho * u
        M = p.unsqueeze(-1) * Wd.unsqueeze(0) - p.unsqueeze(-1) * (p @ Wd).unsqueeze(1)
        exact += (1 - rho) ** 2 * (M / q.unsqueeze(-1)).reshape(-1, D).T @ M.reshape(-1, D)
    est = {}
    for name in ('sampled', 'targets_from_p', 'head_A_fisher'):
        G = torch.zeros(D, D, dtype=torch.float64, device=DEV)
        for i in range(0, n, 4):
            hh = h[i:i + 4].clone().requires_grad_(True)
            la = hh @ W.T + b
            lc = la if name == 'head_A_fisher' else copy_logits(la, log_u, rho)
            y = n8.sample((la if name == 'targets_from_p' else lc).detach(), us[i:i + 4])
            loss = F.cross_entropy(lc.reshape(-1, vocab), y.reshape(-1), reduction='sum')
            (gr,) = torch.autograd.grad(loss, hh)
            gr = gr.double().reshape(-1, D)
            G += gr.T @ gr
        est[name] = {'top_subspace_overlap': n8.overlap(n8.top(G), n8.top(exact)),
                     'relative_error': ((G / G.trace() * exact.trace() - exact).norm() / exact.norm()).item()}
    ev = torch.linalg.eigvalsh(exact).flip(0)
    return {'positions': n * SEQ, 'vocab': vocab, 'rho': rho, 'eigengap_R_over_R+1': (ev[R - 1] / ev[R]).item(), **est,
            'pass': est['sampled']['top_subspace_overlap'] >= n8.TOY_MIN}


def recorded_n8(n8_store, seed):
    """N8's recorded target-free overlap of head A with the trained noisy head: Drive first, the repository's copy after."""
    rel = f'results/f_n7_noisy_seed{seed}.json'
    if n8_store.pull(rel):
        return json.loads(n8_store.L(rel).read_text())['fisher']['overlap_AB'], 'drive'
    repo = HERE / 'results' / 'n8' / f'f_n7_noisy_seed{seed}.json'
    if repo.exists() and not TEST:
        return json.loads(repo.read_text())['fisher']['overlap_AB'], 'repository'
    return None, 'missing'


def measure_model(m, arm, mx, u, log_u, mys, probs, n8_store, seed):
    t0 = time.time()
    if arm == 'noisy':
        G = head_covs(m, mx, u, head_fns(log_u, RHOS))
        res = {'n9a': pair(G, 'A', copy_name(RHO)), 'n9b': pair(G, 'B', copy_name(RHO)), 's3': pair(G, 'A', 'B'),
               's1': {copy_name(r): pair(G, 'A', copy_name(r)) for r in RHOS},
               'stability': {k: n8.overlap(n8.top(G[k][0]), n8.top(G[k][1])) for k in G},
               'rotated_control': n8.scalars(n8.measure_f(n8.rotated_control(m), mx, u)),
               's2': prediction_space(m, mx, mys[0], n7.fixed_targets('noisy', mys, probs, n7.MEAS_NOISE_SEED)[1], log_u, RHOS)}
        rec, rec_from = recorded_n8(n8_store, seed)
        res['loading'] = {'recorded_n8_overlap_A_B': rec, 'recorded_from': rec_from,
                          'difference': None if rec is None else res['s3']['overlap_AB'] - rec,
                          'pass': rec is not None and abs(res['s3']['overlap_AB'] - rec) <= LOAD_TOL}
    else:
        G = head_covs(m, mx, u, head_fns(log_u, [0.0, RHO], with_b=False))
        res = {'s4': {copy_name(r): pair(G, 'A', copy_name(r)) for r in (0.0, RHO)},
               'stability': {k: n8.overlap(n8.top(G[k][0]), n8.top(G[k][1])) for k in G}}
    res['seconds'] = time.time() - t0
    return res


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
    src = study.Store(work.parent / 'n7_harder', local.parent / '_n9_src_n7_harder')
    n8_store = study.Store(work.parent / 'n8_fisher', local.parent / '_n9_src_n8_fisher')
    out = Path(args.out)
    (out / 'results').mkdir(parents=True, exist_ok=True)
    log('device:', DEV, '| test mode:', TEST, '| study folder (data):', source, '| N9 folder:', work, '| local:', local)
    log('weights from:', src.drive / 'final', '| N8 records from:', n8_store.drive / 'results')
    store.pull_dir('results')
    try:
        meta = study.prepare_data(data_store, None)
        probs = n7.unigram(study.load_tokens(store, meta, 'train'))
        log_u = probs.float().log().to(DEV)
        summary_path = store.L('results/n9_summary.json')
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

        def save_json(rel, obj):
            store.L(rel).write_text(json.dumps(obj, indent=2 if rel.endswith('n9_summary.json') else None))
            store.push(rel)

        allx, allys = n6.windows(study.load_tokens(store, meta, 'test'), limit=MEAS[1])
        mx, mys = allx[MEAS[0]:MEAS[1]], [y[MEAS[0]:MEAS[1]] for y in allys]
        u = n8.uniforms(len(mx))
        log(f'measured on test-split windows {MEAS[0] + 1}-{MEAS[0] + len(mx)}')

        if 'checks' not in summary:
            enc = study.gpt2_encoding()
            summary['checks'] = {'analytic': analytic_check(),
                                 'identical_heads': n8.identical_heads_check(SEEDS[0], mx, u),
                                 'test_windows_shifted': all(torch.equal(mys[k - 1][:, :-k], mx[:, k:]) for k in range(1, 9)),
                                 'unigram': {'sum': probs.sum().item(), 'zeros': int((probs == 0).sum()),
                                             'most_frequent_tokens': [enc.decode([t]) for t in torch.topk(probs, 5).indices.tolist()]}}
            summary['windows'] = {'split': 'test', 'measure': [MEAS[0] + 1, MEAS[0] + len(mx)]}
            summary['rho'], summary['rhos'], summary['fisher_seed'] = RHO, RHOS, n8.FISHER_SEED
            summary['readout_rank'], summary['subspace_rank'], summary['seeds'], summary['arms'] = n4.RANK, R, SEEDS, ARMS
            save_json('results/n9_summary.json', summary)
        c = summary['checks']
        log('analytic check on mixtures, top-subspace overlap:', round(c['analytic']['sampled']['top_subspace_overlap'], 4),
            '| identical heads:', c['identical_heads']['agreement_ratio'], '| windows shifted:', c['test_windows_shifted'],
            '| unigram sum', round(c['unigram']['sum'], 6), 'top tokens', c['unigram']['most_frequent_tokens'])
        if not (c['analytic']['pass'] and c['identical_heads']['pass'] and c['test_windows_shifted'] and abs(c['unigram']['sum'] - 1) < 1e-6):
            raise SystemExit('A check of the estimator, the windows or the unigram failed before any model was measured. Stopping.')

        for seed in SEEDS:                                   # the primary models first
            for arm in ['noisy'] + [a for a in ARMS if a != 'noisy']:
                tag = f'{arm}_seed{seed}'
                rel = f'results/c_{tag}.json'
                if store.L(rel).exists():
                    continue
                if not src.pull(f'final/{tag}.pt'):
                    raise SystemExit(f'weights {src.drive / "final" / (tag + ".pt")} are missing on Drive.')
                m = n4.new_n4(seed, n8.BASE)
                m.load_state_dict(torch.load(src.L(f'final/{tag}.pt'), map_location=DEV, weights_only=False))
                res = {'tag': tag, 'seed': seed, 'arm': arm, **measure_model(m, arm, mx, u, log_u, mys, probs, n8_store, seed)}
                save_json(rel, res)
                if arm == 'noisy':
                    log(f"c_{tag}: A vs copy {res['n9a']['agreement_ratio']:.3f} | noisy head vs copy {res['n9b']['agreement_ratio']:.3f} | "
                        f"A vs noisy head {res['s3']['agreement_ratio']:.3f} (overlap {res['s3']['overlap_AB']:.3f}, N8 {res['loading']['recorded_n8_overlap_A_B']}) | "
                        f"copy at 0 {res['s1']['copy0']['agreement_ratio']:.6f} | rotated {res['rotated_control']['agreement_ratio']:.3f} | {res['seconds']:.0f} s")
                else:
                    log(f"c_{tag}: A vs copy {res['s4'][copy_name(RHO)]['agreement_ratio']:.3f} | copy at 0 {res['s4']['copy0']['agreement_ratio']:.6f} | {res['seconds']:.0f} s")
        log('N9 complete: all measurements finished')
    finally:
        dest = out / 'results'
        for f in (store.local / 'results').glob('*.json'):
            (dest / f.name).write_bytes(f.read_bytes())
        log(f'copied {len(list(dest.glob("*.json")))} result files for upload')
        log('waiting for background Drive saves to finish (up to 15 minutes)')
        store.flush()
        failures = store.failures + data_store.failures + src.failures + n8_store.failures
        (out / 'drive_sync.json').write_text(json.dumps({'failures': failures}, indent=2))
        log('Drive saves finished' if not failures else f'Drive problems this session: {failures}')


if __name__ == '__main__':
    main()
