"""Apply G1's preregistered decision rule to the saved records and write G1_RESULTS.md.

Written and committed before any real G1 run. Every number in the report comes from the
JSON records in the results folder. The decision rule is the one in amendment A1 of
G1_PREREGISTRATION.md: the primary statistic is the log curvature ratio between the two
control values, and the original Cohen's d is reported as a secondary measure.

    python g1_analyze.py [results folder] [report path]   # default: results/g1 here, G1_RESULTS.md
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'results' / 'g1'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'G1_RESULTS.md'

SEEDS = [3001, 3002, 3003, 3004, 3005]
NEEDED = 4                     # "in at least 4 of 5"
LARGE, SMALL = math.log(2), math.log(1.25)   # amendment A1: supported at a curvature ratio >= 2; refuted below 1.25
SPLIT_MAX = 0.1                # must match: odd against even windows, difference in log ratio
MANIP_MIN = 0.5                # manipulation check, in nats
EXCLUDED_FLAG = 0.01


def load(name):
    p = RES / f'{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


def cohen_d(a, b):
    """(mean a - mean b) / pooled standard deviation."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * a.var(ddof=1) + (nb - 1) * b.var(ddof=1)) / (na + nb - 2))
    return float((a.mean() - b.mean()) / sp)


def log_ratio(a, b):
    """Mean over windows of log curvature at one control value minus the other: the log of the curvature ratio."""
    return float(np.mean(np.asarray(a, float) - np.asarray(b, float)))


def paired_dz(a, b):
    diff = np.asarray(a, float) - np.asarray(b, float)
    return float(diff.mean() / diff.std(ddof=1))


def classifier_accuracy(geo):
    """S6: predict u from the curvature-by-depth profile; trained on windows 1, 3, 5..., tested on 2, 4, 6..."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    P, M = np.asarray(geo['plus']['logk_by_depth'], float), np.asarray(geo['minus']['logk_by_depth'], float)
    X = {s: np.concatenate([P[s::2], M[s::2]]) for s in (0, 1)}
    y = {s: np.concatenate([np.ones(len(P[s::2])), np.zeros(len(M[s::2]))]) for s in (0, 1)}
    if not (np.isfinite(X[0]).all() and np.isfinite(X[1]).all()):
        return None
    sc = StandardScaler().fit(X[0])
    clf = LogisticRegression(max_iter=2000).fit(sc.transform(X[0]), y[0])
    return float(clf.score(sc.transform(X[1]), y[1]))


def f(x, d=3, sign=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 'n/a'
    return f'{x:+.{d}f}' if sign else f'{x:.{d}f}'


def main():
    summary = load('g1_summary') or {}
    checks = summary.get('checks', {})
    geo = {(v, s): load(f'geom_{v}_seed{s}') for v in ('main', 'shuf') for s in SEEDS}
    trn = {(v, s): load(f'train_{v}_seed{s}') for v in ('main', 'shuf') for s in SEEDS}
    main_seeds = [s for s in SEEDS if geo[('main', s)] and trn[('main', s)]]
    shuf_seeds = [s for s in SEEDS if geo[('shuf', s)]]
    complete = len(main_seeds) == len(SEEDS) and len(shuf_seeds) == len(SEEDS)

    def lk(v, s, u, key='logk_mean'):
        return geo[(v, s)]['controls'][u][key]

    d = {(v, s): log_ratio(lk(v, s, 'plus'), lk(v, s, 'minus')) for v, ss in (('main', main_seeds), ('shuf', shuf_seeds)) for s in ss}
    split = {s: log_ratio(lk('main', s, 'plus')[0::2], lk('main', s, 'minus')[0::2])
             - log_ratio(lk('main', s, 'plus')[1::2], lk('main', s, 'minus')[1::2]) for s in main_seeds}
    manip = {s: trn[('main', s)]['final_losses']['next_uminus'] - trn[('main', s)]['final_losses']['next_uplus'] for s in main_seeds}
    excluded = {(v, s): max(g['controls'][u]['excluded_points'] / g['controls'][u]['points'] for u in g['controls'])
                for (v, s), g in geo.items() if g}

    # ---------------------------------------------------------------- preconditions and decision
    pre = {
        '1. Manipulation check: wrong control costs at least 0.5 nats on the next-token task, in at least 4 of 5 main models':
            sum(manip[s] >= MANIP_MIN for s in main_seeds) >= NEEDED,
        '2. Analytic check: the curvature code returns textbook curvature within 0.1%':
            bool(checks.get('analytic', {}).get('pass')),
        '3. Must fail: shuffled-control models give a curvature ratio below 1.25 in at least 4 of 5':
            sum(abs(d[('shuf', s)]) < SMALL for s in shuf_seeds) >= NEEDED,
        '4. Must match: the log ratio from odd and from even windows differs by less than 0.1 in every main model':
            bool(main_seeds) and all(abs(split[s]) < SPLIT_MAX for s in main_seeds),
    }
    n_large = sum(abs(d[('main', s)]) >= LARGE for s in main_seeds)
    n_small = sum(abs(d[('main', s)]) < SMALL for s in main_seeds)
    if not complete:
        verdict = f'**Not reached: only {len(main_seeds)} of 5 main and {len(shuf_seeds)} of 5 shuffled-control models have records.**'
    elif not all(pre.values()):
        failed = [k.split(':')[0] for k, ok in pre.items() if not ok]
        verdict = f'**Untestable: precondition {", ".join(failed)} failed.** The values below decide nothing.'
    elif n_large >= NEEDED:
        verdict = f'**Supported: distinct regimes.** Curvature ratio at least 2 in {n_large} of 5 main models.'
    elif n_small >= NEEDED:
        verdict = f'**Refuted: no distinct regimes.** Curvature ratio below 1.25 in {n_small} of 5 main models, and the models use u.'
    else:
        verdict = (f'**Intermediate: some separation.** Curvature ratio at least 2 in {n_large} and below 1.25 in {n_small} '
                   'of 5 main models; neither rule is met.')

    L = ['# G1 results', '',
         'Generated by `g1_analyze.py` from the records in `results/g1/`. Preregistration: `G1_PREREGISTRATION.md`.', '',
         '## Verdict', '', verdict, '', '## Preconditions', '', '| Precondition | Met |', '|---|---|']
    L += [f'| {k} | {"yes" if ok else "**no**"} |' for k, ok in pre.items()]

    def ratio(x):
        return None if x is None else math.exp(abs(x))

    L += ['', '## Primary outcome: the curvature ratio between u = +1 and u = −1', '',
          'The log ratio is the mean over windows of window-mean log curvature at u = +1 minus that at u = −1; positive means '
          'trajectories curve more at u = +1. The ratio is exp(|log ratio|), the larger curvature over the smaller. '
          'The shuffled-control models are the must-fail control.', '',
          '| Seed | Main log ratio | Main ratio | Shuffled log ratio | Shuffled ratio | Odd − even log ratio (main) | Wrong-control cost (nats) | Excluded points (max over u) |',
          '|---|---|---|---|---|---|---|---|']
    for s in SEEDS:
        ex = [excluded.get((v, s)) for v in ('main', 'shuf')]
        ex = max([e for e in ex if e is not None], default=None)
        L.append(f"| {s} | {f(d.get(('main', s)))} | {f(ratio(d.get(('main', s))), sign=False)} | {f(d.get(('shuf', s)))} | "
                 f"{f(ratio(d.get(('shuf', s))), sign=False)} | {f(split.get(s))} | "
                 f"{f(manip.get(s))} | {'n/a' if ex is None else f'{100 * ex:.3f}%' + (' **over 1%**' if ex > EXCLUDED_FLAG else '')} |")
    L += ['', f'Main models with a ratio of at least 2: {n_large}. Below 1.25: {n_small}.']

    # ---------------------------------------------------------------- secondary
    L += ['', '## Secondary and exploratory measures', '', 'None of these enters the decision rule.', '',
          '### Other versions of the contrast, and its direction (S0, S0b, S0c, S9, S1)', '',
          'S0c is the originally preregistered primary statistic, Cohen\'s d with the pooled standard deviation across windows, '
          'kept for comparison; amendment A1 explains why it is not used for the decision.', '',
          '| Seed | Model | Cohen\'s d, log κ (S0c) | d, plain mean κ (S0) | Paired d_z, log κ (S0b) | Log ratio at the Euler states (S9) | Curves more (S1) |',
          '|---|---|---|---|---|---|---|']
    for v, ss in (('main', main_seeds), ('shuf', shuf_seeds)):
        for s in ss:
            dc = cohen_d(lk(v, s, 'plus'), lk(v, s, 'minus'))
            d0 = cohen_d(lk(v, s, 'plus', 'k_mean'), lk(v, s, 'minus', 'k_mean'))
            dz = paired_dz(lk(v, s, 'plus'), lk(v, s, 'minus'))
            d9 = log_ratio(lk(v, s, 'plus', 'euler_logk_mean'), lk(v, s, 'minus', 'euler_logk_mean'))
            L.append(f"| {s} | {v} | {f(dc)} | {f(d0)} | {f(dz)} | {f(d9)} | {'u = +1' if d[(v, s)] > 0 else 'u = −1'} |")

    if main_seeds:
        depths = geo[('main', main_seeds[0])]['depths']
        L += ['', '### S2: curvature by depth, main models', '',
              'Mean log κ over windows, then over models; the log ratio at each depth averaged over models.', '',
              '| Depth τ | u = +1 | u = −1 | u = 0 | Mean log ratio (+1 vs −1) |', '|---|---|---|---|---|']
        for k, t in enumerate(depths):
            col = {u: np.mean([np.mean(np.asarray(lk('main', s, u, 'logk_by_depth'))[:, k]) for s in main_seeds])
                   for u in ('plus', 'minus', 'zero')}
            dk = np.mean([log_ratio(np.asarray(lk('main', s, 'plus', 'logk_by_depth'))[:, k],
                                    np.asarray(lk('main', s, 'minus', 'logk_by_depth'))[:, k]) for s in main_seeds])
            L.append(f"| {t:.4f} | {col['plus']:.3f} | {col['minus']:.3f} | {col['zero']:.3f} | {f(float(dk))} |")

    L += ['', '### S3, S4, S5: speed, divergence and solver effort, by model', '',
          'Speed is the mean ‖v‖ per token. Divergence is the Hutchinson estimate of trace(∂f/∂h) per token, '
          'averaged over its five depths. Solver effort is dopri5 function evaluations per batch of windows.', '',
          '| Seed | Model | Speed +1 | Speed −1 | d speed | Div +1 | Div −1 | d div | NFE +1 (1e-3) | NFE −1 (1e-3) | NFE +1 (1e-5) | NFE −1 (1e-5) |',
          '|---|---|---|---|---|---|---|---|---|---|---|---|']
    for v, ss in (('main', main_seeds), ('shuf', shuf_seeds)):
        for s in ss:
            sp = {u: lk(v, s, u, 'speed_mean') for u in ('plus', 'minus')}
            dv = {u: np.asarray(lk(v, s, u, 'div_by_depth')).mean(1) for u in ('plus', 'minus')}
            nfe = {(u, t): np.mean(lk(v, s, u, f'nfe_tol{t}_per_batch')) for u in ('plus', 'minus') for t in ('1e-3', '1e-5')}
            L.append(f"| {s} | {v} | {np.mean(sp['plus']):.4f} | {np.mean(sp['minus']):.4f} | {f(cohen_d(sp['plus'], sp['minus']))} | "
                     f"{dv['plus'].mean():+.4f} | {dv['minus'].mean():+.4f} | {f(cohen_d(dv['plus'], dv['minus']))} | "
                     f"{nfe[('plus', '1e-3')]:.1f} | {nfe[('minus', '1e-3')]:.1f} | {nfe[('plus', '1e-5')]:.1f} | {nfe[('minus', '1e-5')]:.1f} |")

    L += ['', '### S6 and S7: a regime classifier, and where u = 0 lies', '',
          'S6: logistic regression predicting u from the 17-depth curvature profile, trained on windows 1, 3, 5... and '
          'tested on 2, 4, 6...; chance is 0.5. S7: position of u = 0 on the line from u = −1 (0) to u = +1 (1), '
          'by mean log κ; values outside 0 to 1 mean u = 0 lies beyond either regime.', '',
          '| Seed | Model | Classifier accuracy (S6) | Position of u = 0 (S7) | Log ratio, u = 0 vs u = +1 | Log ratio, u = 0 vs u = −1 |',
          '|---|---|---|---|---|---|']
    for v, ss in (('main', main_seeds), ('shuf', shuf_seeds)):
        for s in ss:
            g = geo[(v, s)]['controls']
            acc = classifier_accuracy(g)
            m = {u: np.mean(g[u]['logk_mean']) for u in ('plus', 'minus', 'zero')}
            pos = (m['zero'] - m['minus']) / (m['plus'] - m['minus']) if m['plus'] != m['minus'] else float('nan')
            L.append(f"| {s} | {v} | {f(acc, sign=False)} | {f(float(pos), 2)} | "
                     f"{f(log_ratio(g['zero']['logk_mean'], g['plus']['logk_mean']))} | "
                     f"{f(log_ratio(g['zero']['logk_mean'], g['minus']['logk_mean']))} |")

    L += ['', '### S8: validation losses on the whole split (nats)', '',
          'Rows are models; "next" is the next-token task, "skip" the token after next.', '',
          '| Seed | Model | next, u=+1 | next, u=−1 | next, u=0 | skip, u=−1 | skip, u=+1 | skip, u=0 | final α | u matched task |',
          '|---|---|---|---|---|---|---|---|---|---|']
    for v in ('main', 'shuf'):
        for s in SEEDS:
            r = trn[(v, s)]
            if not r:
                continue
            fl = r['final_losses']
            L.append(f"| {s} | {v} | {fl['next_uplus']:.4f} | {fl['next_uminus']:.4f} | {fl['next_uzero']:.4f} | "
                     f"{fl['skip_uminus']:.4f} | {fl['skip_uplus']:.4f} | {fl['skip_uzero']:.4f} | "
                     f"{r['alpha_final']:.3f} | {r['u_matches_task_fraction']:.3f} |")

    an = checks.get('analytic', {})
    L += ['', '## Checks', '', '### Analytic curvature check', '', '| Field | Worst error | Kind |', '|---|---|---|']
    L += [f"| {k} | {v['worst_error']:.2e} | {v['kind']} |" for k, v in an.items() if k != 'pass']
    b = checks.get('batch', {})
    L += ['', f"Targets at i + 1 and i + 2 confirmed on the first training batch: {b.get('targets_are_i+1_and_i+2')}."]
    for name in ('next', 'skip'):
        ex = b.get(f'{name}_example')
        if ex:
            L.append(f"- {name} task: input ends {ex['input_end']!r}; targets end {ex['target_end']!r}")
    sm = checks.get('same_model', [])
    L += ['', 'Exact and adjoint versions match before training: ' +
          ', '.join(f"seed {x['seed']} {'yes' if x['pass'] else 'NO'} (loss diff {x['loss_rel_diff']:.1e}, cosine {x['downstream_cos']:.7f})" for x in sm) + '.']
    times = [trn[k]['train_seconds'] / 3600 for k in trn if trn[k]]
    if times:
        L += ['', f'Training time per run: {np.mean(times):.3f} h on average, over {len(times)} runs.']
    L.append('')
    OUT.write_text('\n'.join(L))
    print('\n'.join(L[:40]))


if __name__ == '__main__':
    main()
