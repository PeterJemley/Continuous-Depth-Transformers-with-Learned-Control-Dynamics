"""Apply G1d's preregistered decision rule to the saved records and write G1D_RESULTS.md.

Written and committed before any real G1d run. Every number in the report comes from the
JSON records in the results folder.

    python g1d_analyze.py [results folder] [report path]   # default: results/g1d here, G1D_RESULTS.md
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from g1_analyze import classifier_accuracy, log_ratio  # noqa: E402
from g1c_analyze import score  # noqa: E402

RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'results' / 'g1d'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'G1D_RESULTS.md'

SEEDS = [3201, 3202, 3203, 3204, 3205]
ARMS = ['control', 'capped', 'sham']
NEEDED = 4
MARGIN = 0.05
LEVEL = 0.95
SD_SENSITIVE = 0.04
CAPPED_MAX = 3.0
MANIP_MIN = 0.5


def load(name):
    p = RES / f'{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


def interval(diffs, level=LEVEL):
    n = len(diffs)
    m = st.mean(diffs)
    sd = st.stdev(diffs) if n > 1 else float('nan')
    half = stats.t.ppf(1 - (1 - level) / 2, n - 1) * sd / math.sqrt(n) if n > 1 else float('nan')
    return m, sd, m - half, m + half


def f(x, d=4, sign=True):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 'n/a'
    return f'{x:+.{d}f}' if sign else f'{x:.{d}f}'


def main():
    summary = load('g1d_summary') or {}
    checks = summary.get('checks', {})
    lam = summary.get('lambda')
    trn = {(a, s): load(f'train_{a}_seed{s}') for a in ARMS for s in SEEDS}
    con = {(a, s): load(f'contr_{a}_seed{s}') for a in ARMS for s in SEEDS}
    geo = {(a, s): load(f'geom_{a}_seed{s}') for a in ARMS for s in SEEDS}
    complete_seeds = [s for s in SEEDS if all(trn[(a, s)] and con[(a, s)] for a in ARMS)]

    def loss(a, s, key):
        return trn[(a, s)]['final_losses'][key]

    def rates(a, s):
        r = con[(a, s)]['controls']
        return float(np.mean(r['plus']['c'])), float(np.mean(r['minus']['c']))

    def D(a, s):
        """(farther-task change) - (next-token change), arm a against control, each task with its right control."""
        return ((loss(a, s, 'skip_uminus') - loss('control', s, 'skip_uminus'))
                - (loss(a, s, 'next_uplus') - loss('control', s, 'next_uplus')))

    d_main = [D('capped', s) for s in complete_seeds]
    d_sham = [D('sham', s) for s in complete_seeds]
    prim = interval(d_main) if len(d_main) >= 2 else None
    null = interval(d_sham) if len(d_sham) >= 2 else None
    manip_u = {s: loss('control', s, 'next_uminus') - loss('control', s, 'next_uplus') for s in SEEDS if trn[('control', s)]}

    pre = {
        '1. The cap was imposed: a λ in the grid met the pilot rule': lam is not None,
        '2. Manipulation check: capped models\' c̄(−1) at most 3.0, in at least 4 of 5':
            sum(rates('capped', s)[1] <= CAPPED_MAX for s in SEEDS if con[('capped', s)]) >= NEEDED,
        '3. Positive control: control models\' contraction ratio R ≥ 2 (scored as in G1c), in at least 4 of 5':
            sum(score(*rates('control', s))[2] == 'large' for s in SEEDS if con[('control', s)]) >= NEEDED,
        '4. Use of u: wrong control costs at least 0.5 nats on the next-token task, in at least 4 of 5 control models':
            sum(v >= MANIP_MIN for v in manip_u.values()) >= NEEDED,
        '5. Null contrast: the sham arm\'s D interval lies within ±0.05':
            null is not None and -MARGIN <= null[2] and null[3] <= MARGIN,
        '6. Analytic checks: the training penalty (trace and gradient within 2%) and the contraction measure':
            bool(checks.get('penalty', {}).get('pass')) and bool(checks.get('divergence', {}).get('pass')),
    }

    if lam is None and summary.get('pilot') is not None and 'lambda' in summary:
        verdict = '**Untestable: precondition 1 failed.** No λ in the grid held the contraction down, so no main runs were made.'
    elif len(complete_seeds) < len(SEEDS):
        verdict = f'**Not reached: all three arms have records for only {len(complete_seeds)} of 5 seeds.**'
    elif not all(pre.values()):
        failed = [k.split(':')[0] for k, ok in pre.items() if not ok]
        verdict = f'**Untestable: precondition {", ".join(failed)} failed.** The values below decide nothing.'
    else:
        m, sd, lo, hi = prim
        if lo > MARGIN:
            verdict = f'**Supported: the contraction is functional.** D = {m:+.4f}, 95% interval [{lo:+.4f}, {hi:+.4f}], wholly above +0.05.'
        elif hi < MARGIN and sd <= SD_SENSITIVE:
            verdict = f'**Refuted: no material differential cost.** D = {m:+.4f}, 95% interval [{lo:+.4f}, {hi:+.4f}], wholly below +0.05.'
        elif hi < MARGIN:
            verdict = (f'**Inconclusive: the design was not sensitive enough.** D = {m:+.4f}, interval [{lo:+.4f}, {hi:+.4f}], '
                       f'but its standard deviation, {sd:.4f}, exceeds 0.04, so "refuted" is not claimed.')
        elif lo > 0 or hi < 0:
            verdict = f'**Some effect, size unclear.** D = {m:+.4f}, 95% interval [{lo:+.4f}, {hi:+.4f}].'
        else:
            verdict = f'**Inconclusive.** D = {m:+.4f}, 95% interval [{lo:+.4f}, {hi:+.4f}].'

    L = ['# G1d results', '',
         'Generated by `g1d_analyze.py` from the records in `results/g1d/`. Preregistration: `G1D_PREREGISTRATION.md`.',
         f"Contraction measured on validation windows {summary.get('windows', ['?', '?'])[0]}–{summary.get('windows', ['?', '?'])[1]}.", '',
         '## Verdict', '', verdict, '', '## Preconditions', '', '| Precondition | Met |', '|---|---|']
    L += [f'| {k} | {"yes" if ok else "**no**"} |' for k, ok in pre.items()]

    L += ['', '## The pilot: choosing λ', '', 'The smallest λ at which the capped model\'s measured c̄(−1) was at most 3.0. '
          'Grid tried in order until one qualified.', '', '| λ | c̄(−1) |', '|---|---|']
    L += [f'| {k} | {v:.4f} |' for k, v in (summary.get('pilot') or {}).items()]
    L += ['', f'Chosen λ: {lam if lam is not None else "none"}.']

    L += ['', '## Primary outcome: D, by seed', '',
          'D = (farther-task loss, arm − control) − (next-token loss, arm − control), on the whole validation split, '
          'each task with its right control value. Positive D means the farther task is hurt more.', '',
          '| Seed | Farther, control | Farther, capped | Next, control | Next, capped | D (capped) | D (sham) |',
          '|---|---|---|---|---|---|---|']
    for s in complete_seeds:
        L.append(f"| {s} | {loss('control', s, 'skip_uminus'):.4f} | {loss('capped', s, 'skip_uminus'):.4f} | "
                 f"{loss('control', s, 'next_uplus'):.4f} | {loss('capped', s, 'next_uplus'):.4f} | "
                 f"{D('capped', s):+.4f} | {D('sham', s):+.4f} |")
    for name, iv in (('capped (primary)', prim), ('sham (null contrast)', null)):
        if iv:
            L.append(f'\nD, {name}: mean {iv[0]:+.4f}, standard deviation {iv[1]:.4f}, 95% interval [{iv[2]:+.4f}, {iv[3]:+.4f}].')

    # ---------------------------------------------------------------- secondary
    L += ['', '## Secondary and exploratory measures', '', 'None of these enters the decision rule.', '',
          '### S1: each task separately (capped − control, 95% intervals)', '', '| Task | Mean change | 95% interval |', '|---|---|---|']
    for task, key in (('Farther (u = −1)', 'skip_uminus'), ('Next token (u = +1)', 'next_uplus')):
        diffs = [loss('capped', s, key) - loss('control', s, key) for s in complete_seeds]
        if len(diffs) >= 2:
            m, sd, lo, hi = interval(diffs)
            L.append(f'| {task} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] |')

    L += ['', '### S2: how the cap was met — contraction by arm', '',
          '| Seed | Arm | c̄(+1) | c̄(−1) | c̄(0) | R | Scored as |', '|---|---|---|---|---|---|---|']
    for s in SEEDS:
        for a in ARMS:
            if con[(a, s)]:
                cp, cm = rates(a, s)
                c0 = float(np.mean(con[(a, s)]['controls']['zero']['c']))
                _, r, cat = score(cp, cm)
                L.append(f'| {s} | {a} | {cp:.4f} | {cm:.4f} | {c0:.4f} | {f(r, 3, False)} | {cat} |')
    capped = [s for s in SEEDS if con[('capped', s)]]
    if capped:
        times = con[('capped', capped[0])]['times']
        L += ['', 'Capped models by depth (mean over models):', '', '| Depth | c̄(+1) | c̄(−1) |', '|---|---|---|']
        for k, t in enumerate(times):
            m = {u: np.mean([np.asarray(con[('capped', s)]['controls'][u]['c'])[:, k].mean() for s in capped]) for u in ('plus', 'minus')}
            L.append(f"| {t:.2f} | {m['plus']:.4f} | {m['minus']:.4f} |")

    L += ['', '### S3: the penalty during training (capped arm)', '',
          'Unweighted penalty P (mean of max(0, c − 2)² over u = −1 sequences and Euler states), at the start, middle and end.', '',
          '| Seed | First | Middle | Last | Mean |', '|---|---|---|---|---|']
    for s in SEEDS:
        r = trn[('capped', s)]
        p = (r or {}).get('penalty_every_50') or []
        if p:
            v = [x[1] for x in p]
            L.append(f'| {s} | {v[0]:.4f} | {v[len(v) // 2]:.4f} | {v[-1]:.4f} | {np.mean(v):.4f} |')

    L += ['', '### S4: other geometry (curvature ratio, classifier, speed)', '',
          '| Seed | Arm | Curvature ratio | Classifier accuracy | Speed +1 | Speed −1 |', '|---|---|---|---|---|---|']
    for s in SEEDS:
        for a in ARMS:
            gr = geo[(a, s)]
            if gr:
                gcn = gr['controls']
                lr = log_ratio(gcn['plus']['logk_mean'], gcn['minus']['logk_mean'])
                acc = classifier_accuracy(gcn)
                L.append(f"| {s} | {a} | {math.exp(abs(lr)):.3f} | {f(acc, 3, False)} | "
                         f"{np.mean(gcn['plus']['speed_mean']):.3f} | {np.mean(gcn['minus']['speed_mean']):.3f} |")

    L += ['', '### S5 and S6: wrong-control costs, α and training time', '',
          '| Seed | Arm | Wrong-control cost, next (nats) | Wrong-control cost, farther (nats) | Final α | Training hours |',
          '|---|---|---|---|---|---|']
    for s in SEEDS:
        for a in ARMS:
            r = trn[(a, s)]
            if r:
                fl = r['final_losses']
                L.append(f"| {s} | {a} | {fl['next_uminus'] - fl['next_uplus']:+.3f} | {fl['skip_uplus'] - fl['skip_uminus']:+.3f} | "
                         f"{r['alpha_final']:.3f} | {r['train_seconds'] / 3600:.3f} |")

    L += ['', '## Checks', '']
    pc = checks.get('penalty', {})
    if pc:
        L.append(f"Training penalty on a linear field: estimate {pc['estimate']:.4f} against {pc['truth']:.4f} "
                 f"(error {pc['relative_error']:.2%}); gradient error {pc['gradient_relative_error']:.2%}.")
    dv = checks.get('divergence', {})
    if dv:
        L.append(f"\nContraction measure: diagonal error {dv['diagonal']['worst_relative_error']:.1e}, dense error {dv['dense']['relative_error']:.2%}.")
    sh = checks.get('sham', {})
    if sh:
        L.append(f"\nSham penalty on one batch: value {sh['sham_value']}, all gradients zero: {sh['sham_gradient_all_zero']}; "
                 f"control and sham start identically: {sh['same_initial_weights']} (weights), {sh['same_first_batch']} (first batch).")
    b = checks.get('batch', {})
    L.append(f"\nTargets at i + 1 and i + 2 confirmed on the first training batch: {b.get('targets_are_i+1_and_i+2')}.")
    sm = checks.get('same_model', [])
    L.append('\nExact and adjoint versions match before training: ' +
             ', '.join(f"seed {x['seed']} {'yes' if x['pass'] else 'NO'}" for x in sm) + '.\n')
    OUT.write_text('\n'.join(L))
    print('\n'.join(L[:40]))


if __name__ == '__main__':
    main()
