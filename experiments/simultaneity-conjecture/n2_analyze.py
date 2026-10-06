"""Apply N2's preregistered decision rules (with amendment A1) to the saved records and write N2_RESULTS.md.

Written and committed before any real N2 run. Every number in the report comes from the
JSON records in the results folder.

    python n2_analyze.py [results folder] [report path]   # default: results/n2 here, N2_RESULTS.md
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

import numpy as np
from scipy import stats

HERE = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'results' / 'n2'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'N2_RESULTS.md'

SEEDS = [3401, 3402, 3403, 3404, 3405]
NEEDED = 4
R = 16
# Amendment A1: subspace contraction is per dimension, so the near-zero floor is in per-dimension units.
FLOOR, CLEAR = 0.002, 0.004
LARGE, SMALL, LOW = 2.0, 1.25, 0.8
SPLIT_MAX = 0.1
EQUAL_MAX = math.log(1.25)
MARGIN, SD_SENSITIVE = 0.05, 0.04


def load(name):
    p = RES / f'{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


def score(ca, cb):
    """G1c's table with per-dimension floors. Returns (case, R or None, category)."""
    if ca >= FLOOR and cb >= FLOOR:
        r = cb / ca
        return 'both contract', r, 'large' if r >= LARGE else 'small' if r < SMALL else 'middle'
    if cb < FLOOR:
        return 'B does not contract', None, 'small'
    if cb >= CLEAR:
        return 'B contracts, A does not', None, 'large'
    return 'B contracts weakly, A does not', None, 'middle'


def rates(sub, a=f'A{R}', b=f'B{R}'):
    c = sub['contraction']
    return float(np.mean(c[a])), float(np.mean(c[b]))


def agree(p, q, tol):
    sp, sq = score(*p), score(*q)
    if sp[1] is not None and sq[1] is not None:
        d = abs(math.log(sp[1]) - math.log(sq[1]))
        return d, d < tol
    return None, sp[0] == sq[0]


def interval(diffs, level=0.95):
    n = len(diffs)
    m = st.mean(diffs)
    sd = st.stdev(diffs) if n > 1 else float('nan')
    half = stats.t.ppf(1 - (1 - level) / 2, n - 1) * sd / math.sqrt(n) if n > 1 else float('nan')
    return m, sd, m - half, m + half


def overlap(U, V):
    U, V = np.asarray(U), np.asarray(V)
    return float(((U.T @ V) ** 2).sum() / U.shape[1])


def f(x, d=4, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 'n/a'
    return f'{x:+.{d}f}' if sign else f'{x:.{d}f}'


def main():
    summary = load('n2_summary') or {}
    checks = summary.get('checks', {})
    lam = summary.get('lambda')
    trn = {(a, s): load(f'{a}_seed{s}') for a in ('dual', 'same', 'equalised', 'sham') for s in SEEDS}
    sub = {(a, s): load(f'sub_{a}_seed{s}') for a in ('dual', 'same', 'equalised', 'sham') for s in SEEDS}
    mid = {s: load(f'sub_dual_seed{s}_mid') for s in SEEDS}
    dual_seeds = [s for s in SEEDS if sub[('dual', s)] and trn[('dual', s)]]
    same_seeds = [s for s in SEEDS if sub[('same', s)]]

    scored = {k: score(*rates(v)) for k, v in sub.items() if v}
    split = {s: agree(rates(sub[('dual', s)], f'A{R}_odd', f'B{R}_odd'), rates(sub[('dual', s)], f'A{R}_even', f'B{R}_even'), SPLIT_MAX)
             for s in dual_seeds}

    def same_ok(s):
        ca, cb = rates(sub[('same', s)])
        _, r, _ = scored[('same', s)]
        if r is not None:
            return LOW <= r < SMALL
        return ca < FLOOR and cb < FLOOR and abs(cb - ca) < FLOOR

    pre1 = {
        '1. Analytic checks: projected estimator and training penalty':
            bool(checks.get('projected', {}).get('pass')) and bool(checks.get('penalty', {}).get('pass')),
        '2. Batch and same-model checks':
            bool(checks.get('batch', {}).get('targets_are_i+1_and_i+2')) and all(x['pass'] for x in checks.get('same_model', [{'pass': False}])),
    }
    pre_a = {
        'Must fail: dual-same models give R_sub from 0.8 to below 1.25, in at least 4 of 5':
            sum(same_ok(s) for s in same_seeds) >= NEEDED,
        'Must match: odd and even estimation windows agree (log R within 0.1, or the same case), every dual model':
            bool(dual_seeds) and all(split[s][1] for s in dual_seeds),
    }
    n_large = sum(scored[('dual', s)][2] == 'large' for s in dual_seeds)
    n_small = sum(scored[('dual', s)][2] == 'small' for s in dual_seeds)
    if len(dual_seeds) < 5 or len(same_seeds) < 5:
        v2a = f'**Not reached: {len(dual_seeds)} of 5 dual and {len(same_seeds)} of 5 dual-same models measured.**'
    elif not (all(pre1.values()) and all(pre_a.values())):
        failed = [k.split(':')[0] for k, ok in {**pre1, **pre_a}.items() if not ok]
        v2a = f'**Untestable: a precondition failed ({"; ".join(failed)}).**'
    elif n_large >= NEEDED:
        v2a = f'**Supported: per-reading timescales exist.** R_sub ≥ 2 in {n_large} of 5 dual models.'
    elif n_small >= NEEDED:
        v2a = f'**Refuted.** R_sub < 1.25 in {n_small} of 5 dual models.'
    else:
        v2a = f'**Intermediate.** R_sub ≥ 2 in {n_large} and < 1.25 in {n_small} of 5 dual models.'

    # ---------------------------------------------------------------- N2b
    eq_seeds = [s for s in SEEDS if trn[('equalised', s)] and trn[('sham', s)] and trn[('dual', s)]]

    def joint(a, s):
        return trn[(a, s)]['final_loss_A'] + trn[(a, s)]['final_loss_B']

    def equalised_ok(s):
        ca, cb = rates(sub[('equalised', s)])
        if ca > 0 and cb > 0 and ca >= FLOOR and cb >= FLOOR:
            return abs(math.log(cb / ca)) <= EQUAL_MAX
        return abs(cb - ca) < FLOOR

    prim = interval([joint('equalised', s) - joint('dual', s) for s in eq_seeds]) if len(eq_seeds) >= 2 else None
    null = interval([joint('sham', s) - joint('dual', s) for s in eq_seeds]) if len(eq_seeds) >= 2 else None
    pre_b = {
        'There is something to equalise: N2a supported (R_sub ≥ 2 in at least 4 of 5 dual models)': n_large >= NEEDED and 'Supported' in v2a,
        'The pilot found a λ that equalises': lam is not None,
        'The equalising took hold: |log R_sub| ≤ log 1.25 in at least 4 of 5 equalised models':
            sum(equalised_ok(s) for s in SEEDS if sub[('equalised', s)]) >= NEEDED,
        'Null contrast: the sham arm\'s D_joint interval lies within ±0.05': null is not None and -MARGIN <= null[2] and null[3] <= MARGIN,
    }
    if lam is None and 'lambda' in summary:
        v2b = '**Untestable: no λ in the grid equalised the pilot.**'
    elif len(eq_seeds) < 5:
        v2b = f'**Not reached: all arms have records for only {len(eq_seeds)} of 5 seeds.**'
    elif not all(pre_b.values()):
        failed = [k.split(':')[0] for k, ok in pre_b.items() if not ok]
        v2b = f'**Untestable: a precondition failed ({"; ".join(failed)}).**'
    else:
        m, sd, lo, hi = prim
        if lo > MARGIN:
            v2b = f'**Supported: the regimes are needed when readings coexist.** D_joint = {m:+.4f}, 95% interval [{lo:+.4f}, {hi:+.4f}].'
        elif hi < MARGIN and sd <= SD_SENSITIVE:
            v2b = f'**Refuted: not needed.** D_joint = {m:+.4f}, 95% interval [{lo:+.4f}, {hi:+.4f}].'
        elif lo > 0 or hi < 0:
            v2b = f'**Some effect, size unclear.** D_joint = {m:+.4f}, 95% interval [{lo:+.4f}, {hi:+.4f}].'
        else:
            v2b = f'**Inconclusive.** D_joint = {m:+.4f}, 95% interval [{lo:+.4f}, {hi:+.4f}].'

    L = ['# N2 results', '',
         'Generated by `n2_analyze.py` from the records in `results/n2/`. Preregistration: `N2_PREREGISTRATION.md`, with amendment A1.',
         f"Subspaces estimated on validation windows {summary.get('windows', {}).get('estimate')}; contraction measured on "
         f"{summary.get('windows', {}).get('measure')}.", '',
         '## Verdicts', '', f'**N2a (existence):** {v2a}', '', f'**N2b (necessity):** {v2b}', '',
         '## Preconditions', '', '| Precondition | Met |', '|---|---|']
    L += [f'| {k} | {"yes" if ok else "**no**"} |' for k, ok in {**pre1, **pre_a, **pre_b}.items()]

    L += ['', f'## N2a: contraction per dimension within each reading\'s {R}-dimensional subspace', '',
          f'R_sub = c̄_B / c̄_A (B: token after next; A: next token). Rates below {FLOOR} per dimension are scored as near zero (amendment A1).', '',
          '| Seed | Arm | c̄_A | c̄_B | R_sub | Scored as | Odd − even log R | Overlap of A and B |', '|---|---|---|---|---|---|---|---|']
    for a in ('dual', 'same'):
        for s in SEEDS:
            if sub[(a, s)]:
                ca, cb = rates(sub[(a, s)])
                _, r, cat = scored[(a, s)]
                sp = split.get(s, (None,))[0] if a == 'dual' else None
                L.append(f"| {s} | {a} | {ca:.5f} | {cb:.5f} | {f(r, 3)} | {cat} | {f(sp, 4)} | {sub[(a, s)]['overlap_AB'][str(R)]:.3f} |")

    L += ['', '## N2b: equalising the two rates from step 1,440', '', f'λ chosen by the pilot: {lam}.', '',
          '| λ (pilot) | c̄_A | c̄_B |', '|---|---|---|']
    for k, v in (summary.get('pilot') or {}).items():
        L.append(f"| {k} | {v['cA']:.5f} | {v['cB']:.5f} |")
    L += ['', '| Seed | Joint loss, dual | Joint loss, equalised | Joint loss, sham | D_joint | D_sham | Equalised c̄_A | Equalised c̄_B |',
          '|---|---|---|---|---|---|---|---|']
    for s in eq_seeds:
        ca, cb = rates(sub[('equalised', s)]) if sub[('equalised', s)] else (float('nan'), float('nan'))
        L.append(f"| {s} | {joint('dual', s):.4f} | {joint('equalised', s):.4f} | {joint('sham', s):.4f} | "
                 f"{joint('equalised', s) - joint('dual', s):+.4f} | {joint('sham', s) - joint('dual', s):+.4f} | {ca:.5f} | {cb:.5f} |")
    for name, iv in (('D_joint (primary)', prim), ('D_sham (null contrast)', null)):
        if iv:
            L.append(f'\n{name}: mean {iv[0]:+.4f}, standard deviation {iv[1]:.4f}, 95% interval [{iv[2]:+.4f}, {iv[3]:+.4f}].')

    # ---------------------------------------------------------------- secondary
    L += ['', '## Secondary and exploratory measures', '', 'None of these enters a decision rule.', '',
          '### S2: each reading\'s loss change in N2b (equalised − dual, 95% intervals)', '', '| Reading | Mean | 95% interval |', '|---|---|---|']
    for key, label in (('final_loss_A', 'A (next token)'), ('final_loss_B', 'B (token after next)')):
        d = [trn[('equalised', s)][key] - trn[('dual', s)][key] for s in eq_seeds]
        if len(d) >= 2:
            m, sd, lo, hi = interval(d)
            L.append(f'| {label} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] |')
    L += ['', '### S3, S4, S5: 32 dimensions, whole-state contraction, subspace drift', '',
          '| Seed | Arm | R_sub at 32 dims | Whole-state contraction | Drift: overlap of A, mid vs end | Drift: overlap of B, mid vs end |',
          '|---|---|---|---|---|---|']
    for a in ('dual', 'same'):
        for s in SEEDS:
            r = sub[(a, s)]
            if not r:
                continue
            _, r32, _ = score(*rates(r, 'A32', 'B32'))
            whole = float(np.mean(r['contraction']['whole']))
            dA = dB = None
            if a == 'dual' and mid[s]:
                dA = overlap(mid[s]['subspaces'][f'A{R}'], r['subspaces'][f'A{R}'])
                dB = overlap(mid[s]['subspaces'][f'B{R}'], r['subspaces'][f'B{R}'])
            L.append(f'| {s} | {a} | {f(r32, 3)} | {whole:+.4f} | {f(dA, 3)} | {f(dB, 3)} |')
    d6 = [trn[('dual', s)]['final_loss_A'] - trn[('same', s)]['final_loss_A'] for s in SEEDS if trn[('dual', s)] and trn[('same', s)]]
    if len(d6) >= 2:
        m, sd, lo, hi = interval(d6)
        L += ['', f'S6, the cost of reading twice: head A (next token), dual − dual-same: {m:+.4f}, 95% interval [{lo:+.4f}, {hi:+.4f}].']
    fp = [(s, trn[('dual', s)].get('batch_fingerprint_at_mid'), trn[('equalised', s)].get('batch_fingerprint_at_mid'),
           trn[('sham', s)].get('batch_fingerprint_at_mid')) for s in eq_seeds]
    L += ['', '## Checks', '', 'Same batch at step 1,440 in dual, equalised and sham: ' +
          ', '.join(f'seed {s} {"yes" if a == b == c and a is not None else "NO"}' for s, a, b, c in fp) + '.']
    pj = checks.get('projected', {})
    if pj:
        L.append(f"\nProjected estimator: dense error {pj['dense_relative_error']:.2%}, diagonal error {pj['diagonal_worst_relative_error']:.1e}, "
                 f"gradient error {pj['gradient_relative_error']:.2%}.")
    L.append('')
    OUT.write_text('\n'.join(L))
    print('\n'.join(L[:20]))


if __name__ == '__main__':
    main()
