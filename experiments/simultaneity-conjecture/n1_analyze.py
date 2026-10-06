"""Apply N1's preregistered decision rules to the saved records and write N1_RESULTS.md.

Written and committed before any real N1 run. Every number in the report comes from the
JSON records in the results folder.

    python n1_analyze.py [results folder] [report path]   # default: results/n1 here, N1_RESULTS.md
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
from g1c_analyze import score  # noqa: E402

RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'results' / 'n1'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'N1_RESULTS.md'

SEEDS = [3301, 3302, 3303, 3304, 3305]
ARMS = ['full', 'post', 'pre', 'none']
TASKS = [('next', 'next_uplus', 'next-token task (u = +1)'), ('farther', 'skip_uminus', 'farther task (u = −1)')]
NONE_KEY = {'next': 'next_uplus', 'farther': 'skip_uminus'}
MARGIN, SD_SENSITIVE, MANIP_MIN = 0.05, 0.04, 0.5
FAMILY_LEVEL = 1 - 0.05 / 2            # Bonferroni over the two N1b contrasts: 97.5%
NEEDED, OFFSET_HIGH, OFFSET_LOW, INJECTED_MAX = 4, 0.8, 0.5, 0.1


def load(name):
    p = RES / f'{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


def interval(diffs, level):
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
    summary = load('n1_summary') or {}
    checks = summary.get('checks', {})
    trn = {(a, s): load(f'train_{a}_seed{s}') for a in ARMS for s in SEEDS}
    off = {s: load(f'offset_full_seed{s}') for s in SEEDS}
    off0 = {s: load(f'offset_init_seed{s}') for s in SEEDS}
    con = {(a, s): load(f'contr_{a}_seed{s}') for a in ('full', 'pre') for s in SEEDS}
    complete = [s for s in SEEDS if all(trn[(a, s)] for a in ARMS)]

    def loss(a, s, key):
        return trn[(a, s)]['final_losses'][key]

    # ---------------------------------------------------------------- N1b
    contrasts = {}
    for name, key, _ in TASKS:
        diffs = [loss('post', s, key) - loss('full', s, key) for s in complete]
        contrasts[name] = interval(diffs, FAMILY_LEVEL) if len(diffs) >= 2 else None
    pos = interval([loss('none', s, 'skip_uminus') - loss('full', s, 'skip_uminus') for s in complete], FAMILY_LEVEL) \
        if len(complete) >= 2 else None
    manip = {s: loss('full', s, 'next_uminus') - loss('full', s, 'next_uplus') for s in SEEDS if trn[('full', s)]}
    mm = checks.get('must_match', [])
    pre = {
        '1. Must match by construction: with the control zeroed, all four arms give bit-identical logits, every seed':
            bool(mm) and all(x['pass'] for x in mm),
        '2. Positive control: none − full on the farther task, 97.5% interval wholly above +0.05':
            pos is not None and pos[2] > MARGIN,
        '3. The full models use u: wrong control costs at least 0.5 nats on the next-token task, in at least 4 of 5':
            sum(v >= MANIP_MIN for v in manip.values()) >= NEEDED,
    }
    if len(complete) < len(SEEDS):
        v1b = f'**Not reached: all four arms have records for only {len(complete)} of 5 seeds.**'
    elif not all(pre.values()):
        failed = [k.split(':')[0] for k, ok in pre.items() if not ok]
        v1b = f'**Untestable: precondition {", ".join(failed)} failed.** The contrasts below decide nothing.'
    else:
        iv = list(contrasts.values())
        if any(lo > MARGIN for _, _, lo, _ in iv):
            v1b = '**Refuted: the dynamics do work an offset cannot.** At least one interval lies wholly above +0.05.'
        elif all(hi < MARGIN for _, _, _, hi in iv) and all(sd <= SD_SENSITIVE for _, sd, _, _ in iv):
            v1b = '**Supported: an offset is enough.** Both upper bounds lie below +0.05.'
        elif any(lo > 0 for _, _, lo, _ in iv):
            v1b = '**Some effect, size unclear.** At least one interval excludes 0 on the positive side; neither rule is met.'
        else:
            v1b = '**Inconclusive.**'

    # ---------------------------------------------------------------- N1a
    have = [s for s in SEEDS if off[s]]
    inj_ok = bool(have) and all(off[s]['injected_multiplicative_fraction'] <= INJECTED_MAX for s in have)
    n_high = sum(off[s]['offset_fraction'] >= OFFSET_HIGH for s in have)
    n_low = sum(off[s]['offset_fraction'] <= OFFSET_LOW for s in have)
    if len(have) < len(SEEDS):
        v1a = f'**Not reached: offset measurements for only {len(have)} of 5 full models.**'
    elif not inj_ok:
        v1a = ('**Untestable: the measure could not fail.** An injected multiplicative control read above 0.1 in at least '
               'one trained full model, so a high offset fraction would not distinguish an offset from a scaling.')
    elif n_high >= NEEDED:
        v1a = f'**Supported: the trained effect is an offset.** Offset fraction ≥ 0.8 in {n_high} of 5 full models.'
    elif n_low >= NEEDED:
        v1a = f'**Refuted: training made the effect depend on the state.** Offset fraction ≤ 0.5 in {n_low} of 5 full models.'
    else:
        v1a = f'**Intermediate.** Offset fraction ≥ 0.8 in {n_high} and ≤ 0.5 in {n_low} of 5 full models.'

    L = ['# N1 results', '',
         'Generated by `n1_analyze.py` from the records in `results/n1/`. Preregistration: `N1_PREREGISTRATION.md`.',
         f"Offset and contraction measured on validation windows {summary.get('windows', ['?', '?'])[0]}–"
         f"{summary.get('windows', ['?', '?'])[1]}.", '',
         '## Verdicts', '', f'**N1b (primary):** {v1b}', '', f'**N1a (secondary hypothesis):** {v1a}', '',
         '## Preconditions for N1b', '', '| Precondition | Met |', '|---|---|']
    L += [f'| {k} | {"yes" if ok else "**no**"} |' for k, ok in pre.items()]
    if pos:
        L += ['', f'Positive control, none − full on the farther task: mean {pos[0]:+.4f}, 97.5% interval [{pos[2]:+.4f}, {pos[3]:+.4f}].']

    L += ['', '## N1b: post − full, by task (97.5% intervals, Bonferroni over 2)', '',
          '| Task | Mean | SD | 97.5% interval |', '|---|---|---|---|']
    for name, _, label in TASKS:
        iv = contrasts.get(name)
        if iv:
            L.append(f'| {label} | {iv[0]:+.4f} | {iv[1]:.4f} | [{iv[2]:+.4f}, {iv[3]:+.4f}] |')
    L += ['', '### Losses by seed and arm (whole validation split, right control)', '',
          '| Seed | Arm | Next token (u = +1) | Farther (u = −1) | Wrong-control cost, next | Final α | Hours |',
          '|---|---|---|---|---|---|---|']
    for s in SEEDS:
        for a in ARMS:
            r = trn[(a, s)]
            if r:
                fl = r['final_losses']
                L.append(f"| {s} | {a} | {fl['next_uplus']:.4f} | {fl['skip_uminus']:.4f} | "
                         f"{fl['next_uminus'] - fl['next_uplus']:+.3f} | {r['alpha_final']:.3f} | {r['train_seconds'] / 3600:.3f} |")

    L += ['', '## N1a: the control\'s effect on the block\'s output', '',
          'Offset fraction: the share of Δ = out(u = −1) − out(u = +1) that is one constant vector over all tokens. '
          'The injected check scales the output by 0.9 against 1.1 and must read at most 0.1.', '',
          '| Seed | Offset fraction (trained) | Offset fraction (untrained) | Top singular share | Δ size relative to output | Injected check |',
          '|---|---|---|---|---|---|']
    for s in SEEDS:
        o, o0 = off[s], off0[s]
        if o:
            L.append(f"| {s} | {o['offset_fraction']:.4f} | {f(o0['offset_fraction'], 4, False) if o0 else 'n/a'} | "
                     f"{o['top_singular_share']:.4f} | {o['relative_size']:.4f} | {o['injected_multiplicative_fraction']:.4f} |")

    # ---------------------------------------------------------------- secondary
    L += ['', '## Secondary and exploratory measures', '', 'None of these enters a decision rule.', '',
          '### S1: the pre arm (pre − full, 95% intervals)', '', '| Task | Mean | 95% interval |', '|---|---|---|']
    for name, key, label in TASKS:
        diffs = [loss('pre', s, key) - loss('full', s, key) for s in complete]
        if len(diffs) >= 2:
            m, sd, lo, hi = interval(diffs, 0.95)
            L.append(f'| {label} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] |')
    L += ['', '### S3: contraction (G1c\'s measure)', '', '| Seed | Arm | c̄(+1) | c̄(−1) | R | Scored as |', '|---|---|---|---|---|---|']
    for s in SEEDS:
        for a in ('full', 'pre'):
            r = con[(a, s)]
            if r:
                cp, cm = (float(np.mean(r['controls'][u]['c'])) for u in ('plus', 'minus'))
                _, R, cat = score(cp, cm)
                L.append(f'| {s} | {a} | {cp:.4f} | {cm:.4f} | {f(R, 3, False)} | {cat} |')

    L += ['', '## Checks', '']
    for x in mm:
        L.append(f"- Seed {x['seed']}: arms identical with the control zeroed: {x['identical_with_control_zeroed']}; "
                 f"arms differ otherwise: {x['differ_otherwise']}.")
    b = checks.get('batch', {})
    L.append(f"\nTargets at i + 1 and i + 2 confirmed on the first training batch: {b.get('targets_are_i+1_and_i+2')}.")
    sm = checks.get('same_model', [])
    L.append('\nExact and adjoint versions match before training: ' +
             ', '.join(f"seed {x['seed']} {'yes' if x['pass'] else 'NO'}" for x in sm) + '.\n')
    OUT.write_text('\n'.join(L))
    print('\n'.join(L[:24]))


if __name__ == '__main__':
    main()
