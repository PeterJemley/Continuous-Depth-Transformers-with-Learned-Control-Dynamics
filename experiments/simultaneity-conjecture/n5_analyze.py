"""Apply N5's preregistered decision rules to the saved records and write N5_RESULTS.md.

Written and committed before any real N5 run. Every number in the report comes from the
JSON records in the results folders (N5's, and N4's for the width-64 point of the curve).

    python n5_analyze.py [N5 results folder] [report path] [N4 results folder]
    # defaults: results/n5, N5_RESULTS.md, results/n4 here
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

from scipy import stats

HERE = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'results' / 'n5'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'N5_RESULTS.md'
RES4 = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / 'results' / 'n4'

SEEDS = [3501, 3502, 3503, 3504, 3505]
ARMS = ('dual-sep', 'dual-tied', 'same-sep', 'same-tied')
FULL, NARROW, WIDE = 256, 32, 128          # read from the summary if it says otherwise (test mode)
MARGIN, VANISH, SD_SENSITIVE = 0.05, 0.02, 0.04
LEVEL = 0.975                              # Bonferroni over the family {N5a, N5b}


def load(folder, name):
    p = folder / f'{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


def interval(xs, level=LEVEL):
    n = len(xs)
    m = st.mean(xs)
    sd = st.stdev(xs) if n > 1 else float('nan')
    half = stats.t.ppf(1 - (1 - level) / 2, n - 1) * sd / math.sqrt(n) if n > 1 else float('nan')
    return m, sd, m - half, m + half


def fmt(iv, level=LEVEL):
    return f'{iv[0]:+.4f} (standard deviation {iv[1]:.4f}), {level:.1%} interval [{iv[2]:+.4f}, {iv[3]:+.4f}]'


def main():
    global FULL, NARROW, WIDE
    summary = load(RES, 'n5_summary') or {}
    if summary.get('widths'):
        FULL, NARROW, WIDE = summary['widths']
    checks = summary.get('checks', {})
    widths = sorted({FULL, NARROW, WIDE})
    trn = {(w, a, s): load(RES, f'w{w}_{a}_seed{s}') for w in widths for a in ARMS for s in SEEDS}
    sub = {(w, a, s): load(RES, f'sub_w{w}_{a}_seed{s}') for w in widths for a in ARMS for s in SEEDS}
    for a in ARMS:                                       # N4's runs: width 64, exploratory only
        for s in SEEDS:
            trn[(64, a, s)] = load(RES4, f'{a}_seed{s}')
            sub[(64, a, s)] = load(RES4, f'sub_{a}_seed{s}')
    all_w = sorted({w for (w, _, _) in trn})

    def J(w, a, s):
        return trn[(w, a, s)]['final_loss_A'] + trn[(w, a, s)]['final_loss_B']

    def seeds_at(w):
        return [s for s in SEEDS if all(trn[(w, a, s)] for a in ARMS)]

    def Dw(w, s):
        return (J(w, 'dual-tied', s) - J(w, 'dual-sep', s)) - (J(w, 'same-tied', s) - J(w, 'same-sep', s))

    def same_data(w):
        return all(len({trn[(w, a, s)].get('batch_fingerprint_step0') for a in ARMS}) == 1 for s in seeds_at(w))

    pre = {
        '1. Must match: separate and tied give bit-identical logits at initialisation, every width and seed':
            bool(checks.get('must_match')) and all(x['pass'] for x in checks['must_match']),
        '2. Batch and same-model checks, and the same batches in all four arms of each width and seed':
            bool(checks.get('batch', {}).get('targets_are_i+1_and_i+2')) and all(x['pass'] for x in checks.get('same_model', [{'pass': False}]))
            and all(same_data(w) for w in (FULL, NARROW, WIDE) if seeds_at(w)),
        f'3. The width-{FULL} readout is full rank at initialisation, every seed':
            bool(checks.get('full_rank')) and all(x['pass'] for x in checks['full_rank']),
    }
    pre_ok = all(pre.values())
    failed = '; '.join(k.split(':')[0] for k, ok in pre.items() if not ok)

    # ---------------------------------------------------------------- N5a
    s_full = seeds_at(FULL)
    iv_a = interval([Dw(FULL, s) for s in s_full]) if len(s_full) >= 2 else None
    if len(s_full) < 5:
        v5a = f'**Not reached: all four arms at width {FULL} have records for only {len(s_full)} of 5 seeds.**'
    elif not pre_ok:
        v5a = f'**Untestable: a precondition failed ({failed}).**'
    else:
        m, sd, lo, hi = iv_a
        tail = f'D({FULL}) = {m:+.4f}, 97.5% interval [{lo:+.4f}, {hi:+.4f}], standard deviation {sd:.4f}.'
        if hi < VANISH:
            v5a = f'**Supported: without competition, the cost vanishes.** {tail}'
        elif lo > MARGIN:
            v5a = f'**Refuted: the cost persists without any restriction.** {tail}'
        elif lo > 0 and hi < MARGIN:
            v5a = f'**Partly: a smaller cost remains.** {tail}'
        else:
            v5a = f'**Inconclusive.** {tail}'

    # ---------------------------------------------------------------- N5b
    s_b = [s for s in SEEDS if s in seeds_at(NARROW) and s in seeds_at(WIDE)]
    iv_b = interval([Dw(NARROW, s) - Dw(WIDE, s) for s in s_b]) if len(s_b) >= 2 else None
    if len(s_b) < 5:
        v5b = f'**Not reached: widths {NARROW} and {WIDE} both have records for only {len(s_b)} of 5 seeds.**'
    elif not pre_ok:
        v5b = f'**Untestable: a precondition failed ({failed}).**'
    else:
        m, sd, lo, hi = iv_b
        tail = f'D({NARROW}) − D({WIDE}) = {m:+.4f}, 97.5% interval [{lo:+.4f}, {hi:+.4f}], standard deviation {sd:.4f}.'
        if lo > MARGIN:
            v5b = f'**Supported: tightening the competition raises the cost.** {tail}'
        elif hi < MARGIN and sd <= SD_SENSITIVE:
            v5b = f'**Refuted.** {tail}'
        elif lo > 0 or hi < 0:
            v5b = f'**Some effect, size unclear.** {tail}'
        else:
            v5b = f'**Inconclusive.** {tail}'

    L = ['# N5 results', '',
         'Generated by `n5_analyze.py` from the records in `results/n5/`, and N4\'s records in `results/n4/` for width 64 (exploratory only). '
         'Preregistration: `N5_PREREGISTRATION.md`.', '',
         '## Verdicts', '', f'**N5a (primary: without competition, the cost vanishes):** {v5a}', '',
         f'**N5b (tightening the competition raises the cost):** {v5b}', '',
         'Both use Bonferroni-corrected 97.5% t-intervals over 5 seeds.', '',
         '## Preconditions', '', '| Precondition | Met |', '|---|---|']
    L += [f'| {k} | {"yes" if ok else "**no**"} |' for k, ok in pre.items()]
    fr = checks.get('full_rank', [])
    if fr:
        L.append('\nSmallest over largest singular value of the initial width-' + str(FULL) + ' readout: '
                 + ', '.join(f"seed {x['seed']} {x['smallest_over_largest_singular_value']:.4f}" for x in fr) + '.')

    L += ['', '## D by width and seed', '',
          'D(w) = (J_dual-tied − J_dual-sep) − (J_same-tied − J_same-sep), J = L_A + L_B on the whole validation split. '
          'Width 64 is N4\'s runs on the same seeds.', '',
          '| Seed | ' + ' | '.join(f'D({w})' for w in all_w) + f' | D({NARROW}) − D({WIDE}) |',
          '|---|' + '---|' * (len(all_w) + 1)]
    for s in SEEDS:
        cells = [f'{Dw(w, s):+.4f}' if s in seeds_at(w) else 'n/a' for w in all_w]
        diff = f'{Dw(NARROW, s) - Dw(WIDE, s):+.4f}' if s in s_b else 'n/a'
        L.append(f'| {s} | ' + ' | '.join(cells) + f' | {diff} |')

    L += ['', '## Secondary and exploratory measures', '', 'None of these enters a decision rule.', '',
          '### S1: the whole curve (95% intervals)', '', '| Width | D | 95% interval | Cost of tying, dual | Cost of tying, same |', '|---|---|---|---|---|']
    for w in all_w:
        ss = seeds_at(w)
        if len(ss) >= 2:
            d = interval([Dw(w, s) for s in ss], 0.95)
            cd = interval([J(w, 'dual-tied', s) - J(w, 'dual-sep', s) for s in ss], 0.95)
            cs = interval([J(w, 'same-tied', s) - J(w, 'same-sep', s) for s in ss], 0.95)
            L.append(f'| {w} | {d[0]:+.4f} | [{d[2]:+.4f}, {d[3]:+.4f}] | {cd[0]:+.4f} [{cd[2]:+.4f}, {cd[3]:+.4f}] | '
                     f'{cs[0]:+.4f} [{cs[2]:+.4f}, {cs[3]:+.4f}] |')

    L += ['', '### S2: how many directions one reading needs (same-sep next-token loss, mean over seeds)', '',
          '| Width | Same-sep L_A | Dual-sep L_A | Dual-sep L_B |', '|---|---|---|---|']
    for w in all_w:
        ss = seeds_at(w)
        if ss:
            L.append(f"| {w} | {st.mean(trn[(w, 'same-sep', s)]['final_loss_A'] for s in ss):.4f} | "
                     f"{st.mean(trn[(w, 'dual-sep', s)]['final_loss_A'] for s in ss):.4f} | {st.mean(trn[(w, 'dual-sep', s)]['final_loss_B'] for s in ss):.4f} |")

    L += ['', '### S3: which reading pays when directions are shared (dual pair, tied − separate, 95% intervals)', '',
          '| Width | Reading A (next token) | Reading B (token after next) |', '|---|---|---|']
    for w in all_w:
        ss = seeds_at(w)
        if len(ss) >= 2:
            row = []
            for key in ('final_loss_A', 'final_loss_B'):
                m, _, lo, hi = interval([trn[(w, 'dual-tied', s)][key] - trn[(w, 'dual-sep', s)][key] for s in ss], 0.95)
                row.append(f'{m:+.4f} [{lo:+.4f}, {hi:+.4f}]')
            L.append(f'| {w} | ' + ' | '.join(row) + ' |')

    L += ['', '### S4: overlap of the two readings\' subspaces at the ODE block\'s output (mean over seeds)', '',
          'Untrained values are from the job\'s own measurement on the same seeds and windows (N4\'s untrained values are in `results/n4_exploratory/`).', '',
          '| Width | Untrained, dual | Untrained, same | dual-sep | dual-tied | same-sep | same-tied | Readout overlap, dual-sep | Readout overlap, same-sep |',
          '|---|---|---|---|---|---|---|---|---|']
    unt = summary.get('untrained_overlap', {})
    unt4 = load(HERE / 'results' / 'n4_exploratory', 'untrained_overlap') or {}
    for w in all_w:
        def mean_ov(a, key='overlap_AB'):
            v = [sub[(w, a, s)][key] for s in SEEDS if sub[(w, a, s)]]
            return f'{st.mean(v):.3f}' if v else 'n/a'
        if w == 64:
            ud = [unt4.get(f'dual-sep_seed{s}') for s in SEEDS if f'dual-sep_seed{s}' in unt4]
            us = [unt4.get(f'same-sep_seed{s}') for s in SEEDS if f'same-sep_seed{s}' in unt4]
        else:
            ud = [unt[f'w{w}_seed{s}']['dual-sep'] for s in SEEDS if f'w{w}_seed{s}' in unt]
            us = [unt[f'w{w}_seed{s}']['same-sep'] for s in SEEDS if f'w{w}_seed{s}' in unt]
        L.append(f"| {w} | {f'{st.mean(ud):.3f}' if ud else 'n/a'} | {f'{st.mean(us):.3f}' if us else 'n/a'} | "
                 f"{mean_ov('dual-sep')} | {mean_ov('dual-tied')} | {mean_ov('same-sep')} | {mean_ov('same-tied')} | "
                 f"{mean_ov('dual-sep', 'readout_overlap')} | {mean_ov('same-sep', 'readout_overlap')} |")
    for name, iv in (('N5a, D(' + str(FULL) + ')', iv_a), (f'N5b, D({NARROW}) − D({WIDE})', iv_b)):
        if iv:
            L.append(f'\n{name}: {fmt(iv)}.')
    L.append('')
    OUT.write_text('\n'.join(L))
    print('\n'.join(L[:14]))


if __name__ == '__main__':
    main()
