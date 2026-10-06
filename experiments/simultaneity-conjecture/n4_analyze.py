"""Apply N4's preregistered decision rules to the saved records and write N4_RESULTS.md.

Written and committed before any real N4 run. Every number in the report comes from the
JSON records in the results folder.

    python n4_analyze.py [results folder] [report path]   # default: results/n4 here, N4_RESULTS.md
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

from scipy import stats

HERE = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'results' / 'n4'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'N4_RESULTS.md'

SEEDS = [3501, 3502, 3503, 3504, 3505]
ARMS = ('dual-sep', 'dual-tied', 'same-sep', 'same-tied')
NEEDED = 4
MARGIN, SD_SENSITIVE = 0.05, 0.04
SEPARATE_MAX, SAME_MIN = 0.8, 0.95
NO_BIND = 0.01                       # S3: amendment A1


def load(name):
    p = RES / f'{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


def interval(diffs, level=0.95):
    n = len(diffs)
    m = st.mean(diffs)
    sd = st.stdev(diffs) if n > 1 else float('nan')
    half = stats.t.ppf(1 - (1 - level) / 2, n - 1) * sd / math.sqrt(n) if n > 1 else float('nan')
    return m, sd, m - half, m + half


def fmt_iv(iv):
    return f'{iv[0]:+.4f} (standard deviation {iv[1]:.4f}), 95% interval [{iv[2]:+.4f}, {iv[3]:+.4f}]'


def main():
    summary = load('n4_summary') or {}
    checks = summary.get('checks', {})
    trn = {(a, s): load(f'{a}_seed{s}') for a in ARMS for s in SEEDS}
    sub = {(a, s): load(f'sub_{a}_seed{s}') for a in ARMS for s in SEEDS}
    seeds = [s for s in SEEDS if all(trn[(a, s)] for a in ARMS)]

    def J(a, s):
        return trn[(a, s)]['final_loss_A'] + trn[(a, s)]['final_loss_B']

    pre = {
        '1. Must match: separate and tied give bit-identical logits at initialisation, every seed':
            bool(checks.get('must_match')) and all(x['pass'] for x in checks['must_match']),
        '2. Batch and same-model checks':
            bool(checks.get('batch', {}).get('targets_are_i+1_and_i+2')) and all(x['pass'] for x in checks.get('same_model', [{'pass': False}])),
        '2b. Same data order in all four arms (batch fingerprint at step 0)':
            bool(seeds) and all(len({trn[(a, s)].get('batch_fingerprint_step0') for a in ARMS}) == 1 for s in seeds),
    }

    # ---------------------------------------------------------------- N4b
    D = [(J('dual-tied', s) - J('dual-sep', s)) - (J('same-tied', s) - J('same-sep', s)) for s in seeds]
    cost_dual = [J('dual-tied', s) - J('dual-sep', s) for s in seeds]
    cost_same = [J('same-tied', s) - J('same-sep', s) for s in seeds]
    prim = interval(D) if len(seeds) >= 2 else None
    iv_dual = interval(cost_dual) if len(seeds) >= 2 else None
    iv_same = interval(cost_same) if len(seeds) >= 2 else None
    no_bind = bool(iv_dual and iv_same and all(-NO_BIND <= iv[2] and iv[3] <= NO_BIND for iv in (iv_dual, iv_same)))
    if len(seeds) < 5:
        v4b = f'**Not reached: all four arms have records for only {len(seeds)} of 5 seeds.**'
    elif not all(pre.values()):
        v4b = f'**Untestable: a precondition failed ({"; ".join(k.split(":")[0] for k, ok in pre.items() if not ok)}).**'
    else:
        m, sd, lo, hi = prim
        tail = f'D = {m:+.4f}, 95% interval [{lo:+.4f}, {hi:+.4f}], standard deviation {sd:.4f}.'
        if lo > MARGIN:
            v4b = f'**Supported: the readings need separate directions.** {tail}'
        elif hi < MARGIN and sd <= SD_SENSITIVE:
            v4b = f'**Refuted.** {tail}'
        elif lo > 0 or hi < 0:
            v4b = f'**Some effect, size unclear.** {tail}'
        else:
            v4b = f'**Inconclusive.** {tail}'
        if no_bind:
            v4b = (f'**Untestable at this width: the restriction did not bind** (S3: both pairs\' costs of tying lie within '
                   f'±{NO_BIND}; amendment A1). {tail}')

    # ---------------------------------------------------------------- N4a
    ov = {(a, s): sub[(a, s)]['overlap_AB'] for a in ARMS for s in SEEDS if sub[(a, s)]}
    sep_seeds = [s for s in SEEDS if ('dual-sep', s) in ov and ('same-sep', s) in ov]
    must_fail = sum(ov[('same-sep', s)] >= SAME_MIN for s in sep_seeds) >= NEEDED
    n_apart = sum(ov[('dual-sep', s)] <= SEPARATE_MAX for s in sep_seeds)
    n_together = sum(ov[('dual-sep', s)] >= SAME_MIN for s in sep_seeds)
    pre_a = {f'Must fail: same-sep overlap ≥ {SAME_MIN} in at least 4 of 5': must_fail}
    if len(sep_seeds) < 5:
        v4a = f'**Not reached: {len(sep_seeds)} of 5 seeds measured in both separate arms.**'
    elif not (pre['1. Must match: separate and tied give bit-identical logits at initialisation, every seed']
              and pre['2. Batch and same-model checks'] and must_fail):
        failed = [k.split(':')[0] for k, ok in {**pre, **pre_a}.items() if not ok]
        v4a = f'**Untestable: a precondition failed ({"; ".join(failed)}).**'
    elif n_apart >= NEEDED:
        v4a = f'**Supported: the separation replicates.** Dual-sep overlap ≤ {SEPARATE_MAX} in {n_apart} of 5.'
    elif n_together >= NEEDED:
        v4a = f'**Refuted.** Dual-sep overlap ≥ {SAME_MIN} in {n_together} of 5.'
    else:
        v4a = f'**Intermediate.** Dual-sep overlap ≤ {SEPARATE_MAX} in {n_apart} and ≥ {SAME_MIN} in {n_together} of 5.'

    L = ['# N4 results', '',
         'Generated by `n4_analyze.py` from the records in `results/n4/`. Preregistration: `N4_PREREGISTRATION.md`.',
         f"Readout rank {summary.get('readout_rank')}; N4a subspaces of dimension {summary.get('subspace_rank')}, "
         f"estimated on validation windows {summary.get('windows', {}).get('measure')}.", '',
         '## Verdicts', '', f'**N4b (primary, the readings need separate directions):** {v4b}', '',
         f'**N4a (the separation replicates):** {v4a}', '',
         '## Preconditions', '', '| Precondition | Met |', '|---|---|']
    L += [f'| {k} | {"yes" if ok else "**no**"} |' for k, ok in {**pre, **pre_a}.items()]

    L += ['', '## N4b: combined validation loss J = L_A + L_B, whole validation split', '',
          '| Seed | dual-sep | dual-tied | same-sep | same-tied | Cost of tying, dual | Cost of tying, same | D |',
          '|---|---|---|---|---|---|---|---|']
    for i, s in enumerate(seeds):
        L.append(f"| {s} | {J('dual-sep', s):.4f} | {J('dual-tied', s):.4f} | {J('same-sep', s):.4f} | {J('same-tied', s):.4f} | "
                 f'{cost_dual[i]:+.4f} | {cost_same[i]:+.4f} | {D[i]:+.4f} |')
    if prim:
        L += ['', f'D (primary): {fmt_iv(prim)}.']

    L += ['', '## N4a: overlap of the two readings\' subspaces at the ODE block\'s output', '',
          'Mean squared cosine of the principal angles; 1 is identical. Odd and even halves of the windows are shown as a stability check (not a decision rule).', '',
          '| Seed | Arm | Overlap of A and B | Odd half | Even half | Stability of A, odd vs even | Stability of B |', '|---|---|---|---|---|---|---|']
    for a in ARMS:
        for s in SEEDS:
            r = sub[(a, s)]
            if r:
                L.append(f"| {s} | {a} | {r['overlap_AB']:.3f} | {r['overlap_AB_odd']:.3f} | {r['overlap_AB_even']:.3f} | "
                         f"{r['stability_A_odd_even']:.3f} | {r['stability_B_odd_even']:.3f} |")

    L += ['', '## Secondary and exploratory measures', '', 'None of these enters a decision rule except S3\'s note on N4b.', '',
          '### S1: which reading pays when directions are shared (tied − separate, 95% intervals)', '',
          '| Pair | Reading | Mean | 95% interval |', '|---|---|---|---|']
    for t in ('dual', 'same'):
        for key, label in (('final_loss_A', 'A (next token)'), ('final_loss_B', 'B (token after next)' if t == 'dual' else 'B (next token)')):
            d = [trn[(f'{t}-tied', s)][key] - trn[(f'{t}-sep', s)][key] for s in seeds]
            if len(d) >= 2:
                m, sd, lo, hi = interval(d)
                L.append(f'| {t} | {label} | {m:+.4f} | [{lo:+.4f}, {hi:+.4f}] |')
    L += ['', '### S2: overlap of the readouts\' row spaces, separate arms', '',
          'The overlap of the row spaces of P_A and P_B (1 at initialisation, since P_B starts as a copy of P_A).', '',
          '| Seed | dual-sep | same-sep |', '|---|---|---|']
    for s in SEEDS:
        if sub[('dual-sep', s)] or sub[('same-sep', s)]:
            L.append(f"| {s} | {sub[('dual-sep', s)]['readout_overlap'] if sub[('dual-sep', s)] else float('nan'):.3f} | "
                     f"{sub[('same-sep', s)]['readout_overlap'] if sub[('same-sep', s)] else float('nan'):.3f} |")
    L += ['', f'### S3: does the restriction bind? (amendment A1: not binding if both intervals lie within ±{NO_BIND})', '']
    for name, iv in (('Cost of tying, dual pair (J_dual-tied − J_dual-sep)', iv_dual), ('Cost of tying, same pair (J_same-tied − J_same-sep)', iv_same)):
        if iv:
            L.append(f'- {name}: {fmt_iv(iv)}.')
    if iv_dual:
        L.append(f'- The restriction {"did not bind" if no_bind else "bound"} by this rule.')
    n2 = [json.loads(p.read_text()) for p in sorted((HERE / 'results' / 'n2').glob('dual_seed*.json'))]
    n2s = [json.loads(p.read_text()) for p in sorted((HERE / 'results' / 'n2').glob('same_seed*.json'))]
    if n2 and seeds:
        def mean(xs):
            return sum(xs) / len(xs)
        L += ['', '### S4: against N2\'s unrestricted models (different seeds; descriptive only)', '',
              '| Models | Mean L_A | Mean L_B |', '|---|---|---|',
              f"| N2 dual, full-rank heads | {mean([r['final_loss_A'] for r in n2]):.4f} | {mean([r['final_loss_B'] for r in n2]):.4f} |"]
        if n2s:
            L.append(f"| N2 dual-same, full-rank heads | {mean([r['final_loss_A'] for r in n2s]):.4f} | {mean([r['final_loss_B'] for r in n2s]):.4f} |")
        for a in ARMS:
            L.append(f"| N4 {a} | {mean([trn[(a, s)]['final_loss_A'] for s in seeds]):.4f} | {mean([trn[(a, s)]['final_loss_B'] for s in seeds]):.4f} |")
    mm = checks.get('must_match', [])
    if mm:
        L += ['', '## Checks', '', 'Must-match at initialisation: ' + ', '.join(f"seed {x['seed']} {'yes' if x['pass'] else 'NO'}" for x in mm) + '.']
    L.append('')
    OUT.write_text('\n'.join(L))
    print('\n'.join(L[:12]))


if __name__ == '__main__':
    main()
