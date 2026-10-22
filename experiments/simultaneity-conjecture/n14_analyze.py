"""Apply N14's preregistered decision rules to the saved records and write N14_RESULTS.md.

Written and committed before any real N14 run. Every number in the report comes from the JSON records: N14's own
(results/n14: m_* for the new arms, r_* for N12's two ends re-measured, the training records, n14_summary.json) and N12's
committed records (results/n12), which precondition 2 compares the re-measurements with. For the calibration run (one
seed) it reports the same quantities and checks, with no verdicts.

a: the inner softmax's agreement with head A (for a single softmax, head B's). delta_A: the inner softmax's clean-target
loss minus head A's. The ladder: (delta_A, a) of sub20, sub10, sub05, sub025 in each seed; a_hat(x) interpolates a at
delta_A = x between the two ladder points that bracket x (the ladder has a fifth rung, sub0125, by amendment A1). b = a_hat - a, the
residual. D(d) = a(sub20) - a(d).
N14a: b(n80-mix) (interval rule). N14b: D(r10) against D(n80-mix) (count rule). N14c: a(sub20) - a(sub20-rw) (interval rule).

    python n14_analyze.py [results folder] [report path] [N12 records folder]
    # defaults: results/n14 here, N14_RESULTS.md, and results/n12 or results/n12_calib by the summary's mode
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

from scipy import stats

HERE = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'results' / 'n14'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'N14_RESULTS.md'

REAL_SEEDS = [4101, 4102, 4103, 4104, 4105]
DOSE = ('r10', 'r20', 'r40')
LADDER = ('sub10', 'sub05', 'sub025', 'sub0125')              # sub0125: amendment A1
CONTROL = 'sub20-rw'
NEW = DOSE + LADDER + (CONTROL,)
OLD = ('sub20', 'n80-mix')
RUNG = ('sub20',) + LADDER                            # the ladder, top to bottom
DOSES = ('sub20', 'r10', 'r20', 'r40', 'n80-mix')     # the dose, in order
MIX_ARMS = ('r10', 'r20', 'r40', 'n80-mix')
DESIGN = {'sub20': (0.2, 0.0), 'r10': (0.2, 0.1), 'r20': (0.2, 0.2), 'r40': (0.2, 0.4), 'n80-mix': (0.2, 0.8),
          'sub10': (0.1, 0.0), 'sub05': (0.05, 0.0), 'sub025': (0.025, 0.0), 'sub0125': (0.0125, 0.0), 'sub20-rw': (0.2, 0.0)}   # (clean fraction, dose)
WINDOW = 128 + 8
NEEDED, LEVEL = 4, 0.975
BEYOND, BEYOND_NONE, BEYOND_SD = 0.03, 0.01, 0.03       # N14a: the residual's margin, the refuting margin, the SD condition
HALF, THREE_QUARTERS = 0.5, 0.75                        # N14b
RW_NONE, RW_MOST, RW_SD = 0.03, 0.06, 0.03              # N14c
REPRO_LOSS, REPRO_RATIO, MATCH_MIN, SEPARATE, MIN_TRACE, FRAC_TOL = 1e-3, 1e-2, 0.999, 0.60, 0.02, 0.02


def load(folder, name):
    p = folder / f'{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


def interval(xs, level=LEVEL):
    n = len(xs)
    m = st.mean(xs)
    sd = st.stdev(xs) if n > 1 else float('nan')
    half = stats.t.ppf(1 - (1 - level) / 2, n - 1) * sd / math.sqrt(n) if n > 1 else float('nan')
    return m, sd, m - half, m + half


def fmt(xs, digits=3):
    """Mean with its 97.5% interval, or the value alone when there is one."""
    if not xs:
        return 'n/a'
    if len(xs) == 1:
        return f'{xs[0]:+.{digits}f}'
    m, sd, lo, hi = interval(xs)
    return f'{m:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]'


def interval_verdict(xs, support, refute, above, sd_max, supported_text, label):
    """above: the claim is that the interval lies above `support` (refutation below `refute`); else the reverse."""
    m, sd, lo, hi = interval(xs)
    tail = f'{label} = {m:+.4f}, 97.5% interval [{lo:+.4f}, {hi:+.4f}], standard deviation {sd:.4f}, {len(xs)} seeds.'
    if (lo > support) if above else (hi < support):
        return f'**Supported: {supported_text}** {tail}'
    if ((hi < refute) if above else (lo > refute)) and sd <= sd_max:
        return f'**Refuted.** {tail}'
    if lo > 0 or hi < 0:
        return f'**Some effect, size unclear.** {tail}'
    return f'**Inconclusive.** {tail}'


def main():
    summary = load(RES, 'n14_summary') or {}
    mode = summary.get('mode', 'real')
    n12_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / 'results' / ('n12_calib' if mode == 'calibration' else 'n12')
    want = summary.get('seeds', REAL_SEEDS) if mode == 'calibration' else REAL_SEEDS
    c = summary.get('checks', {})
    cut = summary.get('held_out', {}).get('cut_token')
    m = {(a, s): load(RES, f'm_{a}_seed{s}') for a in NEW for s in want}
    r = {(a, s): load(RES, f'r_{a}_seed{s}') for a in OLD for s in want}
    trn = {(a, s): load(RES, f'{a}_seed{s}') for a in NEW for s in want}
    ref = {(a, s): load(n12_dir, f'm_{a}_seed{s}') for a in OLD + ('clean',) for s in want}
    ref_trn = {s: load(n12_dir, f'n80_seed{s}') for s in want}
    seeds = [s for s in want if all(m[(a, s)] for a in NEW) and all(r[(a, s)] for a in OLD) and all(trn[(a, s)] for a in NEW)
             and all(ref[(a, s)] for a in OLD + ('clean',)) and ref_trn[s]]
    rec = lambda a, s: m[(a, s)] if a in NEW else r[(a, s)]

    def a_of(a, s):
        x = rec(a, s)
        return (x.get('B_inner_vs_A') or x['B_vs_A'])['agreement_ratio']

    def dA(a, s):
        f = rec(a, s)['functional']
        return f['B_inner_clean'] - f['A_clean']

    def ladder(s):
        return [(dA(a, s), a_of(a, s)) for a in RUNG]

    def ordered(s):
        xs = [p[0] for p in ladder(s)]
        return all(xs[i] < xs[i + 1] for i in range(len(xs) - 1))

    def bracketed(s, x):
        xs = [p[0] for p in ladder(s)]
        return min(xs) <= x <= max(xs)

    def a_hat(s, x):
        """Linear interpolation of a at delta_A = x on the seed's ladder (sorted by delta_A); None outside its range."""
        pts = sorted(ladder(s))
        if not (pts[0][0] <= x <= pts[-1][0]):
            return None
        for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
            if x0 <= x <= x1:
                return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
        return None

    def b(a, s):
        h = a_hat(s, dA(a, s))
        return None if h is None else h - a_of(a, s)

    def D(a, s):
        return a_of('sub20', s) - a_of(a, s)

    def loo(a, s):
        """Leave-one-out residual of an inner rung: the line through its two neighbours, evaluated at its delta_A, minus its a."""
        i = RUNG.index(a)
        (x0, y0), (x1, y1), (x2, y2) = ladder(s)[i - 1], ladder(s)[i], ladder(s)[i + 1]
        if x2 == x0:
            return None
        return y0 + (y2 - y0) * (x1 - x0) / (x2 - x0) - y1

    def ratio(x, key):
        return x[key]['agreement_ratio']

    nest = {n['seed']: n for n in c.get('nesting', [])} if c.get('nesting') else {}
    all_measured = [m[(a, s)] for a in NEW for s in seeds] + [r[(a, s)] for a in OLD for s in seeds]

    def run_ok(a, s):
        t, n80, n = trn[(a, s)], ref_trn[s], nest.get(s)
        fk, fd = DESIGN[a]
        return (t['training_tokens'] == cut and t['max_offset'] + WINDOW <= cut
                and t['batch_fingerprint_step0'] == n80['batch_fingerprint_step0']
                and t['replaced_positions_fingerprint_step0'] == n80['replaced_positions_fingerprint_step0']
                and t['keep_subset_of_K_step0'] and t['random_subset_of_R_step0']
                and abs(t['kept_fraction_step0'] - fk) <= FRAC_TOL and abs(t['random_fraction_step0'] - fd) <= FRAC_TOL
                and n is not None and n['fingerprints'].get(a) == [t['keep_fingerprint_step0'], t['random_fingerprint_step0']])

    n_needed = min(NEEDED, len(want))
    pre = {
        '1. Setup: batch, same-model, identical-heads and analytic checks; the loss identity at the dose\'s ends; the position sets nested, '
        'subsets of n80\'s K and R, at the design fractions, and the same in every run; windows; held-out tail untouched; batch fingerprints equal N12\'s n80\'s':
            bool(c.get('batch', {}).get('targets_are_i+1_to_i+8')) and bool(c.get('windows_shifted')) and bool(c.get('identical_heads', {}).get('pass'))
            and bool(c.get('analytic', {}).get('pass')) and bool(c.get('loss_identity', {}).get('pass'))
            and bool(c.get('nesting')) and all(n['pass'] for n in c.get('nesting', [])) and all(x['pass'] for x in c.get('same_model', [{'pass': False}]))
            and cut is not None and bool(seeds) and all(run_ok(a, s) for a in NEW for s in seeds),
        f'2. Must match: re-measuring N12\'s sub20 and n80-mix reproduces N12\'s records (losses within {REPRO_LOSS:g}, ratios within {REPRO_RATIO:g}); '
        f'the constructed mixture check, at 0.8 in every model and at the arm\'s own ρ in the mixture arms; the zero-noise copy (≥ {MATCH_MIN}) in every model':
            bool(seeds) and all(r[(a, s)].get('reproduction', {}).get('pass') for a in OLD for s in seeds)
            and all(x['constructed_mixture']['pass'] and ratio(x, 'copy0_vs_A') >= MATCH_MIN for x in all_measured)
            and all((rec(a, s)['at_rho']['constructed'] or {}).get('pass') for a in MIX_ARMS for s in seeds),
        f'3. Must differ: rotated control ≤ {SEPARATE} in at least {NEEDED} of 5, in every arm':
            bool(seeds) and all(sum(ratio(rec(a, s), 'rotated_control') <= SEPARATE for s in seeds) >= n_needed for a in NEW + OLD),
        f'4. The ladder brackets: Δ_A(n80-mix) within the ladder\'s range in at least {NEEDED} of 5 seeds':
            sum(bracketed(s, dA('n80-mix', s)) for s in seeds) >= n_needed,
        f'5. The ladder is ordered: Δ_A increases along sub20, sub10, sub05, sub025, sub0125 in at least {NEEDED} of 5 seeds':
            sum(ordered(s) for s in seeds) >= n_needed,
        f'6. The trace exists: D(n80-mix) ≥ {MIN_TRACE} in at least {NEEDED} of 5 seeds':
            sum(D('n80-mix', s) >= MIN_TRACE for s in seeds) >= n_needed,
    }
    names = list(pre)
    good = [s for s in seeds if bracketed(s, dA('n80-mix', s)) and ordered(s)]

    def gate(needed):
        if len(seeds) < 5:
            return f'**Not reached: complete records for only {len(seeds)} of 5 seeds.**'
        bad = [names[i].split(':')[0] for i in needed if not pre[names[i]]]
        return f'**Untestable: a precondition failed ({"; ".join(bad)}).**' if bad else None

    L = ['# N14 results' if mode == 'real' else '# N14 calibration run', '',
         f'Generated by `n14_analyze.py` from the records in `{RES.name}/`, compared with N12\'s in `{n12_dir.name}/`. Preregistration: `N14_PREREGISTRATION.md`.',
         f"Measured on the held-out tail's windows {summary.get('held_out', {}).get('measure')}, N12's. a is the inner softmax's agreement with head A "
         "(head B's own, for a single softmax); Δ_A is the inner softmax's clean-target loss minus head A's; â is a's linear interpolation at the model's Δ_A "
         'on the seed\'s clean-target ladder (sub20, sub10, sub05, sub025, sub0125); b = â − a; D(d) = a(sub20) − a(d). Intervals are 97.5% t-intervals.', '']
    if mode == 'real':
        v_a = gate([0, 1, 2, 3, 4])
        if not v_a and len(good) < NEEDED:
            v_a = f'**Untestable: the ladder brackets and orders n80-mix in only {len(good)} of 5 seeds.**'
        if not v_a:
            v_a = interval_verdict([b('n80-mix', s) for s in good], BEYOND, BEYOND_NONE, True, BEYOND_SD,
                                   'the trace of the random targets is beyond their functional cost.', 'b(n80-mix)')
        v_b = gate([0, 1, 2, 5])
        if not v_b:
            n_half = sum(D('r10', s) < HALF * D('n80-mix', s) for s in seeds)
            n_tq = sum(D('r10', s) >= THREE_QUARTERS * D('n80-mix', s) for s in seeds)
            if n_half >= NEEDED:
                v_b = f'**Supported: the trace grows with the dose.** D(r10) < D(n80-mix) / 2 in {n_half} of 5 seeds.'
            elif n_tq >= NEEDED:
                v_b = f'**Refuted.** D(r10) ≥ 0.75 · D(n80-mix) in {n_tq} of 5 seeds: the trace is complete at the smallest dose.'
            else:
                v_b = f'**Intermediate.** D(r10) < D(n80-mix) / 2 in {n_half} of 5 seeds; ≥ 0.75 · D(n80-mix) in {n_tq}.'
        v_c = gate([0, 1, 2]) or interval_verdict([a_of('sub20', s) - a_of(CONTROL, s) for s in seeds], RW_NONE, RW_MOST, False, RW_SD,
                                                     'the discount\'s reweighting of the kept targets is not the trace.', 'a(sub20) − a(sub20-rw)')
        L += ['## Verdicts', '',
              f'**N14a (primary: the trace is beyond function):** {v_a}', '',
              f'**N14b (the trace grows with the dose):** {v_b}', '',
              f'**N14c (the discount\'s reweighting of the kept targets is not the trace):** {v_c}', '']
    else:
        L += ['This is the calibration run of the preregistration\'s calibration step: one seed, on windows the real run does not use. '
              'It decides nothing, and no verdict is computed. Whether the ladder brackets n80-mix (precondition 4) is the check it exists for.', '']
    L += ['## Preconditions', '', '| Precondition | Met |', '|---|---|']
    L += [f'| {k} | {"yes" if ok else "**no**"} |' for k, ok in pre.items()]

    if seeds:
        def mean(f, over=None):
            xs = [f(s) for s in (over or seeds)]
            xs = [x for x in xs if x is not None]
            return st.mean(xs) if xs else float('nan')
        L += ['', f'The ladder brackets and orders n80-mix in {len(good)} of {len(seeds)} seeds: {good}.', '',
              '## By arm (means over seeds)', '',
              '| Arm | Clean fraction | Dose | Δ_A | a: agreement with head A | â: the ladder at the same Δ_A | b = â − a | Head A, minus N12\'s clean arm | Rotated | Agreement with the copy at the arm\'s ρ |',
              '|---|---|---|---|---|---|---|---|---|---|']
        for a in DOSES + LADDER + (CONTROL,):
            fk, fd = DESIGN[a]
            bs = [b(a, s) for s in seeds]
            bs = [x for x in bs if x is not None]
            hat = f'{mean(lambda s: a_hat(s, dA(a, s))):.3f}' if bs and a not in RUNG else ('(on the ladder)' if a in RUNG else 'n/a')
            bb = f'{st.mean(bs):+.3f}' if bs and a not in RUNG else ('0' if a in RUNG else 'n/a')
            L.append(f'| {a} | {fk} | {fd} | {mean(lambda s: dA(a, s)):+.3f} | {mean(lambda s: a_of(a, s)):.3f} | {hat} | {bb} | '
                     f"{mean(lambda s: rec(a, s)['functional']['A_clean'] - ref[('clean', s)]['functional']['A_clean']):+.4f} | "
                     f"{mean(lambda s: ratio(rec(a, s), 'rotated_control')):.3f} | {mean(lambda s: ratio(rec(a, s)['at_rho'], 'B_vs_copy')):.3f} |")
        L += ['', '## By seed', '',
              '| Seed | Δ_A(n80-mix) | a(sub20) | a(n80-mix) | D(n80-mix) | â(n80-mix) | b(n80-mix) | D(r10) | D(r10) / D(n80-mix) | a(sub20) − a(sub20-rw) | Bracketed, ordered |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
        for s in seeds:
            h, bb = a_hat(s, dA('n80-mix', s)), b('n80-mix', s)
            dn = D('n80-mix', s)
            L.append(f"| {s} | {dA('n80-mix', s):+.3f} | {a_of('sub20', s):.3f} | {a_of('n80-mix', s):.3f} | {dn:+.3f} | "
                     f"{'n/a' if h is None else f'{h:.3f}'} | {'n/a' if bb is None else f'{bb:+.3f}'} | {D('r10', s):+.3f} | "
                     f"{'n/a' if dn == 0 else f'{D(chr(114) + str(10), s) / dn:.2f}'} | {a_of('sub20', s) - a_of(CONTROL, s):+.3f} | "
                     f"{'yes' if bracketed(s, dA('n80-mix', s)) else 'no'}, {'yes' if ordered(s) else 'no'} |")
        L += ['', '## The clean-target ladder by seed: (Δ_A, a)', '',
              '| Seed | ' + ' | '.join(RUNG) + ' | Leave-one-out residuals, ' + ', '.join(RUNG[1:-1]) + ' |', '|---|' + '---|' * (len(RUNG) + 1)]
        for s in seeds:
            cells = [f'({x:+.3f}, {y:.3f})' for x, y in ladder(s)]
            los = [loo(a, s) for a in RUNG[1:-1]]
            L.append(f'| {s} | ' + ' | '.join(cells) + ' | ' + ', '.join('n/a' if v is None else f'{v:+.3f}' for v in los) + ' |')
        L += ['', '## Secondary and exploratory measures', '', 'None of these enters a decision rule.', '']
        res_lines = []
        for a in DOSE + (CONTROL,):
            xs = [b(a, s) for s in seeds]
            xs = [x for x in xs if x is not None]
            res_lines.append(f'{a} {fmt(xs)} ({len(xs)} seeds bracketed)')
        L.append('- S1, the residuals against the ladder at the other doses and for the control: ' + '; '.join(res_lines) + '.')
        los = {a: [x for x in (loo(a, s) for s in seeds) if x is not None] for a in RUNG[1:-1]}
        L.append('- S2, the ladder\'s own noise (leave-one-out residuals): ' + ', '.join(f'{a} {fmt(v)}' for a, v in los.items()) + '; the largest in magnitude '
                 f'{max((abs(x) for v in los.values() for x in v), default=float("nan")):.3f}.')
        n_order = sum(all(a_of(DOSES[i], s) >= a_of(DOSES[i + 1], s) for i in range(len(DOSES) - 1)) for s in seeds)
        L.append(f'- S3, the ordering of a along the dose (sub20 ≥ r10 ≥ r20 ≥ r40 ≥ n80-mix) holds in full in {n_order} of {len(seeds)} seeds; '
                 'D(d): ' + ', '.join(f'{a} {fmt([D(a, s) for s in seeds])}' for a in DOSES[1:]) + '.')
        L.append('- S4, Δ_A by dose: ' + ', '.join(f'{a} {fmt([dA(a, s) for s in seeds])}' for a in DOSES) + '; by ladder rung: '
                 + ', '.join(f'{a} {fmt([dA(a, s) for s in seeds])}' for a in RUNG) + '.')
        L.append('- S5, head A against N12\'s clean arm: ' + ', '.join(
            f"{a} {fmt([rec(a, s)['functional']['A_clean'] - ref[('clean', s)]['functional']['A_clean'] for s in seeds])}" for a in DOSES + LADDER + (CONTROL,)) + '.')
        L.append('- S7, agreement with the copy at the arm\'s ρ, beside the head A measure: ' + ', '.join(
            f"{a} {mean(lambda s: ratio(rec(a, s)['at_rho'], 'B_vs_copy')):.3f} / {mean(lambda s: a_of(a, s)):.3f}" for a in DOSES) + '.')
        steps = [0, 300, 1500, 2850] if mode == 'real' or len(trn[(NEW[0], seeds[0])]['r_every_50']) > 20 else [0, 20, 40]
        L += ['', '## S6: the discount over training (means over seeds)', '',
              'Mean r(y) on the kept and on the random positions, and the random targets\' share of the summed weight, at the recorded step nearest '
              'each listed step, at the arm\'s own ρ (0.8 for the arms without random targets, where it is recorded only).', '',
              '| Arm | ' + ' | '.join(f'step {x}: kept / random / share' for x in steps) + ' |', '|---|' + '---|' * len(steps)]
        for a in NEW:
            cells = []
            for x in steps:
                recs = [min(trn[(a, s)]['r_every_50'], key=lambda e: abs(e[0] - x)) for s in seeds]
                kept = st.mean(e[1] for e in recs)
                rnd = [e[2] for e in recs if e[2] is not None]
                sh = [e[3] for e in recs if e[3] is not None]
                cells.append(f'{kept:.3f} / ' + (f'{st.mean(rnd):.3f} / {st.mean(sh):.2f}' if rnd else 'none / none'))
            L.append(f'| {a} | ' + ' | '.join(cells) + ' |')
        envs = sorted({(x['environment']['gpu'], x['environment']['torch'], x['environment']['code_commit'][:7])
                       for x in all_measured + [trn[(a, s)] for a in NEW for s in seeds]})
        L += ['', '## Run environment', ''] + [f'- {gpu}, PyTorch {v}, code {cm}' for gpu, v, cm in envs] + ['']
    OUT.write_text('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
