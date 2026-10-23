"""Apply N15's preregistered decision rules to the saved records and write N15_RESULTS.md.

Written and committed before any real N15 run. Every number in the report comes from the JSON records: N15's own
(results/n15: m_* for the new arms, r_* for the saved models re-measured, the training records, n15_summary.json), N12's
n80 training records (fingerprints) and N13's n80-rw training records (the share at beta = 1). For the calibration run
(one seed) it reports the same quantities and checks, with no verdicts.

delta_A: the inner reading's clean-target loss minus head A's; a: its agreement with head A. G = delta_A(rw-b1) - delta_A(sub20),
the shortfall; F(arm) = (delta_A(rw-b1) - delta_A(arm)) / G, the recovered fraction; s(arm): the random targets' share of the
summed weight at the end of training; P(arm) = 1 - s(arm) / s(rw-b1), the removed share. The ladder: (delta_A, a) of clean,
sub20, sub10, sub05, sub025, sub0125; a_hat interpolates a at a model's delta_A; b = a_hat - a.
N15a: F(rw-b4) - 0.75 P(rw-b4) (interval rule). N15b: the per-seed mean of b over the new arms (interval rule, two-sided).
N15c: F(rw-oracle) - 0.75 P(rw-oracle) (interval rule).

    python n15_analyze.py [results folder] [report path] [N12 records folder] [N13 records folder]
    # defaults: results/n15 here, N15_RESULTS.md, results/n12 and results/n13 (or the _calib folders by the summary's mode)
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

from scipy import stats

HERE = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'results' / 'n15'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'N15_RESULTS.md'

REAL_SEEDS = [4101, 4102, 4103, 4104, 4105]
NEW = ('rw-b2', 'rw-b4', 'rw-oracle')
LABEL = {'rw-b1': 'β = 1 (N13)', 'rw-b2': 'β = 2', 'rw-b4': 'β = 4', 'rw-oracle': 'head A judges'}
RUNG_ALL = ('clean', 'sub20', 'sub10', 'sub05', 'sub025', 'sub0125')
WINDOW = 128 + 8
NEEDED = 4
LEVEL = 1 - 0.05 / 3                                     # three interval rules: Bonferroni
HIGH, LOW, F_SD = 0.75, 0.25, 0.15                       # N15a, N15c: the fractions of P, and the SD cap for refutation
LADDER_TOL, LADDER_SD = 0.04, 0.03                       # N15b
SHARE_FRACTION, TRAINED, MIN_G = 0.75, 1.0, 0.10         # preconditions 4, 5, 6
REPRO_LOSS, REPRO_RATIO, MATCH_MIN, SEPARATE, NOISE_TOL = 1e-3, 1e-2, 0.999, 0.60, 0.02


def load(folder, name):
    p = folder / f'{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


def interval(xs, level=LEVEL):
    n = len(xs)
    m = st.mean(xs)
    sd = st.stdev(xs) if n > 1 else float('nan')
    half = stats.t.ppf(1 - (1 - level) / 2, n - 1) * sd / math.sqrt(n) if n > 1 else float('nan')
    return m, sd, m - half, m + half


def fmt(xs, digits=3, level=LEVEL):
    if not xs:
        return 'n/a'
    if len(xs) == 1:
        return f'{xs[0]:+.{digits}f}'
    m, sd, lo, hi = interval(xs, level)
    return f'{m:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]'


def main():
    summary = load(RES, 'n15_summary') or {}
    mode = summary.get('mode', 'real')
    cal = mode == 'calibration'
    n12_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / 'results' / ('n12_calib' if cal else 'n12')
    n13_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else HERE / 'results' / ('n13_calib' if cal else 'n13')
    want = summary.get('seeds', REAL_SEEDS) if cal else REAL_SEEDS
    rung = RUNG_ALL[:-1] if cal else RUNG_ALL
    refs = ('rw-b1',) + rung
    c = summary.get('checks', {})
    cut = summary.get('held_out', {}).get('cut_token')
    m = {(a, s): load(RES, f'm_{a}_seed{s}') for a in NEW for s in want}
    r = {(a, s): load(RES, f'r_{a}_seed{s}') for a in refs for s in want}
    trn = {(a, s): load(RES, f'{a}_seed{s}') for a in NEW for s in want}
    n80 = {s: load(n12_dir, f'n80_seed{s}') for s in want}
    b1 = {s: load(n13_dir, f'n80-rw_seed{s}') for s in want}
    seeds = [s for s in want if all(m[(a, s)] for a in NEW) and all(r[(a, s)] for a in refs) and all(trn[(a, s)] for a in NEW) and n80[s] and b1[s]]
    rec = lambda a, s: m[(a, s)] if a in NEW else r[(a, s)]

    def a_of(a, s):
        x = rec(a, s)
        return (x.get('B_inner_vs_A') or x['B_vs_A'])['agreement_ratio']

    def dA(a, s):
        f = rec(a, s)['functional']
        return f['B_inner_clean'] - f['A_clean']

    def share(a, s):
        if a == 'rw-b1':
            e, fr = b1[s]['r_every_50'][-1], b1[s].get('noise_fraction_step0', 0.8)
            return fr * e[3] / (fr * e[3] + (1 - fr) * e[2])
        return trn[(a, s)]['w_every_50'][-1][4]

    G = lambda s: dA('rw-b1', s) - dA('sub20', s)
    Fv = lambda a, s: (dA('rw-b1', s) - dA(a, s)) / G(s)
    Pv = lambda a, s: 1 - share(a, s) / share('rw-b1', s)

    def ladder(s):
        return [(dA(a, s), a_of(a, s)) for a in rung]

    def ordered(s):
        xs = [p[0] for p in ladder(s)]
        return all(xs[i] < xs[i + 1] for i in range(len(xs) - 1))

    def a_hat(s, x):
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

    def ratio(x, key):
        return x[key]['agreement_ratio']

    all_measured = [m[(a, s)] for a in NEW for s in seeds] + [r[(a, s)] for a in refs for s in seeds]

    def run_ok(a, s):
        t = trn[(a, s)]
        return (t['training_tokens'] == cut and t['max_offset'] + WINDOW <= cut
                and t['batch_fingerprint_step0'] == n80[s]['batch_fingerprint_step0']
                and t['replaced_positions_fingerprint_step0'] == n80[s]['replaced_positions_fingerprint_step0']
                and abs(t['noise_fraction_step0'] - 0.8) <= NOISE_TOL)

    n_needed = min(NEEDED, len(want))
    pre = {
        '1. Setup: batch, same-model, identical-heads and analytic checks; the weight identity at β = 1; the oracle\'s detachment; windows; '
        'held-out tail untouched; fingerprints and noise fractions equal N12\'s n80\'s':
            bool(c.get('batch', {}).get('targets_are_i+1_to_i+8')) and bool(c.get('windows_shifted')) and bool(c.get('identical_heads', {}).get('pass'))
            and bool(c.get('analytic', {}).get('pass')) and bool(c.get('weight_identity', {}).get('pass')) and bool(c.get('oracle_detachment', {}).get('pass'))
            and all(x['pass'] for x in c.get('same_model', [{'pass': False}])) and cut is not None and bool(seeds) and all(run_ok(a, s) for a in NEW for s in seeds),
        f'2. Must match: re-measuring the saved models reproduces their records (losses within {REPRO_LOSS:g}, ratios within {REPRO_RATIO:g}); '
        f'the constructed mixture check and the zero-noise copy (≥ {MATCH_MIN}) in every model':
            bool(seeds) and all(r[(a, s)].get('reproduction', {}).get('pass') for a in refs for s in seeds)
            and all(x['constructed_mixture']['pass'] and ratio(x, 'copy0_vs_A') >= MATCH_MIN for x in all_measured),
        f'3. Must differ: rotated control ≤ {SEPARATE} in at least {NEEDED} of 5, in every arm':
            bool(seeds) and all(sum(ratio(rec(a, s), 'rotated_control') <= SEPARATE for s in seeds) >= n_needed for a in NEW + refs),
        f'4a. The manipulation took (β = 4): s(rw-b4) ≤ {SHARE_FRACTION} · s(rw-b1) in at least {NEEDED} of 5 seeds':
            sum(share('rw-b4', s) <= SHARE_FRACTION * share('rw-b1', s) for s in seeds) >= n_needed,
        f'4c. The manipulation took (oracle): s(rw-oracle) ≤ {SHARE_FRACTION} · s(rw-b1) in at least {NEEDED} of 5 seeds':
            sum(share('rw-oracle', s) <= SHARE_FRACTION * share('rw-b1', s) for s in seeds) >= n_needed,
        f'5a. The head trained (β = 4): Δ_A(rw-b4) ≤ {TRAINED} in at least {NEEDED} of 5 seeds':
            sum(dA('rw-b4', s) <= TRAINED for s in seeds) >= n_needed,
        f'5c. The head trained (oracle): Δ_A(rw-oracle) ≤ {TRAINED} in at least {NEEDED} of 5 seeds':
            sum(dA('rw-oracle', s) <= TRAINED for s in seeds) >= n_needed,
        f'6. The shortfall exists: G = Δ_A(rw-b1) − Δ_A(sub20) ≥ {MIN_G} nats in every seed':
            bool(seeds) and all(G(s) >= MIN_G for s in seeds),
        f'7. The ladder brackets and is ordered: at least {NEEDED} seeds with an ordered ladder and at least two new arms bracketed':
            sum(ordered(s) and sum(b(a, s) is not None for a in NEW) >= 2 for s in seeds) >= n_needed,
    }
    names = list(pre)
    pooled_seeds = [s for s in seeds if ordered(s) and sum(b(a, s) is not None for a in NEW) >= 2]

    def gate(needed):
        if len(seeds) < 5:
            return f'**Not reached: complete records for only {len(seeds)} of 5 seeds.**'
        bad = [names[i].split(':')[0] for i in needed if not pre[names[i]]]
        return f'**Untestable: a precondition failed ({"; ".join(bad)}).**' if bad else None

    def f_verdict(arm, label):
        x75 = [Fv(arm, s) - HIGH * Pv(arm, s) for s in seeds]
        x25 = [Fv(arm, s) - LOW * Pv(arm, s) for s in seeds]
        xf = [Fv(arm, s) for s in seeds]
        m75, _, lo75, hi75 = interval(x75)
        m25, sd25, lo25, hi25 = interval(x25)
        mf, sdf, lof, hif = interval(xf)
        tail = (f'F = {mf:+.3f} [{lof:+.3f}, {hif:+.3f}], P = {st.mean([Pv(arm, s) for s in seeds]):+.3f}; F − 0.75 P = {m75:+.3f} [{lo75:+.3f}, {hi75:+.3f}]; '
                f'F − 0.25 P = {m25:+.3f} [{lo25:+.3f}, {hi25:+.3f}], standard deviation {sd25:.3f}; {LEVEL:.1%} intervals, {len(seeds)} seeds.')
        if lo75 > 0:
            return f'**Supported: {label} recovers the shortfall in proportion to the share it removes.** {tail}'
        if hi25 < 0 and sd25 <= F_SD:
            return f'**Refuted.** {tail}'
        if lof > 0 or hif < 0:
            return f'**Some effect, size unclear.** {tail}'
        return f'**Inconclusive.** {tail}'

    L = ['# N15 results' if not cal else '# N15 calibration run', '',
         f'Generated by `n15_analyze.py` from the records in `{RES.name}/`, with N12\'s `n80` records in `{n12_dir.name}/` and N13\'s `n80-rw` records in `{n13_dir.name}/`. Preregistration: `N15_PREREGISTRATION.md`.',
         f"Measured on the held-out tail's windows {summary.get('held_out', {}).get('measure')}, N12's. Δ_A is the inner reading's clean-target loss minus head A's; a its agreement with head A; "
         'G = Δ_A(rw-b1) − Δ_A(sub20) the shortfall; F the fraction of G an arm recovers; s the random targets\' share of the summed weight at the end of training; '
         'P = 1 − s(arm) / s(rw-b1) the share removed; â the clean-target ladder\'s agreement at the model\'s Δ_A; b = â − a. '
         f'Intervals are Bonferroni-corrected t-intervals at {LEVEL:.1%}.', '']
    if not cal:
        v_a = gate([0, 1, 2, 3, 5, 7]) or f_verdict('rw-b4', 'the sharper discount')
        v_c = gate([0, 1, 2, 4, 6, 7]) or f_verdict('rw-oracle', 'the discount judged by head A')
        v_b = gate([0, 1, 2, 8])
        if not v_b:
            xs = [st.mean(b(a, s) for a in NEW if b(a, s) is not None) for s in pooled_seeds]
            mb, sdb, lob, hib = interval(xs)
            tail = f'pooled residual = {mb:+.4f} [{lob:+.4f}, {hib:+.4f}], standard deviation {sdb:.4f}, {len(xs)} seeds.'
            if lob > -LADDER_TOL and hib < LADDER_TOL:
                v_b = f'**Supported: the sharpened heads sit on the clean-target ladder.** {tail}'
            elif (hib < -LADDER_TOL or lob > LADDER_TOL) and sdb <= LADDER_SD:
                v_b = f'**Refuted ({"below" if lob > LADDER_TOL else "above"} the ladder).** {tail}'
            elif lob > 0 or hib < 0:
                v_b = f'**Some effect, size unclear.** {tail}'
            else:
                v_b = f'**Inconclusive.** {tail}'
        L += ['## Verdicts', '',
              f'**N15a (primary: the sharper discount recovers the shortfall in proportion to the share it removes):** {v_a}', '',
              f'**N15b (geometry follows function along the sharpness ladder):** {v_b}', '',
              f'**N15c (a better judge recovers the shortfall in proportion to the share it removes):** {v_c}', '']
    else:
        L += ['This is the calibration run of the preregistration\'s calibration step: one seed, on windows the real run does not use. '
              'It decides nothing, and no verdict is computed. Whether the sharpened heads train and remove share (preconditions 4 and 5) is what it exists for.', '']
    L += ['## Preconditions', '', '| Precondition | Met |', '|---|---|']
    L += [f'| {k} | {"yes" if ok else "**no**"} |' for k, ok in pre.items()]

    if seeds:
        def mean(f, over=None):
            xs = [v for v in (f(s) for s in (over or seeds)) if v is not None]
            return st.mean(xs) if xs else float('nan')
        if cal:
            s_ = seeds[0]
            rule = (f"the largest β in {{2, 4, 8}} whose head trains (Δ_A ≤ {TRAINED}) and whose share is at most {SHARE_FRACTION} of β = 1's. "
                    f"Here: β = 2: Δ_A {dA('rw-b2', s_):.3f}, share {share('rw-b2', s_):.3f}; β = 4: Δ_A {dA('rw-b4', s_):.3f}, share {share('rw-b4', s_):.3f}; "
                    f"β = 1's share {share('rw-b1', s_):.3f}. β = 4 meets the rule: {'yes' if dA('rw-b4', s_) <= TRAINED and share('rw-b4', s_) <= SHARE_FRACTION * share('rw-b1', s_) else 'no'}.")
            L += ['', f'The calibration\'s rule for N15a\'s arm: {rule}']
        L += ['', '## By arm (means over seeds)', '',
              '| Arm | Weight | Δ_A | a: agreement with head A | Share at the end, s | Removed share, P | Recovered fraction, F | â: the ladder at the same Δ_A | b = â − a | Head A, minus clean | Rotated |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
        for a in ('rw-b1',) + NEW:
            bs = [v for v in (b(a, s) for s in seeds) if v is not None]
            L.append(f"| {a} | {LABEL[a]} | {mean(lambda s: dA(a, s)):+.3f} | {mean(lambda s: a_of(a, s)):.3f} | {mean(lambda s: share(a, s)):.3f} | "
                     f"{'—' if a == 'rw-b1' else f'{mean(lambda s: Pv(a, s)):+.3f}'} | {'—' if a == 'rw-b1' else f'{mean(lambda s: Fv(a, s)):+.3f}'} | "
                     f"{mean(lambda s: a_hat(s, dA(a, s))):.3f} | {f'{st.mean(bs):+.3f}' if bs else 'n/a'} | "
                     f"{mean(lambda s: rec(a, s)['functional']['A_clean'] - r[('clean', s)]['functional']['A_clean']):+.4f} | {mean(lambda s: ratio(rec(a, s), 'rotated_control')):.3f} |")
        for a in rung:
            L.append(f"| {a} | ladder | {mean(lambda s: dA(a, s)):+.3f} | {mean(lambda s: a_of(a, s)):.3f} | — | — | "
                     f"{'1 by definition' if a == 'sub20' else '—'} | (on the ladder) | 0 | "
                     f"{mean(lambda s: rec(a, s)['functional']['A_clean'] - r[('clean', s)]['functional']['A_clean']):+.4f} | {mean(lambda s: ratio(rec(a, s), 'rotated_control')):.3f} |")
        L += ['', '## By seed', '',
              '| Seed | G | s(b1) | s(b2) | s(b4) | s(oracle) | F(b2) | F(b4) | F(oracle) | P(b4) | P(oracle) | b(b2) | b(b4) | b(oracle) | Ladder ordered |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|']
        for s in seeds:
            bb = ['n/a' if b(a, s) is None else f'{b(a, s):+.3f}' for a in NEW]
            L.append(f"| {s} | {G(s):+.3f} | {share('rw-b1', s):.3f} | {share('rw-b2', s):.3f} | {share('rw-b4', s):.3f} | {share('rw-oracle', s):.3f} | "
                     f"{Fv('rw-b2', s):+.3f} | {Fv('rw-b4', s):+.3f} | {Fv('rw-oracle', s):+.3f} | {Pv('rw-b4', s):+.3f} | {Pv('rw-oracle', s):+.3f} | "
                     + ' | '.join(bb) + f" | {'yes' if ordered(s) else 'no'} |")
        L += ['', '## The clean-target ladder by seed: (Δ_A, a)', '', '| Seed | ' + ' | '.join(rung) + ' |', '|---|' + '---|' * len(rung)]
        for s in seeds:
            L.append(f'| {s} | ' + ' | '.join(f'({x:+.3f}, {y:.3f})' for x, y in ladder(s)) + ' |')
        L += ['', '## Secondary and exploratory measures', '', 'None of these enters a decision rule.', '']
        L.append('- S1, the cost per unit of share, (Δ_A(arm) − Δ_A(sub20)) / s(arm): ' + ', '.join(
            f'{a} {fmt([(dA(a, s) - dA("sub20", s)) / share(a, s) for s in seeds], 2)}' for a in ('rw-b1',) + NEW) + ' (N14 found 0.30–0.41 along the dose).')
        L.append(f"- S2, β = 2: F {fmt([Fv('rw-b2', s) for s in seeds])}, P {fmt([Pv('rw-b2', s) for s in seeds])}; F and P in the same order across β = 1, 2, 4 in "
                 f"{sum((Fv('rw-b2', s) <= Fv('rw-b4', s)) == (Pv('rw-b2', s) <= Pv('rw-b4', s)) for s in seeds)} of {len(seeds)} seeds.")
        avg = lambda a, s: (b1[s].get('noise_fraction_step0', 0.8) * st.mean(e[3] for e in b1[s]['r_every_50']) / (b1[s].get('noise_fraction_step0', 0.8) * st.mean(e[3] for e in b1[s]['r_every_50']) + (1 - b1[s].get('noise_fraction_step0', 0.8)) * st.mean(e[2] for e in b1[s]['r_every_50']))) if a == 'rw-b1' else st.mean(e[4] for e in trn[(a, s)]['w_every_50'])
        L.append('- S4, the schedule-average share, and F against the removed average share: ' + ', '.join(
            f"{a} share {mean(lambda s: avg(a, s)):.3f}" + ('' if a == 'rw-b1' else f", F − 0.75 P_avg {fmt([Fv(a, s) - HIGH * (1 - avg(a, s) / avg('rw-b1', s)) for s in seeds])}")
            for a in ('rw-b1',) + NEW) + '.')
        L.append('- S6, head A against clean (same seed): ' + ', '.join(
            f"{a} {fmt([rec(a, s)['functional']['A_clean'] - r[('clean', s)]['functional']['A_clean'] for s in seeds])}" for a in ('rw-b1',) + NEW) + '; against rw-b1: ' + ', '.join(
            f"{a} {fmt([m[(a, s)]['functional']['A_clean'] - r[('rw-b1', s)]['functional']['A_clean'] for s in seeds])}" for a in NEW) + '.')
        L.append('- S7, agreement with the copy at 0.8, beside the head A measure: ' + ', '.join(
            f"{a} {mean(lambda s: ratio(rec(a, s), 'B_vs_copy')):.3f} / {mean(lambda s: a_of(a, s)):.3f}" for a in ('rw-b1',) + NEW) + '.')
        steps = [0, 300, 1500, 2850] if not cal or len(trn[(NEW[0], seeds[0])]['w_every_50']) > 20 else [0, 20, 40]
        L += ['', '## S3 and S5: the weights over training (means over seeds)', '',
              'Mean weight on the kept and on the replaced positions, the random targets\' share of the summed weight, and the fraction of kept positions with weight below 0.01, '
              'at the recorded step nearest each listed step. For rw-b1 the record is N13\'s (no ignored fraction).', '',
              '| Arm | ' + ' | '.join(f'step {x}: kept / replaced / share / ignored' for x in steps) + ' |', '|---|' + '---|' * len(steps)]
        for a in ('rw-b1',) + NEW:
            cells = []
            for x in steps:
                if a == 'rw-b1':
                    recs = [min(b1[s]['r_every_50'], key=lambda e: abs(e[0] - x)) for s in seeds]
                    fr = [b1[s].get('noise_fraction_step0', 0.8) for s in seeds]
                    sh = [f * e[3] / (f * e[3] + (1 - f) * e[2]) for f, e in zip(fr, recs)]
                    cells.append(f'{st.mean(e[2] for e in recs):.3f} / {st.mean(e[3] for e in recs):.3f} / {st.mean(sh):.2f} / —')
                else:
                    recs = [min(trn[(a, s)]['w_every_50'], key=lambda e: abs(e[0] - x)) for s in seeds]
                    cells.append(f'{st.mean(e[2] for e in recs):.3f} / {st.mean(e[3] for e in recs):.3f} / {st.mean(e[4] for e in recs):.2f} / {st.mean(e[5] for e in recs):.2f}')
            L.append(f'| {a} | ' + ' | '.join(cells) + ' |')
        envs = sorted({(x['environment']['gpu'], x['environment']['torch'], x['environment']['code_commit'][:7])
                       for x in all_measured + [trn[(a, s)] for a in NEW for s in seeds]})
        L += ['', '## Run environment', ''] + [f'- {gpu}, PyTorch {v}, code {cm}' for gpu, v, cm in envs] + ['']
    OUT.write_text('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
