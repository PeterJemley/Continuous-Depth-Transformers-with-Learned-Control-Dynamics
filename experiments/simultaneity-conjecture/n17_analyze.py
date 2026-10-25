"""Apply N17's preregistered decision rules to the saved records and write N17_RESULTS.md.

Written and committed before any real N17 run. Every number in the report comes from the JSON records: N17's own
(results/n17: m_* for the new arms, r_* for the saved models re-measured, the training records, n17_summary.json), N12's
n80 training records (fingerprints), N13's n80-rw training records (rw-b1's weights over training), N14's training records
(the nested positions' fingerprints) and the training records of the saved rw-oracle (N16's, or N15's calibration model's).
For the calibration run (one seed) it reports the same quantities and checks, with no verdicts, and applies the calibration's
one rule, which chooses only the loop-free rungs the real run trains.

delta_A: head B's clean-target loss minus head A's, on the measurement windows. x: the judge's accuracy, its clean-target loss
minus head A's (head C's in a new arm; x_self = delta_A(rw-b1), head B judging itself; 0 for rw-oracle, head A judging).
G = delta_A(rw-b1) - delta_A(sub20); F(arm) = (delta_A(rw-b1) - delta_A(arm)) / G. The loop-free line, within seed: the rungs'
points (x, F), sorted by x and joined by straight lines; F_lf(x) inside their range, nothing outside.
N17a: F* = F_lf(x_self) wholly above 0.75 (refuted wholly below 0.25, sd at most 0.15).
N17b: d = F(rw-peer) - F_lf(x_peer) wholly below -0.5 (refuted wholly above -0.25, sd at most 0.15).

    python n17_analyze.py [results folder] [report path] [N12 records] [N13 records] [N14 records] [the saved rw-oracle's records]
    # defaults: results/n17 here and N17_RESULTS.md; for a calibration summary, results/n17_calib and N17_CALIBRATION.md;
    # results/n12, n13, n14 and n16 (or n12_calib, n13_calib, n14_calib and n15_calib for a calibration summary)
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

from scipy import stats

HERE = Path(__file__).resolve().parent
REAL_SEEDS = [4101, 4102, 4103, 4104, 4105]
RUNG_ORDER = ('rw-j20', 'rw-j10', 'rw-j05', 'rw-j025', 'rw-j0125')
FRAC = {'rw-j20': 0.2, 'rw-j10': 0.1, 'rw-j05': 0.05, 'rw-j025': 0.025, 'rw-j0125': 0.0125}
N14_TAG = {'rw-j10': 'sub10', 'rw-j05': 'sub05', 'rw-j025': 'sub025', 'rw-j0125': 'sub0125'}
DEFAULT = ('rw-j10', 'rw-j05')
CALIB_ONLY = ('rw-b1-plain', 'rw-b1-passenger')
LADDER_REAL = ('clean', 'sub20', 'sub10', 'sub05', 'sub025', 'sub0125')
WINDOW = 128 + 8
NEEDED = 4
LEVEL = 1 - 0.05 / 2                                     # two interval rules: Bonferroni
EXPLORATORY = 0.95
A_SUP, A_REF, A_SD = 0.75, 0.25, 0.15                    # N17a, on F*
D_SUP, D_REF, D_SD = -0.5, -0.25, 0.15                   # N17b, on d
LINE_HIGH, MIN_GAP, SPARE = 0.75, 0.05, 0.03             # precondition 6b; precondition 6a; the calibration's rule
TRAINED, MIN_G = 1.0, 0.10                               # preconditions 4 and 5
REPRO_LOSS, REPRO_RATIO, MATCH_MIN, SEPARATE, NOISE_TOL, FRAC_TOL = 1e-3, 1e-2, 0.999, 0.60, 0.02, 0.02
CHECKS = ('weight_identity', 'forward_identity', 'detachment', 'judge_source', 'passenger', 'copied_peer', 'overflow', 'nesting')


def load(folder, name):
    p = folder / f'{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


def interval(xs, level=LEVEL):
    n = len(xs)
    m = st.mean(xs)
    sd = st.stdev(xs) if n > 1 else float('nan')
    half = stats.t.ppf(1 - (1 - level) / 2, n - 1) * sd / math.sqrt(n) if n > 1 else float('nan')
    return m, sd, m - half, m + half


def fmt(xs, digits=3, level=EXPLORATORY, sign='+'):
    xs = [x for x in xs if x is not None]
    if not xs:
        return 'n/a'
    if len(xs) == 1:
        return f'{xs[0]:{sign}.{digits}f}'
    m, sd, lo, hi = interval(xs, level)
    return f'{m:{sign}.{digits}f} [{lo:{sign}.{digits}f}, {hi:{sign}.{digits}f}]'


def interp(points, x):
    """Straight lines between points sorted by their first value; None outside their range."""
    pts = sorted(points)
    if x is None or not pts or not (pts[0][0] <= x <= pts[-1][0]):
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[0][1] if len(pts) == 1 and pts[0][0] == x else None


def choose_rungs(xs, targets):
    """The calibration's rule: the shortest run of adjacent rungs, at most three, ordered by fraction (x rising as the fraction
    falls), whose x span every target with SPARE to spare at each end; a tie goes to the run nearer 10% and 5%."""
    lo, hi = min(targets) - SPARE, max(targets) + SPARE
    found = []
    for n in (2, 3):
        for i in range(len(RUNG_ORDER) - n + 1):
            run = RUNG_ORDER[i:i + n]
            v = [xs.get(a) for a in run]
            if None in v or not all(v[j] < v[j + 1] for j in range(n - 1)):
                continue
            if min(v) <= lo and max(v) >= hi:
                found.append((n, abs(i - 1), run))
        if found:
            return min(found)[2]
    return None


def main():
    res_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    RES = res_arg or HERE / 'results' / 'n17'
    summary = load(RES, 'n17_summary') or {}
    cal = summary.get('mode') == 'calibration'
    if res_arg is None and not summary:
        RES = HERE / 'results' / 'n17_calib'
        summary = load(RES, 'n17_summary') or {}
        cal = summary.get('mode') == 'calibration'
    OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / ('N17_CALIBRATION.md' if cal else 'N17_RESULTS.md')
    sub = lambda k: HERE / 'results' / (f'{k}_calib' if cal else k)
    arg = lambda i, default: Path(sys.argv[i]) if len(sys.argv) > i else default
    n12_dir, n13_dir, n14_dir = arg(3, sub('n12')), arg(4, sub('n13')), arg(5, sub('n14'))
    oracle_dir = arg(6, HERE / 'results' / ('n15_calib' if cal else 'n16'))
    want = summary.get('seeds', REAL_SEEDS) if cal else REAL_SEEDS
    arms = list(summary.get('arms', {}))
    rungs = [a for a in RUNG_ORDER if a in arms]
    new = ['rw-peer'] + rungs + [a for a in CALIB_ONLY if a in arms]
    ladder_arms = LADDER_REAL[:-1] if cal else LADDER_REAL
    refs = ['rw-b1', 'rw-oracle'] + list(ladder_arms)
    c = summary.get('checks', {})
    cut = summary.get('held_out', {}).get('cut_token')
    m = {(a, s): load(RES, f'm_{a}_seed{s}') for a in new for s in want}
    r = {(a, s): load(RES, f'r_{a}_seed{s}') for a in refs for s in want}
    trn = {(a, s): load(RES, f'{a}_seed{s}') for a in new for s in want}
    n80 = {s: load(n12_dir, f'n80_seed{s}') for s in want}
    b1 = {s: load(n13_dir, f'n80-rw_seed{s}') for s in want}
    orc = {s: load(oracle_dir, f'rw-oracle_seed{s}') for s in want}
    n14 = {(a, s): load(n14_dir, f'{N14_TAG[a]}_seed{s}') for a in rungs if a in N14_TAG for s in want}
    rec = lambda a, s: r[(a, s)] if a in refs else m.get((a, s))
    seeds = [s for s in want if all(m[(a, s)] and trn[(a, s)] for a in new) and all(r[(a, s)] for a in refs) and n80[s] and b1[s]]

    def a_of(a, s):
        x = rec(a, s)
        return (x.get('B_inner_vs_A') or x['B_vs_A'])['agreement_ratio']

    def dA(a, s):
        f = rec(a, s)['functional']
        return f['B_inner_clean'] - f['A_clean']

    def xj(a, s):
        if a == 'rw-b1':
            return dA('rw-b1', s)
        if a == 'rw-oracle':
            return 0.0
        return rec(a, s)['judge_measure']['x']

    G = lambda s: dA('rw-b1', s) - dA('sub20', s)
    Fv = lambda a, s: (dA('rw-b1', s) - dA(a, s)) / G(s)
    line = lambda s: [(xj(a, s), Fv(a, s)) for a in rungs]
    F_lf = lambda s, x: interp(line(s), x)
    F_star = lambda s: F_lf(s, xj('rw-b1', s))
    d_of = lambda s: None if F_lf(s, xj('rw-peer', s)) is None else Fv('rw-peer', s) - F_lf(s, xj('rw-peer', s))

    def ordered(s):
        v = [xj(a, s) for a in rungs]
        return all(v[i] < v[i + 1] and v[i + 1] - v[i] >= MIN_GAP for i in range(len(v) - 1))

    ok6a = lambda s: ordered(s) and F_star(s) is not None
    ok6b = lambda s: F_lf(s, xj('rw-peer', s)) is not None and F_lf(s, xj('rw-peer', s)) >= LINE_HIGH

    def sharp(a, s):
        """(kept mean / replaced mean, random share) at the last logged step of training."""
        if a == 'rw-b1':
            e, fr = b1[s]['r_every_50'][-1], b1[s].get('noise_fraction_step0', 0.8)
            return e[2] / e[3], fr * e[3] / (fr * e[3] + (1 - fr) * e[2])
        e = (orc[s] if a == 'rw-oracle' else trn[(a, s)])['w_every_50'][-1]
        return e[2] / e[3], e[4]

    def ratio(x, key):
        return x[key]['agreement_ratio']

    def run_ok(a, s):
        t = trn[(a, s)]
        ok = (t['training_tokens'] == cut and t['max_offset'] + WINDOW <= cut
              and t['batch_fingerprint_step0'] == n80[s]['batch_fingerprint_step0']
              and t['replaced_positions_fingerprint_step0'] == n80[s]['replaced_positions_fingerprint_step0']
              and abs(t['noise_fraction_step0'] - 0.8) <= NOISE_TOL)
        if a in FRAC:
            ok = ok and t['judge_keep_inside_K_step0'] and abs(t['judge_kept_fraction_step0'] - FRAC[a]) <= FRAC_TOL
            ref = n14.get((a, s))
            if ref is not None:
                ok = ok and t['judge_keep_fingerprint_step0'] == ref['keep_fingerprint_step0']
        return ok

    measured = [rec(a, s) for a in new + refs for s in seeds]
    n_needed = min(NEEDED, len(want))
    trained_ok = lambda arms_: all(sum(dA(a, s) <= TRAINED and xj(a, s) <= TRAINED for s in seeds) >= n_needed for a in arms_)
    pre = {
        '1. Setup: batch, same-model, identical-heads and analytic checks; the weight identity; forward_z equals N4\'s forward; head C\'s detachment; '
        'the judge-source, passenger and copied-peer checks; the overflow test; the nesting of head C\'s positions; windows; held-out tail untouched; '
        'each new run\'s fingerprints and noise fraction equal N12\'s n80\'s, and each loop-free head C\'s positions are N14\'s, inside K':
            bool(c.get('batch', {}).get('targets_are_i+1_to_i+8')) and bool(c.get('windows_shifted')) and bool(c.get('identical_heads', {}).get('pass'))
            and bool(c.get('analytic', {}).get('pass')) and all(bool(c.get(k, {}).get('pass')) for k in CHECKS)
            and all(x['pass'] for x in c.get('same_model', [{'pass': False}]))
            and cut is not None and bool(seeds) and all(run_ok(a, s) for a in new for s in seeds),
        f'2. Must match: re-measuring the saved models reproduces their records (losses within {REPRO_LOSS:g}, ratios within {REPRO_RATIO:g}); '
        f'the constructed mixture check and the zero-noise copy (≥ {MATCH_MIN}) in every model':
            bool(seeds) and all(r[(a, s)].get('reproduction', {}).get('pass') for a in refs for s in seeds)
            and all(x['constructed_mixture']['pass'] and ratio(x, 'copy0_vs_A') >= MATCH_MIN for x in measured),
        f'3. Must differ: rotated control ≤ {SEPARATE} in at least {NEEDED} of 5, in every arm':
            bool(seeds) and all(sum(ratio(rec(a, s), 'rotated_control') <= SEPARATE for s in seeds) >= n_needed for a in new + refs),
        f'4a. The heads trained, for N17a: Δ_A of head B and x of head C ≤ {TRAINED} in at least {NEEDED} of 5 seeds, in each loop-free rung':
            bool(seeds) and trained_ok(rungs),
        f'4b. The heads trained, for N17b: as 4a, and in rw-peer':
            bool(seeds) and trained_ok(rungs + ['rw-peer']),
        f'5. The shortfall exists: G = Δ_A(rw-b1) − Δ_A(sub20) ≥ {MIN_G} nats in every seed':
            bool(seeds) and all(G(s) >= MIN_G for s in seeds),
        f'6a. The loop-free line reaches head B\'s accuracy: the rungs\' x ordered by fraction, at least {MIN_GAP} apart, and bracketing x_self, in at least {NEEDED} of 5 seeds':
            sum(ok6a(s) for s in seeds) >= n_needed,
        f'6b. The loop-free line reaches the peer\'s accuracy, and is high there: the rungs bracket x_peer and F_lf(x_peer) ≥ {LINE_HIGH}, in at least {NEEDED} of 5 seeds':
            sum(ok6b(s) for s in seeds) >= n_needed,
    }
    names = list(pre)

    def gate(needed):
        if len(seeds) < 5:
            return f'**Not reached: complete records for only {len(seeds)} of 5 seeds.**'
        bad = [names[i].split(':')[0] for i in needed if not pre[names[i]]]
        return f'**Untestable: a precondition failed ({"; ".join(bad)}).**' if bad else None

    def one_sided(xs, quantity, sup_side, sup_margin, ref_margin, sd_cap, sup_text, digits=3):
        """sup_side 'above': supported if the interval lies wholly above sup_margin, refuted if wholly below ref_margin with sd at
        most sd_cap; 'below': the reverse. Some effect if the interval excludes 0; else inconclusive."""
        mv, sd, lo, hi = interval(xs)
        tail = f'{quantity} = {mv:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}], standard deviation {sd:.{digits}f}; {LEVEL:.1%} interval, {len(xs)} seeds.'
        sup = lo > sup_margin if sup_side == 'above' else hi < sup_margin
        ref = (hi < ref_margin if sup_side == 'above' else lo > ref_margin) and sd <= sd_cap
        if sup:
            return f'**Supported: {sup_text}** {tail}'
        if ref:
            return f'**Refuted.** {tail}'
        if lo > 0 or hi < 0:
            return f'**Some effect, size unclear.** {tail}'
        return f'**Inconclusive.** {tail}'

    L = ['# N17 results' if not cal else '# N17 calibration run', '',
         f'Generated by `n17_analyze.py` from the records in `{RES.name}/`, with N12\'s `n80` records in `{n12_dir.name}/`, N13\'s `n80-rw` records in `{n13_dir.name}/`, '
         f'N14\'s training records in `{n14_dir.name}/` and the saved `rw-oracle`\'s training records in `{oracle_dir.name}/`. Preregistration: `N17_PREREGISTRATION.md`.',
         f"Measured on the held-out tail's windows {summary.get('held_out', {}).get('measure')}, N12's. Δ_A is head B's clean-target loss minus head A's; "
         'x the judge\'s accuracy, its clean-target loss minus head A\'s (head C\'s in a new arm, x_self = Δ_A(rw-b1), 0 for rw-oracle); '
         'G = Δ_A(rw-b1) − Δ_A(sub20) the shortfall; F = (Δ_A(rw-b1) − Δ_A(arm)) / G the recovered fraction; F_lf the loop-free line through the rungs\' (x, F); '
         'F* = F_lf(x_self); d = F(rw-peer) − F_lf(x_peer). '
         f'The two claims\' intervals are Bonferroni-corrected t-intervals at {LEVEL:.1%}; every other interval is a {EXPLORATORY:.0%} t-interval and decides nothing.', '']
    if not cal:
        pa = [s for s in seeds if ok6a(s)]
        pb = [s for s in seeds if ok6b(s)]
        v_a = gate([0, 1, 2, 3, 5, 6]) or one_sided([F_star(s) for s in pa], 'F*', 'above', A_SUP, A_REF, A_SD,
                                                    'without the loop, a judge as accurate as head B recovers the shortfall.')
        v_b = gate([0, 1, 2, 4, 5, 7]) or one_sided([d_of(s) for s in pb], 'd', 'below', D_SUP, D_REF, D_SD,
                                                    'with the loop, a judge that is not head B does far worse than a loop-free judge of its accuracy.')
        L += ['## Verdicts', '',
              f'**N17a (primary: without the loop, a judge as accurate as head B recovers the shortfall):** {v_a}', '',
              f'**N17b (with the loop, a judge that is not head B does not):** {v_b}', '',
              'A support is read as consistent with the loop, not as showing that the loop alone is the cause: what the judge is trained on, '
              'its course over training and its sharpness are not separated (the preregistration\'s rivals R3–R5; S2 and S3 below).', '']
    else:
        L += ['This is the calibration run of the preregistration\'s calibration step: one seed, on windows the real run does not use. '
              'It decides nothing but which loop-free rungs the real run trains, and no verdict is computed. '
              '`rw-oracle` is N15\'s calibration model of the same seed, re-measured.', '']
    L += ['## Preconditions', '', '| Precondition | Met |', '|---|---|']
    L += [f'| {k} | {"yes" if ok else "**no**"} |' for k, ok in pre.items()]
    L += ['', '**The setup checks:** ' + '; '.join(
        f"{k} {'passed' if c.get(k, {}).get('pass') else '**failed**'}" for k in CHECKS)
        + (f". Passenger: largest difference {c['passenger']['max_difference']:.1e} over {c['passenger']['steps']} steps; copied peer: largest differences "
           f"{c['copied_peer']['max_main_difference']:.1e} (main) and {c['copied_peer']['max_head_C_difference']:.1e} (head C against head B)." if 'passenger' in c and 'copied_peer' in c else '.')]

    if seeds:
        def mean(f, over=None):
            xs = [v for v in (f(s) for s in (over or seeds)) if v is not None]
            return st.mean(xs) if xs else float('nan')

        if cal:
            s_ = seeds[0]
            xs = {a: xj(a, s_) for a in rungs}
            x_self, x_peer = xj('rw-b1', s_), xj('rw-peer', s_)
            both = choose_rungs(xs, [x_self, x_peer])
            only = None if both else choose_rungs(xs, [x_self])
            pp = summary.get('passenger_pair', {})
            L += ['', '## The calibration\'s rule for the rungs', '',
                  f'In seed {s_}: x_self = {x_self:+.3f}, x_peer = {x_peer:+.3f}; the rungs\' x: '
                  + ', '.join(f'{a} {xs[a]:+.3f}' for a in rungs) + '.', '']
            if both:
                L.append(f'**The real run trains {", ".join(both)}**: the shortest run of adjacent rungs, ordered by fraction, spanning both x_self and x_peer '
                         f'with at least {SPARE} nats to spare at each end' + (' (the default pair).' if tuple(both) == DEFAULT else '.')
                         + f' The real job\'s argument: `--rungs {",".join(both)}`.')
            elif only:
                L.append(f'**No run of rungs spans both; the real run trains {", ".join(only)}**, the shortest spanning x_self. '
                         'N17b is likely to be untestable by precondition 6b; this is recorded before the real run. '
                         f'The real job\'s argument: `--rungs {",".join(only)}`.')
            else:
                L.append('**No run of rungs spans x_self.** By the preregistration, the real run is not queued; the calibration\'s values go to the author, '
                         'and any change to the arms would be an amendment.')
            L += ['', f"**The full-length passenger pair:** rw-b1 retrained twice in N17's code, once plain and once with a passenger head C: "
                  + ('**bit-identical**' if pp.get('identical') else ('**not bit-identical**' if pp else 'no record'))
                  + (f" (largest difference {pp['max_difference']:.1e})" if pp else '') + '. '
                  + ('By the preregistration, the real run is not queued until the cause is found and fixed. ' if pp and not pp.get('identical') else '')
                  + 'Against N13\'s saved seed-3001 model, re-measured on the same windows, for the record: '
                  + ', '.join(f"{a} Δ_A {dA(a, s_):+.4f} against {dA('rw-b1', s_):+.4f}, agreement {a_of(a, s_):.4f} against {a_of('rw-b1', s_):.4f}"
                              for a in CALIB_ONLY if a in arms) + '.']
            L += ['', f'**Seed {s_}\'s values, for the record:** G {G(s_):+.3f}; F(rw-peer) {Fv("rw-peer", s_):+.3f}; '
                  + ', '.join(f'F({a}) {Fv(a, s_):+.3f}' for a in rungs) + '; the default pair\'s F* '
                  + (lambda v: 'n/a (unbracketed)' if v is None else f'{v:+.3f}')(interp([(xj(a, s_), Fv(a, s_)) for a in DEFAULT if a in rungs], x_self))
                  + ' and d ' + (lambda v: 'n/a (unbracketed)' if v is None else f'{Fv("rw-peer", s_) - v:+.3f}')(interp([(xj(a, s_), Fv(a, s_)) for a in DEFAULT if a in rungs], x_peer))
                  + '. These decide nothing.']

        L += ['', '## By arm (means over seeds)', '',
              '| Arm | Judge | Head C trained on | x: the judge\'s accuracy | Δ_A of head B | F | a: agreement with head A | Sharpness, kept / replaced | Random share, s | Head A, minus clean | Rotated |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
        trained_on = {'rw-peer': 'head B\'s targets, its own discount', 'rw-b1-plain': '—', 'rw-b1-passenger': 'head B\'s targets, its own discount (not judging)',
                      **{a: f'clean targets at {FRAC[a]:.2%}, full weight' for a in FRAC}}
        judge_name = {'rw-b1': 'head B itself', 'rw-oracle': 'head A', 'rw-b1-plain': 'head B itself', 'rw-b1-passenger': 'head B itself'}
        for a in ['rw-b1', 'rw-oracle'] + new:
            L.append(f"| {a} | {judge_name.get(a, 'head C')} | {trained_on.get(a, '— (saved model)')} | {mean(lambda s: xj(a, s)):+.3f} | {mean(lambda s: dA(a, s)):+.3f} | "
                     f"{mean(lambda s: Fv(a, s)):+.3f} | {mean(lambda s: a_of(a, s)):.3f} | {mean(lambda s: sharp(a, s)[0]):.2f} | {mean(lambda s: sharp(a, s)[1]):.3f} | "
                     f"{mean(lambda s: rec(a, s)['functional']['A_clean'] - r[('clean', s)]['functional']['A_clean']):+.4f} | {mean(lambda s: ratio(rec(a, s), 'rotated_control')):.3f} |")
        for a in ladder_arms:
            L.append(f"| {a} | — | — | — | {mean(lambda s: dA(a, s)):+.3f} | {mean(lambda s: Fv(a, s)):+.3f} | {mean(lambda s: a_of(a, s)):.3f} | — | — | "
                     f"{mean(lambda s: rec(a, s)['functional']['A_clean'] - r[('clean', s)]['functional']['A_clean']):+.4f} | {mean(lambda s: ratio(rec(a, s), 'rotated_control')):.3f} |")
        cell = lambda v, k=3: 'n/a' if v is None else f'{v:+.{k}f}'
        L += ['', '## By seed', '',
              '| Seed | G | x_self | ' + ' | '.join(f'{a}: x, F' for a in rungs) + ' | F* | x_peer | F(rw-peer) | F_lf(x_peer) | d | 6a | 6b |',
              '|---|---|---|' + '---|' * len(rungs) + '---|---|---|---|---|---|---|']
        for s in seeds:
            L.append(f"| {s} | {G(s):+.3f} | {xj('rw-b1', s):+.3f} | " + ' | '.join(f'{xj(a, s):+.3f}, {Fv(a, s):+.3f}' for a in rungs)
                     + f" | {cell(F_star(s))} | {xj('rw-peer', s):+.3f} | {Fv('rw-peer', s):+.3f} | {cell(F_lf(s, xj('rw-peer', s)))} | {cell(d_of(s))} | "
                     f"{'yes' if ok6a(s) else 'no'} | {'yes' if ok6b(s) else 'no'} |")
        L += ['', '## Secondary and exploratory measures', '', 'None of these enters a decision rule.', '',
              '### S1: the peer\'s own recovery', '',
              f"F(rw-peer) = {fmt([Fv('rw-peer', s) for s in seeds])}, at the peer's accuracy x_peer = {fmt([xj('rw-peer', s) for s in seeds])}, "
              f"against head B's own x_self = {fmt([xj('rw-b1', s) for s in seeds])}. Does a second self-judged head, which needs no clean labels, help head B?", '',
              '### S2: each judge\'s sharpness and share at the end', '',
              'From the training-batch records at the last logged step: the mean weight on kept targets over the mean on replaced ones, and the random targets\' share '
              'of the summed weight (rw-b1 from N13\'s record, rw-oracle from its own).', '',
              '| Judge (arm) | x | Sharpness | Share, s | F |', '|---|---|---|---|---|']
        for a in ['rw-b1', 'rw-oracle', 'rw-peer'] + rungs:
            L.append(f"| {a} | {fmt([xj(a, s) for s in seeds])} | {fmt([sharp(a, s)[0] for s in seeds], 2, sign='')} | {fmt([sharp(a, s)[1] for s in seeds], sign='')} | {fmt([Fv(a, s) for s in seeds])} |")
        n_steps = len(trn[('rw-peer', seeds[0])]['judge_every_50'])
        marks = [0, 100, 200, 300, 500, 1000, 1500, 2000, 2850] if n_steps > 20 else [e[0] for e in trn[('rw-peer', seeds[0])]['judge_every_50']]
        at = lambda a, s, x: (lambda e: e[1] - e[2])(min(trn[(a, s)]['judge_every_50'], key=lambda e: abs(e[0] - x)))
        L += ['', '### S3: each judge\'s accuracy over training (means over seeds)', '',
              'The judge\'s loss on the batch\'s clean targets minus head A\'s, at the logged step nearest each listed step. N13 recorded head B\'s validation loss on its noisy '
              'targets, not on clean ones, so rw-b1\'s own course is not available; the peer, a self-judged head in the same runs, stands in for it.', '',
              '| Arm | ' + ' | '.join(f'step {x}' for x in marks) + ' |', '|---|' + '---|' * len(marks)]
        for a in ['rw-peer'] + rungs:
            L.append(f'| {a} | ' + ' | '.join(f"{mean(lambda s: at(a, s, x)):+.3f}" for x in marks) + ' |')
        jm = lambda a, s, k: rec(a, s)['judge_measure'][k]
        L += ['', '### S4: how far each judge\'s blind spot is head B\'s (means over seeds)', '',
              f'On the measurement windows\' clean targets: the fraction ignored (weight below {IGNORED_TXT}) under the judge and under head B\'s own discount, '
              'the share of head B\'s own ignored set that the judge also ignores (1 for rw-b1 by construction), and head B\'s loss on each set.', '',
              '| Arm | Judge ignores | Head B\'s own discount ignores | Share of own ignored also judge-ignored | Head B\'s loss on judge-ignored / own-ignored |', '|---|---|---|---|---|']
        for a in ['rw-b1', 'rw-oracle'] + new:
            L.append(f"| {a} | {mean(lambda s: jm(a, s, 'judge_ignored_fraction')):.4f} | {mean(lambda s: jm(a, s, 'own_ignored_fraction')):.4f} | "
                     f"{mean(lambda s: jm(a, s, 'share_of_own_ignored_also_judge_ignored')):.3f} | "
                     f"{mean(lambda s: jm(a, s, 'loss_B_on_judge_ignored')):.2f} / {mean(lambda s: jm(a, s, 'loss_B_on_own_ignored')):.2f} |")
        L += ['', '### S5: head A\'s cost', '',
              'Head A against clean (same seed): ' + ', '.join(f"{a} {fmt([rec(a, s)['functional']['A_clean'] - r[('clean', s)]['functional']['A_clean'] for s in seeds], 4)}"
                                                           for a in ['rw-b1', 'rw-oracle'] + new) + '.']

        def a_hat(s, x):
            return interp([(dA(a, s), a_of(a, s)) for a in ladder_arms], x)
        b_of = lambda a, s: None if a_hat(s, dA(a, s)) is None else a_hat(s, dA(a, s)) - a_of(a, s)
        L += ['', '### S6: geometry against the clean-target ladder', '',
              'b = â − a, with â the ladder\'s agreement at head B\'s Δ_A (straight lines between the rungs either side; '
              + ', '.join(ladder_arms) + '): ' + ', '.join(
                  f"{a} {fmt([b_of(a, s) for s in seeds], 4)} ({sum(b_of(a, s) is not None for s in seeds)} of {len(seeds)} bracketed)" for a in ['rw-b1', 'rw-oracle'] + new) + '.']
        L += ['', '### S7: the whole loop-free line', '',
              'Means over seeds of (x, F), from the most accurate loop-free judge, head A in rw-oracle (x = 0), down the rungs: '
              + '; '.join(f"{a} ({mean(lambda s: xj(a, s)):+.3f}, {mean(lambda s: Fv(a, s)):+.3f})" for a in ['rw-oracle'] + rungs)
              + f"; and the self-judged points, rw-b1 ({mean(lambda s: xj('rw-b1', s)):+.3f}, 0 by definition) and rw-peer "
                f"({mean(lambda s: xj('rw-peer', s)):+.3f}, {mean(lambda s: Fv('rw-peer', s)):+.3f}). Description only."]
        skipped = {a: sum(trn[(a, s)]['skipped_steps_fp16'] for s in seeds) for a in new}
        c_only = {a: sum(trn[(a, s)]['head_C_only_skipped_steps'] for s in seeds) for a in new}
        L += ['', '## The runs', '', '16-bit steps skipped by the main group, and steps skipped by head C alone, summed over seeds: '
              + ', '.join(f'{a} {skipped[a]} / {c_only[a]}' for a in new) + '. Resumes: '
              + ', '.join(f"{a} {sum(trn[(a, s)]['resumes'] for s in seeds)}" for a in new) + '.']
        envs = sorted({(x['environment']['gpu'], x['environment']['torch'], x['environment']['code_commit'][:7])
                       for x in measured + [trn[(a, s)] for a in new for s in seeds]})
        L += ['', '## Run environment', ''] + [f'- {gpu}, PyTorch {v}, code {cm}' for gpu, v, cm in envs] + ['']
    OUT.write_text('\n'.join(L))
    print('\n'.join(L))


IGNORED_TXT = '0.01'

if __name__ == '__main__':
    main()
