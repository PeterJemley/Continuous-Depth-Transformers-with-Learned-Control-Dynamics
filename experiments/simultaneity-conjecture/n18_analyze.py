"""Apply N18's preregistered decision rule to the saved records and write N18_RESULTS.md.

Written and committed before any real N18 run. Every number in the report comes from the JSON records: N18's own
(results/n18: m_* for the new arms, r_* for the saved models re-measured, c_* for the code identity against N17's saved models,
the training records, n18_summary.json), N12's n80 training records (fingerprints), N13's n80-rw training records (rw-b1's
weights over training), N14's training records (the nested positions' fingerprints) and the training records of the saved
rw-oracle (N16's, or N15's calibration model's). For the calibration run (one seed) it reports the same quantities and checks,
with no verdict, and whether the real run may be queued.

delta_A: head B's clean-target loss minus head A's, on the measurement windows.
R = (delta_A(rw-cloop) - delta_A(rw-ctwin)) / (delta_A(rw-peer) - delta_A(rw-j20)), within seed: the loop's cost to head B on clean
targets, as a share of the peer's gap to rw-j20. W = (delta_A(rw-ctwin) - delta_A(rw-j20)) / the same gap (secondary).
N18a: R's 95% interval wholly above 0.67 (refuted wholly below 0.33; some effect if it excludes 0).
G = delta_A(rw-b1) - delta_A(sub20); F(arm) = (delta_A(rw-b1) - delta_A(arm)) / G. x: the judge's plain cross-entropy minus head
A's; x_m: its mixture-form loss minus head A's copy's. The loop-free line (S8): the rungs' points (x, F), sorted, joined by
straight lines, nothing outside them; d(J) = F(J) - the line at J's x.

    python n18_analyze.py [results folder] [report path] [N12 records] [N13 records] [N14 records] [the saved rw-oracle's records]
    # defaults: results/n18 here and N18_RESULTS.md; for a calibration summary, results/n18_calib and N18_CALIBRATION.md;
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
CLAIM = ('rw-cloop', 'rw-ctwin', 'rw-j20', 'rw-peer')
SECONDARY = ('rw-j025', 'rw-j0125')
EXPLORATORY_ARM = 'rw-nfree'
FRAC = {'rw-j20': 0.2, 'rw-j10': 0.1, 'rw-j05': 0.05, 'rw-j025': 0.025, 'rw-j0125': 0.0125}
N14_TAG = {'rw-j025': 'sub025', 'rw-j0125': 'sub0125'}
LADDER_REAL = ('clean', 'sub20', 'sub10', 'sub05', 'sub025', 'sub0125')
WINDOW = 128 + 8
NEEDED = 4
LEVEL = 0.95
SUP, REF = 0.67, 0.33
TRAINED, MIN_DEN = 1.0, 0.05
REPRO_LOSS, REPRO_RATIO, MATCH_MIN, SEPARATE, NOISE_TOL, FRAC_TOL = 1e-3, 1e-2, 0.999, 0.60, 0.02, 0.02
CHECKS = ('weight_identity', 'forward_identity', 'detachment', 'weight_sources', 'loss_forms', 'unmixing', 'passenger', 'copied_peer', 'overflow', 'positions')
S_MARKS = (300, 500, 1000, 2000, 2850)
IGNORED_TXT = '0.01'


def load(folder, name):
    p = folder / f'{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


def interval(xs, level=LEVEL):
    n = len(xs)
    m = st.mean(xs)
    sd = st.stdev(xs) if n > 1 else float('nan')
    half = stats.t.ppf(1 - (1 - level) / 2, n - 1) * sd / math.sqrt(n) if n > 1 else float('nan')
    return m, sd, m - half, m + half


def fmt(xs, digits=3, sign='+'):
    xs = [x for x in xs if x is not None]
    if not xs:
        return 'n/a'
    if len(xs) == 1:
        return f'{xs[0]:{sign}.{digits}f}'
    m, sd, lo, hi = interval(xs)
    return f'{m:{sign}.{digits}f} [{lo:{sign}.{digits}f}, {hi:{sign}.{digits}f}]'


def interp(points, x):
    """Straight lines between points sorted by their first value; None outside their range."""
    pts = sorted(p for p in points if p[0] is not None)
    if x is None or not pts or not (pts[0][0] <= x <= pts[-1][0]):
        return None
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= x <= x1:
            return y0 if x1 == x0 else y0 + (y1 - y0) * (x - x0) / (x1 - x0)
    return pts[0][1] if len(pts) == 1 and pts[0][0] == x else None


def setup_parts(c):
    """Precondition 1's setup checks, split into those N18a needs and those that concern only rw-nfree."""
    det, ws, lf = c.get('detachment', {}), c.get('weight_sources', {}), c.get('loss_forms', {})
    claim_ok = (all(bool(c.get(k, {}).get('pass')) for k in ('weight_identity', 'forward_identity', 'passenger', 'copied_peer', 'overflow', 'positions'))
                and bool(det) and all(all(v.values()) for a, v in det.items() if a not in ('pass', EXPLORATORY_ARM))
                and bool(ws) and all(all(v.values()) for a, v in ws.items() if a not in ('pass', EXPLORATORY_ARM))
                and bool(lf) and all(v['pass'] for k, v in lf.items() if k != 'pass' and EXPLORATORY_ARM not in k)
                and bool(c.get('batch', {}).get('targets_are_i+1_to_i+8')) and bool(c.get('windows_shifted'))
                and bool(c.get('identical_heads', {}).get('pass')) and bool(c.get('analytic', {}).get('pass'))
                and all(x['pass'] for x in c.get('same_model', [{'pass': False}]))
                and abs(c.get('unigram', {}).get('sum', 0) - 1) < 1e-6)
    nfree_ok = (bool(c.get('unmixing', {}).get('pass')) and all(det.get(EXPLORATORY_ARM, {'x': False}).values())
                and all(ws.get(EXPLORATORY_ARM, {'x': False}).values()) and all(v['pass'] for k, v in lf.items() if k != 'pass' and EXPLORATORY_ARM in k))
    return claim_ok, nfree_ok


def main():
    res_arg = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    RES = res_arg or HERE / 'results' / 'n18'
    summary = load(RES, 'n18_summary') or {}
    cal = summary.get('mode') == 'calibration'
    if res_arg is None and not summary:
        RES = HERE / 'results' / 'n18_calib'
        summary = load(RES, 'n18_summary') or {}
        cal = summary.get('mode') == 'calibration'
    OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / ('N18_CALIBRATION.md' if cal else 'N18_RESULTS.md')
    sub = lambda k: HERE / 'results' / (f'{k}_calib' if cal else k)
    arg = lambda i, default: Path(sys.argv[i]) if len(sys.argv) > i else default
    n12_dir, n13_dir, n14_dir = arg(3, sub('n12')), arg(4, sub('n13')), arg(5, sub('n14'))
    oracle_dir = arg(6, HERE / 'results' / ('n15_calib' if cal else 'n16'))
    want = summary.get('seeds', REAL_SEEDS) if cal else REAL_SEEDS
    arms = list(summary.get('arms', {}))
    new = [a for a in CLAIM + SECONDARY + (EXPLORATORY_ARM,) if a in arms]
    n17_rungs = ['n17-rw-j20', 'n17-rw-j10', 'n17-rw-j05', 'n17-rw-j025', 'n17-rw-j0125'] if cal else ['n17-rw-j10', 'n17-rw-j05']
    ladder_arms = LADDER_REAL[:-1] if cal else LADDER_REAL
    refs = ['rw-b1', 'rw-oracle', *ladder_arms, 'n17-rw-peer', *n17_rungs]
    line_arms = n17_rungs if cal else ['rw-j20', 'n17-rw-j10', 'n17-rw-j05', 'rw-j025', 'rw-j0125']
    identity_arms = ('rw-j20', 'rw-peer') if cal else ('rw-peer',)
    c = summary.get('checks', {})
    cut = summary.get('held_out', {}).get('cut_token')
    m = {(a, s): load(RES, f'm_{a}_seed{s}') for a in new for s in want}
    r = {(a, s): load(RES, f'r_{a}_seed{s}') for a in refs for s in want}
    trn = {(a, s): load(RES, f'{a}_seed{s}') for a in new for s in want}
    cid = {(a, s): load(RES, f'c_{a}_seed{s}') for a in identity_arms for s in want}
    n80 = {s: load(n12_dir, f'n80_seed{s}') for s in want}
    b1 = {s: load(n13_dir, f'n80-rw_seed{s}') for s in want}
    orc = {s: load(oracle_dir, f'rw-oracle_seed{s}') for s in want}
    n14 = {(a, s): load(n14_dir, f'{N14_TAG[a]}_seed{s}') for a in N14_TAG if a in new for s in want}
    rec = lambda a, s: r[(a, s)] if a in refs else m.get((a, s))
    complete = lambda s: (all(m[(a, s)] and trn[(a, s)] for a in CLAIM if a in new) and all(r[(a, s)] for a in refs)
                          and all(cid[(a, s)] for a in identity_arms) and n80[s] and b1[s])
    seeds = [s for s in want if complete(s)]
    have = lambda a, s: rec(a, s) is not None and (a in refs or trn.get((a, s)) is not None)

    def a_of(a, s):
        x = rec(a, s)
        return (x.get('B_inner_vs_A') or x['B_vs_A'])['agreement_ratio']

    def dA(a, s):
        f = rec(a, s)['functional']
        return f['B_inner_clean'] - f['A_clean']

    def jx(a, s, key='x'):
        if a == 'rw-b1':
            return dA('rw-b1', s) if key == 'x' else rec(a, s)['judge_measure'][key]
        if a == 'rw-oracle' and key == 'x':
            return 0.0
        jm = rec(a, s).get('judge_measure')
        return jm[key] if jm else None

    twin_x = lambda s: m[('rw-ctwin', s)]['twin_measure']['x']
    den = lambda s: dA('rw-peer', s) - dA('rw-j20', s)
    R = lambda s: (dA('rw-cloop', s) - dA('rw-ctwin', s)) / den(s)
    Wsh = lambda s: (dA('rw-ctwin', s) - dA('rw-j20', s)) / den(s)
    rest = lambda s: (dA('rw-peer', s) - dA('rw-cloop', s)) / den(s)
    G = lambda s: dA('rw-b1', s) - dA('sub20', s)
    Fv = lambda a, s: (dA('rw-b1', s) - dA(a, s)) / G(s)
    line = lambda s, key: [(jx(a, s, key), Fv(a, s)) for a in line_arms if on_line(a, s)]
    d_of = lambda a, s, key='x': (lambda v: None if v is None else Fv(a, s) - v)(interp(line(s, key), jx(a, s, key)))

    def ratio(x, key):
        return x[key]['agreement_ratio']

    def run_ok(a, s):
        t = trn[(a, s)]
        ok = (t['training_tokens'] == cut and t['max_offset'] + WINDOW <= cut
              and t['batch_fingerprint_step0'] == n80[s]['batch_fingerprint_step0']
              and t['replaced_positions_fingerprint_step0'] == n80[s]['replaced_positions_fingerprint_step0']
              and abs(t['noise_fraction_step0'] - 0.8) <= NOISE_TOL)
        if a in ('rw-cloop', 'rw-ctwin', 'rw-j20'):
            ok = ok and t['judge_keep_is_K_step0'] is True
        if a == 'rw-ctwin':
            ok = ok and t['twin_keep_is_K_step0'] is True
        if a in N14_TAG:
            ok = ok and t['judge_keep_inside_K_step0'] and abs(t['judge_kept_fraction_step0'] - FRAC[a]) <= FRAC_TOL
            ref = n14.get((a, s))
            ok = ok and ref is not None and t['judge_keep_fingerprint_step0'] == ref['keep_fingerprint_step0']
        return ok

    on_line = lambda a, s: have(a, s) and (a in refs or run_ok(a, s))           # a rung trained here enters the line only if its run checks held
    claim_setup, nfree_setup = setup_parts(c)
    ident_ok = lambda s: all(cid[(a, s)].get('available') and cid[(a, s)].get('pass') for a in identity_arms)
    claim_new = [a for a in CLAIM if a in new]
    measured = [rec(a, s) for a in new + refs for s in seeds if rec(a, s) is not None]
    n_needed = min(NEEDED, len(want))
    pre = {
        '1. Setup: batch, same-model, identical-heads and analytic checks; the weight identity; the forward identity; the passenger check with two '
        'passenger heads; the copied-peer check; the overflow test, extended; detachment; the weight sources; the loss and gradient forms; '
        'the positions (head C\'s in rw-cloop, rw-ctwin and rw-j20, and the twin\'s, exactly K); windows; held-out tail untouched; each claim run\'s '
        'fingerprints and noise fraction equal N12\'s n80\'s; code identity with N17\'s saved models (bit-identical where the environment matches). '
        'A check that concerns only rw-nfree is reported under S7':
            claim_setup and cut is not None and bool(seeds) and all(run_ok(a, s) for a in claim_new for s in seeds) and all(ident_ok(s) for s in seeds),
        f'2. Must match: re-measuring the saved models reproduces their records (losses within {REPRO_LOSS:g}, ratios within {REPRO_RATIO:g}); '
        f'the constructed mixture check and the zero-noise copy (≥ {MATCH_MIN}) in every model':
            bool(seeds) and all(r[(a, s)].get('reproduction', {}).get('pass') for a in refs for s in seeds)
            and all(x['constructed_mixture']['pass'] and ratio(x, 'copy0_vs_A') >= MATCH_MIN for x in measured),
        f'3. Must differ: rotated control ≤ {SEPARATE} in at least {NEEDED} of 5 models of each of rw-cloop, rw-ctwin, rw-j20 and rw-peer':
            bool(seeds) and all(sum(ratio(rec(a, s), 'rotated_control') <= SEPARATE for s in seeds) >= n_needed for a in claim_new),
        f'4. The runs trained, in every seed: rw-j20\'s and rw-peer\'s Δ_A of head B and judge x, and rw-ctwin\'s twin x, at most {TRAINED}; '
        'every logged loss of the four arms finite':
            bool(seeds) and all(dA(a, s) <= TRAINED and jx(a, s) <= TRAINED for a in ('rw-j20', 'rw-peer') for s in seeds)
            and all(twin_x(s) <= TRAINED for s in seeds) and all(trn[(a, s)].get('logged_losses_finite') for a in claim_new for s in seeds),
        f'5. The peer\'s gap to rw-j20 exists: Δ_A(rw-peer) − Δ_A(rw-j20) ≥ {MIN_DEN} nats in every seed':
            bool(seeds) and all(den(s) >= MIN_DEN for s in seeds),
    }
    names = list(pre)
    nfree_ok = nfree_setup and EXPLORATORY_ARM in new and bool(seeds) and all(trn.get((EXPLORATORY_ARM, s)) and run_ok(EXPLORATORY_ARM, s) for s in seeds)

    L = ['# N18 results' if not cal else '# N18 calibration run', '',
         f'Generated by `n18_analyze.py` from the records in `{RES.name}/`, with N12\'s `n80` records in `{n12_dir.name}/`, N13\'s `n80-rw` records in `{n13_dir.name}/`, '
         f'N14\'s training records in `{n14_dir.name}/` and the saved `rw-oracle`\'s training records in `{oracle_dir.name}/`. Preregistration: `N18_PREREGISTRATION.md`.',
         f"Measured on the held-out tail's windows {summary.get('held_out', {}).get('measure')}, N12's. Δ_A is head B's clean-target loss minus head A's. "
         'R = (Δ_A(rw-cloop) − Δ_A(rw-ctwin)) / (Δ_A(rw-peer) − Δ_A(rw-j20)), the loop\'s cost to head B on clean targets as a share of the peer\'s gap to rw-j20; '
         'W = (Δ_A(rw-ctwin) − Δ_A(rw-j20)) / the same gap, the weighting\'s share without the loop. '
         'G = Δ_A(rw-b1) − Δ_A(sub20); F = (Δ_A(rw-b1) − Δ_A(arm)) / G; x a judge\'s plain cross-entropy on the clean targets minus head A\'s; '
         'x_m its mixture-form loss minus head A\'s copy\'s. '
         f'The claim\'s interval is a two-sided {LEVEL:.0%} t-interval; every other interval is a {LEVEL:.0%} t-interval and decides nothing.', '']
    if not cal:
        if len(seeds) < 5:
            verdict = f'**Not reached: complete records for only {len(seeds)} of 5 seeds.**'
        elif not all(pre.values()):
            verdict = f'**Untestable: a precondition failed ({"; ".join(k.split(":")[0] for k in names if not pre[k])}).**'
        else:
            mv, sd, lo, hi = interval([R(s) for s in seeds])
            tail = f'R = {mv:+.3f} [{lo:+.3f}, {hi:+.3f}], standard deviation {sd:.3f}; {LEVEL:.0%} interval, {len(seeds)} seeds.'
            if lo > SUP:
                verdict = ('**Supported: with the data and the weighting fixed, the judge\'s own loop is enough to produce most of the peer\'s gap.** ' + tail
                           + (' The interval lies wholly above 1: the loop on clean targets costs head B more than the peer\'s whole gap, which P and I both predict.' if lo > 1 else ''))
            elif hi < REF:
                reading = ('it contains 0: the loop\'s cost cannot be told from none, as W and D predict' if lo <= 0 <= hi else
                           'it lies wholly between 0 and 0.33: the loop costs a little' if lo > 0 else
                           'it lies wholly below 0: the loop helped, which no named rival predicts')
                verdict = f'**Refuted.** {tail} The reading: {reading}.'
            elif lo > 0 or hi < 0:
                verdict = f'**Some effect, size unclear.** {tail}'
            else:
                verdict = f'**Inconclusive.** {tail}'
        L += ['## Verdict', '', f'**N18a (the loop is enough to produce most of the peer\'s gap):** {verdict}', '',
              'A support is read as "the loop, with what it does to the weighter\'s accuracy and course", not as a cost beyond the judge\'s plain accuracy '
              '(rival P, tested by S8), nor as showing that the peer\'s own gap is the loop\'s (rival I).', '']
    else:
        L += ['This is the calibration run of the preregistration\'s calibration step: one seed, on windows the real run does not use. '
              'It may change nothing in the design, and no verdict is computed. `rw-oracle` is N15\'s calibration model of the same seed, re-measured; '
              'the loop-free line (S8) runs through N17\'s five calibration rungs.', '']
        gates = {'every setup check passed, other than one concerning only rw-nfree': claim_setup and bool(seeds) and all(run_ok(a, s) for a in claim_new for s in seeds),
                 'the code identity held where the environment matches': bool(seeds) and all(ident_ok(s) for s in seeds),
                 'precondition 4 held in seed 3001': pre[names[3]],
                 'precondition 5 held in seed 3001': pre[names[4]]}
        L += ['## May the real run be queued?', '', '| Condition | Met |', '|---|---|'] + [f'| {k} | {"yes" if v else "**no**"} |' for k, v in gates.items()]
        L += ['', ('**Yes: the real run may be queued.**' if all(gates.values()) else
                   '**No.** By the preregistration, the real run is not queued, and the values go to the author.'), '']
    L += ['## Preconditions', '', '| Precondition | Met |', '|---|---|']
    L += [f'| {k} | {"yes" if ok else "**no**"} |' for k, ok in pre.items()]
    if c:
        lf = c.get('loss_forms', {})
        L += ['', '**The setup checks:** ' + '; '.join(f"{k} {'passed' if c.get(k, {}).get('pass') else '**failed**'}" for k in CHECKS) + '.']
        if 'passenger' in c and 'copied_peer' in c:
            L += [f"Passenger, with two passenger heads: largest difference {c['passenger']['max_difference']:.1e} over {c['passenger']['steps']} steps; copied peer: "
                  f"largest differences {c['copied_peer']['max_main_difference']:.1e} (main) and {c['copied_peer']['max_head_C_difference']:.1e} (head C against head B)."]
        if lf:
            L += ['The loss and gradient forms (relative differences; tolerance 1e-05): ' + '; '.join(
                f"{k}: loss {v['relative_loss_difference']:.1e}, gradient {v['relative_gradient_difference']:.1e}" for k, v in lf.items() if k != 'pass') + '.']
        if 'unmixing' in c:
            u_ = c['unmixing']
            L += [f"The unmixing identity: relative difference {u_['identity_relative_difference']:.1e}; the floor: {u_['targets_below_floor']} of {u_['targets']} "
                  f"constructed targets below 0.8 u, all weighted exactly 0 ({u_['zero_below_floor']}), elsewhere relative difference {u_['floor_relative_difference_elsewhere']:.1e}."]
    if seeds:
        L += ['', '**Code identity with N17\'s saved models:** ' + '; '.join(
            f"{a} seed {s}: " + ('not available' if not cid[(a, s)].get('available') else
                                 f"environment {'matches' if cid[(a, s)]['environment_matches'] else 'differs'}, "
                                 f"{'bit-identical' if cid[(a, s)]['identical'] else 'not bit-identical'} (largest difference {cid[(a, s)]['max_difference']})")
            for a in identity_arms for s in seeds) + '.']

    if seeds:
        def mean(f, over=None):
            xs = [v for v in (f(s) for s in (over or seeds)) if v is not None]
            return st.mean(xs) if xs else float('nan')

        cell = lambda v, k=3: 'n/a' if v is None else f'{v:+.{k}f}'
        L += ['', '## By seed', '',
              '| Seed | Δ_A rw-cloop | Δ_A rw-ctwin | Δ_A rw-j20 | Δ_A rw-peer | The peer\'s gap to rw-j20 | R | W | 1 − R − W |',
              '|---|---|---|---|---|---|---|---|---|']
        for s in seeds:
            L.append(f"| {s} | {dA('rw-cloop', s):+.4f} | {dA('rw-ctwin', s):+.4f} | {dA('rw-j20', s):+.4f} | {dA('rw-peer', s):+.4f} | {den(s):+.4f} | "
                     f"{R(s):+.3f} | {Wsh(s):+.3f} | {rest(s):+.3f} |")
        if cal:
            s_ = seeds[0]
            L += ['', f"**Seed {s_}'s values, for the record:** R {R(s_):+.3f}, W {Wsh(s_):+.3f}, 1 − R − W {rest(s_):+.3f}; the denominator "
                  f"{den(s_):+.4f} nats; G {G(s_):+.3f}. These decide nothing."]
        judge_name = {'rw-b1': 'head B itself', 'rw-oracle': 'head A'}
        trained_on = {'rw-cloop': 'clean targets at K, its own discount', 'rw-ctwin': "clean targets at K, the twin's discount",
                      'rw-j20': 'clean targets at K, full weight', 'rw-j025': 'clean targets at 2.5%, full weight', 'rw-j0125': 'clean targets at 1.25%, full weight',
                      'rw-peer': "head B's targets, its own discount", 'rw-nfree': "head B's targets, full weight (weight unmixed)",
                      'n17-rw-peer': "N17's peer, saved", **{f'n17-{a}': f'N17\'s, clean targets at {FRAC[a]:.2%}, saved' for a in FRAC}}
        shown = ['rw-b1', 'rw-oracle'] + new + ['n17-rw-peer'] + n17_rungs
        L += ['', '## S1: by arm (means over seeds)', '',
              '| Arm | Judge | The judge trained on | x | x_m | Δ_A of head B | F | a: agreement with head A | Head A, minus clean | Rotated |',
              '|---|---|---|---|---|---|---|---|---|---|']
        for a in shown:
            L.append(f"| {a} | {judge_name.get(a, 'head C')} | {trained_on.get(a, '—')} | {mean(lambda s: jx(a, s)):+.3f} | {mean(lambda s: jx(a, s, 'x_m')):+.3f} | "
                     f"{mean(lambda s: dA(a, s)):+.3f} | {mean(lambda s: Fv(a, s)):+.3f} | {mean(lambda s: a_of(a, s)):.3f} | "
                     f"{mean(lambda s: rec(a, s)['functional']['A_clean'] - r[('clean', s)]['functional']['A_clean']):+.4f} | {mean(lambda s: ratio(rec(a, s), 'rotated_control')):.3f} |")
        for a in ladder_arms:
            L.append(f"| {a} | — | — | — | — | {mean(lambda s: dA(a, s)):+.3f} | {mean(lambda s: Fv(a, s)):+.3f} | {mean(lambda s: a_of(a, s)):.3f} | "
                     f"{mean(lambda s: rec(a, s)['functional']['A_clean'] - r[('clean', s)]['functional']['A_clean']):+.4f} | {mean(lambda s: ratio(rec(a, s), 'rotated_control')):.3f} |")
        L += ['', f"The twin's x in rw-ctwin: {fmt([twin_x(s) for s in seeds])}. W, the weighting's share without the loop: {fmt([Wsh(s) for s in seeds])}; "
              f"1 − R − W, the data's share with the loop present: {fmt([rest(s) for s in seeds])}."]

        def at(a, s, x_, field, f):
            """f of the training record's entry in field logged nearest step x_."""
            t = trn.get((a, s))
            if not t or not t.get(field):
                return None
            return f(min(t[field], key=lambda e: abs(e[0] - x_)))

        sharp_f = lambda e: e[2] / e[3] if e[3] else None
        share_f = lambda e: e[4]
        plain_f = lambda e: e[1] - e[2]
        mix_f = lambda e: e[4] - e[5]
        twin_f = lambda e: None if e[6] is None else e[6] - e[2]
        n_log = len(trn[(claim_new[0], seeds[0])]['w_every_50'])
        marks = S_MARKS if n_log > 20 else tuple(e[0] for e in trn[(claim_new[0], seeds[0])]['w_every_50'])
        L += ['', '## Secondary and exploratory measures', '', 'None of these enters the decision rule.', '',
              '### S2: each judge\'s sharpness and random share over training (means over seeds)', '',
              'From the training-batch records at the logged step nearest each listed step: head B\'s mean weight on kept targets over its mean on replaced ones, '
              'and the random targets\' share of the summed weight.', '',
              '| Arm | ' + ' | '.join(f'step {x_}: sharpness, share' for x_ in marks) + ' |', '|---|' + '---|' * len(marks)]
        for a in [x_ for x_ in ('rw-cloop', 'rw-ctwin', 'rw-j20', 'rw-peer', 'rw-nfree') if x_ in new]:
            L.append(f'| {a} | ' + ' | '.join(f"{mean(lambda s: at(a, s, x_, 'w_every_50', sharp_f)):.2f}, {mean(lambda s: at(a, s, x_, 'w_every_50', share_f)):.3f}" for x_ in marks) + ' |')
        L += ['', '### S3: each judge\'s accuracy over training (means over seeds)', '',
              'On the batch\'s clean targets at the logged step nearest each listed step: the judge\'s plain cross-entropy minus head A\'s, and its mixture-form loss '
              'minus head A\'s copy\'s; for rw-ctwin, the twin\'s plain cross-entropy minus head A\'s as well.', '',
              '| Arm | Measure | ' + ' | '.join(f'step {x_}' for x_ in marks) + ' |', '|---|---|' + '---|' * len(marks)]
        for a in [x_ for x_ in ('rw-cloop', 'rw-ctwin', 'rw-j20', 'rw-peer', 'rw-j025', 'rw-j0125', 'rw-nfree') if x_ in new]:
            L.append(f'| {a} | plain | ' + ' | '.join(f"{mean(lambda s: at(a, s, x_, 'judge_every_50', plain_f)):+.3f}" for x_ in marks) + ' |')
            L.append(f'| {a} | mixture | ' + ' | '.join(f"{mean(lambda s: at(a, s, x_, 'judge_every_50', mix_f)):+.3f}" for x_ in marks) + ' |')
            if a == 'rw-ctwin':
                L.append(f'| {a} | the twin, plain | ' + ' | '.join(f"{mean(lambda s: at(a, s, x_, 'judge_every_50', twin_f)):+.3f}" for x_ in marks) + ' |')
        jm = lambda a, s, k: (rec(a, s).get('judge_measure') or {}).get(k)
        L += ['', '### S4: the blind spots (means over seeds)', '',
              f'On the measurement windows\' clean targets: the fraction ignored (weight below {IGNORED_TXT}) under the judge and under head B\'s own discount, '
              'the share of each set also in the other, and head B\'s loss on each set.', '',
              '| Arm | Judge ignores | Head B\'s own discount ignores | Share of own also judge-ignored | Share of judge-ignored also own | Head B\'s loss on judge-ignored / own-ignored |',
              '|---|---|---|---|---|---|']
        for a in ['rw-b1', 'rw-oracle'] + new:
            L.append(f"| {a} | {mean(lambda s: jm(a, s, 'judge_ignored_fraction')):.4f} | {mean(lambda s: jm(a, s, 'own_ignored_fraction')):.4f} | "
                     f"{mean(lambda s: jm(a, s, 'share_of_own_ignored_also_judge_ignored')):.3f} | {mean(lambda s: jm(a, s, 'share_of_judge_ignored_also_own_ignored')):.3f} | "
                     f"{mean(lambda s: jm(a, s, 'loss_B_on_judge_ignored')):.2f} / {mean(lambda s: jm(a, s, 'loss_B_on_own_ignored')):.2f} |")
        L += ['', '### S5: head A\'s cost', '',
              'Head A against clean (same seed): ' + ', '.join(f"{a} {fmt([rec(a, s)['functional']['A_clean'] - r[('clean', s)]['functional']['A_clean'] for s in seeds], 4)}"
                                                           for a in new) + '.']

        def a_hat(s, x_):
            return interp([(dA(a, s), a_of(a, s)) for a in ladder_arms], x_)
        b_of = lambda a, s: None if a_hat(s, dA(a, s)) is None else a_hat(s, dA(a, s)) - a_of(a, s)
        L += ['', '### S6: geometry against the clean-target ladder', '',
              'b = â − a, with â the ladder\'s agreement at head B\'s Δ_A (straight lines between the rungs either side; ' + ', '.join(ladder_arms) + '): '
              + ', '.join(f"{a} {fmt([b_of(a, s) for s in seeds], 4)} ({sum(b_of(a, s) is not None for s in seeds)} of {len(seeds)} bracketed)" for a in new) + '.']
        L += ['', '### S7: rw-nfree, the loop-free judge on noisy data (exploratory)', '']
        if EXPLORATORY_ARM in new and all(m.get((EXPLORATORY_ARM, s)) for s in seeds):
            if not nfree_ok:
                L.append('**A check concerning rw-nfree failed, so its results are unreportable** (by the preregistration). The values below are not to be read.')
            nf = EXPLORATORY_ARM
            L.append(f"F {fmt([Fv(nf, s) for s in seeds])}; x_m, its mixture-form gap to head A's copy on the clean targets, {fmt([jx(nf, s, 'x_m') for s in seeds])}; "
                     f"the fraction of clean targets its weight floors at 0, {fmt([jx(nf, s, 'floored_fraction') for s in seeds], 4, '')}; "
                     f"the mean mass of max(q, 0.8 u) before normalisation, {fmt([jx(nf, s, 'mean_mass_before_normalisation') for s in seeds], 4, '')}. "
                     f"Δ_A(rw-nfree) − Δ_A(rw-peer), the loop with noisy data: {fmt([dA(nf, s) - dA('rw-peer', s) for s in seeds], 4)}; "
                     f"Δ_A(rw-nfree) − Δ_A(rw-j20), the data without the loop: {fmt([dA(nf, s) - dA('rw-j20', s) for s in seeds], 4)}. "
                     'Each of these also changes the estimator.')
        else:
            L.append('No records.')
        L += ['', '### S8: P, tested directly (secondary)', '',
              'The loop-free line on x within seed through ' + ', '.join(line_arms) + '; d = F − the line at the judge\'s x, over the seeds where the line reaches it. '
              'P predicts d near 0 for rw-cloop and rw-ctwin; d(rw-cloop) below 0 would mean the loop costs head B beyond what its judge\'s plain accuracy predicts. '
              'The same on x_m is description; a judge more accurate in mixture form than the line\'s top rung is beyond it.', '',
              '| Judge | x | d on x (seeds reached) | x_m | d on x_m (seeds reached) |', '|---|---|---|---|---|']
        for a in [x_ for x_ in ('rw-cloop', 'rw-ctwin', 'rw-peer', 'rw-j20') if x_ in new]:
            dx = [d_of(a, s) for s in seeds]
            dm = [d_of(a, s, 'x_m') for s in seeds]
            L.append(f"| {a} | {fmt([jx(a, s) for s in seeds])} | {fmt(dx)} ({sum(v is not None for v in dx)} of {len(seeds)}) | "
                     f"{fmt([jx(a, s, 'x_m') for s in seeds])} | {fmt(dm)} ({sum(v is not None for v in dm)} of {len(seeds)}) |")
        off = [f'{a} seed {s}' for a in line_arms for s in seeds if have(a, s) and not on_line(a, s)]
        if off:
            L += ['', 'Left out of the line because their run checks (fingerprints, positions, held-out tail) failed: ' + ', '.join(off) + '.']
        L += ['', 'The line, means over seeds of (x, x_m, F): ' + '; '.join(
            f"{a} ({mean(lambda s: jx(a, s) if have(a, s) else None):+.3f}, {mean(lambda s: jx(a, s, 'x_m') if have(a, s) else None):+.3f}, "
            f"{mean(lambda s: Fv(a, s) if have(a, s) else None):+.3f})" for a in line_arms) + '.']
        skipped = {a: sum(trn[(a, s)]['skipped_steps_fp16'] for s in seeds if trn.get((a, s))) for a in new}
        c_only = {a: sum(trn[(a, s)]['head_C_only_skipped_steps'] for s in seeds if trn.get((a, s))) for a in new}
        t_only = {a: sum(trn[(a, s)].get('twin_only_skipped_steps', 0) for s in seeds if trn.get((a, s))) for a in new}
        L += ['', '## The runs', '', '16-bit steps skipped by the main group, by head C alone and by the twin alone, summed over seeds: '
              + ', '.join(f'{a} {skipped[a]} / {c_only[a]} / {t_only[a]}' for a in new) + '. Resumes: '
              + ', '.join(f"{a} {sum(trn[(a, s)]['resumes'] for s in seeds if trn.get((a, s)))}" for a in new) + '.']
        envs = sorted({(x['environment']['gpu'] or 'cpu', x['environment']['torch'], x['environment']['code_commit'][:7])
                       for x in measured + [trn[(a, s)] for a in new for s in seeds if trn.get((a, s))]})
        L += ['', '## Run environment', ''] + [f'- {gpu}, PyTorch {v}, code {cm}' for gpu, v, cm in envs] + ['']
    OUT.write_text('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
