"""Apply N16's preregistered decision rules to the saved records and write N16_RESULTS.md.

Written and committed before any real N16 run. Every number in the report comes from the JSON records: N16's own
(results/n16: m_* for the new arms, r_* for the saved models re-measured, the training records, n16_summary.json), N12's
n80 training records (fingerprints) and N13's n80-rw training records (the share at beta = 1). For the calibration run
(one seed) it reports the same quantities and checks, with no verdicts, and applies the calibration's one rule, which
decides only whether each exploratory arm runs; there rw-oracle is N15's calibration model, re-measured, with its training
record in results/n15_calib.

delta_A: the inner reading's clean-target loss minus head A's; a: its agreement with head A. G = delta_A(rw-b1) - delta_A(sub20),
the shortfall; F = (delta_A(rw-b1) - delta_A(rw-oracle)) / G, the recovered fraction; s(arm): the random targets' share of the
summed weight at the end of training; P = 1 - s(rw-oracle) / s(rw-b1), the removed share; C = delta_A(sub20-oracle) -
delta_A(sub20-rw), the clean-target cost of head A's judgement against head B's own. The ladder: (delta_A, a) of clean,
sub20, sub10, sub05, sub025, sub0125; a_hat interpolates a at a model's delta_A; b = a_hat - a.
N16a: F wholly above 0.75. N16b: P wholly below 0.5. N16c: C wholly below -0.30. N16d: b(rw-oracle) wholly within 0.04.

Two readings fixed at build time, before the calibration (see the preregistration's deviation log): preconditions 1-3 are
applied to the confirmatory arms and the references, since the exploratory arms carry no claim (their checks are reported
under S2); precondition 4 is applied per head, 4a (rw-oracle) gating N16a and N16b, 4b (sub20-oracle) gating N16c.

    python n16_analyze.py [results folder] [report path] [N12 records] [N13 records] [N15 records, calibration only]
    # defaults: results/n16 here, N16_RESULTS.md, results/n12, results/n13 (or the _calib folders by the summary's mode), results/n15_calib
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

from scipy import stats

HERE = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'results' / 'n16'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'N16_RESULTS.md'

REAL_SEEDS = [4101, 4102, 4103, 4104, 4105]
CONF = ('rw-oracle', 'sub20-oracle')                     # the confirmatory new arms
EXPL = ('rw-b2w', 'rw-b4w')                              # exploratory, no claim; the calibration decides whether they run
LABEL = {'rw-b1': 'head B judges, β = 1 (N13)', 'rw-oracle': 'head A judges', 'sub20-oracle': 'head A judges, clean targets only',
         'sub20-rw': 'head B judges, clean targets only (N14)', 'rw-b2w': 'head B judges, β = 2 after the warm-up', 'rw-b4w': 'head B judges, β = 4 after the warm-up'}
RUNG_ALL = ('clean', 'sub20', 'sub10', 'sub05', 'sub025', 'sub0125')
WINDOW = 128 + 8
NEEDED = 4
LEVEL = 1 - 0.05 / 4                                     # four interval rules: Bonferroni
F_HIGH, F_LOW, F_SD = 0.75, 0.25, 0.15                   # N16a
P_LOW, P_HIGH, P_SD = 0.5, 0.75, 0.15                    # N16b: supported below P_LOW, refuted above P_HIGH
C_SUP, C_REF, C_SD = -0.30, -0.10, 0.10                  # N16c
LADDER_TOL, LADDER_SD = 0.04, 0.03                       # N16d
TRAINED, MIN_G = 1.0, 0.10                               # preconditions 4 and 5
REPRO_LOSS, REPRO_RATIO, MATCH_MIN, SEPARATE, NOISE_TOL = 1e-3, 1e-2, 0.999, 0.60, 0.02
IGNORED = 0.01


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
    xs = [x for x in xs if x is not None]
    if not xs:
        return 'n/a'
    if len(xs) == 1:
        return f'{xs[0]:+.{digits}f}'
    m, sd, lo, hi = interval(xs, level)
    return f'{m:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}]'


def main():
    summary = load(RES, 'n16_summary') or {}
    mode = summary.get('mode', 'real')
    cal = mode == 'calibration'
    n12_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / 'results' / ('n12_calib' if cal else 'n12')
    n13_dir = Path(sys.argv[4]) if len(sys.argv) > 4 else HERE / 'results' / ('n13_calib' if cal else 'n13')
    n15_dir = Path(sys.argv[5]) if len(sys.argv) > 5 else HERE / 'results' / 'n15_calib'
    want = summary.get('seeds', REAL_SEEDS) if cal else REAL_SEEDS
    rung = RUNG_ALL[:-1] if cal else RUNG_ALL
    dropped = set(summary.get('dropped', []))
    trained = [a for a in CONF + EXPL if not (cal and a == 'rw-oracle') and a not in dropped]
    refs = ('rw-b1', 'sub20-rw') + rung + (('rw-oracle',) if cal else ())
    c = summary.get('checks', {})
    cut = summary.get('held_out', {}).get('cut_token')
    m = {(a, s): load(RES, f'm_{a}_seed{s}') for a in trained for s in want}
    r = {(a, s): load(RES, f'r_{a}_seed{s}') for a in refs for s in want}
    trn = {(a, s): load(RES, f'{a}_seed{s}') for a in trained for s in want}
    if cal:
        for s in want:
            trn[('rw-oracle', s)] = load(n15_dir, f'rw-oracle_seed{s}')
    n80 = {s: load(n12_dir, f'n80_seed{s}') for s in want}
    b1 = {s: load(n13_dir, f'n80-rw_seed{s}') for s in want}
    rec = lambda a, s: r[(a, s)] if a in refs else m.get((a, s))
    seeds = [s for s in want if all(rec(a, s) for a in CONF) and all(r[(a, s)] for a in refs) and all(trn.get((a, s)) for a in CONF) and n80[s] and b1[s]]
    expl = {a: [s for s in seeds if m.get((a, s)) and trn.get((a, s))] for a in EXPL if a in trained}
    expl_ran = [a for a in expl if expl[a]]

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

    def ignored_end(a, s):
        return trn[(a, s)]['w_every_50'][-1][5]

    G = lambda s: dA('rw-b1', s) - dA('sub20', s)
    Fv = lambda a, s: (dA('rw-b1', s) - dA(a, s)) / G(s)
    Pv = lambda a, s: 1 - share(a, s) / share('rw-b1', s)
    Cv = lambda s: dA('sub20-oracle', s) - dA('sub20-rw', s)
    S3 = lambda s: dA('sub20-oracle', s) - dA('sub20', s)

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

    def ng(a, s, key):
        return rec(a, s).get('neglect', {}).get(key)

    def run_ok(a, s):
        t = trn[(a, s)]
        return (t['training_tokens'] == cut and t['max_offset'] + WINDOW <= cut
                and t['batch_fingerprint_step0'] == n80[s]['batch_fingerprint_step0']
                and t['replaced_positions_fingerprint_step0'] == n80[s]['replaced_positions_fingerprint_step0']
                and abs(t['noise_fraction_step0'] - 0.8) <= NOISE_TOL)

    claim_arms = CONF + refs
    all_measured = [rec(a, s) for a in claim_arms for s in seeds]
    n_needed = min(NEEDED, len(want))
    pre = {
        '1. Setup: batch, same-model, identical-heads and analytic checks; the weight identity at β = 1; the oracle\'s detachment; the warm-up switch; windows; '
        'held-out tail untouched; each confirmatory run\'s fingerprints and noise fraction equal N12\'s n80\'s (sub20-oracle\'s kept set is K)':
            bool(c.get('batch', {}).get('targets_are_i+1_to_i+8')) and bool(c.get('windows_shifted')) and bool(c.get('identical_heads', {}).get('pass'))
            and bool(c.get('analytic', {}).get('pass')) and bool(c.get('weight_identity', {}).get('pass')) and bool(c.get('oracle_detachment', {}).get('pass'))
            and bool(c.get('warmup_switch', {}).get('pass')) and all(x['pass'] for x in c.get('same_model', [{'pass': False}]))
            and cut is not None and bool(seeds) and all(run_ok(a, s) for a in CONF for s in seeds),
        f'2. Must match: re-measuring the saved models reproduces their records (losses within {REPRO_LOSS:g}, ratios within {REPRO_RATIO:g}); '
        f'the constructed mixture check and the zero-noise copy (≥ {MATCH_MIN}) in every model':
            bool(seeds) and all(r[(a, s)].get('reproduction', {}).get('pass') for a in refs for s in seeds)
            and all(x['constructed_mixture']['pass'] and ratio(x, 'copy0_vs_A') >= MATCH_MIN for x in all_measured),
        f'3. Must differ: rotated control ≤ {SEPARATE} in at least {NEEDED} of 5, in every arm':
            bool(seeds) and all(sum(ratio(rec(a, s), 'rotated_control') <= SEPARATE for s in seeds) >= n_needed for a in claim_arms),
        f'4a. The head trained: Δ_A(rw-oracle) ≤ {TRAINED} in at least {NEEDED} of 5 seeds':
            sum(dA('rw-oracle', s) <= TRAINED for s in seeds) >= n_needed,
        f'4b. The head trained: Δ_A(sub20-oracle) ≤ {TRAINED} in at least {NEEDED} of 5 seeds':
            sum(dA('sub20-oracle', s) <= TRAINED for s in seeds) >= n_needed,
        f'5. The shortfall exists: G = Δ_A(rw-b1) − Δ_A(sub20) ≥ {MIN_G} nats in every seed':
            bool(seeds) and all(G(s) >= MIN_G for s in seeds),
        f'6. The ladder brackets rw-oracle and is ordered in at least {NEEDED} of 5 seeds':
            sum(ordered(s) and b('rw-oracle', s) is not None for s in seeds) >= n_needed,
    }
    names = list(pre)
    pooled_seeds = [s for s in seeds if ordered(s) and b('rw-oracle', s) is not None]

    def gate(needed):
        if len(seeds) < 5:
            return f'**Not reached: complete records for only {len(seeds)} of 5 seeds.**'
        bad = [names[i].split(':')[0] for i in needed if not pre[names[i]]]
        return f'**Untestable: a precondition failed ({"; ".join(bad)}).**' if bad else None

    def one_sided(xs, quantity, sup_side, sup_margin, ref_margin, sd_cap, sup_text, digits=3):
        """sup_side 'above': supported if the interval lies wholly above sup_margin, refuted if wholly below ref_margin;
        'below': the reverse. Some effect if the interval excludes 0; else inconclusive."""
        mv, sd, lo, hi = interval(xs)
        tail = f'{quantity} = {mv:+.{digits}f} [{lo:+.{digits}f}, {hi:+.{digits}f}], standard deviation {sd:.{digits}f}; {LEVEL:.2%} interval, {len(xs)} seeds.'
        sup = lo > sup_margin if sup_side == 'above' else hi < sup_margin
        ref = (hi < ref_margin if sup_side == 'above' else lo > ref_margin) and sd <= sd_cap
        if sup:
            return f'**Supported: {sup_text}** {tail}'
        if ref:
            return f'**Refuted.** {tail}'
        if lo > 0 or hi < 0:
            return f'**Some effect, size unclear.** {tail}'
        return f'**Inconclusive.** {tail}'

    L = ['# N16 results' if not cal else '# N16 calibration run', '',
         f'Generated by `n16_analyze.py` from the records in `{RES.name}/`, with N12\'s `n80` records in `{n12_dir.name}/` and N13\'s `n80-rw` records in `{n13_dir.name}/`'
         + (f', and N15\'s calibration `rw-oracle` training record in `{n15_dir.name}/`' if cal else '') + '. Preregistration: `N16_PREREGISTRATION.md`.',
         f"Measured on the held-out tail's windows {summary.get('held_out', {}).get('measure')}, N12's. Δ_A is the inner reading's clean-target loss minus head A's; a its agreement with head A; "
         'G = Δ_A(rw-b1) − Δ_A(sub20) the shortfall; F = (Δ_A(rw-b1) − Δ_A(rw-oracle)) / G the recovered fraction; s the random targets\' share of the summed weight at the end of training; '
         'P = 1 − s(rw-oracle) / s(rw-b1) the share removed; C = Δ_A(sub20-oracle) − Δ_A(sub20-rw) the clean-target cost of head A\'s judgement against head B\'s own; '
         'â the clean-target ladder\'s agreement at the model\'s Δ_A; b = â − a. '
         f'Intervals are Bonferroni-corrected t-intervals at {LEVEL:.2%}.', '']
    if not cal:
        v_a = gate([0, 1, 2, 3, 5]) or one_sided([Fv('rw-oracle', s) for s in seeds], 'F', 'above', F_HIGH, F_LOW, F_SD,
                                                 'the head-A-judged discount recovers the shortfall.')
        v_b = gate([0, 1, 2, 3, 5]) or one_sided([Pv('rw-oracle', s) for s in seeds], 'P', 'below', P_LOW, P_HIGH, P_SD,
                                                 'the share is not the mechanism; the recovery came with less than half the random share removed.')
        v_c = gate([0, 1, 2, 4]) or one_sided([Cv(s) for s in seeds], 'C', 'below', C_SUP, C_REF, C_SD,
                                              'on clean targets alone, head A\'s judgement costs far less than head B\'s own: self-reference is the mechanism.')
        v_d = gate([0, 1, 2, 6])
        if not v_d:
            xs = [b('rw-oracle', s) for s in pooled_seeds]
            mb, sdb, lob, hib = interval(xs)
            tail = f'b = {mb:+.4f} [{lob:+.4f}, {hib:+.4f}], standard deviation {sdb:.4f}; {LEVEL:.2%} interval, {len(xs)} seeds.'
            if lob > -LADDER_TOL and hib < LADDER_TOL:
                v_d = f'**Supported: the head-A-judged head sits on the clean-target ladder.** {tail}'
            elif (hib < -LADDER_TOL or lob > LADDER_TOL) and sdb <= LADDER_SD:
                v_d = f'**Refuted ({"below" if lob > LADDER_TOL else "above"} the ladder).** {tail}'
            elif lob > 0 or hib < 0:
                v_d = f'**Some effect, size unclear.** {tail}'
            else:
                v_d = f'**Inconclusive.** {tail}'
        L += ['## Verdicts', '',
              f'**N16a (primary: the head-A-judged discount recovers the shortfall):** {v_a}', '',
              f'**N16b (the share is not the mechanism):** {v_b}', '',
              f'**N16c (self-reference is the mechanism):** {v_c}', '',
              f'**N16d (geometry follows function):** {v_d}', '']
    else:
        L += ['This is the calibration run of the preregistration\'s calibration step: one seed, on windows the real run does not use. '
              'It decides nothing but whether each exploratory arm runs, and no verdict is computed. `rw-oracle` is N15\'s calibration model of the same seed, re-measured.', '']
    L += ['## Preconditions', '', '| Precondition | Met |', '|---|---|']
    L += [f'| {k} | {"yes" if ok else "**no**"} |' for k, ok in pre.items()]
    if dropped:
        L += ['', f'Exploratory arms dropped before this run by the calibration\'s rule: {", ".join(sorted(dropped))} (see the preregistration\'s deviation log).']

    if seeds:
        def mean(f, over=None):
            xs = [v for v in (f(s) for s in (over or seeds)) if v is not None]
            return st.mean(xs) if xs else float('nan')

        def hat_cell(a, over):
            h = mean(lambda s: a_hat(s, dA(a, s)), over)
            return 'n/a' if math.isnan(h) else f'{h:.3f}'
        with_share = ['rw-b1', 'rw-oracle'] + expl_ran
        if cal:
            s_ = seeds[0]
            parts, drop = [], []
            for a in EXPL:
                if a in expl_ran:
                    ok = dA(a, s_) <= TRAINED
                    parts.append(f'{a}: Δ_A {dA(a, s_):.3f}, share {share(a, s_):.3f}, ignored kept {ignored_end(a, s_):.3f}, so it {"runs" if ok else "is dropped"}')
                    if not ok:
                        drop.append(a)
                else:
                    parts.append(f'{a}: no record')
            L += ['', f'**The calibration\'s rule for the exploratory arms** (a warm-up arm runs on the real seeds if its head trains in seed {s_}, Δ_A ≤ {TRAINED}): '
                  + '; '.join(parts) + f'. The real job\'s `--drop` argument: `{",".join(drop) if drop else ""}`' + (' (none dropped).' if not drop else '.'),
                  '', f'**Seed {s_}\'s values, for the record:** G {G(s_):+.3f}; rw-oracle F {Fv("rw-oracle", s_):+.3f}, P {Pv("rw-oracle", s_):+.3f}, b {b("rw-oracle", s_):+.4f}' if b('rw-oracle', s_) is not None else
                  f'**Seed {s_}\'s values, for the record:** G {G(s_):+.3f}; rw-oracle F {Fv("rw-oracle", s_):+.3f}, P {Pv("rw-oracle", s_):+.3f}, b n/a (unbracketed)']
            L[-1] += (f'; C = Δ_A(sub20-oracle) − Δ_A(sub20-rw) {Cv(s_):+.3f}; sub20-oracle − sub20 {S3(s_):+.3f}, its residual '
                      + (f'{b("sub20-oracle", s_):+.4f}' if b('sub20-oracle', s_) is not None else 'n/a') + f'; ladder ordered: {"yes" if ordered(s_) else "no"}.')
        L += ['', '## By arm (means over seeds)', '',
              '| Arm | Weight | Δ_A | a: agreement with head A | Share at the end, s | Removed share, P | Recovered fraction, F | â: the ladder at the same Δ_A | b = â − a | Head A, minus clean | Rotated | Ignored clean targets: own / head A |',
              '|---|---|---|---|---|---|---|---|---|---|---|---|']
        for a in ['rw-b1', 'rw-oracle', 'sub20-oracle', 'sub20-rw'] + expl_ran:
            over = expl.get(a) or seeds
            bs = [v for v in (b(a, s) for s in over) if v is not None]
            if a in with_share:
                s_cell = f"{mean(lambda s: share(a, s), over):.3f}"
                p_cell = '—' if a == 'rw-b1' else f'{mean(lambda s: Pv(a, s), over):+.3f}'
                f_cell = '—' if a == 'rw-b1' else f'{mean(lambda s: Fv(a, s), over):+.3f}'
            else:
                s_cell = p_cell = f_cell = '—'
            L.append(f"| {a} | {LABEL[a]} | {mean(lambda s: dA(a, s), over):+.3f} | {mean(lambda s: a_of(a, s), over):.3f} | {s_cell} | {p_cell} | {f_cell} | "
                     f"{hat_cell(a, over)} | {f'{st.mean(bs):+.3f}' if bs else 'n/a'} | "
                     f"{mean(lambda s: rec(a, s)['functional']['A_clean'] - r[('clean', s)]['functional']['A_clean'], over):+.4f} | {mean(lambda s: ratio(rec(a, s), 'rotated_control'), over):.3f} | "
                     f"{mean(lambda s: ng(a, s, 'own_ignored_fraction'), over):.3f} / {mean(lambda s: ng(a, s, 'A_ignored_fraction'), over):.3f} |")
        for a in rung:
            L.append(f"| {a} | ladder | {mean(lambda s: dA(a, s)):+.3f} | {mean(lambda s: a_of(a, s)):.3f} | — | — | "
                     f"{'1 by definition' if a == 'sub20' else '—'} | (on the ladder) | 0 | "
                     f"{mean(lambda s: rec(a, s)['functional']['A_clean'] - r[('clean', s)]['functional']['A_clean']):+.4f} | {mean(lambda s: ratio(rec(a, s), 'rotated_control')):.3f} | "
                     f"{mean(lambda s: ng(a, s, 'own_ignored_fraction')):.3f} / {mean(lambda s: ng(a, s, 'A_ignored_fraction')):.3f} |")
        L += ['', '## By seed', '',
              '| Seed | G | s(rw-b1) | s(rw-oracle) | F | P | C | sub20-oracle − sub20 | b(rw-oracle) | b(sub20-oracle) | Ladder ordered |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
        for s in seeds:
            bb = ['n/a' if b(a, s) is None else f'{b(a, s):+.4f}' for a in CONF]
            L.append(f"| {s} | {G(s):+.3f} | {share('rw-b1', s):.3f} | {share('rw-oracle', s):.3f} | {Fv('rw-oracle', s):+.3f} | {Pv('rw-oracle', s):+.3f} | {Cv(s):+.3f} | {S3(s):+.3f} | "
                     + ' | '.join(bb) + f" | {'yes' if ordered(s) else 'no'} |")
        L += ['', '## The clean-target ladder by seed: (Δ_A, a)', '', '| Seed | ' + ' | '.join(rung) + ' |', '|---|' + '---|' * len(rung)]
        for s in seeds:
            L.append(f'| {s} | ' + ' | '.join(f'({x:+.3f}, {y:.3f})' for x, y in ladder(s)) + ' |')
        L += ['', '## Secondary and exploratory measures', '', 'None of these enters a decision rule.', '']
        L += ['### S1: neglect at the end (means over seeds)', '',
              f'On the measurement windows: the fraction of clean targets whose weight is below {IGNORED} under head B\'s own discount (ρ = 0.8) and under head A\'s judgement, '
              'and the inner reading\'s clean-target loss on each ignored set against the rest.', '',
              '| Arm | Ignored by own discount | Ignored by head A | Loss on own-ignored / rest | Loss on A-ignored / rest |', '|---|---|---|---|---|']
        for a in ['rw-b1', 'rw-oracle', 'sub20-oracle', 'sub20-rw'] + expl_ran + list(rung):
            over = expl.get(a) or seeds
            L.append(f"| {a} | {mean(lambda s: ng(a, s, 'own_ignored_fraction'), over):.3f} | {mean(lambda s: ng(a, s, 'A_ignored_fraction'), over):.3f} | "
                     f"{mean(lambda s: ng(a, s, 'loss_on_own_ignored'), over):.2f} / {mean(lambda s: ng(a, s, 'loss_on_rest_own'), over):.2f} | "
                     f"{mean(lambda s: ng(a, s, 'loss_on_A_ignored'), over):.2f} / {mean(lambda s: ng(a, s, 'loss_on_rest_A'), over):.2f} |")
        L.append('')
        L.append(f"The conjecture expects rw-b1's own discount to ignore a larger fraction than head A's judgement does, and its loss on the targets its own discount ignores to be far above its loss on the rest. "
                 f"rw-b1: own {mean(lambda s: ng('rw-b1', s, 'own_ignored_fraction')):.3f} against head A {mean(lambda s: ng('rw-b1', s, 'A_ignored_fraction')):.3f}; "
                 f"loss on its own-ignored targets {mean(lambda s: ng('rw-b1', s, 'loss_on_own_ignored')):.2f} against {mean(lambda s: ng('rw-b1', s, 'loss_on_rest_own')):.2f} on the rest.")
        L += ['', '### S2: the exploratory warm-up arms', '']
        for a in EXPL:
            if a in dropped:
                L.append(f'- {a}: dropped before the run by the calibration\'s rule.')
            elif a not in expl_ran:
                L.append(f'- {a}: no records.')
            else:
                over = expl[a]
                k = sum(dA(a, s) <= TRAINED for s in over)
                bs = [b(a, s) for s in over]
                checks = (f"fingerprints as n80's in {sum(run_ok(a, s) for s in over)} of {len(over)}, rotated ≤ {SEPARATE} in {sum(ratio(rec(a, s), 'rotated_control') <= SEPARATE for s in over)} of {len(over)}, "
                          f"constructed mixture and zero-noise copy in {sum(rec(a, s)['constructed_mixture']['pass'] and ratio(rec(a, s), 'copy0_vs_A') >= MATCH_MIN for s in over)} of {len(over)}")
                L.append(f"- {a} ({LABEL[a]}): trained (Δ_A ≤ {TRAINED}) in {k} of {len(over)} seeds; Δ_A {fmt([dA(a, s) for s in over])}, F {fmt([Fv(a, s) for s in over])}, "
                         f"share {fmt([share(a, s) for s in over])}, P {fmt([Pv(a, s) for s in over])}, ignored kept at the end {fmt([ignored_end(a, s) for s in over])}, "
                         f"residual b {fmt(bs, 4)} ({sum(x is not None for x in bs)} of {len(over)} bracketed). Checks: {checks}.")
        L += ['', '### S3: sub20-oracle against sub20', '',
              f'Δ_A(sub20-oracle) − Δ_A(sub20) = {fmt([S3(s) for s in seeds])}. Below zero would mean head A teaches; at or above, the judge only removes the blind spot.']
        steps = [0, 300, 1500, 2850] if not cal or len(trn[('sub20-oracle', seeds[0])]['w_every_50']) > 20 else [0, 20, 40]
        L += ['', '### S4: the weights over training (means over seeds)', '',
              'Mean weight on the kept and on the replaced positions, the random targets\' share of the summed weight, and the fraction of kept positions with weight below 0.01, '
              'at the recorded step nearest each listed step. For rw-b1 the record is N13\'s (no ignored fraction); for the clean-target-only arm there are no replaced positions.', '',
              '| Arm | ' + ' | '.join(f'step {x}: kept / replaced / share / ignored' for x in steps) + ' |', '|---|' + '---|' * len(steps)]
        for a in ['rw-b1', 'rw-oracle', 'sub20-oracle'] + expl_ran:
            over = expl.get(a) or seeds
            cells = []
            for x in steps:
                if a == 'rw-b1':
                    recs = [min(b1[s]['r_every_50'], key=lambda e: abs(e[0] - x)) for s in over]
                    fr = [b1[s].get('noise_fraction_step0', 0.8) for s in over]
                    sh = [f * e[3] / (f * e[3] + (1 - f) * e[2]) for f, e in zip(fr, recs)]
                    cells.append(f'{st.mean(e[2] for e in recs):.3f} / {st.mean(e[3] for e in recs):.3f} / {st.mean(sh):.2f} / —')
                else:
                    recs = [min(trn[(a, s)]['w_every_50'], key=lambda e: abs(e[0] - x)) for s in over]
                    if recs[0][3] is None:
                        cells.append(f'{st.mean(e[2] for e in recs):.3f} / — / — / {st.mean(e[5] for e in recs):.2f}')
                    else:
                        cells.append(f'{st.mean(e[2] for e in recs):.3f} / {st.mean(e[3] for e in recs):.3f} / {st.mean(e[4] for e in recs):.2f} / {st.mean(e[5] for e in recs):.2f}')
            L.append(f'| {a} | ' + ' | '.join(cells) + ' |')
        every = [e[0] for e in trn[('rw-oracle', seeds[0])]['w_every_50']]
        marks = [x for x in every if x % 300 == 0] if len(every) > 20 else every
        def share_at(a, s, x):
            if a == 'rw-b1':
                e, f = min(b1[s]['r_every_50'], key=lambda e: abs(e[0] - x)), b1[s].get('noise_fraction_step0', 0.8)
                return f * e[3] / (f * e[3] + (1 - f) * e[2])
            return min(trn[(a, s)]['w_every_50'], key=lambda e: abs(e[0] - x))[4]
        L += ['', 'The oracle\'s share against β = 1\'s, step by step (rw-b1 / rw-oracle, means over seeds): '
              + ', '.join(f'step {x}: {mean(lambda s: share_at("rw-b1", s, x)):.2f} / {mean(lambda s: share_at("rw-oracle", s, x)):.2f}' for x in marks) + '.']
        L += ['', '### S5: head A\'s cost', '',
              'Head A against clean (same seed): ' + ', '.join(f"{a} {fmt([rec(a, s)['functional']['A_clean'] - r[('clean', s)]['functional']['A_clean'] for s in (expl.get(a) or seeds)], 4)}"
                                                           for a in ['rw-b1', 'rw-oracle', 'sub20-oracle', 'sub20-rw'] + expl_ran)
              + '; against rw-b1: ' + ', '.join(f"{a} {fmt([rec(a, s)['functional']['A_clean'] - r[('rw-b1', s)]['functional']['A_clean'] for s in (expl.get(a) or seeds)], 4)}"
                                                for a in ['rw-oracle', 'sub20-oracle'] + expl_ran) + '.']
        L += ['', '### S6: agreement with the copy, and sub20-oracle against the ladder', '',
              'Agreement with the copy at 0.8, beside the head A measure: ' + ', '.join(
                  f"{a} {mean(lambda s: ratio(rec(a, s), 'B_vs_copy'), expl.get(a) or seeds):.3f} / {mean(lambda s: a_of(a, s), expl.get(a) or seeds):.3f}"
                  for a in ['rw-b1', 'rw-oracle', 'sub20-oracle', 'sub20-rw'] + expl_ran)
              + f". The residual of sub20-oracle against the ladder: b = {fmt([b('sub20-oracle', s) for s in seeds], 4)} "
              f"({sum(b('sub20-oracle', s) is not None for s in seeds)} of {len(seeds)} bracketed)."]
        envs = sorted({(x['environment']['gpu'], x['environment']['torch'], x['environment']['code_commit'][:7])
                       for x in all_measured + [trn[(a, s)] for a in trained for s in seeds if trn.get((a, s))]})
        L += ['', '## Run environment', ''] + [f'- {gpu}, PyTorch {v}, code {cm}' for gpu, v, cm in envs] + ['']
    OUT.write_text('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
