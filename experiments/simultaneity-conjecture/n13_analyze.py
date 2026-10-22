"""Apply N13's preregistered decision rules to the saved records and write N13_RESULTS.md.

Written and committed before any real N13 run. Every number in the report comes from the JSON records: N13's own
(results/n13: m_* for the new arms, r_* for N12's models re-measured, and the training records) and N12's committed
records (results/n12), which precondition 2 compares the re-measurements with. For the calibration run (one seed) it
reports the same quantities and checks, with no verdicts.

The inner reading against head A: delta_A = head B's inner softmax's clean-target loss minus head A's, within a model.
N13a: |delta_A(n80-rw) - delta_A(n80-mix)| <= TAU in at least 4 of 5 seeds (count rule).
N13b: S(n80 read out in the mixture form), against the same seed's n80 yardstick (interval rule).
N13c: D = delta_A(n80-w02) - delta_A(n80) (interval rule).

    python n13_analyze.py [results folder] [report path] [N12 records folder]
    # defaults: results/n13 here, N13_RESULTS.md, and results/n12 or results/n12_calib by the summary's mode
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

from scipy import stats

HERE = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'results' / 'n13'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'N13_RESULTS.md'

REAL_SEEDS = [4101, 4102, 4103, 4104, 4105]
NEW = ('n80-rw', 'n80-mix-rep', 'n80-w02')
OLD = ('n80', 'n80-mix', 'sub20')
WINDOW = 128 + 8
NEEDED, LEVEL = 4, 0.975
TAU, FAR = 0.03, 0.10                                 # N13a: the tolerance, and the refuting difference
FORM_NONE, FORM_SOME, FORM_SD = 0.10, 0.33, 0.05      # N13b
GAIN_QUARTER, GAIN_HALF, GAIN_SD = -0.27, -0.53, 0.10  # N13c: a quarter and a half of N12's improvement of 1.062 nats
REPRO_LOSS, REPRO_RATIO, MATCH_MIN, SEPARATE, MIN_GAP, NOISE_TOL = 1e-3, 1e-2, 0.999, 0.60, 0.30, 0.05
N12_IMPROVEMENT = 1.062


def load(folder, name):
    p = folder / f'{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


def interval(xs, level=LEVEL):
    n = len(xs)
    m = st.mean(xs)
    sd = st.stdev(xs) if n > 1 else float('nan')
    half = stats.t.ppf(1 - (1 - level) / 2, n - 1) * sd / math.sqrt(n) if n > 1 else float('nan')
    return m, sd, m - half, m + half


def interval_verdict(xs, support, refute, above, sd_max, supported_text, label):
    """above: the claim is that the interval lies above `support` (and refutation below `refute`); else the reverse."""
    m, sd, lo, hi = interval(xs)
    tail = f'{label} = {m:+.4f}, 97.5% interval [{lo:+.4f}, {hi:+.4f}], standard deviation {sd:.4f}.'
    if (lo > support) if above else (hi < support):
        return f'**Supported: {supported_text}** {tail}'
    if ((hi < refute) if above else (lo > refute)) and sd <= sd_max:
        return f'**Refuted.** {tail}'
    if lo > 0 or hi < 0:
        return f'**Some effect, size unclear.** {tail}'
    return f'**Inconclusive.** {tail}'


def main():
    summary = load(RES, 'n13_summary') or {}
    mode = summary.get('mode', 'real')
    n12_dir = Path(sys.argv[3]) if len(sys.argv) > 3 else HERE / 'results' / ('n12_calib' if mode == 'calibration' else 'n12')
    want = summary.get('seeds', REAL_SEEDS) if mode == 'calibration' else REAL_SEEDS
    c = summary.get('checks', {})
    cut = summary.get('held_out', {}).get('cut_token')
    m = {(a, s): load(RES, f'm_{a}_seed{s}') for a in NEW + ('n80-w02-mixform',) for s in want}
    r = {(a, s): load(RES, f'r_{a}_seed{s}') for a in OLD + ('n80-mixform',) for s in want}
    trn = {(a, s): load(RES, f'{a}_seed{s}') for a in NEW for s in want}
    ref = {(a, s): load(n12_dir, f'm_{a}_seed{s}') for a in OLD + ('clean',) for s in want}
    ref_trn = {s: load(n12_dir, f'n80_seed{s}') for s in want}
    seeds = [s for s in want if all(m[(a, s)] for a in NEW + ('n80-w02-mixform',)) and all(r[(a, s)] for a in OLD + ('n80-mixform',))
             and all(trn[(a, s)] for a in NEW) and all(ref[(a, s)] for a in OLD + ('clean',)) and ref_trn[s]]

    def dA(rec):
        return rec['functional']['B_inner_clean'] - rec['functional']['A_clean']

    def gap_n80(s):
        return r[('n80', s)]['functional']['B_clean'] - r[('n80', s)]['functional']['copy_clean']

    def S(rec, s):
        return 1 - (rec['functional']['B_clean'] - r[('n80', s)]['functional']['copy_clean']) / gap_n80(s)

    def ratio(rec, key):
        return rec[key]['agreement_ratio']

    def inner_vs_A(rec):
        return ratio(rec, 'B_inner_vs_A') if rec.get('B_inner_vs_A') else ratio(rec, 'B_vs_A')

    all_measured = [m[(a, s)] for a in NEW + ('n80-w02-mixform',) for s in seeds] + [r[(a, s)] for a in OLD + ('n80-mixform',) for s in seeds]
    fingerprints_ok = bool(seeds) and all(trn[(a, s)]['batch_fingerprint_step0'] == ref_trn[s]['batch_fingerprint_step0']
                                          and trn[(a, s)]['replaced_positions_fingerprint_step0'] == ref_trn[s]['replaced_positions_fingerprint_step0']
                                          for a in NEW for s in seeds)
    held_out = cut is not None and bool(seeds) and all(trn[(a, s)]['training_tokens'] == cut and trn[(a, s)]['max_offset'] + WINDOW <= cut for a in NEW for s in seeds)
    noise_ok = bool(seeds) and all(abs(trn[(a, s)]['noise_fraction_step0'] - 0.8) <= NOISE_TOL for a in NEW for s in seeds)
    n_needed = min(NEEDED, len(want))
    pre = {
        '1. Setup: batch, same-model, identical-heads, analytic and gradient-identity checks; windows; held-out tail untouched; noise fractions; '
        'each new run\'s batch and replaced-positions fingerprints equal N12\'s n80\'s':
            bool(c.get('batch', {}).get('targets_are_i+1_to_i+8')) and bool(c.get('windows_shifted')) and bool(c.get('identical_heads', {}).get('pass'))
            and bool(c.get('analytic', {}).get('pass')) and bool(c.get('gradient_identity', {}).get('pass'))
            and all(x['pass'] for x in c.get('same_model', [{'pass': False}])) and held_out and fingerprints_ok and noise_ok,
        f'2. Must match: re-measuring N12\'s n80 and n80-mix reproduces N12\'s records (losses within {REPRO_LOSS:g}, ratios within {REPRO_RATIO:g}); '
        f'the constructed mixture check and the zero-noise copy (≥ {MATCH_MIN}) in every model':
            bool(seeds) and all(r[(a, s)].get('reproduction', {}).get('pass') for a in ('n80', 'n80-mix') for s in seeds)
            and all(rec['constructed_mixture']['pass'] and ratio(rec, 'copy0_vs_A') >= MATCH_MIN for rec in all_measured),
        f'3. The replicate floor is below the tolerance: |Δ_A(n80-mix-rep) − Δ_A(n80-mix)| ≤ {TAU} in at least {NEEDED} of 5 seeds':
            sum(abs(dA(m[('n80-mix-rep', s)]) - dA(r[('n80-mix', s)])) <= TAU for s in seeds) >= n_needed,
        f'4. Must differ: rotated control ≤ {SEPARATE} in at least {NEEDED} of 5, in every arm':
            bool(seeds) and all(sum(ratio(rec[(a, s)], 'rotated_control') <= SEPARATE for s in seeds) >= n_needed
                                for rec, arms in ((m, NEW), (r, OLD)) for a in arms),
        f'5. The yardstick leaves a gap: gap(n80) ≥ {MIN_GAP} nats in every seed':
            bool(seeds) and all(gap_n80(s) >= MIN_GAP for s in seeds),
    }
    names = list(pre)

    def gate(needed):
        if len(seeds) < 5:
            return f'**Not reached: complete records for only {len(seeds)} of 5 seeds.**'
        bad = [names[i].split(':')[0] for i in needed if not pre[names[i]]]
        return f'**Untestable: a precondition failed ({"; ".join(bad)}).**' if bad else None

    L = ['# N13 results' if mode == 'real' else '# N13 calibration run', '',
         f'Generated by `n13_analyze.py` from the records in `{RES.name}/`, compared with N12\'s in `{n12_dir.name}/`. Preregistration: `N13_PREREGISTRATION.md`.',
         f"Measured on the held-out tail's windows {summary.get('held_out', {}).get('measure')}, N12's. Δ_A is head B's inner softmax's clean-target loss "
         'minus head A\'s, within a model; S is N12\'s share of the gap closed, against the same seed\'s n80 yardstick.', '']
    if mode == 'real':
        a_diff = [abs(dA(m[('n80-rw', s)]) - dA(r[('n80-mix', s)])) for s in seeds]
        v_a = gate([0, 1, 2, 3])
        if not v_a:
            n_in, n_far = sum(d <= TAU for d in a_diff), sum(d > FAR for d in a_diff)
            if n_in >= NEEDED:
                v_a = f'**Supported: the pipeline follows the identity.** |Δ_A(n80-rw) − Δ_A(n80-mix)| ≤ {TAU} in {n_in} of 5 seeds.'
            elif n_far >= NEEDED:
                v_a = f'**Refuted.** The difference exceeds {FAR} in {n_far} of 5 seeds while the replicate stays within {TAU}.'
            else:
                v_a = f'**Intermediate.** Within {TAU} in {n_in} of 5 seeds; beyond {FAR} in {n_far}.'
        v_b = gate([0, 1, 3, 4]) or interval_verdict([S(r[('n80-mixform', s)], s) for s in seeds], FORM_NONE, FORM_SOME, False, FORM_SD,
                                                     'the output form, applied at readout to the standard head, closes none of the gap.', 'S(n80, mixture readout)')
        v_c = gate([0, 1, 3]) or interval_verdict([dA(m[('n80-w02', s)]) - dA(r[('n80', s)]) for s in seeds], GAIN_QUARTER, GAIN_HALF, True, GAIN_SD,
                                                  'the gain is not from head B pulling less on the shared network.', 'D = Δ_A(n80-w02) − Δ_A(n80)')
        L += ['## Verdicts', '',
              f'**N13a (primary: the pipeline follows the identity):** {v_a}', '',
              f'**N13b (the output form, applied at readout to the standard head, closes none of the gap):** {v_b}', '',
              f'**N13c (the gain is not from head B pulling less on the shared network):** {v_c}', '']
    else:
        L += ['This is the calibration run of the preregistration\'s calibration step: one seed, on windows the real run does not use. '
              'It decides nothing, and no verdict is computed. Precondition 3, the replicate floor, is the check it exists for.', '']
    L += ['## Preconditions', '', '| Precondition | Met |', '|---|---|']
    L += [f'| {k} | {"yes" if ok else "**no**"} |' for k, ok in pre.items()]

    if seeds:
        def mean(f):
            return st.mean([f(s) for s in seeds])
        rows = [('n80 (N12, re-measured)', lambda s: r[('n80', s)]), ('n80-mix (N12, re-measured)', lambda s: r[('n80-mix', s)]),
                ('sub20 (N12, re-measured)', lambda s: r[('sub20', s)]), ('n80-rw', lambda s: m[('n80-rw', s)]),
                ('n80-mix-rep', lambda s: m[('n80-mix-rep', s)]), ('n80-w02', lambda s: m[('n80-w02', s)])]
        L += ['', '## By arm (means over seeds)', '',
              '| Arm | Δ_A: inner reading minus head A (nats) | Inner softmax\'s agreement with head A | Head A, minus N12\'s clean arm | S, mixture form | Rotated |',
              '|---|---|---|---|---|---|']
        for name, get in rows:
            arm = name.split(' ')[0]
            mixrec = {'n80': lambda s: r[('n80-mixform', s)], 'n80-w02': lambda s: m[('n80-w02-mixform', s)]}.get(arm, get)
            L.append(f'| {name} | {mean(lambda s: dA(get(s))):+.3f} | {mean(lambda s: inner_vs_A(get(s))):.3f} | '
                     f"{mean(lambda s: get(s)['functional']['A_clean'] - ref[('clean', s)]['functional']['A_clean']):+.4f} | "
                     f'{mean(lambda s: S(mixrec(s), s)):+.3f} | {mean(lambda s: ratio(get(s), "rotated_control")):.3f} |')
        L += ['', '## By seed', '',
              '| Seed | Δ_A(n80-rw) | Δ_A(n80-mix) | difference | Δ_A(n80-mix-rep) − Δ_A(n80-mix) | S(n80, mixture readout) | D = Δ_A(n80-w02) − Δ_A(n80) |',
              '|---|---|---|---|---|---|---|']
        for s in seeds:
            L.append(f"| {s} | {dA(m[('n80-rw', s)]):+.4f} | {dA(r[('n80-mix', s)]):+.4f} | {dA(m[('n80-rw', s)]) - dA(r[('n80-mix', s)]):+.4f} | "
                     f"{dA(m[('n80-mix-rep', s)]) - dA(r[('n80-mix', s)]):+.4f} | {S(r[('n80-mixform', s)], s):+.3f} | "
                     f"{dA(m[('n80-w02', s)]) - dA(r[('n80', s)]):+.4f} |")
        L += ['', '## Against the N12 partner: per-position divergence on the measurement windows (means over seeds)', '',
              '| Arm | Partner | Inner softmax, symmetrised KL | Mixture form, symmetrised KL | Head A, symmetrised KL |', '|---|---|---|---|---|']
        for a in NEW:
            L.append(f"| {a} | {m[(a, seeds[0])]['partner']['arm']} | {mean(lambda s: m[(a, s)]['partner']['inner_symmetrised']):.4f} | "
                     f"{mean(lambda s: m[(a, s)]['partner']['mix_symmetrised']):.4f} | {mean(lambda s: m[(a, s)]['partner']['A_symmetrised']):.4f} |")
        steps = [0, 300, 1500, 2850] if mode == 'real' or len(trn[(NEW[0], seeds[0])]['r_every_50']) > 20 else [0, 20, 40]
        L += ['', '## S1: the discount over training (means over seeds)', '',
              'Mean r(y) on the kept and on the replaced positions of the batch, at the recorded step nearest each listed step. '
              'In n80-rw it is the discount in use; in the other arms it is computed from their logits and recorded only.', '',
              '| Arm | ' + ' | '.join(f'step {x}: kept / replaced' for x in steps) + ' |', '|---|' + '---|' * len(steps)]
        for a in NEW:
            cells = []
            for x in steps:
                vals = []
                for s in seeds:
                    rec = min(trn[(a, s)]['r_every_50'], key=lambda e: abs(e[0] - x))
                    vals.append((rec[2], rec[3] if rec[3] is not None else float('nan')))
                cells.append(f'{st.mean(v[0] for v in vals):.3f} / {st.mean(v[1] for v in vals):.3f}')
            L.append(f'| {a} | ' + ' | '.join(cells) + ' |')
        L += ['', '## Secondary and exploratory measures', '', 'None of these enters a decision rule.', '',
              f"- S2, the replicate floor: Δ_A differs by {mean(lambda s: abs(dA(m[('n80-mix-rep', s)]) - dA(r[('n80-mix', s)]))):.4f} on average "
              f"(largest {max(abs(dA(m[('n80-mix-rep', s)]) - dA(r[('n80-mix', s)])) for s in seeds):.4f}); S by "
              f"{mean(lambda s: abs(S(m[('n80-mix-rep', s)], s) - S(r[('n80-mix', s)], s))):.4f}; the inner softmax's agreement with head A by "
              f"{mean(lambda s: abs(inner_vs_A(m[('n80-mix-rep', s)]) - inner_vs_A(r[('n80-mix', s)]))):.4f}.",
              f"- S3, S in the mixture form: n80-rw {mean(lambda s: S(m[('n80-rw', s)], s)):+.3f}, n80-w02 {mean(lambda s: S(m[('n80-w02-mixform', s)], s)):+.3f}, "
              f"against n80-mix {mean(lambda s: S(r[('n80-mix', s)], s)):+.3f} and sub20 {mean(lambda s: S(r[('sub20', s)], s)):+.3f}.",
              f"- S4, head A's cost against n80 (same seed): n80-rw {mean(lambda s: m[('n80-rw', s)]['functional']['A_clean'] - r[('n80', s)]['functional']['A_clean']):+.4f}, "
              f"n80-mix-rep {mean(lambda s: m[('n80-mix-rep', s)]['functional']['A_clean'] - r[('n80', s)]['functional']['A_clean']):+.4f}, "
              f"n80-w02 {mean(lambda s: m[('n80-w02', s)]['functional']['A_clean'] - r[('n80', s)]['functional']['A_clean']):+.4f}.",
              f"- S5, the inner softmax's agreement with head A: n80-rw {mean(lambda s: inner_vs_A(m[('n80-rw', s)])):.3f}, "
              f"n80-mix-rep {mean(lambda s: inner_vs_A(m[('n80-mix-rep', s)])):.3f}, n80-w02 {mean(lambda s: inner_vs_A(m[('n80-w02', s)])):.3f}; "
              f"N12's n80-mix {mean(lambda s: inner_vs_A(r[('n80-mix', s)])):.3f}, n80 {mean(lambda s: inner_vs_A(r[('n80', s)])):.3f}.",
              f"- N13c's margins as fractions of N12's improvement ({N12_IMPROVEMENT} nats): a quarter is {GAIN_QUARTER}, a half is {GAIN_HALF}.", '']
        envs = sorted({(x['environment']['gpu'], x['environment']['torch'], x['environment']['code_commit'][:7])
                       for x in all_measured + [trn[(a, s)] for a in NEW for s in seeds]})
        L += ['## Run environment', ''] + [f'- {gpu}, PyTorch {v}, code {cm}' for gpu, v, cm in envs] + ['']
    OUT.write_text('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
