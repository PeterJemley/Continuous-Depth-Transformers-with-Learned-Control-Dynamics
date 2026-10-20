"""Apply N11's preregistered decision rules to the saved records and write N11_RESULTS.md.

Written and committed before any real N11 run. Every number in the report comes from the JSON
records in the results folder.

The share of the gap closed, per seed: gap(arm) = head B's clean-target loss in that arm minus the clean-target
loss of the ideal copy of the same seed's n80 head A (both on the measurement windows), and
S(arm) = 1 - gap(arm) / gap(n80).

Preconditions 1-4 apply to every claim; 5f to N11a and N11c; 5g to N11c; 6 to none.

    python n11_analyze.py [results folder] [report path]   # default: results/n11 here, N11_RESULTS.md
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

from scipy import stats

HERE = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'results' / 'n11'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'N11_RESULTS.md'

SEEDS = [4001, 4002, 4003, 4004, 4005]
ARMS = {'clean': 0.0, 'n80': 0.8, 'n80-f8': 0.8, 'n80-f32': 0.8, 'n80-f128': 0.8, 'n80-geom': 0.8}
PAY = ('n80-f8', 'n80-f32', 'n80-f128', 'n80-geom')
NEEDED, LEVEL = 4, 0.975
PRESERVE, BEATS, SD_SENSITIVE = 0.67, 0.30, 0.05
SAME_MIN, CLEAN_GAP, MATCH_MIN, SEPARATE = 0.95, 0.05, 0.999, 0.60
MIN_GAP, KL_FRACTION, PINNED, STABLE_MIN, NOISE_TOL = 0.30, 0.5, 0.90, 0.80, 0.05


def load(name):
    p = RES / f'{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


def interval(xs, level=LEVEL):
    n = len(xs)
    m = st.mean(xs)
    sd = st.stdev(xs) if n > 1 else float('nan')
    half = stats.t.ppf(1 - (1 - level) / 2, n - 1) * sd / math.sqrt(n) if n > 1 else float('nan')
    return m, sd, m - half, m + half


def interval_verdict(xs, margin, supported, label):
    m, sd, lo, hi = interval(xs)
    tail = f'{label} = {m:+.4f}, 97.5% interval [{lo:+.4f}, {hi:+.4f}], standard deviation {sd:.4f}.'
    if lo > margin:
        return f'**Supported: {supported}** {tail}'
    if hi < margin and sd <= SD_SENSITIVE:
        return f'**Refuted.** {tail}'
    if lo > 0 or hi < 0:
        return f'**Some effect, size unclear.** {tail}'
    return f'**Inconclusive.** {tail}'


def main():
    summary = load('n11_summary') or {}
    c = summary.get('checks', {})
    trn = {(a, s): load(f'{a}_seed{s}') for a in ARMS for s in SEEDS}
    mes = {(a, s): load(f'm_{a}_seed{s}') for a in ARMS for s in SEEDS}
    seeds = [s for s in SEEDS if all(trn[(a, s)] and mes[(a, s)] for a in ARMS)]

    def fn(a, s, k):
        return mes[(a, s)]['functional'][k]

    def gap(a, s, where='functional'):
        return mes[(a, s)][where]['B_clean'] - mes[('n80', s)][where]['copy_clean']

    def S(a, s, where='functional'):
        return 1 - gap(a, s, where) / gap('n80', s, where)

    def ratio(a, s, key='B_vs_copy'):
        return mes[(a, s)][key]['agreement_ratio']

    def pkl(a, s):
        return mes[(a, s)]['payload_windows']['kl_B_to_copy']

    same_data = bool(seeds) and all(len({trn[(a, s)].get('batch_fingerprint_step0') for a in ARMS}) == 1 for s in seeds)
    noise_ok = bool(seeds) and all(abs(trn[(a, s)]['noise_fraction_step0'] - ARMS[a]) <= NOISE_TOL for a in ARMS if ARMS[a] > 0 for s in seeds)
    pre = {
        '1. Setup: batch, same-model, identical-heads and analytic checks; windows; noise fractions; same batches; payload windows':
            bool(c.get('batch', {}).get('targets_are_i+1_to_i+8')) and bool(c.get('test_windows_shifted'))
            and bool(c.get('identical_heads', {}).get('pass')) and bool(c.get('analytic', {}).get('pass'))
            and bool(c.get('payload_windows', {}).get('targets_shifted'))
            and all(x['pass'] for x in c.get('same_model', [{'pass': False}])) and same_data and noise_ok,
        f'2. Must match: clean arm ratio ≥ {SAME_MIN} and |head B − head A| ≤ {CLEAN_GAP} nats in at least 4 of 5; zero-noise copy ≥ {MATCH_MIN} in every model':
            sum(ratio('clean', s) >= SAME_MIN for s in seeds) >= NEEDED
            and sum(abs(fn('clean', s, 'B_clean') - fn('clean', s, 'A_clean')) <= CLEAN_GAP for s in seeds) >= NEEDED
            and bool(seeds) and all(ratio(a, s, 'copy0_vs_A') >= MATCH_MIN for a in ARMS for s in seeds),
        f'3. Must differ: rotated control ≤ {SEPARATE} in at least 4 of 5, in every arm':
            bool(seeds) and all(sum(mes[(a, s)]['rotated_control']['agreement_ratio'] <= SEPARATE for s in seeds) >= NEEDED for a in ARMS),
        f'4. The yardstick leaves a gap: gap(n80) ≥ {MIN_GAP} nats in every seed':
            bool(seeds) and all(gap('n80', s) >= MIN_GAP for s in seeds),
        f'5f. Functional payload took hold: n80-f128 payload-window KL ≤ {KL_FRACTION} × n80\'s in at least 4 of 5':
            sum(pkl('n80-f128', s) <= KL_FRACTION * pkl('n80', s) for s in seeds) >= NEEDED,
        f'5g. Geometric payload took hold: readout overlap ≥ {PINNED} in every n80-geom model':
            bool(seeds) and all(mes[('n80-geom', s)]['readout_overlap_A_B'] >= PINNED for s in seeds),
    }
    names = list(pre)
    stable = bool(seeds) and all(min(v for k, v in mes[(a, s)]['stability'].items() if k != 'A') >= STABLE_MIN for a in ARMS for s in seeds)

    def gate(needed):
        if len(seeds) < 5:
            return f'**Not reached: all six arms have records for only {len(seeds)} of 5 seeds.**'
        bad = [names[i].split(':')[0] for i in needed if not pre[names[i]]]
        return f'**Untestable: a precondition failed ({"; ".join(bad)}).**' if bad else None

    v_a = gate([0, 1, 2, 3, 4]) or interval_verdict([S('n80-f128', s) for s in seeds], PRESERVE,
                                                      'a functional payload of sufficient strength preserves the reading\'s function.', 'S(n80-f128)')
    v_b = gate([0, 1, 2, 3])
    if not v_b:
        n_ord = sum(S('n80-f8', s) < S('n80-f32', s) < S('n80-f128', s) for s in seeds)
        n_rev = sum(S('n80-f128', s) <= S('n80-f8', s) for s in seeds)
        if n_ord >= NEEDED:
            v_b = f'**Supported: the effect grows with the strength.** f8 < f32 < f128 in {n_ord} of 5 seeds.'
        elif n_rev >= NEEDED:
            v_b = f'**Refuted.** S(f128) ≤ S(f8) in {n_rev} of 5 seeds.'
        else:
            v_b = f'**Partial.** Ordered in {n_ord} of 5 seeds; S(f128) ≤ S(f8) in {n_rev}.'
    v_c = gate([0, 1, 2, 3, 4, 5]) or interval_verdict([S('n80-f128', s) - S('n80-geom', s) for s in seeds], BEATS,
                                                         'function, not geometry.', 'S(n80-f128) − S(n80-geom)')

    L = ['# N11 results', '',
         'Generated by `n11_analyze.py` from the records in `results/n11/`. Preregistration: `N11_PREREGISTRATION.md`.',
         f"Measured on {summary.get('windows', {}).get('split')}-split windows {summary.get('windows', {}).get('measure')}. "
         'S is the share of the gap closed against a fixed yardstick: the ideal noisy copy of the same seed\'s n80 head A.', '',
         '## Verdicts', '',
         f'**N11a (primary: a functional payload of sufficient strength preserves the reading\'s function):** {v_a}', '',
         f'**N11b (the effect grows with the strength):** {v_b}', '',
         f'**N11c (function, not geometry):** {v_c}', '',
         '## Preconditions', '', '| Precondition | Met |', '|---|---|']
    L += [f'| {k} | {"yes" if ok else "**no**"} |' for k, ok in pre.items()]
    L.append(f"| 6. Stable subspaces (secondary outcomes only): head B and the copies ≥ {STABLE_MIN} in every model | {'yes' if stable else '**no**'} |")

    if seeds:
        L += ['', '## By arm (means over seeds)', '',
              '| Arm | λ | Gap (nats) | S | Payload-window KL | Ratio, head B vs its copy | Readout overlap A, B | Head A, clean | Rotated |',
              '|---|---|---|---|---|---|---|---|---|']
        for a in ARMS:
            lam = mes[(a, seeds[0])].get('lambda')
            L.append(f"| {a} | {'—' if lam is None else f'{lam:g}'} | {st.mean(gap(a, s) for s in seeds):+.3f} | "
                     f"{'—' if a == 'clean' else f'{st.mean(S(a, s) for s in seeds):+.3f}'} | {st.mean(pkl(a, s) for s in seeds):.3f} | "
                     f"{st.mean(ratio(a, s) for s in seeds):.3f} | {st.mean(mes[(a, s)]['readout_overlap_A_B'] for s in seeds):.3f} | "
                     f"{st.mean(fn(a, s, 'A_clean') for s in seeds):.3f} | {st.mean(mes[(a, s)]['rotated_control']['agreement_ratio'] for s in seeds):.3f} |")
        L += ['', '## By seed: S', '', '| Seed | ' + ' | '.join(PAY) + ' | gap(n80) |', '|' + '---|' * (len(PAY) + 2)]
        for s in seeds:
            L.append(f'| {s} | ' + ' | '.join(f'{S(a, s):+.3f}' for a in PAY) + f" | {gap('n80', s):.3f} |")

        L += ['', '## Secondary and exploratory measures', '', 'None of these enters a decision rule.', '',
              '| Measure | ' + ' | '.join(PAY) + ' |', '|' + '---|' * (len(PAY) + 1)]
        L.append('| S1: ratio of head B with its own copy | ' + ' | '.join(f"{st.mean(ratio(a, s) for s in seeds):.3f}" for a in PAY) + ' |')
        L.append('| S2: head A\'s cost (nats, vs n80) | ' + ' | '.join(f"{st.mean(fn(a, s, 'A_clean') - fn('n80', s, 'A_clean') for s in seeds):+.3f}" for a in PAY) + ' |')
        L.append('| S3: S on the payload windows | ' + ' | '.join(f"{st.mean(S(a, s, 'payload_windows') for s in seeds):+.3f}" for a in PAY) + ' |')
        L.append('| S3: S on the measurement windows | ' + ' | '.join(f"{st.mean(S(a, s) for s in seeds):+.3f}" for a in PAY) + ' |')
        L.append('| S4: payload-window KL ÷ n80\'s | ' + ' | '.join(f"{st.mean(pkl(a, s) / pkl('n80', s) for s in seeds):.3f}" for a in PAY) + ' |')
        n10 = HERE / 'results' / 'n10'
        if n10.is_dir():
            m10 = [json.loads(p.read_text()) for p in sorted(n10.glob('m_n80_seed*.json'))]
            if m10:
                L.append('')
                L.append(f"S5, replication of N10's n80 (different seeds and windows, descriptive only): ratio with the copy "
                         f"{st.mean(ratio('n80', s) for s in seeds):.3f} (N10 {st.mean(r['B_vs_copy']['agreement_ratio'] for r in m10):.3f}); "
                         f"gap {st.mean(gap('n80', s) for s in seeds):.3f} (N10 {st.mean(r['functional']['G'] for r in m10):.3f}); "
                         f"head A {st.mean(fn('n80', s, 'A_clean') for s in seeds):.3f} (N10 {st.mean(r['functional']['A_clean'] for r in m10):.3f}).")
    L.append('')
    OUT.write_text('\n'.join(L))
    print('\n'.join(L[:14]))


if __name__ == '__main__':
    main()
