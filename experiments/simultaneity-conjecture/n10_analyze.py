"""Apply N10's preregistered decision rules to the saved records and write N10_RESULTS.md.

Written and committed before any real N10 run. Every number in the report comes from the
JSON records in the results folder.

Preconditions 1, 2 and 4 apply to every claim; 3 and 6 to N10c and N10d; 5 to N10a and N10b
(its geometric part to N10b only, its functional part to both).

    python n10_analyze.py [results folder] [report path]   # default: results/n10 here, N10_RESULTS.md
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

from scipy import stats

HERE = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'results' / 'n10'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'N10_RESULTS.md'

SEEDS = [3901, 3902, 3903, 3904, 3905]
ARMS = {'clean': 0.0, 'n20': 0.2, 'n50': 0.5, 'n80': 0.8, 'n80-func': 0.8, 'n80-geom': 0.8, 'n80-data': 0.8}
DOSE = ('n20', 'n50', 'n80')
NEEDED, LEVEL = 4, 0.975
MARGIN, SD_SENSITIVE = 0.20, 0.05
SAME_MIN, CLEAN_G, MATCH_MIN = 0.95, 0.05, 0.999
RECOGNISE, SEPARATE, CONVERGED = 0.85, 0.60, 0.85
PINNED, KL_FRACTION, STABLE_MIN, NOISE_TOL = 0.90, 0.5, 0.80, 0.05


def load(name):
    p = RES / f'{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


def interval(xs, level=LEVEL):
    n = len(xs)
    m = st.mean(xs)
    sd = st.stdev(xs) if n > 1 else float('nan')
    half = stats.t.ppf(1 - (1 - level) / 2, n - 1) * sd / math.sqrt(n) if n > 1 else float('nan')
    return m, sd, m - half, m + half


def main():
    summary = load('n10_summary') or {}
    c = summary.get('checks', {})
    trn = {(a, s): load(f'{a}_seed{s}') for a in ARMS for s in SEEDS}
    mes = {(a, s): load(f'm_{a}_seed{s}') for a in ARMS for s in SEEDS}
    seeds = [s for s in SEEDS if all(trn[(a, s)] and mes[(a, s)] for a in ARMS)]

    def ratio(a, s, key='B_vs_copy'):
        return mes[(a, s)][key]['agreement_ratio']

    def G(a, s):
        return mes[(a, s)]['functional']['G']

    def pkl(a, s):
        return mes[(a, s)]['payload_windows']['kl_B_to_copy']

    # ---------------------------------------------------------------- preconditions
    same_data = bool(seeds) and all(len({trn[(a, s)].get('batch_fingerprint_step0') for a in ARMS}) == 1 for s in seeds)
    noise_ok = bool(seeds) and all(abs(trn[(a, s)]['noise_fraction_step0'] - ARMS[a]) <= NOISE_TOL
                                   for a in ARMS if ARMS[a] > 0 for s in seeds)
    p1 = (bool(c.get('batch', {}).get('targets_are_i+1_to_i+8')) and bool(c.get('test_windows_shifted'))
          and bool(c.get('identical_heads', {}).get('pass')) and bool(c.get('analytic', {}).get('pass'))
          and bool(c.get('payload_windows', {}).get('targets_shifted')) and bool(c.get('payload_windows', {}).get('disjoint_from_measurement'))
          and all(x['pass'] for x in c.get('same_model', [{'pass': False}])) and same_data and noise_ok)
    p2 = (sum(ratio('clean', s) >= SAME_MIN for s in seeds) >= NEEDED
          and sum(abs(G('clean', s)) <= CLEAN_G for s in seeds) >= NEEDED
          and bool(seeds) and all(ratio(a, s, 'copy0_vs_A') >= MATCH_MIN for a in ARMS for s in seeds))
    p3 = bool(seeds) and all(sum(ratio(a, s, 'A_vs_copy') >= RECOGNISE for s in seeds) >= NEEDED for a in ARMS)
    p4 = bool(seeds) and all(sum(mes[(a, s)]['rotated_control']['agreement_ratio'] <= SEPARATE for s in seeds) >= NEEDED for a in ARMS)
    p5g = bool(seeds) and all(mes[('n80-geom', s)]['readout_overlap_A_B'] >= PINNED for s in seeds)
    p5f = sum(pkl('n80-func', s) <= KL_FRACTION * pkl('n80', s) for s in seeds) >= NEEDED
    p6 = bool(seeds) and all(min(mes[(a, s)]['stability'][k] for k in mes[(a, s)]['stability'] if k != 'A') >= STABLE_MIN
                             for a in DOSE for s in seeds)
    pre = {
        '1. Setup: batch, same-model, identical-heads and analytic checks; windows; noise fractions; same batches; payload windows': p1,
        f'2. Must match: clean arm ratio ≥ {SAME_MIN} and |G| ≤ {CLEAN_G} in at least 4 of 5; zero-noise copy ≥ {MATCH_MIN} in every model': p2,
        f'3. The measure recognises copies: head A vs its copy ≥ {RECOGNISE} in at least 4 of 5, in every arm': p3,
        f'4. Must differ: rotated control ≤ {SEPARATE} in at least 4 of 5, in every arm': p4,
        f'5a. Geometric payload applied: readout overlap of P_A and P_B ≥ {PINNED} in every n80-geom model': p5g,
        f'5b. Functional payload applied: payload-window KL ≤ {KL_FRACTION} × n80\'s in at least 4 of 5 seeds': p5f,
        f'6. Stable subspaces: head B and the copies ≥ {STABLE_MIN} in every n20, n50 and n80 model': p6,
    }
    names = list(pre)

    def gate(needed):
        if len(seeds) < 5:
            return f'**Not reached: all seven arms have records for only {len(seeds)} of 5 seeds.**'
        bad = [names[i].split(':')[0] for i in needed if not pre[names[i]]]
        return f'**Untestable: a precondition failed ({"; ".join(bad)}).**' if bad else None

    # ---------------------------------------------------------------- claims
    gf = [G('n80-func', s) for s in seeds]
    v_a = gate([0, 1, 3, 5])
    if not v_a:
        m, sd, lo, hi = interval(gf)
        _, _, dlo, dhi = interval([G('n80', s) - G('n80-func', s) for s in seeds])
        tail = f'G(n80-func) = {m:+.4f}, 97.5% interval [{lo:+.4f}, {hi:+.4f}], standard deviation {sd:.4f}.'
        if hi < MARGIN:
            v_a = f'**Supported: the functional payload preserves the reading.** {tail}'
        elif lo > MARGIN and sd <= SD_SENSITIVE:
            v_a = f'**Refuted.** {tail}'
        elif dlo > 0 or dhi < 0:
            v_a = f'**Some effect, size unclear.** {tail} Paired reduction from n80: [{dlo:+.4f}, {dhi:+.4f}].'
        else:
            v_a = f'**Inconclusive.** {tail}'

    dgf = [G('n80-geom', s) - G('n80-func', s) for s in seeds]
    v_b = gate([0, 1, 3, 4, 5])
    if not v_b:
        m, sd, lo, hi = interval(dgf)
        tail = f'G(n80-geom) − G(n80-func) = {m:+.4f}, 97.5% interval [{lo:+.4f}, {hi:+.4f}], standard deviation {sd:.4f}.'
        if lo > MARGIN:
            v_b = f'**Supported: function, not geometry.** {tail}'
        elif hi < MARGIN and sd <= SD_SENSITIVE:
            v_b = f'**Refuted.** {tail}'
        elif lo > 0 or hi < 0:
            v_b = f'**Some effect, size unclear.** {tail}'
        else:
            v_b = f'**Inconclusive.** {tail}'

    v_c = gate([0, 1, 2, 3, 6])
    if not v_c:
        n_hi = sum(ratio('n20', s) >= CONVERGED for s in seeds)
        n_lo = sum(ratio('n20', s) <= SEPARATE for s in seeds)
        if n_hi >= NEEDED:
            v_c = f'**Supported: light noise does not change what the head reads.** Ratio ≥ {CONVERGED} in {n_hi} of 5.'
        elif n_lo >= NEEDED:
            v_c = f'**Refuted.** Ratio ≤ {SEPARATE} in {n_lo} of 5.'
        else:
            v_c = f'**Intermediate.** Ratio ≥ {CONVERGED} in {n_hi} and ≤ {SEPARATE} in {n_lo} of 5.'

    v_d = gate([0, 1, 2, 3, 6])
    if not v_d:
        n_ord = sum(ratio('n20', s) > ratio('n50', s) > ratio('n80', s) for s in seeds)
        n_rev = sum(ratio('n20', s) <= ratio('n80', s) for s in seeds)
        if n_ord >= NEEDED:
            v_d = f'**Supported: the change grows with the noise.** n20 > n50 > n80 in {n_ord} of 5 seeds.'
        elif n_rev >= NEEDED:
            v_d = f'**Refuted.** n20 ≤ n80 in {n_rev} of 5 seeds.'
        else:
            v_d = f'**Partial.** Ordered in {n_ord} of 5 seeds; n20 ≤ n80 in {n_rev}.'

    # ---------------------------------------------------------------- report
    L = ['# N10 results', '',
         'Generated by `n10_analyze.py` from the records in `results/n10/`. Preregistration: `N10_PREREGISTRATION.md`.',
         f"Readout rank {summary.get('readout_rank')}; subspaces of dimension {summary.get('subspace_rank')}, measured on "
         f"{summary.get('windows', {}).get('split')}-split windows {summary.get('windows', {}).get('measure')}. "
         'G is head B\'s loss on the clean targets minus its ideal copy\'s, in nats; ratios are N9\'s target-free agreement ratio.', '',
         '## Verdicts', '',
         f'**N10a (primary: a small functional payload preserves the reading under heavy noise):** {v_a}', '',
         f'**N10b (function, not geometry):** {v_b}', '',
         f'**N10c (light noise does not change what the head reads):** {v_c}', '',
         f'**N10d (the change grows with the noise):** {v_d}', '',
         '## Preconditions', '', '| Precondition | Met |', '|---|---|']
    L += [f'| {k} | {"yes" if ok else "**no**"} |' for k, ok in pre.items()]

    L += ['', '## By arm (means over seeds)', '',
          '| Arm | ρ | Ratio, head B vs its copy | G (nats) | KL(B ‖ copy) | Payload-window KL | Readout overlap A, B | Head A vs copy | Rotated |',
          '|---|---|---|---|---|---|---|---|---|']
    for a in ARMS:
        if seeds:
            L.append(f"| {a} | {ARMS[a]} | {st.mean(ratio(a, s) for s in seeds):.3f} | {st.mean(G(a, s) for s in seeds):+.3f} | "
                     f"{st.mean(mes[(a, s)]['functional']['kl_B_to_copy'] for s in seeds):.3f} | {st.mean(pkl(a, s) for s in seeds):.3f} | "
                     f"{st.mean(mes[(a, s)]['readout_overlap_A_B'] for s in seeds):.3f} | {st.mean(ratio(a, s, 'A_vs_copy') for s in seeds):.3f} | "
                     f"{st.mean(mes[(a, s)]['rotated_control']['agreement_ratio'] for s in seeds):.3f} |")

    L += ['', '## By seed: G (nats)', '', '| Seed | ' + ' | '.join(ARMS) + ' |', '|' + '---|' * (len(ARMS) + 1)]
    for s in seeds:
        L.append(f'| {s} | ' + ' | '.join(f'{G(a, s):+.3f}' for a in ARMS) + ' |')
    L += ['', '## By seed: ratio of head B with its ideal copy', '', '| Seed | ' + ' | '.join(ARMS) + ' |', '|' + '---|' * (len(ARMS) + 1)]
    for s in seeds:
        L.append(f'| {s} | ' + ' | '.join(f'{ratio(a, s):.3f}' for a in ARMS) + ' |')

    L += ['', '## Secondary and exploratory measures', '', 'None of these enters a decision rule.', '']
    if len(seeds) >= 2:
        for label, xs in (('S1: G(n80) − G(n80-data), the data payload\'s reduction', [G('n80', s) - G('n80-data', s) for s in seeds]),
                          ('S1: G(n80-data) − G(n80-func)', [G('n80-data', s) - G('n80-func', s) for s in seeds]),
                          ('S2: ratio of head B with its copy, n80-geom minus n80', [ratio('n80-geom', s) - ratio('n80', s) for s in seeds]),
                          ('S2: G(n80) − G(n80-geom), the geometric payload\'s reduction', [G('n80', s) - G('n80-geom', s) for s in seeds])):
            m, sd, lo, hi = interval(xs, 0.95)
            L.append(f'- {label}: mean {m:+.4f}, 95% interval [{lo:+.4f}, {hi:+.4f}].')
        L.append(f"- S4: n80 replicates N9? Ratio with the copy {st.mean(ratio('n80', s) for s in seeds):.3f} (N9: 0.31–0.35); "
                 f"G {st.mean(G('n80', s) for s in seeds):+.3f} (N9: +0.61).")
        L += ['', '### S5: losses (whole validation split, means over seeds)', '',
              '| Arm | Head A | Head B, own targets | Head B, clean targets |', '|---|---|---|---|']
        for a in ARMS:
            L.append(f"| {a} | {st.mean(trn[(a, s)]['final_loss_A'] for s in seeds):.4f} | {st.mean(trn[(a, s)]['final_loss_B'] for s in seeds):.4f} | "
                     f"{st.mean(trn[(a, s)]['final_loss_B_clean_targets'] for s in seeds):.4f} |")
    L.append('')
    OUT.write_text('\n'.join(L))
    print('\n'.join(L[:16]))


if __name__ == '__main__':
    main()
