"""Apply N8's preregistered decision rules to the saved records and write N8_RESULTS.md.

Written and committed before any real N8 run. Every number in the report comes from the
JSON records in the results folder.

Amendment A1 adds precondition 3b (a must-differ control) and S4 (readout drift).
Preconditions 2-4 are applied to each set of models separately: N7's models for N8a and N8b, N6r's
models for N8c. Precondition 1 (the estimator) applies to all three.

    python n8_analyze.py [results folder] [report path]   # default: results/n8 here, N8_RESULTS.md
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

from scipy import stats

HERE = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'results' / 'n8'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'N8_RESULTS.md'

SETS = {'n7': ([3801, 3802, 3803, 3804, 3805], ('same', 'k4', 'k8', 'noisy')),
        'n6r': ([3701, 3702, 3703, 3704, 3705], ('same', 'k2', 'k4', 'k8'))}
DECIDING = {'n7': ('k8', 'noisy'), 'n6r': ('k2', 'k8')}      # arms whose heads enter a decision rule
NEEDED, LEVEL = 4, 0.975
MARGIN_A, MARGIN_C, SD_SENSITIVE = 0.30, 0.10, 0.05
SAME_MIN, STABLE_MIN = 0.95, 0.80
CONVERGED, SEPARATE = 0.85, 0.60


def load(name):
    p = RES / f'{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


def interval(xs, level=LEVEL):
    n = len(xs)
    m = st.mean(xs)
    sd = st.stdev(xs) if n > 1 else float('nan')
    half = stats.t.ppf(1 - (1 - level) / 2, n - 1) * sd / math.sqrt(n) if n > 1 else float('nan')
    return m, sd, m - half, m + half


def interval_verdict(diffs, margin, supported, refuted):
    m, sd, lo, hi = interval(diffs)
    tail = f'mean {m:+.4f}, 97.5% interval [{lo:+.4f}, {hi:+.4f}], standard deviation {sd:.4f}.'
    if lo > margin:
        return f'**Supported: {supported}** {tail}'
    if hi < margin and sd <= SD_SENSITIVE:
        return f'**Refuted{": " + refuted if refuted else ""}.** {tail}'
    if lo > 0 or hi < 0:
        return f'**Some effect, size unclear.** {tail}'
    return f'**Inconclusive.** {tail}'


def main():
    summary = load('n8_summary') or {}
    c = summary.get('checks', {})
    rec = {(sn, a, s): load(f'f_{sn}_{a}_seed{s}') for sn, (seeds, arms) in SETS.items() for s in seeds for a in arms}
    done = {sn: [s for s in seeds if all(rec[(sn, a, s)] for a in arms)] for sn, (seeds, arms) in SETS.items()}

    def f(sn, a, s, key='overlap_AB'):
        return rec[(sn, a, s)]['fisher'][key]

    def old(sn, a, s, key='overlap_AB'):
        return rec[(sn, a, s)]['old'][key]

    est_ok = bool(c.get('analytic', {}).get('pass')) and bool(c.get('identical_heads', {}).get('pass')) and bool(c.get('test_windows_shifted'))
    pre, pre_ok, failed = {}, {}, {}
    for sn, (seeds, arms) in SETS.items():
        ss = done[sn]
        p = {
            '1. The estimator works: the analytic check, the identical-heads check, and the windows': est_ok,
            f'2. The right models are loaded: the old overlap within 0.05 of the recorded value, for every model':
                bool(ss) and all(rec[(sn, a, s)]['loading']['pass'] for a in arms for s in ss),
            f'3. Must fail: the same arm\'s target-free agreement ratio ≥ {SAME_MIN} in at least 4 of 5':
                sum(f(sn, 'same', s, 'agreement_ratio') >= SAME_MIN for s in ss) >= NEEDED,
            f'3b. Must differ (amendment A1): the rotated-readout control\'s target-free agreement ratio ≤ {SEPARATE} in at least 4 of 5':
                sum((rec[(sn, 'same', s)].get('rotated_control') or {}).get('agreement_ratio', 1.0) <= SEPARATE for s in ss) >= NEEDED,
            f'4. Stable subspaces: every head in arms {", ".join(DECIDING[sn])} has target-free stability ≥ {STABLE_MIN}':
                bool(ss) and all(min(f(sn, a, s, 'stability_A_odd_even'), f(sn, a, s, 'stability_B_odd_even')) >= STABLE_MIN
                                 for a in DECIDING[sn] for s in ss),
        }
        pre[sn], pre_ok[sn] = p, all(p.values())
        failed[sn] = '; '.join(k.split(':')[0] for k, ok in p.items() if not ok)

    def gate(sn):
        if len(done[sn]) < 5:
            return f'**Not reached: {sn} models have records for only {len(done[sn])} of 5 seeds.**'
        if not pre_ok[sn]:
            return f'**Untestable: a precondition failed for the {sn} models ({failed[sn]}).**'
        return None

    s7, s6 = done['n7'], done['n6r']
    diff_a = [f('n7', 'noisy', s) - f('n7', 'k8', s) for s in s7]
    v8a = gate('n7') or interval_verdict(diff_a, MARGIN_A, 'with errors removed, a harder copy of the same reading shares what it uses.',
                                         '').replace('mean', 'o_F(noisy) − o_F(8) = mean', 1)

    n_conv = sum(f('n7', 'noisy', s, 'agreement_ratio') >= CONVERGED for s in s7)
    n_sep = sum(f('n7', 'noisy', s, 'agreement_ratio') <= SEPARATE for s in s7)
    v8b = gate('n7')
    if not v8b:
        if n_conv >= NEEDED:
            v8b = f'**Supported: the noisy head converges.** Its target-free agreement ratio ≥ {CONVERGED} in {n_conv} of 5.'
        elif n_sep >= NEEDED:
            v8b = f'**Refuted.** Its target-free agreement ratio ≤ {SEPARATE} in {n_sep} of 5.'
        else:
            v8b = f'**Intermediate.** Ratio ≥ {CONVERGED} in {n_conv} and ≤ {SEPARATE} in {n_sep} of 5.'

    diff_c = [f('n6r', 'k2', s) - f('n6r', 'k8', s) for s in s6]
    v8c = gate('n6r') or interval_verdict(diff_c, MARGIN_C, 'the ladder persists.', 'it flattens').replace('mean', 'o_F(2) − o_F(8) = mean', 1)

    an = c.get('analytic', {})
    L = ['# N8 results', '',
         'Generated by `n8_analyze.py` from the records in `results/n8/`. Preregistration: `N8_PREREGISTRATION.md`.',
         f"Readout rank {summary.get('readout_rank')}; subspaces of dimension {summary.get('subspace_rank')}, measured on "
         f"{summary.get('windows', {}).get('split')}-split windows {summary.get('windows', {}).get('measure')}; "
         f"shared uniforms from seed {summary.get('fisher_seed')}. o_F is the target-free overlap.", '',
         '## Verdicts', '',
         f'**N8a (primary: with errors removed, a harder copy of the same reading shares what it uses):** {v8a}', '',
         f'**N8b (and it converges):** {v8b}', '',
         f'**N8c (the readings further ahead really do use different directions):** {v8c}', '',
         '## Preconditions', '', '| Models | Precondition | Met |', '|---|---|---|']
    for sn in SETS:
        L += [f'| {sn} | {k} | {"yes" if ok else "**no**"} |' for k, ok in pre[sn].items()]
    if an:
        L.append(f"\nAnalytic check ({an.get('positions')} positions, vocabulary {an.get('vocab')}, eigengap "
                 f"{an.get('eigengap_R_over_R+1', float('nan')):.2f}): sampled estimate's top-subspace overlap with the exact Fisher "
                 f"{an['sampled']['top_subspace_overlap']:.4f}, relative error {an['sampled']['relative_error']:.3f}. "
                 f"For contrast, tokens drawn uniformly: {an['uniform_tokens']['top_subspace_overlap']:.4f} "
                 f"(error {an['uniform_tokens']['relative_error']:.3f}); most likely token: "
                 f"{an['argmax_tokens']['top_subspace_overlap']:.4f} (error {an['argmax_tokens']['relative_error']:.3f}).")
    ih = c.get('identical_heads', {})
    if ih:
        L.append(f"Identical-heads check: agreement ratio {ih.get('agreement_ratio')}.")

    for sn, title in (('n7', "N7's models"), ('n6r', "N6r's models")):
        seeds, arms = SETS[sn]
        ss = done[sn]
        last = ('noisy', 'k8') if sn == 'n7' else ('k2', 'k8')
        L += ['', f'## {title}: target-free overlap / agreement ratio with the clean next-token head, by seed', '',
              '| Seed | ' + ' | '.join(arms) + f' | o_F({"noisy" if sn == "n7" else 2}) − o_F(8) |',
              '|' + '---|' * (len(arms) + 2)]
        for s in ss:
            cells = [f"{f(sn, a, s):.3f} / {f(sn, a, s, 'agreement_ratio'):.3f}" for a in arms]
            L.append(f'| {s} | ' + ' | '.join(cells) + f' | {f(sn, last[0], s) - f(sn, last[1], s):+.3f} |')
        diffs = diff_a if sn == 'n7' else diff_c
        if len(ss) >= 2:
            m, sd, lo, hi = interval(diffs)
            L.append(f'\nMean {m:+.4f}, standard deviation {sd:.4f}, 97.5% interval [{lo:+.4f}, {hi:+.4f}].')

    L += ['', '## Secondary and exploratory measures', '', 'None of these enters a decision rule.', '',
          '### S1 and S3: both measures side by side (means over seeds, 97.5% intervals for the overlaps)', '',
          '| Models | Arm | Target-free overlap | Old overlap | Target-free ratio | Old ratio | Target-free stability A / B | Old stability A / B |',
          '|---|---|---|---|---|---|---|---|']
    for sn, (seeds, arms) in SETS.items():
        ss = done[sn]
        if len(ss) < 2:
            continue
        for a in arms:
            mf, _, flo, fhi = interval([f(sn, a, s) for s in ss])
            mo, _, olo, ohi = interval([old(sn, a, s) for s in ss])
            L.append(f"| {sn} | {a} | {mf:.3f} [{flo:.3f}, {fhi:.3f}] | {mo:.3f} [{olo:.3f}, {ohi:.3f}] | "
                     f"{st.mean(f(sn, a, s, 'agreement_ratio') for s in ss):.3f} | {st.mean(old(sn, a, s, 'agreement_ratio') for s in ss):.3f} | "
                     f"{st.mean(f(sn, a, s, 'stability_A_odd_even') for s in ss):.3f} / {st.mean(f(sn, a, s, 'stability_B_odd_even') for s in ss):.3f} | "
                     f"{st.mean(old(sn, a, s, 'stability_A_odd_even') for s in ss):.3f} / {st.mean(old(sn, a, s, 'stability_B_odd_even') for s in ss):.3f} |")
    L += ['', '### S2: old overlap minus target-free overlap, by arm (mean over seeds)', '', '| Models | Arm | Old − target-free |', '|---|---|---|']
    for sn, (seeds, arms) in SETS.items():
        ss = done[sn]
        for a in arms:
            if ss:
                L.append(f'| {sn} | {a} | {st.mean(old(sn, a, s) - f(sn, a, s) for s in ss):+.3f} |')
    L += ['', '### Loading check', '', '| Models | Arm | Largest |old overlap − recorded| over seeds | Recorded from |', '|---|---|---|---|']
    for sn, (seeds, arms) in SETS.items():
        ss = done[sn]
        for a in arms:
            if ss:
                ds = [rec[(sn, a, s)]['loading']['difference'] for s in ss]
                big = max(abs(d) for d in ds) if all(d is not None for d in ds) else None
                L.append(f"| {sn} | {a} | {'missing' if big is None else f'{big:.3f}'} | "
                         f"{', '.join(sorted({rec[(sn, a, s)]['loading']['recorded_from'] for s in ss}))} |")
    L += ['', '### Rotated-readout control (precondition 3b), by seed', '', '| Models | Seed | Overlap | Agreement ratio |', '|---|---|---|---|']
    for sn in SETS:
        for s in done[sn]:
            rc = rec[(sn, 'same', s)].get('rotated_control')
            if rc:
                L.append(f"| {sn} | {s} | {rc['overlap_AB']:.3f} | {rc['agreement_ratio']:.3f} |")
    L += ['', '### S4 (amendment A1): how far each readout moved from the projection both heads started as (means over seeds)', '',
          'Row-space overlap of the trained projection with the initial one (1 = not moved; 0.25 = random at rank 64 of 256).', '',
          '| Models | Arm | P_A vs initial | P_B vs initial | P_A vs P_B |', '|---|---|---|---|---|']
    for sn, (seeds, arms) in SETS.items():
        ss = [s for s in done[sn] if all(rec[(sn, a, s)].get('readout_drift') for a in arms)]
        for a in arms:
            if ss:
                dr = [rec[(sn, a, s)]['readout_drift'] for s in ss]
                L.append(f"| {sn} | {a} | {st.mean(d['A_vs_initial'] for d in dr):.3f} | {st.mean(d['B_vs_initial'] for d in dr):.3f} | "
                         f"{st.mean(d['A_vs_B'] for d in dr):.3f} |")
    unt = summary.get('untrained', {})
    if unt:
        L += ['', '### Untrained models under the target-free measure', '',
              '| Seed | Overlap | Agreement ratio | Stability A / B | Rotated control ratio |', '|---|---|---|---|---|']
        for k, v in unt.items():
            rc = v.get('rotated_control', {}).get('agreement_ratio')
            L.append(f"| {k[4:]} | {v['overlap_AB']:.3f} | {v['agreement_ratio']:.3f} | {v['stability_A_odd_even']:.3f} / "
                     f"{v['stability_B_odd_even']:.3f} | {'' if rc is None else f'{rc:.3f}'} |")
    L.append('')
    OUT.write_text('\n'.join(L))
    print('\n'.join(L[:14]))


if __name__ == '__main__':
    main()
