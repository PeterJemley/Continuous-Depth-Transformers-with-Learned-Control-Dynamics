"""Apply N12's preregistered decision rules to the saved records and write N12_RESULTS.md.

Written and committed before any real N12 run. Every number in the report comes from the JSON records in the
results folder. For the calibration run (one seed) it reports the same quantities and checks, with no verdicts.

The share of the gap closed, per seed: gap(x) = the loss on the clean targets of x's prediction of the noisy target,
minus that of the ideal copy of the same seed's n80 head A, on the held-out windows; S(x) = 1 - gap(x) / gap(n80).
x is an arm's head B (n80-mix: its mixture output; sub20: the mixture formed from its softmax) or a frozen fit.

Preconditions 1-4 apply to every claim; 5 to N12d; 6 to N12b.

    python n12_analyze.py [results folder] [report path]   # default: results/n12 here, N12_RESULTS.md
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

from scipy import stats

HERE = Path(__file__).resolve().parent
RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'results' / 'n12'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'N12_RESULTS.md'

REAL_SEEDS = [4101, 4102, 4103, 4104, 4105]
ARMS = {'clean': 0.0, 'n80': 0.8, 'n80-mix': 0.8, 'sub20': 0.8}
FITS = ('single', 'mix')
WINDOW = 128 + 8                                      # a training window with its 8 extra targets
NEEDED, LEVEL = 4, 0.975
KEEPS, SAME_READING, DIFFERENT_READING, SD_SENSITIVE = 0.67, 0.85, 0.60, 0.05
SAME_MIN, CLEAN_GAP, MATCH_MIN, SEPARATE = 0.95, 0.05, 0.999, 0.60
MIN_GAP, FIT_MIX_MIN, STABLE_MIN, NOISE_TOL = 0.30, 0.95, 0.80, 0.05
CONVERGED_REL, AT_TARGET = 0.02, 1e-3                # candidate amendment A1: or the fit's loss is below AT_TARGET
N11 = {'ratio': 0.323, 'gap': 0.492, 'saturated_payload_kl': 0.038}


def load(name):
    p = RES / f'{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


def fit_converged(cv):
    """Precondition 5's convergence test, recomputed from the recorded means (candidate amendment A1)."""
    return cv.get('last') is not None and (cv['relative_change'] <= CONVERGED_REL or cv['last'] <= AT_TARGET)


def interval(xs, level=LEVEL):
    n = len(xs)
    m = st.mean(xs)
    sd = st.stdev(xs) if n > 1 else float('nan')
    half = stats.t.ppf(1 - (1 - level) / 2, n - 1) * sd / math.sqrt(n) if n > 1 else float('nan')
    return m, sd, m - half, m + half


def interval_verdict(xs, margin, above, supported, label, null):
    """above: the claim is that the interval lies above the margin (N12a, N12c), else below it (N12d).
    null: the value 'some effect' must exclude (0 for a share closed, 1 for a fit with no bottleneck)."""
    m, sd, lo, hi = interval(xs)
    tail = f'{label} = {m:+.4f}, 97.5% interval [{lo:+.4f}, {hi:+.4f}], standard deviation {sd:.4f}.'
    if (lo > margin) if above else (hi < margin):
        return f'**Supported: {supported}** {tail}'
    if ((hi < margin) if above else (lo > margin)) and sd <= SD_SENSITIVE:
        return f'**Refuted.** {tail}'
    if lo > null or hi < null:
        return f'**Some effect, size unclear.** {tail}'
    return f'**Inconclusive.** {tail}'


def main():
    summary = load('n12_summary') or {}
    c = summary.get('checks', {})
    mode = summary.get('mode', 'real')
    want = summary.get('seeds', REAL_SEEDS) if mode == 'calibration' else REAL_SEEDS
    cut = summary.get('held_out', {}).get('cut_token')
    trn = {(a, s): load(f'{a}_seed{s}') for a in ARMS for s in want}
    mes = {(a, s): load(f'm_{a}_seed{s}') for a in ARMS for s in want}
    fit = {(f, s): load(f'f_{f}_seed{s}') for f in FITS for s in want}
    seeds = [s for s in want if all(trn[(a, s)] and mes[(a, s)] for a in ARMS) and all(fit[(f, s)] for f in FITS)]

    def fn(a, s, k):
        return mes[(a, s)]['functional'][k]

    def yard(s):
        return fn('n80', s, 'copy_clean')

    def gap(a, s):
        return fn(a, s, 'B_clean') - yard(s)

    def S(a, s):
        return 1 - gap(a, s) / gap('n80', s)

    def S_fit(f, s):
        return 1 - (fit[(f, s)]['functional']['B_clean'] - yard(s)) / gap('n80', s)

    def ratio(a, s, key='B_vs_copy'):
        return mes[(a, s)][key]['agreement_ratio']

    held_out = (cut is not None and bool(seeds)
                and all(trn[(a, s)]['training_tokens'] == cut and trn[(a, s)]['max_offset'] + WINDOW <= cut for a in ARMS for s in seeds)
                and all(fit[(f, s)]['max_offset'] + WINDOW <= cut for f in FITS for s in seeds))
    same_data = bool(seeds) and all(len({trn[(a, s)]['batch_fingerprint_step0'] for a in ARMS}) == 1 for s in seeds)
    same_mask = bool(seeds) and all(len({trn[(a, s)]['replaced_positions_fingerprint_step0'] for a in ('n80', 'n80-mix', 'sub20')}) == 1 for s in seeds)
    noise_ok = bool(seeds) and all(abs(trn[(a, s)]['noise_fraction_step0'] - 0.8) <= NOISE_TOL for a in ('n80', 'n80-mix', 'sub20') for s in seeds)
    pre = {
        '1. Setup: batch, same-model, identical-heads and analytic checks; windows; held-out tail untouched; noise fractions; same batches; sub20 kept exactly n80\'s unreplaced positions':
            bool(c.get('batch', {}).get('targets_are_i+1_to_i+8')) and bool(c.get('windows_shifted'))
            and bool(c.get('identical_heads', {}).get('pass')) and bool(c.get('analytic', {}).get('pass'))
            and all(x['pass'] for x in c.get('same_model', [{'pass': False}])) and held_out and same_data and same_mask and noise_ok,
        f'2. Must match: clean arm ratio ≥ {SAME_MIN} and |head B − head A| ≤ {CLEAN_GAP} nats in at least {NEEDED} of 5; '
        f'zero-noise copy ≥ {MATCH_MIN} and the constructed mixture check in every model':
            sum(ratio('clean', s, 'B_vs_A') >= SAME_MIN for s in seeds) >= min(NEEDED, len(want))
            and sum(abs(fn('clean', s, 'B_clean') - fn('clean', s, 'A_clean')) <= CLEAN_GAP for s in seeds) >= min(NEEDED, len(want))
            and bool(seeds) and all(ratio(a, s, 'copy0_vs_A') >= MATCH_MIN and mes[(a, s)]['constructed_mixture']['pass'] for a in ARMS for s in seeds),
        f'3. Must differ: rotated control ≤ {SEPARATE} in at least {NEEDED} of 5, in every arm':
            bool(seeds) and all(sum(mes[(a, s)]['rotated_control']['agreement_ratio'] <= SEPARATE for s in seeds) >= min(NEEDED, len(want)) for a in ARMS),
        f'4. The yardstick leaves a gap: gap(n80) ≥ {MIN_GAP} nats in every seed':
            bool(seeds) and all(gap('n80', s) >= MIN_GAP for s in seeds),
        f'5. The fits: converged (last 300 steps within 2% of the 300 before, or a loss below {AT_TARGET:g} nats), and fit-mix reached S ≥ {FIT_MIX_MIN}, in every model':
            bool(seeds) and all(fit_converged(fit[(f, s)]['convergence']) for f in FITS for s in seeds) and all(S_fit('mix', s) >= FIT_MIX_MIN for s in seeds),
        f'6. Stable subspaces (N12b only): head B and the copy ≥ {STABLE_MIN} in every n80-mix model':
            bool(seeds) and all(min(mes[('n80-mix', s)]['stability'][k] for k in ('B', 'copy')) >= STABLE_MIN for s in seeds),
    }
    names = list(pre)

    def gate(needed):
        if len(seeds) < 5:
            return f'**Not reached: complete records for only {len(seeds)} of 5 seeds.**'
        bad = [names[i].split(':')[0] for i in needed if not pre[names[i]]]
        return f'**Untestable: a precondition failed ({"; ".join(bad)}).**' if bad else None

    L = ['# N12 results' if mode == 'real' else '# N12 calibration run', '',
         'Generated by `n12_analyze.py` from the records in `results/` (' + RES.name + '). Preregistration: `N12_PREREGISTRATION.md`.',
         f"Measured on the held-out tail of the training text, windows {summary.get('held_out', {}).get('measure')} "
         f"(boundary at token {cut} of {summary.get('held_out', {}).get('training_tokens')}). S is the share of the gap closed against a fixed yardstick: "
         'the ideal noisy copy of the same seed\'s n80 head A.', '']
    if mode == 'real':
        v_a = gate([0, 1, 2, 3]) or interval_verdict([S('n80-mix', s) for s in seeds], KEEPS, True,
                                                     'a head that can represent the mixture keeps the reading, from the noisy targets alone.', 'S(n80-mix)', 0)
        v_b = gate([0, 1, 2, 3, 5])
        if not v_b:
            n_same = sum(ratio('n80-mix', s) >= SAME_READING for s in seeds)
            n_diff = sum(ratio('n80-mix', s) <= DIFFERENT_READING for s in seeds)
            if n_same >= NEEDED:
                v_b = f'**Supported: it reads the same directions as its ideal copy.** Ratio ≥ {SAME_READING} in {n_same} of 5 seeds.'
            elif n_diff >= NEEDED:
                v_b = f'**Refuted.** Ratio ≤ {DIFFERENT_READING} in {n_diff} of 5 seeds.'
            else:
                v_b = f'**Intermediate.** Ratio ≥ {SAME_READING} in {n_same} of 5 seeds; ≤ {DIFFERENT_READING} in {n_diff}.'
        v_c = gate([0, 1, 2, 3]) or interval_verdict([S('sub20', s) for s in seeds], KEEPS, True,
                                                     'scarcity alone does not cause the drift.', 'S(sub20)', 0)
        v_d = gate([0, 1, 2, 3, 4]) or interval_verdict([S_fit('single', s) for s in seeds], KEEPS, False,
                                                        'the bottleneck is large.', 'S(fit-single)', 1)
        L += ['## Verdicts', '',
              f'**N12a (primary: a head that can represent the mixture keeps the reading):** {v_a}', '',
              f'**N12b (and reads the same directions as its ideal copy):** {v_b}', '',
              f'**N12c (scarcity alone does not cause the drift):** {v_c}', '',
              f'**N12d (the bottleneck is large):** {v_d}', '']
    else:
        L += ['This is the calibration run of the preregistration\'s calibration step: one seed, on windows the real run does not use. '
              'It decides nothing, and no verdict is computed. Precondition 5 is the check it exists for.', '']
    L += ['## Preconditions', '', '| Precondition | Met |', '|---|---|']
    L += [f'| {k} | {"yes" if ok else "**no**"} |' for k, ok in pre.items()]

    if seeds:
        def mean(f):
            return st.mean([f(s) for s in seeds])
        L += ['', '## By arm (means over seeds)', '',
              '| Arm | Gap (nats) | S | Ratio, head B vs its copy | Head B vs head A | Inner softmax vs head A | Head A, clean | Head B, noisy targets | Copy, noisy targets | Rotated | Constructed check |',
              '|---|---|---|---|---|---|---|---|---|---|---|']
        for a in ARMS:
            inner = '—' if mes[(a, seeds[0])]['B_inner_vs_A'] is None else f"{mean(lambda s: ratio(a, s, 'B_inner_vs_A')):.3f}"
            s_col = '— (a clean head, not a noisy-target predictor)' if a == 'clean' else f'{mean(lambda s: S(a, s)):+.3f}'
            L.append(f"| {a} | {mean(lambda s: gap(a, s)):+.3f} | {s_col} | {mean(lambda s: ratio(a, s)):.3f} | "
                     f"{mean(lambda s: ratio(a, s, 'B_vs_A')):.3f} | {inner} | {mean(lambda s: fn(a, s, 'A_clean')):.3f} | "
                     f"{mean(lambda s: fn(a, s, 'B_noisy')):.3f} | {mean(lambda s: fn(a, s, 'copy_noisy')):.3f} | "
                     f"{mean(lambda s: mes[(a, s)]['rotated_control']['agreement_ratio']):.3f} | "
                     f"{min(mes[(a, s)]['constructed_mixture']['agreement_ratio'] for s in seeds):.4f} |")
        L += ['', '## The frozen fits (means over seeds)', '',
              '| Fit | S | Final fit KL(copy ‖ head B) | Relative change, last 300 steps | Ratio, head B vs copy | Copy\'s clean loss, minus n80\'s |',
              '|---|---|---|---|---|---|']
        for f in FITS:
            L.append(f"| fit-{f} | {mean(lambda s: S_fit(f, s)):+.3f} | {mean(lambda s: fit[(f, s)]['curve_every_50'][-1][1]):.4f} | "
                     f"{max(fit[(f, s)]['convergence']['relative_change'] or float('nan') for s in seeds):.4f} (largest) | "
                     f"{mean(lambda s: fit[(f, s)]['B_vs_copy']['agreement_ratio']):.3f} | "
                     f"{max(abs(fit[(f, s)]['functional']['copy_clean'] - yard(s)) for s in seeds):.2e} (largest) |")
        L += ['', '## By seed: S', '', '| Seed | n80-mix | sub20 | fit-single | fit-mix | gap(n80) | Ratio, n80-mix vs its copy |', '|---|---|---|---|---|---|---|']
        for s in seeds:
            L.append(f"| {s} | {S('n80-mix', s):+.3f} | {S('sub20', s):+.3f} | {S_fit('single', s):+.3f} | {S_fit('mix', s):+.3f} | "
                     f"{gap('n80', s):.3f} | {ratio('n80-mix', s):.3f} |")
        L += ['', '## Secondary and exploratory measures', '', 'None of these enters a decision rule.', '',
              '| Measure | n80 | n80-mix | sub20 |', '|---|---|---|---|',
              '| S1: inner softmax\'s clean loss minus head A\'s | — | ' + ' | '.join(
                  f"{mean(lambda s: fn(a, s, 'B_inner_clean') - fn(a, s, 'A_clean')):+.3f}" for a in ('n80-mix', 'sub20')) + ' |',
              '| S2: head A\'s cost (nats, vs n80), measurement windows | — | ' + ' | '.join(
                  f"{mean(lambda s: fn(a, s, 'A_clean') - fn('n80', s, 'A_clean')):+.4f}" for a in ('n80-mix', 'sub20')) + ' |',
              '| S3: head B\'s loss on the noisy targets, minus n80\'s | — | ' + ' | '.join(
                  f"{mean(lambda s: fn(a, s, 'B_noisy') - fn('n80', s, 'B_noisy')):+.4f}" for a in ('n80-mix', 'sub20')) + ' |', '',
              f"S4: fit-single's final fit divergence, {mean(lambda s: fit[('single', s)]['curve_every_50'][-1][1]):.4f}, against N11's saturated payload's "
              f"{N11['saturated_payload_kl']} (different windows and seeds; N11's was measured on its payload windows).", '',
              f"S5, replication of N11's n80 (different seeds and windows, descriptive only): ratio with the copy {mean(lambda s: ratio('n80', s)):.3f} "
              f"(N11 {N11['ratio']}); gap {mean(lambda s: gap('n80', s)):.3f} (N11 {N11['gap']}).", '']
        envs = sorted({(r['environment']['gpu'], r['environment']['torch'], r['environment']['code_commit'][:7])
                       for r in [trn[k] for k in trn if trn[k]] + [mes[k] for k in mes if mes[k]] + [fit[k] for k in fit if fit[k]]})
        L += ['## Run environment', ''] + [f'- {gpu}, PyTorch {v}, code {cm}' for gpu, v, cm in envs] + ['']
    OUT.write_text('\n'.join(L))
    print('\n'.join(L))


if __name__ == '__main__':
    main()
