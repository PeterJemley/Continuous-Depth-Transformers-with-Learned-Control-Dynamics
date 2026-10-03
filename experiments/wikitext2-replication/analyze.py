"""Apply the preregistered decision rules to the saved results and write RESULTS.md.

Reads only the JSON files in the results directory; every number in the report
comes from them, and each section names the files it used. Missing results are
reported as missing, never filled in.

    python analyze.py
"""
import json
import math
import statistics as st

from common import RESULTS, SEEDS, sign_flip_p, t_interval, wilson, binom_two_sided

T80 = {1: 1.376, 2: 1.061, 3: 0.978, 4: 0.941, 5: 0.920, 6: 0.906, 7: 0.896, 8: 0.889, 9: 0.883, 10: 0.879}
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}


def t975(df):
    df = max(1, int(math.floor(df)))
    for k in sorted(T975, reverse=True):
        if df >= k:
            return T975[k]
    return T975[1]


def load(name):
    p = RESULTS / f'{name}.json'
    return json.loads(p.read_text())['result'] if p.exists() else None


def f(x, d=4):
    return 'n/a' if x is None else (f'{x:.{d}f}' if isinstance(x, float) else str(x))


def pct(x, d=1):
    return 'n/a' if x is None else f'{100 * x:.{d}f}%'


class Report:
    def __init__(self):
        self.lines, self.verdicts = [], []

    def add(self, *ls):
        self.lines.extend(ls)

    def verdict(self, hyp, claim, outcome, basis):
        self.verdicts.append((hyp, claim, outcome, basis))


def main():
    R = Report()
    train = {k: {s: load(f'train_{k}_seed{s}') for s in SEEDS} for k in ('baseline', 'hybrid')}
    have = [s for s in SEEDS if train['baseline'][s] and train['hybrid'][s]]

    # ---- reproduction check
    R.add('## Reproduction check against the original run', '')
    b42, h42 = train['baseline'][42], train['hybrid'][42]
    if b42 and h42:
        checks = [('Baseline step-0 loss', b42['step0_loss'], 11.0107, 0.05), ('Model A step-0 loss', h42['step0_loss'], 10.9800, 0.05),
                  ('Baseline last-50 training loss', b42['last50_train_loss'], 6.4707, 0.1), ('Model A last-50 training loss', h42['last50_train_loss'], 6.4496, 0.1)]
        R.add('| Quantity | This run | Original | Allowed gap | Within? |', '|---|---|---|---|---|')
        ok = True
        for name, got, orig, tol in checks:
            w = abs(got - orig) <= tol
            ok &= w
            R.add(f'| {name} | {got:.4f} | {orig:.4f} | {tol} | {"yes" if w else "no"} |')
        R.add('', f'Source: `train_baseline_seed42.json` and `train_hybrid_seed42.json`, keys `step0_loss` and `last50_train_loss`.', '')
        R.verdict('R0', 'Seed 42 reproduces the original run', 'reproduced' if ok else 'not reproduced: investigate the data pipeline before reading H1 to H8', 'all four within tolerance' if ok else 'at least one outside tolerance')
    else:
        R.add('Missing: seed 42 training results.', '')

    # ---- H1
    R.add('## H1. Training is stable', '')
    ctl = load('h1_control')
    rows = []
    events = 0
    for s in have:
        log = train['hybrid'][s]['log']
        e_post = sum(1 for v in log['ode_grad_norm'] if v < 1e-6 or v > 100)
        e_pre = sum(1 for v in log['ode_grad_norm_preclip'] if v < 1e-6 or v > 100)
        events += e_post
        rows.append(f"| {s} | {e_post} | {max(log['ode_grad_norm']):.4f} | {e_pre} | {max(log['ode_grad_norm_preclip']):.4f} |")
    R.add('| Seed | Events, paper detector | Largest ODE gradient norm, after clipping | Events before clipping, exploratory | Largest before clipping |', '|---|---|---|---|---|', *rows, '')
    if ctl:
        fired = ctl['post_clip_flags']['vanishing'] + ctl['post_clip_flags']['exploding'] > 0
        R.add(f"Must-fail control, an unstable run with α = 10 and learning rate 0.1: the paper's detector flagged {ctl['post_clip_flags']}; "
              f"before clipping, {ctl['pre_clip_flags']}. Largest norm after clipping {ctl['post_clip_max']:.4f}; before clipping {ctl['pre_clip_max']:.4f}.", '')
        if not fired:
            out = 'untestable with this detector'
        else:
            out = 'refuted' if events else 'supported'
        R.verdict('H1', 'Training is stable', out, f'{events} events across {len(have)} seeds; control detector fired: {fired}')
    R.add('Source: `train_hybrid_seed*.json`, `log.ode_grad_norm` and `log.ode_grad_norm_preclip`; `h1_control.json`.', '')

    # ---- H2
    R.add('## H2. Validation loss against the baseline', '')
    if len(have) >= 2:
        vb = [train['baseline'][s]['val_loss'] for s in have]
        vh = [train['hybrid'][s]['val_loss'] for s in have]
        d = [h - b for h, b in zip(vh, vb)]
        R.add('| Seed | Baseline | Model A | Difference |', '|---|---|---|---|',
              *[f'| {s} | {b:.4f} | {h:.4f} | {h - b:+.4f} |' for s, b, h in zip(have, vb, vh)], '')
        m, sd, lo, hi = t_interval(d)
        p = sign_flip_p(d) if len(d) <= 20 else None
        n = len(d)
        mde = (T975.get(n - 1, 2.0) + T80.get(n - 1, 0.85)) * sd / math.sqrt(n)
        # unpaired Welch interval, reported alongside
        mb, mh = st.fmean(vb), st.fmean(vh)
        se = math.sqrt(st.variance(vb) / n + st.variance(vh) / n)
        dfw = se ** 4 / ((st.variance(vb) / n) ** 2 / (n - 1) + (st.variance(vh) / n) ** 2 / (n - 1))
        wlo, whi = (mh - mb) - t975(dfw) * se, (mh - mb) + t975(dfw) * se
        R.add(f'Paired difference, Model A minus baseline: mean {m:+.4f} nats, standard deviation {sd:.4f}, t-based 95% interval [{lo:+.4f}, {hi:+.4f}], '
              f'exact sign-flip p = {f(p)}. Model A lower in {sum(x < 0 for x in d)} of {n} seeds.',
              '', f'Unpaired Welch interval, reported alongside: [{wlo:+.4f}, {whi:+.4f}].',
              '', f'Smallest effect detectable with about 80% power at this spread: {mde:.4f} nats. The paper\'s gap was 0.0211 nats.', '')
        tb = [train['baseline'][s]['last50_train_loss'] for s in have]
        th = [train['hybrid'][s]['last50_train_loss'] for s in have]
        R.add(f'Secondary, the paper\'s own measure: mean last-50-step training loss {st.fmean(tb):.4f} for the baseline and {st.fmean(th):.4f} for Model A.', '')
        if m < 0 and p is not None and p < 0.05:
            out = 'supported: slightly better'
        elif lo >= -0.05 and hi <= 0.05:
            out = 'supported: comparable'
        elif lo > 0:
            out = 'refuted: worse'
        else:
            out = 'inconclusive'
        R.verdict('H2', 'Comparable to, or slightly better than, the baseline', out, f'mean {m:+.4f}, 95% [{lo:+.4f}, {hi:+.4f}], p = {f(p)}, n = {n}')
    else:
        R.add('Missing: fewer than two complete seeds.', '')
    R.add('Source: `train_*_seed*.json`, keys `val_loss` and `last50_train_loss`.', '')

    # ---- H3
    R.add('## H3. Final α', '')
    al = [(s, train['hybrid'][s]['alpha_final']) for s in SEEDS if train['hybrid'][s]]
    if al:
        vals = [a for _, a in al]
        R.add('| Seed | Final α | Smallest α during training | Largest α during training |', '|---|---|---|---|',
              *[f"| {s} | {a:.4f} | {min(train['hybrid'][s]['log']['alpha']):.4f} | {max(train['hybrid'][s]['log']['alpha']):.4f} |" for s, a in al], '')
        mean_a = st.fmean(vals)
        ok = 0.055 <= mean_a <= 0.075 and all(0.045 <= a <= 0.085 for a in vals)
        R.add(f'Mean final α {mean_a:.4f}.', '')
        R.verdict('H3', 'α converges to about 0.065', 'supported' if ok else 'refuted', f'mean {mean_a:.4f}, range {min(vals):.4f} to {max(vals):.4f}')
    R.add('Source: `train_hybrid_seed*.json`, keys `alpha_final` and `log.alpha`.', '')

    # ---- H4
    R.add('## H4. Solver dependence', '')
    sol = {s: load(f'solver_seed{s}') for s in SEEDS}
    sol = {s: v for s, v in sol.items() if v}
    accurate = ['euler8', 'euler16', 'rk4_4', 'dopri5_1e-5']
    if sol:
        R.add('Relative change in validation loss against the training solver, 4 Euler steps:', '',
              '| Seed | Model | ' + ' | '.join(accurate) + ' |', '|---|---|' + '---|' * len(accurate))
        trained_fail, control_fail = False, False
        for s, v in sol.items():
            for label, block in [('trained', v['trained'])] + [('untrained', v['untrained'])] + [(f'α × {k}', b) for k, b in v['scaled'].items()]:
                cells = []
                for a in accurate:
                    r = block['losses'][a]
                    rc = r['rel_change_vs_euler4']
                    failed = rc is None or rc >= 0.01
                    if label == 'trained' and failed:
                        trained_fail = True
                    if label.startswith('α') and failed:
                        control_fail = True
                    cells.append('error' if rc is None else pct(rc, 3))
                R.add(f'| {s} | {label} | ' + ' | '.join(cells) + ' |')
        R.add('', 'Hidden-state error of 4 Euler steps against dopri5, relative to how far the ODE block moves the state:', '',
              '| Seed | Model | Block moves the state by | Error relative to that movement | Error relative to the state |', '|---|---|---|---|---|')
        for s, v in sol.items():
            for label, block in [('trained', v['trained']), ('untrained', v['untrained'])] + [(f'α × {k}', b) for k, b in v['scaled'].items()]:
                dv = block['divergence']
                if 'error' in dv:
                    R.add(f'| {s} | {label} | error | error | error |')
                else:
                    R.add(f"| {s} | {label} | {pct(dv['block_moves_rel_state'], 2)} | {pct(dv['err_rel_movement'], 3)} | {pct(dv['err_rel_state'], 4)} |")
        R.add('', f"Controls used the first {next(iter(sol.values()))['control_limit']} validation windows; the trained model used the whole split.", '')
        if not control_fail:
            out = 'uninformative: the test never failed, even at 16 times the learned α'
        else:
            out = 'refuted' if trained_fail else 'supported'
        R.verdict('H4', 'Behaves as a continuous ODE under more accurate solvers', out, f'trained model failed in any seed: {trained_fail}; test failed for some scaled control: {control_fail}')
    R.add('Source: `solver_seed*.json`.', '')

    # ---- H5
    R.add('## H5. Steering', '')
    for tok, label in [(922, "corrected target ' good', token 922"), (220, "the paper's target, token 220, a space")]:
        st_ = {s: load(f'steer{tok}_seed{s}') for s in SEEDS}
        st_ = {s: v for s, v in st_.items() if v}
        if not st_:
            continue
        R.add(f'### Target: {label}', '',
              '| Seed | P(target), u = +1 | P(bad), u = +1 | P(target), u = −1 | P(bad), u = −1 | P(target), u = 0 | P(bad), u = 0 | Held-out correct | Validation loss before → after |',
              '|---|---|---|---|---|---|---|---|---|')
        succ = neutral = 0
        k_tot = n_tot = 0
        shares, path_ok = [], True
        for s, v in st_.items():
            a, b, c = v['train_prompt_u+1'], v['train_prompt_u-1'], v['train_prompt_u0']
            succ += a['p_pos'] > a['p_neg'] and b['p_neg'] > b['p_pos']
            neutral += c['p_pos'] > c['p_neg']
            k_tot += v['held_out_correct']; n_tot += v['held_out_total']
            shares.append(v['held_out_correct'] / v['held_out_total'])
            path_ok &= v['path_check_max_logit_diff'] <= 1e-6
            R.add(f"| {s} | {pct(a['p_pos'])} | {pct(a['p_neg'])} | {pct(b['p_pos'])} | {pct(b['p_neg'])} | {pct(c['p_pos'])} | {pct(c['p_neg'])} | "
                  f"{v['held_out_correct']}/{v['held_out_total']} | {v['val_loss_before']:.3f} → {v['val_loss_after']:.3f} |")
        lo, hi = wilson(k_tot, n_tot)
        pval = binom_two_sided(k_tot, n_tot)
        mean_share = st.fmean(shares)
        R.add('', f'Held-out prompts pooled: {k_tot}/{n_tot} correct ({pct(k_tot / n_tot)}), 95% Wilson interval {pct(lo)} to {pct(hi)}, '
              f'exact binomial p = {pval:.3g} against 50%. Mean share per seed {pct(mean_share)}. Path check passed in every seed: {path_ok}.', '')
        if tok == 922:
            n = len(st_)
            R.verdict('H5a', 'Steers the training prompt in at least 7 of 8 seeds', 'supported' if succ >= 7 else 'refuted', f'{succ} of {n} seeds')
            if mean_share >= 0.75 and k_tot / n_tot > 0.5 and pval < 0.05:
                o = 'supported'
            elif lo <= 0.5 <= hi or hi < 0.5:
                o = 'refuted'
            else:
                o = 'inconclusive: partial transfer'
            R.verdict('H5b', 'Transfers to held-out prompts', o, f'{k_tot}/{n_tot}, Wilson [{pct(lo)}, {pct(hi)}], mean share {pct(mean_share)}')
            R.verdict('H5c', 'Neutral u favours " good" in at least 7 of 8 seeds', 'supported' if neutral >= 7 else 'refuted', f'{neutral} of {n} seeds')
            R.verdict('H5d', 'Path check: no effect without the control input', 'passed' if path_ok else 'failed: investigate as a bug', 'max logit difference ≤ 1e-6 in every seed' if path_ok else 'a seed exceeded 1e-6')
        R.add(f'Source: `steer{tok}_seed*.json`.', '')

    # ---- H6
    R.add('## H6. Solver effort across the control signal', '')
    for tok in (922, 220):
        nf = {s: load(f'nfe{tok}_seed{s}') for s in SEEDS}
        nf = {s: v for s, v in nf.items() if v}
        if not nf:
            continue
        R.add(f'Steered with token {tok}. Function evaluations at u = −2 to 2 in steps of 0.2; dopri5 makes 2 plus 6 per solver step.', '',
              '| Seed | Evaluations | Mean, u < 0.3 | Mean, u > 0.4 |', '|---|---|---|---|')
        more = 0
        for s, v in nf.items():
            more += v['mean_nfe_u_below_0.3'] > v['mean_nfe_u_above_0.4']
            R.add(f"| {s} | {' '.join(str(r['nfe']) for r in v['rows'])} | {v['mean_nfe_u_below_0.3']:.1f} | {v['mean_nfe_u_above_0.4']:.1f} |")
        R.add('')
        if tok == 922:
            R.verdict('H6', 'Negative u needs more solver effort in at least 7 of 8 seeds', 'supported' if more >= 7 else 'refuted', f'{more} of {len(nf)} seeds')
        R.add(f'Source: `nfe{tok}_seed*.json`.', '')

    # ---- H7
    R.add('## H7. Latency', '')
    lat = load('latency')
    if lat:
        R.add('| Process | Baseline median | Model A median | Ratio |', '|---|---|---|---|',
              *[f"| {r['process']} | {1000 * r['median_baseline_s']:.0f} ms | {1000 * r['median_hybrid_s']:.0f} ms | {r['ratio']:.3f} |" for r in lat['runs']], '')
        ok = all(r['ratio'] <= 1.05 for r in lat['runs'])
        R.verdict('H7', 'Latency parity, ratio at most 1.05', 'supported' if ok else 'refuted', ', '.join(f"{r['ratio']:.3f}" for r in lat['runs']))
    R.add('Source: `latency.json`.', '')

    # ---- H8
    R.add('## H8. Linear probes across depth', '')
    for model in ('steer922', 'steer220', 'untrained'):
        pr = {s: load(f'probe_{model}_seed{s}') for s in SEEDS}
        pr = {s: v for s, v in pr.items() if v}
        if not pr:
            continue
        taus = [r['tau'] for r in next(iter(pr.values()))['rows']]
        R.add(f'Model: {model}. Accuracy / control accuracy on 512 test windows:', '', '| Seed | ' + ' | '.join(f'τ = {t}' for t in taus) + ' |', '|---|' + '---|' * len(taus))
        tau0_ok = True
        for s, v in pr.items():
            R.add(f'| {s} | ' + ' | '.join(f"{r['accuracy']:.3f} / {r['control_accuracy']:.3f}" for r in v['rows']) + ' |')
            lo, hi = v['rows'][0]['accuracy_wilson95']
            tau0_ok &= lo <= 0.5 <= hi
        mean_sel = [st.fmean(v['rows'][i]['selectivity'] for v in pr.values()) for i in range(len(taus))]
        rng = max(mean_sel[1:]) - min(mean_sel[1:])
        R.add('', f"Mean selectivity by τ: {', '.join(f'{t}: {x:+.3f}' for t, x in zip(taus, mean_sel))}. Range over τ > 0: {rng:.3f}. "
              f'τ = 0 at chance in every seed: {tau0_ok}.', '')
        if model == 'steer922':
            out = 'supported' if rng < 0.05 else 'refuted'
            if not tau0_ok:
                out += '; but τ = 0 was not at chance, so investigate a leak first'
            R.verdict('H8', 'Probe selectivity is flat across τ > 0', out, f'range of mean selectivity {rng:.3f}')
    R.add('Source: `probe_*_seed*.json`.', '')

    # ---- write
    complete = len(have) == len(SEEDS)
    head = ['# Results: WikiText-2 replication', '',
            'Generated by `analyze.py` from the files in `results/`. Every number below comes from those files.', '',
            f'Seeds with complete training results: {len(have)} of {len(SEEDS)}.' + ('' if complete else ' **Incomplete: verdicts below are provisional.**'), '',
            '## Verdicts', '', '| Hypothesis | Claim | Outcome | Basis |', '|---|---|---|---|',
            *[f'| {h} | {c} | {o} | {b} |' for h, c, o, b in R.verdicts], '',
            'Each hypothesis is reported separately, so no multiple-comparison correction is applied. Any claim that combines several would need a Holm correction.', '']
    (RESULTS.parent / ('RESULTS.md' if RESULTS.name == 'results' else f'RESULTS_{RESULTS.name}.md')).write_text('\n'.join(head + R.lines) + '\n')
    for v in R.verdicts:
        print(' | '.join(v))


if __name__ == '__main__':
    main()
