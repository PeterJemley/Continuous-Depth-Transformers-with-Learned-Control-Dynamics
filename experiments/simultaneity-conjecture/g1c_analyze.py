"""Apply G1c's preregistered decision rule to the saved records and write G1C_RESULTS.md.

Written and committed before any real G1c run. Every number in the report comes from the
JSON records in the results folder.

    python g1c_analyze.py [results folder] [report path]   # default: results/g1c here, G1C_RESULTS.md
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from g1_analyze import classifier_accuracy, cohen_d, log_ratio  # noqa: E402

RES = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'results' / 'g1c'
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else HERE / 'G1C_RESULTS.md'

SEEDS = [3101, 3102, 3103, 3104, 3105]
NEEDED = 4
FLOOR, CLEAR = 0.1, 0.2            # contraction rates near zero; "u = -1 contracts" when u = +1 does not
LARGE, SMALL, LOW = 2.0, 1.25, 0.8
SPLIT_MAX, PROBE_MAX = 0.1, 0.05   # must match A and B, in log ratio
INIT_MAX = 0.1                     # must fail B: |c(-1) - c(+1)| in untrained models
MANIP_MIN = 0.5


def load(name):
    p = RES / f'{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


def cbar(rec, u, key='c', rows=slice(None)):
    """Model-level contraction rate: mean over the chosen windows and the five depths."""
    return float(np.asarray(rec['controls'][u][key], float)[rows].mean())


def score(cp, cm):
    """The preregistered scoring table. Returns (case, R or None, category)."""
    if cp >= FLOOR and cm >= FLOOR:
        r = cm / cp
        return 'both contract', r, 'large' if r >= LARGE else 'small' if r < SMALL else 'middle'
    if cm < FLOOR:
        return 'u = -1 does not contract', None, 'small'
    if cm >= CLEAR:
        return 'u = -1 contracts, u = +1 does not', None, 'large'
    return 'u = -1 contracts weakly, u = +1 does not', None, 'middle'


def agree(a, b, tol):
    """Must-match comparison of two (cp, cm) pairs: log R within tol where both define R, else the same case."""
    sa, sb = score(*a), score(*b)
    if sa[1] is not None and sb[1] is not None:
        return abs(math.log(sa[1]) - math.log(sb[1])), abs(math.log(sa[1]) - math.log(sb[1])) < tol
    return None, sa[0] == sb[0]


def f(x, d=3, sign=False):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 'n/a'
    return f'{x:+.{d}f}' if sign else f'{x:.{d}f}'


def main():
    summary = load('g1c_summary') or {}
    checks = summary.get('checks', {})
    con = {(v, s): load(f'contr_{v}_seed{s}') for v in ('main', 'shuf', 'init') for s in SEEDS}
    geo = {(v, s): load(f'geom_{v}_seed{s}') for v in ('main', 'shuf') for s in SEEDS}
    trn = {(v, s): load(f'train_{v}_seed{s}') for v in ('main', 'shuf') for s in SEEDS}
    have = {v: [s for s in SEEDS if con[(v, s)]] for v in ('main', 'shuf', 'init')}
    main_seeds = [s for s in have['main'] if trn[('main', s)]]
    complete = all(len(have[v]) == len(SEEDS) for v in have) and len(main_seeds) == len(SEEDS)

    rates = {k: (cbar(r, 'plus'), cbar(r, 'minus')) for k, r in con.items() if r}
    scored = {k: score(*rates[k]) for k in rates if k[0] != 'init'}
    manip = {s: trn[('main', s)]['final_losses']['next_uminus'] - trn[('main', s)]['final_losses']['next_uplus'] for s in main_seeds}

    def shuf_ok(s):
        cp, cm = rates[('shuf', s)]
        _, r, _ = scored[('shuf', s)]
        if r is not None:
            return LOW <= r < SMALL
        return cp < FLOOR and cm < FLOOR and abs(cm - cp) < INIT_MAX

    split, probe = {}, {}
    for s in main_seeds:
        r = con[('main', s)]
        split[s] = agree((cbar(r, 'plus', rows=slice(0, None, 2)), cbar(r, 'minus', rows=slice(0, None, 2))),
                         (cbar(r, 'plus', rows=slice(1, None, 2)), cbar(r, 'minus', rows=slice(1, None, 2))), SPLIT_MAX)
        probe[s] = agree(rates[('main', s)], (cbar(r, 'plus', 'c_probe2'), cbar(r, 'minus', 'c_probe2')), PROBE_MAX)

    pre = {
        '1. Manipulation check: wrong control costs at least 0.5 nats on the next-token task, in at least 4 of 5 main models':
            sum(manip[s] >= MANIP_MIN for s in main_seeds) >= NEEDED,
        '2. Analytic checks: divergence (diagonal exact, dense within 2%) and curvature':
            bool(checks.get('divergence', {}).get('pass')) and bool(checks.get('curvature', {}).get('pass')),
        '3. Must fail A: shuffled-control models give R from 0.8 to below 1.25, in at least 4 of 5':
            sum(shuf_ok(s) for s in have['shuf']) >= NEEDED,
        '4. Must fail B: untrained models give |c(−1) − c(+1)| below 0.1, in at least 4 of 5':
            sum(abs(rates[('init', s)][1] - rates[('init', s)][0]) < INIT_MAX for s in have['init']) >= NEEDED,
        '5. Must match A: odd and even windows agree (log R within 0.1, or the same case) in every main model':
            bool(main_seeds) and all(split[s][1] for s in main_seeds),
        '6. Must match B: a second probe set agrees (log R within 0.05, or the same case) in every main model':
            bool(main_seeds) and all(probe[s][1] for s in main_seeds),
    }
    n_large = sum(scored[('main', s)][2] == 'large' for s in main_seeds)
    n_small = sum(scored[('main', s)][2] == 'small' for s in main_seeds)
    if not complete:
        verdict = (f"**Not reached: records for {len(main_seeds)} of 5 main, {len(have['shuf'])} of 5 shuffled-control "
                   f"and {len(have['init'])} of 5 untrained models.**")
    elif not all(pre.values()):
        failed = [k.split(':')[0] for k, ok in pre.items() if not ok]
        verdict = f'**Untestable: precondition {", ".join(failed)} failed.** The values below decide nothing.'
    elif n_large >= NEEDED:
        verdict = f'**Supported: the control sets the contraction rate.** R ≥ 2 in {n_large} of 5 main models.'
    elif n_small >= NEEDED:
        verdict = f'**Refuted.** R < 1.25 in {n_small} of 5 main models, and the models use u.'
    else:
        verdict = f'**Intermediate.** R ≥ 2 in {n_large} and R < 1.25 in {n_small} of 5 main models; neither rule is met.'

    L = ['# G1c results', '',
         'Generated by `g1c_analyze.py` from the records in `results/g1c/`. Preregistration: `G1C_PREREGISTRATION.md`.',
         f"Validation windows {summary.get('windows', ['?', '?'])[0]}–{summary.get('windows', ['?', '?'])[1]}.", '',
         '## Verdict', '', verdict, '', '## Preconditions', '', '| Precondition | Met |', '|---|---|']
    L += [f'| {k} | {"yes" if ok else "**no**"} |' for k, ok in pre.items()]

    L += ['', '## Primary outcome: the contraction ratio R = c̄(−1) / c̄(+1)', '',
          'c̄ is minus the per-token divergence of the vector field, averaged over five depths and the windows. '
          'Rates below 0.1 are scored by the preregistered table, not as a ratio.', '',
          '| Seed | Model | c̄(+1) | c̄(−1) | R | Scored as | Case |', '|---|---|---|---|---|---|---|']
    for v in ('main', 'shuf'):
        for s in SEEDS:
            if (v, s) in scored:
                cp, cm = rates[(v, s)]
                case, r, cat = scored[(v, s)]
                L.append(f"| {s} | {v} | {cp:.4f} | {cm:.4f} | {f(r)} | {cat} | {case} |")
    L += ['', f'Main models scored R ≥ 2: {n_large}. Scored R < 1.25: {n_small}.', '',
          '### Controls', '', '| Seed | Untrained c̄(+1) | Untrained c̄(−1) | Untrained difference | Odd − even log R (main) | '
          'Probe set 1 − 2 log R (main) | Wrong-control cost (nats) |', '|---|---|---|---|---|---|---|']
    for s in SEEDS:
        ip = rates.get(('init', s))
        L.append(f"| {s} | {f(ip[0], 4, True) if ip else 'n/a'} | {f(ip[1], 4, True) if ip else 'n/a'} | "
                 f"{f(ip[1] - ip[0], 4, True) if ip else 'n/a'} | {f(split.get(s, (None,))[0], 4)} | "
                 f"{f(probe.get(s, (None,))[0], 4)} | {f(manip.get(s), 3, True)} |")

    # ---------------------------------------------------------------- secondary
    L += ['', '## Secondary and exploratory measures', '', 'None of these enters the decision rule.', '',
          '### S1: contraction by depth, main models (mean over models)', '',
          '| Depth | c̄(+1) | c̄(−1) | c̄(0) | Ratio (−1 / +1) |', '|---|---|---|---|---|']
    if main_seeds:
        times = con[('main', main_seeds[0])]['times']
        for k, t in enumerate(times):
            m = {u: np.mean([np.asarray(con[('main', s)]['controls'][u]['c'])[:, k].mean() for s in main_seeds])
                 for u in ('plus', 'minus', 'zero')}
            L.append(f"| {t:.2f} | {m['plus']:.4f} | {m['minus']:.4f} | {m['zero']:.4f} | "
                     f"{f(m['minus'] / m['plus'] if m['plus'] > 0 else None)} |")

    L += ['', '### S2, S3, S4: gain or shape, field or states, and u = 0', '',
          'S2: normalised contraction ν = c̄ / √(D · mean ‖∂f/∂h‖²_F per token), which does not change if the Jacobian is '
          'scaled; ν(−1)/ν(+1) near 1 alongside a large R means the control mostly sets the overall gain. '
          'S3: c̄ of each control value\'s field at its own states and at the other value\'s states. '
          'S4: position of c̄(0) between c̄(+1) (0) and c̄(−1) (1) on a log scale.', '',
          '| Seed | Model | ν(+1) | ν(−1) | ν ratio | c̄ field +1 at −1 states | c̄ field −1 at +1 states | c̄(0) | Position of u = 0 |',
          '|---|---|---|---|---|---|---|---|---|']
    for v in ('main', 'shuf'):
        for s in SEEDS:
            r = con[(v, s)]
            if not r:
                continue
            D = r['d_model']
            nu = {u: cbar(r, u) / math.sqrt(D * cbar(r, u, 'frob2')) for u in ('plus', 'minus')}
            cp, cm, c0 = cbar(r, 'plus'), cbar(r, 'minus'), cbar(r, 'zero')
            pos = (math.log(c0) - math.log(cp)) / (math.log(cm) - math.log(cp)) if min(cp, cm, c0) > 0 and cm != cp else None
            L.append(f"| {s} | {v} | {nu['plus']:+.4f} | {nu['minus']:+.4f} | {f(nu['minus'] / nu['plus'] if nu['plus'] else None)} | "
                     f"{cbar(r, 'plus', 'c_at_other_states'):.4f} | {cbar(r, 'minus', 'c_at_other_states'):.4f} | "
                     f"{c0:.4f} | {f(pos, 2, True)} |")

    L += ['', '### S5: G1\'s measures on the new seeds and windows', '',
          'Curvature ratio (G1\'s primary under amendment A1: refuted below 1.25), the curvature-profile classifier, and speed.', '',
          '| Seed | Model | Curvature log ratio | Curvature ratio | Classifier accuracy | Speed +1 | Speed −1 |',
          '|---|---|---|---|---|---|---|']
    for v in ('main', 'shuf'):
        for s in SEEDS:
            gr = geo[(v, s)]
            if not gr:
                continue
            gc = gr['controls']
            lr = log_ratio(gc['plus']['logk_mean'], gc['minus']['logk_mean'])
            L.append(f"| {s} | {v} | {lr:+.3f} | {math.exp(abs(lr)):.3f} | {f(classifier_accuracy(gc))} | "
                     f"{np.mean(gc['plus']['speed_mean']):.3f} | {np.mean(gc['minus']['speed_mean']):.3f} |")

    L += ['', '### S6: validation losses on the whole split (nats), and α', '',
          '| Seed | Model | next, u=+1 | next, u=−1 | skip, u=−1 | skip, u=+1 | final α | u matched task |',
          '|---|---|---|---|---|---|---|---|']
    for v in ('main', 'shuf'):
        for s in SEEDS:
            r = trn[(v, s)]
            if r:
                fl = r['final_losses']
                L.append(f"| {s} | {v} | {fl['next_uplus']:.4f} | {fl['next_uminus']:.4f} | {fl['skip_uminus']:.4f} | "
                         f"{fl['skip_uplus']:.4f} | {r['alpha_final']:.3f} | {r['u_matches_task_fraction']:.3f} |")

    dv = checks.get('divergence', {})
    L += ['', '## Checks', '']
    if dv:
        L += [f"Divergence, diagonal field: worst relative error {dv['diagonal']['worst_relative_error']:.1e}. "
              f"Dense field: estimate {dv['dense']['estimate']:.4f} against {dv['dense']['truth']:.4f}, "
              f"relative error {dv['dense']['relative_error']:.2%}; Frobenius² {dv['dense']['frobenius2_estimate']:.3f} "
              f"against {dv['dense']['frobenius2_truth']:.3f}."]
    cv = checks.get('curvature', {})
    if cv:
        L += ['', 'Curvature: ' + ', '.join(f"{k} {v['worst_error']:.1e}" for k, v in cv.items() if k != 'pass') + '.']
    b = checks.get('batch', {})
    L += ['', f"Targets at i + 1 and i + 2 confirmed on the first training batch: {b.get('targets_are_i+1_and_i+2')}."]
    sm = checks.get('same_model', [])
    L += ['', 'Exact and adjoint versions match before training: ' +
          ', '.join(f"seed {x['seed']} {'yes' if x['pass'] else 'NO'}" for x in sm) + '.', '']
    OUT.write_text('\n'.join(L))
    print('\n'.join(L[:45]))


if __name__ == '__main__':
    main()
