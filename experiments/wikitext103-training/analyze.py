"""Apply the preregistered decision rules to the saved results and write RESULTS.md.

Reads only the JSON files in the results folder, which are copied unchanged from
the colab-results branch. Every number in the report comes from them, and each
section names the files it used. Missing results are reported as missing, never
filled in.

    python analyze.py [results folder]    # default: results/ next to this file

Readings of the rules that the preregistration leaves implicit, fixed in this
script before any test loss was seen:

- S1 uses the t-based 95% interval of the mean paired difference, Model A minus
  baseline, with 5 degrees of freedom for 6 seeds.
- S3 counts a scaled control as failing when any more accurate solver changes
  the loss by 1% or more, or when any loss in that control is not finite. As in
  the replication, the controls are pooled: S3 is uninformative only if no
  control fails in any seed. A dopri5 solve stopped by the step cap at the
  learned alpha makes that seed untestable (deviation log, stage 3 safeguard).
"""
import json
import math
import statistics as st
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / 'results'
OUT = HERE / 'RESULTS.md'

SEEDS = [1001, 1002, 1003, 1004, 1005, 1006]
SOLVERS = ['euler8', 'euler16', 'rk4_4', 'dopri5_1e-5']
SCALES = ['2', '4', '8', '16']
MARGIN = 0.02
S2_MEAN, S2_EACH = (0.045, 0.085), (0.035, 0.095)
S3_LIMIT = 0.01
CONVERGED = 0.01

T80 = {1: 1.376, 2: 1.061, 3: 0.978, 4: 0.941, 5: 0.920, 6: 0.906, 7: 0.896, 8: 0.889, 9: 0.883, 10: 0.879}
T975 = {1: 12.706, 2: 4.303, 3: 3.182, 4: 2.776, 5: 2.571, 6: 2.447, 7: 2.365, 8: 2.306, 9: 2.262, 10: 2.228,
        12: 2.179, 15: 2.131, 20: 2.086, 30: 2.042}


def t975(df):
    df = max(1, int(math.floor(df)))
    return next(T975[k] for k in sorted(T975, reverse=True) if df >= k)


def t_interval(xs):
    n = len(xs)
    m = sum(xs) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in xs) / (n - 1))
    half = t975(n - 1) * sd / math.sqrt(n)
    return m, sd, m - half, m + half


def sign_flip_p(diffs):
    import itertools
    obs = abs(sum(diffs) / len(diffs))
    hits = total = 0
    for signs in itertools.product((1, -1), repeat=len(diffs)):
        total += 1
        hits += abs(sum(s * d for s, d in zip(signs, diffs)) / len(diffs)) >= obs - 1e-12
    return hits / total


def welch(a, b):
    va, vb = st.variance(a), st.variance(b)
    se2 = va / len(a) + vb / len(b)
    df = se2 ** 2 / ((va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
    d = st.mean(a) - st.mean(b)
    half = t975(df) * math.sqrt(se2)
    return d - half, d + half, df


def load(name):
    p = RESULTS / f'{name}.json'
    return json.loads(p.read_text()) if p.exists() else None


def f(x, d=4):
    if x is None:
        return 'n/a'
    if isinstance(x, float) and not math.isfinite(x):
        return str(x)
    return f'{x:.{d}f}' if isinstance(x, float) else str(x)


def pct(x, d=3):
    if x is None:
        return 'n/a'
    return str(x) if not math.isfinite(x) else f'{100 * x:.{d}f}%'


def finite(x):
    return x is not None and math.isfinite(x)


def main():
    train = {(k, s): load(f'main_{k}_seed{s}') for k in ('baseline', 'hybrid') for s in SEEDS}
    test = {(k, s): load(f'test_main_{k}_seed{s}') for k in ('baseline', 'hybrid') for s in SEEDS}
    stage1 = load('stage1_summary')
    verdicts, body = [], []

    n_train = sum(v is not None for v in train.values())
    n_test = sum(v is not None for v in test.values())

    # ---------------------------------------------------------------- S1
    pairs = [s for s in SEEDS if test[('baseline', s)] and test[('hybrid', s)]]
    body += ['## S1. Test loss against the baseline, the primary outcome', '']
    if len(pairs) < 2:
        verdicts.append(('S1', 'A properly trained Model A matches or beats the baseline', 'missing',
                         f'{len(pairs)} complete pairs'))
        body += [f'Only {len(pairs)} complete pairs; S1 cannot be evaluated.', '']
    else:
        b = [test[('baseline', s)]['test_loss'] for s in pairs]
        h = [test[('hybrid', s)]['test_loss'] for s in pairs]
        diffs = [y - x for x, y in zip(b, h)]
        m, sd, lo, hi = t_interval(diffs)
        p = sign_flip_p(diffs)
        wlo, whi, wdf = welch(h, b)
        n = len(diffs)
        mde = (t975(n - 1) + T80[n - 1]) * sd / math.sqrt(n)
        if hi < 0:
            outcome = 'supported: better'
        elif lo >= -MARGIN and hi <= MARGIN:
            outcome = 'supported: comparable'
        elif lo > 0:
            outcome = 'refuted: worse'
        else:
            outcome = 'inconclusive: not enough runs to tell'
        verdicts.append(('S1', 'A properly trained Model A matches or beats the baseline', outcome,
                         f'mean {m:+.4f}, 95% [{lo:+.4f}, {hi:+.4f}], p = {p:.4f}, n = {n}'))
        body += ['| Seed | Baseline | Model A | Difference |', '|---|---|---|---|']
        body += [f'| {s} | {x:.4f} | {y:.4f} | {y - x:+.4f} |' for s, x, y in zip(pairs, b, h)]
        body += ['', f'Paired difference, Model A minus baseline: mean {m:+.4f} nats, standard deviation {sd:.4f}, '
                 f't-based 95% interval [{lo:+.4f}, {hi:+.4f}], exact sign-flip p = {p:.4f}. '
                 f'Model A lower in {sum(d < 0 for d in diffs)} of {n} seeds.', '',
                 f'Unpaired Welch interval, reported alongside: [{wlo:+.4f}, {whi:+.4f}], about {wdf:.1f} degrees of freedom.', '',
                 f'Smallest effect detectable with about 80% power at this spread: {mde:.4f} nats. The margin for "comparable" is ±{MARGIN} nats.', '',
                 'Source: `test_main_*_seed*.json`, key `test_loss`.', '']

    # ---------------------------------------------------------------- S2
    body += ['## S2. Final α', '']
    alphas = [(s, train[('hybrid', s)]['alpha_final']) for s in SEEDS if train[('hybrid', s)]]
    if not alphas:
        verdicts.append(('S2', 'α settles in a conservative range', 'missing', 'no Model A runs'))
    else:
        vals = [a for _, a in alphas]
        mean_a = st.mean(vals)
        ok = S2_MEAN[0] <= mean_a <= S2_MEAN[1] and all(S2_EACH[0] <= a <= S2_EACH[1] for a in vals)
        verdicts.append(('S2', 'α settles in a conservative range', 'supported' if ok else 'refuted',
                         f'mean {mean_a:.4f}, range {min(vals):.4f} to {max(vals):.4f}; '
                         f'predicted mean in [{S2_MEAN[0]}, {S2_MEAN[1]}], each in [{S2_EACH[0]}, {S2_EACH[1]}]'))
        body += ['| Seed | Final α | α at step 500 | α at 25% | α at 50% | α at 75% | Largest during training |',
                 '|---|---|---|---|---|---|---|']
        for s, a in alphas:
            path = train[('hybrid', s)]['alpha_every_50']
            total = train[('hybrid', s)]['total_steps']

            def at(frac):
                target = frac * total
                return min(path, key=lambda r: abs(r[0] - target))[1] if path else None
            a500 = min(path, key=lambda r: abs(r[0] - 500))[1] if path else None
            body.append(f'| {s} | {a:.4f} | {f(a500)} | {f(at(0.25))} | {f(at(0.5))} | {f(at(0.75))} | '
                        f'{f(max([v for _, v in path] + [a]))} |')
        body += ['', f'Mean final α {mean_a:.4f}. α starts at 0.1.', '',
                 'Source: `main_hybrid_seed*.json`, keys `alpha_final` and `alpha_every_50`.', '']

    # ---------------------------------------------------------------- S3
    body += ['## S3. Solver dependence', '']
    hyb = [s for s in SEEDS if test[('hybrid', s)]]
    if not hyb:
        verdicts.append(('S3', 'The trained Model A behaves as a continuous ODE', 'missing', 'no Model A test records'))
    else:
        failed_seeds, untestable, control_fail_any = [], [], False
        rows = ['| Seed | Model | ' + ' | '.join(SOLVERS) + ' |', '|---|---|' + '---|' * len(SOLVERS)]
        errors = []
        for s in hyb:
            r = test[('hybrid', s)]

            def cells(tab):
                out = []
                for name in SOLVERS:
                    c = tab.get(name, {})
                    out.append('error' if c.get('loss') is None else pct(c.get('rel_change_vs_euler4')))
                return out
            t = r['solver_test']
            rows.append(f'| {s} | trained, α {r["alpha"]:.3f} | ' + ' | '.join(cells(t)) + ' |')
            for name in SOLVERS:
                c = t.get(name, {})
                if c.get('loss') is None:
                    errors.append(f'seed {s}, learned α, {name}: {c.get("error")}')
                    if name == 'dopri5_1e-5':
                        untestable.append(s)
                elif not finite(c.get('rel_change_vs_euler4')) or c['rel_change_vs_euler4'] >= S3_LIMIT:
                    failed_seeds.append(s)
            for sc in SCALES:
                tab = r['solver_controls'].get(sc, {})
                rows.append(f'| {s} | α × {sc} | ' + ' | '.join(cells(tab)) + ' |')
                for name, c in tab.items():
                    if c.get('loss') is None:
                        errors.append(f'seed {s}, α × {sc}, {name}: {c.get("error")}')
                        continue
                    if not finite(c['loss']) or (name != 'euler4' and (not finite(c.get('rel_change_vs_euler4'))
                                                                        or c['rel_change_vs_euler4'] >= S3_LIMIT)):
                        control_fail_any = True
        failed_seeds = sorted(set(failed_seeds))
        untestable = sorted(set(untestable) - set(failed_seeds))
        if failed_seeds:
            outcome = 'refuted'
        elif untestable:
            outcome = 'untestable in some seeds'
        elif not control_fail_any:
            outcome = 'uninformative'
        else:
            outcome = 'supported'
        verdicts.append(('S3', 'The trained Model A behaves as a continuous ODE', outcome,
                         f'seeds with a change of 1% or more: {failed_seeds or "none"}; '
                         f'untestable seeds: {untestable or "none"}; some scaled control failed: {control_fail_any}'))
        body += ['Relative change in test loss against the training solver, 4 Euler steps. '
                 'The trained model uses the whole test split; the scaled controls use the first 512 test windows.', '']
        body += rows
        body += ['', 'Errors recorded: ' + ('none.' if not errors else ''), '']
        body += [f'- {e}' for e in errors]
        mm = [(s, test[('hybrid', s)].get('must_match_max_diff')) for s in hyb]
        body += ['', 'Must-match check, solver-switching code against the model\'s own forward pass: largest logit difference '
                 + ', '.join(f'{s}: {f(d, 7) if isinstance(d, float) else d}' for s, d in mm) + '.', '',
                 'Source: `test_main_hybrid_seed*.json`, keys `solver_test`, `solver_controls` and `must_match_max_diff`.', '']

    # ---------------------------------------------------------------- exploratory
    body += ['## Exploratory measures', '', '### Validation loss and convergence', '',
             '| Seed | Baseline validation | Model A validation | Baseline gain, last 10% | Model A gain, last 10% |',
             '|---|---|---|---|---|']
    for s in SEEDS:
        bt, ht = train[('baseline', s)], train[('hybrid', s)]
        body.append(f'| {s} | {f(bt and bt["final_val_loss"])} | {f(ht and ht["final_val_loss"])} | '
                    f'{f(bt and bt["improvement_last_10pct"])} | {f(ht and ht["improvement_last_10pct"])} |')
    gains = [r['improvement_last_10pct'] for r in train.values() if r and r['improvement_last_10pct'] is not None]
    conv = sum(g < CONVERGED for g in gains)
    body += ['', f'A run counts as converged if validation loss improved by less than {CONVERGED} nats over its last 10% of steps. '
             f'Converged: {conv} of {len(gains)} runs.', '', '### Training cost', '']
    bs = [train[('baseline', s)]['train_seconds'] for s in SEEDS if train[('baseline', s)]]
    hs = [train[('hybrid', s)]['train_seconds'] for s in SEEDS if train[('hybrid', s)]]
    if bs and hs:
        body += [f'Mean training time per run on the T4: baseline {st.mean(bs) / 3600:.2f} h, Model A {st.mean(hs) / 3600:.2f} h, '
                 f'ratio {st.mean(hs) / st.mean(bs):.2f}. This is training time; the paper\'s latency claim concerned inference.', '']
    body += ['### Early training, for comparison with the WikiText-2 replication', '',
             '| Seed | Baseline validation at step 500 | Model A validation at step 500 |', '|---|---|---|']
    for s in SEEDS:
        bt, ht = train[('baseline', s)], train[('hybrid', s)]
        body.append(f'| {s} | {f(bt and bt["val_at_500"])} | {f(ht and ht["val_at_500"])} |')
    body += ['', 'Source: `main_*_seed*.json`, keys `final_val_loss`, `improvement_last_10pct`, `train_seconds` and `val_at_500`.', '']

    # ---------------------------------------------------------------- provenance
    body += ['## Provenance', '', '| Run | Peak learning rate | Steps | Resumes | fp16 skipped steps | GPU | Code at end of training | Code at test |',
             '|---|---|---|---|---|---|---|---|']
    for k in ('baseline', 'hybrid'):
        for s in SEEDS:
            r, t = train[(k, s)], test[(k, s)]
            if not r:
                body.append(f'| main_{k}_seed{s} | missing | | | | | | |')
                continue
            env = r.get('environment', {})
            body.append(f'| main_{k}_seed{s} | {r["peak_lr"]} | {r["total_steps"]} | {r["resumes"]} | {r["skipped_steps_fp16"]} | '
                        f'{env.get("gpu")} | `{str(env.get("code_commit"))[:7]}` | '
                        f'{"`" + str(t["environment"].get("code_commit"))[:7] + "`" if t else "missing"} |')
    datas = {json.dumps(r['data'], sort_keys=True) for r in train.values() if r}
    body += ['', f'Data fingerprints identical across all runs: {len(datas) == 1}.']
    if stage1:
        ch = stage1.get('choice', {})
        body += ['', 'Learning rates chosen in stage 1: ' + ', '.join(f'{k} {v.get("chosen_lr")}' for k, v in ch.items()) + '.']
    body.append('')

    head = ['# Results: WikiText-103 study', '',
            'Generated by `analyze.py` from the files in `results/`. Every number below comes from those files.', '',
            f'Training records: {n_train} of 12. Test records: {n_test} of 12.', '',
            '## Verdicts', '', '| Hypothesis | Claim | Outcome | Basis |', '|---|---|---|---|']
    head += [f'| {a} | {b} | {c} | {d} |' for a, b, c, d in verdicts]
    head += ['', 'Each hypothesis is reported separately, so no multiple-comparison correction is applied.', '']
    OUT.write_text('\n'.join(head + body))
    print(OUT.read_text())


if __name__ == '__main__':
    main()
