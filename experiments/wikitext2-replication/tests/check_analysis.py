"""Must-fail check for analyze.py: feed it synthetic results whose verdicts are known.

Builds two complete fake result sets from the test-mode files, one where every
prediction holds and one where every prediction fails, and checks that analyze.py
reaches the expected verdict for each hypothesis. Run with REPL_TEST=1 after the
test-mode pipeline has produced results_test/.
"""
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
SRC = HERE / 'results_test'
SEEDS = [42, 0, 1, 2, 3, 4, 5, 6]


def tmpl(name):
    return json.loads((SRC / f'{name}.json').read_text())


def put(d, name, result):
    rec = tmpl(name.split('_seed')[0].replace('train_baseline', 'train_baseline').split('.')[0] + ('_seed42' if '_seed' in name else ''))
    rec['result'] = result
    (d / f'{name}.json').write_text(json.dumps(rec))


def scenario(d, good):
    for i, s in enumerate(SEEDS):
        for kind in ('baseline', 'hybrid'):
            r = copy.deepcopy(tmpl(f'train_{kind}_seed42')['result'])
            r['seed'] = s
            base = 6.30 + 0.01 * i
            r['val_loss'] = base if kind == 'baseline' else base + (-0.03 if good else 0.08) + 0.002 * (i % 3)
            if s == 42:
                r['step0_loss'] = (11.0107 if kind == 'baseline' else 10.98) + (0.01 if good else 0.5)
                r['last50_train_loss'] = (6.4707 if kind == 'baseline' else 6.4496) + (0.02 if good else 0.5)
            if kind == 'hybrid':
                r['alpha_final'] = 0.065 + 0.002 * (i % 3) if good else 0.12
                n = len(r['log']['loss'])
                r['log']['alpha'] = [r['alpha_final']] * n
                r['log']['ode_grad_norm'] = [0.03] * n if good else [0.03] * (n - 1) + [0.0]
                r['log']['ode_grad_norm_preclip'] = r['log']['ode_grad_norm']
            put(d, f'train_{kind}_seed{s}', r)
        sv = copy.deepcopy(tmpl('solver_seed42')['result'])
        for name, r in sv['trained']['losses'].items():
            r['rel_change_vs_euler4'] = 0.0 if name == 'euler4' else (0.001 if good else 0.05)
        for k, blk in sv['scaled'].items():
            for name, r in blk['losses'].items():
                r['rel_change_vs_euler4'] = 0.0 if name == 'euler4' else (0.2 if int(k) >= 8 else 0.002)
        put(d, f'solver_seed{s}', sv)
        for tok in (922, 220):
            st = copy.deepcopy(tmpl(f'steer{tok}_seed42')['result'])
            st['train_prompt_u+1'] = {'p_pos': 0.9 if good else 0.1, 'p_neg': 0.05 if good else 0.5}
            st['train_prompt_u-1'] = {'p_pos': 0.05, 'p_neg': 0.85}
            st['train_prompt_u0'] = {'p_pos': 0.8 if good else 0.1, 'p_neg': 0.1 if good else 0.8}
            st['held_out_correct'] = 34 if good else 21
            put(d, f'steer{tok}_seed{s}', st)
            nf = copy.deepcopy(tmpl(f'nfe{tok}_seed42')['result'])
            nf['mean_nfe_u_below_0.3'], nf['mean_nfe_u_above_0.4'] = (20.0, 14.0) if good else (14.0, 14.0)
            put(d, f'nfe{tok}_seed{s}', nf)
        for m in ('steer922', 'untrained'):
            pr = copy.deepcopy(tmpl(f'probe_{m}_seed42')['result'])
            for j, row in enumerate(pr['rows']):
                row['selectivity'] = 0.0 if j == 0 else (0.30 if good else 0.30 + 0.05 * j)
            put(d, f'probe_{m}_seed{s}', pr)
    ctl = copy.deepcopy(tmpl('h1_control')['result'])
    ctl['post_clip_flags'] = {'vanishing': 0, 'exploding': 3}
    put(d, 'h1_control', ctl)
    lat = copy.deepcopy(tmpl('latency')['result'])
    for r in lat['runs']:
        r['ratio'] = 0.99 if good else 1.2
    put(d, 'latency', lat)


EXPECT = {True: {'R0': 'reproduced', 'H1': 'supported', 'H2': 'supported: slightly better', 'H3': 'supported',
                 'H4': 'supported', 'H5a': 'supported', 'H5b': 'supported', 'H5c': 'supported', 'H5d': 'passed',
                 'H6': 'supported', 'H7': 'supported', 'H8': 'supported'},
          False: {'R0': 'not reproduced', 'H1': 'refuted', 'H2': 'refuted: worse', 'H3': 'refuted', 'H4': 'refuted',
                  'H5a': 'refuted', 'H5b': 'refuted', 'H5c': 'refuted', 'H5d': 'passed', 'H6': 'refuted',
                  'H7': 'refuted', 'H8': 'refuted'}}

failures = 0
for good in (True, False):
    with tempfile.TemporaryDirectory() as tmp:
        d = Path(tmp) / 'results'
        d.mkdir()
        scenario(d, good)
        out = subprocess.run([sys.executable, '-W', 'ignore', str(HERE / 'analyze.py')], capture_output=True, text=True,
                             env={**os.environ, 'REPL_RESULTS_DIR': str(d)}, cwd=HERE)
        if out.returncode:
            print(out.stderr); sys.exit(1)
        got = {line.split(' | ')[0]: line.split(' | ')[2] for line in out.stdout.strip().splitlines()}
        for h, want in EXPECT[good].items():
            ok = got.get(h, '').startswith(want)
            failures += not ok
            print(f"{'all-hold' if good else 'all-fail'} {h:4s} expected '{want}' got '{got.get(h)}' {'OK' if ok else 'MISMATCH'}")
print('ALL MATCH' if failures == 0 else f'{failures} MISMATCHES')
sys.exit(1 if failures else 0)
