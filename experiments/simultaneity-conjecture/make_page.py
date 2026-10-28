"""Build the interactive page, Voices and Directions, from the saved records. It shows the same findings as the static figures,
with every value on hover and a table under each figure.

Reads the records through make_figures.py's readers, plus the full-budget WikiText-103 records and T1's intervals,
which the static figures do not draw. Fills page/template.html with the data and writes ../../docs/index.html, which
GitHub Pages serves. The page's prose was checked against the notes by hand; only its data is generated here.

    python make_page.py
"""
import json
import math
import re
import statistics as st
from pathlib import Path

import numpy as np
from scipy import stats

import make_figures as mf

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / 'page' / 'template.html'
DOCS = HERE.parent.parent / 'docs'
WT103 = HERE.parent / 'wikitext103-training' / 'results'
FAMILY_ALPHA, N_CONTRASTS = 0.05, 4        # T1's four confirmatory contrasts, Bonferroni-corrected (t1_analyze.py)
JUDGED_HEADS = (('rw-cloop', 'N18: judge weights itself, clean targets'), ('rw-ctwin', 'N18: judge weighted by a loop-free twin'),
                ('rw-j20', 'N18: judge at full weight, clean targets'), ('rw-peer', 'N17/N18: the peer, judges itself on scrambled targets'))
# N18a's claim, as its preregistration states it, for the one row the ledger does not yet carry
N18A_GUESS = 'With the data and the weighting fixed, the judge\'s own loop is enough to produce most of the peer\'s gap'


def load(path):
    return json.loads(Path(path).read_text())


def interval(xs, level):
    m, sd = st.mean(xs), st.stdev(xs)
    h = stats.t.ppf(1 - (1 - level) / 2, len(xs) - 1) * sd / math.sqrt(len(xs))
    return [m - h, m + h]


def shortcut():
    """T1's contrasts (means and intervals, as t1_analyze.py computes them) and the full-budget WikiText-103 gap."""
    sc = mf.shortcut_data()
    level = 1 - FAMILY_ALPHA / N_CONTRASTS
    seeds = sorted(int(p.stem.rsplit('seed', 1)[1]) for p in WT103.glob('test_main_hybrid_seed*.json'))
    full = [load(WT103 / f'test_main_hybrid_seed{s}.json')['test_loss'] - load(WT103 / f'test_main_baseline_seed{s}.json')['test_loss'] for s in seeds]
    keys = {'gap': 'whole', 'training_method': 'G', 'ode_form': 'F', 'repeated_block': 'C2', 'separate_weights': 'P'}
    out = {k: st.mean(sc[c]) for k, c in keys.items()}
    out['ci'] = {k: interval(sc[c], 0.95 if c == 'whole' else level) for k, c in keys.items()}
    out['exact_trained_behind_baseline'] = st.mean(f + c for f, c in zip(sc['F'], sc['C2']))     # AX − B6
    out['full_budget_gap_as_built'] = st.mean(full)
    return out


def flow():
    """figure 5's ratios, with the must-fail controls' ranges scored the same way (G1's shuffled control, N2's same-target model)."""
    fl = mf.flow_data()
    shuf = []
    for s in mf.SEEDS_G1:
        u = mf.load('g1', f'geom_shuf_seed{s}')['controls']
        shuf.append(math.exp(abs(float(np.mean(np.asarray(u['plus']['logk_mean'], float) - np.asarray(u['minus']['logk_mean'], float))))))
    same = []
    for s in mf.SEEDS_N2:
        c = mf.load('n2', f'sub_same_seed{s}')['contraction']
        same.append(float(np.mean(c['B16'])) / float(np.mean(c['A16'])))
    return {'curvature_G1': fl['G1'], 'curvature_shuffled_range': [min(shuf), max(shuf)], 'pace_N2': fl['N2'],
            'pace_same_target_range': [min(same), max(same)], 'contraction_G1c': fl['G1c'], 'contraction_removed_cost_G1d': fl['G1d'],
            'predicted': 2.0, 'refute_below': 1.25}


def judged_heads():
    """Head B's (Δ_A, agreement) by run for N18's judged heads, as figure 2 places its heads."""
    def rec(a, s):
        p = mf.RES / 'n18' / f'm_{a}_seed{s}.json'
        return load(p if p.exists() else mf.RES / 'n18' / f'r_{a}_seed{s}.json')
    return {label: [(mf.dA(rec(a, s)), mf.agree(rec(a, s))) for s in mf.SEEDS12] for a, label in JUDGED_HEADS}


def ledger():
    """The ledger's main rows with each claim and verdict, in its order; N18a from its note while the ledger lacks it."""
    section = (HERE / 'CRITICAL_REVIEW.md').read_text().split('## The ledger', 1)[1].split('\n## ', 1)[0]
    rows = []
    for line in section.splitlines():
        cells = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(cells) < 4 or not re.match(r'(WikiText|T\d|G\d|N\d)', cells[0]):
            continue
        name = {'WikiText-2 replication': 'WT-2', 'WikiText-103': 'WT-103'}.get(cells[0], cells[0])
        verdict = re.sub(r'\[\^\d+\]', '', cells[-1]).replace('**', '').strip()
        bold = re.search(r'\*\*(.+?)\*\*', cells[-1])
        for t in name.split(', '):
            rows.append({'test': t, 'guess': cells[1], 'verdict': verdict, 'kind': mf.kind_of(bold.group(1) if bold else cells[-1])})
    if not any(r['test'] == 'N18a' for r in rows):
        word = re.search(r'\*\*N18a: (\w+)', (HERE / 'N18_NOTE.md').read_text()).group(1)
        rows.append({'test': 'N18a', 'guess': N18A_GUESS, 'verdict': word.capitalize(), 'kind': mf.kind_of(word)})
    return rows


def data():
    ladder, noisy, rand, rotated = mf.distance_data()
    acc = mf.accuracy_data()
    judges = mf.judge_data()
    rows = ledger()
    return {
        'flow': flow(),
        'shortcut': shortcut(),
        'ladder_overlap': {'same word': ladder[1], '2 words ahead': ladder[2], '4 words ahead': ladder[4], '8 words ahead': ladder[8],
                           'next word, 80% scrambled': noisy, 'mismatched control': rotated, 'random directions': [rand]},
        'accuracy_ladder': {mf.ACC_NAMES[a]: pts for a, pts in acc.items()},
        'accuracy_ladder_judged_heads': judged_heads(),
        'judges': judges,
        'n18_steps': {s: {k: judges[s][a]['dA'] for k, a in (('j20', 'rw-j20'), ('ctwin', 'rw-ctwin'), ('cloop', 'rw-cloop'), ('peer', 'rw-peer'))}
                      for s in mf.SEEDS12},
        'ledger': rows,
        'ledger_counts': {k: sum(r['kind'] == k for r in rows) for k in mf.KIND},
    }


def main():
    page = TEMPLATE.read_text()
    assert page.count('/*DATA*/') == 1, 'the template must hold one /*DATA*/ placeholder'
    DOCS.mkdir(exist_ok=True)
    (DOCS / 'index.html').write_text(page.replace('/*DATA*/', json.dumps(data(), separators=(',', ':'))))
    (DOCS / '.nojekyll').write_text('')      # serve the page as it is, without GitHub Pages' Jekyll step
    print('wrote', DOCS / 'index.html')


if __name__ == '__main__':
    main()
