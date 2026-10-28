"""Draw the figures from the saved records. Every measured value comes from a JSON record in results/ (T1's from
../too-linear-conjecture/results/t1/); the ledger comes from CRITICAL_REVIEW.md's ledger, with N18a's verdict from
N18_NOTE.md until the ledger carries it; figure 10 is imagined and plots no measurement.

Writes, in figures/, each figure in a light and a dark version (PNG, at twice display size) and FIGURE_DATA.md, the
table of every plotted value by run. Figures 1 to 3 appear in the plain summary.

  1. The distance ladder: how much the second reading shares directions with the next-word reading, by how far ahead it
     reads (N8's target-free overlap on N6r's models), with the next-word reading trained on 80% scrambled answers
     (N8 on N7's models) and the level of random directions.
  2. The accuracy ladder: agreement with the clean head against how much worse than the clean head a head predicts the
     genuine answers, for heads given fewer genuine answers (the ladder), heads trained with scrambled answers (N14's
     doses, N12's blended head, N16's clean-judged head), and discounted heads given no scrambled answers (N14, N16).
  3. N16's two-by-two: who judges the answers, and whether the scrambled answers are present.
  4. T1: where the paper's model's gap to the baseline came from.
  5. The flow: the ratio between the two readings for G1's curvature, N2's per-reading contraction and G1c's whole-state
     contraction.
  6. The angle fan: figure 1's overlaps drawn as the angle whose squared cosine equals each overlap.
  7. N18's judges: the share of the shortfall each judge's head recovers against the judge's own loss gap, with the
     loop-free line and each judge's distance below it.
  8. N18's steps: head B's gap from the judge at full weight to the peer, one change at a time.
  9. The ledger: every preregistered verdict, by chapter.
 10. Imagined: the findings as a jigsaw around an empty centre, with four imagined connections.

    python make_figures.py
"""
import json
import math
import re
import statistics as st
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
T1RES = HERE.parent / 'too-linear-conjecture' / 'results' / 't1'
OUT = HERE / 'figures'
SEEDS6R = [3701, 3702, 3703, 3704, 3705]
SEEDS7 = [3801, 3802, 3803, 3804, 3805]
SEEDS12 = [4101, 4102, 4103, 4104, 4105]
SEEDS_T1 = [2001, 2002, 2003, 2004, 2005]
SEEDS_G1, SEEDS_G1C, SEEDS_G1D, SEEDS_N2 = [3001, 3002, 3003, 3004, 3005], [3101, 3102, 3103, 3104, 3105], [3201, 3202, 3203, 3204, 3205], [3401, 3402, 3403, 3404, 3405]

# Three colours chosen to stay distinct for colour-blind readers, each stepped for a light and a dark background, and grey chrome.
# The status colours mark verdicts only, and always with a glyph; violet marks imagined material only.
THEMES = {
    'light': dict(faint=0.35, surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e', muted='#898781', grid='#e1e0d9', axis='#c3c2b7',
                  s1='#2a78d6', s2='#eb6834', s3='#1baf7a', imagined='#6250d6',
                  good='#0ca30c', warning='#fab219', serious='#ec835a', critical='#d03b3b', none='#9aa0ab'),
    'dark': dict(faint=0.6, surface='#1a1a19', ink='#f0efec', ink2='#c3c2b7', muted='#898781', grid='#2c2c2a', axis='#383835',
                 s1='#3987e5', s2='#d95926', s3='#199e70', imagined='#9085e9',
                 good='#0ca30c', warning='#fab219', serious='#ec835a', critical='#d03b3b', none='#6f7581'),
}
PX = 0.72            # one display pixel in points, at 100 dpi
DPI_DISPLAY, SCALE = 100, 2


def load(folder, name):
    return json.loads((RES / folder / f'{name}.json').read_text())


def dA(rec):
    f = rec['functional']
    return f['B_inner_clean'] - f['A_clean']


def agree(rec):
    return (rec.get('B_inner_vs_A') or rec['B_vs_A'])['agreement_ratio']


def sgn(v, k=3):
    """v with its sign, and a true minus sign."""
    return f'{v:+.{k}f}'.replace('-', '−')


def style(ax, c):
    ax.set_facecolor(c['surface'])
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)
    for side in ('left', 'bottom'):
        ax.spines[side].set_color(c['axis']); ax.spines[side].set_linewidth(1 * PX)
    ax.tick_params(colors=c['muted'], labelcolor=c['ink2'], width=1 * PX, labelsize=9.5)
    ax.grid(axis='y', color=c['grid'], linewidth=1 * PX)
    ax.set_axisbelow(True)


def figure(c, w=7.2, h=4.6):
    fig, ax = plt.subplots(figsize=(w, h), dpi=DPI_DISPLAY)
    fig.patch.set_facecolor(c['surface'])
    style(ax, c)
    return fig, ax


def titles(fig, c, title, subtitle):
    fig.text(0.075, 0.955, title, color=c['ink'], fontsize=11.5, fontweight='bold', va='top')
    fig.text(0.075, 0.905, subtitle, color=c['ink2'], fontsize=9.5, va='top', linespacing=1.4)


def save(fig, name, mode, tight=False):
    """tight: trim the empty canvas around the drawing (for a figure whose shape follows its content)."""
    extra = dict(bbox_inches='tight', pad_inches=0.25) if tight else {}
    fig.savefig(OUT / f'{name}_{mode}.png', dpi=DPI_DISPLAY * SCALE, facecolor=fig.get_facecolor(), **extra)
    plt.close(fig)


# ------------------------------------------------------------------ data

def distance_data():
    ladder = {}
    for arm, k in (('same', 1), ('k2', 2), ('k4', 4), ('k8', 8)):
        ladder[k] = [load('n8', f'f_n6r_{arm}_seed{s}')['fisher']['overlap_AB'] for s in SEEDS6R]
    noisy = [load('n8', f'f_n7_noisy_seed{s}')['fisher']['overlap_AB'] for s in SEEDS7]
    rank = load('n8', f'f_n6r_k2_seed{SEEDS6R[0]}')['fisher']['rank']
    rotated = [load('n8', f'f_n6r_same_seed{s}')['rotated_control']['overlap_AB'] for s in SEEDS6R]
    return ladder, noisy, rank / 256, rotated


RUNGS = ('clean', 'sub20', 'sub10', 'sub05', 'sub025', 'sub0125')
RUNG_LABEL = {'clean': 'all', 'sub20': '20%', 'sub10': '10%', 'sub05': '5%', 'sub025': '2.5%', 'sub0125': '1.25%'}
SCRAMBLED = {'r10': ('n14', 'm_r10'), 'r20': ('n14', 'm_r20'), 'r40': ('n14', 'm_r40'), 'n80-mix': ('n14', 'r_n80-mix'),
             'rw-oracle': ('n16', 'm_rw-oracle')}
NOSCRAMBLE = {'sub20-rw': ('n16', 'r_sub20-rw'), 'sub20-oracle': ('n16', 'm_sub20-oracle')}
ACC_NAMES = {'clean': 'every genuine answer', 'sub20': '20% genuine answers', 'sub10': '10%', 'sub05': '5%', 'sub025': '2.5%', 'sub0125': '1.25%',
             'r10': '10% scrambled + 20% genuine (N14)', 'r20': '20% scrambled + 20% genuine (N14)', 'r40': '40% scrambled + 20% genuine (N14)',
             'n80-mix': 'blended head, 80% scrambled (N12)', 'rw-oracle': 'discounted, clean head judges, 80% scrambled (N16)',
             'sub20-rw': 'discounted, judges itself, no scrambled (N14)', 'sub20-oracle': 'discounted, clean head judges, no scrambled (N16)'}


def accuracy_data():
    pts = {}
    for a in RUNGS:
        pts[a] = [(dA(r), agree(r)) for r in (load('n16', f'r_{a}_seed{s}') for s in SEEDS12)]
    for a, (folder, pre) in {**SCRAMBLED, **NOSCRAMBLE}.items():
        pts[a] = [(dA(r), agree(r)) for r in (load(folder, f'{pre}_seed{s}') for s in SEEDS12)]
    return pts


CELLS = {('self', 'genuine'): 'sub20-rw', ('self', 'scrambled'): 'rw-b1', ('clean', 'genuine'): 'sub20-oracle', ('clean', 'scrambled'): 'rw-oracle'}


def twobytwo_data():
    out = {}
    for key, a in CELLS.items():
        pre = 'm' if a in ('rw-oracle', 'sub20-oracle') else 'r'
        out[a] = [dA(load('n16', f'{pre}_{a}_seed{s}')) for s in SEEDS12]
    out['sub20'] = [dA(load('n16', f'r_sub20_seed{s}')) for s in SEEDS12]
    return out


# T1's contrasts, as t1_analyze.py takes them: G + F + C2 is the chain from the paper's model (AA) to the baseline (B6); P stands beside it
SHORTCUT = {'whole': ('AA', 'B6'), 'G': ('AA', 'AX'), 'F': ('AX', 'T4'), 'C2': ('T4', 'B6'), 'P': ('B5', 'B6')}


def shortcut_data():
    val = {(a, s): json.loads((T1RES / f'main_{a}_seed{s}.json').read_text())['final_val_loss']
           for a in ('B6', 'B5', 'T4', 'AX', 'AA') for s in SEEDS_T1}
    return {k: [val[(a, s)] - val[(b, s)] for s in SEEDS_T1] for k, (a, b) in SHORTCUT.items()}


def flow_data():
    """Each model's ratio between the two readings, scored as its analysis script scores it, and G1d's D by seed."""
    def g1(s):   # the larger window-mean curvature over the smaller (g1_analyze.py)
        u = load('g1', f'geom_main_seed{s}')['controls']
        return math.exp(abs(float(np.mean(np.asarray(u['plus']['logk_mean'], float) - np.asarray(u['minus']['logk_mean'], float)))))

    def n2(s):   # contraction per dimension within the farther reading's 16 directions over the nearer's (n2_analyze.py)
        c = load('n2', f'sub_dual_seed{s}')['contraction']
        return float(np.mean(c['B16'])) / float(np.mean(c['A16']))

    def g1c(s):  # whole-state contraction under u = -1 over u = +1 (g1c_analyze.py)
        u = load('g1c', f'contr_main_seed{s}')['controls']
        return float(np.mean(u['minus']['c'])) / float(np.mean(u['plus']['c']))

    def g1d(s):  # the farther task's cost of the cap minus the nearer task's (g1d_analyze.py)
        loss = lambda a, k: load('g1d', f'train_{a}_seed{s}')['final_losses'][k]
        return (loss('capped', 'skip_uminus') - loss('control', 'skip_uminus')) - (loss('capped', 'next_uplus') - loss('control', 'next_uplus'))
    return {'G1': [g1(s) for s in SEEDS_G1], 'N2': [n2(s) for s in SEEDS_N2], 'G1c': [g1c(s) for s in SEEDS_G1C], 'G1d': [g1d(s) for s in SEEDS_G1D]}


LINE = (('rw-j20', '20%'), ('n17-rw-j10', '10%'), ('n17-rw-j05', '5%'), ('rw-j025', '2.5%'), ('rw-j0125', '1.25%'))   # full weight, genuine answers only
JUDGED = ('rw-oracle', 'rw-ctwin', 'rw-peer', 'rw-b1', 'rw-cloop')


def judge_data():
    """N18's judges by seed, as n18_analyze.py takes them: x the judge's loss gap (head A judging: 0; head B judging
    itself: its own Δ_A), F = (Δ_A(rw-b1) − Δ_A(arm)) / (Δ_A(rw-b1) − Δ_A(sub20)), and the judged head's Δ_A."""
    def rec(a, s):
        p = RES / 'n18' / f'm_{a}_seed{s}.json'
        return json.loads((p if p.exists() else RES / 'n18' / f'r_{a}_seed{s}.json').read_text())
    out = {}
    for s in SEEDS12:
        b1, G = dA(rec('rw-b1', s)), dA(rec('rw-b1', s)) - dA(rec('sub20', s))
        out[s] = {}
        for a in [a for a, _ in LINE] + list(JUDGED):
            r = rec(a, s)
            x = 0.0 if a == 'rw-oracle' else b1 if a == 'rw-b1' else r['judge_measure']['x']
            out[s][a] = {'x': x, 'F': (b1 - dA(r)) / G, 'dA': dA(r)}
    return out


def d_within(J, a):
    """N18's d (S8): within each run, F minus the loop-free line through that run's rungs at the judge's x; the mean over runs."""
    return st.mean(J[s][a]['F'] - line_at(sorted((J[s][r]['x'], J[s][r]['F']) for r, _ in LINE), J[s][a]['x']) for s in SEEDS12)


def line_at(pts, xv):
    """The loop-free line through the rung means, read at x (None outside it)."""
    for (x0, y0), (x1, y1) in zip(pts, pts[1:]):
        if x0 <= xv <= x1:
            return y0 + (y1 - y0) * (xv - x0) / (x1 - x0)
    return None


KIND = {'supported': ('good', '✓', 'Supported'), 'refuted': ('critical', '✕', 'Refuted'), 'unclear': ('warning', '~', 'Intermediate or unclear size'),
        'inconclusive': ('serious', '?', 'Inconclusive'), 'untestable': ('none', '∅', 'Untestable')}
CHAPTERS = ('The paper', 'Ch. 1: the flow', 'Ch. 2: directions', 'Ch. 3: different or harder', 'Ch. 4: scrambled answers, the judge')


VERDICT_WORDS = (('untestable', 'untestable'), ('refuted', 'refuted'), ('supported', 'supported'), ('inconclusive', 'inconclusive'),
                 ('intermediate', 'unclear'), ('size unclear', 'unclear'))


def kind_of(verdict):
    v = verdict.lower()
    for word, kind in VERDICT_WORDS:
        if word in v:
            return kind
    raise ValueError(f'a verdict make_figures.py does not know how to draw: {verdict!r}')


def chapter_of(test):
    if test.startswith('WT') or test[0] == 'T':
        return 0
    if test.startswith('G'):
        return 1
    n = int(re.match(r'N(\d+)', test).group(1))
    return 1 if n <= 2 else 2 if n <= 6 else 3 if n <= 9 else 4


def ledger_data():
    """(test, kind) in the ledger's order, and the tests taken from a note because the ledger lacks them. A row naming
    two tests counts for each; N15, withdrawn, is kept as None."""
    section = (HERE / 'CRITICAL_REVIEW.md').read_text().split('## The ledger', 1)[1].split('\n## ', 1)[0]
    rows = []
    for line in section.splitlines():
        cells = [x.strip() for x in line.strip().strip('|').split('|')]
        if len(cells) < 4 or not re.match(r'(WikiText|T\d|G\d|N\d)', cells[0]):
            continue
        bold = re.search(r'\*\*(.+?)\*\*', cells[-1])
        name = {'WikiText-2 replication': 'WT-2', 'WikiText-103': 'WT-103'}.get(cells[0], cells[0])
        if '–' in name or '-' in name.replace('WT-', ''):
            raise ValueError(f'a ledger row naming a range of tests: {name!r}; list them, separated by commas')
        rows += [(t, kind_of(bold.group(1) if bold else cells[-1])) for t in name.split(', ')]
    added = []
    if 'N18a' not in dict(rows):
        rows.append(('N18a', kind_of(re.search(r'\*\*N18a: (\w+)', (HERE / 'N18_NOTE.md').read_text()).group(1))))
        added.append('N18a')
    if 'N15' not in dict(rows):
        at = max(i for i, (t, _) in enumerate(rows) if t.startswith('N14')) + 1
        rows.insert(at, ('N15', None))
    return rows, added


# ------------------------------------------------------------------ figures

def fig_distance(c, data):
    ladder, noisy, rand, rotated = data
    fig, ax = figure(c)
    fig.subplots_adjust(left=0.1, right=0.72, top=0.79, bottom=0.14)
    xs = [1, 2, 4, 8]
    means = [st.mean(ladder[k]) for k in xs]
    lab = dict(color=c['ink2'], fontsize=9)
    ax.axhspan(min(rotated), max(rotated), color=c['muted'], alpha=0.16, linewidth=0, zorder=1)
    ax.text(8.35, st.mean(rotated), 'a mismatched head', va='center', ha='left', clip_on=False, **lab)
    ax.plot([0.85, 8.0], [rand, rand], color=c['muted'], linewidth=1 * PX, zorder=1)
    ax.text(8.35, rand, 'random directions', va='center', ha='left', clip_on=False, **lab)
    for k in xs:
        ax.scatter([k] * len(ladder[k]), ladder[k], s=(5 * PX) ** 2 * 2, color=c['s1'], alpha=c['faint'], linewidths=0, zorder=2)
    ax.plot(xs, means, color=c['s1'], linewidth=2 * PX, solid_capstyle='round', solid_joinstyle='round', zorder=3)
    ax.scatter(xs, means, s=(9 * PX) ** 2, color=c['s1'], edgecolors=c['surface'], linewidths=2 * PX, zorder=4)
    ax.scatter([1] * len(noisy), noisy, s=(5 * PX) ** 2 * 2, color=c['s2'], alpha=c['faint'], linewidths=0, marker='D', zorder=2)
    ax.scatter([1], [st.mean(noisy)], s=(9 * PX) ** 2, color=c['s2'], edgecolors=c['surface'], linewidths=2 * PX, marker='D', zorder=4)
    ax.text(4.4, 0.33, f'the ladder (N6r\'s models):\n{means[-1]:.2f} at 8 words ahead', va='center', ha='left', linespacing=1.3, **lab)
    ax.text(1.12, st.mean(noisy) - 0.035, f'next word, 80% scrambled\nanswers (N7\'s models): {st.mean(noisy):.2f}', va='top', ha='left', linespacing=1.3, **lab)
    ax.text(1.2, means[0], f'two heads, same word: {means[0]:.2f}', va='center', ha='left', **lab)
    ax.set_xscale('log', base=2)
    ax.set_xticks(xs); ax.set_xticklabels(['1', '2', '4', '8'])
    ax.set_xlim(0.85, 8.6); ax.set_ylim(0, 1.02)
    ax.set_xlabel('How many words ahead the second reading predicts (each step doubles)', color=c['ink2'], fontsize=9.5)
    ax.set_ylabel('Shared directions with the next-word reading', color=c['ink2'], fontsize=9.5)
    titles(fig, c, 'The further ahead a reading looks, the fewer directions it shares',
           'Overlap of the directions each reading depends on, measured without the right answers (N8).\nFaint marks: five runs each; solid marks and line: their mean.')
    return fig


def fig_accuracy(c, pts):
    fig, ax = figure(c, h=5.2)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.79, bottom=0.12)
    mean = lambda a: (st.mean(p[0] for p in pts[a]), st.mean(p[1] for p in pts[a]))
    lab = dict(color=c['ink2'], fontsize=8.5)
    lead = dict(color=c['muted'], linewidth=1 * PX)
    ax.axhline(0.85, color=c['muted'], linewidth=1 * PX, zorder=1)
    ax.text(0.905, 0.846, 'the 0.85 bar for "the same reading"', ha='right', va='top', **lab)
    lx, ly = zip(*[mean(a) for a in RUNGS])
    ax.plot(lx, ly, color=c['s1'], linewidth=2 * PX, solid_capstyle='round', solid_joinstyle='round', zorder=2)
    ax.scatter(lx, ly, s=(9 * PX) ** 2, color=c['s1'], edgecolors=c['surface'], linewidths=2 * PX, zorder=3,
               label='Genuine answers only, at fewer positions: the accuracy ladder')
    ax.text(lx[0] + 0.018, ly[0], 'genuine answers at every position', ha='left', va='center', **lab)
    ax.text(lx[-1], ly[-1] - 0.022, 'at 1.25% of positions', ha='center', va='top', **lab)
    sx, sy = zip(*[mean(a) for a in SCRAMBLED])
    ax.scatter(sx, sy, s=(9 * PX) ** 2, color=c['s2'], edgecolors=c['surface'], linewidths=2 * PX, marker='D', zorder=4,
               label='Trained with scrambled answers')
    nx, ny = zip(*[mean(a) for a in NOSCRAMBLE])
    ax.scatter(nx, ny, s=(9 * PX) ** 2, color=c['s3'], edgecolors=c['surface'], linewidths=2 * PX, marker='s', zorder=4,
               label='Discounted genuine answers, no scrambled ones')
    # the three N14 doses: one label, a leader to each
    tx, ty = 0.03, 0.755
    ax.text(tx, ty, '20% genuine plus 10%, 20%\nand 40% scrambled (N14)', ha='left', va='center', linespacing=1.3, **lab)
    for a in ('r10', 'r20', 'r40'):
        x, y = mean(a)
        ax.plot([0.17, x - 0.004], [ty + 0.024, y - 0.006], zorder=1, **lead)
    note = {'rw-oracle': ('clean head judges (N16)', (0.0, 0.82)),
            'n80-mix': ('blended head (N12)', (0.47, 0.805)),
            'sub20-oracle': ('clean head judges,\ngenuine answers only (N16)', (0.40, 0.885)),
            'sub20-rw': ('judges itself,\ngenuine answers only (N14)', (0.66, 0.73))}
    for a, (txt, xy) in note.items():
        ax.annotate(txt, mean(a), xy, ha='left', va='center', linespacing=1.3,
                    arrowprops=dict(arrowstyle='-', shrinkA=2, shrinkB=5, **lead), **lab)
    ax.set_xlim(-0.03, 0.92); ax.set_ylim(0.55, 1.0)
    ax.set_xlabel('How much worse than the clean head it predicts the genuine answers (nats)', color=c['ink2'], fontsize=9.5)
    ax.set_ylabel('Agreement with the clean head\'s directions', color=c['ink2'], fontsize=9.5)
    ax.legend(loc='lower left', frameon=False, fontsize=8.5, labelcolor=c['ink2'], handletextpad=0.4, borderaxespad=0.6)
    titles(fig, c, 'Heads trained with scrambled answers sit close to the accuracy ladder',
           'Second heads placed by the accuracy they reached (N14, N16; means of five runs each).\nNo verdict on whether they sit on it (N14a, N16d inconclusive): small gaps, but uncertain.')
    return fig


def rounded_bar(ax, x, h, width, color, radius_px=4):
    """A bar with a rounded data-end and a square base, the corner radius in display pixels."""
    bb = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    scale = ax.figure.dpi / DPI_DISPLAY
    rx = radius_px * scale / (bb.width / (x1 - x0))
    ry = radius_px * scale / (bb.height / (y1 - y0))
    L, R = x - width / 2, x + width / 2
    verts = [(L, 0), (L, h - ry), (L, h), (L + rx, h), (R - rx, h), (R, h), (R, h - ry), (R, 0), (L, 0)]
    codes = [MPath.MOVETO, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CLOSEPOLY]
    ax.add_patch(PathPatch(MPath(verts, codes), facecolor=color, edgecolor='none', zorder=3))


def fig_twobytwo(c, data):
    fig, ax = figure(c, h=4.6)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.78, bottom=0.14)
    ax.set_xlim(-0.55, 1.55); ax.set_ylim(0, 0.85)
    fig.canvas.draw()
    bb = ax.get_window_extent()
    width = 24 / (bb.width / (ax.get_xlim()[1] - ax.get_xlim()[0]))   # 24 display pixels
    gap = width * 0.3
    groups = (('self', 'The head judges its own answers'), ('clean', 'The clean head judges them'))
    for gi, (g, glabel) in enumerate(groups):
        for bi, (kind, col) in enumerate((('genuine', c['s3']), ('scrambled', c['s2']))):
            a = CELLS[(g, kind)]
            v = st.mean(data[a])
            x = gi + (bi - 0.5) * (width + gap)
            rounded_bar(ax, x, v, width, col)
            ax.text(x, v + 0.015, f'{v:.2f}', color=c['ink'], fontsize=9.5, ha='center', va='bottom', zorder=6,
                    bbox=dict(facecolor=c['surface'], edgecolor='none', pad=1.0))
    ref = st.mean(data['sub20'])
    ax.axhline(ref, color=c['muted'], linewidth=1 * PX, zorder=2)
    ax.text(1.53, ref + 0.012, f'genuine answers only,\nat full weight: {ref:.2f}', color=c['ink2'], fontsize=8.5, ha='right', va='bottom', linespacing=1.3)
    ax.set_xticks([0, 1]); ax.set_xticklabels([g[1] for g in groups])
    ax.tick_params(axis='x', length=0, labelsize=9.5, labelcolor=c['ink'])
    ax.set_ylabel('Gap behind the clean head (nats)', color=c['ink2'], fontsize=9.5)
    ax.scatter([], [], marker='s', s=(9 * PX) ** 2, color=c['s3'], label='Genuine answers only, discounted')
    ax.scatter([], [], marker='s', s=(9 * PX) ** 2, color=c['s2'], label='Plus the scrambled answers')
    ax.legend(loc='upper right', frameon=False, fontsize=8.5, labelcolor=c['ink2'], handletextpad=0.3)
    titles(fig, c, 'The gap shrank with a clean judge, and with the scrambled answers',
           'How far each discounted head falls behind the clean head on the genuine answers; lower is better.\nFive-run means (N16). The judge comparisons are N16\'s claims; the rest is exploratory.')
    return fig


def hbar(ax, y, a, b, height, color, round_a=False, round_b=True, radius_px=4):
    """A horizontal bar from a to b (a < b) at row y; a rounded end gets a corner radius in display pixels."""
    bb = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    scale = ax.figure.dpi / DPI_DISPLAY
    rx = min(radius_px * scale / (bb.width / (x1 - x0)), (b - a) / 2)
    ry = radius_px * scale / (bb.height / abs(y1 - y0))
    lo, hi = y - height / 2, y + height / 2
    v, k = [(a + rx if round_a else a, lo)], [MPath.MOVETO]
    if round_b:
        v += [(b - rx, lo), (b, lo), (b, lo + ry), (b, hi - ry), (b, hi), (b - rx, hi)]; k += [MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3]
    else:
        v += [(b, lo), (b, hi)]; k += [MPath.LINETO, MPath.LINETO]
    if round_a:
        v += [(a + rx, hi), (a, hi), (a, hi - ry), (a, lo + ry), (a, lo), (a + rx, lo)]; k += [MPath.LINETO, MPath.CURVE3, MPath.CURVE3, MPath.LINETO, MPath.CURVE3, MPath.CURVE3]
    else:
        v += [(a, hi), (a, lo)]; k += [MPath.LINETO, MPath.LINETO]
    v.append(v[0]); k.append(MPath.CLOSEPOLY)
    ax.add_patch(PathPatch(MPath(v, k), facecolor=color, edgecolor='none', zorder=3))


def row_chart(c, h, left):
    """A figure for labelled horizontal rows: x gridlines, no y spine."""
    fig, ax = figure(c, h=h)
    fig.subplots_adjust(left=left, right=0.95, top=0.72, bottom=0.16)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=c['grid'], linewidth=1 * PX)
    ax.spines['left'].set_visible(False)
    ax.tick_params(axis='y', length=0, labelcolor=c['ink'], labelsize=9)
    return fig, ax


def fig_shortcut(c, d):
    rows = (('whole', 'The whole gap, as trained', c['muted']), ('G', 'Closed by training with exact gradients', c['s1']),
            ('F', 'The continuous form (size unclear)', c['s1']), ('C2', 'Reusing one block four times', c['s1']),
            ('P', 'Separately: one block’s missing weights', c['s1']))
    ys = [0, 1, 2, 3, 4.4]
    fig, ax = row_chart(c, 3.9, 0.37)
    ax.set_xlim(0, 0.95); ax.set_ylim(5.0, -0.6)
    fig.canvas.draw()
    ax.axvline(0.05, color=c['muted'], linewidth=1 * PX, zorder=2)
    ax.text(0.056, 4.95, 'margin 0.05', color=c['ink2'], fontsize=8.5, va='bottom')
    for (k, _, col), y in zip(rows, ys):
        m = st.mean(d[k])
        hbar(ax, y, 0, m, 0.5, col)
        ax.text(max(m, 0.05) + 0.012, y, f'{m:.3f} nats' if m < 0.1 else f'{m:.2f} nats', color=c['ink'], fontsize=9, va='center')
    ax.set_yticks(ys); ax.set_yticklabels([r[1] for r in rows])
    ax.get_yticklabels()[0].set_fontweight('bold')
    ax.set_xlabel('Loss gap on held-out text (nats)', color=c['ink2'], fontsize=9.5)
    titles(fig, c, 'Most of Model A\'s gap at this budget came from the training shortcut',
           'Model A against the baseline at a tenth of the WikiText-103 budget, five-run means (T1); at the full\nbudget the gap was 1.10 nats. The first three pieces add up to the whole gap; the missing weights\nare a separate contrast. The paper\'s refuted claims stay refuted.')
    return fig


def fig_flow(c, d):
    rows = (('G1', 'Curvature (G1)'), ('N2', 'Pull, per reading (N2)'), ('G1c', 'Pull, whole state (G1c)'))
    fig, ax = figure(c, h=3.6)
    fig.subplots_adjust(left=0.24, right=0.96, top=0.64, bottom=0.17)
    ax.set_xscale('log', base=2); ax.set_xlim(0.9, 12); ax.set_ylim(2.6, -0.6)
    ax.axvspan(0.9, 1.25, color=c['critical'], alpha=0.08, linewidth=0, zorder=0)
    for v, s, ha in ((1.25, 'refuted below 1.25 ', 'right'), (2, ' predicted 2', 'left')):
        ax.axvline(v, color=c['muted'], linewidth=1 * PX, zorder=1)
        ax.text(v, -0.66, s, color=c['ink2'], fontsize=8.5, ha=ha, va='bottom')
    for i, (k, _) in enumerate(rows):
        ax.scatter(d[k], [i] * len(d[k]), s=(8 * PX) ** 2, color=c['s1'], edgecolors=c['surface'], linewidths=1.5 * PX, zorder=3)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[1] for r in rows])
    ax.tick_params(axis='y', length=0, labelcolor=c['ink'], labelsize=9)
    ax.spines['left'].set_visible(False)
    ax.set_xticks([1, 2, 4, 8]); ax.set_xticklabels(['1×', '2×', '4×', '8×']); ax.minorticks_off()
    ax.set_xlabel('Ratio between the two readings, one dot per model (1 = no difference)', color=c['ink2'], fontsize=9.5)
    titles(fig, c, 'The flow\'s bend and per-reading pull differed only slightly, not twice',
           f'Over the whole state the pull did differ, 4 to 10 times (G1c); capping it cost the farther task no\nmore than the nearer (G1d: {sgn(st.mean(d["G1d"]))} nats). G1, G1c and G1d added a control signal into the\nflow; N2\'s model had none. All at a tenth of the training budget.')
    return fig


def fig_angles(c, data):
    ladder, noisy, rand, rotated = data
    items = [('the same word', ladder[1], c['s1']), ('2 words ahead', ladder[2], c['s1']), ('4 words ahead', ladder[4], c['s1']),
             ('8 words ahead', ladder[8], c['s1']), ('next word, 80% scrambled', noisy, c['s2']),
             ('a mismatched head', rotated, c['muted']), ('random directions', [rand], c['muted'])]
    ang = lambda o: math.acos(math.sqrt(o))
    fig = plt.figure(figsize=(7.2, 5.0), dpi=DPI_DISPLAY)
    fig.patch.set_facecolor(c['surface'])
    ax = fig.add_axes([0.05, 0.05, 0.52, 0.72])
    ax.set_aspect('equal'); ax.axis('off'); ax.set_xlim(-0.03, 1.1); ax.set_ylim(-0.1, 1.06)
    for deg in range(0, 91, 15):
        r = math.radians(deg)
        ax.plot([0, math.cos(r)], [0, math.sin(r)], color=c['grid'], linewidth=1 * PX, zorder=1)
        if deg in (0, 30, 90):
            ax.text(1.05 * math.cos(r), 1.05 * math.sin(r), f'{deg}°', color=c['muted'], fontsize=8.5, ha='left' if deg < 90 else 'center', va='center')
    t = np.linspace(0, math.pi / 2, 200)
    ax.plot(np.cos(t), np.sin(t), color=c['axis'], linewidth=1 * PX, zorder=1)
    ax.plot([0, 1], [0, 0], color=c['ink2'], linewidth=3 * PX, solid_capstyle='round', zorder=3)
    ax.text(1.0, -0.04, 'the next-word reading', color=c['ink2'], fontsize=8.5, ha='right', va='top')
    fig.canvas.draw()
    to_fig = lambda x, y: fig.transFigure.inverted().transform(ax.transData.transform((x, y)))
    items.sort(key=lambda it: -ang(st.mean(it[1])))
    prev = 0.8
    for k, vals, col in items:
        a = ang(st.mean(vals))
        ex, ey = math.cos(a), math.sin(a)
        for v in vals:
            ax.scatter([math.cos(ang(v))], [math.sin(ang(v))], s=(4 * PX) ** 2, color=col, linewidths=0, zorder=3)
        ax.plot([0, ex], [0, ey], color=col, linewidth=2 * PX, solid_capstyle='round', zorder=2)
        ax.scatter([ex], [ey], s=(8 * PX) ** 2, color=col, edgecolors=c['surface'], linewidths=1.5 * PX, zorder=4)
        ly = min(to_fig(ex, ey)[1], prev - 0.075)
        prev = ly
        ax.annotate(f'{k} · {st.mean(vals):.2f} · {math.degrees(a):.0f}°', (ex, ey), (0.62, ly), textcoords='figure fraction', va='center',
                    color=c['ink2'], fontsize=8.5, arrowprops=dict(arrowstyle='-', color=c['axis'], linewidth=1 * PX, shrinkA=3, shrinkB=5, relpos=(0, 0.5)))
    titles(fig, c, 'The further ahead a reading looks, the further its directions turn',
           'Each ray sits at the angle whose squared cosine equals its overlap with the next-word reading (N8,\nmeasured without the right answers), so the angle is to scale, though one angle stands for sixteen.')
    return fig


def fig_judges(c, J):
    fig, ax = figure(c, h=5.2)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.77, bottom=0.12)
    m = lambda a, k: st.mean(J[s][a][k] for s in SEEDS12)
    pts = [(m(a, 'x'), m(a, 'F')) for a, _ in LINE]
    ax.set_xlim(-0.03, 0.88); ax.set_ylim(-1.4, 1.3)
    ax.axhline(0, color=c['axis'], linewidth=1 * PX, zorder=1)
    lab = dict(color=c['ink2'], fontsize=8.5)
    ax.plot(*zip(*pts), color=c['ink2'], linewidth=2 * PX, solid_joinstyle='round', zorder=2,
            label='The loop-free line: full weight, genuine\nanswers at 20% to 1.25% of positions')
    ax.scatter(*zip(*pts), s=(6 * PX) ** 2, color=c['ink2'], zorder=3)
    for (x, y), (_, s) in zip(pts, LINE):
        ax.text(x + 0.012, y + 0.05, s, ha='left', va='bottom', **lab)
    style_of = {'rw-oracle': (c['s3'], 's'), 'rw-ctwin': (c['s3'], 's'), 'rw-peer': (c['s2'], 'D'), 'rw-b1': (c['s2'], 'D'), 'rw-cloop': (c['s2'], 'D')}
    for a in JUDGED:
        col, mk = style_of[a]
        x, y = m(a, 'x'), m(a, 'F')
        ln = line_at(pts, x) if a in ('rw-ctwin', 'rw-peer', 'rw-cloop') else None
        if ln is not None:
            ax.plot([x, x], [ln, y], color=c['muted'], linewidth=1 * PX, zorder=1)
        ax.scatter([J[s][a]['x'] for s in SEEDS12], [J[s][a]['F'] for s in SEEDS12], s=(4 * PX) ** 2, color=col, alpha=c['faint'], linewidths=0, marker=mk, zorder=3)
        ax.scatter([x], [y], s=(9 * PX) ** 2, color=col, edgecolors=c['surface'], linewidths=2 * PX, marker=mk, zorder=4)
    d = lambda a: d_within(J, a)
    lead = dict(arrowstyle='-', color=c['muted'], linewidth=1 * PX, shrinkA=2, shrinkB=6)
    ax.text(m('rw-oracle', 'x') + 0.02, m('rw-oracle', 'F'), 'the clean head judges (N16)', va='center', **lab)
    for i, (a, txt) in enumerate((('rw-ctwin', f'weighted by a loop-free twin · d = {sgn(d("rw-ctwin"), 2)}'),
                                  ('rw-peer', f'the peer (N17, N18) · d = {sgn(d("rw-peer"), 2)}'),
                                  ('rw-b1', 'the head judges itself (0 by definition)'))):
        ax.annotate(txt, (m(a, 'x'), m(a, 'F')), (0.30, -0.2 - 0.17 * i), ha='right', va='center', arrowprops=dict(relpos=(1, 0.5), **lead), **lab)
    xc, yc = m('rw-cloop', 'x'), m('rw-cloop', 'F')
    ax.text(xc - 0.012, (yc + line_at(pts, xc)) / 2, f'd = {sgn(d("rw-cloop"), 2)}', ha='right', va='center', **lab)
    ax.text(xc - 0.02, yc - 0.05, 'the judge weights itself,\ngenuine answers only (N18)', ha='right', va='top', linespacing=1.3, **lab)
    ax.scatter([], [], marker='s', s=(9 * PX) ** 2, color=c['s3'], label='A judge without a loop')
    ax.scatter([], [], marker='D', s=(9 * PX) ** 2, color=c['s2'], label='A judge with a loop')
    ax.legend(loc='lower left', frameon=False, fontsize=8.5, labelcolor=c['ink2'], handletextpad=0.4, borderaxespad=0.6)
    ax.set_xlabel('The judge\'s own loss gap behind the clean head (nats)', color=c['ink2'], fontsize=9.5)
    ax.set_ylabel('Share of the shortfall its head recovered', color=c['ink2'], fontsize=9.5)
    titles(fig, c, 'The self-weighting judge fell furthest below the loop-free line',
           'Five-run means (N18); faint marks are the runs. d is the distance below the line within each run (S8,\nsecondary). The loop-free twin fell below it too, by less. The peer\'s judge also weights itself, on\nscrambled answers; the self-weighting judge trained on genuine answers only.')
    return fig


def fig_steps(c, J):
    v = lambda a: st.mean(J[s][a]['dA'] for s in SEEDS12)
    j20, ct, cl, peer = v('rw-j20'), v('rw-ctwin'), v('rw-cloop'), v('rw-peer')
    rows = (('Judge at full weight (rw-j20)', 0, j20, c['muted'], True), ('+ the weighting, from a loop-free twin', j20, ct, c['s3'], False),
            ('+ the loop: weights from the judge itself', ct, cl, c['s2'], False), ('+ the scrambled answers, as in the peer', cl, peer, c['s1'], False),
            ('= the peer (rw-peer)', 0, peer, c['muted'], True))
    fig, ax = row_chart(c, 3.9, 0.37)
    ax.set_xlim(0, 0.66); ax.set_ylim(4.6, -0.6)
    fig.canvas.draw()
    for i, (_, a, b, col, total) in enumerate(rows):
        lo, hi = min(a, b), max(a, b)
        hbar(ax, i, lo, hi, 0.5, col, round_a=not total, round_b=True)
        if 0 < i < len(rows) - 1:
            ax.plot([a, a], [i - 0.75, i - 0.25], color=c['axis'], linewidth=1 * PX, zorder=2)
        ax.text(hi + 0.008, i, f'{b:.3f}' if total else sgn(b - a), color=c['ink'], fontsize=9, va='center')
    ax.set_yticks(range(len(rows))); ax.set_yticklabels([r[0] for r in rows])
    for t in (ax.get_yticklabels()[0], ax.get_yticklabels()[-1]):
        t.set_fontweight('bold')
    ax.set_xlabel('Head B\'s gap behind the clean head on the genuine answers (Δ_A, nats)', color=c['ink2'], fontsize=9.5)
    titles(fig, c, f'On genuine answers alone, the loop cost about twice the peer\'s gap',
           f'Five-run means (N18). The peer\'s gap: {peer:.2f} − {j20:.2f} = {peer - j20:.2f} nats. Each step\'s size depends on\nthe steps before it. Adding the scrambled answers lowered the gap by {cl - peer:.2f} nats; N18 cannot\nsay how much of that is the loop tempered, nor that the peer\'s own gap is the loop\'s.')
    return fig


def fig_ledger(c, led):
    rows, added = led
    fig = plt.figure(figsize=(7.2, 3.8), dpi=DPI_DISPLAY)
    fig.patch.set_facecolor(c['surface'])
    ax = fig.add_axes([0.03, 0.03, 0.94, 0.55])
    ax.axis('off')
    n = max(sum(chapter_of(t) == g for t, _ in rows) for g in range(len(CHAPTERS)))
    ax.set_xlim(-13.2, n + 0.2); ax.set_ylim(len(CHAPTERS) * 1.3, -0.2); ax.set_aspect('equal'); ax.set_anchor('N')
    for g, name in enumerate(CHAPTERS):
        y = g * 1.3 + 0.45
        ax.text(-13.0, y, name, color=c['ink2'], fontsize=8.5, va='center')
        for i, (t, k) in enumerate((t, k) for t, k in rows if chapter_of(t) == g):
            if k is None:
                ax.add_patch(plt.Rectangle((i, y - 0.45), 0.9, 0.9, facecolor='none', edgecolor=c['none'], linewidth=1.5 * PX))
                ax.text(i + 0.45, y, '—', color=c['muted'], fontsize=8, ha='center', va='center')
                continue
            col, glyph, _ = KIND[k]
            ax.add_patch(plt.Rectangle((i, y - 0.45), 0.9, 0.9, facecolor=c[col], edgecolor='none'))
            ax.text(i + 0.45, y, glyph, color='#0b0b0b' if k == 'unclear' else '#ffffff', fontsize=8, fontweight='bold', ha='center', va='center')
    counts = {k: sum(kk == k for _, kk in rows) for k in KIND}
    renderer = fig.canvas.get_renderer()
    x, y = 0.075, 0.69
    for k, (col, glyph, name) in KIND.items():
        t = fig.text(x + 0.02, y, f'{glyph} {name} · {counts[k]}', color=c['ink2'], fontsize=8.5, va='center')
        w = t.get_window_extent(renderer).width / fig.bbox.width
        if x + 0.02 + w > 0.97:   # wrap to a second row
            x, y = 0.075, y - 0.065
            t.set_position((x + 0.02, y))
        fig.patches.append(plt.Rectangle((x, y - 0.013), 0.014, 0.026, transform=fig.transFigure, facecolor=c[col], edgecolor='none'))
        x += 0.02 + w + 0.03
    extra = f', with {" and ".join(added)} from {"its note" if len(added) == 1 else "their notes"}' if added else ''
    titles(fig, c, f'{sum(counts.values())} verdicts in the ledger\'s main rows, in order',
           f'The main claims in CRITICAL_REVIEW.md\'s ledger{extra}; secondary claims and T1\'s rivals are in the\nnotes. A row naming two tests counts for each. WT-2 is "mostly refuted". N15, withdrawn before its\nreal run, is drawn hollow and not counted.')
    return fig


# The imagined jigsaw: each piece a finding (glued where its verdict decided something, half where it decided part,
# loose where it did not), the empty centre the project's starting idea, and four imagined threads. Nothing here is measured.
PIECES = {(0, 0): ('The paper refuted; the gap mostly the shortcut', 'WT-2, WT-103, T1', 'glued'), (0, 1): ('The flow differs a little, not twice', 'G1, N2', 'glued'),
          (0, 2): ('The flow’s pull: real; not needed, by this test', 'G1c, G1d', 'glued'), (0, 3): ('A fixed shift suffices to switch', 'N1', 'glued'),
          (1, 0): ('Sharing directions costs accuracy', 'N4', 'glued'), (1, 3): ('Why apart? a competition?', 'N5', 'loose'),
          (2, 0): ('Further ahead, more separate', 'N6r', 'glued'), (2, 3): ('Not a noisy copy: a different reading', 'N7–N11', 'glued'),
          (3, 0): ('The remedy weighs each answer', 'N12, N13', 'glued'), (3, 1): ('Directions follow accuracy?', 'N14a, N16d', 'loose'),
          (3, 2): ('Who judges matters', 'N16', 'glued'), (3, 3): ('A judge that grades itself', 'N17, N18', 'half')}
THREADS = (((2, 0), 'C', 'Two readings already behave a little like separate voices: the further ahead a reading looks, the less its directions overlap the next-word reading’s, and forcing them together costs accuracy. A many-voiced model might hold each voice in directions of its own.'),
           ((2, 3), (3, 3), 'A head trained on scrambled answers came to read differently, not as a noisy copy of the right reading; a judge that weights its own training is a head shaping its own answers. Perhaps the two are one phenomenon seen twice.'),
           ((0, 1), (1, 0), 'At this scale the flow’s bend and per-reading pull differed only slightly (G1, N2), and its 4- to 10-fold difference in whole-state pull was not needed (G1c, G1d). If the readings live anywhere here, it may be in which directions each one uses.'),
           ((3, 2), 'C', 'With the clean head as judge, a diagnostic rather than a method, the shortfall closed (N16); a judge weighting its own training did harm when it trained on genuine answers alone (N18a). Whether independence or accuracy helped is not separated. Perhaps many readings could check one another.'))


def mix(a, b, t):
    """The colour t of the way from a to b."""
    pa, pb = (np.array([int(h[i:i + 2], 16) for i in (1, 3, 5)]) for h in (a, b))
    return '#' + ''.join(f'{int(round(x)):02x}' for x in pa + (pb - pa) * t)


def fig_jigsaw(c):
    cell, ch, gx, gy = 180, 140.4, 20, 14
    W, H = gx * 2 + cell * 4, gy * 2 + ch * 4
    fig = plt.figure(figsize=(7.2, 9.4), dpi=DPI_DISPLAY)
    fig.patch.set_facecolor(c['surface'])
    ax = fig.add_axes([0.07, 0.31, 0.86, 0.86 * 7.2 / 9.4 * (H + 34) / (W + 28)])
    ax.set_xlim(-14, W + 14); ax.set_ylim(H + 26, -8); ax.axis('off')   # y runs down the board; the margin holds the tilted pieces
    h = [[1, -1, 1, -1], [-1, 1, -1, 1], [1, 1, -1, -1]]   # knob directions between rows, then between columns
    v = [[1, -1, 1], [-1, 1, -1], [1, -1, 1], [-1, 1, -1]]

    def edge(x0, y0, x1, y1, ty):
        if not ty:
            return [(x1, y1)], [MPath.LINETO]
        dx, dy = x1 - x0, y1 - y0
        ln = math.hypot(dx, dy); ux, uy = dx / ln, dy / ln; nx, ny = uy * ty, -ux * ty; k = min(ln, 105)
        P = lambda a, b: (x0 + ux * ln * a + nx * k * b, y0 + uy * ln * a + ny * k * b)
        return ([P(0.36, 0), P(0.42, 0.12), P(0.30, 0.24), P(0.5, 0.24), P(0.70, 0.24), P(0.58, 0.12), P(0.64, 0), (x1, y1)],
                [MPath.LINETO] + [MPath.CURVE4] * 6 + [MPath.LINETO])

    def piece(r, col, place=lambda p: p):
        x, y = gx + col * cell, gy + r * ch
        top, bottom = (-h[r - 1][col] if r > 0 else 0), (h[r][col] if r < 3 else 0)
        left, right = (-v[r][col - 1] if col > 0 else 0), (v[r][col] if col < 3 else 0)
        verts, codes = [(x, y)], [MPath.MOVETO]
        for seg in (edge(x, y, x + cell, y, top), edge(x + cell, y, x + cell, y + ch, right), edge(x + cell, y + ch, x, y + ch, bottom), edge(x, y + ch, x, y, left)):
            verts += seg[0]; codes += seg[1]
        verts.append((x, y)); codes.append(MPath.CLOSEPOLY)
        return MPath([place(p) for p in verts], codes)

    glued, half = mix(c['surface'], c['ink'], 0.09), mix(c['surface'], c['ink'], 0.045)
    centres, words = {}, []
    for r in range(4):
        for col in range(4):
            cx, cy = gx + col * cell + cell / 2, gy + r * ch + ch / 2
            if (r, col) not in PIECES:
                ax.add_patch(PathPatch(piece(r, col), facecolor='none', edgecolor=c['grid'], linewidth=1 * PX, linestyle=(0, (3, 4))))
                continue
            title, tests, state = PIECES[(r, col)]
            tilt = math.radians({'loose': 5 if col > 1 else -5, 'half': 2.5}.get(state, 0))
            off = {'loose': 10, 'half': 4}.get(state, 0)
            dxo = off * (1 if col > 1 else -1)
            place = lambda p, cx=cx, cy=cy, tilt=tilt, dxo=dxo, off=off: (cx + (p[0] - cx) * math.cos(tilt) - (p[1] - cy) * math.sin(tilt) + dxo,
                                                                         cy + (p[0] - cx) * math.sin(tilt) + (p[1] - cy) * math.cos(tilt) + off)
            ax.add_patch(PathPatch(piece(r, col, place), facecolor={'glued': glued, 'half': half}.get(state, c['surface']),
                                   edgecolor=c['ink2'] if state == 'glued' else c['muted'], linewidth=(1.2 if state == 'glued' else 1.4) * PX))
            lines = textwrap.wrap(title, 16)
            centres[(r, col)] = (cx, cy, 62, (len(lines) * 16 + 16) / 2 + 8)
            words.append((lines, tests, state, place, cx, cy, math.degrees(tilt)))
    mx, my = gx + 2 * cell, gy + 2 * ch
    block = lambda key: (mx, my, 86, 46) if key == 'C' else centres[key]

    def clip(b, tx, ty):
        cx, cy, hw, hh = b
        dx, dy = tx - cx, ty - cy
        k = min(hw / max(abs(dx), 1e-6), hh / max(abs(dy), 1e-6))
        return cx + dx * k, cy + dy * k
    nums = []
    for a, b, _ in THREADS:
        A, B = block(a), block(b)
        qx, qy = (A[0] + B[0]) / 2 + (B[1] - A[1]) * 0.25, (A[1] + B[1]) / 2 - (B[0] - A[0]) * 0.25
        s0, e0 = clip(A, qx, qy), clip(B, qx, qy)
        ax.add_patch(PathPatch(MPath([s0, (qx, qy), e0], [MPath.MOVETO, MPath.CURVE3, MPath.CURVE3]), facecolor='none', edgecolor=c['imagined'], linewidth=1.6 * PX))
        ax.scatter([s0[0], e0[0]], [s0[1], e0[1]], s=(5 * PX) ** 2, color=c['imagined'], linewidths=0, zorder=3)
        nums.append((0.25 * s0[0] + 0.5 * qx + 0.25 * e0[0], 0.25 * s0[1] + 0.5 * qy + 0.25 * e0[1]))
    for lines, tests, state, place, cx, cy, tilt in words:
        halo = [pe.withStroke(linewidth=3, foreground={'glued': glued, 'half': half}.get(state, c['surface']))]
        y0 = cy - (len(lines) * 16 + 16) / 2 + 8
        for i, ln in enumerate(lines + [tests]):
            tx, ty = place((cx, y0 + i * 16 + (4 if i == len(lines) else 0)))
            ax.text(tx, ty, ln, color=c['muted'] if i == len(lines) else c['ink'], fontsize=7.5,
                    fontweight='normal' if i == len(lines) else 'bold', ha='center', va='center', rotation=-tilt, path_effects=halo, zorder=5)
    halo = [pe.withStroke(linewidth=3, foreground=c['surface'])]
    ax.text(mx, my - 16, 'Several readings\nheld at once', color=c['imagined'], fontsize=13, family='DejaVu Serif', ha='center', va='center',
            linespacing=1.2, path_effects=halo, zorder=5)
    ax.text(mx, my + 26, 'the idea · untested', color=c['muted'], fontsize=8, ha='center', va='center', path_effects=halo, zorder=5)
    for i, (bx, by) in enumerate(nums):
        ax.scatter([bx], [by], s=(20 * PX) ** 2, color=c['surface'], edgecolors=c['imagined'], linewidths=1.5 * PX, zorder=6)
        ax.text(bx, by, str(i + 1), color=c['imagined'], fontsize=8, fontweight='bold', ha='center', va='center', zorder=7)
    y = 0.295
    for i, (_, _, txt) in enumerate(THREADS):
        wrapped = textwrap.fill(txt, 104)
        fig.text(0.05, y, str(i + 1), color=c['imagined'], fontsize=8.5, fontweight='bold', va='top')
        fig.text(0.075, y, wrapped, color=c['ink2'], fontsize=8.5, va='top', linespacing=1.35)
        y -= 0.0172 * (wrapped.count('\n') + 1) + 0.013
    titles(fig, c, 'Imagined: the findings as a jigsaw, the centre still empty',
           'Not a result. Each piece gathers the tests named on it: filled where they decided something,\nhalf-filled where they decided part, empty and tilted where they did not. The threads are imagined.')
    return fig


def write_data(dist, acc, tb, sc, fl, jd, led):
    ladder, noisy, rand, rotated = dist
    L = ['# Figure data', '', 'Every value plotted in the figures, by run. Generated by `make_figures.py` from the records in `results/` (T1\'s in `../too-linear-conjecture/results/t1/`) and, for figure 9, the ledger in `CRITICAL_REVIEW.md` (with N18a\'s verdict from `N18_NOTE.md` until the ledger has it). Figures 1 to 3 appear in the plain summary. Figure 10 is imagined and plots no measurement.', '',
         '## Figure 1: the distance ladder', '',
         f'Target-free overlap of the directions head B and the next-word head A depend on (N8). Random directions, in theory: {rand:.4f}. '
         f'The deliberately mismatched head (N8\'s rotated control on N6r\'s same-word models): ' + ', '.join(f'{v:.3f}' for v in rotated) + '. '
         f'Seeds: {SEEDS6R[0]}–{SEEDS6R[-1]} for N6r\'s models, {SEEDS7[0]}–{SEEDS7[-1]} for N7\'s; run 1 is the first seed of each.', '',
         '| Second reading | ' + ' | '.join(f'run {i + 1}' for i in range(5)) + ' | Mean |', '|---|' + '---|' * 6]
    for k in (1, 2, 4, 8):
        name = 'the same word' if k == 1 else f'{k} words ahead'
        L.append(f'| {name} (N6r models, `results/n8/f_n6r_*`) | ' + ' | '.join(f'{v:.3f}' for v in ladder[k]) + f' | {st.mean(ladder[k]):.3f} |')
    L.append('| the next word, 80% scrambled answers (N7 models, `results/n8/f_n7_noisy_*`) | ' + ' | '.join(f'{v:.3f}' for v in noisy) + f' | {st.mean(noisy):.3f} |')
    L += ['', '## Figure 2: the accuracy ladder', '',
          'Δ_A: how much worse than the clean head the head predicts the genuine answers, in nats; a: its agreement with the clean head\'s directions. Runs are seeds 4101–4105; ladder rungs and `sub20-rw` from `results/n16/r_*`, N16\'s heads from `results/n16/m_*`, N14\'s doses from `results/n14/m_*`, the blended head from `results/n14/r_n80-mix_*`.', '',
          '| Head | ' + ' | '.join(f'run {i + 1} (Δ_A, a)' for i in range(5)) + ' | Mean |', '|---|' + '---|' * 6]
    for a in list(RUNGS) + list(SCRAMBLED) + list(NOSCRAMBLE):
        p = acc[a]
        L.append(f'| {ACC_NAMES[a]} | ' + ' | '.join(f'({x:.3f}, {y:.3f})' for x, y in p) + f' | ({st.mean(x for x, _ in p):.3f}, {st.mean(y for _, y in p):.3f}) |')
    L += ['', '## Figure 3: N16\'s two-by-two', '', 'Δ_A in nats, seeds 4101–4105, from `results/n16/`.', '',
          '| Head | ' + ' | '.join(f'run {i + 1}' for i in range(5)) + ' | Mean |', '|---|' + '---|' * 6]
    for a, name in (('sub20-rw', 'judges itself, genuine answers only'), ('rw-b1', 'judges itself, plus scrambled answers'),
                    ('sub20-oracle', 'clean head judges, genuine answers only'), ('rw-oracle', 'clean head judges, plus scrambled answers'),
                    ('sub20', 'reference: genuine answers at full weight')):
        L.append(f'| {name} | ' + ' | '.join(f'{v:.3f}' for v in tb[a]) + f' | {st.mean(tb[a]):.3f} |')
    run_head = lambda first: [f'| {first} | ' + ' | '.join(f'run {i + 1}' for i in range(5)) + ' | Mean |', '|---|' + '---|' * 6]
    row = lambda name, xs, k=3: f'| {name} | ' + ' | '.join(f'{x:.{k}f}' for x in xs) + f' | {st.mean(xs):.{k}f} |'
    L += ['', '## Figure 4: where the paper\'s gap came from (T1)', '',
          f'Validation loss differences in nats, seeds {SEEDS_T1[0]}–{SEEDS_T1[-1]}, from `../too-linear-conjecture/results/t1/main_*`. '
          'AA is the paper\'s model trained the paper\'s way, AX the same model with exact gradients, T4 one standard block reused four times, '
          'B6 the baseline, B5 the baseline less one block. Intervals and verdicts are in `../too-linear-conjecture/T1_NOTE.md`.', ''] + run_head('Contrast')
    for k, name in (('whole', 'the whole gap, AA − B6'), ('G', 'G: exact gradients, AA − AX'), ('F', 'F: the continuous form, AX − T4'),
                    ('C2', 'C2: reusing one block, T4 − B6'), ('P', 'P, separately: one block\'s missing weights, B5 − B6')):
        L.append(row(name, sc[k], 4))
    L += ['', '## Figure 5: the flow', '',
          f'The ratio between the two readings, per model, scored as each analysis script scores it. G1: seeds {SEEDS_G1[0]}–{SEEDS_G1[-1]}, '
          f'`results/g1/geom_main_*`; N2: seeds {SEEDS_N2[0]}–{SEEDS_N2[-1]}, `results/n2/sub_dual_*`; G1c: seeds {SEEDS_G1C[0]}–{SEEDS_G1C[-1]}, '
          f'`results/g1c/contr_main_*`. G1d\'s D (nats): seeds {SEEDS_G1D[0]}–{SEEDS_G1D[-1]}, `results/g1d/train_*`.', ''] + run_head('Measure')
    for k, name in (('G1', 'curvature (G1)'), ('N2', 'pull per dimension, within each reading\'s 16 directions (N2)'),
                    ('G1c', 'pull, whole state (G1c)'), ('G1d', 'G1d: D, the farther task\'s cost of the cap minus the nearer task\'s')):
        L.append(row(name, fl[k], 4 if k == 'G1d' else 3))
    L += ['', '## Figure 6: the angle fan', '',
          'Figure 1\'s overlaps (and the mismatched head and random directions) as angles: θ = arccos(√overlap), from the mean overlap. '
          'One angle stands for sixteen: the overlap is the mean squared cosine of the sixteen principal angles.', '',
          '| Second reading | Mean overlap | Angle |', '|---|---|---|']
    for name, vals in (('the same word', ladder[1]), ('2 words ahead', ladder[2]), ('4 words ahead', ladder[4]), ('8 words ahead', ladder[8]),
                       ('the next word, 80% scrambled answers', noisy), ('a mismatched head', rotated), ('random directions, in theory', [rand])):
        L.append(f'| {name} | {st.mean(vals):.3f} | {math.degrees(math.acos(math.sqrt(st.mean(vals)))):.1f}° |')
    jm = lambda a, k: st.mean(jd[s][a][k] for s in SEEDS12)
    pts = [(jm(a, 'x'), jm(a, 'F')) for a, _ in LINE]
    L += ['', '## Figures 7 and 8: N18\'s judges and steps', '',
          f'Seeds {SEEDS12[0]}–{SEEDS12[-1]}, from `results/n18/` (`m_*` where present, else `r_*`). x: the judge\'s plain cross-entropy on the '
          'genuine answers minus head A\'s (head A judging: 0; head B judging itself: its own Δ_A). F = (Δ_A(rw-b1) − Δ_A(arm)) / '
          '(Δ_A(rw-b1) − Δ_A(sub20)). d (S8, secondary): within each run, F minus the loop-free line through that run\'s five rungs at the judge\'s x, then the mean over runs, as `n18_analyze.py` computes it. The drawn drops run from the line through the rung means to the mean F.', '',
          '| Arm | x by run | F by run | Δ_A by run | Mean x | Mean F | Mean Δ_A | d |', '|---|---|---|---|---|---|---|---|']
    for a in [a for a, _ in LINE] + list(JUDGED):
        dd = d_within(jd, a) if a in ('rw-ctwin', 'rw-peer', 'rw-cloop') else None
        L.append(f'| `{a}` | ' + ', '.join(f'{jd[s][a]["x"]:.3f}' for s in SEEDS12) + ' | ' + ', '.join(f'{jd[s][a]["F"]:+.3f}' for s in SEEDS12) + ' | '
                 + ', '.join(f'{jd[s][a]["dA"]:.3f}' for s in SEEDS12) + f' | {jm(a, "x"):.3f} | {jm(a, "F"):+.3f} | {jm(a, "dA"):.3f} | '
                 + (f'{dd:+.3f}' if dd is not None else '') + ' |')
    led, added = led
    L += ['', '## Figure 9: the ledger', '', 'The main rows of the ledger, in its order' + (f', with {", ".join(added)} from its note' if added else '') + '. Secondary claims and T1\'s rivals are in the notes. N15 was withdrawn before its real run and is not counted.', '',
          '| Chapter | Tests and verdicts |', '|---|---|']
    for g, name in enumerate(CHAPTERS):
        L.append(f'| {name} | ' + ', '.join(f'{t} {KIND[k][2].lower() if k else "withdrawn"}' for t, k in led if chapter_of(t) == g) + ' |')
    L.append('| Counts | ' + ', '.join(f'{KIND[k][2].lower()} {sum(kk == k for _, kk in led)}' for k in KIND) + ' |')
    L += ['', '## Figure 10: imagined', '', 'No measurement. Each piece\'s state is a judgement from the ledger; the threads are conjectures.', '',
          '| Piece | Tests | State |', '|---|---|---|'] + [f'| {t} | {tests} | {state} |' for t, tests, state in PIECES.values()]
    (OUT / 'FIGURE_DATA.md').write_text('\n'.join(L) + '\n')


def main():
    plt.rcParams['font.family'] = 'DejaVu Sans'
    OUT.mkdir(exist_ok=True)
    dist, acc, tb = distance_data(), accuracy_data(), twobytwo_data()
    sc, fl, jd, led = shortcut_data(), flow_data(), judge_data(), ledger_data()
    for mode, c in THEMES.items():
        save(fig_distance(c, dist), 'distance_ladder', mode)
        save(fig_accuracy(c, acc), 'accuracy_ladder', mode)
        save(fig_twobytwo(c, tb), 'judge_two_by_two', mode)
        save(fig_shortcut(c, sc), 'shortcut_gap', mode)
        save(fig_flow(c, fl), 'flow_ratios', mode)
        save(fig_angles(c, dist), 'angle_fan', mode)
        save(fig_judges(c, jd), 'judges_and_the_line', mode)
        save(fig_steps(c, jd), 'n18_steps', mode)
        save(fig_ledger(c, led), 'ledger', mode, tight=True)
        save(fig_jigsaw(c), 'imagined_jigsaw', mode)
    write_data(dist, acc, tb, sc, fl, jd, led)
    print('wrote', sorted(p.name for p in OUT.iterdir()))


if __name__ == '__main__':
    main()
