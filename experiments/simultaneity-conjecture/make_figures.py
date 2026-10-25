"""Draw the plain summary's figures from the saved records. Every plotted value comes from a JSON record in results/.

Writes, in figures/, each figure in a light and a dark version (PNG, at twice display size) and FIGURE_DATA.md, the
table of every plotted value by run.

  1. The distance ladder: how much the second reading shares directions with the next-word reading, by how far ahead it
     reads (N8's target-free overlap on N6r's models), with the next-word reading trained on 80% scrambled answers
     (N8 on N7's models) and the level of random directions.
  2. The accuracy ladder: agreement with the clean head against how much worse than the clean head a head predicts the
     genuine answers, for heads given fewer genuine answers (the ladder), heads trained with scrambled answers (N14's
     doses, N12's blended head, N16's clean-judged head), and discounted heads given no scrambled answers (N14, N16).
  3. N16's two-by-two: who judges the answers, and whether the scrambled answers are present.

    python make_figures.py
"""
import json
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import PathPatch  # noqa: E402
from matplotlib.path import Path as MPath  # noqa: E402

HERE = Path(__file__).resolve().parent
RES = HERE / 'results'
OUT = HERE / 'figures'
SEEDS6R = [3701, 3702, 3703, 3704, 3705]
SEEDS7 = [3801, 3802, 3803, 3804, 3805]
SEEDS12 = [4101, 4102, 4103, 4104, 4105]

# The reference palette's first three categorical slots (validated all-pairs in both modes) and its chrome.
THEMES = {
    'light': dict(surface='#fcfcfb', ink='#0b0b0b', ink2='#52514e', muted='#898781', grid='#e1e0d9', axis='#c3c2b7',
                  s1='#2a78d6', s2='#eb6834', s3='#1baf7a'),
    'dark': dict(surface='#1a1a19', ink='#f0efec', ink2='#c3c2b7', muted='#898781', grid='#2c2c2a', axis='#383835',
                 s1='#3987e5', s2='#d95926', s3='#199e70'),
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


def save(fig, name, mode):
    fig.savefig(OUT / f'{name}_{mode}.png', dpi=DPI_DISPLAY * SCALE, facecolor=fig.get_facecolor())
    plt.close(fig)


# ------------------------------------------------------------------ data

def distance_data():
    ladder = {}
    for arm, k in (('same', 1), ('k2', 2), ('k4', 4), ('k8', 8)):
        ladder[k] = [load('n8', f'f_n6r_{arm}_seed{s}')['fisher']['overlap_AB'] for s in SEEDS6R]
    noisy = [load('n8', f'f_n7_noisy_seed{s}')['fisher']['overlap_AB'] for s in SEEDS7]
    rank = load('n8', f'f_n6r_k2_seed{SEEDS6R[0]}')['fisher']['rank']
    return ladder, noisy, rank / 256


RUNGS = ('clean', 'sub20', 'sub10', 'sub05', 'sub025', 'sub0125')
RUNG_LABEL = {'clean': 'all', 'sub20': '20%', 'sub10': '10%', 'sub05': '5%', 'sub025': '2.5%', 'sub0125': '1.25%'}
SCRAMBLED = {'r10': ('n14', 'm_r10'), 'r20': ('n14', 'm_r20'), 'r40': ('n14', 'm_r40'), 'n80-mix': ('n14', 'r_n80-mix'),
             'rw-oracle': ('n16', 'm_rw-oracle')}
NOSCRAMBLE = {'sub20-rw': ('n16', 'r_sub20-rw'), 'sub20-oracle': ('n16', 'm_sub20-oracle')}


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


# ------------------------------------------------------------------ figures

def fig_distance(c, data):
    ladder, noisy, rand = data
    fig, ax = figure(c)
    fig.subplots_adjust(left=0.1, right=0.74, top=0.79, bottom=0.14)
    xs = [1, 2, 4, 8]
    means = [st.mean(ladder[k]) for k in xs]
    ax.axhline(rand, color=c['muted'], linewidth=1 * PX, zorder=1)
    ax.text(8.25, rand, 'random directions', color=c['ink2'], fontsize=9, va='center', ha='left', clip_on=False)
    for k in xs:
        ax.scatter([k] * len(ladder[k]), ladder[k], s=(5 * PX) ** 2 * 2, color=c['s1'], alpha=0.35, linewidths=0, zorder=2)
    ax.plot(xs, means, color=c['s1'], linewidth=2 * PX, solid_capstyle='round', solid_joinstyle='round', zorder=3)
    ax.scatter(xs, means, s=(9 * PX) ** 2, color=c['s1'], edgecolors=c['surface'], linewidths=2 * PX, zorder=4)
    ax.scatter([1] * len(noisy), noisy, s=(5 * PX) ** 2 * 2, color=c['s2'], alpha=0.35, linewidths=0, marker='D', zorder=2)
    ax.scatter([1], [st.mean(noisy)], s=(9 * PX) ** 2, color=c['s2'], edgecolors=c['surface'], linewidths=2 * PX, marker='D', zorder=4)
    ax.text(8.25, means[-1], f'clean readings\n(N6r, N8): {means[-1]:.2f} at 8', color=c['ink2'], fontsize=9, va='center', ha='left', clip_on=False)
    ax.text(1.25, st.mean(noisy) - 0.03, f'the next word, 80% scrambled\nanswers (N7, N8): {st.mean(noisy):.2f}', color=c['ink2'], fontsize=9, va='top', ha='left', linespacing=1.3)
    ax.text(1.25, means[0], f'two heads, same word: {means[0]:.2f}', color=c['ink2'], fontsize=9, va='center', ha='left')
    ax.set_xscale('log', base=2)
    ax.set_xticks(xs); ax.set_xticklabels(['1', '2', '4', '8'])
    ax.set_xlim(0.85, 8.6); ax.set_ylim(0, 1.02)
    ax.set_xlabel('How many words ahead the second reading predicts', color=c['ink2'], fontsize=9.5)
    ax.set_ylabel('Shared directions with the next-word reading', color=c['ink2'], fontsize=9.5)
    titles(fig, c, 'The further ahead a reading looks, the fewer directions it shares',
           'Overlap of the directions each reading depends on, measured without the right answers.\nFaint marks: five runs each; solid marks and line: their mean.')
    return fig


def fig_accuracy(c, pts):
    fig, ax = figure(c, h=5.2)
    fig.subplots_adjust(left=0.1, right=0.97, top=0.79, bottom=0.12)
    mean = lambda a: (st.mean(p[0] for p in pts[a]), st.mean(p[1] for p in pts[a]))
    lab = dict(color=c['ink2'], fontsize=8.5)
    ax.axhline(0.85, color=c['muted'], linewidth=1 * PX, zorder=1)
    ax.text(0.905, 0.846, 'the 0.85 bar for "the same reading"', ha='right', va='top', **lab)
    lx, ly = zip(*[mean(a) for a in RUNGS])
    ax.plot(lx, ly, color=c['s1'], linewidth=2 * PX, solid_capstyle='round', solid_joinstyle='round', zorder=2)
    ax.scatter(lx, ly, s=(9 * PX) ** 2, color=c['s1'], edgecolors=c['surface'], linewidths=2 * PX, zorder=3,
               label='Given only genuine answers, at fewer and fewer positions: the accuracy ladder')
    for a, (x, y) in zip(RUNGS, zip(lx, ly)):
        if a == 'clean':
            ax.text(x + 0.018, y, 'all positions', ha='left', va='center', **lab)
        else:
            ax.text(x, y - 0.022, RUNG_LABEL[a], ha='center', va='top', **lab)
    sx, sy = zip(*[mean(a) for a in SCRAMBLED])
    ax.scatter(sx, sy, s=(9 * PX) ** 2, color=c['s2'], edgecolors=c['surface'], linewidths=2 * PX, marker='D', zorder=4,
               label='Trained with scrambled answers')
    nx, ny = zip(*[mean(a) for a in NOSCRAMBLE])
    ax.scatter(nx, ny, s=(9 * PX) ** 2, color=c['s3'], edgecolors=c['surface'], linewidths=2 * PX, marker='s', zorder=4,
               label='Discounted genuine answers, no scrambled ones')
    note = {'rw-oracle': ('clean head judges, with scrambled answers (N16)', (0.19, 0.93), 'left'),
            'r20': ('10%, 20%, 40%\nscrambled (N14)', (0.03, 0.80), 'left'),
            'n80-mix': ('blended head (N12)', (0.47, 0.805), 'left'),
            'sub20-oracle': ('clean head judges,\ngenuine answers only (N16)', (0.40, 0.885), 'left'),
            'sub20-rw': ('judges itself,\ngenuine answers only (N14)', (0.66, 0.73), 'left')}
    for a, (txt, xy, ha) in note.items():
        ax.annotate(txt, mean(a), xy, ha=ha, va='center', linespacing=1.3,
                    arrowprops=dict(arrowstyle='-', color=c['muted'], linewidth=1 * PX, shrinkA=2, shrinkB=5), **lab)
    ax.set_xlim(-0.03, 0.92); ax.set_ylim(0.55, 1.0)
    ax.set_xlabel('How much worse than the clean head it predicts the genuine answers (nats)', color=c['ink2'], fontsize=9.5)
    ax.set_ylabel('Agreement with the clean head\'s directions', color=c['ink2'], fontsize=9.5)
    ax.legend(loc='lower left', frameon=False, fontsize=8.5, labelcolor=c['ink2'], handletextpad=0.4, borderaxespad=0.6)
    titles(fig, c, 'Heads trained with scrambled answers sit close to the accuracy ladder',
           'Second heads placed by the accuracy they reached; N14 and N16, same text, means of five runs.\nN14\'s primary rule was inconclusive: the gaps from the ladder were small, but within its noise.')
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
    ax.axhline(ref, color=c['muted'], linewidth=1 * PX, zorder=4)
    ax.text(-0.52, ref + 0.012, f'genuine answers only,\nat full weight: {ref:.2f}', color=c['ink2'], fontsize=8.5, ha='left', va='bottom', linespacing=1.3)
    ax.set_xticks([0, 1]); ax.set_xticklabels([g[1] for g in groups])
    ax.tick_params(axis='x', length=0, labelsize=9.5, labelcolor=c['ink'])
    ax.set_ylabel('Shortfall behind the clean head (nats)', color=c['ink2'], fontsize=9.5)
    ax.scatter([], [], marker='s', s=(9 * PX) ** 2, color=c['s3'], label='Genuine answers only, discounted')
    ax.scatter([], [], marker='s', s=(9 * PX) ** 2, color=c['s2'], label='Plus the scrambled answers')
    ax.legend(loc='upper right', frameon=False, fontsize=8.5, labelcolor=c['ink2'], handletextpad=0.3)
    titles(fig, c, 'The shortfall depends on who judges, and on the scrambled answers',
           'How far each discounted head falls behind the clean head on the genuine answers; lower is better.\nMeans of five runs (N16); an exploratory comparison, not one of N16\'s claims.')
    return fig


def write_data(dist, acc, tb):
    ladder, noisy, rand = dist
    L = ['# Figure data', '', 'Every value plotted in the plain summary\'s figures, by run. Generated by `make_figures.py` from the records in `results/`.', '',
         '## Figure 1: the distance ladder', '',
         f'Target-free overlap of the directions head B and the next-word head A depend on (N8). Random directions: {rand:.4f}.', '',
         '| Second reading | ' + ' | '.join(f'run {i + 1}' for i in range(5)) + ' | Mean |', '|---|' + '---|' * 6]
    for k in (1, 2, 4, 8):
        name = 'the same word' if k == 1 else f'{k} words ahead'
        L.append(f'| {name} (N6r models, `results/n8/f_n6r_*`) | ' + ' | '.join(f'{v:.3f}' for v in ladder[k]) + f' | {st.mean(ladder[k]):.3f} |')
    L.append('| the next word, 80% scrambled answers (N7 models, `results/n8/f_n7_noisy_*`) | ' + ' | '.join(f'{v:.3f}' for v in noisy) + f' | {st.mean(noisy):.3f} |')
    L += ['', '## Figure 2: the accuracy ladder', '',
          'Δ_A: how much worse than the clean head the head predicts the genuine answers, in nats; a: its agreement with the clean head\'s directions. Runs are seeds 4101–4105; ladder rungs and `sub20-rw` from `results/n16/r_*`, N16\'s heads from `results/n16/m_*`, N14\'s doses from `results/n14/m_*`, the blended head from `results/n14/r_n80-mix_*`.', '',
          '| Head | ' + ' | '.join(f'run {i + 1} (Δ_A, a)' for i in range(5)) + ' | Mean |', '|---|' + '---|' * 6]
    names = {'clean': 'every genuine answer', 'sub20': '20% genuine answers', 'sub10': '10%', 'sub05': '5%', 'sub025': '2.5%', 'sub0125': '1.25%',
             'r10': '10% scrambled + 20% genuine (N14)', 'r20': '20% scrambled + 20% genuine (N14)', 'r40': '40% scrambled + 20% genuine (N14)',
             'n80-mix': 'blended head, 80% scrambled (N12)', 'rw-oracle': 'discounted, clean head judges, 80% scrambled (N16)',
             'sub20-rw': 'discounted, judges itself, no scrambled (N14)', 'sub20-oracle': 'discounted, clean head judges, no scrambled (N16)'}
    for a in list(RUNGS) + list(SCRAMBLED) + list(NOSCRAMBLE):
        p = acc[a]
        L.append(f'| {names[a]} | ' + ' | '.join(f'({x:.3f}, {y:.3f})' for x, y in p) + f' | ({st.mean(x for x, _ in p):.3f}, {st.mean(y for _, y in p):.3f}) |')
    L += ['', '## Figure 3: N16\'s two-by-two', '', 'Δ_A in nats, seeds 4101–4105, from `results/n16/`.', '',
          '| Head | ' + ' | '.join(f'run {i + 1}' for i in range(5)) + ' | Mean |', '|---|' + '---|' * 6]
    for a, name in (('sub20-rw', 'judges itself, genuine answers only'), ('rw-b1', 'judges itself, plus scrambled answers'),
                    ('sub20-oracle', 'clean head judges, genuine answers only'), ('rw-oracle', 'clean head judges, plus scrambled answers'),
                    ('sub20', 'reference: genuine answers at full weight')):
        L.append(f'| {name} | ' + ' | '.join(f'{v:.3f}' for v in tb[a]) + f' | {st.mean(tb[a]):.3f} |')
    (OUT / 'FIGURE_DATA.md').write_text('\n'.join(L) + '\n')


def main():
    plt.rcParams['font.family'] = 'DejaVu Sans'
    OUT.mkdir(exist_ok=True)
    dist, acc, tb = distance_data(), accuracy_data(), twobytwo_data()
    for mode, c in THEMES.items():
        save(fig_distance(c, dist), 'distance_ladder', mode)
        save(fig_accuracy(c, acc), 'accuracy_ladder', mode)
        save(fig_twobytwo(c, tb), 'judge_two_by_two', mode)
    write_data(dist, acc, tb)
    print('wrote', sorted(p.name for p in OUT.iterdir()))


if __name__ == '__main__':
    main()
