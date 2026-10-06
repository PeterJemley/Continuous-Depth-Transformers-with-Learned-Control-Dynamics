"""N5: is N4's cost a competition for directions? A readout-width sweep.

Implements N5_PREREGISTRATION.md in this folder. N4's four arms (dual-sep, dual-tied, same-sep,
same-tied) at readout widths 256, 32 and 128, in that order, on N4's seeds. Everything except the width
is N4's code: this script sets n4_directions.RANK for each width and calls N4's functions.

    python n5_widths.py --out OUT --work /content/drive/MyDrive/cdt-colab/n5_widths
    STUDY_TEST=1 python n5_widths.py --out OUT --work W --source LOCAL_STUDY_FOLDER   # small test
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import n4_directions as n4  # noqa: E402  (imports the study, G1 and N2 code)

g, study, DEV, TEST, log = n4.g, n4.study, n4.DEV, n4.TEST, n4.log
SEEDS = n4.SEEDS
D_MODEL = study.CFG['d_model']
WIDTHS = [D_MODEL, D_MODEL // 8, D_MODEL // 2]      # 256, 32, 128 (test mode: 32, 4, 16); the full width first
RANK_TOL = 1e-4


def set_width(w):
    n4.RANK = w


def rank_check(seed):
    """Precondition 3: at full width the initial P has full rank (smallest singular value above
    RANK_TOL times the largest)."""
    set_width(D_MODEL)
    m = n4.new_n4(seed, 'dual-sep')
    s = torch.linalg.svdvals(m.proj_a.weight.detach().double())
    ratio = (s[-1] / s[0]).item()
    return {'seed': seed, 'width': D_MODEL, 'smallest_over_largest_singular_value': ratio, 'pass': ratio > RANK_TOL}


def untrained_overlap(seed, w, mx, my1, my2):
    """S4: the overlap of the two heads' subspaces on the untrained model, at width w."""
    set_width(w)
    out = {}
    for arm in ('dual-sep', 'same-sep'):
        Ua, Ub = n4.estimate_subspaces(n4.new_n4(seed, arm), mx, my1, my2, arm, n4.R)
        out[arm] = n4.n2.overlap(Ua, Ub)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--source', default=None)
    ap.add_argument('--local', default=None)
    args = ap.parse_args()
    work = Path(args.work)
    source = Path(args.source) if args.source else work.parent / 'wt103_study'
    if args.local:
        local = Path(args.local)
    elif Path('/content').is_dir() and not TEST:
        local = Path('/content/cdt_local') / work.name
    else:
        local = work.parent / f'_local_{work.name}'
    data_store = study.Store(source, local)
    store = study.Store(work, local)
    out = Path(args.out)
    (out / 'results').mkdir(parents=True, exist_ok=True)
    log('device:', DEV, '| test mode:', TEST, '| study folder (data):', source, '| N5 folder:', work, '| local:', local)
    store.pull_dir('results')
    try:
        meta = study.prepare_data(data_store, None)
        train = study.load_tokens(store, meta, 'train')
        summary_path = store.L('results/n5_summary.json')
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

        def save_summary():
            summary_path.write_text(json.dumps(summary, indent=2))
            store.push('results/n5_summary.json')

        val = study.load_tokens(store, meta, 'validation')
        mx, my1, my2 = (t[n4.MEAS[0]:n4.MEAS[1]] for t in g.g1_windows(val, limit=n4.MEAS[1]))

        if 'checks' not in summary:
            mm = []
            for w in WIDTHS:
                set_width(w)
                mm += [{'width': w, **n4.must_match_check(train, s)} for s in SEEDS]
            summary['checks'] = {'must_match': mm, 'full_rank': [rank_check(s) for s in SEEDS],
                                 'batch': n4.n2.batch_check_dual(train), 'batch_g1': g.batch_check(train),
                                 'same_model': [g.same_model_check(train, s) for s in SEEDS]}
            summary['widths'], summary['subspace_rank'] = WIDTHS, n4.R
            summary['windows'] = {'overlap': [n4.MEAS[0] + 1, n4.MEAS[1]]}
            save_summary()
        c = summary['checks']
        log('must-match check (separate = tied at step 0), every width and seed:', all(x['pass'] for x in c['must_match']))
        log('full-rank check at width', D_MODEL, ':', [round(x['smallest_over_largest_singular_value'], 4) for x in c['full_rank']])
        log('batch check, targets at i+1 and i+2:', c['batch']['targets_are_i+1_and_i+2'])
        log('same-model check:', [x['pass'] for x in c['same_model']])
        if not (all(x['pass'] for x in c['must_match']) and all(x['pass'] for x in c['full_rank'])
                and c['batch']['targets_are_i+1_and_i+2'] and all(x['pass'] for x in c['same_model'])):
            raise SystemExit('A check of the setup failed before training. Stopping.')

        if 'untrained_overlap' not in summary:
            summary['untrained_overlap'] = {f'w{w}_seed{s}': untrained_overlap(s, w, mx, my1, my2) for w in WIDTHS for s in SEEDS}
            save_summary()

        for w in WIDTHS:
            set_width(w)
            for seed in SEEDS:
                for arm in n4.ARMS:
                    tag = f'w{w}_{arm}_seed{seed}'
                    n4.train_n4(store, meta, arm, seed, tag)
                    rel = f'results/sub_{tag}.json'
                    if store.L(rel).exists():
                        continue
                    if not store.pull(f'final/{tag}.pt'):
                        raise SystemExit(f'weights final/{tag}.pt are missing locally and on Drive.')
                    m = n4.new_n4(seed, arm)
                    m.load_state_dict(torch.load(store.L(f'final/{tag}.pt'), map_location=DEV, weights_only=False))
                    res = {'tag': tag, 'seed': seed, 'arm': arm, 'width': w, **n4.measure(m, arm, mx, my1, my2)}
                    store.L(rel).write_text(json.dumps(res))
                    store.push(rel)
                    log(f"sub_{tag}: overlap of A and B {res['overlap_AB']:.3f}, readout overlap {res['readout_overlap']:.3f}")
            log(f'width {w} complete')
        log('N5 complete: all runs and measurements finished')
    finally:
        dest = out / 'results'
        for f in (store.local / 'results').glob('*.json'):
            (dest / f.name).write_bytes(f.read_bytes())
        log(f'copied {len(list(dest.glob("*.json")))} result files for upload')
        log('waiting for background Drive saves to finish (up to 15 minutes)')
        store.flush()
        failures = store.failures + data_store.failures
        (out / 'drive_sync.json').write_text(json.dumps({'failures': failures}, indent=2))
        log('Drive saves finished' if not failures else f'Drive problems this session: {failures}')


if __name__ == '__main__':
    main()
