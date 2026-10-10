"""N6r: the half-overlap test, with its must-fail control set for its measurement.

Implements N6R_PREREGISTRATION.md in this folder. Everything is N6's (n6_halfoverlap.py): the model, the four
arms (head B at i+1, i+2, i+4, i+8), the training and the overlap measure. Three things change:
  - the seeds (3701-3705);
  - the overlap is measured on 256 windows of the WikiText-103 test split;
  - the must-fail control uses the agreement ratio: head A's subspace from one half of the windows against
    head B's from the other half, averaged both ways, divided by the geometric mean of the heads' own
    split-half stabilities. Two heads reading the same directions give about 1.

    python n6r_halfoverlap.py --out OUT --work /content/drive/MyDrive/cdt-colab/n6r_halfoverlap
    STUDY_TEST=1 python n6r_halfoverlap.py --out OUT --work W --source LOCAL_STUDY_FOLDER   # small test
"""
import argparse
import json
import math
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import n6_halfoverlap as n6  # noqa: E402  (imports N4, N2, G1 and the study)

n4, g, study, DEV, TEST, log = n6.n4, n6.g, n6.study, n6.DEV, n6.TEST, n6.log
SEEDS = [3701, 3702] if TEST else [3701, 3702, 3703, 3704, 3705]
n6.SEEDS = g.SEEDS = SEEDS
ARMS, R, BASE = n6.ARMS, n6.R, n6.BASE
N_MEAS = 32 if TEST else 256                       # test-split windows 1-256


def overlap(U, V):
    return n4.n2.overlap(U, V)


def measure(model, mx, ya, yb):
    """N6's overlap on all windows, each head's split-half stability, and the agreement ratio from
    independent halves (odd and even windows)."""
    Ua, Ub = n4.estimate_subspaces(model, mx, ya, yb, BASE, R)
    Uao, Ubo = n4.estimate_subspaces(model, mx[0::2], ya[0::2], yb[0::2], BASE, R)
    Uae, Ube = n4.estimate_subspaces(model, mx[1::2], ya[1::2], yb[1::2], BASE, R)
    stab_a, stab_b = overlap(Uao, Uae), overlap(Ubo, Ube)
    cross = 0.5 * (overlap(Uao, Ube) + overlap(Uae, Ubo))
    return {'overlap_AB': overlap(Ua, Ub), 'stability_A_odd_even': stab_a, 'stability_B_odd_even': stab_b,
            'cross_half_overlap_AB': cross, 'agreement_ratio': cross / math.sqrt(stab_a * stab_b),
            'readout_overlap': n4.readout_overlap(model), 'rank': R, 'readout_rank': n4.RANK,
            'subspaces': {'A': Ua.cpu().tolist(), 'B': Ub.cpu().tolist()}}


def identical_heads_check(seed, mx, mys):
    """The control's upper end: a model whose two heads have identical weights must give an agreement
    ratio of exactly 1."""
    m = n4.new_n4(seed, BASE)
    with torch.no_grad():
        m.head2.weight.copy_(m.head.weight)
        m.proj_b.weight.copy_(m.proj_a.weight)
    r = measure(m, mx, *n6.pair('same', mys))
    return {'seed': seed, 'agreement_ratio': r['agreement_ratio'], 'overlap_AB': r['overlap_AB'],
            'pass': abs(r['agreement_ratio'] - 1) < 1e-9}


def untrained(seed, mx, mys):
    """S4, and the control's lower end: every arm on the untrained model."""
    out = {}
    for arm in ARMS:
        r = measure(n4.new_n4(seed, BASE), mx, *n6.pair(arm, mys))
        out[arm] = {k: r[k] for k in ('overlap_AB', 'stability_A_odd_even', 'stability_B_odd_even',
                                      'cross_half_overlap_AB', 'agreement_ratio')}
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
    log('device:', DEV, '| test mode:', TEST, '| study folder (data):', source, '| N6r folder:', work, '| local:', local)
    store.pull_dir('results')
    try:
        meta = study.prepare_data(data_store, None)
        train = study.load_tokens(store, meta, 'train')
        summary_path = store.L('results/n6r_summary.json')
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

        def save_json(rel, obj):
            store.L(rel).write_text(json.dumps(obj, indent=2 if rel.endswith('n6r_summary.json') else None))
            store.push(rel)

        mx, mys = n6.windows(study.load_tokens(store, meta, 'test'), limit=N_MEAS)
        log(f'overlap measured on test-split windows 1-{len(mx)}')

        if 'checks' not in summary:
            enc = study.gpt2_encoding()
            summary['checks'] = {'batch': n6.batch_check(train), 'batch_g1': g.batch_check(train),
                                 'same_model': [g.same_model_check(train, s) for s in SEEDS],
                                 'test_windows': {'input_end': enc.decode(mx[0, -10:].tolist()),
                                                  **{f'target_i+{k}_end': enc.decode(mys[k - 1][0, -10:].tolist()) for k in (1, 2, 4, 8)},
                                                  'shifted': all(torch.equal(mys[k - 1][:, :-k], mx[:, k:]) for k in range(1, 9))},
                                 'identical_heads': identical_heads_check(SEEDS[0], mx, mys)}
            summary['windows'] = {'split': 'test', 'measure': [1, len(mx)]}
            summary['readout_rank'], summary['subspace_rank'], summary['arms'] = n4.RANK, R, ARMS
            save_json('results/n6r_summary.json', summary)
        c = summary['checks']
        log('batch check, targets at i+1 to i+8:', c['batch']['targets_are_i+1_to_i+8'], '| test windows shifted correctly:', c['test_windows']['shifted'])
        log('same-model check:', [x['pass'] for x in c['same_model']])
        log('identical-heads check, agreement ratio:', c['identical_heads']['agreement_ratio'])
        if not (c['batch']['targets_are_i+1_to_i+8'] and c['test_windows']['shifted'] and c['identical_heads']['pass']
                and all(x['pass'] for x in c['same_model'])):
            raise SystemExit('A check of the setup failed before training. Stopping.')

        if 'untrained' not in summary:
            summary['untrained'] = {f'seed{s}': untrained(s, mx, mys) for s in SEEDS}
            save_json('results/n6r_summary.json', summary)

        for seed in SEEDS:
            for arm in ARMS:
                tag = f'{arm}_seed{seed}'
                n6.train_n6(store, meta, arm, seed, tag)
                rel = f'results/sub_{tag}.json'
                if store.L(rel).exists():
                    continue
                if not store.pull(f'final/{tag}.pt'):
                    raise SystemExit(f'weights final/{tag}.pt are missing locally and on Drive.')
                m = n4.new_n4(seed, BASE)
                m.load_state_dict(torch.load(store.L(f'final/{tag}.pt'), map_location=DEV, weights_only=False))
                res = {'tag': tag, 'seed': seed, 'arm': arm, 'k': ARMS[arm], **measure(m, mx, *n6.pair(arm, mys))}
                save_json(rel, res)
                log(f"sub_{tag}: overlap {res['overlap_AB']:.3f}, agreement ratio {res['agreement_ratio']:.3f}, "
                    f"stability A {res['stability_A_odd_even']:.3f}, B {res['stability_B_odd_even']:.3f}")
        log('N6r complete: all runs and measurements finished')
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
