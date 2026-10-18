"""N11: a payload strong enough to test, and a yardstick it cannot move.

Implements N11_PREREGISTRATION.md in this folder. N10's model, training, payloads and measurements
(n10_payload.py), with six arms on new seeds:
  clean      rho = 0, no payload (must-match control)
  n80        rho = 0.8, no payload (the yardstick: its head A's ideal noisy copy)
  n80-f8, n80-f32, n80-f128   rho = 0.8, functional payload at lambda = 8, 32, 128
  n80-geom   rho = 0.8, geometric payload at lambda = 10 (as N10)
Measured on WikiText-103 test-split windows 1793-2213. The share of the gap closed, S, is computed by
n11_analyze.py from the saved records: gap(arm) = head B's clean-target loss in that arm minus the clean-target
loss of the ideal copy of the same seed's n80 head A, and S = 1 - gap(arm) / gap(n80).

    python n11_strength.py --out OUT --work /content/drive/MyDrive/cdt-colab/n11_strength
    STUDY_TEST=1 python n11_strength.py --out OUT --work W --source LOCAL_STUDY_FOLDER   # small test
"""
import argparse
import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import n10_payload as n10  # noqa: E402  (imports N9, N8, N7, N6r, N6, N4, N2, G1 and the study)

n9, n8, n7, n6, n4, g, study, DEV, TEST, log = n10.n9, n10.n8, n10.n7, n10.n6, n10.n4, n10.g, n10.study, n10.DEV, n10.TEST, n10.log
SEEDS = [4001, 4002] if TEST else [4001, 4002, 4003, 4004, 4005]
n6.SEEDS = g.SEEDS = SEEDS
ARMS = {'clean': (0.0, None, None), 'n80': (0.8, None, None),
        'n80-f8': (0.8, 'func', 8.0), 'n80-f32': (0.8, 'func', 32.0), 'n80-f128': (0.8, 'func', 128.0),
        'n80-geom': (0.8, 'geom', 10.0)}
n10.ARMS = ARMS                                     # n10.measure and n10.train_n10 read rho and kind from here
MEAS = (80, 96) if TEST else (1792, 2213)           # test-split windows 1793-2213, the last 421


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
    log('device:', DEV, '| test mode:', TEST, '| study folder (data):', source, '| N11 folder:', work, '| local:', local)
    store.pull_dir('results')
    try:
        meta = study.prepare_data(data_store, None)
        train = study.load_tokens(store, meta, 'train')
        probs = n7.unigram(train)
        log_u = probs.float().log().to(DEV)
        payload = n10.payload_windows(train)
        summary_path = store.L('results/n11_summary.json')
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

        def save_json(rel, obj):
            store.L(rel).write_text(json.dumps(obj, indent=2 if rel.endswith('n11_summary.json') else None))
            store.push(rel)

        allx, allys = n6.windows(study.load_tokens(store, meta, 'test'), limit=MEAS[1])
        mx, mys = allx[MEAS[0]:MEAS[1]], [y[MEAS[0]:MEAS[1]] for y in allys]
        u = n8.uniforms(len(mx))
        log(f'measured on test-split windows {MEAS[0] + 1}-{MEAS[0] + len(mx)}')

        if 'checks' not in summary:
            enc = study.gpt2_encoding()
            summary['checks'] = {
                'batch': n6.batch_check(train), 'batch_g1': g.batch_check(train),
                'same_model': [g.same_model_check(train, s) for s in SEEDS],
                'test_windows_shifted': all(torch.equal(mys[k - 1][:, :-k], mx[:, k:]) for k in range(1, 9)),
                'analytic': n9.analytic_check(),
                'identical_heads': n8.identical_heads_check(SEEDS[0], mx, u),
                'payload_windows': {'count': len(payload[0]), 'split': 'train', 'seed': n10.PAYLOAD_SEED,
                                    'starts_first_5': payload[2][:5].tolist(),
                                    'targets_shifted': bool(torch.equal(payload[1][:, :-1], payload[0][:, 1:])),
                                    'first_text': enc.decode(payload[0][0, :20].tolist())},
                'unigram': {'sum': probs.sum().item(), 'zeros': int((probs == 0).sum())}}
            summary['windows'] = {'split': 'test', 'measure': [MEAS[0] + 1, MEAS[0] + len(mx)]}
            summary['arms'] = {a: {'rho': r, 'payload': k, 'lambda': lam} for a, (r, k, lam) in ARMS.items()}
            summary['seeds'], summary['readout_rank'], summary['subspace_rank'] = SEEDS, n4.RANK, n9.R
            summary['payload'] = {'windows': n10.N_PAYLOAD, 'per_step': n10.PAYLOAD_PER_STEP, 'seed': n10.PAYLOAD_SEED}
            save_json('results/n11_summary.json', summary)
        c = summary['checks']
        log('batch check:', c['batch']['targets_are_i+1_to_i+8'], '| test windows shifted:', c['test_windows_shifted'],
            '| analytic:', round(c['analytic']['sampled']['top_subspace_overlap'], 4), '| identical heads:', c['identical_heads']['agreement_ratio'],
            '| payload windows shifted:', c['payload_windows']['targets_shifted'])
        log('same-model check:', [x['pass'] for x in c['same_model']])
        if not (c['batch']['targets_are_i+1_to_i+8'] and c['test_windows_shifted'] and c['analytic']['pass'] and c['identical_heads']['pass']
                and c['payload_windows']['targets_shifted'] and all(x['pass'] for x in c['same_model'])):
            raise SystemExit('A check of the setup failed before training. Stopping.')

        for seed in SEEDS:
            for arm, (rho, kind, lam) in ARMS.items():
                tag = f'{arm}_seed{seed}'
                n10.train_n10(store, meta, arm, seed, tag, probs, log_u, payload, lam=lam)
                rel = f'results/m_{tag}.json'
                if store.L(rel).exists():
                    continue
                if not store.pull(f'final/{tag}.pt'):
                    raise SystemExit(f'weights final/{tag}.pt are missing locally and on Drive.')
                m = n4.new_n4(seed, n10.BASE)
                m.load_state_dict(torch.load(store.L(f'final/{tag}.pt'), map_location=DEV, weights_only=False))
                res = {'tag': tag, 'seed': seed, 'arm': arm, 'lambda': lam, **n10.measure(m, arm, seed, mx, mys, u, log_u, probs, payload)}
                save_json(rel, res)
                fn = res['functional']
                log(f"m_{tag}: head B clean {fn['B_clean']:.3f} | copy clean {fn['copy_clean']:.3f} | head A clean {fn['A_clean']:.3f} | "
                    f"B vs copy {res['B_vs_copy']['agreement_ratio']:.3f} | payload-window KL {res['payload_windows']['kl_B_to_copy']:.3f} | "
                    f"readouts A,B {res['readout_overlap_A_B']:.3f} | rotated {res['rotated_control']['agreement_ratio']:.3f} | {res['seconds']:.0f} s")
        log('N11 complete: all runs and measurements finished')
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
