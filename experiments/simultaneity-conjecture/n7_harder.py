"""N7: does sharing track how different the readings are, or how hard?

Implements N7_PREREGISTRATION.md in this folder. N6r's model, training and measure (n6_halfoverlap.py,
n6r_halfoverlap.py), with four arms. Head A always predicts token i+1; head B predicts:
  same   token i+1 (must-fail control);
  k4     token i+4;
  k8     token i+8;
  noisy  token i+1, with each target replaced, with probability 0.8 (amendment A1), by a token drawn from the training
         data's unigram distribution.
The training noise is drawn per step from a generator seeded by the run's seed and the step, so a resumed
run draws the same noise. Validation and measurement targets are corrupted once, with fixed seeds.
The overlap is measured on WikiText-103 test-split windows 257-512.

    python n7_harder.py --out OUT --work /content/drive/MyDrive/cdt-colab/n7_harder
    STUDY_TEST=1 python n7_harder.py --out OUT --work W --source LOCAL_STUDY_FOLDER   # small test
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import n6r_halfoverlap as n6r  # noqa: E402  (imports N6, N4, N2, G1 and the study)

n6, n4, g, study, DEV, TEST, AMP, log = n6r.n6, n6r.n4, n6r.g, n6r.study, n6r.DEV, n6r.TEST, n6r.n6.AMP, n6r.log
SEQ, BATCH, VOCAB = g.SEQ, g.BATCH, g.VOCAB
SEEDS = [3801, 3802] if TEST else [3801, 3802, 3803, 3804, 3805]
n6.SEEDS = g.SEEDS = SEEDS
ARMS = {'same': (1, False), 'k4': (4, False), 'k8': (8, False), 'noisy': (1, True)}
NOISE = 0.8                                          # amendment A1 (was 0.6)
TOTAL, PEAK_LR, BASE = n6.TOTAL, n6.PEAK_LR, n6.BASE
MEAS = (16, 32) if TEST else (256, 512)             # test-split windows 257-512
VAL_NOISE_SEED, MEAS_NOISE_SEED = 271828, 314159


# ------------------------------------------------------------------ targets and noise

def unigram(train):
    """The training token file's unigram distribution, as a tensor for sampling."""
    counts = np.zeros(VOCAB, dtype=np.float64)
    for i in range(0, len(train), 10_000_000):                     # in chunks, to keep memory small
        counts += np.bincount(np.asarray(train[i:i + 10_000_000], dtype=np.int64), minlength=VOCAB)
    return torch.from_numpy(counts / counts.sum())


def corrupt(y, probs, gen):
    """Replace each target, independently with probability NOISE, by a draw from probs."""
    mask = torch.rand(y.shape, generator=gen) < NOISE
    draws = torch.multinomial(probs.float(), y.numel(), replacement=True, generator=gen).view(y.shape)
    return torch.where(mask, draws, y), mask


def pair(arm, ys):
    """Head A's and head B's clean targets for an arm (the noise, if any, is applied separately)."""
    return ys[0], ys[ARMS[arm][0] - 1]


def fixed_targets(arm, ys, probs, seed):
    """Head A's and head B's targets for validation or measurement, with the noisy arm corrupted once."""
    ya, yb = pair(arm, ys)
    if ARMS[arm][1]:
        yb, _ = corrupt(yb, probs, torch.Generator().manual_seed(seed))
    return ya, yb


# ------------------------------------------------------------------ training

def train_n7(store, meta, arm, seed, tag, probs):
    """One run, resumable. As n6.train_n6, with head B's targets corrupted in the noisy arm."""
    res_path = store.L(f'results/{tag}.json')
    if res_path.exists():
        log(f'{tag}: already finished, skipping')
        return json.loads(res_path.read_text())
    ck_rel = f'ckpt/{tag}.pt'
    store.pull(ck_rel)
    ck_path = store.L(ck_rel)
    train = study.load_tokens(store, meta, 'train')
    vx, vys = n6.windows(study.load_tokens(store, meta, 'validation'), limit=g.VAL_LIMIT)
    vya, vyb = fixed_targets(arm, vys, probs, VAL_NOISE_SEED)
    model = n4.new_n4(seed, BASE)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR)
    scaler = torch.amp.GradScaler('cuda', enabled=AMP)
    gen = torch.Generator().manual_seed(seed)
    state = {'step': 0, 'train_loss': [], 'val': [], 'alpha': [], 'skipped_steps': 0, 'seconds': 0.0, 'resumes': 0}
    if ck_path.exists():
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); scaler.load_state_dict(ck['scaler'])
        gen.set_state(ck['gen'])
        torch.set_rng_state(ck['torch_rng'])
        if DEV.type == 'cuda' and ck.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
        state = ck['state']; state['resumes'] += 1
        log(f"{tag}: resuming at step {state['step']}")
    else:
        log(f"{tag}: starting, head B at i+{ARMS[arm][0]}{' with noise' if ARMS[arm][1] else ''}, seed {seed}")

    def save_ckpt():
        tmp = ck_path.with_suffix('.tmp')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(),
                    'gen': gen.get_state(), 'state': state, 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if DEV.type == 'cuda' else None}, tmp)
        os.replace(tmp, ck_path)
        store.push(ck_rel)

    model.train()
    last_ck, t_seg = time.time(), time.time()
    while state['step'] < TOTAL:
        step = state['step']
        for grp in opt.param_groups:
            grp['lr'] = study.lr_at(step, TOTAL, PEAK_LR)
        x, ys = n6.draw_batch(train, gen)
        if step == 0:
            state['batch_fingerprint_step0'] = int((x.long() * torch.arange(1, x.numel() + 1).view_as(x)).sum())
        ya, yb = pair(arm, ys)
        if ARMS[arm][1]:
            yb, mask = corrupt(yb, probs, torch.Generator().manual_seed(seed * 100003 + step))
            if step == 0:
                state['noise_fraction_step0'] = mask.float().mean().item()
        x, ya, yb = x.to(DEV), ya.to(DEV), yb.to(DEV)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=AMP):
            la, lb = n4.forward_n4(model, x)
            loss = 0.5 * (F.cross_entropy(la.view(-1, VOCAB), ya.reshape(-1)) + F.cross_entropy(lb.view(-1, VOCAB), yb.reshape(-1)))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if AMP and scaler.get_scale() < before:
            state['skipped_steps'] += 1
        state['step'] = step + 1
        if step % 50 == 0:
            state['train_loss'].append([step, loss.item()])
            state['alpha'].append([step, model.ode_block.func.output_scale.item()])
        if state['step'] == 500 or state['step'] % g.VAL_EVERY == 0 or state['step'] == TOTAL:
            l_a, l_b = n6.losses(model, vx, vya, vyb)
            state['val'].append([state['step'], l_a, l_b])
            log(f"{tag}: step {state['step']}/{TOTAL}, validation loss head A {l_a:.4f}, head B {l_b:.4f}")
        if time.time() - last_ck > study.CKPT_SECONDS:
            state['seconds'] += time.time() - t_seg; t_seg = time.time()
            save_ckpt(); last_ck = time.time()
    state['seconds'] += time.time() - t_seg
    save_ckpt()
    torch.save(model.state_dict(), store.L(f'final/{tag}.pt'))
    store.push(f'final/{tag}.pt')
    fx, fys = n6.windows(study.load_tokens(store, meta, 'validation'), limit=g.FINAL_LIMIT)
    fya, fyb = fixed_targets(arm, fys, probs, VAL_NOISE_SEED)
    l_a, l_b = n6.losses(model, fx, fya, fyb)
    l_b_clean = n6.losses(model, fx, fya, pair(arm, fys)[1])[1] if ARMS[arm][1] else l_b
    result = {'tag': tag, 'arm': arm, 'k': ARMS[arm][0], 'noisy': ARMS[arm][1], 'noise_rate': NOISE if ARMS[arm][1] else 0.0,
              'seed': seed, 'readout_rank': n4.RANK,
              'final_loss_A': l_a, 'final_loss_B': l_b, 'final_loss_B_clean_targets': l_b_clean,
              'final_val_windows': len(fx), 'val_curve': state['val'],
              'train_loss_every_50': state['train_loss'], 'alpha_every_50': state['alpha'],
              'alpha_final': model.ode_block.func.output_scale.item(),
              'batch_fingerprint_step0': state.get('batch_fingerprint_step0'),
              'noise_fraction_step0': state.get('noise_fraction_step0'),
              'skipped_steps_fp16': state['skipped_steps'], 'train_seconds': state['seconds'], 'resumes': state['resumes'],
              'n_params': sum(p.numel() for p in model.parameters()), 'environment': study.env_record()}
    res_path.write_text(json.dumps(result, indent=2))
    store.push(f'results/{tag}.json')
    log(f'{tag}: finished in {state["seconds"] / 3600:.2f} h; head A {l_a:.4f}, head B {l_b:.4f} (clean targets {l_b_clean:.4f})')
    return result


# ------------------------------------------------------------------ checks

def noise_check(train, probs):
    """One batch's noisy targets: the fraction replaced, and how the replaced tokens compare with the unigram."""
    x, ys = n6.draw_batch(train, torch.Generator().manual_seed(SEEDS[0]))
    y = ys[0]
    yn, mask = corrupt(y, probs, torch.Generator().manual_seed(SEEDS[0] * 100003))
    enc = study.gpt2_encoding()
    top = torch.topk(probs, 5).indices.tolist()
    return {'fraction_replaced': mask.float().mean().item(), 'fraction_unchanged': (yn == y).float().mean().item(),
            'clean_end': enc.decode(y[0, -10:].tolist()), 'noisy_end': enc.decode(yn[0, -10:].tolist()),
            'most_frequent_tokens': [enc.decode([t]) for t in top], 'pass': abs(mask.float().mean().item() - NOISE) < 0.05}


def untrained(seed, mx, mys, probs):
    """S4 and calibration: every arm on the untrained model, with the noisy arm's fixed noisy targets."""
    out = {}
    for arm in ARMS:
        ya, yb = fixed_targets(arm, mys, probs, MEAS_NOISE_SEED)
        r = n6r.measure(n4.new_n4(seed, BASE), mx, ya, yb)
        out[arm] = {k: r[k] for k in ('overlap_AB', 'stability_A_odd_even', 'stability_B_odd_even',
                                      'cross_half_overlap_AB', 'agreement_ratio')}
    return out


# ------------------------------------------------------------------ main

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
    log('device:', DEV, '| test mode:', TEST, '| study folder (data):', source, '| N7 folder:', work, '| local:', local)
    store.pull_dir('results')
    try:
        meta = study.prepare_data(data_store, None)
        train = study.load_tokens(store, meta, 'train')
        probs = unigram(train)
        summary_path = store.L('results/n7_summary.json')
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

        def save_json(rel, obj):
            store.L(rel).write_text(json.dumps(obj, indent=2 if rel.endswith('n7_summary.json') else None))
            store.push(rel)

        allx, allys = n6.windows(study.load_tokens(store, meta, 'test'), limit=MEAS[1])
        mx, mys = allx[MEAS[0]:MEAS[1]], [y[MEAS[0]:MEAS[1]] for y in allys]
        log(f'overlap measured on test-split windows {MEAS[0] + 1}-{MEAS[1]}')

        if 'checks' not in summary:
            summary['checks'] = {'batch': n6.batch_check(train), 'batch_g1': g.batch_check(train),
                                 'same_model': [g.same_model_check(train, s) for s in SEEDS],
                                 'test_windows_shifted': all(torch.equal(mys[k - 1][:, :-k], mx[:, k:]) for k in range(1, 9)),
                                 'noise': noise_check(train, probs),
                                 'identical_heads': n6r.identical_heads_check(SEEDS[0], mx, mys)}
            summary['windows'] = {'split': 'test', 'measure': [MEAS[0] + 1, MEAS[1]]}
            summary['noise_rate'], summary['readout_rank'], summary['subspace_rank'] = NOISE, n4.RANK, n6.R
            summary['arms'] = {a: {'k': k, 'noisy': nz} for a, (k, nz) in ARMS.items()}
            save_json('results/n7_summary.json', summary)
        c = summary['checks']
        log('batch check:', c['batch']['targets_are_i+1_to_i+8'], '| test windows shifted:', c['test_windows_shifted'],
            '| noise fraction:', round(c['noise']['fraction_replaced'], 3), '| identical heads:', c['identical_heads']['agreement_ratio'])
        log('same-model check:', [x['pass'] for x in c['same_model']])
        if not (c['batch']['targets_are_i+1_to_i+8'] and c['test_windows_shifted'] and c['noise']['pass']
                and c['identical_heads']['pass'] and all(x['pass'] for x in c['same_model'])):
            raise SystemExit('A check of the setup failed before training. Stopping.')

        if 'untrained' not in summary:
            summary['untrained'] = {f'seed{s}': untrained(s, mx, mys, probs) for s in SEEDS}
            save_json('results/n7_summary.json', summary)

        for seed in SEEDS:
            for arm in ARMS:
                tag = f'{arm}_seed{seed}'
                train_n7(store, meta, arm, seed, tag, probs)
                rel = f'results/sub_{tag}.json'
                if store.L(rel).exists():
                    continue
                if not store.pull(f'final/{tag}.pt'):
                    raise SystemExit(f'weights final/{tag}.pt are missing locally and on Drive.')
                m = n4.new_n4(seed, BASE)
                m.load_state_dict(torch.load(store.L(f'final/{tag}.pt'), map_location=DEV, weights_only=False))
                ya, yb = fixed_targets(arm, mys, probs, MEAS_NOISE_SEED)
                res = {'tag': tag, 'seed': seed, 'arm': arm, 'k': ARMS[arm][0], 'noisy': ARMS[arm][1], **n6r.measure(m, mx, ya, yb)}
                save_json(rel, res)
                log(f"sub_{tag}: overlap {res['overlap_AB']:.3f}, agreement ratio {res['agreement_ratio']:.3f}, "
                    f"stability A {res['stability_A_odd_even']:.3f}, B {res['stability_B_odd_even']:.3f}")
        log('N7 complete: all runs and measurements finished')
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
