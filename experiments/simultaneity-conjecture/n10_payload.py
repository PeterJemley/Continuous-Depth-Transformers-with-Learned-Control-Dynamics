"""N10: how much noise changes what a head learns, and which payload preserves it.

Implements N10_PREREGISTRATION.md in this folder. N7's model and training (n7_harder.py): head A predicts the
next token; head B predicts the next token with a fraction rho of its targets replaced by draws from the
training data's unigram u. Seven arms:
  clean      rho = 0                    (must-match control)
  n20, n50, n80   rho = 0.2, 0.5, 0.8
  n80-func   rho = 0.8, plus a functional payload: on 64 fixed training windows, KL(copy_rho(A) || head B),
             where copy_rho(A) = (1 - rho) p_A + rho u is head A's ideal noisy copy (n9_noisycopy), not
             differentiated through
  n80-geom   rho = 0.8, plus a geometric payload: ||P_B - P_A||^2 / ||P_A||^2, P_A not differentiated through
  n80-data   rho = 0.8, plus a data payload: head B's cross-entropy on the clean targets of the 64 windows
Loss = 0.5 (loss_A + loss_B) + 0.5 * lambda * payload term. Each step uses 4 of the 64 payload windows in turn.
Each model is then measured on WikiText-103 test-split windows 1281-1792:
  - geometric: N9's target-free agreement ratio of head B (and of head A) with the copy at the arm's rho;
  - functional: G = head B's loss on the clean targets minus the copy's, and KL between them;
  - the controls (zero-noise copy, rotated readout) and the payload manipulation checks.

    python n10_payload.py --out OUT --work /content/drive/MyDrive/cdt-colab/n10_payload
    STUDY_TEST=1 python n10_payload.py --out OUT --work W --source LOCAL_STUDY_FOLDER   # small test
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import n9_noisycopy as n9  # noqa: E402  (imports N8, N7, N6r, N6, N4, N2, G1 and the study)

n8, n7, n6, n4, g, study, DEV, TEST, log = n9.n8, n9.n7, n9.n6, n9.n4, n9.g, n9.study, n9.DEV, n9.TEST, n9.log
AMP = n6.AMP
SEQ, VOCAB, R, BASE = g.SEQ, g.VOCAB, n9.R, n8.BASE
SEEDS = [3901, 3902] if TEST else [3901, 3902, 3903, 3904, 3905]
n6.SEEDS = g.SEEDS = SEEDS
ARMS = {'clean': (0.0, None), 'n20': (0.2, None), 'n50': (0.5, None), 'n80': (0.8, None),
        'n80-func': (0.8, 'func'), 'n80-geom': (0.8, 'geom'), 'n80-data': (0.8, 'data')}
LAMBDA = {'func': 1.0, 'data': 1.0, 'geom': 10.0}
N_PAYLOAD, PAYLOAD_SEED, PAYLOAD_PER_STEP = 64, 4242, 4
TOTAL, PEAK_LR = n6.TOTAL, n6.PEAK_LR
MEAS = (64, 80) if TEST else (1280, 1792)           # test-split windows 1281-1792
VAL_NOISE_SEED, MEAS_NOISE_SEED = n7.VAL_NOISE_SEED, n7.MEAS_NOISE_SEED
BATCH = n9.BATCH


# ------------------------------------------------------------------ targets, noise and payload windows

def corrupt(y, probs, gen, rho):
    """Replace each target, independently with probability rho, by a draw from probs."""
    if rho == 0:
        return y, torch.zeros(y.shape, dtype=torch.bool)
    mask = torch.rand(y.shape, generator=gen) < rho
    draws = torch.multinomial(probs.float(), y.numel(), replacement=True, generator=gen).view(y.shape)
    return torch.where(mask, draws, y), mask


def fixed_noisy(y, probs, rho, seed):
    return corrupt(y, probs, torch.Generator().manual_seed(seed), rho)[0]


def payload_windows(train):
    """The 64 payload windows: inputs and their clean next-token targets, from the training split, fixed seed."""
    ix = torch.randint(len(train) - SEQ - 1, (N_PAYLOAD,), generator=torch.Generator().manual_seed(PAYLOAD_SEED)).numpy()
    b = torch.from_numpy(np.stack([np.asarray(train[i:i + SEQ + 1], dtype=np.int64) for i in ix]))
    return b[:, :SEQ], b[:, 1:SEQ + 1], ix


def kl(log_p, log_q):
    """Mean over positions of KL(p || q), from log-probabilities over the vocabulary."""
    return (log_p.exp() * (log_p - log_q)).sum(-1).mean()


def payload_term(model, kind, px, py, log_u, rho):
    if kind == 'geom':
        pa = model.proj_a.weight.detach()
        return (model.proj_b.weight - pa).pow(2).sum() / pa.pow(2).sum()
    la, lb = n4.forward_n4(model, px)
    if kind == 'data':
        return F.cross_entropy(lb.float().reshape(-1, VOCAB), py.reshape(-1))
    lc = n9.copy_logits(la.detach(), log_u, rho)                     # the copy, not differentiated through
    return kl(lc, torch.log_softmax(lb.float(), -1))


def final_state(model, x):
    """The normalised final state ln_f(z) that both readouts read (as n4.forward_n4, up to the readouts)."""
    T = x.shape[1]
    h = model.tok_emb(x) + model.pos_emb(torch.arange(T, device=x.device))
    for b in model.early_blocks:
        h = b(h)
    z = model.ode_block(h, None)
    for b in model.late_blocks:
        z = b(z)
    return model.ln_f(z)


def readout_payload_grads(model, px, log_u, rho, lam):
    """N11 amendment A1: the functional payload for head B's readout only. The shared state and head A's copy are
    computed without gradient, so the payload cannot reach the trunk or head A. Returns the payload term's
    gradients for head B's readout, unclipped so that lam sets the payload's weight (Adam bounds each update),
    and the term's value."""
    with torch.no_grad():
        z = final_state(model, px).float()
        lc = n9.copy_logits(model.head(model.proj_a(z)), log_u, rho)
    params = [model.proj_b.weight, model.head2.weight]
    term = kl(lc, torch.log_softmax(model.head2(model.proj_b(z)).float(), -1))
    grads = torch.autograd.grad(0.5 * lam * term, params)
    return params, list(grads), term.detach()


# ------------------------------------------------------------------ training

def train_n10(store, meta, arm, seed, tag, probs, log_u, payload, lam=None):
    """One run, resumable. As n7.train_n7, with head B's noise rate and payload set by the arm. lam overrides
    the payload strength (N11); by default it is LAMBDA[kind], as in N10."""
    rho, kind = ARMS[arm][:2]
    lam = LAMBDA.get(kind) if lam is None else lam
    res_path = store.L(f'results/{tag}.json')
    if res_path.exists():
        log(f'{tag}: already finished, skipping')
        return json.loads(res_path.read_text())
    ck_rel = f'ckpt/{tag}.pt'
    store.pull(ck_rel)
    ck_path = store.L(ck_rel)
    train = study.load_tokens(store, meta, 'train')
    vx, vys = n6.windows(study.load_tokens(store, meta, 'validation'), limit=g.VAL_LIMIT)
    vya, vyb = vys[0], fixed_noisy(vys[0], probs, rho, VAL_NOISE_SEED)
    px, py = payload[0].to(DEV), payload[1].to(DEV)
    model = n4.new_n4(seed, BASE)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR)
    scaler = torch.amp.GradScaler('cuda', enabled=AMP)
    gen = torch.Generator().manual_seed(seed)
    state = {'step': 0, 'train_loss': [], 'payload_term': [], 'val': [], 'alpha': [], 'skipped_steps': 0, 'seconds': 0.0, 'resumes': 0}
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
        log(f'{tag}: starting, noise {rho}, payload {kind}, seed {seed}')

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
        ya, yb = ys[0], ys[0]
        if rho > 0:
            yb, mask = corrupt(yb, probs, torch.Generator().manual_seed(seed * 100003 + step), rho)
            if step == 0:
                state['noise_fraction_step0'] = mask.float().mean().item()
        x, ya, yb = x.to(DEV), ya.to(DEV), yb.to(DEV)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=AMP):
            la, lb = n4.forward_n4(model, x)
            loss = 0.5 * (F.cross_entropy(la.view(-1, VOCAB), ya.reshape(-1)) + F.cross_entropy(lb.view(-1, VOCAB), yb.reshape(-1)))
            pterm = None
            if kind is not None and kind != 'func-readout':
                j = (PAYLOAD_PER_STEP * step) % N_PAYLOAD
                pterm = payload_term(model, kind, px[j:j + PAYLOAD_PER_STEP], py[j:j + PAYLOAD_PER_STEP], log_u, rho)
                loss = loss + 0.5 * lam * pterm
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        if kind == 'func-readout':                                        # N11 amendment A1: added after clipping the main loss
            j = (PAYLOAD_PER_STEP * step) % N_PAYLOAD
            params, grads, pterm = readout_payload_grads(model, px[j:j + PAYLOAD_PER_STEP], log_u, rho, lam)
            for prm, gr in zip(params, grads):
                prm.grad.add_(gr)
        before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if AMP and scaler.get_scale() < before:
            state['skipped_steps'] += 1
        state['step'] = step + 1
        if step % 50 == 0:
            state['train_loss'].append([step, loss.item()])
            state['alpha'].append([step, model.ode_block.func.output_scale.item()])
            if pterm is not None:
                state['payload_term'].append([step, pterm.item()])
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
    l_a, l_b = n6.losses(model, fx, fys[0], fixed_noisy(fys[0], probs, rho, VAL_NOISE_SEED))
    l_b_clean = n6.losses(model, fx, fys[0], fys[0])[1]
    result = {'tag': tag, 'arm': arm, 'rho': rho, 'payload': kind, 'lambda': lam, 'seed': seed, 'readout_rank': n4.RANK,
              'final_loss_A': l_a, 'final_loss_B': l_b, 'final_loss_B_clean_targets': l_b_clean,
              'final_val_windows': len(fx), 'val_curve': state['val'], 'train_loss_every_50': state['train_loss'],
              'payload_term_every_50': state['payload_term'], 'alpha_every_50': state['alpha'],
              'alpha_final': model.ode_block.func.output_scale.item(),
              'batch_fingerprint_step0': state.get('batch_fingerprint_step0'), 'noise_fraction_step0': state.get('noise_fraction_step0'),
              'skipped_steps_fp16': state['skipped_steps'], 'train_seconds': state['seconds'], 'resumes': state['resumes'],
              'n_params': sum(p.numel() for p in model.parameters()), 'environment': study.env_record()}
    res_path.write_text(json.dumps(result, indent=2))
    store.push(f'results/{tag}.json')
    log(f'{tag}: finished in {state["seconds"] / 3600:.2f} h; head A {l_a:.4f}, head B {l_b:.4f} (clean targets {l_b_clean:.4f})')
    return result


# ------------------------------------------------------------------ measurement

@torch.no_grad()
def functional(model, xs, y_clean, y_noisy, log_u, rho, batch=BATCH):
    """Losses of head A, head B and the copy on clean and noisy targets, and the KL divergence both ways
    between head B and the copy, per position."""
    model.eval()
    tot = {k: 0.0 for k in ('A_clean', 'B_clean', 'copy_clean', 'B_noisy', 'copy_noisy', 'kl_B_to_copy', 'kl_copy_to_B')}
    cnt = 0
    for i in range(0, len(xs), batch):
        x, yc, yn = xs[i:i + batch].to(DEV), y_clean[i:i + batch].to(DEV), y_noisy[i:i + batch].to(DEV)
        la, lb = n4.forward_n4(model, x)
        lpa, lpb, lpc = torch.log_softmax(la.float(), -1), torch.log_softmax(lb.float(), -1), n9.copy_logits(la, log_u, rho)
        n = yc.numel()
        tot['A_clean'] -= lpa.gather(-1, yc.unsqueeze(-1)).sum().item()
        tot['B_clean'] -= lpb.gather(-1, yc.unsqueeze(-1)).sum().item()
        tot['copy_clean'] -= lpc.gather(-1, yc.unsqueeze(-1)).sum().item()
        tot['B_noisy'] -= lpb.gather(-1, yn.unsqueeze(-1)).sum().item()
        tot['copy_noisy'] -= lpc.gather(-1, yn.unsqueeze(-1)).sum().item()
        tot['kl_B_to_copy'] += kl(lpb, lpc).item() * n
        tot['kl_copy_to_B'] += kl(lpc, lpb).item() * n
        cnt += n
    out = {k: v / cnt for k, v in tot.items()}
    out['G'] = out['B_clean'] - out['copy_clean']
    return out


def measure(model, arm, seed, mx, mys, u, log_u, probs, payload):
    rho = ARMS[arm][0]
    t0 = time.time()
    rhos = sorted({0.0, rho})
    G = n9.head_covs(model, mx, u, n9.head_fns(log_u, rhos))
    cp = n9.copy_name(rho)
    res = {'rho': rho,
           'B_vs_copy': n9.pair(G, 'B', cp), 'A_vs_copy': n9.pair(G, 'A', cp), 'copy0_vs_A': n9.pair(G, 'A', 'copy0'),
           'stability': {k: n8.overlap(n8.top(G[k][0]), n8.top(G[k][1])) for k in G},
           'rotated_control': n8.scalars(n8.measure_f(n8.rotated_control(model), mx, u)),
           'functional': functional(model, mx, mys[0], fixed_noisy(mys[0], probs, rho, MEAS_NOISE_SEED), log_u, rho),
           'payload_windows': {k: v for k, v in functional(model, payload[0], payload[1], payload[1], log_u, rho).items()
                               if k in ('kl_B_to_copy', 'kl_copy_to_B', 'B_clean', 'copy_clean', 'G')},
           'readout_overlap_A_B': n8.rowspace_overlap(model.proj_a.weight, model.proj_b.weight),
           'readout_drift': n8.readout_drift(model, seed)}
    res['seconds'] = time.time() - t0
    return res


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
    log('device:', DEV, '| test mode:', TEST, '| study folder (data):', source, '| N10 folder:', work, '| local:', local)
    store.pull_dir('results')
    try:
        meta = study.prepare_data(data_store, None)
        train = study.load_tokens(store, meta, 'train')
        probs = n7.unigram(train)
        log_u = probs.float().log().to(DEV)
        payload = payload_windows(train)
        summary_path = store.L('results/n10_summary.json')
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

        def save_json(rel, obj):
            store.L(rel).write_text(json.dumps(obj, indent=2 if rel.endswith('n10_summary.json') else None))
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
                'payload_windows': {'count': len(payload[0]), 'split': 'train', 'seed': PAYLOAD_SEED,
                                    'starts_first_5': payload[2][:5].tolist(),
                                    'targets_shifted': bool(torch.equal(payload[1][:, :-1], payload[0][:, 1:])),
                                    'first_text': enc.decode(payload[0][0, :20].tolist()),
                                    'disjoint_from_measurement': True},       # different split
                'unigram': {'sum': probs.sum().item(), 'zeros': int((probs == 0).sum()),
                            'most_frequent_tokens': [enc.decode([t]) for t in torch.topk(probs, 5).indices.tolist()]}}
            summary['windows'] = {'split': 'test', 'measure': [MEAS[0] + 1, MEAS[0] + len(mx)]}
            summary['arms'] = {a: {'rho': r, 'payload': k, 'lambda': LAMBDA.get(k)} for a, (r, k) in ARMS.items()}
            summary['seeds'], summary['readout_rank'], summary['subspace_rank'] = SEEDS, n4.RANK, R
            summary['payload'] = {'windows': N_PAYLOAD, 'per_step': PAYLOAD_PER_STEP, 'seed': PAYLOAD_SEED}
            save_json('results/n10_summary.json', summary)
        c = summary['checks']
        log('batch check:', c['batch']['targets_are_i+1_to_i+8'], '| test windows shifted:', c['test_windows_shifted'],
            '| analytic:', round(c['analytic']['sampled']['top_subspace_overlap'], 4), '| identical heads:', c['identical_heads']['agreement_ratio'],
            '| payload windows shifted:', c['payload_windows']['targets_shifted'])
        log('same-model check:', [x['pass'] for x in c['same_model']])
        if not (c['batch']['targets_are_i+1_to_i+8'] and c['test_windows_shifted'] and c['analytic']['pass'] and c['identical_heads']['pass']
                and c['payload_windows']['targets_shifted'] and all(x['pass'] for x in c['same_model'])):
            raise SystemExit('A check of the setup failed before training. Stopping.')

        for seed in SEEDS:
            for arm in ARMS:
                tag = f'{arm}_seed{seed}'
                train_n10(store, meta, arm, seed, tag, probs, log_u, payload)
                rel = f'results/m_{tag}.json'
                if store.L(rel).exists():
                    continue
                if not store.pull(f'final/{tag}.pt'):
                    raise SystemExit(f'weights final/{tag}.pt are missing locally and on Drive.')
                m = n4.new_n4(seed, BASE)
                m.load_state_dict(torch.load(store.L(f'final/{tag}.pt'), map_location=DEV, weights_only=False))
                res = {'tag': tag, 'seed': seed, 'arm': arm, **measure(m, arm, seed, mx, mys, u, log_u, probs, payload)}
                save_json(rel, res)
                fn = res['functional']
                log(f"m_{tag}: B vs copy {res['B_vs_copy']['agreement_ratio']:.3f} | A vs copy {res['A_vs_copy']['agreement_ratio']:.3f} | "
                    f"G {fn['G']:+.3f} | KL(B||copy) {fn['kl_B_to_copy']:.3f} | payload-window KL {res['payload_windows']['kl_B_to_copy']:.3f} | "
                    f"readouts A,B {res['readout_overlap_A_B']:.3f} | rotated {res['rotated_control']['agreement_ratio']:.3f} | {res['seconds']:.0f} s")
        log('N10 complete: all runs and measurements finished')
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
