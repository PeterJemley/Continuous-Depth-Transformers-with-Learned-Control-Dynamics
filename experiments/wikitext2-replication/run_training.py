"""Train the baseline and Model A for one seed (H1, H2, H3 and the reproduction check).

Mirrors train_model() from the original notebook (commit 6b65d64, cell 12):
AdamW, 500 steps, batch 32, length 64, learning rate 3e-4, clipping at 1.0, and
gradient statistics computed after clipping, as the original did. Two additions,
both logged in the preregistration's deviation log before any run:
- batches are drawn from a generator forked from the CPU random state right after
  model construction, which reproduces the original GPU run's batch order;
- gradient norms before clipping are also recorded, as an exploratory measure.

    python run_training.py --seed 42
"""
import argparse
import time

import torch
import torch.nn.functional as F

from common import (TRAIN_CFG, ckpt_path, new_model, pm, tokens, val_loss, write_result)


def get_batch_from(gen, toks, batch_size, seq_len):
    """Same sampling as pm.get_batch, drawing from a given generator."""
    ix = torch.randint(len(toks) - seq_len - 1, (batch_size,), generator=gen)
    x = torch.stack([toks[i:i + seq_len] for i in ix])
    y = torch.stack([toks[i + 1:i + seq_len + 1] for i in ix])
    return x, y


def train(kind, seed):
    model = new_model(kind, seed)
    batch_gen = torch.Generator()
    batch_gen.set_state(torch.get_rng_state())
    n_params = sum(p.numel() for p in model.parameters())
    toks = tokens('train')
    opt = torch.optim.AdamW(model.parameters(), lr=TRAIN_CFG['lr'])
    log = {k: [] for k in ('loss', 'grad_norm', 'ode_grad_norm', 'ode_grad_norm_preclip', 'alpha')}
    model.train()
    t0 = time.time()
    for step in range(TRAIN_CFG['n_steps']):
        x, y = get_batch_from(batch_gen, toks, TRAIN_CFG['batch_size'], TRAIN_CFG['seq_len'])
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        pre = pm.compute_gradient_stats(model)            # exploratory: before clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        stats = pm.compute_gradient_stats(model)          # the paper's statistic: after clipping
        opt.step()
        log['loss'].append(loss.item())
        log['grad_norm'].append(stats['total_grad_norm'])
        if 'ode_grad_norm' in stats:
            log['ode_grad_norm'].append(stats['ode_grad_norm'])
            log['ode_grad_norm_preclip'].append(pre['ode_grad_norm'])
            log['alpha'].append(model.ode_block.func.output_scale.item())
        if step % 50 == 0:
            print(f'{kind} seed {seed} step {step:4d} loss {loss.item():.4f}', flush=True)
    train_seconds = time.time() - t0
    torch.save(model.state_dict(), ckpt_path(f'{kind}_seed{seed}'))
    t1 = time.time()
    vl = val_loss(model)
    payload = {'kind': kind, 'seed': seed, 'n_params': n_params, 'config': TRAIN_CFG,
               'step0_loss': log['loss'][0], 'last50_train_loss': sum(log['loss'][-50:]) / 50,
               'val_loss': vl, 'train_seconds': train_seconds, 'val_seconds': time.time() - t1, 'log': log}
    if kind == 'hybrid':
        payload['alpha_final'] = log['alpha'][-1]
    write_result(f'train_{kind}_seed{seed}', payload)
    print(f'{kind} seed {seed}: val loss {vl:.4f}, train {train_seconds:.0f}s', flush=True)


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--kinds', nargs='+', default=['baseline', 'hybrid'])
    args = ap.parse_args()
    for kind in args.kinds:
        train(kind, args.seed)
