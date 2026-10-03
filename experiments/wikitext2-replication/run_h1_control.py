"""H1 must-fail control: can the paper's gradient detector flag a deliberately unstable run?

Model A with alpha initialised at 10, learning rate 1e-1, 50 steps. The paper's
detector counts steps whose ODE-block gradient norm, measured after clipping, is
below 1e-6 or above 100. The same counts before clipping are exploratory.
"""
import math

import torch
import torch.nn.functional as F

from common import TRAIN_CFG, new_model, pm, tokens, write_result
from run_training import get_batch_from


def flags(norms):
    return {'vanishing': sum(1 for v in norms if v < 1e-6), 'exploding': sum(1 for v in norms if v > 100 or not math.isfinite(v))}


def main():
    model = new_model('hybrid', 0)
    with torch.no_grad():
        model.ode_block.func.output_scale.fill_(10.0)
    gen = torch.Generator()
    gen.set_state(torch.get_rng_state())
    opt = torch.optim.AdamW(model.parameters(), lr=1e-1)
    post, pre, losses = [], [], []
    model.train()
    for _ in range(50):
        x, y = get_batch_from(gen, tokens('train'), TRAIN_CFG['batch_size'], TRAIN_CFG['seq_len'])
        opt.zero_grad()
        logits = model(x)
        loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        pre.append(pm.compute_gradient_stats(model).get('ode_grad_norm', 0.0))
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        post.append(pm.compute_gradient_stats(model).get('ode_grad_norm', 0.0))
        opt.step()
        losses.append(loss.item())
    payload = {'post_clip_flags': flags(post), 'pre_clip_flags': flags(pre), 'post_clip_max': max(post),
               'pre_clip_max': max(pre), 'losses': losses, 'post_clip': post, 'pre_clip': pre}
    write_result('h1_control', payload)
    print('paper detector flags:', payload['post_clip_flags'], ' before clipping:', payload['pre_clip_flags'])


if __name__ == '__main__':
    main()
