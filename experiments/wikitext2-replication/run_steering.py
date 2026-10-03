"""H5: the paper's three-phase steering procedure on one trained Model A.

The phases follow the original notebook (commit 6b65d64, cell 18) step for step:
100 steps on all parameters (AdamW, lr 1e-4), 200 steps on control_embed only
(lr 1e-2), 200 steps on the ODE block only (lr 1e-4), with "The movie was" as the
only training prompt. The random state is reseeded to seed + 1000 first, as
logged in the deviation log, so each run is reproducible.

    python run_steering.py --seed 42 --pos-token 922    # corrected ' good'
    python run_steering.py --seed 42 --pos-token 220    # the paper's token, a space
"""
import argparse
import copy

import torch
import torch.nn.functional as F

from common import (CONTEXT_IDS, HELD_OUT_NOUNS, NEG_TOKEN, ckpt_path, load_model, val_loss, write_result)
from pinned_data import gpt2_encoding


def probs(model, ids, u):
    with torch.no_grad():
        p = F.softmax(model(torch.tensor([ids]), control=u)[0, -1], dim=-1)
    return p


def steer(model, pos_token, neg_token=NEG_TOKEN):
    context = torch.tensor([CONTEXT_IDS])
    u_pos = torch.zeros(1, 4); u_pos[0, 0] = 1.0
    u_neg = torch.zeros(1, 4); u_neg[0, 0] = -1.0
    tgt_pos, tgt_neg = torch.tensor([pos_token]), torch.tensor([neg_token])

    def loss_fn():
        return (F.cross_entropy(model(context, control=u_pos)[:, -1, :], tgt_pos) +
                F.cross_entropy(model(context, control=u_neg)[:, -1, :], tgt_neg))

    losses = {}
    # Phase 1: warmup, all parameters
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    model.train()
    losses['phase1'] = []
    for _ in range(100):
        optimizer.zero_grad(); loss = loss_fn(); loss.backward(); optimizer.step(); losses['phase1'].append(loss.item())
    # Phase 2: control_embed only
    for p in model.parameters():
        p.requires_grad = False
    for p in model.ode_block.func.control_embed.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW(model.ode_block.func.control_embed.parameters(), lr=1e-2)
    losses['phase2'] = []
    for _ in range(200):
        optimizer.zero_grad(); loss = loss_fn(); loss.backward(); optimizer.step(); losses['phase2'].append(loss.item())
    for p in model.parameters():
        p.requires_grad = True
    # Phase 3: hybrid unfreeze, ODE block only
    for p in model.parameters():
        p.requires_grad = False
    for p in model.ode_block.parameters():
        p.requires_grad = True
    optimizer = torch.optim.AdamW(model.ode_block.parameters(), lr=1e-4)
    losses['phase3'] = []
    for _ in range(200):
        optimizer.zero_grad(); loss = loss_fn(); loss.backward(); optimizer.step(); losses['phase3'].append(loss.item())
    for p in model.parameters():
        p.requires_grad = True
    model.eval()
    return losses


def evaluate(model, pos_token, neg_token=NEG_TOKEN):
    u = {s: torch.zeros(1, 4) for s in ('+1', '-1', '0')}
    u['+1'][0, 0], u['-1'][0, 0] = 1.0, -1.0
    out = {}
    # H5a and H5c: the training prompt at u = +1, -1 and 0
    for s, uu in u.items():
        p = probs(model, CONTEXT_IDS, uu)
        out[f'train_prompt_u{s}'] = {'p_pos': p[pos_token].item(), 'p_neg': p[neg_token].item()}
    # H5b: held-out prompts "The X was"
    enc = gpt2_encoding()
    held = []
    for noun in HELD_OUT_NOUNS:
        ids = enc.encode(f'The {noun} was')
        pp, pn = probs(model, ids, u['+1']), probs(model, ids, u['-1'])
        held.append({'noun': noun, 'ids': ids,
                     'plus': {'p_pos': pp[pos_token].item(), 'p_neg': pp[neg_token].item()},
                     'minus': {'p_pos': pn[pos_token].item(), 'p_neg': pn[neg_token].item()},
                     'plus_correct': pp[pos_token].item() > pp[neg_token].item(),
                     'minus_correct': pn[neg_token].item() > pn[pos_token].item()})
    out['held_out'] = held
    out['held_out_correct'] = sum(h['plus_correct'] + h['minus_correct'] for h in held)
    out['held_out_total'] = 2 * len(held)
    # H5d: with the control input path ablated, the two signs must give identical outputs
    ablated = copy.deepcopy(model)
    with torch.no_grad():
        ablated.ode_block.func.control_embed.weight.zero_()
        lp = ablated(torch.tensor([CONTEXT_IDS]), control=u['+1'])
        ln = ablated(torch.tensor([CONTEXT_IDS]), control=u['-1'])
    out['path_check_max_logit_diff'] = (lp - ln).abs().max().item()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--pos-token', type=int, required=True)
    args = ap.parse_args()
    enc = gpt2_encoding()
    model = load_model('hybrid', f'hybrid_seed{args.seed}')
    before = val_loss(model)
    torch.manual_seed(args.seed + 1000)
    losses = steer(model, args.pos_token)
    result = evaluate(model, args.pos_token)
    after = val_loss(model)
    torch.save(model.state_dict(), ckpt_path(f'steer{args.pos_token}_seed{args.seed}'))
    payload = {'seed': args.seed, 'pos_token': args.pos_token, 'pos_token_text': enc.decode([args.pos_token]),
               'neg_token': NEG_TOKEN, 'neg_token_text': enc.decode([NEG_TOKEN]),
               'context_text': enc.decode(CONTEXT_IDS), 'losses': losses,
               'val_loss_before': before, 'val_loss_after': after, **result}
    write_result(f'steer{args.pos_token}_seed{args.seed}', payload)
    r = result
    print(f"seed {args.seed} token {args.pos_token}: +1 {r['train_prompt_u+1']}  -1 {r['train_prompt_u-1']}  "
          f"held-out {r['held_out_correct']}/{r['held_out_total']}", flush=True)


if __name__ == '__main__':
    main()
