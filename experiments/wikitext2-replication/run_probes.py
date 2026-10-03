"""H8: linear probes for the sign of u at intermediate depths.

1,024 validation windows chosen at random with the seed, each given u = +1 or -1
on dimension 0 at random. The ODE trajectory is solved with dopri5 (tolerance
1e-3, as in the original probe) at tau = 0, 0.5, 0.67, 0.8, 1.0. Logistic-regression
probes on the sequence-mean hidden state train on 512 windows and test on 512.
Control task: the same probe on randomly permuted labels.

    python run_probes.py --seed 42 --model steer922
    python run_probes.py --seed 42 --model untrained
"""
import argparse

import torch
from sklearn.linear_model import LogisticRegression

from common import load_model, new_model, odeint, val_windows, wilson, write_result

TIMES = [0.0, 0.5, 0.67, 0.8, 1.0]


@torch.no_grad()
def features(model, x, u, chunk=128):
    feats = []
    for i in range(0, len(x), chunk):
        xb, ub = x[i:i + chunk], u[i:i + chunk]
        T = xb.shape[1]
        h = model.tok_emb(xb) + model.pos_emb(torch.arange(T))
        for b in model.early_blocks:
            h = b(h)
        model.ode_block.func.control = ub
        traj = odeint(model.ode_block.func, h, torch.tensor(TIMES), method='dopri5', rtol=1e-3, atol=1e-3)
        feats.append(traj.mean(dim=2))              # [times, batch, d_model]
    return torch.cat(feats, dim=1)


def probe(feat, y):
    clf = LogisticRegression(max_iter=2000).fit(feat[:512].numpy(), y[:512].numpy())
    return float(clf.score(feat[512:].numpy(), y[512:].numpy()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--model', required=True, help='steer922, steer220 or untrained')
    args = ap.parse_args()
    model = new_model('hybrid', args.seed).eval() if args.model == 'untrained' else load_model('hybrid', f'{args.model}_seed{args.seed}')
    xs, _ = val_windows()
    g = torch.Generator().manual_seed(args.seed)
    idx = torch.randperm(len(xs), generator=g)[:1024]
    x = xs[idx]
    sign = torch.randint(0, 2, (1024,), generator=g)
    u = torch.zeros(1024, 4)
    u[:, 0] = sign.float() * 2 - 1
    y_control = sign[torch.randperm(1024, generator=g)]
    feat = features(model, x, u)
    rows = []
    for i, t in enumerate(TIMES):
        acc, ctl = probe(feat[i], sign), probe(feat[i], y_control)
        lo, hi = wilson(round(acc * 512), 512)
        rows.append({'tau': t, 'accuracy': acc, 'control_accuracy': ctl, 'selectivity': acc - ctl,
                     'accuracy_wilson95': [lo, hi]})
    write_result(f'probe_{args.model}_seed{args.seed}', {'seed': args.seed, 'model': args.model, 'rows': rows,
                                                        'n_train': 512, 'n_test': 512})
    print(f"seed {args.seed} {args.model}: " + '  '.join(f"tau {r['tau']}: {r['accuracy']:.3f}/{r['control_accuracy']:.3f}" for r in rows), flush=True)


if __name__ == '__main__':
    main()
