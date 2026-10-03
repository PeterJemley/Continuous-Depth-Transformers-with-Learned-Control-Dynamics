"""H6: solver effort across the control signal, as in the paper's probe Q2.

Loads a steered model into the original InstrumentedHybridTransformer (dopri5,
tolerance 1e-3), sets u to the same value on all four dimensions as the paper did,
and counts function evaluations at 21 values from -2 to 2 on "The movie was".

    python run_nfe.py --seed 42 --pos-token 922
"""
import argparse

import numpy as np
import torch

from common import CONTEXT_IDS, HYBRID_KW, MODEL_CFG, ckpt_path, pm, write_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--pos-token', type=int, required=True)
    args = ap.parse_args()
    kw = {k: v for k, v in HYBRID_KW.items() if k != 'n_steps'}
    im = pm.InstrumentedHybridTransformer(**MODEL_CFG, **kw)
    sd = torch.load(ckpt_path(f'steer{args.pos_token}_seed{args.seed}'), map_location='cpu')
    sd.pop('ode_block.integration_times', None)
    keys = im.load_state_dict(sd, strict=False)
    im.eval()
    context = torch.tensor([CONTEXT_IDS])
    rows = []
    with torch.no_grad():
        for v in np.linspace(-2, 2, 21):
            im(context, control=torch.ones(1, 4) * float(v))
            nfe = im.get_last_nfe()
            rows.append({'u': round(float(v), 4), 'nfe': nfe, 'solver_steps': (nfe - 2) / 6})
    neg = [r['nfe'] for r in rows if r['u'] < 0.3]
    pos = [r['nfe'] for r in rows if r['u'] > 0.4]
    payload = {'seed': args.seed, 'pos_token': args.pos_token, 'rows': rows,
               'missing_keys': keys.missing_keys, 'unexpected_keys': keys.unexpected_keys,
               'mean_nfe_u_below_0.3': sum(neg) / len(neg), 'mean_nfe_u_above_0.4': sum(pos) / len(pos)}
    write_result(f'nfe{args.pos_token}_seed{args.seed}', payload)
    print(f"seed {args.seed} token {args.pos_token}: NFE {[r['nfe'] for r in rows]}", flush=True)


if __name__ == '__main__':
    main()
