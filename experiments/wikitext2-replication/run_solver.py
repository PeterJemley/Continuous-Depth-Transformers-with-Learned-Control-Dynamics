"""H4: the solver-dependence test of Ott et al. (2020) on one trained Model A.

Validation loss with the training solver (4 Euler steps) and with more accurate
solvers, on the trained model, on the untrained model of the same seed, and on
the trained model with alpha multiplied by 2, 4, 8 and 16. Also hidden-state
divergence measured against how far the ODE block moves the state.

    python run_solver.py --seed 42 [--control-limit N]
"""
import argparse
import time

import torch

from common import hybrid_forward, load_model, new_model, val_loss, val_windows, write_result

SOLVERS = {'euler4': dict(method='euler', n_steps=4), 'euler8': dict(method='euler', n_steps=8),
           'euler16': dict(method='euler', n_steps=16), 'rk4_4': dict(method='rk4', n_steps=4),
           'dopri5_1e-5': dict(method='dopri5', rtol=1e-5, atol=1e-5)}
SCALES = [2, 4, 8, 16]


def losses_for(model, limit, scale=None):
    out = {}
    for name, kw in SOLVERS.items():
        t = time.time()
        try:
            out[name] = {'val_loss': val_loss(model, forward=lambda m, x: hybrid_forward(m, x, scale=scale, **kw), limit=limit)}
        except Exception as e:  # e.g. the adaptive solver's step size underflows
            out[name] = {'val_loss': None, 'error': f'{type(e).__name__}: {e}'[:300]}
        out[name]['seconds'] = time.time() - t
    base = out['euler4']['val_loss']
    for name, r in out.items():
        r['rel_change_vs_euler4'] = None if (r['val_loss'] is None or base is None) else abs(r['val_loss'] - base) / base
    return out


@torch.no_grad()
def divergence(model, scale=None, n=32):
    x, _ = val_windows(limit=n)
    T = x.shape[1]
    h0 = model.tok_emb(x) + model.pos_emb(torch.arange(T))
    for b in model.early_blocks:
        h0 = b(h0)

    def ode_out(**kw):
        # run only the ODE block, reusing hybrid_forward's solver handling
        from common import odeint
        func = model.ode_block.func
        func.control = None
        saved = func.output_scale.data.clone()
        if scale:
            func.output_scale.data.mul_(scale)
        try:
            if kw['method'] in ('euler', 'rk4'):
                return odeint(func, h0, torch.linspace(0, 1, kw['n_steps'] + 1), method=kw['method'])[-1]
            return odeint(func, h0, torch.tensor([0.0, 1.0]), method='dopri5', rtol=kw['rtol'], atol=kw['atol'])[-1]
        finally:
            func.output_scale.data.copy_(saved)
    try:
        ref = ode_out(**SOLVERS['dopri5_1e-5'])
    except Exception as e:
        return {'error': f'{type(e).__name__}: {e}'[:300]}
    he = ode_out(**SOLVERS['euler4'])
    move = (ref - h0).norm().item()
    err = (he - ref).norm().item()
    return {'block_moves_rel_state': move / h0.norm().item(), 'err_rel_state': err / ref.norm().item(),
            'err_rel_movement': err / move if move > 0 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--control-limit', type=int, default=None, help='validation windows used for the controls')
    args = ap.parse_args()
    trained = load_model('hybrid', f'hybrid_seed{args.seed}')
    x, _ = val_windows(limit=8)
    with torch.no_grad():
        same = (hybrid_forward(trained, x) - trained(x)).abs().max().item()
    if same != 0.0:
        raise SystemExit(f'hybrid_forward differs from the model by {same}; fix before measuring')
    untrained = new_model('hybrid', args.seed).eval()
    payload = {'seed': args.seed, 'must_match_max_diff': same, 'control_limit': args.control_limit,
               'trained': {'losses': losses_for(trained, None), 'divergence': divergence(trained)},
               'untrained': {'losses': losses_for(untrained, args.control_limit), 'divergence': divergence(untrained)},
               'scaled': {}}
    for s in SCALES:
        payload['scaled'][str(s)] = {'losses': losses_for(trained, args.control_limit, scale=s), 'divergence': divergence(trained, scale=s)}
    write_result(f'solver_seed{args.seed}', payload)
    print(f"seed {args.seed}: trained dopri5 rel change {payload['trained']['losses']['dopri5_1e-5']['rel_change_vs_euler4']}", flush=True)


if __name__ == '__main__':
    main()
