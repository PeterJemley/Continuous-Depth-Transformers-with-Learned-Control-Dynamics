"""G1: does a trained control signal partition the vector field into distinct dynamical regimes?

Implements G1_PREREGISTRATION.md in this folder. Model A (the paper's HybridODETransformer)
is trained with exact gradients, as arm AX in T1, while its control u selects the task:
u = (+1, 0, 0, 0) asks for the next token, u = (-1, 0, 0, 0) for the token after next.
Shuffled-control models see the same data and tasks with u drawn independently of the task.
Each trained model's ODE block is then measured: trajectory curvature at u = +1, -1 and 0 on
the first 256 validation windows, plus the secondary measures.

    python g1_regimes.py --out OUT --work /content/drive/MyDrive/cdt-colab/g1_regimes
    STUDY_TEST=1 python g1_regimes.py --out OUT --work W --source LOCAL_STUDY_FOLDER   # small test

Token files are read from the WikiText-103 study's Drive folder without writing to it; this
job's checkpoints, final weights and results go to its own folder, so a session that ends is
resumed by running the job again.
"""
import argparse
import json
import math
import os
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.func import jvp

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'wikitext103-training'))

import study  # noqa: E402  (sets the model config, test mode and device)
import paper_model as pm  # noqa: E402
from torchdiffeq import odeint as odeint_exact  # noqa: E402

DEV, TEST, AMP, log = study.DEV, study.TEST, study.AMP, study.log
SEQ, BATCH = study.SEQ, study.BATCH
VOCAB = study.CFG['vocab_size']

TOTAL = 60 if TEST else 2879                 # 10% of the WikiText-103 budget, as in T1
PEAK_LR = 2.4e-3                             # T1's choice for AX, not re-tuned
SEEDS = [3001, 3002] if TEST else [3001, 3002, 3003, 3004, 3005]
VARIANTS = ['main', 'shuf']                  # u tells the truth / u independent of the task
VAL_EVERY = 20 if TEST else 400
VAL_LIMIT = 64 if TEST else 512              # windows for the curves during training
FINAL_LIMIT = 64 if TEST else None           # the whole validation split for final losses
N_GEOM = 16 if TEST else 256                 # the first 256 validation windows
GEOM_BATCH = 8 if TEST else 32
DEPTHS = torch.linspace(0, 1, 17)            # 17 evenly spaced depths
GEOM_TOL, NFE_TOL, MAX_STEPS = 1e-5, 1e-3, 10000
DIV_IDX = [0, 4, 8, 12, 16]                  # depths 0, 0.25, 0.5, 0.75, 1 for divergence
DIV_PROBES = 8
CONTROLS = {'plus': 1.0, 'minus': -1.0, 'zero': 0.0}


# ------------------------------------------------------------------ the model

def _exact_ode_forward(self, x, control=None):
    """ODEFlowBlock.forward with ordinary backpropagation through the Euler steps, as AX in T1."""
    self.func.control = control
    return odeint_exact(self.func, x, self.integration_times, method='euler')[-1]


def new_model(seed, exact=True):
    torch.manual_seed(seed)
    m = pm.HybridODETransformer(**study.CFG, **study.HYB)
    if exact:
        m.ode_block.forward = types.MethodType(_exact_ode_forward, m.ode_block)
    return m.to(DEV)


def control(values, n=None):
    """u = (value, 0, 0, 0) per sequence. `values` is a tensor of signs, or one number for n sequences."""
    if not torch.is_tensor(values):
        values = torch.full((n,), float(values))
    u = torch.zeros(len(values), study.HYB['control_dim'])
    u[:, 0] = values.float()
    return u.to(DEV)


# ------------------------------------------------------------------ data

def g1_windows(tokens, limit=None):
    """Non-overlapping 128-token inputs, with the targets at i + 1 and at i + 2."""
    n = (len(tokens) - 2) // SEQ
    if limit:
        n = min(n, limit)
    t = torch.from_numpy(np.asarray(tokens[:n * SEQ + 2], dtype=np.int64))
    idx = torch.arange(n)[:, None] * SEQ + torch.arange(SEQ)[None, :]
    return t[idx], t[idx + 1], t[idx + 2]


def draw_batch(train, gen, task_gen):
    """130-token windows from the run's generator, and one task per sequence: +1 next token, -1 the one after."""
    ix = torch.randint(len(train) - SEQ - 2, (BATCH,), generator=gen).numpy()
    b = torch.from_numpy(np.stack([np.asarray(train[i:i + SEQ + 2], dtype=np.int64) for i in ix]))
    task = 1 - 2 * torch.randint(2, (BATCH,), generator=task_gen)
    return b[:, :SEQ], b[:, 1:SEQ + 1], b[:, 2:SEQ + 2], task


@torch.no_grad()
def task_loss(model, xs, ys, u_value, batch=32):
    """Mean cross-entropy in nats against targets ys, with every sequence given u = (u_value, 0, 0, 0). fp32."""
    was = model.training
    model.eval()
    total, count = 0.0, 0
    for i in range(0, len(xs), batch):
        x, y = xs[i:i + batch].to(DEV), ys[i:i + batch].to(DEV)
        logits = model(x, control=control(u_value, len(x))).float()
        total += F.cross_entropy(logits.reshape(-1, VOCAB), y.reshape(-1), reduction='sum').item()
        count += y.numel()
    model.train(was)
    return total / count


# ------------------------------------------------------------------ training

def train_g1(store, meta, variant, seed):
    """One training run, resumable. The seed fixes the initial weights, the data order and the tasks,
    so a seed's main and shuffled models differ only in u."""
    tag = f'train_{variant}_seed{seed}'
    res_path = store.L(f'results/{tag}.json')
    if res_path.exists():
        log(f'{tag}: already finished, skipping')
        return json.loads(res_path.read_text())
    ck_rel = f'ckpt/{tag}.pt'
    store.pull(ck_rel)
    ck_path = store.L(ck_rel)
    train = study.load_tokens(store, meta, 'train')
    vx, vy1, vy2 = g1_windows(study.load_tokens(store, meta, 'validation'), limit=VAL_LIMIT)
    model = new_model(seed)
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR)
    scaler = torch.amp.GradScaler('cuda', enabled=AMP)
    gen = torch.Generator().manual_seed(seed)
    task_gen = torch.Generator().manual_seed(seed * 10 + 1)
    u_gen = torch.Generator().manual_seed(seed * 10 + 2)
    state = {'step': 0, 'train_loss': [], 'val': [], 'alpha': [], 'skipped_steps': 0, 'seconds': 0.0, 'resumes': 0,
             'u_matches_task': [0, 0]}
    if ck_path.exists():
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); scaler.load_state_dict(ck['scaler'])
        gen.set_state(ck['gen']); task_gen.set_state(ck['task_gen']); u_gen.set_state(ck['u_gen'])
        state = ck['state']; state['resumes'] += 1
        torch.set_rng_state(ck['torch_rng'])
        if DEV.type == 'cuda' and ck.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
        log(f"{tag}: resuming at step {state['step']}")
    else:
        log(f'{tag}: starting, seed {seed}, peak lr {PEAK_LR}, {TOTAL} steps')

    def save_ckpt():
        tmp = ck_path.with_suffix('.tmp')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(),
                    'gen': gen.get_state(), 'task_gen': task_gen.get_state(), 'u_gen': u_gen.get_state(),
                    'state': state, 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if DEV.type == 'cuda' else None}, tmp)
        os.replace(tmp, ck_path)
        store.push(ck_rel)

    model.train()
    last_ck, t_seg, bad = time.time(), time.time(), 0
    while state['step'] < TOTAL:
        step = state['step']
        for g in opt.param_groups:
            g['lr'] = study.lr_at(step, TOTAL, PEAK_LR)
        x, y1, y2, task = draw_batch(train, gen, task_gen)
        u_sign = task if variant == 'main' else 1 - 2 * torch.randint(2, (BATCH,), generator=u_gen)
        state['u_matches_task'][0] += int((u_sign == task).sum()); state['u_matches_task'][1] += BATCH
        y = torch.where(task[:, None] == 1, y1, y2)
        x, y = x.to(DEV), y.to(DEV)
        opt.zero_grad(set_to_none=True)
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=AMP):
            loss = F.cross_entropy(model(x, control=control(u_sign)).view(-1, VOCAB), y.reshape(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scale_before = scaler.get_scale()
        scaler.step(opt)
        scaler.update()
        if AMP and scaler.get_scale() < scale_before:
            state['skipped_steps'] += 1
        if not math.isfinite(loss.item()):
            bad += 1
            if bad >= 100:
                raise SystemExit(f'{tag}: 100 consecutive non-finite losses. Stopping this run.')
        else:
            bad = 0
        state['step'] = step + 1
        if step % 50 == 0:
            state['train_loss'].append([step, loss.item()])
            state['alpha'].append([step, model.ode_block.func.output_scale.item()])
        if state['step'] == 500 or state['step'] % VAL_EVERY == 0 or state['step'] == TOTAL:
            ln, ls = task_loss(model, vx, vy1, +1), task_loss(model, vx, vy2, -1)
            state['val'].append([state['step'], ln, ls])
            log(f"{tag}: step {state['step']}/{TOTAL}, validation loss next-token (u=+1) {ln:.4f}, "
                f"token after next (u=-1) {ls:.4f}")
        if time.time() - last_ck > study.CKPT_SECONDS:
            state['seconds'] += time.time() - t_seg; t_seg = time.time()
            save_ckpt(); last_ck = time.time()
    state['seconds'] += time.time() - t_seg
    save_ckpt()
    torch.save(model.state_dict(), store.L(f'final/{tag}.pt'))
    store.push(f'final/{tag}.pt')

    # S8 and the manipulation check: both tasks under every control value, on the whole validation split.
    fx, fy1, fy2 = g1_windows(study.load_tokens(store, meta, 'validation'), limit=FINAL_LIMIT)
    final = {f'{task}_u{name}': task_loss(model, fx, ys, val)
             for task, ys in (('next', fy1), ('skip', fy2)) for name, val in CONTROLS.items()}
    result = {'tag': tag, 'variant': variant, 'seed': seed, 'peak_lr': PEAK_LR, 'total_steps': TOTAL,
              'final_losses': final, 'final_val_windows': len(fx), 'val_curve': state['val'],
              'train_loss_every_50': state['train_loss'], 'alpha_every_50': state['alpha'],
              'alpha_final': model.ode_block.func.output_scale.item(),
              'u_matches_task_fraction': state['u_matches_task'][0] / max(1, state['u_matches_task'][1]),
              'skipped_steps_fp16': state['skipped_steps'], 'train_seconds': state['seconds'], 'resumes': state['resumes'],
              'n_params': sum(p.numel() for p in model.parameters()),
              'config': {**study.CFG, **study.HYB, 'seq': SEQ, 'batch': BATCH, 'warmup': study.WARMUP,
                         'min_lr_ratio': study.MIN_LR_RATIO, 'gradients': 'exact'},
              'data': {s: {k: meta[s][k] for k in ('tokens', 'sha256')} for s in ('train', 'validation')},
              'environment': study.env_record()}
    res_path.write_text(json.dumps(result, indent=2))
    store.push(f'results/{tag}.json')
    log(f'{tag}: finished in {state["seconds"] / 3600:.2f} h; next-token loss with u=+1 {final["next_uplus"]:.4f}, '
        f'with u=-1 {final["next_uminus"]:.4f}')
    return result


# ------------------------------------------------------------------ geometry

def curvature(func, t, h):
    """Per-token speed and curvature of the flow dh/dt = func(t, h) at (t, h).

    Velocity v = func(t, h). Acceleration a = (df/dh) v + df/dt, exactly, by one forward-mode
    Jacobian-vector product of the whole sequence state (attention couples the tokens), then read
    per token. Curvature = |a - (a.v_hat) v_hat| / |v|^2, in float64 from the float32 v and a.
    """
    t = torch.as_tensor(t, dtype=h.dtype, device=h.device)
    v = func(t, h)
    _, a = jvp(lambda tt, hh: func(tt, hh), (t, h), (torch.ones_like(t), v))
    v, a = v.double(), a.double()
    vv = (v * v).sum(-1)
    a_perp = a - ((a * v).sum(-1) / vv).unsqueeze(-1) * v
    return a_perp.norm(dim=-1) / vv, vv.sqrt()


def trajectory(func, h0, times, method='dopri5', tol=GEOM_TOL):
    kw = dict(rtol=tol, atol=tol, options={'max_num_steps': MAX_STEPS}) if method == 'dopri5' else {}
    return odeint_exact(func, h0, times.to(h0.device, h0.dtype), method=method, **kw)


def log_kappa(kappa, speed):
    """log curvature, with points where the state does not move, or does not turn, set to NaN and counted."""
    bad = (speed == 0) | (kappa == 0) | ~torch.isfinite(kappa)
    return torch.where(bad, torch.full_like(kappa, float('nan')), kappa.clamp_min(1e-300).log()), int(bad.sum())


class Counting:
    def __init__(self, func):
        self.func, self.n = func, 0

    def __call__(self, t, h):
        self.n += 1
        return self.func(t, h)


@torch.no_grad()
def measure_geometry(model, xs, seed):
    """Per-window geometry of the ODE block at u = +1, -1 and 0. Dropout off, fp32."""
    model.eval()
    func = model.ode_block.func
    probe_gen = torch.Generator().manual_seed(seed + 77)
    out = {name: {'logk_mean': [], 'k_mean': [], 'logk_by_depth': [], 'speed_mean': [], 'speed_by_depth': [],
                  'div_by_depth': [], 'euler_logk_mean': [], 'euler_logk_by_depth': [],
                  'excluded_points': 0, 'points': 0, 'nfe_tol1e-3_per_batch': [], 'nfe_tol1e-5_per_batch': []}
           for name in CONTROLS}
    for i in range(0, len(xs), GEOM_BATCH):
        x = xs[i:i + GEOM_BATCH].to(DEV)
        B, T = x.shape
        h0 = model.tok_emb(x) + model.pos_emb(torch.arange(T, device=DEV))
        for blk in model.early_blocks:
            h0 = blk(h0)
        probes = [(torch.randint(2, h0.shape, generator=probe_gen).float() * 2 - 1).to(DEV) for _ in range(DIV_PROBES)]
        for name, val in CONTROLS.items():
            o = out[name]
            func.control = control(val, B)
            counter = Counting(func)
            traj = trajectory(counter, h0, DEPTHS)                          # (17, B, T, D)
            o['nfe_tol1e-5_per_batch'].append(counter.n)
            counter = Counting(func)
            trajectory(counter, h0, torch.tensor([0.0, 1.0]), tol=NFE_TOL)  # S5: the paper's own measure
            o['nfe_tol1e-3_per_batch'].append(counter.n)
            logk, kap, spd = [], [], []
            for k, t in enumerate(DEPTHS.tolist()):
                kappa, speed = curvature(func, t, traj[k])
                lk, nbad = log_kappa(kappa, speed)
                o['excluded_points'] += nbad
                o['points'] += kappa.numel()
                logk.append(lk); kap.append(kappa); spd.append(speed)
            logk, kap, spd = torch.stack(logk, 1), torch.stack(kap, 1), torch.stack(spd, 1)  # (B, 17, T)
            o['logk_mean'] += torch.nanmean(logk.flatten(1), 1).tolist()
            o['k_mean'] += torch.where(torch.isfinite(logk), kap, torch.nan).flatten(1).nanmean(1).tolist()
            o['logk_by_depth'] += torch.nanmean(logk, 2).tolist()
            o['speed_mean'] += spd.flatten(1).mean(1).tolist()
            o['speed_by_depth'] += spd.mean(2).tolist()
            # S4: divergence, the trace of df/dh by Hutchinson's estimator, per token.
            div = []
            for k in DIV_IDX:
                t = torch.as_tensor(DEPTHS[k].item(), device=DEV)
                est = []
                for z in probes:
                    _, jz = jvp(lambda hh: func(t, hh), (traj[k],), (z,))
                    est.append((z.double() * jz.double()).flatten(1).sum(1) / T)
                div.append(torch.stack(est).mean(0))
            o['div_by_depth'] += torch.stack(div, 1).tolist()
            # S9: curvature at the four states the trained model's Euler steps visit.
            euler = trajectory(func, h0, torch.linspace(0, 1, 5), method='euler')
            elk = []
            for k in range(4):
                kappa, speed = curvature(func, k / 4, euler[k])
                elk.append(log_kappa(kappa, speed)[0])
            elk = torch.stack(elk, 1)
            o['euler_logk_mean'] += torch.nanmean(elk.flatten(1), 1).tolist()
            o['euler_logk_by_depth'] += torch.nanmean(elk, 2).tolist()
        func.control = None
    return out


def analytic_check():
    """Precondition 2: synthetic fields whose flows are known curves, through the same code."""
    D, B, T = 8, 2, 3
    r = torch.tensor([[0.5, 1.0, 2.0], [3.0, 0.7, 1.5]])          # one radius per token
    c = 0.6

    def start(rad):
        h = torch.zeros(B, T, D)
        h[..., 0] = rad
        return h

    def rot(h, scale=1.0):
        f = torch.zeros_like(h)
        f[..., 0], f[..., 1] = -h[..., 1] * scale, h[..., 0] * scale
        return f

    line_dir = torch.randn(B, T, D, generator=torch.Generator().manual_seed(0))
    fields = {
        'line': (lambda t, h: line_dir + 0 * h + 0 * t, start(r), lambda t, h0: torch.zeros(B, T)),
        'circle': (lambda t, h: rot(h), start(r), lambda t, h0: 1 / r),
        'circle_fast': (lambda t, h: rot(h, 3.0), start(r), lambda t, h0: 1 / r),   # speed must not matter
        'helix': (lambda t, h: rot(h) + torch.nn.functional.pad(torch.full((B, T, 1), c) + 0 * t, (2, D - 3)),
                  start(r), lambda t, h0: r / (r ** 2 + c ** 2)),
        # Time-dependent: v = (1, t), the parabola y = x^2 / 2 from the origin; curvature 1 / (1 + t^2)^1.5.
        'parabola': (lambda t, h: torch.nn.functional.pad(torch.stack([torch.ones(B, T) + 0 * t, torch.ones(B, T) * t], -1)
                                                         + 0 * h[..., :2], (0, D - 2)),
                     torch.zeros(B, T, D), lambda t, h0: torch.full((B, T), 1 / (1 + t ** 2) ** 1.5)),
    }
    res = {}
    for name, (f, h0, truth) in fields.items():
        traj = trajectory(f, h0, DEPTHS)
        worst = 0.0
        for k, t in enumerate(DEPTHS.tolist()):
            kappa, _ = curvature(f, t, traj[k])
            want = truth(t, h0).double()
            err = (kappa - want).abs() / want.abs().clamp_min(1e-12) if name != 'line' else kappa.abs()
            worst = max(worst, err.max().item())
        res[name] = {'worst_error': worst, 'kind': 'absolute' if name == 'line' else 'relative'}
    res['pass'] = all(v['worst_error'] <= (1e-6 if k == 'line' else 1e-3) for k, v in res.items())
    return res


# ------------------------------------------------------------------ checks before training

def same_model_check(train, seed):
    """AX's forward pass matches the adjoint version's: same weights, loss and downstream gradients, with u given."""
    gen, task_gen = torch.Generator().manual_seed(seed), torch.Generator().manual_seed(seed * 10 + 1)
    x, y1, y2, task = draw_batch(train, gen, task_gen)
    y = torch.where(task[:, None] == 1, y1, y2)
    x, y = x.to(DEV), y.to(DEV)
    out = {}
    for exact in (True, False):
        m = new_model(seed, exact=exact)
        m.train()
        torch.manual_seed(seed + 7)
        loss = F.cross_entropy(m(x, control=control(task)).reshape(-1, VOCAB), y.reshape(-1))
        loss.backward()
        down = torch.cat([p.grad.reshape(-1) for blk in m.late_blocks for p in blk.parameters()]
                         + [p.grad.reshape(-1) for p in m.head.parameters()])
        out[exact] = (loss.item(), down, torch.cat([p.detach().reshape(-1) for p in m.parameters()]))
    rel = abs(out[True][0] - out[False][0]) / abs(out[False][0])
    cosd = F.cosine_similarity(out[True][1].double(), out[False][1].double(), dim=0).item()
    same_w = torch.equal(out[True][2], out[False][2])
    return {'seed': seed, 'loss_rel_diff': rel, 'downstream_cos': cosd, 'same_initial_weights': same_w,
            'pass': same_w and rel <= 1e-6 and cosd >= 0.99999}


def batch_check(train):
    """One training batch of each task: decoded, and its targets checked against the input tokens."""
    gen, task_gen = torch.Generator().manual_seed(SEEDS[0]), torch.Generator().manual_seed(SEEDS[0] * 10 + 1)
    x, y1, y2, task = draw_batch(train, gen, task_gen)
    enc = study.gpt2_encoding()
    shifted_ok = bool(torch.equal(y1[:, :-1], x[:, 1:]) and torch.equal(y2[:, :-2], x[:, 2:]) and torch.equal(y2[:, :-1], y1[:, 1:]))
    rec = {'targets_are_i+1_and_i+2': shifted_ok, 'tasks_in_first_batch': task.tolist()}
    for sign, name in ((1, 'next'), (-1, 'skip')):
        j = int((task == sign).nonzero()[0]) if (task == sign).any() else None
        if j is None:
            continue
        ys = y1 if sign == 1 else y2
        rec[f'{name}_example'] = {'input_end': enc.decode(x[j, -12:].tolist()), 'target_end': enc.decode(ys[j, -12:].tolist())}
        log(f'{name}-task example: input ends {rec[name + "_example"]["input_end"]!r}; '
            f'targets end {rec[name + "_example"]["target_end"]!r}')
    return rec


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--work', required=True, help="this job's Drive folder, for checkpoints and results")
    ap.add_argument('--source', default=None, help='the WikiText-103 study folder (default: sibling wt103_study)')
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
    data_store = study.Store(source, local)       # reads token files only; never writes to the study folder
    store = study.Store(work, local)              # checkpoints and results for G1
    out = Path(args.out)
    (out / 'results').mkdir(parents=True, exist_ok=True)
    log('device:', DEV, '| test mode:', TEST, '| study folder (data):', source, '| G1 folder:', work, '| local:', local)
    store.pull_dir('results')
    try:
        meta = study.prepare_data(data_store, None)
        train = study.load_tokens(store, meta, 'train')
        summary_path = store.L('results/g1_summary.json')
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        if 'checks' not in summary:
            summary['checks'] = {'analytic': analytic_check(), 'batch': batch_check(train),
                                 'same_model': [same_model_check(train, s) for s in SEEDS]}
            summary_path.write_text(json.dumps(summary, indent=2))
            store.push('results/g1_summary.json')
        c = summary['checks']
        log('analytic curvature check:', c['analytic']['pass'],
            {k: f"{v['worst_error']:.1e}" for k, v in c['analytic'].items() if k != 'pass'})
        log('batch check, targets at i+1 and i+2:', c['batch']['targets_are_i+1_and_i+2'])
        log('same-model check:', [x['pass'] for x in c['same_model']])
        if not (c['batch']['targets_are_i+1_and_i+2'] and all(x['pass'] for x in c['same_model'])):
            raise SystemExit('A check of the training setup failed before training. Stopping.')
        if not c['analytic']['pass']:
            log('The analytic check failed: G1 will be untestable, as preregistered. Training continues so that '
                'the secondary measures and the cause can be examined.')

        vx_geom = g1_windows(study.load_tokens(store, meta, 'validation'), limit=N_GEOM)[0]
        for seed in SEEDS:
            for variant in VARIANTS:
                train_g1(store, meta, variant, seed)
            for variant in VARIANTS:
                tag = f'geom_{variant}_seed{seed}'
                if store.L(f'results/{tag}.json').exists():
                    log(f'{tag}: already measured, skipping')
                    continue
                if not store.pull(f'final/train_{variant}_seed{seed}.pt'):
                    raise SystemExit(f'final weights for {variant} seed {seed} are missing locally and on Drive.')
                model = new_model(seed)
                model.load_state_dict(torch.load(store.L(f'final/train_{variant}_seed{seed}.pt'), map_location=DEV))
                t0 = time.time()
                geo = measure_geometry(model, vx_geom, seed)
                rec = {'tag': tag, 'variant': variant, 'seed': seed, 'windows': len(vx_geom),
                       'depths': DEPTHS.tolist(), 'div_depths': [DEPTHS[k].item() for k in DIV_IDX],
                       'tolerance': GEOM_TOL, 'nfe_tolerance': NFE_TOL, 'seconds': time.time() - t0,
                       'alpha': model.ode_block.func.output_scale.item(), 'controls': geo,
                       'environment': study.env_record()}
                store.L(f'results/{tag}.json').write_text(json.dumps(rec))
                store.push(f'results/{tag}.json')
                m = {n: float(np.mean(geo[n]['logk_mean'])) for n in CONTROLS}
                log(f'{tag}: measured in {rec["seconds"]:.0f}s; mean log curvature '
                    + ', '.join(f'u={CONTROLS[n]:+g} {m[n]:.3f}' for n in CONTROLS))
        log('G1 complete: all runs and measurements finished')
    finally:
        dest = out / 'results'
        for f in (store.local / 'results').glob('*.json'):
            if f.name.startswith(('train_', 'geom_', 'g1_summary')):
                (dest / f.name).write_bytes(f.read_bytes())
        log(f'copied {len(list(dest.glob("*.json")))} result files for upload')
        log('waiting for background Drive saves to finish (up to 15 minutes)')
        store.flush()
        failures = store.failures + data_store.failures
        (out / 'drive_sync.json').write_text(json.dumps({'failures': failures}, indent=2))
        log('Drive saves finished' if not failures else f'Drive problems this session: {failures}')


if __name__ == '__main__':
    main()
