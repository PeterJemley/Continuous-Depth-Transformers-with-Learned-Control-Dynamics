"""N17: the loop, not the judge's accuracy.

Implements N17_PREREGISTRATION.md in this folder. N12's model, training, batches, windows and measures (n12_bottleneck),
N13's reweighted single-softmax head B (n13_signal), N15's weight (n15_sharpen), on N12's seeds at the 80% noise rate.
Every new arm adds head C, a third readout of the trunk's final state, read detached, which judges head B: head B's loss
is the mean over all positions of stopgrad(w(y)) * CE, with w = 0.2 p_C(y) / (0.2 p_C(y) + 0.8 u(y)).
  rw-peer                 head C is trained as rw-b1's head B is: head B's noisy targets, weighted by its own discount
  rw-j20 ... rw-j0125     head C is trained at full weight on the clean targets at a nested 20 ... 1.25% of positions
                          (N14's positions, inside K): loop-free judges
  rw-b1-plain             calibration only: rw-b1 retrained in this code, head B judging itself, no head C
  rw-b1-passenger         calibration only: the same, with a head C trained as the peer but not judging
  rw-b1 (N13's n80-rw), rw-oracle (N16's), sub20 and clean (N12), sub10 ... sub0125 (N14)   saved models, re-measured
Head C's optimisation is tied to the main group's one way: its loss is scaled by the main group's scale, its gradient
unscaled by the same reciprocal and clipped by the same coefficient, and it steps only when the main group steps.

    python n17_loop.py --out OUT --work /content/drive/MyDrive/cdt-colab/n17_loop [--rungs rw-j10,rw-j05]
    python n17_loop.py --calibration --out OUT --work /content/drive/MyDrive/cdt-colab/n17_calib
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import n12_bottleneck as n12  # noqa: E402
import n13_signal as n13  # noqa: E402  (its patches of n12's loss and output form stay in force: 'rw' is measured in the mixture form)
import n15_sharpen as n15  # noqa: E402  (weight, weight_identity_check)
import n16_judge as n16  # noqa: E402  (neglect)

n10, n9, n8, n6, n4, g, study, DEV, TEST, log = n12.n10, n12.n9, n12.n8, n12.n6, n12.n4, n12.g, n12.study, n12.DEV, n12.TEST, n12.log
AMP, VOCAB, RHO, BASE, TOTAL, PEAK_LR = n12.AMP, n12.VOCAB, n12.RHO, n12.BASE, n12.TOTAL, n12.PEAK_LR
MODES = n12.MODES
FRAC_K = 0.2                                         # n80's kept fraction, which N14's nested sets are drawn at
RUNGS = {'rw-j20': 0.2, 'rw-j10': 0.1, 'rw-j05': 0.05, 'rw-j025': 0.025, 'rw-j0125': 0.0125}
N14_TAG = {'rw-j10': 'sub10', 'rw-j05': 'sub05', 'rw-j025': 'sub025', 'rw-j0125': 'sub0125'}     # the N14 arm with the same positions
SPECS = {'rw-peer': {'c': 'self', 'judge': 'C', 'frac': None},
         **{a: {'c': 'clean', 'judge': 'C', 'frac': f} for a, f in RUNGS.items()},
         'rw-b1-plain': {'c': None, 'judge': 'B', 'frac': None},
         'rw-b1-passenger': {'c': 'self', 'judge': 'B', 'frac': None}}
CALIB_ONLY = ('rw-b1-plain', 'rw-b1-passenger')
DEFAULT_RUNGS = ('rw-j10', 'rw-j05')
REFS = {'rw-b1': ('n13', 'n80-rw', 'n13'), 'rw-oracle': ('n16', 'rw-oracle', 'n16'), 'sub20': ('n12', 'sub20', 'n12'), 'clean': ('n12', 'clean', 'n12'),
        'sub10': ('n14', 'sub10', 'n14'), 'sub05': ('n14', 'sub05', 'n14'), 'sub025': ('n14', 'sub025', 'n14'), 'sub0125': ('n14', 'sub0125', 'n14')}
CALIB_REFS = {'rw-b1': ('n13', 'n80-rw', 'n13'), 'rw-oracle': ('n15', 'rw-oracle', 'n15'), 'sub20': ('n12', 'sub20', 'n12'), 'clean': ('n12', 'clean', 'n12'),
              'sub10': ('n14', 'sub10', 'n14'), 'sub05': ('n14', 'sub05', 'n14'), 'sub025': ('n14', 'sub025', 'n14')}   # the calibration ladder has no sub0125
REF_JUDGE = {'rw-b1': 'B', 'rw-oracle': 'A'}         # whose prediction set head B's weights in the saved references
for _a in SPECS:
    n12.ARMS[_a] = (RHO, 'rw')                       # measured as N13's n80-rw is: a single softmax, its noisy-target prediction in the mixture form
for _a in ('sub10', 'sub05', 'sub025', 'sub0125'):
    n12.ARMS.setdefault(_a, (RHO, 'sub'))
n12.ARMS.setdefault('rw-b1', (RHO, 'rw'))
n12.ARMS.setdefault('rw-oracle', (RHO, 'rw'))
IGNORED = 0.01
CHECK_STEPS = 5 if TEST else 20
C_SEED = 31337                                       # head C's initial values: their own generator, the same in every arm of a seed


# ------------------------------------------------------------------ head C and the forward pass

class HeadC(nn.Module):
    """A third rank-RANK readout of the trunk's final state. Its initial values come from their own generator, so the main
    random state (which the trunk's dropout draws on) is untouched. like=model copies head B's values (the copied-peer check)."""

    def __init__(self, seed, like=None):
        super().__init__()
        with torch.random.fork_rng(devices=[]):
            self.proj = nn.Linear(study.CFG['d_model'], n4.RANK, bias=False)
            self.head = nn.Linear(n4.RANK, VOCAB, bias=False)
        gen = torch.Generator().manual_seed(seed + C_SEED)
        with torch.no_grad():
            for lin in (self.proj, self.head):                              # nn.Linear's default: uniform within 1/sqrt(fan_in)
                b = 1 / math.sqrt(lin.in_features)
                lin.weight.uniform_(-b, b, generator=gen)
            if like is not None:
                self.proj.weight.copy_(like.proj_b.weight)
                self.head.weight.copy_(like.head2.weight)
        self.to(DEV)

    def forward(self, z):
        return self.head(self.proj(z))


def forward_z(model, x):
    """n4.forward_n4's computation, step for step, also returning the trunk's final state z that the heads read."""
    T = x.shape[1]
    h = model.tok_emb(x) + model.pos_emb(torch.arange(T, device=x.device))
    for b in model.early_blocks:
        h = b(h)
    z = model.ode_block(h, None)
    for b in model.late_blocks:
        z = b(z)
    z = model.ln_f(z)
    return model.head(model.proj_a(z)), model.head2(model.proj_b(z)), z


# ------------------------------------------------------------------ targets, weights and losses

def judge_keep(ya, mask, seed, step, spec):
    """A loop-free head C's positions: N14's nested draw, inside K, at the rung's fraction (n14_dose.step_targets)."""
    if spec['c'] != 'clean':
        return None
    v = torch.rand(ya.shape, generator=torch.Generator().manual_seed(seed * 7919 + step * 31 + 17))
    return (~mask) & (v < spec['frac'] / FRAC_K)


def judge_weights(lb, lc, yb, log_u, spec):
    """Head B's per-position weights, without gradient, from the judge's logits: head C's, or head B's own."""
    src = lc if spec['judge'] == 'C' else lb
    with torch.no_grad():
        return n15.weight(src.detach().reshape(-1, VOCAB).float(), yb.reshape(-1), log_u, 1.0)


def head_b_loss(w, lb, yb):
    return (w * F.cross_entropy(lb.reshape(-1, VOCAB).float(), yb.reshape(-1), reduction='none')).mean()


def head_c_loss(lc, w, ya, yb, keep_c, log_u, spec):
    """The peer's: its own discount on head B's targets (w when head C judges, so the weight is computed once). A loop-free
    judge's: full weight on its clean targets, summed over its positions and divided by all positions (N14's convention)."""
    if spec['c'] == 'self':
        r = w if spec['judge'] == 'C' else judge_weights(None, lc, yb, log_u, {'judge': 'C'})
        return (r * F.cross_entropy(lc.reshape(-1, VOCAB).float(), yb.reshape(-1), reduction='none')).mean()
    ce = F.cross_entropy(lc.reshape(-1, VOCAB).float(), ya.reshape(-1), reduction='none')
    return (ce * keep_c.reshape(-1).float()).sum() / ce.numel()


# ------------------------------------------------------------------ one step: the main group as N13-N16, head C tied to it one way

def optim_step(model, hc, opt, opt_c, scaler, loss, loss_c, inject=None):
    """Backward, unscale, clip and step. The main group (trunk, head A, head B) is exactly N16's. Head C's loss is scaled by the
    main scale, its gradient unscaled by the same reciprocal and clipped by the same coefficient, and it steps only when the main
    group steps; an overflow in head C alone skips only head C's step. inject ('C' or 'main') puts an inf into one gradient after
    the backward pass, for the overflow test. Returns (main step skipped, head C step skipped)."""
    scaler.scale(loss).backward()
    if hc is not None:
        (loss_c * scaler._scale.to(device=loss_c.device, non_blocking=True) if scaler.is_enabled() else loss_c).backward()
    if inject == 'C':
        next(hc.parameters()).grad.view(-1)[0] = float('inf')
    elif inject == 'main':
        next(model.parameters()).grad.view(-1)[0] = float('inf')
    scaler.unscale_(opt)
    total = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    c_bad = False
    if hc is not None:
        gc = [p.grad for p in hc.parameters()]
        if scaler.is_enabled():
            found = torch.full((), 0.0, dtype=torch.float32, device=scaler._scale.device)
            torch._amp_foreach_non_finite_check_and_unscale_(gc, found, scaler._scale.double().reciprocal().float())
            c_bad = found.item() > 0
        else:
            c_bad = not all(bool(torch.isfinite(x).all()) for x in gc)
        torch.nn.utils.clip_grads_with_norm_(hc.parameters(), 1.0, total)
    before = scaler.get_scale()
    scaler.step(opt)
    scaler.update()
    main_skipped = scaler.is_enabled() and scaler.get_scale() < before
    c_skipped = hc is not None and (main_skipped or c_bad)
    if hc is not None and not c_skipped:
        opt_c.step()
    return main_skipped, c_skipped


def step_forward(model, hc, x, ya, yb, keep_c, log_u, spec):
    with torch.autocast(DEV.type, dtype=torch.float16, enabled=AMP):
        la, lb, z = forward_z(model, x)
        lc = hc(z.detach()) if hc is not None else None
        w = judge_weights(lb, lc, yb, log_u, spec)
        loss = 0.5 * (F.cross_entropy(la.view(-1, VOCAB), ya.reshape(-1)) + head_b_loss(w, lb, yb))
        loss_c = 0.5 * head_c_loss(lc, w, ya, yb, keep_c, log_u, spec) if hc is not None else None
    return la, lb, lc, w, loss, loss_c


def fresh(seed, spec, copy_b=False):
    model = n4.new_n4(seed, BASE)
    hc = HeadC(seed, like=model if copy_b else None) if spec['c'] is not None else None
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR)
    opt_c = torch.optim.AdamW(hc.parameters(), lr=PEAK_LR) if hc is not None else None
    return model, hc, opt, opt_c, torch.amp.GradScaler(DEV.type, enabled=AMP)


def run_steps(seed, spec, steps, train, probs, log_u, copy_b=False):
    """Train a fresh model for a few steps with this file's step: the passenger and copied-peer checks."""
    model, hc, opt, opt_c, scaler = fresh(seed, spec, copy_b)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for step in range(steps):
        for o in (opt, opt_c):
            if o is not None:
                for grp in o.param_groups:
                    grp['lr'] = study.lr_at(step, TOTAL, PEAK_LR)
        x, ys, _ = n12.draw_batch(train, gen)
        ya = ys[0]
        yb, keep, mask = n12.step_targets(ya, probs, seed, step, RHO, 'rw')
        keep_c = judge_keep(ya, mask, seed, step, spec)
        x, ya, yb = x.to(DEV), ya.to(DEV), yb.to(DEV)
        keep_c = keep_c.to(DEV) if keep_c is not None else None
        opt.zero_grad(set_to_none=True)
        if opt_c is not None:
            opt_c.zero_grad(set_to_none=True)
        *_, loss, loss_c = step_forward(model, hc, x, ya, yb, keep_c, log_u, spec)
        optim_step(model, hc, opt, opt_c, scaler, loss, loss_c)
    return model, hc


def max_difference(a, b):
    return max((x - y).abs().max().item() for x, y in zip(a, b))


# ------------------------------------------------------------------ the setup checks (precondition 1)

def forward_check(model, x):
    """forward_z gives n4.forward_n4's logits exactly."""
    model.eval()
    with torch.no_grad():
        la, lb = n4.forward_n4(model, x)
        la2, lb2, _ = forward_z(model, x)
    model.train()
    return {'pass': bool(torch.equal(la, la2) and torch.equal(lb, lb2))}


def detachment_check(seed, x, ya, yb, keep_c_peer, keep_c_clean, log_u):
    """Head B's loss sends no gradient to head C; head C's loss (peer or loop-free) sends none to the trunk or heads A and B;
    head B's loss does reach the trunk and head B."""
    model, hc = n4.new_n4(seed, BASE), HeadC(seed)
    model.eval()
    out = {}
    for name, spec, kc in (('peer', SPECS['rw-peer'], keep_c_peer), ('loop_free', SPECS['rw-j10'], keep_c_clean)):
        la, lb, z = forward_z(model, x)
        lc = hc(z.detach())
        w = judge_weights(lb, lc, yb, log_u, spec)
        g_c = torch.autograd.grad(head_b_loss(w, lb, yb), list(hc.parameters()), allow_unused=True)
        la, lb, z = forward_z(model, x)
        lc = hc(z.detach())
        w = judge_weights(lb, lc, yb, log_u, spec)
        g_main = torch.autograd.grad(head_c_loss(lc, w, ya, yb, kc, log_u, spec), list(model.parameters()), allow_unused=True)
        la, lb, z = forward_z(model, x)
        lc = hc(z.detach())
        w = judge_weights(lb, lc, yb, log_u, spec)
        g_b = torch.autograd.grad(head_b_loss(w, lb, yb), list(model.parameters()), allow_unused=True)
        out[name] = {'head_C_from_head_B_loss': all(x_ is None or x_.abs().max().item() == 0 for x_ in g_c),
                     'main_from_head_C_loss': all(x_ is None or x_.abs().max().item() == 0 for x_ in g_main),
                     'head_B_loss_reaches_main': max((x_.abs().max().item() for x_ in g_b if x_ is not None), default=0.0) > 0}
    out['pass'] = all(all(v.values()) for v in out.values())
    return out


def judge_source_check(seed, x, yb, log_u):
    """With head C at its own initial values, the weight in head B's loss is exactly the weight from head C's logits, and
    differs from head B's own discount."""
    model, hc = n4.new_n4(seed, BASE), HeadC(seed)
    model.eval()
    with torch.no_grad():
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=AMP):
            la, lb, z = forward_z(model, x)
            lc = hc(z.detach())
        w = judge_weights(lb, lc, yb, log_u, SPECS['rw-peer'])
        from_c = n15.weight(lc.reshape(-1, VOCAB).float(), yb.reshape(-1), log_u, 1.0)
        own = n15.weight(lb.reshape(-1, VOCAB).float(), yb.reshape(-1), log_u, 1.0)
    diff_own = (w - own).abs().max().item()
    return {'equals_head_C_weight': bool(torch.equal(w, from_c)), 'max_difference_from_own_discount': diff_own,
            'pass': bool(torch.equal(w, from_c)) and diff_own > 0}


def passenger_check(seed, train, probs, log_u):
    """CHECK_STEPS steps of rw-b1 with a passenger head C leave the trunk and heads A and B bit-identical to the same steps without."""
    m0, _ = run_steps(seed, SPECS['rw-b1-plain'], CHECK_STEPS, train, probs, log_u)
    m1, _ = run_steps(seed, SPECS['rw-b1-passenger'], CHECK_STEPS, train, probs, log_u)
    a, b = list(m0.state_dict().values()), list(m1.state_dict().values())
    return {'steps': CHECK_STEPS, 'identical': all(torch.equal(x, y) for x, y in zip(a, b)), 'max_difference': max_difference(a, b),
            'pass': all(torch.equal(x, y) for x, y in zip(a, b))}


def copied_peer_check(seed, train, probs, log_u):
    """A peer started as an exact copy of head B, judging head B, gives rw-b1's trunk and heads bit for bit, and stays head B."""
    m0, _ = run_steps(seed, SPECS['rw-b1-plain'], CHECK_STEPS, train, probs, log_u)
    m1, hc = run_steps(seed, SPECS['rw-peer'], CHECK_STEPS, train, probs, log_u, copy_b=True)
    a, b = list(m0.state_dict().values()), list(m1.state_dict().values())
    same_main = all(torch.equal(x, y) for x, y in zip(a, b))
    same_c = torch.equal(hc.proj.weight, m1.proj_b.weight) and torch.equal(hc.head.weight, m1.head2.weight)
    return {'steps': CHECK_STEPS, 'main_identical': same_main, 'max_main_difference': max_difference(a, b), 'head_C_equals_head_B': bool(same_c),
            'max_head_C_difference': max_difference([hc.proj.weight, hc.head.weight], [m1.proj_b.weight, m1.head2.weight]),
            'pass': same_main and bool(same_c)}


def overflow_test(seed, x, ya, yb, keep_c, log_u):
    """An inf injected into head C's gradient alone skips only head C's step and leaves the main scale unchanged; one injected
    into the main group's gradient skips both steps. Run with an enabled scaler on this device, whatever AMP is."""
    out = {}
    for where in ('C', 'main'):
        model, hc = n4.new_n4(seed, BASE), HeadC(seed)
        opt, opt_c = torch.optim.AdamW(model.parameters(), lr=PEAK_LR), torch.optim.AdamW(hc.parameters(), lr=PEAK_LR)
        scaler = torch.amp.GradScaler(DEV.type, enabled=True)
        before_m = [p.detach().clone() for p in model.parameters()]
        before_c = [p.detach().clone() for p in hc.parameters()]
        model.train()
        *_, loss, loss_c = step_forward(model, hc, x, ya, yb, keep_c, log_u, SPECS['rw-peer'])
        scaler.scale(loss.detach())                                         # initialises the scale, as the first step's scaler.scale does
        scale_before = scaler.get_scale()
        main_skipped, c_skipped = optim_step(model, hc, opt, opt_c, scaler, loss, loss_c, inject=where)
        out[where] = {'main_skipped': main_skipped, 'head_C_skipped': c_skipped,
                      'main_moved': any(not torch.equal(a, b) for a, b in zip(before_m, model.parameters())),
                      'head_C_moved': any(not torch.equal(a, b) for a, b in zip(before_c, hc.parameters())),
                      'scale_before': scale_before, 'scale_after': scaler.get_scale()}
    o_c, o_m = out['C'], out['main']
    out['pass'] = (not o_c['main_skipped'] and o_c['head_C_skipped'] and o_c['main_moved'] and not o_c['head_C_moved']
                   and o_c['scale_after'] == o_c['scale_before']
                   and o_m['main_skipped'] and o_m['head_C_skipped'] and not o_m['main_moved'] and not o_m['head_C_moved']
                   and o_m['scale_after'] < o_m['scale_before'])
    return out


# ------------------------------------------------------------------ the judge's measures on the measurement windows

@torch.no_grad()
def judge_measure(model, judge, mx, y_clean, log_u, hc=None, batch=n12.MEASURE_BATCH):
    """On the measurement windows' clean targets: the judge's loss and head A's (x = their difference); the clean targets ignored
    (weight below IGNORED) under the judge and under head B's own discount; the share of head B's own ignored set the judge also
    ignores; head B's loss on each set. judge is 'C' (hc), 'B' or 'A'."""
    model.eval()
    if hc is not None:
        hc.eval()
    tot = dict.fromkeys(('judge', 'A', 'B', 'B_on_judge_ignored', 'B_on_own_ignored'), 0.0)
    n_all = n_j = n_own = n_both = 0
    for i in range(0, len(mx), batch):
        x, y = mx[i:i + batch].to(DEV), y_clean[i:i + batch].to(DEV)
        la, lb, z = forward_z(model, x)
        lj = {'C': lambda: hc(z), 'B': lambda: lb, 'A': lambda: la}[judge]()
        yf = y.reshape(-1)
        laf, lbf, ljf = (t.reshape(-1, VOCAB).float() for t in (la, lb, lj))
        ce_j, ce_a, ce_b = (F.cross_entropy(t, yf, reduction='none') for t in (ljf, laf, lbf))
        ign_j = n15.weight(ljf, yf, log_u, 1.0) < IGNORED
        ign_own = n15.weight(lbf, yf, log_u, 1.0) < IGNORED
        tot['judge'] += ce_j.sum().item(); tot['A'] += ce_a.sum().item(); tot['B'] += ce_b.sum().item()
        tot['B_on_judge_ignored'] += ce_b[ign_j].sum().item(); tot['B_on_own_ignored'] += ce_b[ign_own].sum().item()
        n_all += yf.numel(); n_j += int(ign_j.sum()); n_own += int(ign_own.sum()); n_both += int((ign_j & ign_own).sum())
    return {'judge': judge, 'threshold': IGNORED, 'n_targets': n_all,
            'judge_clean': tot['judge'] / n_all, 'A_clean': tot['A'] / n_all, 'B_clean': tot['B'] / n_all,
            'x': (tot['judge'] - tot['A']) / n_all,
            'judge_ignored_fraction': n_j / n_all, 'own_ignored_fraction': n_own / n_all,
            'share_of_own_ignored_also_judge_ignored': n_both / n_own if n_own else None,
            'loss_B_on_judge_ignored': tot['B_on_judge_ignored'] / n_j if n_j else None,
            'loss_B_on_own_ignored': tot['B_on_own_ignored'] / n_own if n_own else None}


# ------------------------------------------------------------------ training

def train_n17(store, meta, arm, seed, tag, train, probs, log_u):
    spec = SPECS[arm]
    res_path = store.L(f'results/{tag}.json')
    if res_path.exists():
        log(f'{tag}: already finished, skipping')
        return json.loads(res_path.read_text())
    ck_rel = f'ckpt/{tag}.pt'
    store.pull(ck_rel)
    ck_path = store.L(ck_rel)
    vx, vys = n6.windows(study.load_tokens(store, meta, 'validation'), limit=g.VAL_LIMIT)
    vya, vyb = vys[0], n10.fixed_noisy(vys[0], probs, RHO, n12.VAL_NOISE_SEED)
    model, hc, opt, opt_c, scaler = fresh(seed, spec)
    gen = torch.Generator().manual_seed(seed)
    state = {'step': 0, 'train_loss': [], 'val': [], 'alpha': [], 'w': [], 'judge': [], 'skipped_steps': 0, 'c_skipped_steps': 0,
             'seconds': 0.0, 'resumes': 0, 'max_offset': -1}
    if ck_path.exists():
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); scaler.load_state_dict(ck['scaler'])
        if hc is not None:
            hc.load_state_dict(ck['hc']); opt_c.load_state_dict(ck['opt_c'])
        gen.set_state(ck['gen'])
        torch.set_rng_state(ck['torch_rng'])
        if DEV.type == 'cuda' and ck.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
        state = ck['state']; state['resumes'] += 1
        log(f"{tag}: resuming at step {state['step']}")
    else:
        what = ('head C judges itself (the peer)' if spec['c'] == 'self' else
                f"head C at full weight on {spec['frac']:.2%} clean targets" if spec['c'] == 'clean' else 'no head C')
        log(f"{tag}: starting, noise {RHO}, head B weighted by head {spec['judge']}'s prediction, {what}, seed {seed}")

    def save_ckpt():
        tmp = ck_path.with_suffix('.tmp')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(),
                    'hc': hc.state_dict() if hc is not None else None, 'opt_c': opt_c.state_dict() if opt_c is not None else None,
                    'gen': gen.get_state(), 'state': state, 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if DEV.type == 'cuda' else None}, tmp)
        os.replace(tmp, ck_path)
        store.push(ck_rel)

    model.train()
    if hc is not None:
        hc.train()
    last_ck, t_seg = time.time(), time.time()
    while state['step'] < TOTAL:
        step = state['step']
        for o in (opt, opt_c):
            if o is not None:
                for grp in o.param_groups:
                    grp['lr'] = study.lr_at(step, TOTAL, PEAK_LR)
        x, ys, top = n12.draw_batch(train, gen)
        state['max_offset'] = max(state['max_offset'], top)
        ya = ys[0]
        yb, keep, mask = n12.step_targets(ya, probs, seed, step, RHO, 'rw')
        keep_c = judge_keep(ya, mask, seed, step, spec)
        if step == 0:
            state.update({'batch_fingerprint_step0': n12.fingerprint(x), 'replaced_positions_fingerprint_step0': n12.fingerprint(mask),
                          'noise_fraction_step0': mask.float().mean().item(), 'kept_fraction_step0': keep.float().mean().item(),
                          'judge_keep_fingerprint_step0': n12.fingerprint(keep_c) if keep_c is not None else None,
                          'judge_kept_fraction_step0': keep_c.float().mean().item() if keep_c is not None else None,
                          'judge_keep_inside_K_step0': bool((keep_c & mask).sum() == 0) if keep_c is not None else None})
        x, ya, yb, keep = x.to(DEV), ya.to(DEV), yb.to(DEV), keep.to(DEV)
        keep_c = keep_c.to(DEV) if keep_c is not None else None
        opt.zero_grad(set_to_none=True)
        if opt_c is not None:
            opt_c.zero_grad(set_to_none=True)
        la, lb, lc, w, loss, loss_c = step_forward(model, hc, x, ya, yb, keep_c, log_u, spec)
        if step % 50 == 0:                                                      # the weights and the judge's accuracy, no gradient
            with torch.no_grad():
                k = keep.reshape(-1)
                sk, sr = w[k].sum().item(), w[~k].sum().item()
                state['w'].append([step, w.mean().item(), w[k].mean().item(), w[~k].mean().item(), sr / (sr + sk), (w[k] < IGNORED).float().mean().item()])
                lj = lc if spec['judge'] == 'C' else lb
                yf = ya.reshape(-1)
                state['judge'].append([step, F.cross_entropy(lj.reshape(-1, VOCAB).float(), yf).item(), F.cross_entropy(la.reshape(-1, VOCAB).float(), yf).item(),
                                       loss_c.item() if loss_c is not None else None])
        main_skipped, c_skipped = optim_step(model, hc, opt, opt_c, scaler, loss, loss_c)
        state['skipped_steps'] += int(main_skipped)
        state['c_skipped_steps'] += int(c_skipped and not main_skipped)
        state['step'] = step + 1
        if step % 50 == 0:
            state['train_loss'].append([step, loss.item()])
            state['alpha'].append([step, model.ode_block.func.output_scale.item()])
        if state['step'] == 500 or state['step'] % g.VAL_EVERY == 0 or state['step'] == TOTAL:
            l_a, l_b = n12.val_losses(model, vx, vya, vyb, 'rw', log_u)[:2]
            state['val'].append([state['step'], l_a, l_b])
            log(f"{tag}: step {state['step']}/{TOTAL}, validation loss head A {l_a:.4f}, head B {l_b:.4f}")
        if time.time() - last_ck > study.CKPT_SECONDS:
            state['seconds'] += time.time() - t_seg; t_seg = time.time()
            save_ckpt(); last_ck = time.time()
    state['seconds'] += time.time() - t_seg
    save_ckpt()
    torch.save(model.state_dict(), store.L(f'final/{tag}.pt'))
    store.push(f'final/{tag}.pt')
    if hc is not None:
        torch.save(hc.state_dict(), store.L(f'final/{tag}_headC.pt'))
        store.push(f'final/{tag}_headC.pt')
    fx, fys = n6.windows(study.load_tokens(store, meta, 'validation'), limit=g.FINAL_LIMIT)
    l_a, l_b, l_b_clean, l_inner = n12.val_losses(model, fx, fys[0], n10.fixed_noisy(fys[0], probs, RHO, n12.VAL_NOISE_SEED), 'rw', log_u)
    last = state['w'][-1]
    result = {'tag': tag, 'arm': arm, 'seed': seed, 'rho': RHO, **{f'spec_{k}': v for k, v in spec.items()}, 'head_b': 'rw', 'readout_rank': n4.RANK,
              'final_loss_A': l_a, 'final_loss_B': l_b, 'final_loss_B_clean_targets': l_b_clean, 'final_loss_B_inner_clean': l_inner,
              'final_val_windows': len(fx), 'val_curve': state['val'], 'train_loss_every_50': state['train_loss'],
              'w_every_50': state['w'],              # [step, mean w, mean w on kept, mean w on replaced, random share, fraction of kept with w < 0.01]
              'judge_every_50': state['judge'],      # [step, the judge's loss on the batch's clean targets, head A's, head C's own loss (None without head C)]
              'alpha_every_50': state['alpha'], 'alpha_final': model.ode_block.func.output_scale.item(),
              **{k: state.get(k) for k in ('batch_fingerprint_step0', 'replaced_positions_fingerprint_step0', 'noise_fraction_step0', 'kept_fraction_step0',
                                            'judge_keep_fingerprint_step0', 'judge_kept_fraction_step0', 'judge_keep_inside_K_step0')},
              'max_offset': state['max_offset'], 'training_tokens': len(train),
              'skipped_steps_fp16': state['skipped_steps'], 'head_C_only_skipped_steps': state['c_skipped_steps'],
              'train_seconds': state['seconds'], 'resumes': state['resumes'],
              'n_params': sum(p.numel() for p in model.parameters()), 'n_params_head_C': sum(p.numel() for p in hc.parameters()) if hc is not None else 0,
              'environment': study.env_record()}
    res_path.write_text(json.dumps(result, indent=2))
    store.push(f'results/{tag}.json')
    log(f'{tag}: finished in {state["seconds"] / 3600:.2f} h; head A {l_a:.4f}, head B {l_b:.4f} (clean targets {l_b_clean:.4f}, inner {l_inner:.4f}); '
        f'final share {last[4]:.3f}, ignored kept {last[5]:.3f}')
    return result


def load_head_c(store, tag, seed):
    if not store.pull(f'final/{tag}_headC.pt'):
        raise SystemExit(f'head C weights {store.drive / "final" / (tag + "_headC.pt")} are missing locally and on Drive.')
    hc = HeadC(seed)
    hc.load_state_dict(torch.load(store.L(f'final/{tag}_headC.pt'), map_location=DEV, weights_only=False))
    return hc


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--source', default=None)
    ap.add_argument('--local', default=None)
    FOLDERS = (('n12', 'n12_bottleneck', 'n12_calib'), ('n13', 'n13_signal', 'n13_calib'), ('n14', 'n14_dose', 'n14_calib'),
               ('n15', 'n15_sharpen', 'n15_calib'), ('n16', 'n16_judge', 'n16_calib'))
    for k, real, cal in FOLDERS:
        ap.add_argument(f'--{k}', default=None, help=f"{k.upper()}'s Drive folder (default: {real}, or {cal}, next to --work)")
        ap.add_argument(f'--{k}-records', default=None, help=f"{k.upper()}'s committed records (default: results/{k} or results/{k}_calib here)")
    ap.add_argument('--calibration', action='store_true', help="the calibration run: seed 3001, the tail's windows 513-1024, every rung")
    ap.add_argument('--rungs', default=','.join(DEFAULT_RUNGS), help="the real run's loop-free rungs, as the calibration's rule chose them")
    args = ap.parse_args()
    mode = 'calibration' if args.calibration else 'real'
    seeds, (w0, w1) = MODES[mode]['seeds'], MODES[mode]['windows']
    n6.SEEDS = g.SEEDS = seeds
    work = Path(args.work)
    source = Path(args.source) if args.source else work.parent / 'wt103_study'
    folders, records = {}, {}
    for k, real, cal in FOLDERS:
        v = getattr(args, k); r = getattr(args, f'{k}_records')
        folders[k] = Path(v) if v else work.parent / (cal if args.calibration else real)
        records[k] = Path(r) if r else HERE / 'results' / (f'{k}_calib' if args.calibration else k)
    if args.local:
        local = Path(args.local)
    elif Path('/content').is_dir() and not TEST:
        local = Path('/content/cdt_local') / work.name
    else:
        local = work.parent / f'_local_{work.name}'
    data_store = study.Store(source, local)
    store = study.Store(work, local)
    refs = CALIB_REFS if args.calibration else REFS
    srcs = {k: study.Store(folders[k], local.parent / f'_n17_src_{folders[k].name}') for k in {v[0] for v in refs.values()}}
    if args.calibration:
        new_arms = ['rw-peer', *RUNGS, *CALIB_ONLY]
    else:
        rungs = [a for a in args.rungs.split(',') if a]
        bad = [a for a in rungs if a not in RUNGS]
        if bad or not 2 <= len(rungs) <= 3:
            raise SystemExit(f'--rungs must name two or three of {list(RUNGS)}; got {rungs}.')
        new_arms = ['rw-peer', *rungs]
    out = Path(args.out)
    (out / 'results').mkdir(parents=True, exist_ok=True)
    log('device:', DEV, '| test mode:', TEST, '| mode:', mode, '| study folder (data):', source, '| N17 folder:', work, '| local:', local)
    log('arms trained:', new_arms, '| references:', list(refs), '| saved models from:', {k: str(folders[k]) for k in srcs})
    store.pull_dir('results')
    try:
        meta = study.prepare_data(data_store, None)
        full = study.load_tokens(store, meta, 'train')
        cut = int(len(full) * n12.CUT_FRACTION)
        train, tail = full[:cut], full[cut:]
        probs = n12.n7.unigram(train)
        log_u = probs.float().log().to(DEV)
        summary_path = store.L('results/n17_summary.json')
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}

        def save_json(rel, obj):
            store.L(rel).write_text(json.dumps(obj, indent=2 if rel.endswith('n17_summary.json') else None))
            store.push(rel)

        allx, allys = n6.windows(tail, limit=w1)
        mx, mys = allx[w0:w1], [y[w0:w1] for y in allys]
        u = n8.uniforms(len(mx))
        log(f"training region: tokens 0-{cut} of {len(full)}; measured on the held-out tail's windows {w0 + 1}-{w0 + len(mx)}")

        if 'checks' not in summary:
            s0 = seeds[0]
            x, ys, _ = n12.draw_batch(train, torch.Generator().manual_seed(s0))
            yb, keep, mask = n12.step_targets(ys[0], probs, s0, 0, RHO, 'rw')
            kc_clean = judge_keep(ys[0], mask, s0, 0, SPECS['rw-j10'])
            xd, yad, ybd = x.to(DEV), ys[0].to(DEV), yb.to(DEV)
            m0 = n4.new_n4(s0, BASE)
            log(f'setup checks: {CHECK_STEPS}-step passenger and copied-peer runs, the overflow test')
            summary['checks'] = {
                'batch': n6.batch_check(train), 'batch_g1': g.batch_check(train),
                'same_model': [g.same_model_check(train, s) for s in seeds],
                'windows_shifted': all(torch.equal(mys[k - 1][:, :-k], mx[:, k:]) for k in range(1, 9)),
                'analytic': n9.analytic_check(),
                'identical_heads': n8.identical_heads_check(s0, mx, u),
                'weight_identity': n15.weight_identity_check(m0, xd, ybd, log_u),
                'forward_identity': forward_check(m0, xd),
                'detachment': detachment_check(s0, xd, yad, ybd, None, kc_clean.to(DEV), log_u),
                'judge_source': judge_source_check(s0, xd, ybd, log_u),
                'passenger': passenger_check(s0, train, probs, log_u),
                'copied_peer': copied_peer_check(s0, train, probs, log_u),
                'overflow': overflow_test(s0, xd, yad, ybd, None, log_u),
                'nesting': {'inside_K': bool((kc_clean & mask).sum() == 0),
                            'nested': all(bool((judge_keep(ys[0], mask, s0, 0, SPECS[a]) & ~judge_keep(ys[0], mask, s0, 0, SPECS[b])).sum() == 0)
                                          for a, b in zip(list(RUNGS)[1:], list(RUNGS)[:-1])),
                            'top_rung_is_K': bool(torch.equal(judge_keep(ys[0], mask, s0, 0, SPECS['rw-j20']), ~mask)),
                            'fractions': {a: judge_keep(ys[0], mask, s0, 0, SPECS[a]).float().mean().item() for a in RUNGS}},
                'unigram': {'sum': probs.sum().item(), 'zeros': int((probs == 0).sum()), 'from_tokens': cut}}
            nc = summary['checks']['nesting']
            nc['pass'] = nc['inside_K'] and nc['nested'] and nc['top_rung_is_K'] and all(abs(nc['fractions'][a] - f) <= 0.02 for a, f in RUNGS.items())
            summary['mode'] = mode
            summary['held_out'] = {'cut_fraction': n12.CUT_FRACTION, 'cut_token': cut, 'training_tokens': len(full), 'tail_tokens': len(tail),
                                   'tail_windows': (len(tail) - n12.K_MAX) // n12.SEQ, 'measure': [w0 + 1, w0 + len(mx)]}
            summary['arms'] = {a: SPECS[a] for a in new_arms}
            summary['references'] = {a: {'folder': str(folders[v[0]]), 'tag': v[1], 'records': str(records[v[2]])} for a, v in refs.items()}
            summary['seeds'], summary['readout_rank'], summary['subspace_rank'] = seeds, n4.RANK, n9.R
            save_json('results/n17_summary.json', summary)
        c = summary['checks']
        names = ('weight_identity', 'forward_identity', 'detachment', 'judge_source', 'passenger', 'copied_peer', 'overflow', 'nesting')
        log('batch check:', c['batch']['targets_are_i+1_to_i+8'], '| windows shifted:', c['windows_shifted'], '| analytic:', c['analytic']['pass'],
            '| identical heads:', c['identical_heads']['pass'], '|', ' | '.join(f"{k}: {c[k]['pass']}" for k in names))
        log('same-model check:', [x['pass'] for x in c['same_model']], '| passenger largest difference:', c['passenger']['max_difference'],
            '| copied peer largest differences:', c['copied_peer']['max_main_difference'], c['copied_peer']['max_head_C_difference'])
        if not (c['batch']['targets_are_i+1_to_i+8'] and c['windows_shifted'] and c['analytic']['pass'] and c['identical_heads']['pass']
                and all(c[k]['pass'] for k in names) and all(x['pass'] for x in c['same_model']) and abs(c['unigram']['sum'] - 1) < 1e-6):
            raise SystemExit('A check of the setup failed before training. Stopping.')

        def measure_all(m, arm, tag, judge, hc=None):
            return {'tag': tag, 'seed': seed, 'arm': arm, 'model': tag, **n12.measure(m, arm, mx, mys, u, log_u, probs),
                    'neglect': n16.neglect(m, mx, mys, log_u),
                    'judge_measure': judge_measure(m, judge, mx, mys[0], log_u, hc) if judge else None, 'environment': study.env_record()}

        for seed in seeds:
            for arm, (fkey, prefix, rkey) in refs.items():                   # the saved models, re-measured (precondition 2)
                rel = f'results/r_{arm}_seed{seed}.json'
                if store.L(rel).exists():
                    continue
                tag = f'{prefix}_seed{seed}'
                m = n13.load_model(srcs[fkey], tag, seed)
                res = measure_all(m, arm, tag, REF_JUDGE.get(arm))
                res['tag'] = f'{arm}_seed{seed}'
                ref_path = records[rkey] / f'm_{tag}.json'
                res['reproduction'] = n13.reproduction(res, json.loads(ref_path.read_text())) if ref_path.exists() else {'pass': False, 'missing': str(ref_path)}
                save_json(rel, res)
                fn, rep = res['functional'], res['reproduction']
                a = (res.get('B_inner_vs_A') or res['B_vs_A'])['agreement_ratio']
                jm = res['judge_measure']
                log(f"r_{arm}_seed{seed}: inner clean {fn['B_inner_clean']:.3f} | head A clean {fn['A_clean']:.3f} | agreement with A {a:.3f} | "
                    + (f"judge {jm['judge']} x {jm['x']:+.3f} | " if jm else '')
                    + f"reproduces: {rep.get('pass')} (loss {rep.get('max_loss_difference', float('nan')):.1e}, ratio {rep.get('max_ratio_difference', float('nan')):.1e}) | {res['seconds']:.0f} s")
            for arm in new_arms:
                tag = f'{arm}_seed{seed}'
                train_n17(store, meta, arm, seed, tag, train, probs, log_u)
                rel = f'results/m_{tag}.json'
                if store.L(rel).exists():
                    continue
                m = n13.load_model(store, tag, seed)
                hc = load_head_c(store, tag, seed) if SPECS[arm]['c'] is not None else None
                res = measure_all(m, arm, tag, SPECS[arm]['judge'], hc)
                save_json(rel, res)
                fn, jm = res['functional'], res['judge_measure']
                a = (res.get('B_inner_vs_A') or res['B_vs_A'])['agreement_ratio']
                log(f"m_{tag}: inner clean {fn['B_inner_clean']:.3f} | head A clean {fn['A_clean']:.3f} | agreement with A {a:.3f} | "
                    f"judge {jm['judge']} x {jm['x']:+.3f}, ignores {jm['judge_ignored_fraction']:.3f} (own {jm['own_ignored_fraction']:.3f}) | "
                    f"rotated {res['rotated_control']['agreement_ratio']:.3f} | {res['seconds']:.0f} s")
        if args.calibration and 'passenger_pair' not in summary:              # the full-length passenger pair, bit for bit
            s = seeds[0]
            a = torch.load(store.L(f'final/rw-b1-plain_seed{s}.pt'), map_location='cpu', weights_only=False)
            b = torch.load(store.L(f'final/rw-b1-passenger_seed{s}.pt'), map_location='cpu', weights_only=False)
            same = all(torch.equal(a[k], b[k]) for k in a)
            summary['passenger_pair'] = {'seed': s, 'identical': same, 'max_difference': max((a[k] - b[k]).abs().max().item() for k in a)}
            save_json('results/n17_summary.json', summary)
            log(f"full-length passenger pair: bit-identical {same}, largest difference {summary['passenger_pair']['max_difference']:.1e}")
        log(f'N17 {mode} complete: all runs and measurements finished')
    finally:
        dest = out / 'results'
        for f in (store.local / 'results').glob('*.json'):
            (dest / f.name).write_bytes(f.read_bytes())
        log(f'copied {len(list(dest.glob("*.json")))} result files for upload')
        log('waiting for background Drive saves to finish (up to 15 minutes)')
        store.flush()
        failures = store.failures + data_store.failures + sum((s.failures for s in srcs.values()), [])
        (out / 'drive_sync.json').write_text(json.dumps({'failures': failures}, indent=2))
        log('Drive saves finished' if not failures else f'Drive problems this session: {failures}')


if __name__ == '__main__':
    main()
