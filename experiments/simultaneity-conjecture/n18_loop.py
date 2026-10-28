"""N18: the loop, with the judge's data and weighting held fixed.

Implements N18_PREREGISTRATION.md in this folder, built on n17_loop.py: N17's head C, forward pass, position draw and judge
measures, N12's model, training, batches, windows and measures, N13's reweighted single-softmax head B, N15's weight. Every new
arm adds head C, the judge, a further readout of the trunk's final state, read detached; head B's loss is the mean over all
positions of stopgrad(w(y)) * CE, with w from head C's prediction.
  rw-cloop    head C on the clean targets at K, each weighted by head C's own discount (N14's convention for sub20-rw)
  rw-ctwin    head C on the same targets, each weighted by head T's discount; head T, the twin, at full weight on K, as
              rw-j20's judge (N16's convention for sub20-oracle, with the twin in head A's place)
  rw-j20, rw-j025, rw-j0125   N17's loop-free rungs: head C at full weight on N14's nested clean positions
  rw-peer     N17's peer, trained again: head C trained as rw-b1's head B is
  rw-nfree    exploratory: head C at full weight on head B's targets (N12's n80 loss); head B's weight unmixed,
              max(1 - 0.8 u / q_C, 0)
  rw-b1-plain, rw-b1-passenger2   the setup checks only: rw-b1 with no extra head, and with a peer head C and a twin, neither judging
  N17's rw-peer and rungs, N13's n80-rw, N16's (or N15's) rw-oracle, N12's sub20 and clean, N14's ladder   saved models, re-measured
Each extra head has its own AdamW and is tied to the main group one way, as N17's head C: its loss is scaled by the main scale,
its gradient unscaled by the same reciprocal and clipped by the main group's coefficient, it steps only when the main group
steps, and an overflow in its own gradient skips only its own step. The judge and the twin start from N17's initial values.

    python n18_loop.py --out OUT --work /content/drive/MyDrive/cdt-colab/n18_data
    python n18_loop.py --calibration --out OUT --work /content/drive/MyDrive/cdt-colab/n18_calib
"""
import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import n17_loop as n17  # noqa: E402  (imports N12, N13, N15 and N16; their patches of n12 stay in force)

n12, n13, n15, n16 = n17.n12, n17.n13, n17.n15, n17.n16
n10, n9, n8, n6, n4, g, study, DEV, TEST, log = n12.n10, n12.n9, n12.n8, n12.n6, n12.n4, n12.g, n12.study, n12.DEV, n12.TEST, n12.log
AMP, VOCAB, RHO, BASE, TOTAL, PEAK_LR = n12.AMP, n12.VOCAB, n12.RHO, n12.BASE, n12.TOTAL, n12.PEAK_LR
MODES, FRAC_K, IGNORED, CHECK_STEPS = n12.MODES, n17.FRAC_K, n17.IGNORED, n17.CHECK_STEPS
LOG_RHO = math.log(RHO)


def _import_n14():
    """N14's module, for its sub20-rw loss in the loss-form check alone. Importing it replaces n12.b_noisy and n12.head_b_loss,
    which the 'rw' measurement calls by name, and adds entries to n12.ARMS; all three are restored at once."""
    saved = n12.b_noisy, n12.head_b_loss, dict(n12.ARMS)
    import n14_dose
    n12.b_noisy, n12.head_b_loss = saved[:2]
    n12.ARMS.clear()
    n12.ARMS.update(saved[2])
    return n14_dose


n14 = _import_n14()

RUNGS = {'rw-j20': 0.2, 'rw-j025': 0.025, 'rw-j0125': 0.0125}          # the loop-free rungs trained here
N14_TAG = {'rw-j025': 'sub025', 'rw-j0125': 'sub0125'}               # the N14 arm with the same positions
_base = {'c': None, 'frac': None, 'judge': 'C', 'cw': None, 'twin': False, 'bw': 'discount'}
SPECS = {'rw-cloop': {**_base, 'c': 'clean', 'frac': FRAC_K, 'cw': 'self'},
         'rw-ctwin': {**_base, 'c': 'clean', 'frac': FRAC_K, 'cw': 'twin', 'twin': True},
         **{a: {**_base, 'c': 'clean', 'frac': f, 'cw': 'none'} for a, f in RUNGS.items()},
         'rw-peer': {**_base, 'c': 'self'},
         'rw-nfree': {**_base, 'c': 'noisy', 'bw': 'unmix'},
         'rw-b1-plain': {**_base, 'judge': 'B'},
         'rw-b1-passenger2': {**_base, 'c': 'self', 'judge': 'B', 'twin': True}}
CLAIM_ARMS = ('rw-cloop', 'rw-ctwin', 'rw-j20', 'rw-peer')
REAL_ARMS = ('rw-cloop', 'rw-ctwin', 'rw-j20', 'rw-peer', 'rw-j025', 'rw-j0125', 'rw-nfree')
CALIB_ARMS = ('rw-cloop', 'rw-ctwin', 'rw-j20', 'rw-peer', 'rw-nfree')     # rw-nfree last: an exploratory arm's failure cannot delay the claim's
IDENTITY = {'real': ('rw-peer',), 'calibration': ('rw-j20', 'rw-peer')}   # compared with N17's saved models of the same seed
N17_REAL = ('rw-peer', 'rw-j10', 'rw-j05')
N17_CALIB = ('rw-peer', 'rw-j20', 'rw-j10', 'rw-j05', 'rw-j025', 'rw-j0125')
REFS = {'rw-b1': ('n13', 'n80-rw', 'B'), 'rw-oracle': ('n16', 'rw-oracle', 'A'), 'sub20': ('n12', 'sub20', None), 'clean': ('n12', 'clean', None),
        'sub10': ('n14', 'sub10', None), 'sub05': ('n14', 'sub05', None), 'sub025': ('n14', 'sub025', None), 'sub0125': ('n14', 'sub0125', None),
        **{f'n17-{a}': ('n17', a, 'C') for a in N17_REAL}}
CALIB_REFS = {'rw-b1': ('n13', 'n80-rw', 'B'), 'rw-oracle': ('n15', 'rw-oracle', 'A'), 'sub20': ('n12', 'sub20', None), 'clean': ('n12', 'clean', None),
              'sub10': ('n14', 'sub10', None), 'sub05': ('n14', 'sub05', None), 'sub025': ('n14', 'sub025', None),
              **{f'n17-{a}': ('n17', a, 'C') for a in N17_CALIB}}
for _a in list(SPECS) + [f'n17-{a}' for a in N17_CALIB]:
    n12.ARMS[_a] = (RHO, 'rw')                       # measured as N13's n80-rw is: a single softmax, its noisy-target prediction in the mixture form
for _a in ('sub10', 'sub05', 'sub025', 'sub0125'):
    n12.ARMS.setdefault(_a, (RHO, 'sub'))
TWIN_OFFSET = 7_000_003                              # the weight-source checks' distinct twin: a second generator
REL_TOL, UNMIX_TOL = 1e-5, 1e-6


# ------------------------------------------------------------------ positions, weights and losses

def positions(ya, mask, seed, step, spec):
    """Head C's positions (N17's nested draw, for a clean-target judge) and the twin's (N17's rung at 20%, which is K)."""
    keep_c = n17.judge_keep(ya, mask, seed, step, spec)
    keep_t = n17.judge_keep(ya, mask, seed, step, {'c': 'clean', 'frac': FRAC_K}) if spec['twin'] else None
    return keep_c, keep_t


def unmixed_weight(f, y, log_u):
    """max(1 - 0.8 u(y) / q(y), 0) from logits f, computed in log space; 1 where u(y) = 0, a target that must be clean."""
    lq = torch.log_softmax(f, -1).gather(-1, y.unsqueeze(-1)).squeeze(-1)
    lu = log_u[y]
    w = (1 - torch.exp(LOG_RHO + lu - lq)).clamp(min=0)
    return torch.where(torch.isneginf(lu), torch.ones_like(w), w)


def weight_of(f, y, log_u, bw):
    return unmixed_weight(f, y, log_u) if bw == 'unmix' else n15.weight(f, y, log_u, 1.0)


def judge_weights(lb, lc, yb, log_u, spec):
    """Head B's per-position weights, without gradient, from the judge's logits (head C's, or head B's own), by the spec's formula."""
    src = lc if spec['judge'] == 'C' else lb
    with torch.no_grad():
        return weight_of(src.detach().reshape(-1, VOCAB).float(), yb.reshape(-1), log_u, spec['bw'])


def own_weights(src, ya, log_u):
    """The weight on each clean target of head C's own loss, without gradient: the discount from src (head C's or the twin's logits)."""
    with torch.no_grad():
        return n15.weight(src.detach().reshape(-1, VOCAB).float(), ya.reshape(-1), log_u, 1.0)


def head_c_weights(lc, lt, w, ya, yb, log_u, spec):
    """The weight on each term of head C's own loss: the peer's is w (its own discount on head B's targets, as in N17); a
    self-weighted or twin-weighted judge's is the discount on its clean targets; None for full weight."""
    if spec['c'] == 'self':
        return w if spec['judge'] == 'C' else judge_weights(None, lc, yb, log_u, {'judge': 'C', 'bw': 'discount'})
    if spec['cw'] == 'self':
        return own_weights(lc, ya, log_u)
    if spec['cw'] == 'twin':
        return own_weights(lt, ya, log_u)
    return None


def head_c_loss(lc, r_c, ya, yb, keep_c, spec):
    """The peer's: r_c * CE on head B's targets, the mean over positions (N17's). rw-nfree's: CE on head B's targets at full weight,
    the mean (N12's n80). A clean-target judge's: (r_c *) CE on its clean targets, summed over its positions and divided by all."""
    if spec['c'] == 'self':
        return (r_c * F.cross_entropy(lc.reshape(-1, VOCAB).float(), yb.reshape(-1), reduction='none')).mean()
    if spec['c'] == 'noisy':
        return F.cross_entropy(lc.reshape(-1, VOCAB).float(), yb.reshape(-1))
    ce = F.cross_entropy(lc.reshape(-1, VOCAB).float(), ya.reshape(-1), reduction='none')
    if r_c is not None:
        ce = r_c * ce
    return (ce * keep_c.reshape(-1).float()).sum() / ce.numel()


def twin_loss(lt, ya, keep_t):
    """The twin's: full weight on the clean targets at K, summed over them and divided by all positions (rw-j20's judge's)."""
    ce = F.cross_entropy(lt.reshape(-1, VOCAB).float(), ya.reshape(-1), reduction='none')
    return (ce * keep_t.reshape(-1).float()).sum() / ce.numel()


def mixture_logp(lf, log_u, bw):
    """A judge's mixture-form prediction, as log probabilities over the vocabulary: 0.2 p + 0.8 u for a judge with a clean
    prediction (N9's copy form); for rw-nfree, max(q, 0.8 u) normalised to sum to 1."""
    if bw == 'unmix':
        lm = torch.maximum(torch.log_softmax(lf.float(), -1), LOG_RHO + log_u)
        return lm - torch.logsumexp(lm, -1, keepdim=True)
    return n9.copy_logits(lf, log_u, RHO)


# ------------------------------------------------------------------ one step: the main group as N13-N17, each extra head tied to it one way

def optim_step(model, extras, opt, scaler, loss, inject=None):
    """Backward, unscale, clip and step. The main group (trunk, head A, head B) is exactly N16's and N17's. extras is a list of
    (head, its optimiser, its loss); each extra head's loss is scaled by the main scale, its gradient unscaled by the same
    reciprocal and clipped by the main group's coefficient, and it steps only when the main group steps; an overflow in an extra
    head's own gradient skips only that head's step. With one extra head this is N17's optim_step, operation for operation.
    inject ('main' or an index into extras) puts an inf into one gradient after the backward pass, for the overflow test.
    Returns (main step skipped, [each extra head's step skipped])."""
    scaler.scale(loss).backward()
    for h, _, l in extras:
        (l * scaler._scale.to(device=l.device, non_blocking=True) if scaler.is_enabled() else l).backward()
    if inject == 'main':
        next(model.parameters()).grad.view(-1)[0] = float('inf')
    elif inject is not None:
        next(extras[inject][0].parameters()).grad.view(-1)[0] = float('inf')
    scaler.unscale_(opt)
    total = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    bad = []
    for h, _, _ in extras:
        gh = [p.grad for p in h.parameters()]
        if scaler.is_enabled():
            found = torch.full((), 0.0, dtype=torch.float32, device=scaler._scale.device)
            torch._amp_foreach_non_finite_check_and_unscale_(gh, found, scaler._scale.double().reciprocal().float())
            bad.append(found.item() > 0)
        else:
            bad.append(not all(bool(torch.isfinite(x).all()) for x in gh))
        torch.nn.utils.clip_grads_with_norm_(h.parameters(), 1.0, total)
    before = scaler.get_scale()
    scaler.step(opt)
    scaler.update()
    main_skipped = scaler.is_enabled() and scaler.get_scale() < before
    skipped = [main_skipped or b for b in bad]
    for (_, o, _), s in zip(extras, skipped):
        if not s:
            o.step()
    return main_skipped, skipped


def step_forward(model, hc, ht, x, ya, yb, keep_c, keep_t, log_u, spec):
    """One forward pass: the logits, head B's weight w, the weight r_c on head C's own loss, and the three losses."""
    with torch.autocast(DEV.type, dtype=torch.float16, enabled=AMP):
        la, lb, z = n17.forward_z(model, x)
        lc = hc(z.detach()) if hc is not None else None
        lt = ht(z.detach()) if ht is not None else None
        w = judge_weights(lb, lc, yb, log_u, spec)
        loss = 0.5 * (F.cross_entropy(la.view(-1, VOCAB), ya.reshape(-1)) + n17.head_b_loss(w, lb, yb))
        r_c = head_c_weights(lc, lt, w, ya, yb, log_u, spec) if hc is not None else None
        loss_c = 0.5 * head_c_loss(lc, r_c, ya, yb, keep_c, spec) if hc is not None else None
        loss_t = 0.5 * twin_loss(lt, ya, keep_t) if ht is not None else None
    return la, lb, lc, lt, w, r_c, loss, loss_c, loss_t


def fresh(seed, spec, copy_b=False, twin_offset=0):
    model = n4.new_n4(seed, BASE)
    hc = n17.HeadC(seed, like=model if copy_b else None) if spec['c'] is not None else None
    ht = n17.HeadC(seed + twin_offset) if spec['twin'] else None
    opt = torch.optim.AdamW(model.parameters(), lr=PEAK_LR)
    opt_c = torch.optim.AdamW(hc.parameters(), lr=PEAK_LR) if hc is not None else None
    opt_t = torch.optim.AdamW(ht.parameters(), lr=PEAK_LR) if ht is not None else None
    return model, hc, ht, opt, opt_c, opt_t, torch.amp.GradScaler(DEV.type, enabled=AMP)


def extras_of(hc, ht, opt_c, opt_t, loss_c, loss_t):
    return [(h, o, l) for h, o, l in ((hc, opt_c, loss_c), (ht, opt_t, loss_t)) if h is not None]


def set_lr(opts, step):
    for o in opts:
        if o is not None:
            for grp in o.param_groups:
                grp['lr'] = study.lr_at(step, TOTAL, PEAK_LR)


def run_steps(seed, spec, steps, train, probs, log_u, copy_b=False):
    """Train a fresh model for a few steps with this file's step: the passenger and copied-peer checks."""
    model, hc, ht, opt, opt_c, opt_t, scaler = fresh(seed, spec, copy_b)
    gen = torch.Generator().manual_seed(seed)
    model.train()
    for step in range(steps):
        set_lr((opt, opt_c, opt_t), step)
        x, ys, _ = n12.draw_batch(train, gen)
        ya = ys[0]
        yb, keep, mask = n12.step_targets(ya, probs, seed, step, RHO, 'rw')
        keep_c, keep_t = positions(ya, mask, seed, step, spec)
        x, ya, yb = x.to(DEV), ya.to(DEV), yb.to(DEV)
        keep_c = keep_c.to(DEV) if keep_c is not None else None
        keep_t = keep_t.to(DEV) if keep_t is not None else None
        for o in (opt, opt_c, opt_t):
            if o is not None:
                o.zero_grad(set_to_none=True)
        *_, loss, loss_c, loss_t = step_forward(model, hc, ht, x, ya, yb, keep_c, keep_t, log_u, spec)
        optim_step(model, extras_of(hc, ht, opt_c, opt_t, loss_c, loss_t), opt, scaler, loss)
    return model, hc, ht


def rel_difference(a, b):
    return ((a - b).abs().max() / b.abs().max()).item()


# ------------------------------------------------------------------ the setup checks (precondition 1)

def detachment_check(seed, x, ya, yb, mask, log_u):
    """In each arm: the main loss sends no gradient to the extra heads; the judge's and the twin's losses send none to the trunk or
    heads A and B; the judge's loss sends none to the twin; head B's weighted loss reaches head B."""
    out = {}
    for arm in ('rw-cloop', 'rw-ctwin', 'rw-nfree', 'rw-peer', 'rw-j20'):
        spec = SPECS[arm]
        model, hc = n4.new_n4(seed, BASE), n17.HeadC(seed)
        ht = n17.HeadC(seed + TWIN_OFFSET) if spec['twin'] else None
        model.eval()
        kc, kt = positions(ya.cpu(), mask, seed, 0, spec)
        kc = kc.to(DEV) if kc is not None else None
        kt = kt.to(DEV) if kt is not None else None
        la, lb, lc, lt, w, r_c, loss, loss_c, loss_t = step_forward(model, hc, ht, x, ya, yb, kc, kt, log_u, spec)
        main, extra = list(model.parameters()), list(hc.parameters()) + (list(ht.parameters()) if ht is not None else [])
        zero = lambda gs: all(x_ is None or x_.abs().max().item() == 0 for x_ in gs)
        grad = lambda l, ps: torch.autograd.grad(l, ps, allow_unused=True, retain_graph=True)
        res = {'extra_heads_from_main_loss': zero(grad(loss, extra)),
               'main_from_judge_loss': zero(grad(loss_c, main)),
               'head_B_loss_reaches_head_B': max((g_.abs().max().item() for g_ in grad(n17.head_b_loss(w, lb, yb), [model.proj_b.weight, model.head2.weight])
                                                  if g_ is not None), default=0.0) > 0}
        if ht is not None:
            res['main_from_twin_loss'] = zero(grad(loss_t, main))
            res['twin_from_judge_loss'] = zero(grad(loss_c, list(ht.parameters())))
        out[arm] = res
    out['pass'] = all(all(v.values()) for v in out.values())
    return out


def weight_source_check(seed, x, ya, yb, mask, log_u):
    """With the twin at distinct values (TWIN_OFFSET): head B's weight in step_forward is exactly the judge's by the spec's formula,
    and differs from head B's own discount (and the twin's); the weight on head C's own loss is exactly the discount from its own
    logits (rw-cloop) or the twin's (rw-ctwin), and differs from head B's own discount (rw-cloop) or the judge's own (rw-ctwin);
    the twin's loss is the unweighted term."""
    out = {}
    for arm in ('rw-cloop', 'rw-ctwin', 'rw-nfree'):
        spec = SPECS[arm]
        model, hc = n4.new_n4(seed, BASE), n17.HeadC(seed)
        ht = n17.HeadC(seed + TWIN_OFFSET) if spec['twin'] else None
        model.eval()
        kc, kt = positions(ya.cpu(), mask, seed, 0, spec)
        kc = kc.to(DEV) if kc is not None else None
        kt = kt.to(DEV) if kt is not None else None
        with torch.no_grad():
            la, lb, lc, lt, w, r_c, loss, loss_c, loss_t = step_forward(model, hc, ht, x, ya, yb, kc, kt, log_u, spec)
            f = lambda t: t.reshape(-1, VOCAB).float()
            yfb, yfa = yb.reshape(-1), ya.reshape(-1)
            res = {'B_weight_is_judge_weight': bool(torch.equal(w, weight_of(f(lc), yfb, log_u, spec['bw']))),
                   'B_weight_differs_from_own_discount': (w - n15.weight(f(lb), yfb, log_u, 1.0)).abs().max().item() > 0}
            if ht is not None:
                res['B_weight_differs_from_twin'] = (w - weight_of(f(lt), yfb, log_u, spec['bw'])).abs().max().item() > 0
            if spec['cw'] == 'self':
                res['own_weight_is_judge_discount'] = bool(torch.equal(r_c, n15.weight(f(lc), yfa, log_u, 1.0)))
                res['own_weight_differs_from_head_B_discount'] = (r_c - n15.weight(f(lb), yfa, log_u, 1.0)).abs().max().item() > 0
            if spec['cw'] == 'twin':
                res['own_weight_is_twin_discount'] = bool(torch.equal(r_c, n15.weight(f(lt), yfa, log_u, 1.0)))
                res['own_weight_differs_from_judge_discount'] = (r_c - n15.weight(f(lc), yfa, log_u, 1.0)).abs().max().item() > 0
                unweighted = (F.cross_entropy(f(lt), yfa, reduction='none') * kt.reshape(-1).float()).sum() / yfa.numel()
                res['twin_term_is_unweighted'] = bool(torch.equal(loss_t, 0.5 * unweighted))
        out[arm] = res
    out['pass'] = all(all(v.values()) for v in out.values())
    return out


def loss_form_check(seed, x, ya, yb, mask, log_u):
    """Each new term, with other heads' logits substituted, against the earlier study's loss on the same batch: the term and its
    gradient with respect to the logits, relative difference at most REL_TOL (float32, before the factor of 0.5)."""
    model = n4.new_n4(seed, BASE)
    model.eval()
    with torch.no_grad():
        la, lb = n4.forward_n4(model, x)
    K = ~mask.to(DEV)
    zeros = torch.zeros_like(K)
    rows = {}

    def compare(name, mine_fn, ref_fn, leaves):
        ls_m = [t.detach().float().clone().requires_grad_() for t in leaves]
        ls_r = [t.detach().float().clone().requires_grad_() for t in leaves]
        mine, ref = mine_fn(*ls_m), ref_fn(*ls_r)
        gm = torch.autograd.grad(mine, ls_m[0])[0]
        gr = torch.autograd.grad(ref, ls_r[0])[0]
        rows[name] = {'mine': mine.item(), 'reference': ref.item(), 'relative_loss_difference': abs(mine.item() - ref.item()) / abs(ref.item()),
                      'relative_gradient_difference': rel_difference(gm, gr)}
        rows[name]['pass'] = rows[name]['relative_loss_difference'] <= REL_TOL and rows[name]['relative_gradient_difference'] <= REL_TOL

    sc, st_, sn = SPECS['rw-cloop'], SPECS['rw-ctwin'], SPECS['rw-nfree']
    compare('rw-cloop judge term, head B\'s logits, against N14\'s sub20-rw loss',
            lambda l: head_c_loss(l, head_c_weights(l, None, None, ya, yb, log_u, sc), ya, yb, K, sc),
            lambda l: n14.head_b_loss(l, n14.SPECS['sub20-rw'], ya, K, zeros, log_u), [lb])
    compare('rw-ctwin judge term, head B\'s logits for the judge and head A\'s for the twin, against N16\'s sub20-oracle loss',
            lambda l, t: head_c_loss(l, head_c_weights(l, t, None, ya, yb, log_u, st_), ya, yb, K, st_),
            lambda l, t: n16.head_b_loss(t, l, ya, K, log_u, n16.SPECS['sub20-oracle'], 0), [lb, la])
    compare('rw-ctwin twin term, head B\'s logits, against N12\'s sub20 head B loss',
            lambda l: twin_loss(l, ya, K), lambda l: n13._n12_head_b_loss(l, 'sub', ya, K, log_u), [lb])
    compare('rw-nfree judge term, head B\'s logits, against N12\'s n80 head B loss',
            lambda l: head_c_loss(l, None, ya, yb, None, sn), lambda l: n13._n12_head_b_loss(l, 'single', yb, K, log_u), [lb])
    rows['pass'] = all(v['pass'] for v in rows.values())
    return rows


def unmixing_check(seed, probs):
    """In float64, against references written here from probabilities: a constructed judge with q = 0.2 p_A + 0.8 u exactly gives
    the discount from p_A (relative difference at most UNMIX_TOL); a constructed q below 0.8 u at some targets gives weight 0 there
    and max(1 - 0.8 u / q, 0) elsewhere."""
    gen = torch.Generator().manual_seed(seed)
    n, n_uniform = 128, 16                                                  # targets drawn from u, and a few uniformly (u = 0 among them)
    u = probs.double()
    log_u = u.log()
    y = torch.cat([torch.multinomial(u, n - n_uniform, replacement=True, generator=gen), torch.randint(VOCAB, (n_uniform,), generator=gen)])
    pa = torch.softmax(2 * torch.randn(n, VOCAB, generator=gen, dtype=torch.float64), -1)
    q = 0.2 * pa + 0.8 * u
    w = unmixed_weight(q.log(), y, log_u)
    pay, uy = pa.gather(-1, y.unsqueeze(-1)).squeeze(-1), u[y]
    disc = torch.where(uy == 0, torch.ones_like(pay), 0.2 * pay / (0.2 * pay + 0.8 * uy))
    identity = rel_difference(w, disc)
    q2 = torch.softmax(3 * torch.randn(n, VOCAB, generator=gen, dtype=torch.float64), -1)     # mostly below 0.8 u at targets drawn from u
    half = torch.arange(n) % 2 == 1
    q2[half] = 0.1 * q2[half] + 0.9 * F.one_hot(y[half], VOCAB).double()                       # and well above it at these
    w2 = unmixed_weight(q2.log(), y, log_u)
    q2y = q2.gather(-1, y.unsqueeze(-1)).squeeze(-1)
    below = q2y < 0.8 * uy
    ref2 = torch.where(uy == 0, torch.ones_like(q2y), (1 - 0.8 * uy / q2y).clamp(min=0))
    return {'identity_relative_difference': identity, 'targets_below_floor': int(below.sum()), 'targets': n,
            'zero_below_floor': bool((w2[below] == 0).all()), 'floor_relative_difference_elsewhere': rel_difference(w2[~below], ref2[~below]),
            'pass': identity <= UNMIX_TOL and int(below.sum()) > 0 and bool((w2[below] == 0).all()) and rel_difference(w2[~below], ref2[~below]) <= UNMIX_TOL}


def passenger_check(seed, train, probs, log_u):
    """CHECK_STEPS steps of rw-b1 with a passenger head C (trained as the peer) and a passenger twin (trained at full weight on K),
    neither judging, leave the trunk and heads A and B bit-identical to the same steps without them."""
    m0, _, _ = run_steps(seed, SPECS['rw-b1-plain'], CHECK_STEPS, train, probs, log_u)
    m1, _, _ = run_steps(seed, SPECS['rw-b1-passenger2'], CHECK_STEPS, train, probs, log_u)
    a, b = list(m0.state_dict().values()), list(m1.state_dict().values())
    same = all(torch.equal(x, y) for x, y in zip(a, b))
    return {'steps': CHECK_STEPS, 'heads': 2, 'identical': same, 'max_difference': n17.max_difference(a, b), 'pass': same}


def copied_peer_check(seed, train, probs, log_u):
    """N17's: a peer started as an exact copy of head B, judging head B, gives rw-b1's trunk and heads bit for bit, and stays head B."""
    m0, _, _ = run_steps(seed, SPECS['rw-b1-plain'], CHECK_STEPS, train, probs, log_u)
    m1, hc, _ = run_steps(seed, SPECS['rw-peer'], CHECK_STEPS, train, probs, log_u, copy_b=True)
    a, b = list(m0.state_dict().values()), list(m1.state_dict().values())
    same_main = all(torch.equal(x, y) for x, y in zip(a, b))
    same_c = torch.equal(hc.proj.weight, m1.proj_b.weight) and torch.equal(hc.head.weight, m1.head2.weight)
    return {'steps': CHECK_STEPS, 'main_identical': same_main, 'max_main_difference': n17.max_difference(a, b), 'head_C_equals_head_B': bool(same_c),
            'max_head_C_difference': n17.max_difference([hc.proj.weight, hc.head.weight], [m1.proj_b.weight, m1.head2.weight]),
            'pass': same_main and bool(same_c)}


def overflow_test(seed, x, ya, yb, mask, log_u):
    """In rw-ctwin, with an enabled scaler whatever AMP is: an inf in the twin's gradient skips the twin's step only; one in the
    judge's skips the judge's only; in either case the main group steps and the scale is unchanged. One in the main group's
    gradient skips every step and lowers the scale."""
    spec, out = SPECS['rw-ctwin'], {}
    kc, kt = positions(ya.cpu(), mask, seed, 0, spec)
    kc, kt = kc.to(DEV), kt.to(DEV)
    for where, inject in (('twin', 1), ('judge', 0), ('main', 'main')):
        model, hc, ht = n4.new_n4(seed, BASE), n17.HeadC(seed), n17.HeadC(seed + TWIN_OFFSET)
        opt, opt_c, opt_t = (torch.optim.AdamW(p.parameters(), lr=PEAK_LR) for p in (model, hc, ht))
        scaler = torch.amp.GradScaler(DEV.type, enabled=True)
        before = {k: [p.detach().clone() for p in m.parameters()] for k, m in (('main', model), ('judge', hc), ('twin', ht))}
        model.train()
        *_, loss, loss_c, loss_t = step_forward(model, hc, ht, x, ya, yb, kc, kt, log_u, spec)
        scaler.scale(loss.detach())                                         # initialises the scale, as the first step's scaler.scale does
        scale_before = scaler.get_scale()
        main_skipped, (c_skipped, t_skipped) = optim_step(model, extras_of(hc, ht, opt_c, opt_t, loss_c, loss_t), opt, scaler, loss, inject=inject)
        moved = {k: any(not torch.equal(a, b) for a, b in zip(before[k], m.parameters())) for k, m in (('main', model), ('judge', hc), ('twin', ht))}
        out[where] = {'main_skipped': main_skipped, 'judge_skipped': c_skipped, 'twin_skipped': t_skipped,
                      **{f'{k}_moved': v for k, v in moved.items()}, 'scale_before': scale_before, 'scale_after': scaler.get_scale()}
    o_t, o_c, o_m = out['twin'], out['judge'], out['main']
    out['pass'] = (not o_t['main_skipped'] and not o_t['judge_skipped'] and o_t['twin_skipped'] and o_t['main_moved'] and o_t['judge_moved']
                   and not o_t['twin_moved'] and o_t['scale_after'] == o_t['scale_before']
                   and not o_c['main_skipped'] and o_c['judge_skipped'] and not o_c['twin_skipped'] and o_c['main_moved'] and not o_c['judge_moved']
                   and o_c['twin_moved'] and o_c['scale_after'] == o_c['scale_before']
                   and o_m['main_skipped'] and o_m['judge_skipped'] and o_m['twin_skipped'] and not any(o_m[f'{k}_moved'] for k in ('main', 'judge', 'twin'))
                   and o_m['scale_after'] < o_m['scale_before'])
    return out


def positions_check(seed, ya, mask):
    """At step 0: the positions of rw-cloop's, rw-ctwin's and rw-j20's head C, and the twin's, are exactly K; rw-j025's and
    rw-j0125's are nested inside rw-j20's (N14's nested draw), at their fractions."""
    keep = {a: positions(ya, mask, seed, 0, SPECS[a]) for a in ('rw-cloop', 'rw-ctwin', 'rw-j20', 'rw-j025', 'rw-j0125')}
    K = ~mask
    out = {'judge_is_K': {a: bool(torch.equal(keep[a][0], K)) for a in ('rw-cloop', 'rw-ctwin', 'rw-j20')},
           'twin_is_K': bool(torch.equal(keep['rw-ctwin'][1], K)),
           'nested': bool((keep['rw-j025'][0] & ~keep['rw-j20'][0]).sum() == 0) and bool((keep['rw-j0125'][0] & ~keep['rw-j025'][0]).sum() == 0),
           'fractions': {a: keep[a][0].float().mean().item() for a in RUNGS}}
    out['pass'] = (all(out['judge_is_K'].values()) and out['twin_is_K'] and out['nested']
                   and all(abs(out['fractions'][a] - f) <= 0.02 for a, f in RUNGS.items()))
    return out


# ------------------------------------------------------------------ the judge's measures on the measurement windows

@torch.no_grad()
def judge_measure(model, judge, mx, y_clean, log_u, hc=None, bw='discount', batch=n12.MEASURE_BATCH):
    """N17's measures, with the judge's weight by its formula, plus the mixture axis: x_m = the judge's mixture-form loss on the
    clean targets minus head A's copy's (0.2 p_A + 0.8 u). For rw-nfree's judge (bw 'unmix'): the fraction of clean targets its
    weight floors at 0, and the mean mass of max(q, 0.8 u) before normalisation."""
    model.eval()
    if hc is not None:
        hc.eval()
    tot = dict.fromkeys(('judge', 'A', 'B', 'B_on_judge_ignored', 'B_on_own_ignored', 'judge_mix', 'A_mix', 'mass'), 0.0)
    n_all = n_j = n_own = n_both = n_floor = n_rows = 0
    for i in range(0, len(mx), batch):
        x, y = mx[i:i + batch].to(DEV), y_clean[i:i + batch].to(DEV)
        la, lb, z = n17.forward_z(model, x)
        lj = {'C': lambda: hc(z), 'B': lambda: lb, 'A': lambda: la}[judge]()
        yf = y.reshape(-1)
        laf, lbf, ljf = (t.reshape(-1, VOCAB).float() for t in (la, lb, lj))
        ce_j, ce_a, ce_b = (F.cross_entropy(t, yf, reduction='none') for t in (ljf, laf, lbf))
        wj = weight_of(ljf, yf, log_u, bw)
        ign_j = wj < IGNORED
        ign_own = n15.weight(lbf, yf, log_u, 1.0) < IGNORED
        mj = mixture_logp(ljf, log_u, bw).gather(-1, yf.unsqueeze(-1)).squeeze(-1)
        ma = n9.copy_logits(laf, log_u, RHO).gather(-1, yf.unsqueeze(-1)).squeeze(-1)
        tot['judge'] += ce_j.sum().item(); tot['A'] += ce_a.sum().item(); tot['B'] += ce_b.sum().item()
        tot['B_on_judge_ignored'] += ce_b[ign_j].sum().item(); tot['B_on_own_ignored'] += ce_b[ign_own].sum().item()
        tot['judge_mix'] -= mj.sum().item(); tot['A_mix'] -= ma.sum().item()
        if bw == 'unmix':
            n_floor += int((wj == 0).sum())
            tot['mass'] += torch.maximum(torch.softmax(ljf, -1), RHO * log_u.exp()).sum(-1).sum().item()
            n_rows += ljf.shape[0]
        n_all += yf.numel(); n_j += int(ign_j.sum()); n_own += int(ign_own.sum()); n_both += int((ign_j & ign_own).sum())
    return {'judge': judge, 'weight': bw, 'threshold': IGNORED, 'n_targets': n_all,
            'judge_clean': tot['judge'] / n_all, 'A_clean': tot['A'] / n_all, 'B_clean': tot['B'] / n_all,
            'x': (tot['judge'] - tot['A']) / n_all,
            'judge_mix_clean': tot['judge_mix'] / n_all, 'A_mix_clean': tot['A_mix'] / n_all, 'x_m': (tot['judge_mix'] - tot['A_mix']) / n_all,
            'judge_ignored_fraction': n_j / n_all, 'own_ignored_fraction': n_own / n_all,
            'share_of_own_ignored_also_judge_ignored': n_both / n_own if n_own else None,
            'share_of_judge_ignored_also_own_ignored': n_both / n_j if n_j else None,
            'loss_B_on_judge_ignored': tot['B_on_judge_ignored'] / n_j if n_j else None,
            'loss_B_on_own_ignored': tot['B_on_own_ignored'] / n_own if n_own else None,
            'floored_fraction': n_floor / n_all if bw == 'unmix' else None,
            'mean_mass_before_normalisation': tot['mass'] / n_rows if bw == 'unmix' else None}


# ------------------------------------------------------------------ training

def train_n18(store, meta, arm, seed, tag, train, probs, log_u):
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
    model, hc, ht, opt, opt_c, opt_t, scaler = fresh(seed, spec)
    gen = torch.Generator().manual_seed(seed)
    state = {'step': 0, 'train_loss': [], 'val': [], 'alpha': [], 'w': [], 'own_w': [], 'judge': [], 'skipped_steps': 0, 'c_skipped_steps': 0,
             't_skipped_steps': 0, 'seconds': 0.0, 'resumes': 0, 'max_offset': -1}
    if ck_path.exists():
        ck = torch.load(ck_path, map_location='cpu', weights_only=False)
        model.load_state_dict(ck['model']); opt.load_state_dict(ck['opt']); scaler.load_state_dict(ck['scaler'])
        if hc is not None:
            hc.load_state_dict(ck['hc']); opt_c.load_state_dict(ck['opt_c'])
        if ht is not None:
            ht.load_state_dict(ck['ht']); opt_t.load_state_dict(ck['opt_t'])
        gen.set_state(ck['gen'])
        torch.set_rng_state(ck['torch_rng'])
        if DEV.type == 'cuda' and ck.get('cuda_rng') is not None:
            torch.cuda.set_rng_state_all(ck['cuda_rng'])
        state = ck['state']; state['resumes'] += 1
        log(f"{tag}: resuming at step {state['step']}")
    else:
        what = {'self': 'head C judges itself (the peer)', 'noisy': "head C at full weight on head B's targets, head B's weight unmixed"}.get(spec['c'])
        if spec['c'] == 'clean':
            what = f"head C on {spec['frac']:.2%} clean targets, " + {'none': 'at full weight', 'self': 'each weighted by its own discount',
                                                                       'twin': "each weighted by the twin's discount; the twin at full weight on K"}[spec['cw']]
        log(f"{tag}: starting, noise {RHO}, head B weighted by head {spec['judge']}'s prediction, {what or 'no head C'}, seed {seed}")

    def save_ckpt():
        tmp = ck_path.with_suffix('.tmp')
        torch.save({'model': model.state_dict(), 'opt': opt.state_dict(), 'scaler': scaler.state_dict(),
                    'hc': hc.state_dict() if hc is not None else None, 'opt_c': opt_c.state_dict() if opt_c is not None else None,
                    'ht': ht.state_dict() if ht is not None else None, 'opt_t': opt_t.state_dict() if opt_t is not None else None,
                    'gen': gen.get_state(), 'state': state, 'torch_rng': torch.get_rng_state(),
                    'cuda_rng': torch.cuda.get_rng_state_all() if DEV.type == 'cuda' else None}, tmp)
        os.replace(tmp, ck_path)
        store.push(ck_rel)

    model.train()
    for h in (hc, ht):
        if h is not None:
            h.train()
    last_ck, t_seg = time.time(), time.time()
    while state['step'] < TOTAL:
        step = state['step']
        set_lr((opt, opt_c, opt_t), step)
        x, ys, top = n12.draw_batch(train, gen)
        state['max_offset'] = max(state['max_offset'], top)
        ya = ys[0]
        yb, keep, mask = n12.step_targets(ya, probs, seed, step, RHO, 'rw')
        keep_c, keep_t = positions(ya, mask, seed, step, spec)
        if step == 0:
            state.update({'batch_fingerprint_step0': n12.fingerprint(x), 'replaced_positions_fingerprint_step0': n12.fingerprint(mask),
                          'noise_fraction_step0': mask.float().mean().item(), 'kept_fraction_step0': keep.float().mean().item(),
                          'judge_keep_fingerprint_step0': n12.fingerprint(keep_c) if keep_c is not None else None,
                          'judge_kept_fraction_step0': keep_c.float().mean().item() if keep_c is not None else None,
                          'judge_keep_inside_K_step0': bool((keep_c & mask).sum() == 0) if keep_c is not None else None,
                          'judge_keep_is_K_step0': bool(torch.equal(keep_c, ~mask)) if keep_c is not None else None,
                          'twin_keep_is_K_step0': bool(torch.equal(keep_t, ~mask)) if keep_t is not None else None})
        x, ya, yb, keep = x.to(DEV), ya.to(DEV), yb.to(DEV), keep.to(DEV)
        keep_c = keep_c.to(DEV) if keep_c is not None else None
        keep_t = keep_t.to(DEV) if keep_t is not None else None
        for o in (opt, opt_c, opt_t):
            if o is not None:
                o.zero_grad(set_to_none=True)
        la, lb, lc, lt, w, r_c, loss, loss_c, loss_t = step_forward(model, hc, ht, x, ya, yb, keep_c, keep_t, log_u, spec)
        if step % 50 == 0:                                                      # the weights and the judge's accuracy, no gradient
            with torch.no_grad():
                k = keep.reshape(-1)
                sk, sr = w[k].sum().item(), w[~k].sum().item()
                state['w'].append([step, w.mean().item(), w[k].mean().item(), w[~k].mean().item(), sr / (sr + sk) if sr + sk > 0 else None, (w[k] < IGNORED).float().mean().item()])
                if spec['c'] == 'clean' and r_c is not None:
                    kc_ = keep_c.reshape(-1)
                    state['own_w'].append([step, r_c[kc_].mean().item(), (r_c[kc_] < IGNORED).float().mean().item()])
                lj = lc if spec['judge'] == 'C' else lb
                yf = ya.reshape(-1)
                ljf, laf = lj.reshape(-1, VOCAB).float(), la.reshape(-1, VOCAB).float()
                pick = lambda t: t.gather(-1, yf.unsqueeze(-1)).squeeze(-1)
                state['judge'].append([step, F.cross_entropy(ljf, yf).item(), F.cross_entropy(laf, yf).item(),
                                       loss_c.item() if loss_c is not None else None,
                                       -pick(mixture_logp(ljf, log_u, spec['bw'])).mean().item(), -pick(n9.copy_logits(laf, log_u, RHO)).mean().item(),
                                       F.cross_entropy(lt.reshape(-1, VOCAB).float(), yf).item() if lt is not None else None,
                                       loss_t.item() if loss_t is not None else None])
        main_skipped, skipped = optim_step(model, extras_of(hc, ht, opt_c, opt_t, loss_c, loss_t), opt, scaler, loss)
        state['skipped_steps'] += int(main_skipped)
        if hc is not None:
            state['c_skipped_steps'] += int(skipped[0] and not main_skipped)
        if ht is not None:
            state['t_skipped_steps'] += int(skipped[-1] and not main_skipped)
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
    for h, suffix in ((hc, 'headC'), (ht, 'headT')):
        if h is not None:
            torch.save(h.state_dict(), store.L(f'final/{tag}_{suffix}.pt'))
            store.push(f'final/{tag}_{suffix}.pt')
    fx, fys = n6.windows(study.load_tokens(store, meta, 'validation'), limit=g.FINAL_LIMIT)
    l_a, l_b, l_b_clean, l_inner = n12.val_losses(model, fx, fys[0], n10.fixed_noisy(fys[0], probs, RHO, n12.VAL_NOISE_SEED), 'rw', log_u)
    last = state['w'][-1]
    finite = (all(math.isfinite(v) for e in state['train_loss'] + state['val'] for v in e[1:])
              and all(v is None or math.isfinite(v) for e in state['judge'] for v in e[1:])
              and all(math.isfinite(v) for v in (l_a, l_b, l_b_clean, l_inner)))
    result = {'tag': tag, 'arm': arm, 'seed': seed, 'rho': RHO, **{f'spec_{k}': v for k, v in spec.items()}, 'head_b': 'rw', 'readout_rank': n4.RANK,
              'final_loss_A': l_a, 'final_loss_B': l_b, 'final_loss_B_clean_targets': l_b_clean, 'final_loss_B_inner_clean': l_inner,
              'final_val_windows': len(fx), 'val_curve': state['val'], 'train_loss_every_50': state['train_loss'],
              'w_every_50': state['w'],              # [step, mean w, mean w on kept, mean w on replaced, random share, fraction of kept with w < 0.01]
              'own_w_every_50': state['own_w'],      # [step, mean weight on head C's own clean targets, fraction of them below 0.01] (rw-cloop, rw-ctwin)
              'judge_every_50': state['judge'],      # [step, judge CE on the batch's clean targets, head A's, head C's own loss,
                                                     #  judge mixture-form loss on them, head A's copy's, the twin's CE, the twin's own loss]
              'logged_losses_finite': finite,
              'alpha_every_50': state['alpha'], 'alpha_final': model.ode_block.func.output_scale.item(),
              **{k: state.get(k) for k in ('batch_fingerprint_step0', 'replaced_positions_fingerprint_step0', 'noise_fraction_step0', 'kept_fraction_step0',
                                            'judge_keep_fingerprint_step0', 'judge_kept_fraction_step0', 'judge_keep_inside_K_step0',
                                            'judge_keep_is_K_step0', 'twin_keep_is_K_step0')},
              'max_offset': state['max_offset'], 'training_tokens': len(train),
              'skipped_steps_fp16': state['skipped_steps'], 'head_C_only_skipped_steps': state['c_skipped_steps'],
              'twin_only_skipped_steps': state['t_skipped_steps'],
              'train_seconds': state['seconds'], 'resumes': state['resumes'],
              'n_params': sum(p.numel() for p in model.parameters()),
              'n_params_head_C': sum(p.numel() for p in hc.parameters()) if hc is not None else 0,
              'n_params_twin': sum(p.numel() for p in ht.parameters()) if ht is not None else 0,
              'environment': study.env_record()}
    res_path.write_text(json.dumps(result, indent=2))
    store.push(f'results/{tag}.json')
    log(f'{tag}: finished in {state["seconds"] / 3600:.2f} h; head A {l_a:.4f}, head B {l_b:.4f} (clean targets {l_b_clean:.4f}, inner {l_inner:.4f}); '
        f"final share {'n/a' if last[4] is None else f'{last[4]:.3f}'}, ignored kept {last[5]:.3f}")
    return result


def load_head(store, tag, seed, suffix):
    if not store.pull(f'final/{tag}_{suffix}.pt'):
        raise SystemExit(f'weights {store.drive / "final" / (tag + "_" + suffix + ".pt")} are missing locally and on Drive.')
    h = n17.HeadC(seed)
    h.load_state_dict(torch.load(store.L(f'final/{tag}_{suffix}.pt'), map_location=DEV, weights_only=False))
    return h


def code_identity(store, n17_store, n17_records, arm, seed):
    """N18's model against N17's saved model of the same arm and seed: bit-identical weights are required where the GPU and the
    PyTorch version match N17's training record; otherwise the differences are reported."""
    tag = f'{arm}_seed{seed}'
    mine = {**{f'main.{k}': v for k, v in torch.load(store.L(f'final/{tag}.pt'), map_location='cpu', weights_only=False).items()},
            **{f'head_C.{k}': v for k, v in torch.load(store.L(f'final/{tag}_headC.pt'), map_location='cpu', weights_only=False).items()}}
    for rel in (f'final/{tag}.pt', f'final/{tag}_headC.pt'):
        if not n17_store.pull(rel):
            return {'tag': tag, 'available': False, 'missing': str(n17_store.drive / rel)}
    theirs = {**{f'main.{k}': v for k, v in torch.load(n17_store.L(f'final/{tag}.pt'), map_location='cpu', weights_only=False).items()},
              **{f'head_C.{k}': v for k, v in torch.load(n17_store.L(f'final/{tag}_headC.pt'), map_location='cpu', weights_only=False).items()}}
    rec_path = n17_records / f'{tag}.json'
    if not rec_path.exists():
        return {'tag': tag, 'available': False, 'missing': str(rec_path)}
    env_n17 = json.loads(rec_path.read_text())['environment']
    env = study.env_record()
    match = all(env.get(k) == env_n17.get(k) for k in ('torch', 'gpu', 'device', 'amp_fp16'))
    same_keys = set(mine) == set(theirs)
    identical = same_keys and all(torch.equal(mine[k], theirs[k]) for k in mine)
    return {'tag': tag, 'available': True, 'environment_this': env, 'environment_n17': env_n17, 'environment_matches': match,
            'same_parameters': same_keys, 'identical': identical,
            'max_difference': max(((mine[k].float() - theirs[k].float()).abs().max().item() for k in mine if k in theirs), default=None),
            'pass': identical if match else True}


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--source', default=None)
    ap.add_argument('--local', default=None)
    FOLDERS = (('n12', 'n12_bottleneck', 'n12_calib'), ('n13', 'n13_signal', 'n13_calib'), ('n14', 'n14_dose', 'n14_calib'),
               ('n15', 'n15_sharpen', 'n15_calib'), ('n16', 'n16_judge', 'n16_calib'), ('n17', 'n17_loop', 'n17_calib'))
    for k, real, cal in FOLDERS:
        ap.add_argument(f'--{k}', default=None, help=f"{k.upper()}'s Drive folder (default: {real}, or {cal}, next to --work)")
        ap.add_argument(f'--{k}-records', default=None, help=f"{k.upper()}'s committed records (default: results/{k} or results/{k}_calib here)")
    ap.add_argument('--calibration', action='store_true', help="the calibration run: seed 3001, the tail's windows 513-1024")
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
    srcs = {k: study.Store(folders[k], local.parent / f'_n18_src_{folders[k].name}') for k in {v[0] for v in refs.values()} | {'n17'}}
    new_arms = list(CALIB_ARMS if args.calibration else REAL_ARMS)
    identity_arms = IDENTITY[mode]
    out = Path(args.out)
    (out / 'results').mkdir(parents=True, exist_ok=True)
    log('device:', DEV, '| test mode:', TEST, '| mode:', mode, '| study folder (data):', source, '| N18 folder:', work, '| local:', local)
    log('arms trained:', new_arms, '| references:', list(refs), '| saved models from:', {k: str(folders[k]) for k in srcs})
    store.pull_dir('results')
    try:
        meta = study.prepare_data(data_store, None)
        full = study.load_tokens(store, meta, 'train')
        cut = int(len(full) * n12.CUT_FRACTION)
        train, tail = full[:cut], full[cut:]
        probs = n12.n7.unigram(train)
        log_u = probs.float().log().to(DEV)
        summary_path = store.L('results/n18_summary.json')
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        if summary.get('arms') is not None and list(summary['arms']) != new_arms:
            raise SystemExit(f"This folder's earlier session trained {list(summary['arms'])}; this session would train {new_arms}. Stopping.")

        def save_json(rel, obj):
            store.L(rel).write_text(json.dumps(obj, indent=2 if rel.endswith('n18_summary.json') else None))
            store.push(rel)

        allx, allys = n6.windows(tail, limit=w1)
        mx, mys = allx[w0:w1], [y[w0:w1] for y in allys]
        u = n8.uniforms(len(mx))
        log(f"training region: tokens 0-{cut} of {len(full)}; measured on the held-out tail's windows {w0 + 1}-{w0 + len(mx)}")

        if 'checks' not in summary:
            s0 = seeds[0]
            x, ys, _ = n12.draw_batch(train, torch.Generator().manual_seed(s0))
            yb, keep, mask = n12.step_targets(ys[0], probs, s0, 0, RHO, 'rw')
            xd, yad, ybd = x.to(DEV), ys[0].to(DEV), yb.to(DEV)
            m0 = n4.new_n4(s0, BASE)
            log(f'setup checks: {CHECK_STEPS}-step passenger and copied-peer runs, the overflow test, the loss forms')
            summary['checks'] = {
                'batch': n6.batch_check(train), 'batch_g1': g.batch_check(train),
                'same_model': [g.same_model_check(train, s) for s in seeds],
                'windows_shifted': all(torch.equal(mys[k - 1][:, :-k], mx[:, k:]) for k in range(1, 9)),
                'analytic': n9.analytic_check(),
                'identical_heads': n8.identical_heads_check(s0, mx, u),
                'weight_identity': n15.weight_identity_check(m0, xd, ybd, log_u),
                'forward_identity': n17.forward_check(m0, xd),
                'detachment': detachment_check(s0, xd, yad, ybd, mask, log_u),
                'weight_sources': weight_source_check(s0, xd, yad, ybd, mask, log_u),
                'loss_forms': loss_form_check(s0, xd, yad, ybd, mask, log_u),
                'unmixing': unmixing_check(s0, probs),
                'passenger': passenger_check(s0, train, probs, log_u),
                'copied_peer': copied_peer_check(s0, train, probs, log_u),
                'overflow': overflow_test(s0, xd, yad, ybd, mask, log_u),
                'positions': positions_check(s0, ys[0], mask),
                'unigram': {'sum': probs.sum().item(), 'zeros': int((probs == 0).sum()), 'from_tokens': cut}}
            summary['mode'] = mode
            summary['held_out'] = {'cut_fraction': n12.CUT_FRACTION, 'cut_token': cut, 'training_tokens': len(full), 'tail_tokens': len(tail),
                                   'tail_windows': (len(tail) - n12.K_MAX) // n12.SEQ, 'measure': [w0 + 1, w0 + len(mx)]}
            summary['arms'] = {a: SPECS[a] for a in new_arms}
            summary['references'] = {a: {'folder': str(folders[v[0]]), 'tag': v[1], 'records': str(records[v[0]])} for a, v in refs.items()}
            summary['seeds'], summary['readout_rank'], summary['subspace_rank'] = seeds, n4.RANK, n9.R
            save_json('results/n18_summary.json', summary)
        c = summary['checks']
        names = ('weight_identity', 'forward_identity', 'detachment', 'weight_sources', 'loss_forms', 'unmixing', 'passenger', 'copied_peer',
                 'overflow', 'positions')
        log('batch check:', c['batch']['targets_are_i+1_to_i+8'], '| windows shifted:', c['windows_shifted'], '| analytic:', c['analytic']['pass'],
            '| identical heads:', c['identical_heads']['pass'], '|', ' | '.join(f"{k}: {c[k]['pass']}" for k in names))
        log('same-model check:', [x['pass'] for x in c['same_model']], '| passenger largest difference:', c['passenger']['max_difference'],
            '| copied peer largest differences:', c['copied_peer']['max_main_difference'], c['copied_peer']['max_head_C_difference'],
            '| loss forms:', {k: (v['relative_loss_difference'], v['relative_gradient_difference']) for k, v in c['loss_forms'].items() if k != 'pass'})
        exploratory_only = {'unmixing'}                       # a check that concerns only rw-nfree does not stop the run (preregistration)
        nfree_parts = all(c['detachment'].get('rw-nfree', {}).values()) and all(c['weight_sources'].get('rw-nfree', {}).values())
        core = [k for k in names if k not in exploratory_only and k not in ('detachment', 'weight_sources', 'loss_forms')]
        core_ok = (all(c[k]['pass'] for k in core)
                   and all(all(v.values()) for a, v in c['detachment'].items() if a not in ('pass', 'rw-nfree'))
                   and all(all(v.values()) for a, v in c['weight_sources'].items() if a not in ('pass', 'rw-nfree'))
                   and all(v['pass'] for k, v in c['loss_forms'].items() if k != 'pass' and 'rw-nfree' not in k))
        if not (c['batch']['targets_are_i+1_to_i+8'] and c['windows_shifted'] and c['analytic']['pass'] and c['identical_heads']['pass']
                and core_ok and all(x['pass'] for x in c['same_model']) and abs(c['unigram']['sum'] - 1) < 1e-6):
            raise SystemExit('A check of the setup failed before training. Stopping.')
        nfree_ok = nfree_parts and c['unmixing']['pass'] and all(v['pass'] for k, v in c['loss_forms'].items() if 'rw-nfree' in k)
        if not nfree_ok:
            log("a check concerning only rw-nfree failed: its results will be unreportable; N18a is unaffected")

        def measure_all(m, arm, tag, judge, hc=None, bw='discount', ht=None):
            res = {'tag': tag, 'seed': seed, 'arm': arm, 'model': tag, **n12.measure(m, arm, mx, mys, u, log_u, probs),
                   'neglect': n16.neglect(m, mx, mys, log_u),
                   'judge_measure': judge_measure(m, judge, mx, mys[0], log_u, hc, bw) if judge else None, 'environment': study.env_record()}
            if ht is not None:
                res['twin_measure'] = judge_measure(m, 'C', mx, mys[0], log_u, ht)
            return res

        for seed in seeds:
            for arm, (fkey, prefix, judge) in refs.items():                  # the saved models, re-measured (precondition 2)
                rel = f'results/r_{arm}_seed{seed}.json'
                if store.L(rel).exists():
                    continue
                tag = f'{prefix}_seed{seed}'
                m = n13.load_model(srcs[fkey], tag, seed)
                hc = load_head(srcs[fkey], tag, seed, 'headC') if judge == 'C' else None
                res = measure_all(m, arm, tag, judge, hc)
                res['tag'] = f'{arm}_seed{seed}'
                ref_path = records[fkey] / f'm_{tag}.json'
                if ref_path.exists():
                    ref = json.loads(ref_path.read_text())
                    rep = n13.reproduction(res, ref)
                    if judge == 'C':
                        rep['judge_x_difference'] = abs(res['judge_measure']['x'] - ref['judge_measure']['x'])
                        rep['pass'] = rep['pass'] and rep['judge_x_difference'] <= 1e-3
                    res['reproduction'] = rep
                else:
                    res['reproduction'] = {'pass': False, 'missing': str(ref_path)}
                save_json(rel, res)
                fn, rep = res['functional'], res['reproduction']
                a = (res.get('B_inner_vs_A') or res['B_vs_A'])['agreement_ratio']
                jm = res['judge_measure']
                log(f"r_{arm}_seed{seed}: inner clean {fn['B_inner_clean']:.3f} | head A clean {fn['A_clean']:.3f} | agreement with A {a:.3f} | "
                    + (f"judge {jm['judge']} x {jm['x']:+.3f}, x_m {jm['x_m']:+.3f} | " if jm else '')
                    + f"reproduces: {rep.get('pass')} (loss {rep.get('max_loss_difference', float('nan')):.1e}, ratio {rep.get('max_ratio_difference', float('nan')):.1e}) | {res['seconds']:.0f} s")
            for arm in new_arms:
                spec = SPECS[arm]
                tag = f'{arm}_seed{seed}'
                rel = f'results/m_{tag}.json'
                if arm == 'rw-nfree':                                     # exploratory: a failure is recorded, and the claim's runs go on
                    try:
                        train_n18(store, meta, arm, seed, tag, train, probs, log_u)
                        if not store.L(rel).exists():
                            m = n13.load_model(store, tag, seed)
                            res = measure_all(m, arm, tag, 'C', load_head(store, tag, seed, 'headC'), spec['bw'])
                            save_json(rel, res)
                            log(f"m_{tag}: inner clean {res['functional']['B_inner_clean']:.3f} | judge x_m {res['judge_measure']['x_m']:+.3f}, "
                                f"floored {res['judge_measure']['floored_fraction']:.3f} | {res['seconds']:.0f} s")
                    except Exception as e:                                # noqa: BLE001
                        save_json(f'results/x_{tag}.json', {'tag': tag, 'failed': f'{type(e).__name__}: {e}'[:2000]})
                        log(f'{tag}: FAILED ({type(e).__name__}: {e}); recorded in results/x_{tag}.json; the other arms go on')
                    continue
                train_n18(store, meta, arm, seed, tag, train, probs, log_u)
                if not store.L(rel).exists():
                    m = n13.load_model(store, tag, seed)
                    hc = load_head(store, tag, seed, 'headC')
                    ht = load_head(store, tag, seed, 'headT') if spec['twin'] else None
                    res = measure_all(m, arm, tag, spec['judge'], hc, spec['bw'], ht)
                    save_json(rel, res)
                    fn, jm = res['functional'], res['judge_measure']
                    a = (res.get('B_inner_vs_A') or res['B_vs_A'])['agreement_ratio']
                    log(f"m_{tag}: inner clean {fn['B_inner_clean']:.3f} | head A clean {fn['A_clean']:.3f} | agreement with A {a:.3f} | "
                        f"judge {jm['judge']} x {jm['x']:+.3f}, x_m {jm['x_m']:+.3f}, ignores {jm['judge_ignored_fraction']:.3f} (own {jm['own_ignored_fraction']:.3f}) | "
                        + (f"twin x {res['twin_measure']['x']:+.3f} | " if ht is not None else '')
                        + f"rotated {res['rotated_control']['agreement_ratio']:.3f} | {res['seconds']:.0f} s")
                crel = f'results/c_{tag}.json'
                if arm in identity_arms and not store.L(crel).exists():       # code identity against N17's saved model (precondition 1)
                    store.pull(f'final/{tag}.pt'); store.pull(f'final/{tag}_headC.pt')
                    ci = code_identity(store, srcs['n17'], records['n17'], arm, seed)
                    save_json(crel, ci)
                    log(f"c_{tag}: against N17's saved model: environment matches {ci.get('environment_matches')}, bit-identical {ci.get('identical')}, "
                        f"largest difference {ci.get('max_difference')}, passes {ci.get('pass')}")
        log(f'N18 {mode} complete: all runs and measurements finished')
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
