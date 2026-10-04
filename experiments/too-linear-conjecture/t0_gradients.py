"""T0: are Model A's training gradients the gradients of its own forward pass?

Implements T0_PREREGISTRATION.md in this folder. Reads the WikiText-103 study's final
Model A weights and token files from its Drive folder (read-only), computes the
training gradient (odeint_adjoint) and the exact gradient (ordinary backpropagation
through the same four Euler steps, same dropout masks), and writes every measured
number to t0_results.json. t0_analyze.py applies the decision rule.

    python t0_gradients.py --out OUT --work /content/drive/MyDrive/cdt-colab/t0_gradients
    STUDY_TEST=1 python t0_gradients.py --out OUT --work W --source LOCAL_STUDY_FOLDER   # small test

The study folder defaults to the sibling folder wt103_study of --work.
"""
import argparse
import inspect
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'wikitext103-training'))

import study  # noqa: E402  (sets the model config, test mode and device)
import torchdiffeq  # noqa: E402
from torchdiffeq import odeint  # noqa: E402
from torchdiffeq._impl import adjoint as tdq_adjoint  # noqa: E402

DEV, TEST, log = study.DEV, study.TEST, study.log
SEEDS = study.MAIN_SEEDS
N_BATCHES, BATCH, SEQ = (2, 4, study.SEQ) if TEST else (8, 32, study.SEQ)
K_START, K_MAX = (4, 8) if TEST else (16, 64)
BATCH_SEED = 12345
REF_MIN = 0.95
S3_MULTIPLIERS = [0.001, 0.1, 0.5, 1.0, 2.0]
MM4_ALPHA = 0.001


# ------------------------------------------------------------------ gradients

def set_dropout(model, on):
    for m in model.modules():
        if isinstance(m, nn.Dropout):
            m.p = study.CFG['dropout'] if on else 0.0


def groups(model):
    func = model.ode_block.func
    g = {'ode': list(func.parameters()),
         'field': [p for n, p in func.named_parameters() if n != 'output_scale'],
         'alpha': [func.output_scale],
         'upstream': list(model.tok_emb.parameters()) + list(model.pos_emb.parameters())
         + [p for b in model.early_blocks for p in b.parameters()],
         'downstream': [p for b in model.late_blocks for p in b.parameters()]
         + list(model.ln_f.parameters()) + list(model.head.parameters())}
    assert sum(p.numel() for k in ('ode', 'upstream', 'downstream') for p in g[k]) == sum(p.numel() for p in model.parameters())
    return g


def forward(model, x, mode):
    """'adjoint' is the model's own forward pass, as trained. 'exact' and 'loop' run the same
    computation in the same order, differentiated by ordinary backpropagation."""
    if mode == 'adjoint':
        return model(x)
    T = x.shape[1]
    h = model.tok_emb(x) + model.pos_emb(torch.arange(T, device=x.device))
    for b in model.early_blocks:
        h = b(h)
    func, times = model.ode_block.func, model.ode_block.integration_times
    func.control = None
    if mode == 'exact':
        h = odeint(func, h, times, method='euler')[-1]
    else:
        for i in range(len(times) - 1):
            h = h + (times[i + 1] - times[i]) * func(times[i], h)
    for b in model.late_blocks:
        h = b(h)
    return model.head(model.ln_f(h))


def grad(model, x, y, mode, seed, dropout=True, alpha=None, amp=False):
    """Flattened gradient per parameter group, plus the loss."""
    set_dropout(model, dropout)
    model.train()
    func = model.ode_block.func
    saved = func.output_scale.data.clone()
    if alpha is not None:
        func.output_scale.data.fill_(alpha)
    try:
        model.zero_grad(set_to_none=True)
        torch.manual_seed(seed)
        scale = 2.0 ** 16 if amp else 1.0
        with torch.autocast(DEV.type, dtype=torch.float16, enabled=amp):
            logits = forward(model, x, mode)
        loss = F.cross_entropy(logits.float().reshape(-1, logits.size(-1)), y.reshape(-1))
        (loss * scale).backward()
        out = {k: torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in ps]) / scale
               for k, ps in groups(model).items()}
        return out, loss.item()
    finally:
        func.output_scale.data.copy_(saved)
        set_dropout(model, True)


def cos(a, b):
    return F.cosine_similarity(a.double(), b.double(), dim=0).item()


# ------------------------------------------------------------------ measurements

def primary(model, batches, k):
    """Per batch: cosines of the K-draw means, single-draw cosines, and norms. Seeds 0..K-1 are
    shared by the training and exact gradients; seeds K..2K-1 give the noise reference."""
    rows = []
    for bi, (x, y) in enumerate(batches):
        sums = {n: None for n in ('a', 'c', 'c2')}
        single_ac, single_cc, single_ac_f, single_cc_f = [], [], [], []
        for s in range(k):
            ga, _ = grad(model, x, y, 'adjoint', s)
            gc, _ = grad(model, x, y, 'exact', s)
            gc2, _ = grad(model, x, y, 'exact', k + s)
            single_ac.append(cos(ga['ode'], gc['ode']))
            single_cc.append(cos(gc['ode'], gc2['ode']))
            single_ac_f.append(cos(ga['field'], gc['field']))
            single_cc_f.append(cos(gc['field'], gc2['field']))
            for n, g in (('a', ga), ('c', gc), ('c2', gc2)):
                sums[n] = {key: v.clone() for key, v in g.items()} if sums[n] is None else {key: sums[n][key] + g[key] for key in g}
        mean = {n: {key: v / k for key, v in sums[n].items()} for n in sums}
        rows.append({
            'batch': bi,
            'cos_mean_a_c': {key: cos(mean['a'][key], mean['c'][key]) for key in ('ode', 'field', 'upstream', 'downstream')},
            'cos_mean_c_c2': {key: cos(mean['c'][key], mean['c2'][key]) for key in ('ode', 'field', 'upstream', 'downstream')},
            'norm_ratio_ode': (mean['a']['ode'].norm() / mean['c']['ode'].norm()).item(),
            'norm_ratio_field': (mean['a']['field'].norm() / mean['c']['field'].norm()).item(),
            'alpha_grad_mean_a': mean['a']['alpha'].item(), 'alpha_grad_mean_c': mean['c']['alpha'].item(),
            'alpha_grad_mean_c2': mean['c2']['alpha'].item(),
            'single_cos_a_c_ode': float(np.mean(single_ac)),
            'single_cos_c_c2_ode': float(np.mean(single_cc)),
            'single_cos_a_c_field': float(np.mean(single_ac_f)),
            'single_cos_c_c2_field': float(np.mean(single_cc_f)),
        })
    avg = lambda f: float(np.mean([f(r) for r in rows]))  # noqa: E731
    return {'k': k, 'batches': rows,
            'primary_cos': avg(lambda r: r['cos_mean_a_c']['ode']),
            'reference_cos': avg(lambda r: r['cos_mean_c_c2']['ode']),
            'field_cos': avg(lambda r: r['cos_mean_a_c']['field']),
            'field_reference_cos': avg(lambda r: r['cos_mean_c_c2']['field']),
            'norm_ratio_field': avg(lambda r: r['norm_ratio_field']),
            'single_cos_a_c_field': avg(lambda r: r['single_cos_a_c_field']),
            'single_cos_c_c2_field': avg(lambda r: r['single_cos_c_c2_field']),
            'upstream_cos': avg(lambda r: r['cos_mean_a_c']['upstream']),
            'upstream_reference_cos': avg(lambda r: r['cos_mean_c_c2']['upstream']),
            'norm_ratio_ode': avg(lambda r: r['norm_ratio_ode']),
            'single_cos_a_c': avg(lambda r: r['single_cos_a_c_ode']),
            'single_cos_c_c2': avg(lambda r: r['single_cos_c_c2_ode'])}


def dropout_off_cos(model, batches, alpha=None, multiplier=None, no_time=False):
    """Adjoint against exact gradient with dropout off. no_time zeroes the time embedding's output,
    making the vector field time-independent, so that every error the adjoint makes vanishes as
    alpha goes to 0."""
    learned = model.ode_block.func.output_scale.item()
    a = alpha if alpha is not None else learned * multiplier
    last = model.ode_block.func.time_embed[2]
    saved = (last.weight.data.clone(), last.bias.data.clone())
    if no_time:
        last.weight.data.zero_()
        last.bias.data.zero_()
    vals = []
    try:
        for x, y in batches:
            gb, _ = grad(model, x, y, 'adjoint', 0, dropout=False, alpha=a)
            ge, _ = grad(model, x, y, 'exact', 0, dropout=False, alpha=a)
            vals.append((cos(gb['ode'], ge['ode']), cos(gb['field'], ge['field'])))
    finally:
        last.weight.data.copy_(saved[0])
        last.bias.data.copy_(saved[1])
    return {'alpha': a, 'cos_ode': float(np.mean([v[0] for v in vals])), 'cos_field': float(np.mean([v[1] for v in vals])),
            'per_batch': vals}


def amp_cos(model, batches, k):
    vals = []
    for x, y in batches:
        sa = sc = None
        for s in range(k):
            ga, _ = grad(model, x, y, 'adjoint', s, amp=True)
            gc, _ = grad(model, x, y, 'exact', s)
            sa = {k: ga[k] for k in ('ode', 'field')} if sa is None else {k: sa[k] + ga[k] for k in sa}
            sc = {k: gc[k] for k in ('ode', 'field')} if sc is None else {k: sc[k] + gc[k] for k in sc}
        vals.append({k: cos(sa[k], sc[k]) for k in sa})
    return {k: float(np.mean([v[k] for v in vals])) for k in ('ode', 'field')}


def controls(model, x, y):
    """Must-match, must-fail and injected-effect controls on one model and batch."""
    c = {}
    g1, l1 = grad(model, x, y, 'exact', 7)
    g2, _ = grad(model, x, y, 'exact', 7)
    c['mm1_repeatability_cos'] = min(cos(g1[k], g2[k]) for k in g1)
    ga, la = grad(model, x, y, 'adjoint', 7)
    c['mm2_loss_rel_diff'] = abs(la - l1) / abs(l1)
    c['mm2_downstream_cos'] = cos(ga['downstream'], g1['downstream'])
    ge, _ = grad(model, x, y, 'exact', 0, dropout=False)
    gl, _ = grad(model, x, y, 'loop', 0, dropout=False)
    c['mm3_exact_vs_loop_cos'] = min(cos(ge[k], gl[k]) for k in ge)
    c['pass'] = {
        'mm1': c['mm1_repeatability_cos'] >= 0.99999,
        'mm2': c['mm2_loss_rel_diff'] <= 1e-6 and c['mm2_downstream_cos'] >= 0.99999,
        'mm3': c['mm3_exact_vs_loop_cos'] >= 0.99999,
    }
    for key in ('ode', 'field'):
        gen = torch.Generator(device='cpu').manual_seed(99)
        v = g1[key].cpu()
        flipped = v * (torch.randint(0, 2, v.shape, generator=gen) * 2 - 1).to(v.dtype)
        c[f'mf1_sign_flip_cos_{key}'] = cos(flipped, v)
        noise = torch.randn(v.shape, generator=gen, dtype=v.dtype)
        c[f'inj_equal_norm_noise_cos_{key}'] = cos(v + noise * (v.norm() / noise.norm()), v)
        c['pass'][f'mf1_{key}'] = abs(c[f'mf1_sign_flip_cos_{key}']) <= 0.05
        c['pass'][f'injected_{key}'] = abs(c[f'inj_equal_norm_noise_cos_{key}'] - 0.70710678) <= 0.02
    return c


def pull(store, rel, attempts=4):
    """Store.pull with retries: Drive reads on Colab sometimes fail once and then work."""
    for i in range(attempts):
        if store.pull(rel):
            return True
        if i < attempts - 1:
            log(f'retrying {rel} in {15 * (i + 1)} s')
            time.sleep(15 * (i + 1))
    return False


def check_torchdiffeq():
    src = inspect.getsource(tdq_adjoint.OdeintAdjointMethod.backward)
    return {'version': getattr(torchdiffeq, '__version__', 'unknown'),
            'per_interval_state_reset': 'aug_state[1] = y[i - 1]' in src,
            'runs_odeint_per_interval': 't[i - 1:i + 1].flip(0)' in src}


# ------------------------------------------------------------------ main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--work', required=True, help="this job's Drive folder; unused except to find the study folder")
    ap.add_argument('--source', default=None, help='the WikiText-103 study folder (default: sibling wt103_study)')
    ap.add_argument('--local', default=None)
    args = ap.parse_args()
    source = Path(args.source) if args.source else Path(args.work).parent / 'wt103_study'
    local = Path(args.local) if args.local else (
        Path('/content/cdt_local/wt103_study') if Path('/content').is_dir() and not TEST else source)
    store = study.Store(source, local)
    out = Path(args.out)
    (out / 'results').mkdir(parents=True, exist_ok=True)
    t_start = time.time()
    rec = {'environment': study.env_record(), 'torchdiffeq': check_torchdiffeq(), 'config': {
        'seeds': SEEDS, 'n_batches': N_BATCHES, 'batch': BATCH, 'seq': SEQ, 'k_start': K_START, 'k_max': K_MAX,
        'batch_seed': BATCH_SEED, 'reference_min': REF_MIN, 's3_multipliers': S3_MULTIPLIERS, 'mm4_alpha': MM4_ALPHA},
        'source': str(source)}
    log('device:', DEV, '| test mode:', TEST, '| study folder:', source, '| torchdiffeq:', rec['torchdiffeq'])
    if not (rec['torchdiffeq']['per_interval_state_reset'] and rec['torchdiffeq']['runs_odeint_per_interval']):
        raise SystemExit('torchdiffeq adjoint backward differs from the version the preregistration describes. '
                         'Stopping before any gradient, as preregistered.')

    meta_rel = 'data/' + ('tokens_meta_test.json' if TEST else 'tokens_meta.json')
    if not pull(store, meta_rel):
        raise SystemExit(f'{meta_rel} not found in {source}')
    meta = json.loads(store.L(meta_rel).read_text())
    for split in ('train', 'test'):
        if not pull(store, f"data/{meta[split]['file']}") or study.sha256_file(store.L(f"data/{meta[split]['file']}")) != meta[split]['sha256']:
            raise SystemExit(f'{split} token file missing or does not match its fingerprint.')
    train = study.load_tokens(store, meta, 'train')
    gen = torch.Generator().manual_seed(BATCH_SEED)
    batches = []
    for _ in range(N_BATCHES):
        ix = torch.randint(len(train) - SEQ - 1, (BATCH,), generator=gen).numpy()
        t = torch.from_numpy(np.stack([np.asarray(train[i:i + SEQ + 1], dtype=np.int64) for i in ix])).to(DEV)
        batches.append((t[:, :-1], t[:, 1:]))
    sample = study.gpt2_encoding().decode(batches[0][0][0].tolist())
    (out / 'sample_batch.txt').write_text(sample)
    log('first window of batch 1, decoded:', repr(sample[:300]))
    tx, ty = study.windows(study.load_tokens(store, meta, 'test'), limit=64 if TEST else None)

    models = {}
    rec['weight_check'] = {}
    for seed in SEEDS:
        tag = f'main_hybrid_seed{seed}'
        if not (pull(store, f'final/{tag}.pt') and pull(store, f'results/test_{tag}.json')):
            raise SystemExit(f'{tag}: could not read the final weights or test record from {source}. If the log shows a '
                             'Drive error, choose Runtime > Disconnect and delete runtime, then Run all, to reconnect Drive.')
        m = study.new_model('hybrid', seed)
        m.load_state_dict(torch.load(store.L(f'final/{tag}.pt'), map_location='cpu'))
        recorded = json.loads(store.L(f'results/test_{tag}.json').read_text())['test_loss']
        now = study.eval_loss(m, tx, ty)
        rec['weight_check'][tag] = {'recorded': recorded, 'recomputed': now, 'pass': abs(now - recorded) <= 1e-4}
        log(f'{tag}: test loss recorded {recorded:.6f}, recomputed {now:.6f}')
        if not rec['weight_check'][tag]['pass']:
            raise SystemExit(f'{tag}: recomputed test loss does not match the record; wrong weights. Stopping.')
        models[seed] = m

    rec['controls'] = controls(models[SEEDS[0]], *batches[0])
    log('controls:', json.dumps(rec['controls']['pass']))
    rec['mm4'] = {str(s): dropout_off_cos(models[s], batches, alpha=MM4_ALPHA, no_time=True) for s in SEEDS}
    rec['mm4_pass'] = {key: all(v[f'cos_{key}'] >= 0.999 for v in rec['mm4'].values()) for key in ('ode', 'field')}
    log('mm4, time-independent field, alpha 0.001:', {s: round(v['cos_field'], 6) for s, v in rec['mm4'].items()})
    rec['s9_time_floor'] = {str(s): dropout_off_cos(models[s], batches, alpha=MM4_ALPHA) for s in SEEDS}
    # Controls are recorded, not acted on here: t0_analyze.py applies the decision rule, so a failed
    # control never loses the other measurements.
    (out / 'results' / 't0_results.json').write_text(json.dumps(rec, indent=2))

    k = K_START
    while True:
        log(f'primary measurement with {k} draws per gradient type')
        rec['primary'] = {}
        for s in SEEDS:
            t0 = time.time()
            rec['primary'][str(s)] = primary(models[s], batches, k)
            p = rec['primary'][str(s)]
            log(f"seed {s}: done ({time.time() - t0:.0f} s)")
        if all(min(v['reference_cos'], v['field_reference_cos']) >= REF_MIN for v in rec['primary'].values()) or k >= K_MAX:
            break
        k *= 2
        log(f'noise reference below {REF_MIN} in some model; doubling the draws to {k}, as preregistered')
    rec['k_used'] = k
    rec['precondition_met'] = {key: all(v[pre] >= REF_MIN for v in rec['primary'].values())
                               for key, pre in (('ode', 'reference_cos'), ('field', 'field_reference_cos'))}
    (out / 'results' / 't0_results.json').write_text(json.dumps(rec, indent=2))

    log('secondary: dose-response in alpha, dropout off (S2, S3)')
    rec['s3'] = {str(s): {str(mult): dropout_off_cos(models[s], batches, multiplier=mult) for mult in S3_MULTIPLIERS} for s in SEEDS}
    log('secondary: freshly initialised models (S4)')
    rec['s4'] = {str(s): primary(study.new_model('hybrid', s), batches, k) for s in SEEDS}
    log('exploratory: fp16 training gradient (S7)')
    rec['s7'] = {str(s): amp_cos(models[s], batches, k) for s in SEEDS} if DEV.type == 'cuda' else 'skipped: no GPU'
    rec['alpha_learned'] = {str(s): models[s].ode_block.func.output_scale.item() for s in SEEDS}
    rec['seconds'] = time.time() - t_start
    (out / 'results' / 't0_results.json').write_text(json.dumps(rec, indent=2))
    log(f"T0 complete in {rec['seconds'] / 60:.1f} min; results in results/t0_results.json")


if __name__ == '__main__':
    main()
