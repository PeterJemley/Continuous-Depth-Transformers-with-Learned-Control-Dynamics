"""Colab connection test: checks every link in the loop and measures GPU speed.

Writes results.json into --out. The numbers size the next study's compute budget.
"""
import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / 'experiments' / 'wikitext2-replication'))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

# WikiText-103 has 103,227,021 words against WikiText-2's 2,088,628 (Merity et al., 2016).
# Scaling WikiText-2's measured GPT-2 token count gives an estimate, replaced by an exact
# count when WikiText-103 is pinned.
WT103_TOKENS_EST = round(2391884 * 103227021 / 2088628)


def timed_steps(kind, seq_len, amp, steps=20, warmup=5, batch=32):
    import paper_model as pm
    torch.manual_seed(0)
    cfg = dict(vocab_size=50257, d_model=256, n_heads=4, n_layers=6, max_seq_len=max(128, seq_len), dropout=0.1)
    model = (pm.BaselineTransformer(**cfg) if kind == 'baseline'
             else pm.HybridODETransformer(**cfg, ode_start=2, ode_end=4, control_dim=4, n_steps=4)).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=amp)
    x = torch.randint(0, 50257, (batch, seq_len), device='cuda')
    y = torch.randint(0, 50257, (batch, seq_len), device='cuda')
    torch.cuda.reset_peak_memory_stats()
    model.train()
    times = []
    for i in range(warmup + steps):
        torch.cuda.synchronize(); t = time.perf_counter()
        opt.zero_grad()
        with torch.autocast('cuda', dtype=torch.float16, enabled=amp):
            loss = F.cross_entropy(model(x).view(-1, 50257), y.view(-1))
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update()
        torch.cuda.synchronize()
        if i >= warmup:
            times.append(time.perf_counter() - t)
    step = sorted(times)[len(times) // 2]
    tok_s = batch * seq_len / step
    return {'kind': kind, 'seq_len': seq_len, 'amp_fp16': amp, 'median_step_s': step, 'tokens_per_s': tok_s,
            'loss_finite': bool(torch.isfinite(loss).item()), 'peak_mem_gb': torch.cuda.max_memory_allocated() / 1e9,
            'hours_per_wt103_pass_est': WT103_TOKENS_EST / tok_s / 3600}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--work', required=True)
    args = ap.parse_args()
    out, work = Path(args.out), Path(args.work)
    out.mkdir(parents=True, exist_ok=True)
    res = {'checks': {}, 'throughput': [], 'errors': []}

    res['environment'] = {'python': platform.python_version(), 'torch': torch.__version__,
                          'cuda_available': torch.cuda.is_available(),
                          'gpu': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                          'gpu_mem_gb': torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else None}
    print('environment:', res['environment'], flush=True)
    if not torch.cuda.is_available():
        raise SystemExit('No GPU. In Colab: Runtime > Change runtime type > T4 GPU, then run all cells again.')

    try:
        from pinned_data import gpt2_encoding, wikitext2_tokens
        n = len(wikitext2_tokens('train'))
        res['checks']['pinned_wikitext2_tokens'] = n
        res['checks']['pinned_data_ok'] = n == 2391884
        res['checks']['tokenizer_decodes_922'] = gpt2_encoding().decode([922])
        print('pinned data:', n, 'tokens; ok =', n == 2391884, flush=True)
    except Exception:
        res['errors'].append(traceback.format_exc()); res['checks']['pinned_data_ok'] = False

    try:
        work.mkdir(parents=True, exist_ok=True)
        probe = work / 'drive_write_test.txt'
        probe.write_text(time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()))
        res['checks']['drive_ok'] = probe.read_text() != ''
        print('Google Drive write and read: ok', flush=True)
    except Exception:
        res['errors'].append(traceback.format_exc()); res['checks']['drive_ok'] = False

    for kind in ('baseline', 'hybrid'):
        for seq_len in (64, 128):
            for amp in (False, True):
                try:
                    r = timed_steps(kind, seq_len, amp)
                    res['throughput'].append(r)
                    print(f"{kind:8s} len {seq_len:3d} fp16 {str(amp):5s}: {r['tokens_per_s']:,.0f} tokens/s, "
                          f"{r['median_step_s']*1000:.0f} ms/step, ~{r['hours_per_wt103_pass_est']:.1f} h per WikiText-103 pass", flush=True)
                except Exception:
                    res['errors'].append(f'{kind} {seq_len} amp={amp}:\n' + traceback.format_exc())
                    print(f'{kind} len {seq_len} fp16 {amp}: FAILED', flush=True)
                torch.cuda.empty_cache()

    res['wt103_tokens_estimate'] = WT103_TOKENS_EST
    (out / 'results.json').write_text(json.dumps(res, indent=2))
    print('wrote', out / 'results.json', '| errors:', len(res['errors']), flush=True)


if __name__ == '__main__':
    main()
