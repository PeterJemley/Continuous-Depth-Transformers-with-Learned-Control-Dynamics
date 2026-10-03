"""H7: inference latency of Model A against the baseline on this machine's CPU.

Batch 32, length 128, 5 warm-up passes, then 30 timed passes per model, repeated
in 3 separate processes. Weights don't affect speed, so fresh models are used.

    python run_latency.py
"""
import json
import statistics
import subprocess
import sys
import time

import torch

from common import new_model, write_result


def worker():
    base, hyb = new_model('baseline', 0).eval(), new_model('hybrid', 0).eval()
    x = torch.randint(0, 1000, (32, 128), generator=torch.Generator().manual_seed(0))
    out = {}
    with torch.no_grad():
        for name, m in (('baseline', base), ('hybrid', hyb)):
            for _ in range(5):
                m(x)
            times = []
            for _ in range(30):
                t = time.perf_counter(); m(x); times.append(time.perf_counter() - t)
            out[name] = times
    print(json.dumps(out))


def main():
    runs = []
    for i in range(3):
        r = subprocess.run([sys.executable, __file__, '--worker'], capture_output=True, text=True, check=True)
        t = json.loads(r.stdout.strip().splitlines()[-1])
        mb, mh = statistics.median(t['baseline']), statistics.median(t['hybrid'])
        runs.append({'process': i, 'median_baseline_s': mb, 'median_hybrid_s': mh, 'ratio': mh / mb, 'times': t})
        print(f'process {i}: baseline {mb*1000:.0f} ms, hybrid {mh*1000:.0f} ms, ratio {mh/mb:.3f}', flush=True)
    write_result('latency', {'runs': runs, 'batch': 32, 'length': 128, 'warmup': 5, 'timed': 30})


if __name__ == '__main__':
    worker() if '--worker' in sys.argv else main()
