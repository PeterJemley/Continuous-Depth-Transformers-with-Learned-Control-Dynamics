"""Exploratory, added after the N4 run: N4a's overlap on the untrained models of N4's own seeds, on its windows 1,537-1,792.

    python n4_init_baseline.py VALIDATION_U16_FILE results/n4_exploratory/untrained_overlap.json
"""
import json, sys, numpy as np, torch
sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
import n4_directions as n
torch.set_num_threads(4)
n.BATCH_W = 8
val = np.memmap(sys.argv[1], dtype=np.uint16, mode='r')
mx, my1, my2 = (t[1536:1792] for t in n.g.g1_windows(val, limit=1792))
out = {}
for seed in n.SEEDS:
    for arm in ('dual-sep', 'same-sep'):
        m = n.new_n4(seed, arm)
        Ua, Ub = n.estimate_subspaces(m, mx, my1, my2, arm, n.R)
        out[f'{arm}_seed{seed}'] = n.n2.overlap(Ua, Ub)
        print(arm, seed, round(out[f'{arm}_seed{seed}'], 4), flush=True)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
