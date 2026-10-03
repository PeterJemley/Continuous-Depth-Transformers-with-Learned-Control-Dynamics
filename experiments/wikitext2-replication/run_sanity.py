"""Sanity checks from the preregistration, run and recorded before any result is read."""
from common import (CONTEXT_IDS, HELD_OUT_NOUNS, NEG_TOKEN, POS_TOKEN_CORRECT, POS_TOKEN_PAPER, TEST,
                    tokens, val_windows, write_result)
from pinned_data import gpt2_encoding


def guessing_range(n, p=0.5, mass=0.95):
    import math
    probs = [math.comb(n, i) * p ** i * (1 - p) ** (n - i) for i in range(n + 1)]
    tail = (1 - mass) / 2
    lo, c = 0, 0.0
    while c + probs[lo] <= tail:
        c += probs[lo]; lo += 1
    hi, c = n, 0.0
    while c + probs[hi] <= tail:
        c += probs[hi]; hi -= 1
    return lo / n, hi / n


def main():
    enc = gpt2_encoding()
    train = tokens('train')
    xs, _ = val_windows()
    held = {n: enc.encode(f'The {n} was') for n in HELD_OUT_NOUNS}
    payload = {
        'decoded': {'pos_correct': enc.decode([POS_TOKEN_CORRECT]), 'pos_paper': enc.decode([POS_TOKEN_PAPER]),
                    'neg': enc.decode([NEG_TOKEN]), 'context': enc.decode(CONTEXT_IDS)},
        'held_out_ids': held, 'held_out_all_three_tokens': all(len(v) == 3 for v in held.values()),
        'train_tokens': len(train), 'train_tokens_expected': 2391884,
        'train_tokens_match': TEST or len(train) == 2391884,
        'validation_windows': len(xs),
        'first_train_window_text': enc.decode(train[:64].tolist()),
        'first_validation_window_text': enc.decode(xs[0].tolist()),
        'guessing_range_95': {'held_out_per_seed_n40': guessing_range(40), 'held_out_pooled_n320': guessing_range(320),
                              'probe_test_n512': guessing_range(512)},
    }
    write_result('sanity', payload)
    for k, v in payload.items():
        print(k, ':', v if not isinstance(v, str) else repr(v[:160]))
    if not payload['train_tokens_match'] or not payload['held_out_all_three_tokens']:
        raise SystemExit('Sanity check failed; do not run the experiment.')


if __name__ == '__main__':
    main()
