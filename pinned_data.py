"""Offline access to the pinned GPT-2 tokenizer and WikiText-2 copies in data/.

Every load checks the SHA-256 fingerprints recorded in data/MANIFEST.json, so a
run either uses exactly the pinned bytes or stops with an error.

    from pinned_data import gpt2_encoding, wikitext2_tokens

    enc = gpt2_encoding()
    tokens = wikitext2_tokens('train')   # 2,391,884 tokens, as in the original run
"""

import gzip
import hashlib
import json
from functools import lru_cache
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parent / 'data'
MANIFEST_PATH = DATA_DIR / 'MANIFEST.json'
SPLITS = ('train', 'validation', 'test')

__all__ = ['gpt2_encoding', 'wikitext2_rows', 'wikitext2_text', 'wikitext2_tokens']


def _manifest():
    return json.loads(MANIFEST_PATH.read_text(encoding='utf-8'))


def _sha256(data):
    return hashlib.sha256(data).hexdigest()


def _read_verified(relative_path, expected_sha256):
    data = (DATA_DIR / relative_path).read_bytes()
    actual = _sha256(data)
    if actual != expected_sha256:
        raise ValueError(f'{relative_path}: SHA-256 is {actual}, but MANIFEST.json pins {expected_sha256}')
    return data


@lru_cache(maxsize=1)
def gpt2_encoding():
    """The GPT-2 tokenizer, built from the pinned files with no network access."""
    import tiktoken
    from tiktoken.load import data_gym_to_mergeable_bpe_ranks
    from tiktoken_ext.openai_public import ENDOFTEXT, r50k_pat_str

    files = _manifest()['gpt2_tokenizer']['files']
    for name, meta in files.items():  # fail early with a clear message
        _read_verified(name, meta['sha256'])
    ranks = data_gym_to_mergeable_bpe_ranks(
        vocab_bpe_file=str(DATA_DIR / 'gpt2/vocab.bpe'),
        encoder_json_file=str(DATA_DIR / 'gpt2/encoder.json'),
        vocab_bpe_hash=files['gpt2/vocab.bpe']['sha256'],
        encoder_json_hash=files['gpt2/encoder.json']['sha256'],
    )
    return tiktoken.Encoding(
        name='gpt2',
        explicit_n_vocab=50257,
        pat_str=r50k_pat_str,
        mergeable_ranks=ranks,
        special_tokens={ENDOFTEXT: 50256},
    )


def wikitext2_rows(split='train'):
    """The rows of one WikiText-2 raw split, exactly as the Hugging Face dataset returns them."""
    if split not in SPLITS:
        raise ValueError(f'split must be one of {SPLITS}')
    entry = _manifest()['wikitext-2-raw-v1']
    meta = entry['files'].get(f'wikitext-2-raw-v1/{split}.jsonl.gz')
    if meta is None:
        raise FileNotFoundError('WikiText-2 is not pinned yet. Run scripts/fetch_wikitext2.py once.')
    text = gzip.decompress(_read_verified(f'wikitext-2-raw-v1/{split}.jsonl.gz', meta['sha256_gz']))
    if _sha256(text) != meta['sha256_text']:
        raise ValueError(f'{split}: decompressed text does not match MANIFEST.json')
    rows = [json.loads(line) for line in text.decode('utf-8').split('\n')[:-1]]
    if len(rows) != entry['expected']['rows'][split]:
        raise ValueError(f'{split}: {len(rows)} rows, expected {entry["expected"]["rows"][split]}')
    return rows


def wikitext2_text(split='train'):
    """The split's non-empty rows joined the way the original notebook joined them."""
    return '\n'.join(row for row in wikitext2_rows(split) if row.strip())


def wikitext2_tokens(split='train'):
    """The split encoded with the GPT-2 tokenizer, as a 1-D LongTensor."""
    import torch
    return torch.tensor(gpt2_encoding().encode(wikitext2_text(split)), dtype=torch.long)
