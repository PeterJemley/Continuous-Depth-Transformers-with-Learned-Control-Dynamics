"""One-time fetch that pins WikiText-2 raw into data/wikitext-2-raw-v1/.

Downloads the three Hugging Face parquet files (or reads them from --from-dir),
writes one gzip-compressed text file per split, checks the row counts and the
train token count against the original run, and records SHA-256 fingerprints
in data/MANIFEST.json. After this, pinned_data.py works with no network access.

    python scripts/fetch_wikitext2.py                    # download from Hugging Face
    python scripts/fetch_wikitext2.py --from-dir DIR     # use parquet files already on disk

Requires pyarrow (installed with the `datasets` package, and present in Colab).
"""

import argparse
import gzip
import hashlib
import io
import json
import sys
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
DATA_DIR = REPO / 'data'
OUT_DIR = DATA_DIR / 'wikitext-2-raw-v1'
HF_BASE = 'https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/wikitext-2-raw-v1'
SPLITS = ('train', 'validation', 'test')


def parquet_bytes(split, from_dir):
    name = f'{split}-00000-of-00001.parquet'
    if from_dir:
        return (Path(from_dir) / name).read_bytes(), f'file:{name}', None
    url = f'{HF_BASE}/{name}'
    with urllib.request.urlopen(url) as resp:
        return resp.read(), url, resp.headers.get('X-Repo-Commit')


def main():
    import pyarrow.parquet as pq

    ap = argparse.ArgumentParser()
    ap.add_argument('--from-dir', help='directory holding the three parquet files')
    args = ap.parse_args()

    manifest_path = DATA_DIR / 'MANIFEST.json'
    manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    entry = manifest['wikitext-2-raw-v1']
    expected = entry['expected']
    files, sources, outputs, train_rows = {}, {}, {}, None

    for split in SPLITS:
        raw, source, commit = parquet_bytes(split, args.from_dir)
        rows = pq.read_table(io.BytesIO(raw)).column('text').to_pylist()
        if len(rows) != expected['rows'][split]:
            sys.exit(f'{split}: got {len(rows)} rows, expected {expected["rows"][split]}')
        bad = [i for i, r in enumerate(rows) if not r.endswith('\n') or '\n' in r[:-1]]
        if bad:
            sys.exit(f'{split}: rows {bad[:5]} do not hold exactly one trailing newline')
        text = ''.join(rows).encode('utf-8')
        if [l + '\n' for l in text.decode('utf-8').split('\n')[:-1]] != rows:
            sys.exit(f'{split}: text file would not round-trip to the same rows')

        buf = io.BytesIO()
        with gzip.GzipFile(filename='', mode='wb', fileobj=buf, compresslevel=9, mtime=0) as gz:
            gz.write(text)
        gz_bytes = buf.getvalue()
        rel = f'wikitext-2-raw-v1/{split}.txt.gz'
        outputs[rel] = gz_bytes
        if split == 'train':
            train_rows = rows
        files[rel] = {'sha256_gz': hashlib.sha256(gz_bytes).hexdigest(),
                      'sha256_text': hashlib.sha256(text).hexdigest(),
                      'source_parquet_sha256': hashlib.sha256(raw).hexdigest()}
        sources[split] = {'url': source, 'hf_commit': commit}
        print(f'{split}: {len(rows)} rows, {len(gz_bytes):,} bytes compressed')

    # Verify the token count before writing anything, so a bad copy leaves no trace.
    from pinned_data import gpt2_encoding
    n = len(gpt2_encoding().encode('\n'.join(r for r in train_rows if r.strip())))
    print(f'train tokens: {n:,} (expected {expected["train_tokens_gpt2"]:,})')
    if n != expected['train_tokens_gpt2']:
        sys.exit('Token count differs from the original run. Nothing was written.')

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for rel, gz_bytes in outputs.items():
        (DATA_DIR / rel).write_bytes(gz_bytes)
    entry['files'] = files
    entry['sources'] = sources
    entry['status'] = 'pinned'
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n', encoding='utf-8')

    from pinned_data import wikitext2_tokens
    if len(wikitext2_tokens('train')) != n:
        sys.exit('Reading the pinned files back gave a different token count.')
    print('Pinned and verified.')


if __name__ == '__main__':
    main()
