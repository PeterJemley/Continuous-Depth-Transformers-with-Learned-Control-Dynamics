"""Fetch the records of N11's interrupted T4 session from Drive, for the hardware comparison of deviation D1.

Trains and measures nothing. It copies the JSON records in cdt-colab/n11_strength/results (the folder the T4
session used, next to this job's own folder) to the output folder for upload, and writes an inventory of every
file in that folder, so the upload shows exactly what the session left behind. The records are compared with
the A100 runs by n11_hardware.py.

    python n11_fetch_t4.py --out OUT --work /content/drive/MyDrive/cdt-colab/n11_fetch_t4
"""
import argparse
import json
import shutil
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--work', required=True)
    ap.add_argument('--source', default=None, help='the T4 session\'s folder (default: n11_strength next to --work)')
    args = ap.parse_args()
    src = Path(args.source) if args.source else Path(args.work).parent / 'n11_strength'
    out = Path(args.out)
    if not (src / 'results').is_dir():
        raise SystemExit(f'No results folder at {src}. Nothing was fetched.')
    inventory = [{'path': str(p.relative_to(src)), 'bytes': p.stat().st_size,
                  'modified_utc': time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(p.stat().st_mtime))}
                 for p in sorted(src.rglob('*')) if p.is_file()]
    (out / 'results').mkdir(parents=True, exist_ok=True)
    copied = []
    for p in sorted((src / 'results').glob('*.json')):
        json.loads(p.read_text())                        # a record cut off mid-save would fail here, not later
        shutil.copyfile(p, out / 'results' / p.name)
        copied.append(p.name)
    (out / 'inventory.json').write_text(json.dumps({'source': str(src), 'files': inventory}, indent=2))
    print(f'{len(inventory)} files in {src}; copied {len(copied)} records:', flush=True)
    for name in copied:
        print('  ', name, flush=True)


if __name__ == '__main__':
    main()
