"""Shared wrapper for Colab jobs: runs a script and records how it ended.

Writes exit_status.json into the job's --out folder, so that upload_results.py can
tell a finished job from one that stopped early, whether it stopped with a traceback
or with a plain message.
"""
import json
import runpy
import sys
from pathlib import Path


def run(script):
    out = Path(sys.argv[sys.argv.index('--out') + 1]) if '--out' in sys.argv else None
    sys.argv[0] = str(script)

    def record(status, detail=''):
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
            (out / 'exit_status.json').write_text(json.dumps({'status': status, 'detail': str(detail)[:2000]}))

    try:
        runpy.run_path(sys.argv[0], run_name='__main__')
    except SystemExit as e:
        if e.code in (None, 0):
            record('finished')
        else:
            record('stopped', e.code)
        raise
    except BaseException as e:
        record('error', f'{type(e).__name__}: {e}')
        raise
    else:
        record('finished')
