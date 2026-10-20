"""Shared wrapper for Colab jobs: runs a script and records how it ended.

Writes exit_status.json into the job's --out folder, so that upload_results.py can
tell a finished job from one that stopped early, whether it stopped with a traceback
or with a plain message.

A job that must run on one kind of GPU names it (gpu='A100'). The check runs before
the script starts, so a session given another GPU stops before anything is trained.
"""
import json
import runpy
import sys
from pathlib import Path


def gpu_name():
    import torch
    return torch.cuda.get_device_name(0) if torch.cuda.is_available() else None


def run(script, gpu=None):
    out = Path(sys.argv[sys.argv.index('--out') + 1]) if '--out' in sys.argv else None
    sys.argv[0] = str(script)

    def record(status, detail=''):
        if out is not None:
            out.mkdir(parents=True, exist_ok=True)
            (out / 'exit_status.json').write_text(json.dumps({'status': status, 'detail': str(detail)[:2000]}))

    try:
        if gpu is not None:
            name = gpu_name()
            if name is None or gpu.lower() not in name.lower():
                raise SystemExit(f'This job must run on an {gpu} GPU, but this session has {name or "no GPU"}. Nothing was trained. '
                                 f'Choose Runtime > Change runtime type > {gpu} GPU, Save, then Runtime > Run all again.')
            print(f'GPU check passed: {name}', flush=True)
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
