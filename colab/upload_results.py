"""Upload one job's output folder to the colab-results branch. Run by colab/runner.ipynb.

Reads the GitHub token from the GH_TOKEN environment variable and never prints it.
Does not touch Google Drive, so a Drive problem cannot block the upload.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time

TOKEN = os.environ.get('GH_TOKEN', '')


def run(cmd, cwd=None, check=True):
    r = subprocess.run(cmd, cwd=cwd, shell=True, capture_output=True, text=True)
    out = r.stdout + r.stderr
    if TOKEN:
        out = out.replace(TOKEN, '***')
    if check and r.returncode:
        raise RuntimeError(out)
    return out.strip()


def main():
    ap = argparse.ArgumentParser()
    for a in ('--out', '--job', '--run-id', '--commit', '--repo', '--branch'):
        ap.add_argument(a, required=True)
    args = ap.parse_args()
    if not TOKEN:
        print('No GitHub token, so nothing was uploaded. The notebook will download a zip instead.')
        return 2
    url = os.environ.get('UPLOAD_URL_OVERRIDE') or f'https://x-access-token:{TOKEN}@github.com/{args.repo}.git'
    R = '/content/results_repo' if os.path.isdir('/content') else os.path.abspath('results_repo')
    shutil.rmtree(R, ignore_errors=True)
    if run(f'git ls-remote --heads {url} {args.branch}'):
        run(f'git clone -q --depth 1 -b {args.branch} {url} {R}')
    else:
        os.makedirs(R)
        run('git init -q', cwd=R)
        run(f'git checkout -q -b {args.branch}', cwd=R)
        run(f'git remote add origin {url}', cwd=R)
    run('git config user.name "Colab runner"', cwd=R)
    run('git config user.email "colab-runner@users.noreply.github.com"', cwd=R)
    shutil.copytree(args.out, f'{R}/{args.job}/{args.run_id}')
    run('git add -A', cwd=R)
    run(f'git commit -q -m "Colab results: {args.job} {args.run_id}, code {args.commit[:7]}"', cwd=R)
    for attempt in range(4):
        try:
            run(f'git push -q origin {args.branch}', cwd=R)
            break
        except RuntimeError:
            if attempt == 3:
                raise
            run(f'git pull -q --rebase origin {args.branch}', cwd=R, check=False)
            time.sleep(2 ** (attempt + 1))
    run('git remote remove origin', cwd=R)
    where = f'the {args.branch} branch as {args.job}/{args.run_id}'
    log = os.path.join(args.out, 'log.txt')
    status_file = os.path.join(args.out, 'exit_status.json')
    status = json.load(open(status_file)).get('status') if os.path.exists(status_file) else None
    traceback_in_log = os.path.exists(log) and 'Traceback (most recent call last)' in open(log, errors='replace').read()
    if status in ('stopped', 'error') or traceback_in_log:
        print(f'The job STOPPED WITH AN ERROR. Its log was uploaded to {where}. Report that it stopped with an error.')
        return 1
    print(f'Done. Results uploaded to {where}. Report that it finished.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
