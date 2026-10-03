#!/usr/bin/env bash
# Runs the whole preregistered replication in order, committing results after each seed.
# Resumable: any step whose result file already exists is skipped.
set -euo pipefail
cd "$(dirname "$0")"
PY="python3 -W ignore"
CTRL_LIMIT=512

commit_results() {
  git add results RESULTS.md 2>/dev/null || git add results
  if ! git diff --cached --quiet; then
    git commit -q -m "Replication results: $1" -m "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01KWudXwDuQPscSEUAPCSgFf"
    for wait in 2 4 8 16; do git push -q origin HEAD && break || sleep $wait; done
  fi
}

$PY run_sanity.py
for s in 42 0 1 2 3 4 5 6; do
  for k in baseline hybrid; do
    [ -f results/train_${k}_seed$s.json ] || $PY run_training.py --seed $s --kinds $k
  done
  [ -f results/solver_seed$s.json ] || $PY run_solver.py --seed $s --control-limit $CTRL_LIMIT
  for t in 922 220; do
    [ -f results/steer${t}_seed$s.json ] || $PY run_steering.py --seed $s --pos-token $t
    [ -f results/nfe${t}_seed$s.json ] || $PY run_nfe.py --seed $s --pos-token $t
  done
  for m in steer922 steer220 untrained; do
    [ -f results/probe_${m}_seed$s.json ] || $PY run_probes.py --seed $s --model $m
  done
  $PY analyze.py >/dev/null
  commit_results "seed $s"
done
[ -f results/h1_control.json ] || $PY run_h1_control.py
[ -f results/latency.json ] || $PY run_latency.py
$PY analyze.py
commit_results "complete"
echo "ALL DONE"
