"""Colab job wrapper. See experiments/simultaneity-conjecture/n18_loop.py. Runs only on an A100, as N12-N17 did.
The real run: seeds 4101-4105; trains rw-cloop, rw-ctwin, rw-j20, rw-peer, rw-j025, rw-j0125 and rw-nfree; re-measures N17's
peer, rw-j10 and rw-j05 and the saved models of N12-N16; the held-out tail's windows 1-512."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _run import run  # noqa: E402

run(Path(__file__).resolve().parents[2] / 'experiments' / 'simultaneity-conjecture' / 'n18_loop.py', gpu='A100')
