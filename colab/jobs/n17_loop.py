"""Colab job wrapper. See experiments/simultaneity-conjecture/n17_loop.py. Runs only on an A100, as N12-N16 did.
NEXT_JOB.json's args carry --rungs ARM,ARM[,ARM]: the loop-free rungs the calibration's rule chose."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _run import run  # noqa: E402

run(Path(__file__).resolve().parents[2] / 'experiments' / 'simultaneity-conjecture' / 'n17_loop.py', gpu='A100')
