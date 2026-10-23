"""Colab job wrapper. See experiments/simultaneity-conjecture/n16_judge.py. Runs only on an A100, as N12-N15 did.
NEXT_JOB.json's args may carry --drop ARM[,ARM] for the exploratory arms the calibration's rule dropped."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _run import run  # noqa: E402

run(Path(__file__).resolve().parents[2] / 'experiments' / 'simultaneity-conjecture' / 'n16_judge.py', gpu='A100')
