"""Colab job wrapper. See experiments/simultaneity-conjecture/n15_sharpen.py. Runs only on an A100, as N12-N14 did."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _run import run  # noqa: E402

run(Path(__file__).resolve().parents[2] / 'experiments' / 'simultaneity-conjecture' / 'n15_sharpen.py', gpu='A100')
