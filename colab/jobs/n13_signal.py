"""Colab job wrapper. See experiments/simultaneity-conjecture/n13_signal.py. Runs only on an A100, as N12 did."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _run import run  # noqa: E402

run(Path(__file__).resolve().parents[2] / 'experiments' / 'simultaneity-conjecture' / 'n13_signal.py', gpu='A100')
