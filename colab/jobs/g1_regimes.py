"""Colab job wrapper. See experiments/simultaneity-conjecture/g1_regimes.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _run import run  # noqa: E402

run(Path(__file__).resolve().parents[2] / 'experiments' / 'simultaneity-conjecture' / 'g1_regimes.py')
