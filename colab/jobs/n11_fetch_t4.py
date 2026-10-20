"""Colab job wrapper. See experiments/simultaneity-conjecture/n11_fetch_t4.py. Needs no GPU of any particular kind."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _run import run  # noqa: E402

run(Path(__file__).resolve().parents[2] / 'experiments' / 'simultaneity-conjecture' / 'n11_fetch_t4.py')
