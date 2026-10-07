"""Colab job wrapper. See experiments/too-linear-conjecture/t0_gradients.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _run import run  # noqa: E402

run(Path(__file__).resolve().parents[2] / 'experiments' / 'too-linear-conjecture/t0_gradients.py')
