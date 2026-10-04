"""Colab job wrapper for test T0. See experiments/too-linear-conjecture/t0_gradients.py."""
import runpy
import sys
from pathlib import Path

sys.argv[0] = str(Path(__file__).resolve().parents[2] / 'experiments' / 'too-linear-conjecture' / 't0_gradients.py')
runpy.run_path(sys.argv[0], run_name='__main__')
