"""Colab job wrapper: N12's full-length calibration run (seed 3001, the held-out tail's windows 513-1024).
See experiments/simultaneity-conjecture/n12_bottleneck.py. Runs only on an A100."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _run import run  # noqa: E402

if '--calibration' not in sys.argv:
    sys.argv.append('--calibration')
run(Path(__file__).resolve().parents[2] / 'experiments' / 'simultaneity-conjecture' / 'n12_bottleneck.py', gpu='A100')
