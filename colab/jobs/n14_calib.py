"""Colab job wrapper. See experiments/simultaneity-conjecture/n14_dose.py. Runs only on an A100, as N12 and N13 did.
The calibration run: seed 3001, N12's calibration models, the held-out tail's windows 513-1024."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _run import run  # noqa: E402

if '--calibration' not in sys.argv:
    sys.argv.append('--calibration')
run(Path(__file__).resolve().parents[2] / 'experiments' / 'simultaneity-conjecture' / 'n14_dose.py', gpu='A100')
