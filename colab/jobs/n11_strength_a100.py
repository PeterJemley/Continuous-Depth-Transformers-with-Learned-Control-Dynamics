"""Colab job wrapper. See experiments/simultaneity-conjecture/n11_strength.py.

N11 rerun from scratch on an A100 (deviation log, 2026-10-20). The job name gives it its own Drive folder,
cdt-colab/n11_strength_a100, so the interrupted T4 runs in cdt-colab/n11_strength are left untouched.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _run import run  # noqa: E402

run(Path(__file__).resolve().parents[2] / 'experiments' / 'simultaneity-conjecture' / 'n11_strength.py', gpu='A100')
