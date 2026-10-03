"""Colab job wrapper for the WikiText-103 training study. See experiments/wikitext103-training/study.py."""
import runpy
import sys
from pathlib import Path

sys.argv[0] = str(Path(__file__).resolve().parents[2] / 'experiments' / 'wikitext103-training' / 'study.py')
runpy.run_path(sys.argv[0], run_name='__main__')
