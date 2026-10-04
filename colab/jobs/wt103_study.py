"""Colab job wrapper. See experiments/wikitext103-training/study.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _run import run  # noqa: E402

run(Path(__file__).resolve().parents[2] / 'experiments' / 'wikitext103-training/study.py')
