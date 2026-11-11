"""Model script: a short row of seven leather-bound books (period_kit.book_row), spines to -Y."""

import math
import os
import random
import sys

sys.path.insert(0, os.environ.get("ASSETS_MODELS_DIR") or os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import period_kit as K  # noqa: E402

leather = K.book_cloth("book_leather")
K.BOOK_COLOURS[:] = [(0.07, 0.025, 0.012), (0.10, 0.03, 0.015), (0.03, 0.045, 0.02), (0.11, 0.06, 0.025),
                     (0.05, 0.012, 0.01), (0.06, 0.035, 0.018)]
K.book_row("bk", -0.15, 0.15, 0.085, 0.0, 0.17, h_range=(0.20, 0.235), seed=4, lean_end=False, gaps=0.0, stacks=0,
           mat=leather)
