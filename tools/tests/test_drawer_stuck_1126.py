"""A drawing server that gives its card up does not look, to the asset line, like a command to run again."""
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "assets"))


def test_the_stuck_exit_is_not_one_of_the_lines_busy_codes():
    import config as C
    src = (HERE.parent / "qwen_image.py").read_text()
    rc = int(re.search(r"^SERVE_STUCK_RC = (\d+)", src, re.M).group(1))
    assert rc not in C.BUSY_RCS and rc != 0
    assert "unreadable job row" in src
