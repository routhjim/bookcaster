"""Model script: the lit twin of candlestick_brass_01 (same model; SPEC["variant"] == "lit")."""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "candlestick_brass_01", "candlestick_brass_01.py"),
               init_globals={"SPEC": SPEC})  # noqa: F821
