"""Model script: the lit twin of lantern_dark_01 (same model; SPEC["variant"] == "lit")."""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lantern_dark_01", "lantern_dark_01.py"),
               init_globals={"SPEC": SPEC})  # noqa: F821
