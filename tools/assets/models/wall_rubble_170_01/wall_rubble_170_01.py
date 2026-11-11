"""Model script: wall_rubble_stone_01 at the width and seed in this spec (1.70 m)."""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "wall_rubble_stone_01",
                            "wall_rubble_stone_01.py"), init_globals={"SPEC": SPEC})  # noqa: F821
