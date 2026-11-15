"""Model script: wolf-jaw tongs, the "wolf" variant of tongs_flat_jaw_01 (same arms, bowed V-groove jaws)."""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tongs_flat_jaw_01",
                            "tongs_flat_jaw_01.py"), init_globals={"SPEC": SPEC})  # noqa: F821
