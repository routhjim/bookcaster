"""Model script: the lit twin of forge_hearth_stone_01 (same model; SPEC["variant"] == "lit" makes the charcoal
glow)."""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "forge_hearth_stone_01",
                            "forge_hearth_stone_01.py"), init_globals={"SPEC": SPEC})  # noqa: F821
