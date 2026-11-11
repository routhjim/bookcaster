"""Model script: wooden twin of bed_iron_single_01 (same bedding; SPEC["frame"] == "wood" builds the mahogany frame)."""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bed_iron_single_01",
                            "bed_iron_single_01.py"), init_globals={"SPEC": SPEC})  # noqa: F821
