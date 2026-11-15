"""Model script: the lit twin of lamp_oil_table_01 (same model; SPEC["variant"] == "lit" adds the flame and the
glowing globe)."""
import os
import runpy

runpy.run_path(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "lamp_oil_table_01",
                            "lamp_oil_table_01.py"), init_globals={"SPEC": SPEC})  # noqa: F821
