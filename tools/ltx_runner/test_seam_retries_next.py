"""Regression test (CPU, no model): every seam try failing must keep the smoothest try and flag it, not raise.
    ~/tts-ab/.venv-ltx/bin/python tools/ltx_runner/test_seam_retries.py"""
import os, sys, types
os.environ["HIP_VISIBLE_DEVICES"] = ""
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import torch
import ltx_split_next as L

calls = {"n": 0}


def fake_pipe(**kw):
    calls["n"] += 1
    return types.SimpleNamespace(video=[torch.full((5, 8, 8, 3), float(kw["seed"] % 7))])


steps = iter([7.5, 5.1, 6.2])
L.seam_ok = lambda prev, cf, frames, ov: (False, f"seam max step {next(steps):.2f} vs prev median 1.00")
L.log = lambda *a: None
j = {"name": "x_c1", "chain_from": "x_c0", "chain_frame": 3}
frames, j2 = L.gen_with_seam_retries(fake_pipe, {"seed": 42}, j, "cpu", __import__("pathlib").Path("."))
assert calls["n"] == 3, calls
assert "seam_flag" in j2 and "5.10" in j2["seam_flag"], j2
assert frames is not None and float(frames[0, 0, 0, 0]) == float((42 + 1000) % 7), "kept the smoothest (2nd) try"
# a passing try returns at once, no flag
L.seam_ok = lambda *a: (True, "seam max step 1.0 vs prev median 1.0")
calls["n"] = 0
frames, j3 = L.gen_with_seam_retries(fake_pipe, {"seed": 1}, j, "cpu", __import__("pathlib").Path("."))
assert calls["n"] == 1 and "seam_flag" not in j3
# an unchained job: one try
frames, j4 = L.gen_with_seam_retries(fake_pipe, {"seed": 1}, {"name": "y"}, "cpu", __import__("pathlib").Path("."))
assert frames is not None and "seam_flag" not in j4
print("ok: all-retries-failed keeps the best try and flags it")
