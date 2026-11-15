"""Two steps of the outer layer's fit (img3d/bl_layer.py) that need Blender's own mesh tools, run in Blender on small
made-up meshes: one short background run (no GPU, no rendering, about a second; skipped where Blender is not
installed). The script's own statements for each step are taken from its source and run on stand-in data.

1. The seam weld before the crumbs are removed. An image-to-3D model's file splits its vertices at UV seams;
   bl_layer.py counts the loose pieces left by the colour cut and deletes the small ones. Before the seams were
   welded every small UV island counted as a crumb and was deleted, leaving a layer in holes and strips. A sheet
   split into strips at seams, with one real crumb beside it, must come out as one piece; the same statements
   without the weld delete the whole sheet (the control that shows the test can fail).

2. The hang check: a layer passes when its top is at the neck, it is above the floor, it rests on at least one of
   the body's shoulders and most of its area is within reach of the body. On a simple body: a layer over both
   shoulders passes, the same layer over one shoulder only passes (no left-right centring), the layer moved off the
   body fails."""

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ASSETS = Path(__file__).resolve().parents[1] / "assets"
sys.path.insert(0, str(ASSETS))
import config as C  # noqa: E402

LAYER = ASSETS / "img3d" / "bl_layer.py"
STRIPS, COLS = 14, 30

INSIDE = r'''
import ast, json, math, sys
import bmesh
import bpy
import numpy as np
from mathutils import Matrix, Vector
from mathutils.bvhtree import BVHTree

src_path, out_path, strips, cols = sys.argv[sys.argv.index("--") + 1:][:4]
strips, cols = int(strips), int(cols)
src = open(src_path).read()
segs = [ast.get_source_segment(src, n) for n in ast.parse(src).body]


def span(first, last):
    i0 = next(i for i, s in enumerate(segs) if s.startswith(first))
    i1 = next(i for i, s in enumerate(segs) if i > i0 and s.startswith(last))
    return segs[i0:i1 + 1]


out = {}
# ---------------------------------------------------------------- 1. the weld before the crumbs go
step = span('REP["open_edges_before_weld_m"]', "bmesh.ops.delete(bm, geom=small")
out["weld_statements"] = len([s for s in step if "remove_doubles" in s])


def sheet():
    """a sheet of STRIPS x COLS quads, each strip with its own vertices (split at seams), and a loose crumb"""
    bm = bmesh.new()
    for k in range(strips):
        lo = [bm.verts.new((0.01 * c, 0.0, 0.01 * k)) for c in range(cols + 1)]
        hi = [bm.verts.new((0.01 * c, 0.0, 0.01 * (k + 1))) for c in range(cols + 1)]
        for c in range(cols):
            bm.faces.new((lo[c], lo[c + 1], hi[c + 1], hi[c]))
    bm.faces.new([bm.verts.new(p) for p in ((5.0, 0, 0), (5.01, 0, 0), (5.01, 0, 0.01), (5.0, 0, 0.01))])
    bm.verts.index_update()
    return bm


for name, code in (("welded", step), ("not_welded", [s for s in step if "remove_doubles" not in s])):
    bm = sheet()
    ns = {"bm": bm, "bmesh": bmesh, "np": np, "JOB": {}, "REP": {}}
    before = len(bm.verts)
    exec("\n".join(code), ns)
    out[name] = {"verts_before": before, "verts_after": len(bm.verts), "faces_after": len(bm.faces),
                 "pieces": ns["REP"].get("pieces"), "open_edges_before_weld_m": ns["REP"].get("open_edges_before_weld_m")}
    bm.free()

# ---------------------------------------------------------------- 2. the hang check
check = span("Pq = np.array(", 'REP["checks"]["ok"]')
JN = {"upperarm_l": (0.18, 0.0, 1.42), "upperarm_r": (-0.18, 0.0, 1.42), "clavicle_l": (0.03, 0.0, 1.45),
      "clavicle_r": (-0.03, 0.0, 1.45), "spine_05": (0.0, 0.0, 1.38), "neck_01": (0.0, 0.0, 1.50)}


def solid(rx, ry, rz, cz, z_lo, keep=None, shift=(0.0, 0.0, 0.0), name="m"):
    """an ellipsoid's upper half on an elliptic tube down to z_lo (a body with its shoulders, or a layer over it)"""
    bm = bmesh.new()
    seg, rings = 64, 24
    rows = [(cz + rz * math.sin(a), math.cos(a)) for a in np.linspace(math.pi / 2 - 0.02, 0.0, rings)]
    rows += [(z, 1.0) for z in np.linspace(cz, z_lo, 40)[1:]]
    grid = []
    for z, k in rows:
        grid.append([bm.verts.new((rx * k * math.cos(t) + shift[0], ry * k * math.sin(t) + shift[1], z + shift[2]))
                     for t in np.linspace(0, 2 * math.pi, seg, endpoint=False)])
    for r in range(len(grid) - 1):
        for c in range(seg):
            vs = (grid[r][c], grid[r][(c + 1) % seg], grid[r + 1][(c + 1) % seg], grid[r + 1][c])
            if keep is None or all(keep(v.co) for v in vs):
                bm.faces.new(vs)
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    me = bpy.data.meshes.new(name)
    bm.to_mesh(me)
    bm.free()
    ob = bpy.data.objects.new(name, me)
    bpy.context.collection.objects.link(ob)
    return ob


body = solid(0.20, 0.11, 0.30, 1.17, 0.0, name="body")
bpy.context.view_layer.update()
bv = np.array([v.co[:] for v in body.data.vertices])
out["hang"] = {}
cases = {"both_shoulders": {}, "one_shoulder": {"keep": lambda p: p[0] > 0.07},
         "moved_off_the_body": {"shift": (0.5, -0.1, 0.0)}}
for name, kw in cases.items():
    lay = solid(0.23, 0.14, 0.33, 1.17, 0.10, name=name, **kw)
    bpy.context.view_layer.update()
    ns = {"np": np, "bpy": bpy, "BVHTree": BVHTree, "Vector": Vector, "me": lay.data, "body": body, "bv": bv,
          "J": lambda n: np.array(JN[n]), "JOB": {}, "REP": {}}
    exec("\n".join(check), ns)
    out["hang"][name] = ns["REP"]["checks"]
    lay.hide_set(True)
json.dump(out, open(out_path, "w"))
'''


@unittest.skipUnless(Path(C.BLENDER).exists(), "Blender is not installed here")
class LayerInBlender(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        d = Path(tempfile.mkdtemp(prefix="char_layer_test_"))
        (d / "inside.py").write_text(INSIDE)
        out = d / "out.json"
        p = subprocess.run([str(C.BLENDER), "-b", "--factory-startup", "-t", "1", "--python", str(d / "inside.py"),
                            "--", str(LAYER), str(out), str(STRIPS), str(COLS)],
                           capture_output=True, text=True, timeout=120, env=dict(os.environ))
        assert out.exists(), f"the Blender run gave no result:\n{p.stdout[-2500:]}\n{p.stderr[-1500:]}"
        cls.r = json.loads(out.read_text())

    # ---- the weld
    def test_the_step_welds_the_seams(self):
        self.assertEqual(self.r["weld_statements"], 1)

    def test_a_sheet_split_at_seams_comes_out_as_one_piece(self):
        w = self.r["welded"]
        self.assertEqual(w["verts_before"], STRIPS * 2 * (COLS + 1) + 4)
        self.assertEqual(w["pieces"], 1)
        self.assertEqual(w["verts_after"], (STRIPS + 1) * (COLS + 1))       # the seams' doubles merged, the crumb gone
        self.assertEqual(w["faces_after"], STRIPS * COLS)                   # no face of the sheet lost
        self.assertGreater(w["open_edges_before_weld_m"], 0)

    def test_without_the_weld_every_strip_is_a_crumb(self):
        n = self.r["not_welded"]
        self.assertLess(2 * (COLS + 1), 300)            # each strip is under the default crumb size
        self.assertEqual(n["pieces"], 0)
        self.assertEqual(n["verts_after"], 0)

    # ---- the hang check
    def test_a_layer_over_both_shoulders_hangs(self):
        c = self.r["hang"]["both_shoulders"]
        self.assertTrue(c["ok"], c)
        self.assertEqual(sorted(c["rests_on"]), ["l", "r"])
        self.assertGreater(c["area_within_reach"], 0.95)

    def test_a_layer_over_one_shoulder_hangs_too(self):
        c = self.r["hang"]["one_shoulder"]
        self.assertTrue(c["ok"], c)
        self.assertEqual(c["rests_on"], ["l"])                              # the side it lies on, by the body's own side
        self.assertLess(c["shoulders"]["r"]["covered"], 0.42)

    def test_a_layer_moved_off_the_body_does_not(self):
        c = self.r["hang"]["moved_off_the_body"]
        self.assertFalse(c["ok"], c)
        self.assertEqual(c["rests_on"], [])
        both = self.r["hang"]["both_shoulders"]
        self.assertLess(c["area_within_reach"], both["area_within_reach"])

    def test_the_shoulder_patches_are_found_on_the_body(self):
        for name in ("both_shoulders", "one_shoulder", "moved_off_the_body"):
            for side in ("l", "r"):
                self.assertGreater(self.r["hang"][name]["shoulders"][side]["verts"], 10)


if __name__ == "__main__":
    unittest.main()
