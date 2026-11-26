"""A room whose faces carry their own pictures (sets/place_build.py face_pictures, wall_pictures): the panel's picture
is laid over its whole texture by its own UVs, kept by the harness (models/period_kit.py picture_uvs, "keep_uvs"; the
harness's unwrap gave a long wall 38 texels of height); a wall that carries its picture is built bare behind it, so
nothing the shell builds stands proud of the panel (sets/layout.py build_wall); the room's soft lights hang well
under a ceiling that carries its picture (room_light). No engine, no Blender."""
import ast
import json
import sys
import types
from pathlib import Path

TOOLS = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TOOLS / "assets" / "sets"))
sys.path.insert(0, str(TOOLS / "assets" / "places"))
sys.path.insert(1, str(TOOLS / "assets"))
from test_room_faces_1120 import made, rec_files  # noqa: E402

import layout as LY  # noqa: E402
import place_build as PB  # noqa: E402

# a wall as a period kit dresses it: dado, skirting, rails, cornice, raised plates, and a guessed door
TRIM = [{"z_m": 0.0, "height_m": 0.22, "depth_m": 0.025, "material": "trim"},
        {"z_m": 0.88, "height_m": 0.07, "depth_m": 0.04, "material": "trim"},
        {"z_m": 3.04, "height_m": 0.16, "depth_m": 0.14, "material": "trim"}]
DOOR = {"panel_m": [0.22, 3.0], "material": "door", "frame_material": "door", "braced": False, "frame_m": 0.09,
        "header_m": [0.12, 0.05], "kind": "door", "from_m": -1.5, "to_m": -0.5, "sill_m": 0.0, "head_m": 2.15}
WIN = {"kind": "window_band", "frame_material": "frame", "glass_material": "glass", "from_m": 1.0, "to_m": 2.2,
       "sill_m": 0.9, "head_m": 2.3}


def dressed(side, openings=()):
    return {"kind": "wall", "side": side, "material": "plaster", "thickness_m": 0.25,
            "dado": {"height_m": 0.95, "material": "dado"}, "trim": json.loads(json.dumps(TRIM)),
            "plates": [{"z_m": [0.24, 0.86], "size_m": [0.62, 0.62], "gap_m": 0.07, "depth_m": 0.018, "material": "p"}],
            "columns": [{"at_m": [0.0], "depth_m": 0.3, "width_m": 0.3, "material": "c"}],
            "openings": [json.loads(json.dumps(o)) for o in openings]}


class Rec:
    """A stand-in for layout.Scene: every box, beam and member the shell builds."""

    def __init__(self):
        self.boxes, self.others = [], []

    def box(self, c, s, mat, folder, yaw=0.0, tile=True):
        self.boxes.append((c, s, mat))

    def ibeam(self, *a, **k):
        self.others.append(("ibeam", a))

    def member(self, *a, **k):
        self.others.append(("member", a))


def proud_m(wd, W, D, H):
    """How far into the room the most proud thing the shell builds on this wall stands (from its inner face)."""
    S = Rec()
    LY.build_wall(S, {}, wd, W, D, H)
    w = LY.Wall(wd["side"], W, D, wd["thickness_m"])
    inner = w.org[0] * w.out[0] + w.org[1] * w.out[1]
    deep = 0.0
    for c, s, _ in S.boxes:
        o = c[0] * w.out[0] + c[1] * w.out[1] - inner
        deep = max(deep, s[1] / 2 - o)
    return (float("inf") if S.others else deep), S


def room(tmp_path, named=None):
    recs = rec_files(tmp_path)
    spec = {"_concept": {"faces": recs}, "named_places": named or {}}
    pics = PB.face_pictures(spec)
    made(tmp_path, [fs["id"] for fs, _ in pics])
    out = {"size_m": [6.0, 9.0, 3.0], "fixtures": [], "lights": [],
           "structure": [dressed("north", [DOOR, WIN]), dressed("south", [DOOR]), dressed("east"), dressed("west"),
                         {"kind": "ceiling", "height_m": 3.0}]}
    return spec, pics, out


def test_the_dressed_shell_would_stand_through_the_panel():
    # (the rule's own premise: the kit's rail, cornice, plates, pilasters and the door's header stand out further
    # than the panel hangs)
    deep, _ = proud_m(dressed("east"), 6.0, 9.0, 3.0)
    assert deep > PB.FACE_PIC_OFF_M
    deep, _ = proud_m(dict(dressed("north", [DOOR, WIN]), trim=[], plates=[], columns=[]), 6.0, 9.0, 3.0)
    assert deep > PB.FACE_PIC_OFF_M                                # (a window's sill, a door's header)


def test_a_wall_that_carries_its_picture_is_bare_behind_it(tmp_path):
    spec, pics, out = room(tmp_path)
    lay = {"openings": [{"id": "o9", "kind": "door", "wall": "north", "point": [-1.0, 4.5, 0.0], "width_m": 1.0,
                         "stated": False, "name": "door"}]}
    PB.wall_faces(out, lay, spec, tmp_path)
    walls = {w["side"]: w for w in out["structure"] if w.get("kind") == "wall"}
    for side in ("north", "south", "east", "west"):
        assert f"wall_face_{side}_picture" in {x["name"] for x in out["fixtures"]}
        w = walls[side]
        assert not any(w.get(k) for k in PB.SHELL_DRESSING) and not w.get("openings"), side
        deep, S = proud_m(w, 6.0, 9.0, 3.0)
        assert deep <= 1e-6 < PB.FACE_PIC_OFF_M, (side, deep)        # nothing stands out of the wall's face
        assert {m for _, _, m in S.boxes} == {"plaster"}             # (no door, frame, sill or glass drawn)


def test_a_wall_that_keeps_its_asked_for_door_keeps_its_shell(tmp_path):
    spec, pics, out = room(tmp_path, named={"by the door": "o1"})
    lay = {"openings": [{"id": "o1", "kind": "door", "wall": "south", "point": [-1.0, -4.5, 0.0], "width_m": 1.0,
                         "stated": False, "name": "door"}]}
    PB.wall_faces(out, lay, spec, tmp_path)
    walls = {w["side"]: w for w in out["structure"] if w.get("kind") == "wall"}
    assert "wall_face_south_picture" not in {x["name"] for x in out["fixtures"]}
    assert walls["south"]["openings"] and walls["south"]["trim"]    # (no picture on it: built as before)
    assert not walls["north"].get("trim")


def test_every_panel_keeps_its_own_uvs_and_the_rule_is_in_its_id(tmp_path, monkeypatch):
    recs = rec_files(tmp_path)
    spec = {"_concept": {"faces": recs}}
    pics = PB.face_pictures(spec)
    assert pics and all(fs.get("keep_uvs") is True for fs, _ in pics)
    assert PB.FACE_PIC_RULE != "whole-1"                             # (panels made before the UVs were kept: stale)
    ids = {fs["id"] for fs, _ in pics}
    monkeypatch.setattr(PB, "FACE_PIC_RULE", PB.FACE_PIC_RULE + "-older")
    assert not ids & {fs["id"] for fs, _ in PB.face_pictures(spec)}  # (a panel of another rule is never reused)


def kit_fn(name):
    """One function of models/period_kit.py (it imports bpy: its source run alone)."""
    src = (TOOLS / "assets" / "models" / "period_kit.py").read_text()
    node = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == name)
    ns = {}
    exec(compile(ast.Module([node], []), "period_kit.py", "exec"), ns)
    return ns[name]


class FakeMesh:
    """A box's mesh as bpy has it: vertices, loops (one per face corner), UV layers."""

    def __init__(self, w, h, t):
        V = types.SimpleNamespace
        self.vertices = [V(co=V(x=x, y=y, z=z)) for x in (-w / 2, w / 2) for y in (-h / 2, h / 2) for z in (0.0, t)]
        faces = [[0, 2, 6, 4], [1, 5, 7, 3], [0, 1, 3, 2], [4, 6, 7, 5], [0, 4, 5, 1], [2, 3, 7, 6]]
        self.loops = [V(index=i, vertex_index=v) for i, v in enumerate(x for f in faces for x in f)]
        mesh = self

        class Layers(list):
            def new(self, name):
                lay = V(name=name, data=[V(uv=None) for _ in mesh.loops])
                self.append(lay)
                return lay
        self.uv_layers = Layers()
        self.uv_layers.new("old")


def test_a_long_panel_s_picture_spans_the_whole_texture_both_ways():
    picture_uvs = kit_fn("picture_uvs")
    for w, h in ((21.6, 8.0), (10.7, 8.0), (10.8, 21.6)):
        me = FakeMesh(w, h, 0.02)
        picture_uvs(types.SimpleNamespace(data=me))
        assert len(me.uv_layers) == 1
        uv = [d.uv for d in me.uv_layers[0].data]
        top = [uv[lp.index] for lp in me.loops if me.vertices[lp.vertex_index].co.z > 0.01]
        for k in (0, 1):                                  # (the face's corners: 0 and 1 along both axes)
            assert min(u[k] for u in top) == 0.0 and max(u[k] for u in top) == 1.0
        for lp in me.loops:                               # (each corner where its Generated X / Y is)
            co, u = me.vertices[lp.vertex_index].co, uv[lp.index]
            assert abs(u[0] - (co.x / w + 0.5)) < 1e-9 and abs(u[1] - (co.y / h + 0.5)) < 1e-9


def run_script(src, spec, path="script.py"):
    calls = []
    K = types.ModuleType("period_kit")
    K.box = lambda name, size, loc=(0, 0, 0), **kw: calls.append(("box", size)) or types.SimpleNamespace(
        name=name, matrix_basis=1, matrix_world=None)
    K.image_material = lambda *a, **kw: "mat"
    K.picture_uvs = lambda ob: calls.append(("uvs", ob.name)) or ob
    mu = types.ModuleType("mathutils")

    class M:
        def __matmul__(self, o):
            return self
        __rmatmul__ = __matmul__
    mu.Matrix = types.SimpleNamespace(Translation=lambda *a: M(), Rotation=lambda *a: M())
    saved = {k: sys.modules.get(k) for k in ("period_kit", "mathutils")}
    sys.modules.update(period_kit=K, mathutils=mu)
    try:
        exec(compile(src, path, "exec"), {"SPEC": spec, "__file__": path})
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    return calls


def test_both_panel_models_lay_their_picture_over_the_whole_texture(tmp_path):
    import generate as G
    spec = {"id": "x_face", "backdrop": {"w": 21.6, "h": 8.0, "emit": 0.0}}
    for src in (G.FLAT_PANEL_SCRIPT % "/nowhere",
                (TOOLS / "assets" / "models" / "backdrop_view" / "backdrop_view.py").read_text()):
        calls = run_script(src, spec)
        assert ("box", (21.6, 8.0, 0.02)) in calls and ("uvs", "x_face_pic") in calls, calls


def test_the_room_s_lights_hang_well_under_a_ceiling_that_carries_its_picture():
    def lit(size, pictured):
        out = {"size_m": list(size), "structure": [{"kind": "ceiling"}], "lights": [],
               "fixtures": [{"name": "ceiling_face_picture"}] if pictured else []}
        assert PB.room_light(out, {})
        return [l["position"][2] for l in out["lights"]]
    H = 8.0
    zs = lit((10.8, 21.6, H), True)
    gap = max(10.8 / round(10.8 / PB.ROOM_LIGHT_GAP_M), 21.6 / round(21.6 / PB.ROOM_LIGHT_GAP_M))
    assert all(H - z >= 0.5 * gap and z >= PB.ROOM_LIGHT_HEAD_M for z in zs), zs
    assert set(lit((6.0, 9.0, 3.0), True)) == {round(3.0 - PB.ROOM_LIGHT_UNDER_M, 2)}   # (a low room: as before)
    assert set(lit((10.8, 21.6, H), False)) == {round(H - PB.ROOM_LIGHT_UNDER_M, 2)}    # (no picture over it: as before)
