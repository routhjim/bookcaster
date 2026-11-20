"""The picture on a room's own faces (img3d/concept_depth.py room_faces): every texel of a wall of the measured room
takes the picture's colour where the picture sees that wall, and is unknown where something stands before it; the set
builder makes every wall, the floor and a flat ceiling whole from what the picture sees (concept_depth.whole_faces: a
seen wall carried along its unseen run by its own rhythm, else mirrored; the unseen wall continued from a neighbour
round its corner) and hangs them, the tiles only on a wall it cannot face (sets/place_build.py wall_pictures,
wall_faces); what is still unknown is filled from the tile (places/generate.py face_fill, face_picture_of)."""
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[0] / "assets" / "img3d"))
sys.path.insert(0, str(HERE.parents[0] / "assets" / "sets"))
sys.path.insert(0, str(HERE.parents[0] / "assets" / "places"))
sys.path.insert(0, str(HERE.parents[0] / "assets"))
import concept_depth as CD  # noqa: E402

F, W_PX, H_PX = 0.7, 320, 200            # normalised focal (of the width), picture size
EYE_H, BACK, SIDE, CEIL = 1.5, 6.0, 2.5, 3.2
PATCH = (-1.6, -0.6, 1.4, 2.4)           # a red patch on the back wall: x from, to; height from, to
BOX = (0.2, 1.4, 3.0, 3.6, 1.8)          # a blue box standing on the floor: x from, to; distance from, to; height
GREY, RED, BLUE = (120, 120, 120), (220, 30, 30), (30, 30, 220)
WHITE = (240, 240, 240)
STRIPE = 1.0                             # side walls with stripes: a white one this often along the room (metres) ...
STRIPE_ON = 0.35                         # ... this wide


def picture(stripes: bool = False):
    """A box room seen from inside (camera: x right, y down, z forward; the floor EYE_H under the eye): its points,
    the picture (grey, the patch red on the back wall, the box blue), the intrinsics."""
    jj, ii = np.mgrid[0:H_PX, 0:W_PX]
    fy = F * W_PX / H_PX
    r = np.stack([((ii + 0.5) / W_PX - 0.5) / F, ((jj + 0.5) / H_PX - 0.5) / fy, np.ones((H_PX, W_PX))], -1)
    inf = np.full((H_PX, W_PX), np.inf)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = np.minimum.reduce([
            np.where(r[..., 1] > 1e-9, EYE_H / r[..., 1], inf),
            np.where(r[..., 1] < -1e-9, -(CEIL - EYE_H) / r[..., 1], inf),
            BACK / r[..., 2],
            np.where(np.abs(r[..., 0]) > 1e-9, np.where(r[..., 0] > 0, SIDE, -SIDE) / r[..., 0], inf)])
        # the box: slabs in x, y (EYE_H - height .. EYE_H), z
        lo = np.array([BOX[0], EYE_H - BOX[4], BOX[2]])
        hi = np.array([BOX[1], EYE_H, BOX[3]])
        t0, t1 = (lo - 0 * r) / r, (hi - 0 * r) / r
        tn, tf = np.minimum(t0, t1).max(-1), np.maximum(t0, t1).min(-1)
    on_box = (tn <= tf) & (tn > 0) & (tn < t)
    t = np.where(on_box, tn, t)
    pts = r * t[..., None]
    img = np.empty((H_PX, W_PX, 3), np.uint8)
    img[:] = GREY
    back = (np.abs(pts[..., 2] - BACK) < 1e-6) & ~on_box
    hgt = EYE_H - pts[..., 1]
    img[back & (pts[..., 0] > PATCH[0]) & (pts[..., 0] < PATCH[1]) & (hgt > PATCH[2]) & (hgt < PATCH[3])] = RED
    img[on_box] = BLUE
    if stripes:                          # (the side walls' own rhythm: a white band every STRIPE along the room)
        side = (np.abs(np.abs(pts[..., 0]) - SIDE) < 1e-6) & ~on_box
        img[side & (np.mod(pts[..., 2], STRIPE) < STRIPE_ON)] = WHITE
    intr = np.array([[F, 0, 0.5], [0, fy, 0.5], [0, 0, 1]])
    return pts, np.ones((H_PX, W_PX), bool), img, intr


def faces(stripes: bool = False):
    pts, mask, img, intr = picture(stripes)
    fr = {}
    r = CD.room_from(pts, mask, intr, {}, {}, fr)
    return {f["face"]: (f, t) for f, t in CD.room_faces(pts, mask, intr, img, fr)}, fr, r


def metres(rec, tex):
    """Each texel's middle in the set's metres along the face's columns and rows."""
    h, w = tex.shape[:2]
    u = rec["u"][1] + (np.arange(w) + 0.5) / w * (rec["u"][2] - rec["u"][1])
    v = rec["v"][1] + (np.arange(h) + 0.5) / h * (rec["v"][2] - rec["v"][1])
    return np.meshgrid(u, v)


def is_(tex, c, tol=40):
    return (np.abs(tex[..., :3].astype(int) - c).max(-1) < tol) & (tex[..., 3] > 0)


def test_the_patch_lands_where_it_is_on_the_back_wall():
    got, fr, r = faces()
    assert "north" in got
    rec, tex = got["north"]
    assert abs(rec["at_m"] - r["room_m"][1] / 2) < 0.01 and rec["u"][0] == "x" and rec["v"][0] == "z"
    X, Z = metres(rec, tex)
    red = is_(tex, RED)
    assert red.sum() > 500
    # the patch's middle where the room has it: (its x from the camera less the room's middle) and its height
    cx = (PATCH[0] + PATCH[1]) / 2
    want_x = (cx - fr["cx"]) * fr["scale"]
    assert abs(X[red].mean() - want_x) < 0.08
    assert abs(Z[red].mean() - (PATCH[2] + PATCH[3]) / 2 * fr["scale"]) < 0.08
    assert abs((X[red].max() - X[red].min()) - (PATCH[1] - PATCH[0])) < 0.15       # its own size, undistorted
    assert abs((Z[red].max() - Z[red].min()) - (PATCH[3] - PATCH[2])) < 0.15
    assert 0 < rec["share"] <= 1 and rec["share_of_rect"] >= rec["share"]


def test_what_stands_before_the_wall_is_left_unknown_not_smeared_on_it():
    got, fr, _ = faces()
    rec, tex = got["north"]
    X, Z = metres(rec, tex)
    assert not is_(tex, BLUE).any()                      # the box is nowhere on the wall
    # the wall behind the box's middle, seen from the eye: unknown
    bx, bz, bh = (BOX[0] + BOX[1]) / 2, BOX[2], BOX[4] / 2
    wx = (bx * BACK / bz - fr["cx"]) * fr["scale"]
    wz = (EYE_H - (EYE_H - bh) * BACK / bz) * fr["scale"]
    k = np.unravel_index(np.argmin((X - wx) ** 2 + (Z - wz) ** 2), X.shape)
    assert abs(X[k] - wx) < 0.05 and abs(Z[k] - wz) < 0.05
    assert tex[k][3] == 0
    # ... and the wall beside it is known, its own grey
    k2 = np.unravel_index(np.argmin((X - (wx - 2.5)) ** 2 + (Z - 0.5) ** 2), X.shape)
    assert tex[k2][3] == 255 and is_(tex[k2][None], GREY).all()
    # the floor the box hides from the eye is unknown too; of the box only its foot (what is near the floor) lies
    # on the floor, under it and just behind it
    fl, ft = got["floor"]
    FX, FY = metres(fl, ft)
    assert (FY[is_(ft, BLUE)] < (BOX[3] - fr["cy"]) * fr["scale"] + 0.6).all()
    fy = (bz + 1.5 - fr["cy"]) * fr["scale"]
    k3 = np.unravel_index(np.argmin((FX - (bx - fr["cx"]) * fr["scale"]) ** 2 + (FY - fy) ** 2), FX.shape)
    assert ft[k3][3] == 0


def test_the_side_walls_read_as_seen_facing_them():
    got, fr, r = faces()
    W, D, _ = r["room_m"]
    w, e = got["west"][0], got["east"][0]
    assert w["u"][0] == "y" and w["u"][1] < w["u"][2]          # (facing the left wall, the back is on the right)
    assert e["u"][0] == "y" and e["u"][1] > e["u"][2]
    assert abs(w["at_m"] + W / 2) < 0.01 and abs(e["at_m"] - W / 2) < 0.01
    assert "south" not in got                                   # (behind the eye: never seen)


def test_a_room_with_no_pictured_walls_is_built_as_before(tmp_path):
    import place_build as PB
    spec = {"_concept": {"bare": {"walls": {"words": "rough grey stone."}}}}
    lib = tmp_path
    (lib / "prop" / PB.face_spec("rough grey stone.")["id"]).mkdir(parents=True)
    (lib / "prop" / PB.face_spec("rough grey stone.")["id"] / "asset.json").write_text("{}")

    def build(sp):
        out = {"size_m": [6.0, 9.0, 3.0], "structure": [{"kind": "wall"}], "fixtures": []}
        PB.wall_faces(out, {"openings": []}, sp, lib)
        return out
    a = build(spec)
    assert len(a["fixtures"]) == (2 * 2 + 2 * 3) and "face_pictures" not in a
    assert build(dict(spec, _concept=dict(spec["_concept"], faces=[])))["fixtures"] == a["fixtures"]


def test_the_unknown_part_of_a_wall_s_picture_is_filled_from_the_tile():
    import generate as G
    tex = np.zeros((40, 60, 4), np.uint8)
    tex[:, 30:, :3], tex[:, 30:, 3] = 100, 255
    tile = np.zeros((90, 90, 3), np.uint8)
    tile[..., 1] = 200
    out = G.face_fill(tex, 10.0, tile)
    assert out.shape == (40, 60, 3)
    assert (out[:, 30:] == 100).all()                          # what is known is the picture's
    assert (out[:, :30, 1] > 0).all() and (out[:, :30, 0] == 0).all()   # the rest the tile's, toned
    flat = G.face_fill(tex, 10.0, None, "#204060")
    assert tuple(flat[0, 0]) == (0x20, 0x40, 0x60)
    assert json.dumps(flat.tolist())                           # (plain numbers: a picture)


# ---------------------------------------------------------------- the whole room from what the picture sees

def whole(tmp_path, stripes=True):
    """The striped room's faces written beside a result, and made whole."""
    got, fr, r = faces(stripes)
    recs = CD.write_faces(tmp_path / "concept_depth.json", [(f, t) for f, t in got.values()])
    return CD.whole_faces(recs, r["room_m"]), recs, fr, r


def test_a_side_wall_s_rhythm_runs_on_over_what_the_picture_does_not_see(tmp_path):
    w, recs, fr, r = whole(tmp_path)
    W, D, H = r["room_m"]
    tex, ppm, how = w["west"]
    assert "rhythm" in how and tex.shape[1] == round(D * ppm) and tex.shape[0] == round(H * ppm)
    assert ppm <= min(f["px_per_m"] for f in recs) + 1e-6           # (no texel made up: the coarsest seen scale)
    seen0 = next(f for f in recs if f["face"] == "west")["u"][1]   # where the picture begins to see it (set y)
    y = -D / 2 + (np.arange(tex.shape[1]) + 0.5) / ppm             # (west: its columns run along +y)
    rows = slice(int((H - 2.0) * ppm), int((H - 1.0) * ppm))      # (a band of it at 1 to 2 m up)
    white = (tex[rows, :, :3].astype(int).min(-1) > 200).mean(0) > 0.5
    want = np.mod(y / fr["scale"] + fr["cy"], STRIPE) < STRIPE_ON  # (the stripes where the room has them)
    unseen = y < seen0 - 0.1
    assert unseen.sum() > ppm                                      # (a metre and more it never saw ...)
    assert (white[unseen] == want[unseen]).mean() > 0.85           # (... carries the stripes at their own spacing)
    assert (tex[..., 3] == 255).mean() > 0.97


def test_the_wall_behind_the_eye_is_a_side_wall_continued_not_the_far_wall(tmp_path):
    w, recs, fr, r = whole(tmp_path)
    assert "south" not in {f["face"] for f in recs}                # (never seen)
    tex, ppm, how = w["south"]
    assert how.startswith("the west wall") or how.startswith("the east wall")
    red = (np.abs(tex[..., :3].astype(int) - RED).max(-1) < 40)
    assert not red.any()                                           # (the far wall's own patch is not copied there)
    assert (tex[..., :3].astype(int).min(-1) > 200).mean() > 0.1   # (the side walls' stripes are)


def test_a_wall_with_no_rhythm_is_mirrored_at_the_join_without_a_cut(tmp_path):
    w, recs, fr, r = whole(tmp_path, stripes=False)
    tex, ppm, how = w["west"]
    assert "mirrored" in how
    rec = next(f for f in recs if f["face"] == "west")
    D = r["room_m"][1]
    j = int(round((rec["u"][1] + D / 2) * ppm))                    # where its seen run begins
    lum = tex[..., :3].astype(float).mean(-1)
    step = np.abs(np.diff(lum[:, max(0, j - 40):j + 40], axis=1)).mean()
    assert step < 12                                               # (no hard cut where the picture stops)


def test_the_floor_is_made_whole_and_its_light_evened(tmp_path):
    w, recs, fr, r = whole(tmp_path)
    tex, ppm, how = w["floor"]
    W, D, H = r["room_m"]
    assert tex.shape[:2] == (round(D * ppm), round(W * ppm))
    assert (tex[..., 3] == 255).mean() > 0.97                      # (behind the eye and under the box too)
    assert not (np.abs(tex[..., :3].astype(int) - BLUE).max(-1) < 40).any()


def rec_files(tmp_path, faces_=("north", "west", "floor", "ceiling")):
    """Neutral face records of a 6 x 9 x 3 room: each a grey picture, a quarter of it unknown, beside each other."""
    from PIL import Image
    room = [6.0, 9.0, 3.0]
    recs = []
    for name in faces_:
        lu, lv = {"north": (6.0, 3.0), "west": (9.0, 3.0), "floor": (6.0, 9.0), "ceiling": (6.0, 9.0)}[name]
        t = np.zeros((int(lv * 50), int(lu * 50), 4), np.uint8)
        t[..., :3] = 90 + (np.arange(t.shape[1]) % 25)[None, :, None]
        t[..., 3] = 255
        t[: t.shape[0] // 2, : t.shape[1] // 2, 3] = 0
        f = tmp_path / f"concept_face_{name}.png"
        Image.fromarray(t).save(f)
        u = {"north": ["x", -3.0, 3.0], "west": ["y", -4.5, 4.5], "floor": ["x", -3.0, 3.0], "ceiling": ["x", -3.0, 3.0]}[name]
        v = ["z", 3.0, 0.0] if name in ("north", "west") else ["y", 4.5, -4.5]
        recs.append({"face": name, "u": u, "v": v, "at_m": 0.0, "px_per_m": 50.0, "share": 0.75, "share_of_rect": 0.75,
                     "file": str(f), "room_m": room})
    return recs


def made(tmp_path, ids):
    for i in ids:
        (tmp_path / "prop" / i).mkdir(parents=True, exist_ok=True)
        (tmp_path / "prop" / i / "asset.json").write_text("{}")


def test_every_wall_the_floor_and_a_flat_ceiling_wear_the_room_s_picture(tmp_path):
    import place_build as PB
    recs = rec_files(tmp_path)
    spec = {"_concept": {"bare": {"walls": {"words": "rough grey stone."}}, "faces": recs}}
    pics = PB.face_pictures(spec)
    by = {f["face"]: fs for fs, f in pics}
    assert set(by) == {"north", "east", "south", "west", "floor", "ceiling"}
    assert by["north"]["size_m"] == [round(6.0 - 2 * PB.FACE_PIC_END_M, 3), 0.02, 3.0]
    assert by["east"]["size_m"] == [round(9.0 - 2 * PB.FACE_PIC_END_M, 3), 0.02, 3.0]
    assert by["floor"]["face_flat"] and by["floor"]["size_m"][2] == 0.02 and not by["north"]["face_flat"]
    assert len({fs["id"] for fs, _ in pics}) == 6
    made(tmp_path, [PB.face_spec("rough grey stone.")["id"]] + [fs["id"] for fs, _ in pics])
    out = {"size_m": [6.0, 9.0, 3.0], "structure": [{"kind": "wall"}, {"kind": "ceiling", "height_m": 3.0}],
           "fixtures": [{"asset": "x", "name": "drapes", "position": [0.0, 4.375, 0.0]}]}
    lay = {"openings": [{"id": "kind_window", "kind": "window", "wall": "north", "point": [0.0, 4.5, 0.0],
                         "width_m": 1.2, "stated": False, "name": "window"}]}
    PB.wall_faces(out, lay, spec, tmp_path)
    names = {x["name"]: x for x in out["fixtures"]}
    for side in ("north", "east", "south", "west"):
        assert f"wall_face_{side}_picture" in names
        assert not any(n.startswith(f"wall_face_{side}_") and n != f"wall_face_{side}_picture" for n in names)
    assert names["wall_face_north_picture"]["position"] == [0.0, round(4.5 - PB.FACE_PIC_OFF_M, 3), 0.0]
    assert names["floor_face_picture"]["position"] == [0.0, 0.0, PB.FACE_PIC_FLOOR_Z]
    assert names["ceiling_face_picture"]["position"][2] < 3.0
    assert "drapes" not in names                                   # (a guessed window under the picture: no drapes)
    assert set(out["room_faces"]) == {"north", "east", "south", "west", "floor", "ceiling"}


def test_an_open_roof_keeps_its_timbers_and_an_asked_for_door_keeps_its_wall(tmp_path):
    import place_build as PB
    recs = rec_files(tmp_path)
    spec = {"_concept": {"bare": {"walls": {"words": "rough grey stone."}}, "faces": recs}, "_roof": "rafters",
            "named_places": {"by the door": "o1"}}
    pics = PB.face_pictures(spec)
    assert "ceiling" not in {f["face"] for _, f in pics}            # (not even made)
    made(tmp_path, [PB.face_spec("rough grey stone.")["id"]] + [fs["id"] for fs, _ in pics])
    out = {"size_m": [6.0, 9.0, 3.0], "structure": [{"kind": "wall"}, {"kind": "trusses"}], "fixtures": []}
    lay = {"openings": [{"id": "o1", "kind": "door", "wall": "south", "point": [-2.4, -4.5, 0.0], "width_m": 1.0,
                         "stated": False, "name": "door"}]}
    PB.wall_faces(out, lay, spec, tmp_path)
    names = {x["name"] for x in out["fixtures"]}
    assert "wall_face_south_picture" not in names                  # (the door must be seen) ...
    assert any(n.startswith("wall_face_south_") for n in names)    # (... its wall takes the tiles, as before)
    assert "wall_face_north_picture" in names and "floor_face_picture" in names
    assert "ceiling_face_picture" not in names and "open roof" in out["room_faces"]["ceiling"]
    # a room built to another size takes none
    out2 = {"size_m": [8.0, 9.0, 3.0], "structure": [{"kind": "wall"}], "fixtures": []}
    PB.wall_faces(out2, {"openings": []}, spec, tmp_path)
    assert not any(x["name"].endswith("_picture") for x in out2["fixtures"])
    # the room's concept depth carries its faces into the spec
    con = {"space": "room", "depth": {"room_m": [6.0, 9.0, 3.0], "things": {}, "faces": [dict(recs[0], room_m=None)]}}
    sp = PB.with_concept({"pieces": []}, con)
    assert sp["_concept"]["faces"][0]["room_m"] == [6.0, 9.0, 3.0]


def test_each_panel_s_picture_is_its_face_less_its_ends_and_flat_ones_get_a_flat_model(tmp_path):
    import place_build as PB
    import generate as G
    recs = rec_files(tmp_path)
    spec = {"_concept": {"bare": {"walls": {"words": "rough grey stone."}, "floor": {"colour": "#403020"}}, "faces": recs}}
    for fs, f in PB.face_pictures(spec):
        rgb, how = G.face_picture_of(fs, None, "#808080")
        assert rgb is not None and rgb.dtype == np.uint8 and rgb.shape[2] == 3
        w, h = (fs["size_m"][0], fs["size_m"][1]) if fs["face_flat"] else (fs["size_m"][0], fs["size_m"][2])
        assert abs(rgb.shape[1] / 50.0 - w) < 0.05 and abs(rgb.shape[0] / 50.0 - h) < 0.05, (f["face"], rgb.shape)
    compile(G.FLAT_PANEL_SCRIPT % "/nowhere", "flat_panel.py", "exec")
