"""The picture on a room's own faces (img3d/concept_depth.py room_faces): every texel of a wall of the measured room
takes the picture's colour where the picture sees that wall, and is unknown where something stands before it; the set
builder hangs each wall's picture over its part of the wall and the tiles cover only the rest (sets/place_build.py
wall_pictures, wall_faces), the unknown filled from the tile (places/generate.py face_fill)."""
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


def picture():
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
    intr = np.array([[F, 0, 0.5], [0, fy, 0.5], [0, 0, 1]])
    return pts, np.ones((H_PX, W_PX), bool), img, intr


def faces():
    pts, mask, img, intr = picture()
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


def test_a_wall_s_picture_hangs_over_its_part_and_the_tiles_fill_the_rest(tmp_path):
    import place_build as PB
    from PIL import Image
    tex = np.zeros((300, 600, 4), np.uint8)
    tex[..., :3], tex[..., 3] = 90, 255
    tex[:50, :50, 3] = 0
    f = tmp_path / "concept_face_north.png"
    Image.fromarray(tex).save(f)
    rec = {"face": "north", "u": ["x", -3.0, 3.0], "v": ["z", 3.0, 0.0], "x_m": [-3.0, 3.0], "z_m": [0.0, 3.0],
           "at_m": 4.5, "px_per_m": 100.0, "share": 0.9, "share_of_rect": 0.99, "file": str(f), "room_m": [6.0, 9.0, 3.0]}
    spec = {"_concept": {"bare": {"walls": {"words": "rough grey stone."}}, "faces": [rec]}}
    pics = PB.face_pictures(spec)
    assert len(pics) == 1 and pics[0][0]["id"].startswith("wall_face_") and pics[0][0]["size_m"] == [6.0, 0.02, 3.0]
    for i in (PB.face_spec("rough grey stone.")["id"], pics[0][0]["id"]):
        (tmp_path / "prop" / i).mkdir(parents=True)
        (tmp_path / "prop" / i / "asset.json").write_text("{}")
    out = {"size_m": [6.0, 9.0, 3.0], "structure": [{"kind": "wall"}], "fixtures": []}
    PB.wall_faces(out, {"openings": []}, spec, tmp_path)
    pic = [x for x in out["fixtures"] if x["asset"] == pics[0][0]["id"]]
    assert len(pic) == 1 and pic[0]["position"] == [0.0, round(4.5 - PB.FACE_PIC_OFF_M, 3), 0.0]
    assert pic[0]["facing"] == [0.0, -1.0]
    names = {x["name"] for x in out["fixtures"]}
    assert not any(n.startswith("wall_face_north_") and n != "wall_face_north_picture" for n in names)
    assert any(n.startswith("wall_face_east_") for n in names)      # (the walls it does not show keep their tiles)
    # a room built to another size takes none
    out2 = {"size_m": [8.0, 9.0, 3.0], "structure": [{"kind": "wall"}], "fixtures": []}
    PB.wall_faces(out2, {"openings": []}, spec, tmp_path)
    assert not any(x["asset"] == pics[0][0]["id"] for x in out2["fixtures"])
    # the room's concept depth carries its faces into the spec
    con = {"space": "room", "depth": {"room_m": [6.0, 9.0, 3.0], "things": {}, "faces": [rec]}}
    sp = PB.with_concept({"pieces": []}, con)
    assert sp["_concept"]["faces"][0]["room_m"] == [6.0, 9.0, 3.0]


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
