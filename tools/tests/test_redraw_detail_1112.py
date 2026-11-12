"""Detail passes of the redraw (tools/assets/redraw/detail.py): geometry, colour matching, pasting, piece pairs,
the depth stretch, the lit composite and the keep-only-if-not-worse rule, on synthetic pictures."""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets" / "redraw"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "assets"))

import detail as D  # noqa: E402

W, H = 1152, 640


def test_size_class():
    assert D.size_class(None) == "unknown"
    assert D.size_class(D.HEAD_MAX) == "large" and D.size_class(D.HEAD_MAX + 0.1) == "large"
    assert D.size_class(D.HEAD_MAX - 0.01) == "detail"
    assert D.size_class(D.HEAD_MIN) == "detail" and D.size_class(D.HEAD_MIN - 0.01) == "sliver"
    assert D.size_class(0.2, visible=D.VISIBLE_MIN - 0.05) == "sliver"       # mostly out of frame: left alone


def test_head_of():
    j2 = {"head_top": [0.5, 0.2, True], "chin": [0.5, 0.4, True], "FACIAL_L_Eye": [0.52, 0.3, True],
          "FACIAL_R_Eye": [0.48, 0.3, True]}
    cx, top, chin = D.head_of(j2)
    assert abs(cx - 0.5) < 1e-9 and top == 0.2 and chin == 0.4
    assert D.head_of({"chin": [0.5, 0.4, True]}) is None and D.head_of({}) is None


def test_crop_box_inside_and_square():
    for cx, top, chin in ((0.5, 0.3, 0.5), (0.02, 0.0, 0.2), (0.99, 0.75, 0.95), (0.5, 0.05, 0.9)):
        x0, y0, x1, y1 = D.crop_box(cx, top, chin, W, H)
        assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H, (x0, y0, x1, y1)
        assert x1 - x0 == y1 - y0                                             # always square
        hh = (chin - top) * H
        assert x1 - x0 <= min(W, H) and x1 - x0 <= round(D.CROP_FACTOR * hh) + 1
    # in the open: the head sits in the upper part of the crop, the shoulders below it
    x0, y0, x1, y1 = D.crop_box(0.5, 0.3, 0.5, W, H)
    side = x1 - x0
    assert abs(side - D.CROP_FACTOR * 0.2 * H) <= 2
    assert y0 < 0.3 * H and y1 > 0.5 * H + 0.2 * H * 0.5
    assert abs((x0 + x1) / 2 - 0.5 * W) <= 1
    # a head at the frame's edge: the crop slides back in, same size
    xa = D.crop_box(0.02, 0.3, 0.5, W, H)
    assert xa[0] == 0 and xa[2] - xa[0] == side
    # a head taller than a third of the frame's height: the crop is the frame's shorter side at most
    xb = D.crop_box(0.5, 0.05, 0.9, W, H)
    assert xb[3] - xb[1] == H


def test_box_px():
    (x0, y0, x1, y1), (tw, th) = D.box_px([0.6, 0.6, 0.99, 0.99], W, H, pad=1.25)
    assert 0 <= x0 < x1 <= W and 0 <= y0 < y1 <= H
    assert tw % 32 == 0 and th % 32 == 0 and max(tw, th) == D.SIZE
    assert abs(tw / th - (x1 - x0) / (y1 - y0)) < 0.08                       # the aspect kept


def test_order_and_protection():
    ppl = [{"key": "a", "distance_m": 2.0}, {"key": "b", "distance_m": 4.5}, {"key": "c", "distance_m": 3.0},
           {"key": "d"}]
    assert [p["key"] for p in D.paint_order(ppl)] == ["b", "c", "a", "d"]     # farthest first; unknown last
    assert D.nearer_than(ppl[1], ppl) == ["a", "c"] and D.nearer_than(ppl[0], ppl) == []
    assert D.nearer_than(ppl[3], ppl) == []


def test_lab_round_trip():
    rng = np.random.default_rng(1)
    im = rng.integers(0, 256, (16, 16, 3), dtype=np.uint8)
    back = D.lab_to_srgb(D.srgb_to_lab(im))
    assert np.abs(back.astype(int) - im.astype(int)).max() <= 1
    lab = D.srgb_to_lab(np.array([[[255, 255, 255], [0, 0, 0]]], dtype=np.uint8))
    assert abs(lab[0, 0, 0] - 100) < 0.01 and abs(lab[0, 1, 0]) < 0.01


def _scene(shift=(0, 0, 0), gain=1.0):
    rng = np.random.default_rng(3)
    base = rng.integers(60, 180, (128, 128, 3)).astype(np.float64)
    new = np.clip((base - 120) * gain + 120 + np.array(shift), 0, 255)
    return base.astype(np.uint8), new.astype(np.uint8)


def test_colour_match_restores_the_surround():
    old, new = _scene(shift=(30, -10, 5), gain=0.8)                           # the redraw came back warmer and flatter
    ring = np.ones((128, 128), bool)
    ring[40:90, 40:90] = False
    out = D.colour_match(new, old, ring)
    before = np.abs(D.srgb_to_lab(new)[ring].mean(0) - D.srgb_to_lab(old)[ring].mean(0)).max()
    after = np.abs(D.srgb_to_lab(out)[ring].mean(0) - D.srgb_to_lab(old)[ring].mean(0)).max()
    assert before > 3 and after < 0.6, (before, after)
    # no ring to measure on: left as drawn
    assert np.array_equal(D.colour_match(new, old, np.zeros((128, 128), bool)), new)


def test_paste_alpha_and_feather_paste():
    m = np.zeros((200, 200), bool)
    m[60:140, 60:140] = True
    a = D.paste_alpha(m)
    assert a.min() >= 0 and a.max() <= 1 and a[100, 100] > 0.99 and a[5, 5] == 0
    assert 0 < a[100, 58] < 1                                                 # a soft edge, not a step
    assert a[0, :].max() == 0 and a[:, -1].max() == 0                         # nothing at the crop's own border
    # a nearer person's mask is protected
    prot = np.zeros_like(m)
    prot[60:140, 100:140] = True
    ap = D.paste_alpha(m, prot)
    assert ap[100, 125] < 0.02 and ap[100, 75] > 0.9
    # a mask running to the crop's border still fades there (a crop cut by the frame edge)
    m2 = np.zeros((200, 200), bool)
    m2[0:200, 80:120] = True
    a2 = D.paste_alpha(m2)
    assert a2[0, 100] == 0 and a2[100, 100] > 0.99
    base = np.full((300, 400, 3), 50, np.uint8)
    crop = np.full((200, 200, 3), 250, np.uint8)
    out = D.feather_paste(base, crop, (100, 50, 300, 250), a)
    assert out[150, 200].tolist() == [250, 250, 250] and out[10, 10].tolist() == [50, 50, 50]
    assert out[55, 105].tolist() == [50, 50, 50]                              # inside the box, outside the mask
    assert np.array_equal(base, np.full((300, 400, 3), 50, np.uint8))         # the input is not changed


def test_not_worse():
    assert D.not_worse({"a": ["outfit"]}, {})
    assert D.not_worse({"a": ["outfit", "hair"]}, {"a": ["hair"]})
    assert not D.not_worse({}, {"a": ["neck"]})
    assert not D.not_worse({"a": ["outfit"]}, {"a": ["outfit"], "set": ["look armchair"]})
    assert not D.not_worse({"a": ["outfit"]}, {"b": ["outfit"]})


def test_family_and_pairs():
    assert D.family("armchair_2") == "armchair" and D.family("wingback_armchair") == "armchair"
    assert D.family("leather_armchair") == "armchair" and D.family("desk") == "desk"
    things = [{"name": "armchair", "large": True, "box": [0.1, 0.5, 0.4, 0.95]},
              {"name": "armchair_2", "large": True, "box": [0.6, 0.5, 0.9, 0.95]},
              {"name": "desk", "large": True, "box": [0.0, 0.7, 0.3, 1.0]},
              {"name": "bookcase", "large": True, "box": [0.9, 0.0, 1.0, 0.6]},
              {"name": "bookcase_2", "large": True, "box": [0.0, 0.0, 0.004, 0.5]}]           # a sliver: not a pair
    assert D.lookalike_pairs(things) == ["armchair", "armchair_2"]
    assert D.lookalike_pairs(things[1:]) == []


def test_piece_mask_from_box():
    m = D.piece_mask(None, [0.25, 0.5, 0.5, 1.0], 100, 60)
    assert m.shape == (60, 100) and m[40, 30] and not m[10, 30] and not m[40, 80]


def test_prep_depth():
    far = np.linspace(1.0, 10.0, 256)[None, :].repeat(8, 0)                   # metres, larger = farther
    lin = D.prep_depth(far, inverse=False, clip=(0, 100))
    inv = D.prep_depth(far, inverse=True, clip=(0, 100))
    assert lin[0, 0] == 255 and lin[0, -1] == 0 and inv[0, 0] == 255 and inv[0, -1] == 0   # near = white in both
    assert inv[0, 64] < lin[0, 64]                                            # inverse gives the range to the near part
    # percentiles inside a mask: a far outlier outside it does not eat the range
    d = np.full((20, 20), 2.0)
    d[5:15, 5:15] = np.linspace(2.0, 2.4, 10)[None, :]
    d[0, 0] = 500.0
    mk = np.zeros((20, 20), bool)
    mk[5:15, 5:15] = True
    out = D.prep_depth(d, inverse=False, clip=(0, 100), mask=mk)
    assert out[10, 5] == 255 and out[10, 14] == 0


def test_composite_lit():
    key = np.full((60, 60, 3), 40, np.uint8)
    still = np.full((60, 60, 3), 30, np.uint8)
    still[20:30, 28:32] = (255, 220, 120)                                     # the flame, above the wick
    lit = np.zeros((60, 60), bool)
    lit[30:34, 28:32] = True                                                  # the wick's own mask
    out = D.composite_lit(key, still, lit)
    assert out[25, 30, 0] > 200 and out[50, 10].tolist() == [40, 40, 40]
    assert np.array_equal(D.composite_lit(key, still, np.zeros((60, 60), bool)), key)


def test_base_colour_and_block():
    assert D.base_colour("upholstered in dark red patterned damask fabric") == tuple(D.COLOURS["dark red"])
    assert D.base_colour("dark oxblood-brown leather") == tuple(D.COLOURS["oxblood"])
    assert D.base_colour("a thing of no stated colour") is None
    still = np.full((40, 80, 3), (120, 90, 60), np.uint8)
    still[:, ::2] += 10                                                       # its own shading
    a, b = np.zeros((40, 80), bool), np.zeros((40, 80), bool)
    a[:, :40], b[:, 40:] = True, True
    out = D.colour_block(still, [(a, D.COLOURS["burgundy"]), (b, D.COLOURS["gold"])], amount=0.7)
    la, lb = D.srgb_to_lab(out[:, :40]).reshape(-1, 3).mean(0), D.srgb_to_lab(out[:, 40:]).reshape(-1, 3).mean(0)
    assert la[1] > lb[1] + 8 and lb[2] > la[2] + 8                            # one redder, one yellower: told apart
    assert out[:, :40][:, ::2].mean() != out[:, :40][:, 1::2].mean()          # shading kept


def test_person_prompt():
    p = D.person_prompt({"noun": "woman", "trigger": "chr_x", "wear": "a black dress with a high collar"},
                        "hair pinned up, the hair black with grey streaks", "a black net veil over her face",
                        "She faces the camera; her face is seen from the front.", "A film still.")
    assert "one woman (chr_x)" in p and "grey streaks" in p and "veil" in p and "Nobody else" in p
    assert "  " not in p


def test_blocked_still_hook():
    import tempfile
    import redraw as R
    with tempfile.TemporaryDirectory() as td:
        out = Path(td)
        assert R.blocked_still({}, out, "sc", "cam", "/x/still.png") == "/x/still.png"
        assert R.blocked_still({"blocked_stills": True}, out, "sc", "cam", "/x/still.png") == "/x/still.png"   # none made
        b = out / "detail" / "blocked" / "sc__cam.png"
        b.parent.mkdir(parents=True)
        b.write_bytes(b"x")
        assert R.blocked_still({"blocked_stills": True}, out, "sc", "cam", "/x/still.png") == str(b)
        assert R.blocked_still({}, out, "sc", "cam", "/x/still.png") == "/x/still.png"                          # off by default


if __name__ == "__main__":
    n = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            n += 1
    print(f"ok: {n} tests")
