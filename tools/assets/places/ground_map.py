"""A place in the open's ground map: its own land drawn from straight above, and what the vision judge reads in it.

    python tools/assets/scenes/concept.py --map MAP.png PLACES.json OUT.json     (the judge's reading, read_map below)

Why: the set's real ground (img3d/pano_shell.py ground_disk, FLAT_R_M round the point under the eye) showed one small
square of the panorama's bare ground repeated over the whole disk. A river bank or a plain had nothing of its land on
it: every key's redraw invented another river, and nobody could be stood at a water's edge that was not there (11-18).
The panorama is told to show bare ground near the eye (WORLD_WORDS), so it cannot give the land near the people. The
map is a picture of its own: the place's ground seen from straight above, drawn from the place's words with its
concept picture as the reference (places/pieces.py, the world's "map" step), laid over the whole disk.

The map's frame is the set's frame: its middle the point under the panorama's eye (the set's origin, sets/place_build.py
world_stage), +y (north, the top of the picture) away from the concept picture's camera, as the concept's things are
placed (sets/place_build.py open_places), x to the right. A point of the picture in thousandths (u from the left, v
from the top, 0..1000, as the judge answers) is to_m(u, v) metres.

Water is kept as a grid of cells (WATER_CELL_M) over the map, as runs along each row: the ground disk's water faces and
the staging's dry ground (scenes/from_plan.py) read the same cells, so the people and the mesh never disagree.

Nothing here draws, runs a model or touches a file but the map picture it is given; numpy and Pillow only.
"""
from __future__ import annotations

import math
import re

HALF_M = 40.0               # the map covers the set's real ground: this far each way from its middle (pano_shell FLAT_R_M)
MAP_PX = 1024               # the map is drawn this many points a side
WATER_CELL_M = 0.5          # water is known to this many metres (the ground disk's own cells, so the mesh steps as
#                             the staging does)
WATER_GROW_M = 3.0          # the judge's outline is corrected by the picture's own colours up to this far out ...
WATER_TOL = 70.0            # ... where the map is this near the water's colour (summed over red, green, blue). The
#                             judge's outline of a river came back in a handful of corners, metres off the drawn bank
WATER_MIN_CELLS = 8         # less water than this many cells is a puddle the judge saw, not water to keep people from
WATER_MOST = 0.85           # water over more than this share of the map is the judge taking the whole land for water
NEAR_MOST_M = 6.0           # a land place the judge put in the water is moved this far at most to the dry ground by it

# (no person is named anywhere in these words, as for the panorama's: a picture told of a person's place drew one)
# (the reference picture is the land's look, not its plan: a place's picture "seen whole" was a few metres of bank
# close up, 11-18. What the ground is made of and what lies where by the water is taken from it, and carried on over
# the whole map; its way into the picture is the map's top)
MAP_WORDS = (
    # (drawn from words alone: given the place's picture as its reference, the map came out as that picture again,
    # a few metres of bank seen from a height at a slant, its pebbles forty metres of boulders on the set, 11-18)
    "An aerial survey view of open land from very high above, looking straight down: a "
    "flat plan view, square, with no perspective, no horizon and no sky. The picture is {span} metres of land "
    "from edge to edge, so everything in it is small: a single stone or a tuft of grass is too small to see, a "
    "footpath is a thin line, a grown tree would be a small dot. The land: {land}. The same land goes on to "
    "every edge of the picture; the middle of the picture is open dry ground.{places} It shows only the ground "
    "itself and what lies flat on it: any water with its banks and edges, paths and tracks, bare earth, grass, "
    "mud, sand, gravel. Nothing stands up on it: no buildings, no walls, no trees, no bushes, no posts, only the "
    "ground where such things would stand. No people and no animals. No shadows: even, flat light from straight "
    "above. No writing, no labels, no compass, no frame and no grid: the ground fills the whole picture to every "
    "edge.{style}")

WATER_ASK = ("This picture is a map of a piece of land seen from straight above. Outline every stretch of water on it "
             "(a river, a stream, a pond, a lake, the sea, a ford) as a polygon of at most 16 corner points going round "
             "its edge, each point as [x, y]: thousandths of the map's width (x, from the left) and of its height (y, "
             "from the top), 0 to 1000. One stretch of water per line. Where water runs off an edge of the map, its "
             "outline follows that edge. If the map shows no water, answer none.")
PLACE_AT_ASK = ("This picture is a map of a piece of land seen from straight above. Where on it is this part of the "
                "land: {w}? Answer only with the one point as [x, y]: thousandths of the map's width (x, from the left) "
                "and of its height (y, from the top), 0 to 1000. If the map does not show it, answer none.")
PLACE_WET_ASK = ("This picture is a map of a piece of land seen from straight above. Would a person standing at this "
                 "part of the land stand in the water itself: {w}? Answer yes or no.")


def map_desc(land: str, places: dict | None = None, style: str = "") -> str:
    """The map's drawing words: the place's whole land (its own words), the parts of it the plan sends people to,
    the book's way of painting."""
    land = str(land or "").strip().rstrip(".") or "open land"
    ws = list(dict.fromkeys(str(v).strip().rstrip(".") for v in (places or {}).values() if str(v).strip()))
    pl = f" Plainly on it, near its middle: {'; '.join(ws)}." if ws else ""
    st = f" The way it is painted: {style.strip().rstrip('.')}." if str(style or "").strip() else ""
    return MAP_WORDS.format(span=int(round(2 * HALF_M)), land=land, places=pl, style=st)


# ------------------------------------------------------------------ the map's frame
def to_m(u: float, v: float, half: float = HALF_M) -> list:
    """A point of the map in thousandths (u from the left, v from the top) -> [x, y] metres in the set's frame
    (the map's middle at 0, 0; the top of the map at +y)."""
    return [round((float(u) / 1000.0 - 0.5) * 2 * half, 3), round((0.5 - float(v) / 1000.0) * 2 * half, 3)]


def to_uv(x: float, y: float, half: float = HALF_M) -> list:
    """[x, y] metres in the set's frame -> the map's thousandths [u, v] (to_m turned back)."""
    return [(float(x) / (2 * half) + 0.5) * 1000.0, (0.5 - float(y) / (2 * half)) * 1000.0]


def cells(half: float = HALF_M, cell: float = WATER_CELL_M) -> int:
    return int(round(2 * half / cell))


def cell_of(x: float, y: float, half: float = HALF_M, cell: float = WATER_CELL_M) -> tuple:
    """(row, column) of the grid cell holding [x, y]: row 0 along the top of the map (+y), column 0 at its left."""
    return int(math.floor((half - y) / cell)), int(math.floor((x + half) / cell))


def cell_mid(i: int, j: int, half: float = HALF_M, cell: float = WATER_CELL_M) -> list:
    return [-half + (j + 0.5) * cell, half - (i + 0.5) * cell]


# ------------------------------------------------------------------ the judge's answers
def point_of(txt) -> list | None:
    """[u, v] thousandths from the judge's answer, or None (no point, "none", out of the map)."""
    s = str(txt or "")
    if re.search(r"\bnone\b", s, re.I) and not re.search(r"\d", s):
        return None
    v = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", s)[:2]]
    return v if len(v) == 2 and all(0 <= t <= 1000 for t in v) else None


def polys_of(txt) -> list:
    """Each line of the judge's answer that holds three or more [x, y] points -> a polygon of [u, v] thousandths;
    a line of exactly four numbers is taken as a box [x1, y1, x2, y2]."""
    out = []
    for ln in str(txt or "").splitlines():
        v = [min(1000.0, max(0.0, float(x))) for x in re.findall(r"\d+(?:\.\d+)?", ln)]
        if len(v) == 4 and "],[" not in ln.replace(" ", "") and ln.count("[") <= 1:
            x0, y0, x1, y1 = v
            if x1 > x0 and y1 > y0:
                out.append([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
            continue
        pts = [[v[k], v[k + 1]] for k in range(0, len(v) - 1, 2)]
        if len(pts) >= 3:
            out.append(pts)
    return out


def poly_mask(polys: list, n: int):
    """The cells (n x n, row 0 at the top of the map) whose middles lie inside any of the polygons (thousandths;
    even-odd, so a polygon that crosses itself still fills its loops)."""
    import numpy as np
    c = (np.arange(n) + 0.5) / n * 1000.0
    U, V = np.meshgrid(c, c)
    m = np.zeros((n, n), bool)
    for pg in polys:
        p = np.asarray(pg, np.float64)
        inside = np.zeros((n, n), bool)
        for k in range(len(p)):
            (x0, y0), (x1, y1) = p[k], p[(k + 1) % len(p)]
            if y0 == y1:
                continue
            cross = ((y0 > V) != (y1 > V)) & (U < x0 + (V - y0) * (x1 - x0) / (y1 - y0))
            inside ^= cross
        m |= inside
    return m


def _grow(m, k: int):
    """m dilated by k cells (a square reach; the edges do not wrap)."""
    import numpy as np
    out = m.copy()
    for _ in range(k):
        g = out.copy()
        g[1:] |= out[:-1]
        g[:-1] |= out[1:]
        g[:, 1:] |= out[:, :-1]
        g[:, :-1] |= out[:, 1:]
        out = g
    return out


def _shrink(m, k: int):
    return ~_grow(~m, k)


def refine(mask, rgb):
    """The judge's outline corrected by the map's own colours: the outline less a cell of its edge, and every cell
    within WATER_GROW_M of the outline whose colour is near the water's (the middle colour inside the outline).
    rgb: the map picture, any size (resampled to the grid)."""
    import numpy as np
    from PIL import Image
    n = mask.shape[0]
    if not mask.any():
        return mask
    small = np.asarray(Image.fromarray(np.asarray(rgb, np.uint8)).convert("RGB").resize((n, n), Image.BOX), np.float64)
    col = np.median(small[mask], 0)
    near = np.abs(small - col).sum(-1) <= WATER_TOL
    k = max(1, int(round(WATER_GROW_M * n / (2 * HALF_M))))
    return _shrink(mask, 1) | (_grow(mask, k) & near)


def runs_of(m) -> list:
    """A cell mask -> [[row, first column, column after the last], ...] along each row."""
    out = []
    for i, row in enumerate(m):
        j, n = 0, len(row)
        while j < n:
            if row[j]:
                k = j
                while k < n and row[k]:
                    k += 1
                out.append([int(i), int(j), int(k)])
                j = k
            else:
                j += 1
    return out


def mask_of(water: dict | None):
    """The cell mask (numpy bool, row 0 at the top) of a water record ({"cell_m", "half_m", "runs"})."""
    import numpy as np
    w = water or {}
    n = cells(float(w.get("half_m", HALF_M)), float(w.get("cell_m", WATER_CELL_M)))
    m = np.zeros((n, n), bool)
    for i, a, b in w.get("runs") or []:
        m[i, a:b] = True
    return m


def in_water(water: dict | None, x: float, y: float) -> bool:
    """Whether [x, y] (set frame, metres) lies on a water cell. Pure Python: the staging reads it per point."""
    if not water or not water.get("runs"):
        return False
    half, cell = float(water.get("half_m", HALF_M)), float(water.get("cell_m", WATER_CELL_M))
    i, j = cell_of(x, y, half, cell)
    rows = water.get("_rows")
    if rows is None:                                  # (indexed once per record)
        rows = {}
        for r, a, b in water["runs"]:
            rows.setdefault(int(r), []).append((int(a), int(b)))
        water["_rows"] = rows
    return any(a <= j < b for a, b in rows.get(i, ()))


def nearest(water: dict | None, x: float, y: float, wet: bool, most: float = NEAR_MOST_M) -> list | None:
    """The middle of the nearest cell that is water (wet) or dry ground (not wet) to [x, y], within `most` metres;
    the point itself when it already is; None when there is none that near."""
    if bool(in_water(water, x, y)) == wet:
        return [round(x, 3), round(y, 3)]
    half, cell = float((water or {}).get("half_m", HALF_M)), float((water or {}).get("cell_m", WATER_CELL_M))
    i0, j0 = cell_of(x, y, half, cell)
    k = int(math.ceil(most / cell))
    best = None
    for di in range(-k, k + 1):
        for dj in range(-k, k + 1):
            p = cell_mid(i0 + di, j0 + dj, half, cell)
            if abs(p[0]) > half or abs(p[1]) > half or bool(in_water(water, p[0], p[1])) != wet:
                continue
            d = math.dist(p, [x, y])
            if d <= most and (best is None or d < best[0]):
                best = (d, p)
    return [round(best[1][0], 3), round(best[1][1], 3)] if best else None


# ------------------------------------------------------------------ the judge's reading of a map
def map_items(picture: str, places: dict) -> list:
    """The judge's questions about one map: where its water is, and for each land place where it is and whether a
    person there stands in the water."""
    items = [{"id": "water", "image": str(picture), "max_new_tokens": 400, "ask": {"outline": WATER_ASK}}]
    for k, (ph, w) in enumerate((places or {}).items()):
        items.append({"id": f"l:{k}", "image": str(picture), "max_new_tokens": 30,
                      "ask": {"at": PLACE_AT_ASK.format(w=str(w).strip().rstrip(".")),
                              "wet": PLACE_WET_ASK.format(w=str(w).strip().rstrip("."))}})
    return items


def water_of(answer, rgb=None, half: float = HALF_M, cell: float = WATER_CELL_M) -> dict | None:
    """The judge's outline answer (and the map picture, to correct it by colour) -> a water record, or None when it
    shows no water to keep people from (none, a puddle, or the whole map)."""
    n = cells(half, cell)
    m = poly_mask(polys_of(answer), n)
    if rgb is not None:
        m = refine(m, rgb)
    if m.sum() < WATER_MIN_CELLS or m.mean() > WATER_MOST:
        return None
    return {"cell_m": cell, "half_m": half, "runs": runs_of(m), "share": round(float(m.mean()), 4)}


def read_map(picture: str, places: dict, ask, rgb=None) -> dict:
    """The judge's reading of a map -> {"half_m", "water": record or None, "places": {phrase: {"at": [x, y] set
    frame, "wet": bool, "words"}}}. A place the judge says is not in the water but put in it is moved to the dry
    ground by it (a water's edge put a few metres out in the river would stand people in it); one it says is in
    the water is put on the nearest water. A place the map does not show is left out (its people stand on the open
    ground, as before)."""
    a = ask(map_items(picture, places))
    water = water_of((a.get("water") or {}).get("outline"), rgb)
    got = {}
    for k, (ph, w) in enumerate((places or {}).items()):
        r = a.get(f"l:{k}") or {}
        uv = point_of(r.get("at"))
        if not uv:
            continue
        wet = str(r.get("wet") or "").strip().lower().startswith("y") and bool(water)
        x, y = to_m(*uv)
        at = nearest(water, x, y, wet) or (nearest(water, x, y, False, 2 * HALF_M) if not wet else None) or [x, y]
        got[ph] = {"at": at, "wet": bool(wet and in_water(water, *at)), "words": str(w).strip().rstrip(".")}
    if water:
        water.pop("_rows", None)
    return {"half_m": HALF_M, "water": water, "places": got}


# ------------------------------------------------------------------ the ground's picture
def edge_ring(rgb, d, sky, half: float = HALF_M, eye: float = 1.6, bins: int = 360, spread: int = 7):
    """The panorama's own ground just beyond the map's edge, all the way round: for each of `bins` directions the
    colour where the ground plane at `half` metres out is seen in the panorama (H x W x 3, asset frame as
    pano_shell.pano_dirs: the picture's middle ahead at -y), a running median over `spread` bins. Where the picture
    shows no ground there (sky, a thing nearer), the ring's other ground colours. -> bins x 3 float."""
    import numpy as np
    h, w = rgb.shape[:2]
    th = (np.arange(bins) + 0.5) / bins * 2 * np.pi
    X, Y = half * np.cos(th), half * np.sin(th)
    n = np.sqrt(X * X + Y * Y + eye * eye)
    u = (1 - np.arctan2(-X / n, Y / n) / (2 * np.pi)) % 1.0
    v = np.arccos(np.clip(-eye / n, -1, 1)) / np.pi
    px, py = np.clip((u * w).astype(int), 0, w - 1), np.clip((v * h).astype(int), 0, h - 1)
    col = rgb[py, px, :3].astype(np.float64)
    ok = ~sky[py, px] if sky is not None else np.ones(bins, bool)
    if d is not None:
        ok &= np.abs(d[py, px] * (-eye / n) + eye) < 0.12 * half / 3.0
    if not ok.any():
        ok[:] = True
    fill = np.median(col[ok], 0)
    col[~ok] = fill
    out = np.empty_like(col)
    for k in range(bins):
        idx = [(k + t) % bins for t in range(-(spread // 2), spread // 2 + 1)]
        out[k] = np.median(col[idx], 0)
    return out


def blend(map_rgb, ring, fade_m: float, half: float = HALF_M):
    """The ground's picture: the drawn map (row 0 at +y) inside the disk, fading over its outer `fade_m` metres into
    the panorama's own ground colour in each direction (ring: edge_ring), all that colour outside the disk's round
    edge, so the set's ground meets the panorama with no seam. -> uint8 of the map's size."""
    import numpy as np
    m = np.asarray(map_rgb, np.float64)[..., :3]
    n = m.shape[0]
    c = (np.arange(n) + 0.5) / n * 2 * half - half
    X, Yd = np.meshgrid(c, c)
    Y = -Yd                                            # (row 0 is the top of the map: +y)
    r = np.hypot(X, Y)
    th = np.arctan2(Y, X) % (2 * np.pi)
    b = len(ring)
    edge = np.asarray(ring, np.float64)[np.clip((th / (2 * np.pi) * b).astype(int), 0, b - 1)]
    wt = np.clip((half - r) / max(fade_m, 1e-6), 0.0, 1.0)[..., None]
    return (m * wt + edge * (1 - wt)).clip(0, 255).astype(np.uint8)
