"""Lying: a person lying in a bed is placed from the bed's own lie box (stdlib).

A character in a scene spec with "lying_from": "<base id>" and "bed": "<set item name>" (or true: the set's bed
nearest its "position") becomes, before staging, the library's lying stance of that person settled on that bed's
asset ("<base>_lying_<bed asset>", or its "_propped" form with "propped": true): its own recorded pose, the head on
the bed's pillow, the body along the bed. The stance's record gives where its head lies in its own frame ("lie"
"head_m"); the bed's library record gives the lie box (its "supports" entry with posture "lying",
characters/liebox.py). Nothing is solved here: the stance was settled on this bed's mesh when it was built.
"""
from __future__ import annotations

import json
import math
from pathlib import Path


def _rot(v, deg):
    a = math.radians(deg)
    return [v[0] * math.cos(a) - v[1] * math.sin(a), v[0] * math.sin(a) + v[1] * math.cos(a)]


def _ang(v):
    return math.degrees(math.atan2(v[1], v[0]))


def frame(pos, front, sup: dict) -> dict:
    """The bed's lie box in the set: the pillow's centre, the foot end's middle and the lying person's facing
    (head to foot); pos: where the bed stands, front: its foot direction (the asset's front turned into the set)."""
    f = list(front)
    ax, ay = [-f[1], f[0]], [-f[0], -f[1]]                            # asset x and y in the set
    to = lambda p: [pos[0] + p[0] * ax[0] + p[1] * ay[0], pos[1] + p[0] * ax[1] + p[1] * ay[1]]  # noqa: E731
    xs, ys = sup["box"]["x_m"], sup["box"]["y_m"]
    sgn = 1 if sup.get("head", "+y") == "+y" else -1
    py = sup.get("pillow_y_m") or ([ys[1] - 0.35, ys[1]] if sgn > 0 else [ys[0], ys[0] + 0.35])
    xm = (xs[0] + xs[1]) / 2
    return {"head_xy": to([xm, (py[0] + py[1]) / 2]), "foot_xy": to([xm, ys[0] if sgn > 0 else ys[1]]),
            "centre_xy": to([xm, (ys[0] + ys[1]) / 2]), "facing": f if sgn > 0 else [-f[0], -f[1]]}


def beds(library: Path, set_dir: Path) -> dict:
    """{item name: {"pos", "asset", "sup": the lying support, "front": [dx, dy] (world), "frame"}} for the set's
    items whose library record has a support with posture "lying"."""
    spec = json.loads((set_dir / "spec.json").read_text())
    built = {}                                       # the yaw each item was really placed with (the set's scene.json)
    if (set_dir / "scene.json").exists():
        built = {a["name"]: a["yaw"] for a in json.loads((set_dir / "scene.json").read_text()).get("assets", [])}
    out = {}
    for kind in ("fixtures", "props"):
        for f in spec.get(kind, []):
            if f.get("outside") or "row" in f or not f.get("asset"):
                continue
            for cls in ("prop", "set"):
                p = library / cls / f["asset"] / "asset.json"
                if not p.exists():
                    continue
                sup = next((x for x in json.loads(p.read_text()).get("supports") or []
                            if x.get("posture") == "lying"), None)
                if sup:
                    yaw = -built[f["name"]] if f["name"] in built else _ang(f.get("facing", [0, -1])) + 90.0
                    fa = sup.get("front_asset", [0.0, -1.0])
                    front = _rot(fa, yaw)
                    out[f["name"]] = {"name": f["name"], "pos": f["position"][:2], "asset": f["asset"], "sup": sup,
                                      "front": front, "frame": frame(f["position"][:2], front, sup)}
                break
    return out


def stance(library: Path, base: str, bed: dict, propped: bool = False) -> str | None:
    """The library character settled lying on this bed's asset: "<base>_lying_<asset>" (propped: its "_propped"
    form, else the flat one when only that exists)."""
    want = [f"{base}_lying_{bed['asset']}" + ("_propped" if propped else "")]
    if propped:
        want.append(f"{base}_lying_{bed['asset']}")
    return next((w for w in want if (library / "character" / w / "character.json").exists()), None)


LIE_W_M = 0.55            # the width one lying person takes


def place(c: dict, bed: dict, cj: dict, across: float = 0.0) -> dict:
    """The character spec as the lying stance on the bed: its position so that the stance's head lies on the
    pillow's centre, facing head to foot, the stance's own recorded pose."""
    fr = bed["frame"]
    f_ = fr["facing"]
    lft = [-f_[1], f_[0]]
    hx, hy = (cj.get("lie") or {}).get("head_m", [0.0, 0.0, 0.0])[:2]
    out = {k: v for k, v in c.items() if k not in ("lying_from", "bed", "propped", "marks", "facing_to")}
    out["id"] = cj["id"]
    out["position"] = [round(fr["head_xy"][0] - (hx - across) * lft[0] - hy * f_[0], 3),
                       round(fr["head_xy"][1] - (hx - across) * lft[1] - hy * f_[1], 3)]
    out["facing"] = [round(f_[0], 4), round(f_[1], 4)]
    out["body_pose"] = json.loads(json.dumps(cj["body_pose"]))
    sh_ = [across * lft[0], across * lft[1]]               # (across: this person's place across a shared bed)
    out["_lie"] = {"bed": bed["name"], "head_xy": [round(fr["head_xy"][i] + sh_[i], 3) for i in (0, 1)],
                   "foot_xy": [round(fr["foot_xy"][i] + sh_[i], 3) for i in (0, 1)],
                   "top_z_m": bed["sup"]["box"]["top_z_m"]}
    return out


CLOSE = 0.8               # people lying close give up a fifth of their width each


def plan(spec: dict, library: Path) -> dict:
    """Lay every character that names a bed ("lying_from" and "bed"). A bed takes those who fit across its lie box:
    LIE_W_M each (a child their height's share of it), a fifth less when they lie close, laid side by side across
    it in the plan's order. Returns {key: bed name}; edits spec."""
    B = beds(library, library / "set" / spec["set"])
    wid = {n: b["sup"]["box"]["x_m"][1] - b["sup"]["box"]["x_m"][0] for n, b in B.items()}
    need = lambda k: LIE_W_M * float(spec["characters"][k].get("scale") or 1.0)        # noqa: E731
    taken, cjs = {}, {}
    for k, c in list(spec["characters"].items()):
        if not c.get("lying_from"):
            continue
        order = sorted(B, key=lambda n: (n != c.get("bed"), math.dist(B[n]["pos"], (c.get("position") or [0, 0])[:2])))
        for n in order:
            there = [q for q in taken if taken[q] == n]
            if there and (sum(map(need, there)) + need(k)) * CLOSE > wid[n] + 0.05:
                continue
            vid = stance(library, c["lying_from"], B[n], bool(c.get("propped")))
            if not vid:
                continue
            cjs[k] = json.loads((library / "character" / vid / "character.json").read_text())
            taken[k] = n
            break
        else:
            raise SystemExit(f"{k}: no bed in the set has room and a lying stance of {c['lying_from']}")
    for n in set(taken.values()):
        ks = [k for k in taken if taken[k] == n]
        ws = [need(k) for k in ks]
        f = min(1.0, wid[n] / sum(ws))                       # (closer together when they are more than fit apart)
        x = -sum(ws) * f / 2
        for k, w in zip(ks, ws):
            spec["characters"][k] = place(spec["characters"][k], B[n], cjs[k], x + w * f / 2)
            x += w * f
    return {"taken": taken, "beds": B}
