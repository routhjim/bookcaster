"""Engine still -> image-model redraw: job files for the two image models, with the words built from the scene.

    python tools/assets/redraw/redraw.py RUN.json        (stdlib; writes the job files, prints their paths)

RUN: {"scenes": ["/abs/scene_out", ...]   scene tool outputs with truth.json (tools/assets/redraw/guides.py),
      "out": "/abs/folder",
      "method": "edit" | "image",
      "sheet": "/abs/character sheet.png",    identity picture (both models)
      "lora": "/abs/char.safetensors", "trigger": "chr_x", "lora_w": {"1": 0.8, "2": 0.63},   (method image)
      "wear": "what the character wears",     (words; the scene tool does not know clothes yet)
      "wear_back": "what shows from behind",  (method image: said instead of "wear" for a person turned away)
      "set_words": "what the room holds",     (the set's library description)
      "strengths": [0.55, 0.7],               (method image: img2img strength of the engine still)
      "control": {"type": "depth", "scale": 0.8, "end": 1.0} | null,    (method image: the engine's own guide)
      "controls": [{"type": "depth"|"pose"|"canny", ...}, ...]          (several: depth, the bones' OpenPose
                                              picture, the outlines)
      "body_words": false,                    (say each body's orientation apart from the head)
      "mask_people": false | true | "front",  (method image: the still's people masks as the per-pixel strength,
                                               "strength_bg" for the room; "front": only people whose body is
                                               within 100 deg of the camera, the others at "strength_bg")
      "set_facts": true,                      (name the set's things the frame holds, lit ones as lit: guides.py --set)
      "room_all": false,                      (the room's contents for every shot; default: wide shots only)
      "extra_refs": ["/abs/anchor.png", ...], (method image: more reference pictures after the sheet, e.g. the
                                               scene's first accepted redraw, so later shots draw the same room)
      "seed": 7, "only": ["scene/cam", ...]}
Outputs: method edit -> edit_jobs.json for tools/assets/qwen_edit.py; method image -> image_jobs.jsonl for
tools/qwen_image.py --serve (one JSON line per picture). Each job id is "<scene>__<cam>[__s<strength>]".

The words follow general rules and the scene's own facts only: the camera distance (the still's camera kind), each
person's place in frame (their engine mask's box) and which way they face relative to the camera (the angle the
scene measured between the head's direction and the camera), what they wear, what the room holds. A person turned
away is said to be seen from behind with no face visible.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

SHOT = {
    "wide": "A wide shot: the whole room in view, the people small and full length, feet on the floor.",
    "medium": "A medium shot: the people seen from about the waist up, the room behind them.",
    "close": "A close shot: head and shoulders filling the frame, the room behind soft and out of focus.",
}


def facing_words(deg: float) -> str:
    """How a person is turned relative to the camera, from the scene's measured angle (0 = facing the camera)."""
    if deg < 25:
        return "faces the camera; his face is seen from the front"
    if deg < 60:
        return "is turned a little to one side, a three-quarter view; his face is visible"
    if deg < 115:
        return "is seen in profile, side-on to the camera"
    if deg < 150:
        return ("is turned away from the camera, seen three-quarters from behind: we see the back and side of his "
                "head, his ear and the edge of his cheek, and the back of his coat; his face is not visible")
    return ("is seen from directly behind, facing away from the camera: we see the back of his head, his hair and "
            "the back of his coat and shoulders; his face is not visible")


def place_words(box) -> str:
    cx = (box[0] + box[2]) / 2
    side = "on the left" if cx < 0.38 else "on the right" if cx > 0.62 else "in the centre"
    near = "in the foreground, close to the camera" if box[3] - box[1] > 0.95 and box[2] - box[0] > 0.3 else ""
    return side + (", " + near if near else "")


def people(shot: dict, poses: dict | None = None) -> list:
    """[(key, box, angle, posture)] of the people visible in the still, left to right; posture "sitting" when the
    scene posed them seated (pose name), else "standing"."""
    poses = poses or {}
    out = [(k, v["box"], v.get("head_to_camera_deg", 0.0),
            "sitting" if any(w in (poses.get(k) or "") for w in ("sit", "seat")) else "standing",
            v.get("body_to_camera_deg", v.get("head_to_camera_deg", 0.0)), v.get("nearer_shoulder"))
           for k, v in shot["people"].items() if v.get("box") and v.get("area", 0) > 0.002]
    return sorted(out, key=lambda x: (x[1][0] + x[1][2]) / 2)


def kind_of(cam: str, shot: dict) -> str:
    for k in ("close", "medium", "wide"):
        if cam.startswith(k):
            return k
    return "wide" if shot.get("kind") in ("wide", "full") else shot.get("kind", "medium")


def body_words(deg: float, near: str | None, who: str = "His") -> str:
    """The body's own orientation, apart from the head (the scene's hips-to-camera angle and nearer shoulder): a
    redraw kept a turned-away head but swung the body to profile or front when only the head was described
    (11-11, three-quarter-away shots)."""
    sh = f", his {near} shoulder nearer the camera" if near else ""
    if deg < 60:
        return f"{who} body faces the camera."
    if deg <= 100:
        return f"{who} body is side-on to the camera{sh}."
    return (f"{who} body is turned away from the camera: we see his back{sh}; his chest and the front of his coat are "
            f"not visible, and the back of his coat, with its centre seam and the vent between the tails, faces the "
            f"camera.")


def people_words(ppl: list, with_body: bool = False) -> str:
    if not ppl:
        return ""
    n = len(ppl)
    head = "One man is in the picture." if n == 1 else f"{n} men are in the picture, dressed alike."
    parts = [head]
    for k, box, ang, post, bdeg, near in ppl:
        who = "The man" if n == 1 else f"The man {place_words(box).split(',')[0]}"
        parts.append(f"{who} is {post} {place_words(box)} and {facing_words(ang)}." if n == 1 else
                     f"{who} is {post} and {facing_words(ang)}.")
        if with_body:
            parts.append(body_words(bdeg, near))
    return " ".join(parts)


def any_away(ppl: list) -> bool:
    return any(p[2] >= 115 for p in ppl)


def edit_prompt(kind: str, ppl: list, wear: str, room: str, room_all: bool = False, things: str = "") -> str:
    away = any_away(ppl)
    ident = ("Picture 2 shows who he is: his face, his hair and his clothes"
             + (" (use it for his hair, build and clothes; where he is turned away, keep him turned away exactly "
                "as in Picture 1 and do not show his face)" if away else "") + f": {wear}.")
    return " ".join([
        "Picture 1 is a frame rendered by a game engine. Keep everything about it: the same camera position, lens, "
        "framing and perspective; every person in the same place, at the same size, in the same pose and turned the "
        "same way as in Picture 1; the room's layout unchanged, every wall, window, door, fireplace and piece of "
        "furniture where it is, nothing added, nothing removed, nothing covered.",
        "Change only how it looks: make it a photographed frame from a period film, with real skin and hair, woven "
        "wool and linen with natural folds, real wood, plaster, fabric and brass, the light coming from the same "
        "directions, film grain.",
        SHOT[kind], people_words(ppl), ident,
        room_words(kind, room, room_all) if not things or kind != "wide" else f"{room.split(':')[0]}.", things])


def frame_place(box) -> str:
    cx = (box[0] + box[2]) / 2
    return "on the left" if cx < 0.35 else "on the right" if cx > 0.65 else "in the middle"


def set_words(shot: dict) -> str:
    """The set's things the engine sees in this frame (guides.py --set), with their place and whether they burn or
    are lit, plus where the daylight comes from. Nothing that is out of frame is named, except the daylight."""
    th = [t for t in shot.get("things") or [] if t["visible_frac"] >= 0.05 and t["kind"] != "prop" or
          (t.get("lit") and t["visible_frac"] >= 0.02)]
    th = [t for t in th if t["name"] != "rug" or t["visible_frac"] >= 0.2]
    parts = []
    for t in sorted(th, key=lambda t: (t["box"][0] + t["box"][2]) / 2):
        w = t["words"] if t.get("flame_in_view", True) else t["words"].split(" with ")[0].split(",")[0]
        if t["kind"] == "window" and shot.get("daylight"):
            w += ", bright with " + shot["daylight"] + " outside"
        parts.append(f"{w} {frame_place(t['box'])}")
    out = ("In the frame: " + "; ".join(parts) + ".") if parts else ""
    if shot.get("daylight") and not any(t["kind"] == "window" for t in th):
        out += f" Soft {shot['daylight']} comes from a window out of frame."
    lit = [t for t in th if (t.get("fire") or t.get("lit")) and t.get("flame_in_view", True)]
    if lit:
        out += " " + " ".join(
            (f"The fire in {t['words'].split(' with ')[0]} burns brightly with real flames, as in Picture 1."
             if t.get("fire") else f"{t['words'].split(',')[0].capitalize()} is lit, its flame burning, as in Picture 1.")
            for t in lit)
    return out.strip()


def room_words(kind: str, room: str, room_all: bool = False) -> str:
    """The room's contents only for a wide shot: given the list for a closer shot, the edit model widened the
    view to show it all (11-11 run a: every medium and close redraw pulled back and rebuilt the room)."""
    if kind == "wide" or room_all:
        return f"The room: {room}"
    return ("Behind him keep exactly the part of the room that Picture 1 shows, at the same scale and as out of "
            "focus as it is there; do not widen the view, do not show more of the room, add nothing to the walls.")


def wear_words(ppl: list, wear: str, wear_back: str | None) -> str:
    """Clothes for the people who show their front; a person turned away (>= 115 deg) is given only what shows
    from behind (wear_back), when the run has it: with the full list a strong redraw put the cravat and waistcoat
    on the back of a man seen from behind (11-11, strengths 0.8-0.95)."""
    if not wear_back:
        return f"He wears {wear}." if len(ppl) <= 1 else f"Each wears {wear}."
    out = []
    for k, box, ang, post, bdeg, near in ppl:
        who = "He" if len(ppl) == 1 else f"The man {place_words(box).split(',')[0]}"
        out.append(f"{who} is seen from behind: we see only {wear_back}." if max(ang, bdeg) > 100 else f"{who} wears {wear}.")
    return " ".join(out)


def image_prompt(kind: str, ppl: list, wear: str, room: str, trigger: str, things: str = "",
                 wear_back: str | None = None, with_body: bool = False) -> str:
    return " ".join([
        f"A photographed frame from a period film. {SHOT[kind]}", people_words(ppl, with_body).replace(
            "One man", f"One man, {trigger},").replace("men are in the picture", f"men, each {trigger}, are in the picture"),
        wear_words(ppl, wear, wear_back),
        (f"The room: {room}" if not things else f"{room.split(':')[0]}. {things.replace(', as in Picture 1', '')}"),
        "Real skin and hair, woven cloth with natural folds, natural light, film grain."])


def main():
    run = json.loads(Path(sys.argv[1]).read_text())
    out = Path(run["out"])
    out.mkdir(parents=True, exist_ok=True)
    seed = int(run.get("seed", 7))
    only = set(run.get("only") or [])
    jobs = []
    for sd in map(Path, run["scenes"]):
        truth = json.loads((sd / "truth.json").read_text())
        res = json.loads((sd / "result.json").read_text())
        room = run["set_words"]
        for cam, sh in truth["shots"].items():
            sid = f"{sd.name}/{cam}"
            if only and sid not in only:
                continue
            kind = kind_of(cam, sh)
            poses = {k: ((v.get("body_pose") or {}).get("pose") or "") for k, v in res.get("subjects", {}).items()}
            ppl = people(sh, poses)
            base = {"scene": sd.name, "cam": cam, "kind": kind, "still": sh["still"],
                    "people": [{"key": k, "box": b, "head_to_camera_deg": a, "posture": q, "body_to_camera_deg": bd,
                                "nearer_shoulder": ns} for k, b, a, q, bd, ns in ppl]}
            if run["method"] == "edit":
                jobs.append({"id": f"{sd.name}__{cam}", "prompt": edit_prompt(kind, ppl, run["wear"], room, run.get("room_all", False),
                                                     set_words(sh) if run.get("set_facts") else "", run.get("wear_back"),
                                                run.get("body_words", False)),
                             "images": [sh["still"], run["sheet"]], "seed": seed,
                             "path": str(out / f"{sd.name}__{cam}.png"), "size": [1152, 640], "_meta": base})
            else:
                g = sd / "guides"
                n = max(1, len(ppl))
                lw = float((run.get("lora_w") or {}).get(str(min(n, 2)), 0.8 if n == 1 else 0.63))
                for st in run.get("strengths", [0.7]):
                    j = {"id": f"{sd.name}__{cam}__s{int(round(st * 100))}",
                         "prompt": image_prompt(kind, ppl, run["wear"], room, run["trigger"],
                                                set_words(sh) if run.get("set_facts") else "", run.get("wear_back"),
                                                run.get("body_words", False)),
                         "seed": seed, "path": str(out / f"{sd.name}__{cam}__s{int(round(st * 100))}.png"),
                         "size": [1152, 640], "refs": [run["sheet"]] + list(run.get("extra_refs") or []),
                         "loras": [[run["lora"], lw, [0.0, 0.0, 1.0, 1.0]]],
                         "init": {"image": sh["still"], "strength": st}, "_meta": dict(base, strength=st)}
                    cl = run.get("controls") or ([run["control"]] if run.get("control") else [])
                    pics = {"depth": "depth", "pose": "pose", "canny": "lines"}
                    cj = [{"image": str(g / f"{cam}_{pics[c['type']]}.png"), "type": c["type"],
                           "scale": c.get("scale", 0.8), "end": c.get("end", 1.0), "prepared": True}
                          for c in cl if (g / f"{cam}_{pics[c['type']]}.png").exists()]
                    if cj:
                        j["control"] = cj[0] if len(cj) == 1 else cj
                    if run.get("mask_people") and ppl:      # people redrawn harder than the room
                        mk = run.get("mask_people") if isinstance(run.get("mask_people"), str) else "people"
                        j["init"]["mask"] = str(g / f"{cam}_mask_{mk}.png")
                        j["init"]["strength_bg"] = run.get("strength_bg", 0.6)
                    jobs.append(j)
    if run["method"] == "edit":
        p = out / "edit_jobs.json"
        p.write_text(json.dumps(jobs, indent=1))
    else:
        p = out / "image_jobs.jsonl"
        p.write_text("".join(json.dumps(j) + "\n" for j in jobs))
    (out / "meta.json").write_text(json.dumps({j["id"]: dict(j["_meta"], prompt=j["prompt"]) for j in jobs}, indent=1))
    print(p, len(jobs))


if __name__ == "__main__":
    main()
