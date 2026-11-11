"""Engine still -> image-model redraw: job files for the two image models, with the words built from the scene.

    python tools/assets/redraw/redraw.py RUN.json        (stdlib; writes the job files, prints their paths)

RUN: {"scenes": ["/abs/scene_out", ...]   scene tool outputs with truth.json (tools/assets/redraw/guides.py),
      "out": "/abs/folder",
      "method": "edit" | "image",
      "sheet": "/abs/character sheet.png",    identity picture (both models)
      "lora": "/abs/char.safetensors", "trigger": "chr_x", "lora_w": {"1": 0.8, "2": 0.63},   (method image)
      "wear": "what the character wears",     (words; the scene tool does not know clothes yet)
      "set_words": "what the room holds",     (the set's library description)
      "strengths": [0.55, 0.7],               (method image: img2img strength of the engine still)
      "control": {"type": "depth", "scale": 0.8, "end": 1.0} | null,    (method image: the engine's own guide)
      "mask_people": false,                   (method image: the still's people masks as the per-pixel strength,
                                               "strength_bg" for the room)
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


def people(shot: dict) -> list:
    """[(key, box, angle)] of the people visible in the still, left to right."""
    out = [(k, v["box"], v.get("head_to_camera_deg", 0.0)) for k, v in shot["people"].items() if v.get("box")
           and v.get("area", 0) > 0.002]
    return sorted(out, key=lambda x: (x[1][0] + x[1][2]) / 2)


def kind_of(cam: str, shot: dict) -> str:
    for k in ("close", "medium", "wide"):
        if cam.startswith(k):
            return k
    return "wide" if shot.get("kind") in ("wide", "full") else shot.get("kind", "medium")


def people_words(ppl: list) -> str:
    if not ppl:
        return ""
    n = len(ppl)
    head = "One man is in the picture." if n == 1 else f"{n} men are in the picture, dressed alike."
    parts = [head]
    for k, box, ang in ppl:
        who = "The man" if n == 1 else f"The man {place_words(box).split(',')[0]}"
        parts.append(f"{who} is {place_words(box)} and {facing_words(ang)}." if n == 1 else
                     f"{who} {facing_words(ang)}.")
    return " ".join(parts)


def any_away(ppl: list) -> bool:
    return any(a >= 115 for _, _, a in ppl)


def edit_prompt(kind: str, ppl: list, wear: str, room: str) -> str:
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
        SHOT[kind], people_words(ppl), ident, f"The room: {room}"])


def image_prompt(kind: str, ppl: list, wear: str, room: str, trigger: str) -> str:
    return " ".join([
        f"A photographed frame from a period film. {SHOT[kind]}", people_words(ppl).replace(
            "One man", f"One man, {trigger},").replace("men are in the picture", f"men, each {trigger}, are in the picture"),
        f"He wears {wear}." if len(ppl) <= 1 else f"Each wears {wear}.",
        f"The room: {room}", "Real skin and hair, woven cloth with natural folds, natural light, film grain."])


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
            ppl = people(sh)
            base = {"scene": sd.name, "cam": cam, "kind": kind, "still": sh["still"],
                    "people": [{"key": k, "box": b, "head_to_camera_deg": a} for k, b, a in ppl]}
            if run["method"] == "edit":
                jobs.append({"id": f"{sd.name}__{cam}", "prompt": edit_prompt(kind, ppl, run["wear"], room),
                             "images": [sh["still"], run["sheet"]], "seed": seed,
                             "path": str(out / f"{sd.name}__{cam}.png"), "size": [1152, 640], "_meta": base})
            else:
                g = sd / "guides"
                n = max(1, len(ppl))
                lw = float((run.get("lora_w") or {}).get(str(min(n, 2)), 0.8 if n == 1 else 0.63))
                for st in run.get("strengths", [0.7]):
                    j = {"id": f"{sd.name}__{cam}__s{int(round(st * 100))}",
                         "prompt": image_prompt(kind, ppl, run["wear"], room, run["trigger"]),
                         "seed": seed, "path": str(out / f"{sd.name}__{cam}__s{int(round(st * 100))}.png"),
                         "size": [1152, 640], "refs": [run["sheet"]],
                         "loras": [[run["lora"], lw, [0.0, 0.0, 1.0, 1.0]]],
                         "init": {"image": sh["still"], "strength": st}, "_meta": dict(base, strength=st)}
                    c = run.get("control")
                    if c:
                        img = g / f"{cam}_{'depth' if c['type'] == 'depth' else 'lines'}.png"
                        j["control"] = {"image": str(img), "type": c["type"], "scale": c.get("scale", 0.8),
                                        "end": c.get("end", 1.0), "prepared": True}
                    if run.get("mask_people") and ppl:
                        j["init"]["mask"] = str(out / "masks" / f"{sd.name}__{cam}.png")
                        j["init"]["strength_bg"] = run.get("strength_bg", 0.5)
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
