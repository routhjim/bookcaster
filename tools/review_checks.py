"""bookcaster's REVIEW as code: every check the user made by eye on 10-20, as a named, calibrated check (any book).

    python3 tools/review_checks.py list                         the registry
    python3 tools/review_checks.py calibrate [--json out.json]  run every check on its fixtures: the agreement table
    python3 tools/review_checks.py run <book> --chapter N        verdicts for a chapter's keys/clips/staging/assets
                                                                 -> stage/approvals.json "checks" (flow3d.review)
A check = (stage: asset | staging | key | clip, detector: numeric / Qwen3-VL rubric / a tool's own QA output,
threshold, fix). Fixtures: tools/tests/fixtures/review/<check>/{fail,pass}_*.(png|mp4) + case.json (the context the
detector needs: who is in frame, their looks, the set, the speaker...) with the user's verdict. `calibrate` must flag
every "fail" fixture and pass every "pass" one; a disagreement is a rubric to fix.
Detectors reuse the shared code (tools/remake_chapter.py: the VL judge batch, the outfit-word match; tools/video_qa.py
for clips) - one implementation for the flow and for the review.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
FIX = HERE / "tests" / "fixtures" / "review"
POSE_PY = Path.home() / "tts-ab" / ".venv-pose" / "bin" / "python"

# ------------------------------------------------------------------------------------------------- the rubric
# VL questions: (question template, the answer that PASSES, crop: "person" | "frame")
VL = {
    "key.location_interior": ("Is this scene indoors, inside a room or building? Answer yes or no.", "yes", "frame"),
    "key.no_sea_sky_interior": ("Is the sea or an open sky visible in this picture? Answer yes or no.", "no", "frame"),
    "key.set_present": ("Does this picture show {set_desc}? Answer yes or no.", "yes", "frame"),
    "key.no_sky_between_people": ("Is there a patch of sky, clouds or sea visible between or around the people, "
                                  "although they are inside a room? Answer yes or no.", "no", "frame"),
    "key.clean_background": ("Does the background look like a clean painted scene, with no smeared, melted, noisy or "
                             "garish gravel-like texture? Answer yes or no.", "yes", "frame"),
    # calibrated 10-20: with "or stubble" the clean-shaven reference sheet itself read "yes" (shading)
    "key.facial_hair": ("Does this person have a full beard or a moustache (ignore light stubble and shadows)? "
                        "Answer yes or no.", "{beard}", "person"),
    "key.back_hair": ("Look at this person's hair. Is it {hair}? Answer yes or no.", "yes", "person"),
    "key.no_mask_face": ("Does the back or side of this person's head show a face-like, mottled or mask-like pattern? "
                         "Answer yes or no.", "no", "person"),
    "key.story_prop": ("Is there {prop} visible in this picture? Answer yes or no.", "yes", "frame"),
    "key.no_invented_prop": ("Is there {prop} visible in this picture? Answer yes or no.", "no", "frame"),
    "clip.no_text": ("Is there any written text, letters, a sign, a tablet or a caption visible in this picture? "
                     "Answer yes or no.", "no", "frame"),
}
FREE = {   # free-text VL questions judged in code
    "key.outfit": ("Describe this person's clothes in a few words: garment, material and colour.", "person"),
    "key.age_band": ("How old does this person look? Answer with one number.", "person"),
}

CHECKS = [
    # name, stage, detector, threshold / rule, fix
    ("key.location_interior", "key", "vl", "interior set -> indoors", "redraw: place words + bg-hold"),
    ("key.no_sea_sky_interior", "key", "vl", "interior set -> no sea/sky", "redraw: interior-safe style + bg-hold"),
    ("key.set_present", "key", "vl", "the set's description visible", "bg-hold (paint over the set render)"),
    ("key.no_sky_between_people", "key", "vl", "no sky through the people mask", "bg-hold with silhouettes"),
    ("key.clean_background", "key", "vl", "no smeared set texture", "bg-hold at 0.5 with depth+edges"),
    ("key.outfit", "key", "vl-free", "garment+material(+colour) words of the wardrobe", "redraw: outfit stated + LoRA"),
    ("key.facial_hair", "key", "vl", "beard yes/no as look.json", "redraw: 'clean-shaven: no beard'"),
    ("key.age_band", "key", "vl-free", "|read-age| <= 3 (<25) / 8", "redraw: age stated"),
    ("key.back_hair", "key", "vl", "hair as look.json (anyone)", "redraw: hair stated"),
    ("key.no_mask_face", "key", "vl", "back-turned head: no face pattern", "redraw: 'only hair, no face'"),
    ("key.speaker_face_size", "key", "numeric", "speaker face box >= 15 % of frame height", "reframe (scene layer)"),
    ("key.speaker_face_light", "key", "numeric", "speaker face mean luma >= 45", "brighter key light / reframe"),
    ("key.speaker_visible", "key", "numeric", "the speaker's face is detected", "reframe (occluded)"),
    ("key.story_prop", "key", "vl", "every props_state prop visible", "redraw with the prop stated / prop mesh"),
    ("key.no_invented_prop", "key", "vl", "library props NOT in props_state absent", "redraw: drop the prop words"),
    ("clip.no_text", "clip", "vl", "no text on 4 sampled frames", "re-roll (no words in prompts, text negative)"),
    ("clip.no_placeholder", "clip", "numeric", "no flat untextured box-like block (placeholder_check.py)",
     "an approved prop mesh / the scene-layer shot (10-21 forge: raw white anvil boxes)"),
    ("clip.held_props", "clip", "vl", "every staged hand-held prop visible in the hand (45 % frame)", "re-roll"),
    ("clip.strike_contact", "clip", "vl", "tool head touches the work at the contact frame (+ sparks if staged)",
     "re-roll; staging recipe (strike_on_support)"),
    ("clip.speaker_present", "clip", "video_qa", "speaker face in >= 70 % of sampled frames", "re-roll"),
    ("clip.chunk_lips", "clip", "video_qa", "chunk k mouth >= 50 % of chunk 0", "re-roll (chain 0.7, no end lock)"),
    ("clip.lip_sync", "clip", "video_qa", "voiced/pause mouth activity >= 1.25", "re-roll"),
    ("clip.listener_quiet", "clip", "video_qa", "listener mouth <= 0.8 x speaker", "re-roll"),
    ("clip.join_once", "clip", "numeric", "stitched frames == start(last) + n(last)", "stitch_talk (test_stitch.py)"),
    ("clip.camera", "clip", "video_qa", "camera vs camera.path / static", "re-roll"),
    ("staging.contact_in_frame", "staging", "tool", "the blockout shot's contact point inside the frame", "camera/stage"),
    ("staging.arch_on_wall", "staging", "tool", "architectural props anchored to a wall", "world_locate --anchors"),
    ("staging.ots_speaker_size", "staging", "tool", "OTS speaker face >= 15 % height", "lens/camera"),
    ("staging.no_streaks", "staging", "tool", "no set streaks near the camera", "camera / set mesh"),
    ("staging.no_prop_teleport", "staging", "tool", "a prop's position continuous within a scene", "props_state"),
    ("staging.wardrobe_timing", "staging", "tool", "outfit in force at the moment's time", "characters.model_at"),
    ("asset.creature_watertight", "asset", "tool", "watertight, no thin-shell flaps, jaw skin band", "creature_clean"),
    ("asset.creature_head", "asset", "tool", "the creature's head vs its head-only ref", "creature sheet"),
    ("asset.texture_srgb", "asset", "tool", "textures not washed out (sRGB)", "texture projection fix"),
    ("asset.foot_pitch", "asset", "tool", "toes not up, sole flat", "foot pitch + sole-flat"),
    ("asset.prop_rest", "asset", "tool", "a prop rests on its flat face", "rest-orientation rule"),
    ("asset.prop_scale", "asset", "tool", "prop size vs its affordance", "props_bl scale"),
    ("asset.prop_ref", "asset", "tool", "prop ref: the right object, no background junk", "prop ref vision check"),
    ("asset.headshot_ref", "asset", "tool", "a headshot ref shows only the head", "face_likeness --headshot-out"),
]
NAMES = {c[0] for c in CHECKS}


def _vl(items):
    import remake_chapter as R
    return R.vl_batch(items)


def _crop(box, size):
    W, H = size
    return [max(0, box[0] / W), max(0, box[1] / H), min(1, box[2] / W), min(1, box[3] / H)]


def check_key(img: str, ctx: dict, only: set | None = None) -> dict:
    """Run every key check that applies to a drawn keyframe. ctx: {size, kind (interior/exterior), set_desc,
    people: {name: {box, look, outfit, hair, age, beard: yes|no|None, behind: bool}}, speaker, speaker_face: {h_frac,
    luma} | None, props: [...], absent_props: [...]} -> {check: {pass, why}}"""
    import remake_chapter as R
    want = lambda n: only is None or n in only  # noqa: E731
    items, keys, out = [], [], {}
    size = ctx.get("size") or [832, 448]

    def ask(name, q, crop=None, mnt=None, tag=""):
        items.append({"id": f"{name}|{tag}", "image": img, "ask": {"a": q}, **({"crop": crop} if crop else {}),
                      **({"max_new_tokens": mnt} if mnt else {})})
        keys.append((name, tag))
    if ctx.get("kind") == "interior":
        for n in ("key.location_interior", "key.no_sea_sky_interior", "key.no_sky_between_people"):
            if want(n):
                ask(n, VL[n][0])
    if ctx.get("set_desc") and want("key.set_present"):
        ask("key.set_present", VL["key.set_present"][0].format(set_desc=ctx["set_desc"]))
    if want("key.clean_background"):
        ask("key.clean_background", VL["key.clean_background"][0])
    for w, p in (ctx.get("people") or {}).items():
        crop = _crop(p["box"], size) if p.get("box") else None
        if p.get("outfit") and want("key.outfit"):
            ask("key.outfit", FREE["key.outfit"][0], crop, 24, w)
        if p.get("hair") and want("key.back_hair"):
            ask("key.back_hair", VL["key.back_hair"][0].format(hair=p["hair"]), crop, None, w)
        if p.get("behind"):
            if want("key.no_mask_face"):
                ask("key.no_mask_face", VL["key.no_mask_face"][0], crop, None, w)
            continue
        if p.get("beard") in ("yes", "no") and want("key.facial_hair"):
            ask("key.facial_hair", VL["key.facial_hair"][0], crop, None, w)
        if p.get("age") and want("key.age_band"):
            ask("key.age_band", FREE["key.age_band"][0], crop, 4, w)
    for pr in ctx.get("props") or []:
        if want("key.story_prop"):
            ask("key.story_prop", VL["key.story_prop"][0].format(prop=pr), None, None, pr)
    for pr in ctx.get("absent_props") or []:
        if want("key.no_invented_prop"):
            ask("key.no_invented_prop", VL["key.no_invented_prop"][0].format(prop=pr), None, None, pr)
    ans = _vl(items)
    for (name, tag), it in zip(keys, items):
        a = (ans.get(it["id"]) or {}).get("a")
        if a is None:
            continue
        key = f"{name}:{tag}" if tag else name
        if name == "key.outfit":
            p = ctx["people"][tag]
            ok = R.outfit_match(a, R.outfit_words(p["outfit"]))
        elif name == "key.age_band":
            m = re.search(r"\d+", a)
            tgt = ctx["people"][tag]["age"]
            ok = bool(m) and abs(int(m.group(0)) - tgt) <= (3 if tgt < 25 else 8)
        elif name == "key.facial_hair":
            ok = a.startswith(ctx["people"][tag]["beard"])
        else:
            ok = a.startswith(VL[name][1])
        out[key] = {"pass": ok, "why": a}
    sf = ctx.get("speaker_face")
    if ctx.get("speaker") and want("key.speaker_visible"):
        out["key.speaker_visible"] = {"pass": bool(sf), "why": "face found" if sf else "no speaker face"}
    if sf:
        if want("key.speaker_face_size"):
            out["key.speaker_face_size"] = {"pass": sf["h_frac"] >= 0.15, "why": f"{sf['h_frac']:.0%} of the height"}
        if want("key.speaker_face_light"):
            out["key.speaker_face_light"] = {"pass": sf["luma"] >= 45, "why": f"luma {sf['luma']:.0f}"}
    return out


def _sample_frames(mp4: str, tmp: str) -> list:
    """The QA's 4 sampled frames (20/45/70/95 %) as PNGs (same picks as remake_chapter.qa)."""
    n = int(float(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                                  "stream=nb_read_frames", "-of", "csv=p=0", str(mp4)], capture_output=True,
                                 text=True).stdout.strip() or 0))
    out = []
    for k, f in enumerate([0.2, 0.45, 0.7, 0.95]):
        png = Path(tmp) / f"{Path(mp4).stem}_t{k}.png"
        if not png.exists():
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-vf", f"select=eq(n\\,{max(0, int(n * f) - 1)})",
                            "-vframes", "1", str(png)], capture_output=True)
        if png.exists():
            out.append(png)
    return out


def placeholder_verdict(pngs: list) -> dict:
    import placeholder_check as PH
    hits = {p.name: r for p in pngs if (r := PH.check(str(p)))["placeholder"]}
    return {"pass": not hits if pngs else None, "why": hits or f"none in {len(pngs)} frames"}


def check_clip(mp4: str, ctx: dict) -> dict:
    """Clip checks through tools/video_qa.py (+ the text VL on 4 frames). ctx: {speech: wav?, speaker_x?,
    expect?, anchors?, cam_expect?, chunks: [{mp4, wav}], stitched: {n_expected}, only: [names]: just those
    (a CPU-only check calibrates without the GPU)}"""
    out = {}
    only = set(ctx.get("only") or [])
    if not only or "clip.no_placeholder" in only:
        out["clip.no_placeholder"] = placeholder_verdict(_sample_frames(mp4, ctx.get("tmp", "/tmp")))
    if only and only <= {"clip.no_placeholder"}:
        return out
    js = Path(ctx.get("tmp", "/tmp")) / (Path(mp4).stem + "_qa.json")
    args = [str(mp4), "--json", str(js), "--no-identity"]
    for k, flag in (("speech", "--speech"), ("speaker_x", "--speaker-x"), ("expect", "--expect"), ("anchors", "--anchors")):
        if ctx.get(k) is not None:
            args += [flag, str(ctx[k])]
    if ctx.get("cam_expect"):
        args += ["--cam-expect", json.dumps(ctx["cam_expect"])]
    subprocess.run([str(POSE_PY), "-I", str(HERE / "video_qa.py"), *args], capture_output=True, text=True,
                   env=dict(os.environ, OMP_NUM_THREADS="4"), timeout=1800)
    q = json.loads(js.read_text()) if js.exists() else {"flags": ["qa crashed"]}
    fl = " | ".join(q.get("flags", []))
    if ctx.get("speech"):
        out["clip.speaker_present"] = {"pass": "speaker missing" not in fl, "why": q.get("speaker_present")}
        m = q.get("mouth") or {}
        out["clip.lip_sync"] = {"pass": m.get("ok") is not False and m.get("ok") is not None, "why": m}
        if "listener_vs_speaker" in m:
            out["clip.listener_quiet"] = {"pass": m["listener_vs_speaker"] <= 0.8, "why": m["listener_vs_speaker"]}
    out["clip.camera"] = {"pass": not re.search(r"zoom|slide|camera move|cut at", fl), "why": q.get("camera")}
    # text on 4 frames
    n = q.get("frames") or 0
    items = []
    for k, f in enumerate([0.2, 0.45, 0.7, 0.95]):
        png = Path(ctx.get("tmp", "/tmp")) / f"{Path(mp4).stem}_t{k}.png"
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", str(mp4), "-vf", f"select=eq(n\\,{max(0, int(n * f) - 1)})",
                        "-vframes", "1", str(png)], capture_output=True)
        if png.exists():
            items.append({"id": f"t{k}", "image": str(png), "ask": {"a": VL["clip.no_text"][0]}})
    ans = _vl(items)
    hits = [k for k, v in ans.items() if v.get("a", "").startswith("yes")]
    if ans:
        out["clip.no_text"] = {"pass": not hits, "why": f"text in {len(hits)}/{len(ans)} frames"}
    chunks = ctx.get("chunks") or []
    if len(chunks) > 1:
        acts = []
        for c in chunks:
            cj = Path(ctx.get("tmp", "/tmp")) / (Path(c["mp4"]).stem + "_cq.json")
            subprocess.run([str(POSE_PY), "-I", str(HERE / "video_qa.py"), c["mp4"], "--speech", c["wav"], "--no-identity",
                            "--json", str(cj), *(["--speaker-x", str(ctx["speaker_x"])] if ctx.get("speaker_x") else [])],
                           capture_output=True, text=True, env=dict(os.environ, OMP_NUM_THREADS="4"), timeout=1800)
            acts.append(((json.loads(cj.read_text()).get("mouth") or {}) if cj.exists() else {}).get("act_voiced"))
        ok = all(a is not None and a >= 0.5 * (acts[0] or 0) for a in acts[1:]) if acts and acts[0] else None
        out["clip.chunk_lips"] = {"pass": ok, "why": acts}
    if ctx.get("stitched"):
        nf = int(subprocess.run(["ffprobe", "-v", "error", "-count_frames", "-select_streams", "v:0", "-show_entries",
                                 "stream=nb_read_frames", "-of", "csv=p=0", str(mp4)], capture_output=True,
                                text=True).stdout.strip() or 0)
        ne = ctx["stitched"]["n_expected"]
        # a doubled overlap makes the join LONGER than start(last)+n(last); the mux may trim a few frames at the end
        # (-shortest vs the audio): calibrated 10-20, stitched 210 vs 214 = fine, naive concat 222 = doubled
        out["clip.join_once"] = {"pass": ne - 8 <= nf <= ne + 2, "why": f"{nf} vs {ne}"}
    return out


def check_tool(name: str, ctx: dict) -> dict:
    """Staging/asset checks read the owning tool's QA output (blockout/motion/characters/props): ctx {"file", "field",
    "rule"}; a check without its tool output is 'not run' (listed, never a silent pass)."""
    f = Path(ctx.get("file", ""))
    if not f.exists():
        return {name: {"pass": None, "why": f"no tool output {f}"}}
    d = json.loads(f.read_text())
    for k in (ctx.get("field") or "").split("."):
        d = d.get(k) if isinstance(d, dict) and k else d
    rule = ctx.get("rule", "truthy")
    ok = (bool(d) if rule == "truthy" else (d == 0 if rule == "zero" else (not d if rule == "falsy" else None)))
    return {name: {"pass": ok, "why": d}}


# ------------------------------------------------------------------------------------------------- calibration

def calibrate(out_json: str | None = None, only: str | None = None) -> list:
    """Every fixture through its check: the agreement table (the user's verdict vs the registry's)."""
    rows = []
    for case in sorted(FIX.glob("*/*.json")):
        try:
            c = json.loads(case.read_text())
        except ValueError:
            continue
        if not isinstance(c, dict) or "check" not in c or "verdict" not in c:
            continue                     # a tool-output copy, not a case
        if only and not c["check"].startswith(only):
            continue                     # --check: one check (or a prefix) without loading the GPU judge
        name, media, expect = c["check"], case.with_name(c["media"]), c["verdict"]
        stage = name.split(".")[0]
        try:
            if stage == "key":
                res = check_key(str(media), c["ctx"], only={name})
            elif stage == "clip":
                res = check_clip(str(media), {**c["ctx"], "tmp": str(case.parent)})
            else:
                res = check_tool(name, c["ctx"])
        except Exception as ex:  # noqa: BLE001
            res = {name: {"pass": None, "why": f"error {ex}"}}
        got = [v for k, v in res.items() if k == name or k.startswith(name + ":")]
        verdict = None if not got or any(v["pass"] is None for v in got) else ("pass" if all(v["pass"] for v in got) else "fail")
        rows.append({"check": name, "fixture": f"{case.parent.name}/{case.stem}", "user": expect, "registry": verdict,
                     "agree": verdict == expect, "why": [v["why"] for v in got]})
    if out_json:
        Path(out_json).write_text(json.dumps(rows, indent=1, default=str))
    return rows


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=["list", "calibrate"])
    ap.add_argument("--json")
    ap.add_argument("--check", help="calibrate only this check (or name prefix)")
    a = ap.parse_args(argv)
    if a.cmd == "list":
        for n, st, det, thr, fix in CHECKS:
            print(f"{n:30s} {st:8s} {det:9s} {thr:52s} fix: {fix}")
    else:
        rows = calibrate(a.json, a.check)
        for r in rows:
            print(f"{'OK ' if r['agree'] else 'XX '} {r['check']:28s} {r['fixture']:40s} user={r['user']:4s} "
                  f"registry={r['registry']}  {str(r['why'])[:90]}")
        n = sum(r["agree"] for r in rows)
        print(f"agreement {n}/{len(rows)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
