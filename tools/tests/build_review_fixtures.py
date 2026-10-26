"""Collect the 10-20 review findings (the user's/coordinator's verdicts) as regression fixtures for
tools/review_checks.py: tools/tests/fixtures/review/<check>/<case>.{png|mp4|json} + <case>.json {check, media, verdict, ctx}."""
import json, shutil, subprocess, sys
from pathlib import Path
HERE = Path(__file__).resolve().parent
FIX = HERE / "fixtures" / "review"
V = Path("/home/jrouth/audiobooks/knight_forged/work/video")
R3 = V / "01_chapter_1_remake3d"
P1 = R3 / "preview_talk"
REM = V / "01_chapter_1_remake"
LOOK = json.loads((V / "look.json").read_text())["characters"]


def boxes(render):
    ex = json.loads((render / "start_expect.json").read_text())
    return ex["size"], {f["who"]: f["bbox"] for f in ex["figures"] if f.get("bbox")}


def case(check, name, src, verdict, ctx):
    d = FIX / check
    d.mkdir(parents=True, exist_ok=True)
    src = Path(src)
    media = f"{name}{src.suffix}"
    shutil.copy(src, d / media)
    (d / f"{name}.json").write_text(json.dumps({"check": check, "media": media, "verdict": verdict, "ctx": ctx,
                                                "source": str(src)}, indent=1))


edric = "a youth of 18, smooth beardless boyish face, short messy brown hair"
jerkin = "a rough leather jerkin and simple trousers, no helm, no armor, everyday look"
s727, b727 = boxes(P1 / "stage" / "talk-000727" / "render")
s340, b340 = boxes(P1 / "stage" / "talk-000340" / "render")
# --- keys
case("key.location_interior", "fail_771_seascape", P1 / "stage/talk-000771/key.png", "fail", {"kind": "interior"})
case("key.no_sea_sky_interior", "fail_771_seascape", P1 / "stage/talk-000771/key.png", "fail", {"kind": "interior"})
case("key.location_interior", "pass_340_castle_hall", P1 / "stage/talk-000340/key.png", "pass", {"kind": "interior"})
case("key.set_present", "fail_727_door_no_forge", P1 / "stage/talk-000727/key.png", "fail",
     {"set_desc": "the inside of a blacksmith's forge with a glowing hearth fire"})
case("key.outfit", "fail_727_black_shirt", P1 / "stage/talk-000727/key.png", "fail",
     {"size": s727, "people": {"Edric": {"box": b727["Edric"], "outfit": jerkin}}})
case("key.age_band", "fail_727_edric_older", P1 / "stage/talk-000727/key.png", "fail",
     {"size": s727, "people": {"Edric": {"box": b727["Edric"], "age": 18}}})
case("key.facial_hair", "fail_340_bearded_knight", P1 / "stage/talk-000340/key.png", "fail",
     {"size": s340, "people": {"Knight": {"box": b340["Knight"], "beard": "no"}}})
from PIL import Image
ks = Image.open(V / "refs/knight-8a8e3755.png")
case("key.facial_hair", "pass_knight_sheet", V / "refs/knight-8a8e3755.png", "pass",
     {"size": list(ks.size), "people": {"Knight": {"box": [0, 0, ks.size[0] // 2, ks.size[1]], "beard": "no"}}})
case("key.back_hair", "fail_727_hulda_brown_hair", P1 / "stage/talk-000727/key_v1.png", "fail",
     {"size": s727, "people": {"Hulda": {"box": b727["Hulda"], "hair": "long gray hair tied back", "behind": True}}})
case("key.back_hair", "pass_727_hulda_grey", P1 / "stage/talk-000727/key.png", "pass",
     {"size": s727, "people": {"Hulda": {"box": b727["Hulda"], "hair": "long gray hair tied back", "behind": True}}})
case("key.speaker_face_light", "fail_340_dark", P1 / "stage/talk-000340/key.png", "fail",
     {"speaker": "Knight", "speaker_face": {"h_frac": 0.234, "luma": 42.2}})
case("key.speaker_face_light", "pass_771", P1 / "stage/talk-000771/key.png", "pass",
     {"speaker": "Hulda", "speaker_face": {"h_frac": 0.288, "luma": 57.6}})
case("key.story_prop", "fail_727_no_lantern", P1 / "stage/talk-000727/key.png", "fail", {"props": ["a lantern"]})
case("key.story_prop", "pass_727_original_lantern", V / "01_chapter_1/anim/talk-000727_first.png", "pass", {"props": ["a lantern"]})
# --- clips
pad = P1 / "ltx/audio"
case("clip.no_text", "fail_382_tablet", P1 / "ltx/out/r_talk-000382.mp4", "fail", {})
case("clip.no_text", "pass_771", P1 / "ltx/out/r_talk-000771.mp4", "pass", {})
case("clip.speaker_present", "fail_382_tablet", P1 / "ltx/out/r_talk-000382.mp4", "fail",
     {"speech": str(pad / "talk-000382_pad.wav"), "speaker_x": (b340["Knight"][0] + b340["Knight"][2]) / 2})
case("clip.speaker_present", "pass_771", P1 / "ltx/out/r_talk-000771.mp4", "pass", {"speech": str(pad / "talk-000771_pad.wav")})
d793 = FIX / "clip.chunk_lips"
d793.mkdir(parents=True, exist_ok=True)
chunks = []
for k in (0, 1):
    j = json.loads((P1 / f"ltx/out/r_talk-000793_c{k}.json").read_text())
    shutil.copy(P1 / f"ltx/out/r_talk-000793_c{k}.mp4", d793 / f"fail_793_c{k}.mp4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", str(j["audio_start"]), "-t", f"{j['frames'] / 24:.3f}", "-i",
                    str(pad / "talk-000793_pad.wav"), str(d793 / f"fail_793_c{k}.wav")], check=True)
    chunks.append({"mp4": str(d793 / f"fail_793_c{k}.mp4"), "wav": str(d793 / f"fail_793_c{k}.wav")})
case("clip.chunk_lips", "fail_793_frozen_c1", P1 / "ltx/out/r_talk-000793.mp4", "fail",
     {"chunks": chunks, "speaker_x": (b727["Edric"][0] + b727["Edric"][2]) / 2})
# doubled overlap: the naive concat of the 793 chunks vs the stitched line
lst = d793.parent / "clip.join_once"
lst.mkdir(parents=True, exist_ok=True)
(lst / "concat.txt").write_text("".join(f"file '{c['mp4']}'\n" for c in chunks))
subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(lst / "concat.txt"), "-c", "copy",
                str(lst / "naive_793.mp4")], check=True)
jj = [json.loads((P1 / f"ltx/out/r_talk-000793_c{k}.json").read_text()) for k in (0, 1)]
n_exp = jj[1]["chunk"][2] + jj[1]["frames"]
case("clip.join_once", "fail_naive_concat_793", lst / "naive_793.mp4", "fail", {"stitched": {"n_expected": n_exp}})
case("clip.join_once", "pass_stitched_793", P1 / "ltx/out/r_talk-000793.mp4", "pass", {"stitched": {"n_expected": n_exp}})
# --- assets (tool outputs)
M = Path("/home/jrouth/tts_reader/eval/blockout/models")
case("asset.creature_watertight", "fail_rig6_torn_jaw", M / "vorgrath_rig6/qa/q_checks.json", "fail",
     {"file": str(FIX / "asset.creature_watertight" / "fail_rig6_torn_jaw.json.src"), "field": "ok", "rule": "truthy"})
shutil.copy(M / "vorgrath_rig6/qa/q_checks.json", FIX / "asset.creature_watertight" / "fail_rig6_torn_jaw.json.src")
for name, verdict in (("stone", "fail"), ("hammer", "pass")):
    src = V / f"stage/props/{name}/model.json"
    dst = FIX / "asset.prop_rest" / f"{verdict}_{name}.model.json"
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(src, dst)
    case("asset.prop_rest", f"{verdict}_{name}", src, verdict, {"file": str(dst), "field": "rest_check.ok", "rule": "truthy"})
print("fixtures:", sorted(str(p.relative_to(FIX)) for p in FIX.glob("*/*.json")))
