"""The face repaint: one face per staged person, a reference of the view the head is seen from, the drawing kept
when the repaint is worse, and a paste that leaves no edge (11-25)."""
import sys, tempfile, types
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import numpy as np
from PIL import Image
import remake_chapter as R

ok = True
def check(name, cond):
    global ok
    print(("PASS " if cond else "FAIL ") + name)
    ok = ok and bool(cond)

# --- one face per person
boxes = {"Orla": [300, 20, 900, 640], "Tam": [20, 100, 200, 640]}
faces = [{"box": [400, 50, 660, 330]},      # Orla's own, close to the camera
         {"box": [800, 180, 856, 244]},     # someone of the crowd behind her, inside her box
         {"box": [450, 120, 650, 250]},     # a second reading of her own face
         {"box": [320, 180, 380, 250]},     # another of the crowd
         {"box": [80, 120, 140, 190]},      # Tam's
         {"box": [1000, 300, 1040, 350]}]   # nobody's box
got = R.own_faces(faces, boxes, 1.0, 1.0, 640)
check("each staged person gets one face", set(got) == {"Orla", "Tam"})
check("... the largest in their box", got["Orla"]["box"] == [400, 50, 660, 330] and got["Tam"]["box"] == [80, 120, 140, 190])
check("a face in two boxes goes to the smaller figure",
      R.own_faces([{"box": [90, 130, 150, 200]}], {"Orla": [0, 0, 900, 640], "Tam": [20, 100, 200, 640]}, 1, 1, 640).keys() == {"Tam"})
check("boxes are scaled to the key", list(R.own_faces([{"box": [200, 100, 300, 200]}], {"Orla": [100, 20, 200, 300]}, 2.0, 2.0, 640)) == ["Orla"])

# --- the reference follows the view
with tempfile.TemporaryDirectory() as td:
    td = Path(td)
    refs = td / "video" / "charlora" / R.pslug("Orla") / "refs"
    refs.mkdir(parents=True)
    (refs / "photo_three-quarter_left_cut.png").write_bytes(b"x")
    (refs / "photo_profile_right_cut.png").write_bytes(b"x")
    cx = types.SimpleNamespace(W=td, refs=td / "refs", OUT=td / "out")
    real = R.ref_sheet, R.headshot
    R.ref_sheet = lambda refs_, w: td / "sheet.png"
    R.headshot = lambda cx_, w, sheet: "HEAD"
    try:
        check("a head facing the camera: the sheet's headshot", R.face_ref(cx, "Orla", 10.0) == ("HEAD", "front"))
        check("no turn known: the headshot", R.face_ref(cx, "Orla", None) == ("HEAD", "front"))
        p, v = R.face_ref(cx, "Orla", -45.0)
        check("turned left: their own face turned left", v == "three-quarter left" and p.endswith("photo_three-quarter_left_cut.png"))
        p, v = R.face_ref(cx, "Orla", 68.0)
        check("in profile to the right: their profile", v == "profile right" and p.endswith("photo_profile_right_cut.png"))
        check("turned to a side they have no picture of: no reference, never the front one", R.face_ref(cx, "Orla", 45.0) == (None, ""))
        check("someone with no turned pictures: none past the limit", R.face_ref(cx, "Tam", -50.0) == (None, ""))
    finally:
        R.ref_sheet, R.headshot = real

# --- kept only when not worse
V = R.repaint_verdict
check("as like the sheet, as sharp: the repaint is kept", V({"face_sim": 0.72, "age_gap": 1.0}, {"face_sim": 0.75, "age_gap": -1.0}, 90, 70) == "")
check("a hair less like: kept", V({"face_sim": 0.73, "age_gap": 1.0}, {"face_sim": 0.72, "age_gap": 1.0}, 90, 60) == "")
check("less like the sheet: the drawing stays", "less like" in V({"face_sim": 0.78, "age_gap": 3.0}, {"face_sim": 0.74, "age_gap": 3.0}, 90, 90))
check("blurrier: the drawing stays", "blurrier" in V({"face_sim": 0.7}, {"face_sim": 0.8}, 400, 100))
check("no face left: the drawing stays", "no face" in V({"face_sim": 0.7}, {"skipped": "no face found"}, 90, 90))
check("aged past the band: the drawing stays", "age" in V({"face_sim": 0.7, "age_gap": 4.0}, {"face_sim": 0.7, "age_gap": 19.0}, 90, 90))
check("an age reading inside the band is no difference", V({"face_sim": 0.7, "age_gap": 1.0}, {"face_sim": 0.7, "age_gap": 9.0}, 90, 90) == "")
check("nothing measured: sharpness alone decides", V(None, None, 90, 80) == "" and "blurrier" in V(None, None, 90, 10))

# --- sharpness
rng = np.random.default_rng(3)
noisy = Image.fromarray(rng.integers(0, 255, (64, 64, 3), dtype=np.uint8))
flat = Image.new("RGB", (64, 64), (120, 120, 120))
check("a flat picture is not sharp, a busy one is", R.sharpness(flat) == 0 and R.sharpness(noisy) > 100)
check("... measured inside the box", R.sharpness(noisy, [8, 8, 40, 40]) > 100)

# --- the paste
base = Image.new("RGB", (400, 300), (100, 90, 80))
patch = Image.new("RGB", (200, 200), (140, 130, 120))     # a repaint lighter than the drawing all over
c0, fb = [100, 0, 300, 200], [150, 20, 250, 120]          # the crop starts at the picture's top edge
out = np.asarray(R.face_paste(base, patch, c0, fb), np.int32)
b = np.asarray(base, np.int32)
check("outside the crop nothing changes", (out[:, :100] == b[:, :100]).all() and (out[:, 300:] == b[:, 300:]).all() and (out[200:] == b[200:]).all())
check("no step along the crop's sides", abs(out[100, 100] - b[100, 100]).max() <= 1 and abs(out[100, 299] - b[100, 299]).max() <= 1
      and abs(out[199, 200] - b[199, 200]).max() <= 1)
check("the face itself is the repaint's", abs(out[70, 200] - b[70, 200]).max() >= 10)
check("the repaint is shifted towards the drawing's colour, by no more than the limit",
      (out[70, 200] - b[70, 200]).max() <= 40 - R.FACE_TONE_MAX + 1)
far = Image.new("RGB", (200, 200), (250, 250, 250))
o2 = np.asarray(R.face_paste(base, far, c0, fb), np.int32)
check("a very different repaint is not flattened to the drawing", (o2[70, 200] - b[70, 200]).min() >= 100)

print("all passed" if ok else "FAILED")
sys.exit(0 if ok else 1)
