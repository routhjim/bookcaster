"""11-24: a key of someone drawn from photographs is given the face of their sheet as a reference picture; nobody
else's key job or signature changes; the 3D flow finds a photograph sheet by its own name."""
import json, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
import shot_flow as SF
from tts_reader import video
from PIL import Image

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

class R:
    def __init__(self, vd): self.vd = vd
    def person(self, n): return n
    def look(self, n): return "a broad woman of about 50, grey hair"
    def ref_style(self): return "plain ink drawing"
    def aliases(self, n): return [n]

with tempfile.TemporaryDirectory() as td:
    book = Path(td) / "book"
    vd = book / "work" / "video"
    r = R(vd)
    (vd / "refs").mkdir(parents=True)
    check("no photographs: no face", SF.face_ref(book, "01_one", r, "Orla") is None)
    (vd / "faces" / video.slug("Orla")).mkdir(parents=True)
    Image.new("RGB", (64, 64), "grey").save(vd / "faces" / video.slug("Orla") / "photo1.png")
    check("photographs but no sheet yet: no face", SF.face_ref(book, "01_one", r, "Orla") is None)
    sh = video.ref_path(vd / "refs", "Orla", video.sheet_prompt(r.look("Orla"), r.ref_style(), True))
    Image.new("RGB", (1280, 896), "white").save(sh)
    f = SF.face_ref(book, "01_one", r, "Orla")
    check("the face is cut from the sheet's portrait", f is not None and Image.open(f).size == (640, 582),
          f and Image.open(f).size)
    check("it is named by the sheet and kept with the shots", f.name == sh.name and f.parent == SF.shot_dir(book, "01_one") / "faces", f)
    check("someone else has none", SF.face_ref(book, "01_one", r, "Tam") is None)

src = (HERE.parent / "shot_flow.py").read_text()
check("the key job carries the faces after the place", '"refs": ([str(hero)] if hero else []) + faces' in src)
check("the signature changes only where there is a face", "+ ([faces] if faces else [])" in src)
check("the 3D flow looks for the photograph sheet", "bool(video.face_photos(R, p_))" in (HERE.parent / "remake_chapter.py").read_text())

print("\nFAILED: " + ", ".join(fails) if fails else "\nall passed")
sys.exit(1 if fails else 0)
