"""11-25: three key checks that failed good keys: a short beard asked about as a 'full beard', a white gown read as
an unpainted object, and a face judged on a figure too small or cut by the frame's edge."""
import sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parent.parent))
import remake_chapter as R
from PIL import Image, ImageDraw

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

q = R.beard_question("on the left", "yes")
check("a bearded look is asked for any beard", "any beard" in q and "full" not in q and "ignore" not in q, q)
q = R.beard_question("on the left", "no")
check("a clean-shaven look still ignores stubble", "full beard" in q and "ignore light stubble" in q, q)

check("white gown is white clothes", R.wears_white("a woman of 30, red hair, wearing a white muslin gown and a blue sash"))
check("white hair is not white clothes", not R.wears_white("an old man with white hair, wearing a blue coat"))
check("a dark coat is not white clothes", not R.wears_white("a man wearing a black coat"))

figs = {"Orla": [100, 0, 500, 640], "Tam": [700, 100, 900, 640]}
looks = {"Orla": "a woman wearing a white linen gown", "Tam": "a man wearing a black coat"}
check("a patch on the white gown is hers", R.in_white_clothes((0.25, 0.6), figs, looks, (1152, 640)))
check("a patch on the dark coat is not", not R.in_white_clothes((0.7, 0.6), figs, looks, (1152, 640)))
check("a patch on the wall is not", not R.in_white_clothes((0.55, 0.2), figs, looks, (1152, 640)))
check("no patch, no claim", not R.in_white_clothes(None, figs, looks, (1152, 640)))

check("a full figure in frame is judged", R.face_in_frame([300, 40, 520, 640], 1152, 640))
check("a sliver at the right edge is not", not R.face_in_frame([1020, 0, 1152, 640], 1152, 640))
check("a sliver at the left edge is not", not R.face_in_frame([0, 0, 120, 640], 1152, 640))
check("a wide figure at the edge is judged", R.face_in_frame([0, 0, 400, 640], 1152, 640))
check("a far small figure is not", not R.face_in_frame([500, 200, 540, 330], 1152, 640))
check("a figure a third of the frame high is not", not R.face_in_frame([500, 200, 580, 440], 1152, 640))
check("a figure half the frame high is judged", R.face_in_frame([500, 200, 620, 530], 1152, 640))

with tempfile.TemporaryDirectory() as td:
    im = Image.new("RGB", (640, 360), (90, 70, 40))
    ImageDraw.Draw(im).rectangle([64, 180, 255, 323], fill="white")
    im.save(Path(td) / "k.png")
    f, at = R.white_blob(Path(td) / "k.png", centre=True)
    check("the patch's share", abs(f - 0.12) < 0.01, f)
    check("the patch's centre", at and abs(at[0] - 0.25) < 0.02 and abs(at[1] - 0.7) < 0.02, at)
    check("the plain call still gives the share", abs(R.white_blob(Path(td) / "k.png") - f) < 1e-9)
    Image.new("RGB", (640, 360), (90, 70, 40)).save(Path(td) / "n.png")
    check("no patch: no centre", R.white_blob(Path(td) / "n.png", centre=True) == (0.0, None))

with tempfile.TemporaryDirectory() as td:
    lt, dk = Path(td) / "light.png", Path(td) / "dark.png"
    im = Image.new("RGB", (640, 360), (90, 70, 40))
    ImageDraw.Draw(im).rectangle([100, 150, 300, 359], fill=(235, 232, 225))
    im.save(lt)
    Image.new("RGB", (640, 360), (90, 70, 40)).save(dk)
    fg = {"Orla": [100, 0, 500, 640]}
    check("white where the model is white: theirs", R.white_in_still((0.3, 0.7), fg, (1152, 640), lt))
    check("white where the model is dark: a blown-out head", not R.white_in_still((0.3, 0.7), fg, (1152, 640), dk))
    check("white off every person: an object", not R.white_in_still((0.3, 0.7), {"Orla": [700, 0, 900, 640]}, (1152, 640), lt))
    check("no still: no claim", not R.white_in_still((0.3, 0.7), fg, (1152, 640), Path(td) / "none.png"))

print("\nFAILED: " + ", ".join(fails) if fails else "\nall passed")
sys.exit(1 if fails else 0)
