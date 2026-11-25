"""11-24: a place picture showing things out of the book's period is redrawn with those things named as absent;
the best attempt is kept; a judged picture is not judged or redrawn again; no period = no check."""
import json, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import flow3d as F
import remake_chapter as R
from tts_reader import video

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

with tempfile.TemporaryDirectory() as td:
    book = Path(td) / "book"; refs = book / "work" / "video" / "refs"; refs.mkdir(parents=True)
    (book / "chapters").mkdir()
    hall, yard = refs / "hall.png", refs / "yard.png"
    hall.write_bytes(b"hall0"); yard.write_bytes(b"yard0")
    jobs = [{"id": "ref_hall", "size": [1024, 576], "seed": 7, "prompt": "P hall.", "path": str(hall)},
            {"id": "ref_yard", "size": [1024, 576], "seed": 7, "prompt": "P yard.", "path": str(yard)},
            {"id": "ref_orla", "size": [1280, 896], "seed": 7, "prompt": "P orla.", "path": str(refs / "orla.png")}]
    video_jobs, asked, drawn = video.reference_jobs, [], []
    video.reference_jobs = lambda plan, style, refs_: [dict(j) for j in jobs]
    verdict = {b"hall0": "a wall socket, a strip light", b"hall1": "a wall socket", b"hall2": "None.", b"yard0": "none"}
    def vl(items):
        asked.extend(i["id"] for i in items)
        return {i["id"]: {"modern": verdict[Path(i["image"]).read_bytes()]} for i in items}
    def draw(js, d, log=print, on_done=None):
        for j in js:
            drawn.append(j); Path(j["path"]).write_bytes(b"hall%d" % len(drawn))
    R.vl_batch, video.draw_jobs = vl, draw
    check("no period: nothing asked", F.check_places({}, "s", refs, lambda *a: None) == {} and not asked)
    (book / "period.txt").write_text("a made-up age, around the year 300")
    out = F.check_places({}, "s", refs, lambda *a: None)
    check("only places are judged", "ref_orla" not in asked, asked)
    check("the bad place is redrawn twice, the good one never", [j["id"] for j in drawn] == ["ref_hall", "ref_hall"], drawn)
    check("the redraw names what was seen", "none of these: a wall socket, a strip light." in drawn[0]["prompt"]
          and drawn[0]["seed"] != 7, drawn[0])
    check("the clean attempt is kept, nothing left failing", hall.read_bytes() == b"hall2" and out == {}, out)
    n = len(asked)
    F.check_places({}, "s", refs, lambda *a: None)
    check("a judged picture is not judged again", len(asked) == n and len(drawn) == 2)
    # never clean: the attempt with the fewest stays, and is reported
    hall.write_bytes(b"hall0"); drawn.clear()
    verdict.update({b"hall1": "a socket, a lamp, a sign", b"hall2": "a socket, a lamp"})
    out = F.check_places({}, "s", refs, lambda *a: None)
    check("the best attempt stays and is reported", hall.read_bytes() in (b"hall0", b"hall2") and len(out["ref_hall"]) == 2, out)
    video.reference_jobs = video_jobs

print("\nFAILED: " + ", ".join(fails) if fails else "\nall passed")
sys.exit(1 if fails else 0)
