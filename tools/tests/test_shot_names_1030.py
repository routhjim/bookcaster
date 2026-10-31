"""10-30 user ("make the naming useful for humans to read and also machines to decipher"): shot ids say their kind
and their start both ways (quiet-02m26s-001464); what a run left under the old ids moves to the new ones."""
import inspect, os, sys, tempfile
from pathlib import Path
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent)); sys.path.insert(0, str(HERE.parent.parent))
import shot_flow as S
from tts_reader import animate

fails = []
def check(name, ok, got=None):
    print(("PASS " if ok else "FAIL ") + name + ("" if ok else f"  got {got}"))
    ok or fails.append(name)

check("a quiet shot at 146.4 s", S.shot_mid("quiet", 146.4) == "quiet-02m26s-001464", S.shot_mid("quiet", 146.4))
check("an action shot past 20 minutes", S.shot_mid("action", 1329.0) == "action-22m09s-013290")
check("the seed is the old id's (cached takes stay the same takes)",
      S.shot_seed("quiet-02m26s-001464_s0") == S.shot_seed("action-001464_s0") != 0)
check("animate reads the time back", animate._shot_t("talk-01m26s-000868") == 86.8)
check("ids sort by time as text", sorted([S.shot_mid("quiet", 600.0), S.shot_mid("quiet", 59.9)])[0].endswith("000599"))
src = inspect.getsource(S.direct)
check("the director's shots use shot_mid with their own kind", 'shot_mid(kind_, t)' in src and 'shot_mid("talk", t)' in src)

sd = Path(tempfile.mkdtemp(prefix="names_"))
for f in ["action-001464/action-001464_s0/key.png", "action-001464/action-001464_s0/talk_seed",
          "ltx/out/action-001464_s0.mp4", "ltx/out/action-001464_s0_t2.mp4", "ltx/out/talk-000868_s0.sync.json",
          "ltx/emb/t_talk-000868_ab12.pt", "audio/talk-000868_4cb9_speech.wav", "action-009999/action-009999_s0/key.png"]:
    (sd / f).parent.mkdir(parents=True, exist_ok=True); (sd / f).write_text(f)
shots = [{"id": "quiet-02m26s-001464_s0", "kind": "quiet"}, {"id": "talk-01m26s-000868_s0", "kind": "talk"}]
S._migrate_ids(sd, shots, log=lambda m: None)
have = sorted(str(p.relative_to(sd)) for p in sd.rglob("*") if p.is_file())
want = ["action-009999/action-009999_s0/key.png", "audio/talk-01m26s-000868_4cb9_speech.wav",
        "ltx/emb/t_talk-01m26s-000868_ab12.pt", "ltx/out/quiet-02m26s-001464_s0.mp4",
        "ltx/out/quiet-02m26s-001464_s0_t2.mp4", "ltx/out/talk-01m26s-000868_s0.sync.json",
        "quiet-02m26s-001464/quiet-02m26s-001464_s0/key.png", "quiet-02m26s-001464/quiet-02m26s-001464_s0/talk_seed"]
check("keys, clips, takes, sync records, embeddings and voice cuts move; a shot not in the plan stays", have == want, have)
S._migrate_ids(sd, shots, log=lambda m: None)
check("running it again changes nothing", sorted(str(p.relative_to(sd)) for p in sd.rglob("*") if p.is_file()) == want)
print({"fails": fails})
