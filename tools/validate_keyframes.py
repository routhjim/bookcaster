"""Validate the best-of-N keyframe checks against hand labels (plan 10-18, PLAN-keyframes-2026-10-18.md).

For each labelled keyframe of a chapter: the fast yes/no ranking questions, the careful hard-only check and
(optionally) the detectors, compared with the labels: which real faults each catches (recall) and how many
good frames each wrongly fails (false alarms), with timings. Also whether the server returns yes/no
probabilities. Needs the language model up (Strata); run with no render going.

    python -m tools.validate_keyframes <work_dir> <labels.json> [--detect]   (from ~/tts_reader)
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from tts_reader import animate, bookcaster as b, video  # noqa: E402


def main() -> int:
    work, labels = Path(sys.argv[1]), json.loads(Path(sys.argv[2]).read_text())
    use_detect = "--detect" in sys.argv
    animate.DETECT_ON = use_detect
    plan = json.loads((work / "plan.json").read_text())
    plan["look"]["characters"] = json.loads((work.parent / "look.json").read_text())["characters"]
    kf = animate.keyframes(work)
    chars = plan["look"]["characters"]
    animate._add_creatures(kf["frames"], chars)
    text = b._chapter_text(next(f for f in b.chapter_files(work.parent.parent.parent) if f.stem == work.name))
    timeline = json.loads((work.parent.parent / f"{work.name}.timeline.json").read_text())
    starts = video.paragraph_times(video.paragraphs(text), timeline)
    animate.apply_wardrobe(kf, plan, starts)
    kdir = work / "anim" / "keyframes"
    refs_dir = work.parent / "refs"
    known = video.ref_prompts(plan, b.art_style(work.parent.parent.parent) or video.DEFAULT_STYLE)
    url, model = b.LLM_URL, b.LLM_MODEL
    frames = [f for f in kf["frames"] if f.get("mode") == "key" and (kdir / f"{f['shot']}.png").exists()]

    def job(f):
        who = [w for w in f.get("who", []) if w in chars][:3]
        refs = [str(animate._figure(video.ref_path(refs_dir, w, known[w][0]), work / "anim" / "refs"))
                for w in who if w in known]
        return {"id": f["shot"], "path": str(kdir / f"{f['shot']}.png"), "frame": f, "who": who, "refs": refs,
                "states": [], "anchors": {}, "set": None, "seed": 0,
                "creatures": {w: r for w, r in zip(who, refs) if animate._hair(chars.get(w, "")) == "their face"}}

    def one(f):
        j = job(f)
        t = time.time()
        qs = animate._frame_questions(j, chars, plan)
        ans = animate._answer(Path(j["path"]), qs, url, model)
        t_fast = time.time() - t
        t = time.time()
        st = {"seeds": {}, "passed": {}, "log": []}
        animate.check_keyframes([j], st, chars, plan["look"].get("props", {}), url, model, lambda *a: None,
                                False, True, hard_only=True)
        careful = [] if j["id"] in st["passed"] else (st["log"][-1].get("missing", []) + st["log"][-1].get("problems", [])
                                                      if st["log"] else ["?"])
        t_careful = time.time() - t
        t = time.time()
        det = animate._detector_faults(Path(j["path"]), j, chars) if use_detect else ([], [])
        t_det = time.time() - t
        return {"id": j["id"], "questions": [q["q"] for q in qs], "answers": ans,
                "fast_score": animate._score(qs, ans) if ans else None, "careful": careful, "detect": det,
                "t_fast": round(t_fast, 1), "t_careful": round(t_careful, 1), "t_det": round(t_det, 1)}

    with ThreadPoolExecutor(2) as ex:
        rows = list(ex.map(one, frames))
    hard, soft = labels["hard"], labels.get("soft", {})
    good = [r for r in rows if r["id"] not in hard and r["id"] not in soft]
    report = {"frames": len(rows), "hard_labelled": len(hard), "good_labelled": len(good)}
    report["careful_recall"] = sum(1 for r in rows if r["id"] in hard and r["careful"]) / max(1, len(hard))
    report["careful_false_alarms"] = sum(1 for r in good if r["careful"]) / max(1, len(good))
    if use_detect:
        report["detect_recall"] = sum(1 for r in rows if r["id"] in hard and r["detect"][0]) / max(1, len(hard))
        report["detect_false_alarms"] = sum(1 for r in good if r["detect"][0]) / max(1, len(good))
    fast_soft = [r for r in rows if r["id"] in soft and r["answers"] is not None]
    report["fast_flags_soft"] = sum(1 for r in fast_soft if not all(r["answers"])) / max(1, len(fast_soft))
    report["fast_flags_good"] = sum(1 for r in good if r["answers"] and not all(r["answers"])) / max(1, len(good))
    for k in ("t_fast", "t_careful", "t_det"):
        v = sorted(r[k] for r in rows)
        report[k + "_median"] = v[len(v) // 2] if v else None
    try:
        text, toks = video._look_at_probs(url, model, Path(rows[0]["id"] and (kdir / f"{rows[0]['id']}.png")),
                                          'Is there a person in this picture? Reply with only ["yes"] or ["no"].', 20)
        report["server_logprobs"] = bool(toks)
    except Exception as exc:
        report["server_logprobs"] = f"error: {exc}"[:120]
    out = work / "anim" / "validation.json"
    out.write_text(json.dumps({"report": report, "rows": rows}, indent=1))
    print(json.dumps(report, indent=1))
    for r in rows:
        tag = "HARD" if r["id"] in hard else "soft" if r["id"] in soft else "ok"
        print(f"{r['id']} {tag:4s} careful={'FAIL' if r['careful'] else 'pass'} "
              f"fast={r['fast_score']} det={r['detect'][0][:1]} | {'; '.join(r['careful'])[:110]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
