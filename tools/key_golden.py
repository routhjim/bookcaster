#!/usr/bin/env python3
"""The keyframe GOLDEN-SET gate (10-20): any change to the key-draw code/config re-draws the frozen golden cases with
the CURRENT code path (stage draw -> bg_hold -> fast checks) and must match or beat them (checks + a person's look)
before it goes into a job. ONE change per run.

  key_golden.py freeze <book> <stem> SRC_OUT:ID [...]   copy each case's 3D render + its accepted key into golden/
  key_golden.py run <book> <stem> --tag T [--card 05] [--env K=V ...]   -> golden/run_<T>/sheet.jpg + result.json
The env (REMAKE_KEY_STRENGTH=0.4, REMAKE_PERSON_REFS=off, ...) is the change under test."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent


def gdir(book: Path, stem: str) -> Path:
    return Path(book) / "work" / "video" / stem / "flow3d" / "golden"


def freeze(book, stem, cases):
    g = gdir(book, stem) / "cases"
    for c in cases:
        src, i = c.rsplit(":", 1)
        sd = Path(src) / "stage" / i
        d = g / i
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
        shutil.copytree(sd / "render", d / "render", ignore=shutil.ignore_patterns("*.mp4"))
        for f in ("start_color_orig.png", "start_edges_orig.png"):      # a matted copy: freeze the original still
            if (d / "render" / f).exists():
                shutil.move(d / "render" / f, d / "render" / f.replace("_orig", ""))
        if (sd / "render" / "depth_figures.mp4").exists():
            shutil.copy(sd / "render" / "depth_figures.mp4", d / "render")
        if (sd / "key.png").exists():
            shutil.copy(sd / "key.png", d / "golden.png")
        cfg = {"source": str(sd), "frozen": time.strftime("%Y-%m-%d %H:%M")}
        for lf in sorted(Path(src, "stage").glob("draw_list_*.json")):
            try:
                for x in json.loads(lf.read_text()):
                    if x.get("out", "").endswith(f"/{i}/key.png"):
                        cfg["draw"] = x
            except (ValueError, AttributeError):
                pass
        (d / "config.json").write_text(json.dumps(cfg, indent=1))
        print(f"froze {i} <- {sd}")


def run(book, stem, tag, card, envs, rdir="render"):
    env = dict(os.environ, REMAKE_FULL3D="1", **dict(e.split("=", 1) for e in envs))
    g = gdir(book, stem)
    out = g / f"run_{tag}"
    if out.exists():
        shutil.rmtree(out)
    (out / "stage").mkdir(parents=True)
    ids = sorted(p.name for p in (g / "cases").iterdir())
    for i in ids:
        src = g / "cases" / i / rdir if (g / "cases" / i / rdir).exists() else g / "cases" / i / "render"
        shutil.copytree(src, out / "stage" / i / "render")
    (out / ".renders_done").touch()
    (out / "env.json").write_text(json.dumps(dict(e.split("=", 1) for e in envs), indent=1))
    code = f"""
import sys; sys.path.insert(0, {str(HERE)!r})
import remake_chapter as R
from pathlib import Path
R.DRAW_MIN_BATCH = 1
cx = R.Ctx(Path({str(book)!r}), {stem!r}, Path({str(out)!r}))
P = R.plan(cx)
sub = {{**P, "entries": [e for e in P["entries"] if e["id"] in {ids!r}]}}
for e in sub["entries"]:
    e["route"] = "talk" if e["id"].startswith("talk") else "stage"
R.stage(cx, sub, steps="draw", workers=1, seeds=2, card={card!r})
R.bg_hold(cx, sub, {card if card != "any" else "05"!r})
res = R.quick_key_checks(cx, sub)
import json; Path({str(out / "checks.json")!r}).write_text(json.dumps(res, indent=1, default=str))
"""
    t = time.time()
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, cwd=str(HERE.parent))
    (out / "run.log").write_text(r.stdout + r.stderr)
    sheet(g, out, ids)
    print(f"run {tag}: rc {r.returncode}, {time.time() - t:.0f} s -> {out / 'sheet.jpg'}")


VARIANT_CODE = """
import sys, json, subprocess, shutil; sys.path.insert(0, {here!r})
import remake_chapter as R
from pathlib import Path
from PIL import Image, ImageFilter
R.DRAW_MIN_BATCH = 1
out = Path({out!r}); ids = {ids!r}; variants = {variants!r}; card = {card!r}
cx = R.Ctx(Path({book!r}), {stem!r}, out)
P = R.plan(cx)
sub = {{**P, "entries": [e for e in P["entries"] if e["id"] in ids]}}
for e in sub["entries"]:
    e["route"] = "talk" if e["id"].startswith("talk") else "stage"
lists = []
real = subprocess.run
def fake(cmd, *a, **k):          # capture the CURRENT code's draw list instead of drawing it
    if any(str(c).endswith("keyframe_draw.py") for c in cmd):
        lists.append(cmd[cmd.index("--list") + 1]); return subprocess.CompletedProcess(cmd, 0, "", "")
    return real(cmd, *a, **k)
R.subprocess.run = fake
R.stage(cx, sub, steps="draw", workers=1, seeds=1, card=card)
R.subprocess.run = real
shots = [x for lf in lists for x in json.loads(Path(lf).read_text()) if x.get("frame", "start") == "start" and "key4" not in x["out"]]
allv = []
for sh in shots:
    d = Path(sh["out"]).parent
    e = next(e for e in sub["entries"] if e["id"] == d.name)
    fm = R.figure_mask(cx, e, (832, 448))
    if fm is not None:
        fm.resize(Image.open(d / "render" / "start_color.png").size).filter(ImageFilter.MaxFilter(7)).save(d / "render" / "start_figmask.png")
    for v in variants:
        name, st, bg = v["name"], v["strength"], v.get("bg")
        x = {{**sh, "out": str(d / f"key_{{name}}.png"), "strength": st, "seeds": 1}}
        x.pop("mask", None); x.pop("strength_bg", None)
        if bg and fm is not None:
            x["mask"], x["strength_bg"] = str(d / "render" / "start_figmask.png"), bg
        allv.append(x)
lf = out / "stage" / "variants_list.json"
lf.write_text(json.dumps(allv, indent=1))
hip = real([str(R.HOME / "bin" / "hip-index"), card], capture_output=True, text=True).stdout.strip()
import os
r = real([str(R.POSE_PY), "-I", str(R.HERE / "keyframe_draw.py"), "--list", str(lf), "--lock", f"/tmp/gpu-{{card}}.lock"],
         capture_output=True, text=True, env=dict(os.environ, HIP_VISIBLE_DEVICES=hip))
(out / "variants_draw.log").write_text(r.stdout + r.stderr)
R.kernel_check(out)
res = {{}}
for v in variants:          # the fast checks per variant: its own OUT with the variant's key as key.png
    vo = out / f"v_{{v['name']}}"
    for e in sub["entries"]:
        s_ = out / "stage" / e["id"]
        k = s_ / f"key_{{v['name']}}.png"
        if not k.exists() and (s_ / f"key_{{v['name']}}_best.png").exists():
            k = s_ / f"key_{{v['name']}}_best.png"
        if k.exists():
            (vo / "stage" / e["id"]).mkdir(parents=True, exist_ok=True)
            shutil.copy(k, vo / "stage" / e["id"] / "key.png")
            if not (vo / "stage" / e["id"] / "render").exists():
                (vo / "stage" / e["id"] / "render").symlink_to(s_ / "render")
    cxv = R.Ctx(Path({book!r}), {stem!r}, vo)
    q = R.quick_key_checks(cxv, sub)
    for i, rr in q.items():
        dj = out / "stage" / i / f"key_{{v['name']}}_draw.json"
        lk = (json.loads(dj.read_text())["results"][0].get("likeness") if dj.exists() else None)
        res.setdefault(i, {{}})[v["name"]] = {{"flags": rr.get("flags", []), "likeness": lk}}
(out / "variants.json").write_text(json.dumps(res, indent=1))
"""


def variants(book, stem, tag, card, vs):
    g = gdir(book, stem)
    out = g / f"run_{tag}"
    if out.exists():
        shutil.rmtree(out)
    (out / "stage").mkdir(parents=True)
    ids = sorted(p.name for p in (g / "cases").iterdir())
    for i in ids:
        shutil.copytree(g / "cases" / i / "render", out / "stage" / i / "render")
    (out / ".renders_done").touch()
    vv = []
    for v in vs:                       # name:strength[:bg]
        f = v.split(":")
        vv.append({"name": f[0], "strength": float(f[1]), "bg": float(f[2]) if len(f) > 2 else None})
    code = VARIANT_CODE.format(here=str(HERE), out=str(out), ids=ids, variants=vv, card=card, book=str(book), stem=stem)
    t = time.time()
    r = subprocess.run([sys.executable, "-c", code], env=dict(os.environ, REMAKE_FULL3D="1"), capture_output=True, text=True,
                       cwd=str(HERE.parent))
    (out / "run.log").write_text(r.stdout + r.stderr)
    res = json.loads((out / "variants.json").read_text()) if (out / "variants.json").exists() else {}
    from PIL import Image, ImageDraw
    W, H = 416, 224
    cols = 2 + len(vv)
    im = Image.new("RGB", (W * cols, (H + 30) * len(ids)), (20, 20, 20))
    dr = ImageDraw.Draw(im)
    for r_, i in enumerate(ids):
        y = r_ * (H + 30)
        cells = [("still", g / "cases" / i / "render" / "start_color.png"), ("golden", g / "cases" / i / "golden.png")]
        for v in vv:
            k = out / "stage" / i / f"key_{v['name']}.png"
            cells.append((v["name"], k if k.exists() else out / "stage" / i / f"key_{v['name']}_best.png"))
        for c, (nm, p) in enumerate(cells):
            if p.exists():
                im.paste(Image.open(p).convert("RGB").resize((W, H)), (c * W, y + 30))
            rv = (res.get(i) or {}).get(nm) or {}
            lk = rv.get("likeness") or {}
            txt = nm + (" lk " + ",".join(f"{k[:5]}{v:.2f}" for k, v in lk.items()) if lk else "")
            dr.text((c * W + 4, y + 2), txt, fill=(255, 255, 0))
            if rv.get("flags"):
                dr.text((c * W + 4, y + 15), "; ".join(rv["flags"])[:70], fill=(255, 120, 120))
        dr.text((4, y + 30), i, fill=(0, 255, 255))
    im.save(out / "sheet.jpg", quality=82)
    print(f"variants {tag}: rc {r.returncode}, {time.time() - t:.0f} s -> {out / 'sheet.jpg'}")


def sheet(g, out, ids):
    from PIL import Image, ImageDraw
    W, H = 560, 302
    chk = json.loads((out / "checks.json").read_text()) if (out / "checks.json").exists() else {}
    im = Image.new("RGB", (W * 3, (H + 18) * len(ids)), (20, 20, 20))
    dr = ImageDraw.Draw(im)
    for r, i in enumerate(ids):
        y = r * (H + 18)
        for c, p in enumerate((out / "stage" / i / "render" / "start_color.png", g / "cases" / i / "golden.png",
                               out / "stage" / i / "key.png")):
            if p.exists():
                im.paste(Image.open(p).convert("RGB").resize((W, H)), (c * W, y + 18))
        fl = (chk.get(i) or {}).get("flags", [])
        dr.text((4, y + 2), f"{i}   still | golden | NEW   flags: {'; '.join(fl)[:150] or 'none'}", fill=(255, 255, 0))
    im.save(out / "sheet.jpg", quality=82)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["freeze", "run", "variants"])
    ap.add_argument("book")
    ap.add_argument("stem")
    ap.add_argument("cases", nargs="*")
    ap.add_argument("--tag", default=time.strftime("%H%M"))
    ap.add_argument("--card", default="05")
    ap.add_argument("--env", action="append", default=[])
    ap.add_argument("--render-dir", default="render", help="run: the cases' still dir (e.g. render_lit)")
    ap.add_argument("--v", action="append", default=[], help="variants: name:strength[:bg strength]")
    a = ap.parse_args()
    if a.cmd == "freeze":
        freeze(Path(a.book), a.stem, a.cases)
    elif a.cmd == "variants":
        variants(Path(a.book), a.stem, a.tag, a.card, a.v)
    else:
        run(Path(a.book), a.stem, a.tag, a.card, a.env, a.render_dir)


if __name__ == "__main__":
    main()
