"""A synthetic book on disk for tests (10-27: looks, names and LoRAs have one owner, registry.Book over look.json,
so a test gives its people a real book instead of a cx.look lambda)."""
import json, tempfile
from pathlib import Path
from types import SimpleNamespace


def book(looks: dict, cast_labels: dict | None = None, loras=(), stem: str = "x", dress=None, places=None):
    """-> a cx-like namespace with .book (a temp book), .stem, .look(w, t) and .dress(w, t)."""
    b = Path(tempfile.mkdtemp(prefix="fakebook_"))
    vd = b / "work" / "video"
    vd.mkdir(parents=True)
    (vd / "look.json").write_text(json.dumps({"characters": dict(looks), "places": dict(places or {})}))
    (b / "book.cast.json").write_text(json.dumps({"characters": {k: (v or {}) for k, v in (cast_labels or {}).items()}}))
    for w in (looks if loras == "all" else loras):
        d = vd / "charlora" / "".join(c if c.isalnum() else "_" for c in w.lower()).strip("_")
        d.mkdir(parents=True, exist_ok=True)
        (d / "build.json").write_text(json.dumps({"lora": str(d / "l.safetensors"), "trigger": "chr_" + d.name,
                                                  "hash": "h_" + d.name}))
    return SimpleNamespace(book=b, stem=stem, look=lambda w, t=0: looks.get(w, ""),
                           dress=dress or (lambda w, t: ""))
