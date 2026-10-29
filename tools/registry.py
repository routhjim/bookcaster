"""One owner for who and where (10-27 audit: five name resolvers, alias duplicates in look.json with DIFFERENT
looks, plans holding their own copies of the looks, places under several spellings - Watson never got a LoRA,
two Roylotts, two Baker Street bedrooms, Helen's lines moved Holmes's lips).

Every step asks the registry:
    R = registry.Book(book)
    R.person("Stoner")            -> "Helen Stoner"   (the cast name; aliases, Dr./Doctor, a unique surname or
                                                        first name, the narrator label, LoRA folder slugs)
    R.people()                    -> the cast, one name per person (look.json's first-stored spelling)
    R.look("Dr. Watson")          -> his one look (look.json)
    R.place("Baker Street Bedroom (inside)") -> "Baker Street Bedroom"   R.place_look(...) -> its one description
    R.style()                     -> the book's one art style
    R.lora("Dr. Watson")          -> (path, trigger, build hash) or (None, None, None)

look.json is the owner of every look and place description; nothing else is read for them."""
from __future__ import annotations

import json
import re
from pathlib import Path

DEFAULT_STYLE = ("detailed 19th-century book illustration, pen and ink linework with a soft watercolour wash, "
                 "muted period colours")
SIDE_WORD = re.compile(r"\s*\((inside|outside)\)\s*$", re.I)
TITLE = re.compile(r"^(?:the|mr|mrs|miss|ms|sir|lady|lord|dr|doctor|professor|captain|colonel|father|saint|st)\.?$",
                   re.I)


def person_key(name) -> str:
    """One key per person, whatever the spelling: 'Dr. Watson' = 'Doctor Watson' = folder 'doctor_watson'."""
    s = str(name or "").lower().replace("_", " ")
    s = re.sub(r"\bdr\b\.?", "doctor", s)
    return re.sub(r"[^a-z]", "", s)


def place_key(name) -> str:
    """One key per place: 'Baker Street Bedroom (inside)' = 'Baker Street Bedroom' = 'the baker street bedroom'."""
    s = SIDE_WORD.sub("", str(name or "")).lower()
    s = re.sub(r"^the\s+", "", s)
    return re.sub(r"[^a-z0-9]", "", s)


def side_of(name, desc: str = "") -> str:
    """'inside' / 'outside' as the place's own name ("X (inside)") or description ("Inside: ...") says, else ''."""
    m = SIDE_WORD.search(str(name or "")) or re.match(r"\s*(inside|outside)\s*:", str(desc or ""), re.I)
    return m.group(1).lower() if m else ""


def _words(name: str) -> list:
    return [w for w in re.findall(r"[A-Za-z][A-Za-z'\-]+", str(name or "")) if not TITLE.match(w)]


def book_of(path):
    """The book folder holding *path* (the book itself, work/video, a chapter, refs ...), or None."""
    p = Path(path).expanduser()
    for d in (p, *p.parents):
        if (d / "work" / "video").is_dir() or ((d / "work").is_dir() and (d / "chapters").is_dir()):
            return d
    return None


_AT: dict = {}


def at(path):
    """The registry of the book holding *path*, cached until its look.json or book.cast.json changes; None outside
    a book."""
    b = book_of(path) if path else None
    if b is None:
        return None
    sig = tuple((f.stat().st_mtime_ns, f.stat().st_size) if f.exists() else None
                for f in (b / "work" / "video" / "look.json", b / "book.cast.json"))
    hit = _AT.get(str(b))
    if hit and hit[0] == sig:
        return hit[1]
    R = Book(b)
    _AT[str(b)] = (sig, R)
    return R


class Looks(dict):
    """{name: look} whose keys are the given spellings (a chapter's cast) and whose lookups take ANY spelling of a
    person: looks.get("Dr. Watson") is Doctor Watson's look, "Stoner" in looks is Helen (10-27: chars.get(w, "")
    gave an empty look for every spelling but the stored one)."""
    def __init__(self, R: "Book", items):
        super().__init__(items)
        self.R = R

    def key(self, name):
        """The key here that is the same person as *name*, or None."""
        if dict.__contains__(self, name):
            return name
        return self.R.key_in(name, list(dict.keys(self)))

    def __contains__(self, name):
        return self.key(name) is not None

    def __getitem__(self, name):
        k = self.key(name)
        if k is None:
            raise KeyError(name)
        return dict.__getitem__(self, k)

    def get(self, name, default=None):
        k = self.key(name)
        return dict.__getitem__(self, k) if k is not None else default


class Book:
    def __init__(self, book):
        self.book = Path(book)
        self.vd = self.book / "work" / "video"
        lp = self.vd / "look.json"
        self.data = json.loads(lp.read_text()) if lp.exists() else {}
        chars = self.data.get("characters") or {}
        self._people, self._looks = [], {}
        for n, d in chars.items():                 # the first-stored spelling of each person owns their look
            k = person_key(n)
            if k not in self._looks:
                self._people.append(n)
                self._looks[k] = (n, d if isinstance(d, str) else str((d or {}).get("look") or d or ""))
        places = self.data.get("places") or {}
        self._places, self._entries = {}, {}
        for n, d in places.items():                # one description per place: the first non-empty one stored
            if not isinstance(d, str) or not d.strip():
                continue
            k = place_key(n)
            if k and k not in self._places:
                self._places[k] = (SIDE_WORD.sub("", n).strip(), d)
            if k:                                  # every description of it, in stored order, with its side
                self._entries.setdefault(k, []).append((n, d, side_of(n, d)))
        self._voices = {}
        self._narrator = self._narrator_name()
        self._voices = self._voice_map()

    # --- people ---------------------------------------------------------------------------------------------
    def _narrator_name(self):
        try:
            cast = json.loads((self.book / "book.cast.json").read_text())["characters"]
        except (OSError, ValueError, KeyError):
            return None
        n = next((n for n, c in cast.items() if str(c.get("same_as", "")).lower() == "narrator"), None)
        return self.person(n, narrator=False) if n else None

    def _voice_map(self) -> dict:
        """The voice cast's labels (book.cast.json: "Holmes", "Stoner", "Julia", ...) -> cast names, resolved
        together: a label naming one person takes them; a label naming several takes the one no other label has
        ("Julia" is Julia Stoner, so "Stoner" is Helen Stoner). The narrator's label is the narrator."""
        try:
            cast = json.loads((self.book / "book.cast.json").read_text())["characters"]
        except (OSError, ValueError, KeyError):
            return {}
        out, open_ = {}, {}
        for lab, c in cast.items():
            if str(c.get("same_as", "")).lower() == "narrator" and self._narrator:
                out[lab] = self._narrator
                continue
            words = {w.lower() for w in _words(lab)}
            cands = [n for n in self._people if person_key(n) == person_key(lab) or
                     (words and words <= {w.lower() for w in _words(n)})]
            if len(cands) == 1:
                out[lab] = cands[0]
            elif cands:
                open_[lab] = cands
        for lab, cands in open_.items():
            left = [n for n in cands if n not in out.values()]
            if len(left) == 1:
                out[lab] = left[0]
        return out

    def people(self) -> list:
        return list(self._people)

    def narrator(self):
        return self._narrator

    def person(self, name, default=None, narrator: bool = True):
        """The cast name for any spelling of a person, or `default` (never a guess between two people)."""
        s = str(name or "").strip()
        if not s:
            return default
        if narrator and s.lower() in ("narrator", "the narrator", "i", "me"):
            return self._narrator or default
        k = person_key(s)
        if k in self._looks:
            return self._looks[k][0]
        if s in self._voices:              # a voice label ("Stoner" in the timeline: Helen Stoner)
            return self._voices[s]
        words = [w.lower() for w in _words(s)]
        if not words:
            return default
        # every word of the given name in one person's name, and in nobody else's ("Stoner" alone: Helen or Julia?
        # -> no answer; "Helen Stoner" / "Miss Helen" -> Helen)
        hits = [n for n in self._people if set(words) <= {w.lower() for w in _words(n)}]
        return hits[0] if len(hits) == 1 else default

    def same(self, a, b) -> bool:
        pa, pb = self.person(a), self.person(b)
        return pa is not None and pa == pb

    def key_in(self, name, keys):
        """The one key of *keys* (a plan's cast, a wardrobe, a pin list) that is the same person as *name*: the
        exact spelling, else the same person_key, else the same registry person; None when none or two match."""
        keys = list(keys or [])
        if name in keys:
            return name
        hits = [k for k in keys if person_key(k) == person_key(name)]
        if not hits and self.person(name) is not None:
            hits = [k for k in keys if self.same(k, name)]
        if len({person_key(self.person(k) or k) for k in hits}) != 1:
            return None
        p = self.person(name)                  # several spellings of the one person: the registry's own first
        return p if p in hits else hits[0]

    def aliases(self, name) -> list:
        """Every look.json spelling of this person (the first stored first)."""
        p = self.person(name)
        return [n for n in (self.data.get("characters") or {}) if p and self.person(n) == p]

    def looks(self, names=None, fallback=None) -> "Looks":
        """{name: the registry's look} for *names* (default: the cast); a name the registry does not know keeps
        *fallback*'s look (a plan's own copy) or ''."""
        fb = fallback or {}
        names = self.people() if names is None else list(names)
        return Looks(self, {n: (self.look(n) if self.person(n) else str(fb.get(n) or "")) for n in names})

    def look(self, name) -> str:
        p = self.person(name)
        return self._looks[person_key(p)][1] if p else ""

    def slug(self, name) -> str:
        p = self.person(name) or str(name or "")
        return re.sub(r"[^a-z0-9]+", "_", p.lower()).strip("_")[:50] or "x"

    def lora(self, name):
        """(path, trigger, build hash) of the person's LoRA, found under any spelling's folder."""
        cl = self.vd / "charlora"
        p = self.person(name) or str(name or "").strip()     # (a person look.json lacks: their own spelling's folder)
        if not p or not cl.exists():
            return None, None, None
        for d in [cl / self.slug(p)] + sorted(x for x in cl.iterdir() if x.is_dir()):
            if person_key(d.name) == person_key(p) and (d / "build.json").exists():
                b = json.loads((d / "build.json").read_text())
                return b.get("lora"), b.get("trigger"), b.get("hash") or b.get("built")
        return None, None, None

    # --- places ---------------------------------------------------------------------------------------------
    def place(self, name, default=None):
        """The one name of a place, without '(inside)/(outside)'; a place not yet described keeps its own name."""
        s = str(name or "").strip()
        if not s:
            return default
        k = place_key(s)
        return self._places[k][0] if k in self._places else SIDE_WORD.sub("", s).strip()

    def _entry(self, name):
        """(key, description) of the place's one description: the first stored; for a name that says its side ("X
        (outside)") the first stored FOR that side, else the first with no side (10-27: a mill described inside AND
        outside must not draw its outside view from the inside's words)."""
        es = self._entries.get(place_key(name)) or []
        sd = side_of(name)
        fit = sorted((e for e in es if not sd or e[2] in ("", sd)), key=lambda e: bool(sd) and e[2] != sd)   # its side first
        e = (fit or es or [None])[0]
        return (e[0], e[1]) if e else (None, "")

    def place_look(self, name) -> str:
        # one side word at the front (10-29: "Outside: Outside: Outside: Inside: A dark oak writing desk ..." - each run
        # wrote its side word in front of the stored one; the last written is the side it was described from)
        d = self._entry(name)[1]
        m = SIDE_PREFIX.match(d or "")
        return (re.findall(r"(inside|outside)", m.group(0), re.I)[-1].capitalize() + ": " + d[m.end():]) if m else d

    def place_entry(self, name):
        """The look.json key whose description place_look(name) returns, or None."""
        return self._entry(name)[0]

    def places(self) -> list:
        return [n for n, _ in self._places.values()]

    def place_key_in(self, name, keys):
        """The key of *keys* that is the same place as *name* (the exact spelling first), or None."""
        keys = list(keys or [])
        if name in keys:
            return name
        hits = [k for k in keys if place_key(k) == place_key(name)]
        return hits[0] if hits else None

    # --- style ----------------------------------------------------------------------------------------------
    def style(self, stem: str | None = None) -> str:
        """The book's one art style: `bookcaster art` (art_style.txt), else the chapter's style.txt, else the
        default - the same words for keys, place pictures, sheets and LoRA captions (10-27: keys read only
        art_style.txt, absent for the Speckled Band, and were drawn photographic while every sheet and LoRA was
        pen-and-ink watercolour)."""
        a = self.book / "art_style.txt"
        if a.exists() and a.read_text().strip():
            return a.read_text().strip()
        cands = [self.vd / stem / "style.txt"] if stem else []
        cands += sorted(self.vd.glob("[0-9][0-9]_*/style.txt"))
        for f in cands:
            if f.exists() and f.read_text().strip():
                return f.read_text().strip()
        return DEFAULT_STYLE

    def ref_style(self, stem: str | None = None) -> str:
        """The REFERENCE style: what the character sheets are drawn in and their LoRAs captioned with - the chapter's
        own style.txt, else the default; never `bookcaster art` (10-27 user: back to last night's realistic keys
        without retraining the LoRAs - a LoRA carries who someone is, the picture's style is the key's)."""
        cands = [self.vd / stem / "style.txt"] if stem else []
        cands += sorted(self.vd.glob("[0-9][0-9]_*/style.txt"))
        for f in cands:
            if f.exists() and f.read_text().strip():
                return f.read_text().strip()
        return DEFAULT_STYLE

    def picture_style(self) -> str:
        """The PICTURE style of the shots flow's keys and place pictures: `bookcaster art` (art_style.txt), else none -
        the image model's own photographic look (the Speckled Band 10-26 keys the user chose 10-27)."""
        a = self.book / "art_style.txt"
        return a.read_text().strip() if a.exists() else ""


SIDE_PREFIX = re.compile(r"^\s*(?:(?:inside|outside)\s*:\s*)+", re.I)


def place_slot(places: dict, name, desc: str = ""):
    """Where a place goes in a {name: description} store -> (key, already described?). The same place (place_key)
    already described for this side, or with no side, is that entry; a place described only for the OTHER side
    ("The Mill (inside)" stored, "The Mill (outside)" asked) is a place of its own under its own name; a new place
    is stored under its one name, without "(inside)"/"(outside)" (10-27: places were stored under every new spelling -
    two Baker Street bedrooms)."""
    pk, sd = place_key(name), side_of(name, desc)
    same = [(k, side_of(k, v)) for k, v in (places or {}).items()
            if place_key(k) == pk and isinstance(v, str) and v.strip()]
    k = next((k for k, s in sorted(same, key=lambda x: bool(sd) and x[1] != sd) if not sd or s in ("", sd)), None)
    if k is not None:
        return k, True
    if same or any(place_key(k) == pk for k in places or {}):
        return str(name).strip(), False
    return SIDE_WORD.sub("", str(name)).strip() or str(name), False


def write_place(book, name, desc, overwrite: bool = False) -> str:
    """Store a place description in look.json in its place_slot; a stored description stays unless *overwrite*.
    -> the name it is stored under."""
    lp = Path(book) / "work" / "video" / "look.json"
    data = json.loads(lp.read_text()) if lp.exists() else {}
    places = data.setdefault("places", {})
    key, known = place_slot(places, name, desc)
    if overwrite or not known:
        places[key] = desc
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(json.dumps(data, indent=1, ensure_ascii=False))
    return key
