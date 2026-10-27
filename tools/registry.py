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


def _words(name: str) -> list:
    return [w for w in re.findall(r"[A-Za-z][A-Za-z'\-]+", str(name or "")) if not TITLE.match(w)]


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
        self._places = {}
        for n, d in places.items():                # one description per place: the first non-empty one stored
            if not isinstance(d, str) or not d.strip():
                continue
            k = place_key(n)
            if k and k not in self._places:
                self._places[k] = (SIDE_WORD.sub("", n).strip(), d)
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

    def look(self, name) -> str:
        p = self.person(name)
        return self._looks[person_key(p)][1] if p else ""

    def slug(self, name) -> str:
        p = self.person(name) or str(name or "")
        return re.sub(r"[^a-z0-9]+", "_", p.lower()).strip("_")[:50] or "x"

    def lora(self, name):
        """(path, trigger, build hash) of the person's LoRA, found under any spelling's folder."""
        cl = self.vd / "charlora"
        p = self.person(name)
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

    def place_look(self, name) -> str:
        k = place_key(name)
        return self._places[k][1] if k in self._places else ""

    def places(self) -> list:
        return [n for n, _ in self._places.values()]

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
