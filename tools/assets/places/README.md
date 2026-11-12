# places: a place spec's pieces

`pieces.py` turns the piece list of a place spec (place_spec/1: `{"id", "kind", "look", ...}` per piece) into
library assets. Lookup first, a generation request second; nothing here draws, shapes or writes into the library.

- `piece_kinds.json`: PIECE_KINDS, one file for the spec validator, the word-to-kind mapper, the lookup and the
  generation request (synonyms, parent kind, default size, the support to measure: seat or lie).
- `word_to_kind(phrase)`: a plan phrase to one kind name, mechanically; words no kind covers come back as
  `unresolved` for the language model's closed question.
- `lookup(piece, load_records())`: `[{asset_id, score, why}]`, best first. The kind must fit (the same kind, a
  narrower one, or the parent's records for a kind marked `role`); then look words, colour (the look's colour words
  against the record's stored albedo median when it has one, else its colour words) and size class. A scripted
  model scores lower than a generated one unless `allow_scripted`; a record whose remake passed its checks is
  hidden; equal scores go to a piece an accepted set already uses.
- `generation_request(piece)`: `{kind, look, size_m, needs}` and the dry-run plan of the existing prop route
  (reference drawing, source spec, picture-to-3D and the asset chain, the front, the support measure), each step
  with its gate and what a failure does (a new drawing with the next seed, a bounded number of times).
- `resolve(pieces, records)`: per piece `library`, `generate` or `unresolved`, and the counts.
- `submit(request, really=True)`: hands the drawing job to the line; refuses without the flag.

    python tools/assets/places/pieces.py PIECES.json [--library DIR] [--threshold T] [--allow-scripted]

Limits: records carry no measured colour yet (colour is matched on words until an `albedo_lab` field exists);
`prop_one.py` names its result `<id>_tr` and reads `<library>/prop/<id>/source/spec.json`, so the plan writes that
source spec before the shape step; a lie box has no measuring tool yet (the request states the need).
