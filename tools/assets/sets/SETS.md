# Set builder: set spec -> UE level + validation stills

```
python tools/assets/sets/build_set.py tools/assets/sets/specs/hangar_bay_01.json            # plan, check, build, 5 stills
python tools/assets/sets/build_set.py SPEC.json --render-only [--cameras wide_south,low]    # re-render the saved level
python tools/assets/sets/build_set.py SPEC.json --plan-only                                 # expand + plan checks, no UE
python tools/assets/sets/build_set.py --stop-editor
```

A set is a whole room or hall, built by script from a JSON spec, out of the engine's basic shapes (cube, cylinder,
sphere, scaled) with procedural world-aligned materials, plus library assets (`<library>/manifest.json`) as
fixtures and props. Nothing downloaded, scanned or from engine samples. The level is `/Game/Sets/<id>/L_<id>`;
a rebuild empties and refills it, so the map always equals the spec.

| File | Side | What |
|---|---|---|
| `build_set.py` | caller, stdlib | validate, expand, plan checks, start/feed the set editor, write `<out>/<id>/` |
| `layout.py` | caller, stdlib | spec -> explicit primitives / lights / assets / cameras in UE units, plan checks |
| `presets.py` | caller, stdlib | material presets -> master + parameters |
| `ue/set_server.py` | UE editor | job loop: build (via `set_build.py`), spawn cameras + 1.8 m figures, stills |
| `ue/set_build.py` | UE editor | empties/creates the level, spawns everything, saves |
| `ue/set_materials.py` | UE editor | three master materials (made once) and the set's material instances |
| `specs/*.json` | | the two proof sets |

Output: `<out>/<id>/asset.json` (manifest entry, class `set`), `spec.json`, `scene.json`, `ue/<id>_<camera>.png`,
`logs/` (`times.json`, `ue_result.json`). `<out>` is `<library>/set` when the project is the chain's, else
`$ASSETS_UE_HOME/lab/sets` (`SETS_OUT` overrides). Library index: `<library>/manifest.json` gets a `set` entry.

## Frame and units (no hidden conventions)

- Metres. **X = east, Y = north, Z = up.** Origin = centre of the room's floor; Z = 0 is the floor's top surface.
  The room spans X in [-W/2, W/2], Y in [-D/2, D/2] for `size_m: [W, D, H]`.
- Walls are named by the side they close: `north` (Y = +D/2), `south`, `east` (X = +W/2), `west`. A wall's
  inner face lies on the room boundary; its thickness goes outward.
- Positions along a wall (`from_m`, `to_m`, `at_m`) are the X coordinate for north/south walls and the Y
  coordinate for east/west walls (absolute, not relative to the wall's end).
- `facing: [dx, dy]` is the plan direction the front of a library asset points to (assets are modelled front
  toward -Y, so `[0, -1]` is "as modelled"). Example: a machine against the west wall facing into the room has
  `facing: [1, 0]`.
- Colours: sRGB hex. Light: physical units (lux for the sun, lumens for lamps, nits for glowing surfaces,
  kelvin for colour temperature). Exposure: `exposure_ev100` (manual, fixed: auto exposure wrecks dark scenes).
- UE: x = 100 X, y = -100 Y, z = 100 Z (centimetres; the mirror the asset chain uses).

## Spec

```jsonc
{
 "id": "hangar_bay_01",                 // lowercase, digits, underscore; never a book or character name
 "description": "...",                  // the place as the book describes it
 "style": "...",                        // free text: era, wear, palette, mood (for the writer; not parsed)
 "tags": [...],
 "size_m": [30, 60, 14],                // W (X), D (Y), H (Z): the room's inside
 "exposure_ev100": 8.0,                 // ~6 dim interior, 8 lit hall, 9-10 bright interior, 12+ daylight exterior
 "materials": {"name": {"preset": "...", ...overrides}},          // see Materials
 "structure": [ {"kind": ...}, ... ],                              // see Structure kinds
 "lights":    [ {"type": ...}, ... ],                              // see Lights
 "fixtures":  [ placement, ... ],      // fixed things (scene-record "fixtures"): machines, desks, a parked ship
 "props":     [ placement, ... ],      // free props
 "support_surfaces": [{"z_m": 0.9}],   // optional: heights props may stand on that are not library assets
 "validation": {"figures": [{"position": [x, y], "facing": [dx, dy]}]},   // 1.8 m reference figures (stills only)
 "cameras": [{"name", "position": [x, y, z], "target": [x, y, z], "focal_mm"}]   // 36 mm wide sensor
}
```

**Placement** (fixtures, props): `{"asset": "<library id>", "name": "...", "position": [x, y] or [x, y, z],
"facing": [dx, dy]}`, or a row: `{"asset", "name", "facing", "row": {"from": [x, y], "to": [x, y], "count": n,
"z_m": 0}}` (evenly spaced, both ends included). `z` is the height of the asset's base (0 = on the floor).
A prop standing on another placed item: `"on": "<that item's name>", "on_z_m": <its surface height>` (the plan
checks that it lies fully on the host and ignores their overlap).

### Structure kinds

| kind | keys |
|---|---|
| `floor` | `material`, `thickness_m` (0.4), `extend_m` (1.0: slab runs under the walls) |
| `ground` | `material`, `extent_m`: outdoor ground around the building (what windows look down on) |
| `wall` | `side`, `material`, `thickness_m` (0.3), `height_m` (H), `dado: {height_m, material}` (lower band), `openings: [...]`, `columns: [{at_m: [...], depth_m, width_m, material, height_m}]` (I-section pilasters), `trim: [{z_m, height_m, depth_m, material, from_m, to_m}]` (ledges) |
| opening `window_band` / `glass_wall` | `from_m`, `to_m`, `sill_m`, `head_m`, `bay_m` (mullion spacing), `panes: [cols, rows]` per bay, `mullion_m: [width, depth]`, `bar_m`, `frame_material`, `glass_material`, `brace_bays: [bay indices]`, `brace_z_m: [z0, z1]`, `brace_material` |
| opening `door` | `from_m`, `to_m`, `sill_m`, `head_m`, `panel_m: [w, h]` (frame grid), `material` (leaf), `frame_material`, `brace_material`, `braced` (X brace per cell, default true), `frame_m`, `header_m: [h, depth]` |
| `trusses` | `span_axis` (`x`/`y`), `spacing_m`, `bottom_m`, `top_m`, `rise_m` (pitched top chord), `chord_m: [depth, width]` (I-section), `web_m`, `panel_m`, `material`, `columns` (I-columns under each end, default true), `column_m`, `purlins: {spacing_m, size_m: [d, w]}`, `bottom_ties: {at_m: [...]}`, `roof_bracing: {every}`, `deck: {material, thickness_m}` |
| `ceiling` | `height_m`, `material`, `thickness_m`: flat slab |
| `grid_ceiling` | `height_m`, `cell_m: [x, y]`, `bar_m: [w, d]`, `x_m`, `y_m` (extent), `bar_material`, `panel_material`, `plenum: {top_m, material}` |
| `platform` | `name`, `from: [x, y]`, `to: [x, y]`, `height_m` (walking surface), `thickness_m`, `material`, `edge_material`, `supports: {spacing_m, lines: [north/south/east/west], size_m, material}`, `railing: {sides: [...], height_m, post_spacing_m, material}` |
| `pipe_run` | `name`, `points: [[x, y, z], ...]`, `diameter_m`, `material`, `hangers: {spacing_m, to_z_m, material}`, `drops: {at: [[x, y], ...], to_z_m, diameter_m}` |
| `duct_run` | `name`, `points`, `size_m: [w, h]`, `material` (collars every 1.5 m) |
| `floor_lines` | `lines: [{points: [[x, y], ...], width_m, material, closed, thickness_m, z_m}]` (paint, inlaid rails) |
| `box` | `name`, `base: [x, y, z]` (centre of the bottom face), `size_m: [x, y, z]`, `material`, `yaw_deg` |

Big boxes are split into tiles of at most 8 m so Lumen's surface cache keeps its resolution.

### Materials

`{"preset": name, ...overrides}`. Presets: `worn_concrete`, `painted_steel`, `bare_steel`, `galvanized_steel`,
`stainless_steel`, `epoxy_floor`, `white_panel`, `paint_line`, `rubber`, `plastic` (opaque, procedural),
`glass`, `frosted_glass` (see-through), `emissive` (lamp faces). Overrides: `color`, `color2`, `variation`,
`roughness: [a, b]`, `metallic`, `grime` (0..1), `grime_color`, `streaks` (vertical grime), `panel_m: [x, y, z]`
(joint grid; 0 = none), `joint_m`, `joint_darken`, `feature_m` (patch size); glass `tint`, `opacity`,
`roughness`; emissive `color`, `nits`. Every material is world-aligned (no UVs): colour patches, mottle, grime,
roughness patches and joint lines are the same size on a 60 m wall and on a 10 cm bar.

### Lights

| type | keys |
|---|---|
| `sun` | `toward_sun: [dx, dy, dz]` (direction from the scene to the sun), `lux`, `kelvin`, `volumetric` (shaft strength) |
| `sky` | `intensity` (sky light scale; atmosphere on) |
| `fog` | `density`, `falloff`, `volumetric` (light shafts), `extinction`, `distance_m` |
| `strip_lights` | `z_m` (lamp face), `length_m`, `along` (`x`/`y`), `lumens` (each), `kelvin`, `radius_m`, `shadows`, `hang_to_m` (rods up to), `fixture_material`, `tube_material` (emissive), `rows: [{from, to, count}]` |
| `point` / `spot` / `rect` | `position`, `lumens`, `kelvin` or `color`, `radius_m`, `toward` (spot/rect), `size_m: [w, h]` (rect), `cone_deg` (spot), `shadows` |

Every strip-light fixture is a housing + an emissive face + a rect light of the same size (the lamp you see is
the light that lights). Lumen global illumination and reflections, hardware ray tracing when the project has
it (the chain's `DefaultEngine.ini`), manual exposure from `exposure_ev100` (bias = -EV100), hit-lighting
reflections, quality 2. `post: {bloom, vignette, saturation, contrast, lumen_*}` tunes the post volume.

## Checks

- Spec: required keys, id form, every material used is defined, every asset is in the library and imported.
- Plan (stdlib, before UE): each fixture/prop inside the room, not taller than it, no two overlapping
  (footprints rotated by `facing`), nothing with z > 0 unless an asset or `support_surfaces` entry is right under
  it (no floating props). Problems are printed and stored in `asset.json` (`plan_problems`, `checks_passed`).
- Stills: five cameras per set at 1152x640 (two wide from opposite ends, one medium at eye height, one low, one
  from above) with 1.8 m reference figures (the template mannequin, validation only, not saved in the level).

## Writing a spec from a place description (for a language model)

What the writer must decide, explicitly, because nothing is inferred:
1. The room's inside size `[W, D, H]`, and which wall is which (pick north = the far end of the main view).
2. For each wall: material, what openings it has (absolute from/to along the wall, sill and head heights).
3. The roof/ceiling kind (trusses or flat or grid), its heights, and the services (pipes, ducts) with points.
4. Every fixture/prop by library id with position and facing, keeping aisles clear; heights for stacked props.
5. Light sources that exist in the place (sun through which windows, strip lights with counts and lumens,
   practicals) and an EV100 that fits them.
6. Five cameras that show the place from opposite ends, eye height, low and high.
The plan checks catch overlaps, out-of-room and floating items before any GPU time; the writer re-asks itself
with the printed problems.
