# Asset contract (v1)

Every asset that enters the library obeys these rules. `bl_harness.py` applies the mechanical ones and checks the
rest; `run_asset.py` records the result in the asset's manifest entry. Verified on Blender 4.5 LTS and UE 5.8.1.

## Units, axes, origin

| | Blender (source) | glTF export | UE (after Interchange import) |
|---|---|---|---|
| Unit | 1 unit = 1 m, unit scale 1.0 | metres | centimetres (x100, done by the importer) |
| Up | +Z | +Y | +Z |
| Asset front | faces -Y (Blender's Front view) | +Z | faces **+Y** |
| Asset right (seen from the front) | +X | +X | +X |

Measured with an asymmetric probe box (x 0..1, y -2..0, z 0..3 m in Blender -> x 0..100, y 0..200, z 0..300 cm in
UE) for glTF, FBX and USD alike: X and Z carry over, Y flips (right- to left-handed), scale is exact.

- **Origin:** floor props (`placement: floor`, the default) have the centre of the bounding-box floor at the origin:
  centred in X and Y, lowest point at Z = 0. The harness moves the mesh there. Held props (`placement: held`, for
  later) put the grip at the origin with the handle along +X, as the repo's props library already does.
- **Transforms:** applied (location 0, rotation 0, scale 1) before export. One mesh object per static asset.
  Modifiers are applied first, while every object they reference (empties for Mirror/Array, hidden boolean
  cutters) still exists; parented parts keep their world placement; then non-part objects are removed.
- **Size:** `size_m` = [x, y, z] in metres = [width left-right, depth front-back, height]. The built mesh must match
  each axis within `size_tolerance` (default 15 %).

## Naming

- `id`: lowercase letters, digits, underscore; starts with a letter; 3-64 characters; unique in the library;
  `<thing>_<variant>_<nn>` (e.g. `anvil_london_01`). Never a book or character name: assets are reusable.
- Blender object and mesh: `SM_<id>`. Material: `M_<id>`. Textures: `T_<id>_BaseColor`, `T_<id>_Normal`,
  `T_<id>_ORM`.
- UE content path: `/Game/Assets/<Class>s/<id>/` (Interchange makes `StaticMeshes/`, `Materials/`, `Textures/`
  inside); the mesh is renamed `SM_<id>`.

## Materials and textures

- Model scripts may build any Cycles-renderable node material (procedural is fine) on Principled BSDF. Materials
  do not transfer as nodes: the harness **bakes** every material into one PBR texture set and one material:
  - `T_<id>_BaseColor` sRGB, `T_<id>_Normal` linear, tangent space, **OpenGL (+Y green)** as glTF defines it; UE's
    glTF import sets `flip_green_channel` and `TC_NORMALMAP` (checked on import).
  - `T_<id>_ORM` linear: R = ambient occlusion, G = roughness, B = metallic (the glTF packing; UE reads it as is).
- Texture size by tier: hero 2048, mid 1024, background 512 (override with `texture_px`). Bake margin res/128 px.
  - `T_<id>_Emissive` sRGB, only when a material emits (Principled Emission Color x Strength, either linked or
    non-zero): baked to a float image, stored normalised to its peak, the peak written as the material's emission
    strength (glTF `emissiveFactor` + `KHR_materials_emissive_strength`; UE's glTF master material reads both).
- **Glass**: a material slot is glass when the spec lists it in `glass_materials`, the material has the custom
  property `asset_glass`, or its Principled Transmission Weight is >= 0.5 (unlinked). Glass is not baked: its faces
  go to a second material `M_<id>_Glass` with the first glass slot's constant base colour, roughness, IOR and
  transmission, exported as `KHR_materials_transmission` (Interchange: its Transmission master material,
  translucent). One mesh, two slots. Nanite does not draw translucent sections, so an asset with glass imports with
  Nanite off. Tinted dark glass that must hide an interior: a dark base colour (transmission tints by it).
- **Painted maps** (optional, `run_asset.py --paint`, see README): a second set `textures/painted/T_<id>_BaseColor
  / Normal / ORM.png` and `export/<id>_painted.glb` on the same UVs; the procedural set is never overwritten. The
  spec's `texture_set` (`painted` | `procedural`) picks the set imported into UE (default: painted when this run
  painted and `paint.use` is not false). UE texture names stay `T_<id>_<map>` either way.
- Shader coordinates are preserved: the harness stores every part's Object, Generated and world-Position
  coordinates as vertex attributes before applying transforms, joining and recentring, and rewires Texture
  Coordinate (Object, Generated) and Geometry (Position) links to them, so position-based masks and decals bake
  where the model script placed them.
- Masked (alpha-cut) materials are not covered by v1.

## UVs

- Default: the harness replaces UVs with a Smart UV Project (60 degrees) + pack into 0-1, used by the bake and
  by UE. `keep_uvs: true` keeps the model's own UV map (must be non-overlapping and inside 0-1).
- UE: Nanite meshes still need UVs; lightmap UVs are not needed (Lumen, no baked lighting).

## Geometry checks (all recorded; any failure sets `checks_passed: false`)

| Check | Rule |
|---|---|
| size | each axis within tolerance of `size_m` |
| triangles | <= tier budget: hero 250 000, mid 60 000, background 8 000 |
| materials | every face on a slot with a material |
| loose_geometry | no vertex or edge outside a face |
| normals_consistent | no edge whose two faces wind the same way |
| normals_outward | signed volume > 0 (not inside out) |
| degenerate_faces | no zero-area faces |
| floating_parts | every part off the floor touches (bounding box within 2 mm) another part |
| uvs | present and inside 0-1 |

N-gons are triangulated before export (tangents need tris or quads). Separate intersecting parts are allowed.

## UE import

glTF (`.glb`, textures embedded) through Interchange, by the no-GPU commandlet. Nanite on for static opaque
meshes (all v1 assets). Material: an instance of Interchange's glTF master material with the three textures.
FBX is reserved for skeletal meshes and morph targets (later); see README for the format test.

## Tiers

| Tier | Use | Triangles | Texture |
|---|---|---|---|
| hero | in close-up, held, story objects | <= 250k | 2048 |
| mid | in medium shots, furniture | <= 60k | 1024 |
| background | wide shots, set dressing | <= 8k | 512 |

## Rigged assets (reserved, not built in v1)

Manifest field `rig` (null for static assets) will carry: `skeleton` ("ue_mannequin" | "metahuman" | "custom"),
`ue_skeleton_path`, `bones` (root, and any named sockets such as `grip`, `hinge`), `morph_targets` (names),
`physics_asset`, `export_format` ("fbx"), `bind_pose` ("A" | "T" | "rest"). Skeletal assets will export FBX.

## Manifest entry (`<library>/<class>/<id>/asset.json`; index in `<library>/manifest.json`)

`id`, `class` (prop | vehicle | set | outfit | creature | character), `description`, `tags`, `tier`, `size_m`,
`placement`, `source` (scripted | library | generated) and `licence`, `contract_version`, `blender` (version,
source `.blend`, model script, measured `dims_m`, `tris`, `parts`, `texture_px`), `export` paths, `textures`,
`ue` (`path` of the static mesh, materials and textures, `nanite`, `bounds_cm`, import problems),
`thumbnails` (Blender contact sheet, UE stills), `checks` (name -> pass), `checks_passed`, `check_details`, `rig`,
`render` (card, GPU, ray-tracing state, console variables, time), `times_s`, `updated`.
Paths inside an entry are relative to the asset folder.
