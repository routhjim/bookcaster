# Scene tool: library set + library characters + cameras -> keyframe stills

```
python tools/assets/scenes/scene.py SCENE.json                  # resolve, one scheduler UE job, stills
python tools/assets/scenes/scene.py SCENE.json --cameras wide   # only some cameras
python tools/assets/scenes/scene.py SCENE.json --dry-run        # print the UE job
```

A scene puts characters from the library into a set from the library and photographs them with cameras whose
framing is computed from the characters' measured bodies. Stills are 1152x640, rendered in the asset scheduler's
resident editor (`run_asset.py --ue-job`; no editor of its own) with Lumen and hardware ray tracing.

| File | Side | What |
|---|---|---|
| `scene.py` | caller, stdlib | resolve ids, check the one-project rule, ray-tracing settings, submit, times, contact sheet |
| `framing.py` | both, stdlib | shots from a measured subject (close / medium / full), projection, frame checks |
| `ue/scene_job.py` | UE editor | load the set level, spawn and dress characters (the character tools' `char_render.py`, called), pose, measure, cameras, stills, attach checks; saves nothing |
| `ue/pose.py` | UE editor | body poses built from the skeleton: aim / rotate / two-bone IK / finger curl -> one-frame AnimSequence |

## One project

Sets, characters and props live in ONE UE project: the characters' (MetaHuman characters need that project's
plugins and their assembled Blueprints live there). Chosen because it is the cheapest move: a set is a script and
rebuilds anywhere in under a minute (the set builder's three master materials compile once per project, ~27 s),
and library props are imported on demand from their exported glTF by the set job (~0.8 s each), while moving a
MetaHuman means enabling its plugins in another project and migrating ~80 assets that the character tools keep
changing. So:
- build a set into the characters' project: `SETS_UE_PROJECT=<that .uproject> SETS_OUT=<library>/set
  python tools/assets/sets/build_set.py SPEC.json` (its library entry records the project);
- `scene.py` refuses a set built elsewhere and prints that command;
- `scene.py` writes the ray-tracing settings (`r.RayTracing`, `r.Lumen.HardwareRayTracing`, skin cache) into
  the project's `DefaultEngine.ini` when they are missing (`SCENES_UE_RAYTRACING=0`: leave it); the next editor
  start compiles the ray-tracing shaders once.

## Library ids

- Set: `<library>/set/<id>/asset.json` (written by the set builder): level path and project.
- Character: `<library>/character/<id>/character.json`, kept by whoever builds the character:
  `{"project", "blueprint", "garments": [...], "body_swap", "rebuild_from", "garment_material", "bp_deps"}`.
  A re-render picks up whatever those paths hold now.

## Spec

```jsonc
{
 "id": "scene_name",                       // output folder name
 "set": "<set library id>",
 "exposure_ev100": 7.0,                    // optional: overrides the set's for these stills (restored after)
 "characters": {
  "<key>": {"id": "<character library id>",
            "position": [x, y],            // set frame (SETS.md): metres, X east, Y north; the floor mark
            "facing": [dx, dy],            // plan direction the character faces
            "body_pose": {"pose": "stand_relaxed", "params": {...}}     // pose.py, or
                       | {"anim": "/Game/...AnimSequence", "t": s}       // a clip frame, or {"anim", "scan": "arms_up"}
            "face_pose": {"curves": {"CTRL_expressions_jawOpen": 0.3}} | {"anim", "t"}}
 },
 "lights": [ ... ],                        // extra lights, the set builder's point / spot / rect format
 "cameras": [
  {"name": "wide", "kind": "wide", "subject": "<key>", "position": [x, y, z], "target": [x, y, z] | "subject",
   "focal_mm": 22},
  {"name": "medium", "kind": "medium", "subject": "<key>", "focal_mm": 50, "az": 20, "el": 0, "screen_x": 0.57},
  {"name": "close", "kind": "close", "subject": "<key>", "focal_mm": 85, "az": 25, "el": 2, "screen_x": 0.57}
 ]
}
```

Several characters are several keys; each camera frames one `subject` (a two-shot is a `wide`/`full` camera with
an explicit position and target, checked against its subject).

**Computed cameras** (`close`, `medium`, `full`): the editor measures the posed character (eye line from the eye
joints, head top from the face mesh's reference bounds above the head joint, waist from pelvis/spine, feet from
the foot joints) and `framing.py` places the camera: close = 2.1 head heights with the eye line on the upper
third; medium = waist up with the eye line at 0.28 from the top; full = the whole figure with 12% margins.
`az` swings the camera around the subject from straight in front (positive: toward the subject's own left), `el`
raises it, `screen_x` puts the subject off centre (give the side he looks toward the room), the lens sets the
distance. Close and medium get a shallow depth of field focused on the eyes (f/2.8, f/4), wide f/8.

**Checks in every result** (`result.json`): where head top, eyes, chin, waist and feet land in each frame and the
problems a viewer notices first (head cut, feet out of a wide, subject facing away), the feet's height above the
floor, and the head attach check: the head and neck joints seen by the face skeleton and by the body skeleton
must agree to 1 mm at capture. The face follows a new body pose only while it is drawn and once the pose is
re-applied, so the job re-applies poses under a camera that sees the character and waits until the two agree
before it measures, and again before each still; a still where they still differ is marked `reject`.

**Poses** (`ue/pose.py`): named poses built from the character's own joints (`stand`, `stand_relaxed`,
`stand_hand_on_mantel`, `gesture_point`, `seated`, `sit_armchair_relaxed`) with parameters (seat and shelf
heights, weight leg, head and chest turns, arm-rest height and spread) plus raw `steps` in `params`:
`["aim", bone, child, [dx, dy, dz]]`, `["rot", bone, [axis], deg]`, `["ik", upper, lower, end, target, pole]`,
`["curl", side, deg, thumb_deg]`, `["move", bone, [dx, dy, dz]]`, all in the character's frame (x his left,
y forward, z up, centimetres). Garments follow through the character tools' leader pose.

Output: `<out>/<id>/` (`SCENES_OUT`, default `$ASSETS_UE_HOME/lab/scenes/out`): `<camera>.png`, `result.json`,
`times.json` (queue wait, editor time, per-step and per-still seconds), `job.json`, `sheet.png` (Pillow, pasting
only). `SCENES_CHAR_TOOLS` points at another copy of the character tools.
