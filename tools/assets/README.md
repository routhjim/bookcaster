# Asset chain: Blender model script -> checked asset -> UE5 still (M1)

One command takes an asset spec and a Blender model script through the whole chain, unattended:

```
python tools/assets/run_asset.py tools/assets/examples/anvil/anvil.json tools/assets/examples/anvil/anvil.py
python tools/assets/run_asset.py SPEC.json MODEL.py --preview      # modelling loop: shape + checks + quick sheet
python tools/assets/run_asset.py --stop-editor                       # ask the long-lived UE editor to quit
```

Stdlib Python 3 only on the calling side. Several runs may go at once (see Concurrency).

| Step | Where | What |
|---|---|---|
| 1 Blender harness (`bl_harness.py`) | Blender 4.5, Cycles on a GPU | runs the model script in an empty scene; applies the contract (one mesh `SM_<id>`, floor-centre origin, transforms applied); checks; Smart-UV; bakes materials to BaseColor / Normal / ORM (+ Emissive when a material emits; the normal pass is skipped when no material bends the normal); glass slots kept as a transmissive second material; 8-view contact sheet (512 px tiles framed on the bounds, 32 spp + denoise); exports `.glb`; saves the `.blend` |
| 2 UE import (`ue/ue_import.py`) | inside the render editor (or `UnrealEditor-Cmd -nullrhi` with `--skip-render`) | Interchange glTF import to `/Game/Assets/<Class>s/<id>/`, mesh renamed `SM_<id>`, Nanite on, texture settings checked |
| 3 Stage (`ue/ue_stage.py`) | commandlet, once | builds `/Game/AssetStage/L_AssetStage` (floor, physical sky, sun, sky light, fog, post volume with fixed exposure) if missing; `--rebuild-stage` remakes it |
| 4 UE stills (`ue/ue_render.py`) | editor, `-RenderOffscreen`, one XTX | asset at the origin, 1.8 m box and the template mannequin beside it; a camera preset per class at 1152x640 (prop: wide / medium / close / back / detail; vehicle: wide / low medium / close / low rear 3/4 back / side; set; creature) plus the spec's own `cameras`, Lumen; waits counted in frames (90 after set-up, 30 per camera) after shaders and streaming finish |

**Modes.** `--preview` (2.6 s on the anvil): model, contract, checks and a 320 px, 12 spp sheet of the *unbaked*
materials (`sheet/<id>_preview.png`); no bake, export or UE. Default (final): everything.

**Ray tracing.** Hardware ray tracing is the default in both tools: `run_asset.py` writes `r.RayTracing`,
`r.Lumen.HardwareRayTracing` and the skin-cache shader setting into the project's `DefaultEngine.ini` before it
starts an editor (`ASSETS_UE_RAYTRACING=0` writes them off: software Lumen), and Cycles runs with HIP RT
(`--hiprt 0` or `ASSETS_HIPRT=0` turns it off). Cost: the first editor start after a switch compiles shaders
(~67 s measured in a copy project); later fresh starts ~24 s instead of ~15 s. The long-lived editor pays this
once. The stills record `raytracing_metadata` and the ray-tracing console variables.

**Long-lived editor.** The first final run starts a UE editor that stays up, holds its card's lock, and takes
later jobs from `<project>/Saved/AssetServer/queue/`; it quits after `ASSETS_UE_IDLE_S` (300) without work or on
`--stop-editor`. A warm editor does import + 4 stills in ~4.5 s instead of ~15 s. `--fresh-editor` starts and
quits an editor for one asset instead. After editing `ue/*.py`, stop the editor so the next run loads the new code.

Output: `<library>/<class>/<id>/` with `asset.json` (manifest entry), `source/` (model script, spec, `.blend`),
`export/`, `textures/`, `sheet/<id>_sheet.png`, `ue/<id>_<shot>.png`, `logs/`; `<library>/manifest.json` indexes
all assets. The rules are in [CONTRACT.md](CONTRACT.md).

**Cameras.** The spec may set `camera_preset` (default: its class) and `cameras`: a list of `{"name", "position":
[x, y, z], "target": [x, y, z], "focal": mm}` in the asset's Blender frame (metres, front -Y, base on Z = 0); each
adds a still `ue/<id>_<name>.png`.

**Painting (optional, `--paint`).** `paint.py` (also runnable alone on a finished asset folder) renders guide views
of the baked asset, has Qwen-Image-2.1 repaint each (img2img from the guide at low strength with the object mask
and depth ControlNet), divides the guide's shading out, projects the paintings back per texel and writes a second
map set; see the docstring in `paint.py` and "Painted maps" in CONTRACT.md. Spec block `paint` (prompt, views,
close-up region, strength, projection parameters) and `texture_set`.

Other flags: `--preview-px`, `--preview-samples` (preview sheet size and samples), `--sheet-samples`, `--no-sheet`
(final pass without the Blender sheet when only the UE stills matter), `--skip-render` (Blender + import commandlet only), `--source-sheet` (also render the unbaked model),
`--formats glb,fbx,usd`, `--device auto|xtx|igpu|cpu`, `--hiprt 0|1` (Cycles HIP RT, default on), `--warm-frames`, `--settle-frames`,
`--warm-s`, `--settle-s`, `--ue-timeout`.

## Devices

| Step | Device |
|---|---|
| model script, contract, checks, UV unwrap, glTF export, manifest | CPU (all under 1 s) |
| bake, contact sheet | Cycles HIP on one XTX under its lock (Blender prefers card 05, UE card 65); preview prefers the idle iGPU; overflow to the iGPU if both locks are held and it is idle, else CPU |
| UE import | CPU inside the editor |
| UE stills | one XTX (prefers 65), Vulkan SM6, Lumen with hardware ray tracing (`r.RayTracing`, `r.Lumen.HardwareRayTracing`; `ASSETS_UE_RAYTRACING=0` for software Lumen) |

## Concurrency

- Blender runs are independent (own asset folder, own job files); one per card plus the iGPU plus CPU.
- Only one UE editor per project: two editors saving into one project can corrupt it. Starting an editor, the
  stage build and the import commandlet go through `<project>/Saved/AssetServer/project.lock`; jobs from parallel
  runs queue in the one long-lived editor and run one after another.
- Two runs of the *same* asset id at once are not supported (same folder).

## Settings

Environment variables, documented in `config.py`: `ASSETS_BLENDER` (default `$BLENDER`, else
`~/opt/blender/current/blender`, 4.5 LTS), `ASSETS_UE_HOME` (`~/ue`), `ASSETS_UE_ENGINE`, `ASSETS_UE_PROJECT`
(`$ASSETS_UE_HOME/proj/AssetLab/AssetLab.uproject`), `ASSETS_LIBRARY` (`$ASSETS_UE_HOME/library`), `ASSETS_LOGS`,
`ASSETS_GPU_CARDS` (`65,05`), `ASSETS_UE_CARDS` (`65,05`), `ASSETS_BLENDER_CARDS` (`05,65`),
`ASSETS_BLENDER_DEVICE` (`auto`), `ASSETS_IGPU_BUS` (`c4`), `ASSETS_IGPU_MAX_BUSY` (10), `ASSETS_UE_IDLE_S` (300),
`ASSETS_LOCK_WAIT_S` (1200), `ASSETS_MIN_FREE_GB` (6), `ASSETS_THREADS` (16), `ASSETS_STAGE_EXPOSURE_EV` (0),
`ASSETS_UE_RAYTRACING` (1), `ASSETS_HIPRT` (1).

The UE project needs the plugins PythonScriptPlugin, EditorScriptingUtilities, Interchange, InterchangeEditor.
`run_asset.py` copies `ue/init_unreal.py` into the project's `Content/Python/`; it does nothing unless
`ASSETS_UE_JOB` names a render job, so normal editor starts are unaffected.

## Machine rules built in

- GPU work takes the card's lock (`/tmp/gpu-<bus>.lock`) with a non-blocking flock polled every 15 s (Blender
  does not wait in `auto`: it moves to the other card, the iGPU or the CPU). Blender holds it for its run; the UE
  editor for its lifetime (long-lived: until idle). UE gives up after `ASSETS_LOCK_WAIT_S`.
  `DRI_PRIME=pci-0000_<bus>_00_0` puts that card first in Vulkan's list and the editor takes adapter 0; the run
  records VRAM per card to show which one worked.
- A guard (a thread, or a detached watchdog for the long-lived editor) stops the editor by its own PID when
  MemAvailable falls under `ASSETS_MIN_FREE_GB`; a render waits to start until 10 GiB more than that is free. Shader-compile threads are capped in the project's ini.

## Model scripts

A model script is plain `bpy` code run in an empty scene (metric, 1 unit = 1 m) with `SPEC` (the spec dict) in its
globals. It builds the asset with its front facing -Y and its base on Z = 0, and gives every face a Principled-BSDF
material (procedural nodes are fine; they are baked). Modifiers are applied by the harness (before any object is
removed, so modifier targets may be empties or hidden objects); position-based shader masks keep their place. Target Blender 4.5.

## Export format test (UE 5.8.1, Interchange, same anvil)

| Format | Scale / axes | Material in UE | Normal map | Verdict |
|---|---|---|---|---|
| glTF (.glb) | exact | glTF master instance with BaseColor, Normal, ORM (occlusion + metal/rough) | green flip set, TC_NormalMap | **used** |
| FBX | exact | legacy Phong instance: BaseColor + Normal only; roughness, metallic, AO lost | no green flip | skeletal only (later) |
| USD (.usdc) | exact | UsdPreviewSurface instance, two-sided; no occlusion; extra cube-map texture | no green flip (wrong for OpenGL maps) | no |
