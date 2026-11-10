# Asset chain: Blender model script -> checked asset -> UE5 still (M1)

One command takes an asset spec and a Blender model script through the whole chain, unattended:

```
python tools/assets/run_asset.py tools/assets/examples/anvil/anvil.json tools/assets/examples/anvil/anvil.py
```

Run long jobs detached (`setsid nohup ... > log 2>&1 &`). Stdlib Python 3 only on the calling side.

| Step | Where | What |
|---|---|---|
| 1 Blender harness (`bl_harness.py`) | Blender, CPU | runs the model script in an empty scene; applies the contract (one mesh `SM_<id>`, floor-centre origin, transforms applied); checks; Smart-UV; bakes materials to BaseColor / Normal / ORM; 8-view contact sheet (Cycles CPU, 32 spp + denoise); exports `.glb`; saves the `.blend` |
| 2 UE import (`ue/ue_import.py`) | `UnrealEditor-Cmd -run=pythonscript -nullrhi`, no GPU | Interchange glTF import to `/Game/Assets/<Class>s/<id>/`, mesh renamed `SM_<id>`, Nanite on, texture settings checked |
| 3 Stage (`ue/ue_stage.py`) | same, once | builds `/Game/AssetStage/L_AssetStage` (floor, physical sky, sun, sky light, fog, post volume) if missing |
| 4 UE render (`ue/ue_render.py`) | full editor, `-RenderOffscreen`, one XTX | asset at the origin, 1.8 m box and the template mannequin beside it; wide / medium / close / back stills at 1152x640, Lumen |

Output: `<library>/<class>/<id>/` with `asset.json` (manifest entry), `source/` (model script, spec, `.blend`),
`export/`, `textures/`, `sheet/<id>_sheet.png`, `ue/<id>_<shot>.png`, `logs/`; `<library>/manifest.json` indexes
all assets. The rules are in [CONTRACT.md](CONTRACT.md).

Flags: `--skip-render` (CPU-only parts), `--source-sheet` (also render the unbaked model, to see bake loss),
`--formats glb,fbx,usd` (extra exports), `--warm-s`, `--settle-s`, `--ue-timeout`.

## Settings

Environment variables, documented in `config.py`: `ASSETS_BLENDER` (default `$BLENDER`, else
`~/opt/blender/current/blender`, 4.5 LTS), `ASSETS_UE_HOME` (`~/ue`), `ASSETS_UE_ENGINE`, `ASSETS_UE_PROJECT`
(`$ASSETS_UE_HOME/proj/AssetLab/AssetLab.uproject`), `ASSETS_LIBRARY` (`$ASSETS_UE_HOME/library`), `ASSETS_LOGS`,
`ASSETS_GPU_CARDS` (`05,65`), `ASSETS_LOCK_WAIT_S` (1200), `ASSETS_MIN_FREE_GB` (6), `ASSETS_THREADS` (16).

The UE project needs the plugins PythonScriptPlugin, EditorScriptingUtilities, Interchange, InterchangeEditor.
`run_asset.py` copies `ue/init_unreal.py` into the project's `Content/Python/`; it does nothing unless
`ASSETS_UE_JOB` names a render job, so normal editor starts are unaffected.

## Machine rules built in

- The render takes one card's lock (`/tmp/gpu-<bus>.lock`) with a non-blocking flock polled every 15 s, holds it
  for the editor's lifetime only, and gives up after `ASSETS_LOCK_WAIT_S` (the CPU parts are already done).
  `DRI_PRIME=pci-0000_<bus>_00_0` puts that card first in Vulkan's list and the editor takes adapter 0; the run
  records VRAM per card to show which one worked.
- A guard thread stops the editor (its own PID) when MemAvailable falls under `ASSETS_MIN_FREE_GB`; the render
  waits to start until 10 GiB more than that is free. Shader-compile threads are capped in the project's ini.

## Model scripts

A model script is plain `bpy` code run in an empty scene (metric, 1 unit = 1 m) with `SPEC` (the spec dict) in its
globals. It builds the asset with its front facing -Y and its base on Z = 0, and gives every face a Principled-BSDF
material (procedural nodes are fine; they are baked). Modifiers are applied by the harness. Target Blender 4.5.

## Export format test (UE 5.8.1, Interchange, same anvil)

| Format | Scale / axes | Material in UE | Normal map | Verdict |
|---|---|---|---|---|
| glTF (.glb) | exact | glTF master instance with BaseColor, Normal, ORM (occlusion + metal/rough) | green flip set, TC_NormalMap | **used** |
| FBX | exact | legacy Phong instance: BaseColor + Normal only; roughness, metallic, AO lost | no green flip | skeletal only (later) |
| USD (.usdc) | exact | UsdPreviewSurface instance, two-sided; no occlusion; extra cube-map texture | no green flip (wrong for OpenGL maps) | no |
