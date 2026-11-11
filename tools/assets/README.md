# Asset chain: Blender model script -> checked asset -> UE5 still (M1)

One command takes an asset spec and a Blender model script through the whole chain, unattended:

```
python tools/assets/run_asset.py tools/assets/examples/anvil/anvil.json tools/assets/examples/anvil/anvil.py
python tools/assets/run_asset.py SPEC.json MODEL.py --preview      # modelling loop: shape + checks + quick sheet
python tools/assets/run_asset.py --status                            # the scheduler: devices, queue, recent jobs
python tools/assets/run_asset.py --stop-editor                       # unload the resident UE editor
```

Stdlib Python 3 only on the calling side. Any number of runs may go at once: they queue in one scheduler (see
"Scheduler, devices and the box").

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

**Resident editor.** The scheduler starts one UE editor on the UE card when the first final job arrives (in
parallel with that job's Blender stage) and keeps it for `ASSETS_UE_IDLE_S` (2400) without work or until
`--stop-editor`; jobs reach it through `<project>/Saved/AssetServer/ed-<bus>/queue/`. A warm editor does import +
stills in ~4.5 s instead of ~15-28 s. When `ue/*.py` changes, the scheduler restarts the editor between jobs.
`--fresh-editor` (implies `--local`) starts and quits an editor for one asset.

Output: `<library>/<class>/<id>/` with `asset.json` (manifest entry), `source/` (model script, spec, `.blend`),
`export/`, `textures/`, `sheet/<id>_sheet.png`, `ue/<id>_<shot>.png`, `logs/`; `<library>/manifest.json` indexes
all assets. The rules are in [CONTRACT.md](CONTRACT.md).

**Cameras.** The spec may set `camera_preset` (default: its class) and `cameras`: a list of `{"name", "position":
[x, y, z], "target": [x, y, z], "focal": mm}` in the asset's Blender frame (metres, front -Y, base on Z = 0); each
adds a still `ue/<id>_<name>.png`.

**Painting (optional, `--paint`).** `paint.py` (also runnable alone on a finished asset folder) renders guide views
of the baked asset, has Qwen-Image-2.1 repaint each (img2img from the guide at strength 0.65 with the object mask as
the per-pixel strength, background held; ControlNet and the edit model are options, both worse in the method test),
divides the guide's shading out, projects the paintings back per texel and writes a second
map set; see the docstring in `paint.py` and "Painted maps" in CONTRACT.md. Spec block `paint` (prompt, views,
close-up region, strength, projection parameters) and `texture_set`.

Measured 11-10 (iGPU drawer, the cards were busy): anvil 18 views (8 ring, 4 upper, top, 4 + top close) ~50 s a
view + ~50 s model load, projection 50-70 s; winged creature 21 views ~47 s a view, projection 125-170 s at 4096.
On an XTX a view takes ~9-19 s. Results: the anvil's striking face and edges hold at the UE close shot (forged,
scratched, worn); a ghost edge 1-2 cm off a real edge and a hole painted as a stud show where the painter slid
off the shape. The creature reads as one coherent scaled hide from wide to ~2 m frames, but the painter leaves
plain-coloured parts (limbs, tail) plain and redraws the head's horns in other places, so detail on the head
averages into lumps; macro frames do not hold. While the asset scheduler runs, paint.py stays off the cards
(the scheduler would read its card use as an outside release) and uses the iGPU.

Previews and sheets are not denoised (Blender ran OIDN on the CPU here: 0.4-4.4 s per tile, a 12 spp preview took
12-15 s and 30 s of CPU); they path-trace more samples on the card instead (preview 32 spp: 3.8 s wall, sheet
1.7 s; `--denoise` restores 12 spp + OIDN).
Other flags: `--preview-px`, `--preview-samples` (preview sheet size and samples); a final pass makes no Blender
sheet when UE stills follow (`--sheet` = 320 px, 16 spp; `--full-sheet` = 512 px, 32 spp; `--sheet-px`,
`--sheet-samples`; it ran 10-13 s per asset on a shared card), `--skip-render` (Blender + import only, through the editor), `--source-sheet` (also render the unbaked model),
`--formats glb,fbx,usd`, `--device auto|xtx|igpu` (`cpu`: manual debugging only), `--hiprt 0|1` (Cycles HIP RT, default on), `--warm-frames`, `--settle-frames`,
`--warm-s`, `--settle-s`, `--ue-timeout`.

## Scheduler, devices and the box

`run_asset.py` is a thin client of one scheduler per box (`assetd.py`): it sends its command line over
`$ASSETS_SCHED_DIR/sched.sock`, prints the job's log and exits with the job's code (same outputs as before). The
scheduler starts on first use (detached, log `$ASSETS_LOGS/assets_sched_<mmdd>.log`) and exits after
`ASSETS_SCHED_IDLE_S` (60) with no job and no editor. `--local` (or `ASSETS_SCHED=0`) runs the old in-process path (debugging); a
client whose `ASSETS_*` environment differs from the scheduler's runs locally too.

```
python tools/assets/run_asset.py --status          # devices (who holds each card), queue, recent jobs with per-stage times
python tools/assets/run_asset.py --release-cards   # make way for a video run (both cards); prints how long it took
python tools/assets/run_asset.py --reclaim-cards   # back to asset mode: reload the editor, resume the queue
python tools/assets/run_asset.py --release-card 05 # give up the Blender card only (the editor stays)
python tools/assets/run_asset.py --stop-editor     # unload the resident UE editor
python tools/assets/run_asset.py --cancel 12       # cancel job 12 (or by asset id); see --status
```

**Stages.** A job is a chain of stages; each starts the moment its device is free, so stages of different assets
overlap (asset B bakes while asset A renders in UE): `blender_preview`; or `blender_final` (bake, export, then a
small sheet) -> `paint` (with `--paint`; paint.py takes its own locks) -> `ue` (import + stills in the editor) ->
`finalize` (manifest). The UE stage starts as soon as Blender has written the glTF (blender_report.json status
`exported`), while Blender still renders its sheet. One asset id never runs twice at once (a second job of the
same id waits). Every job writes `logs/sched.json` (per stage: device, seconds waited, seconds run) and asset.json
gets `sched`. A client that is killed cancels its job: queued stages are dropped, a running Blender is stopped.

**Fixed homes, GPU only.** Loading models is the expensive part, so nothing moves between devices:

| Work | Home | Notes |
|---|---|---|
| model script, contract, checks, UV unwrap, glTF export, manifest | CPU | CPU work by nature (each under ~1 s); not a fallback. Blender runs with `-t 4` (`ASSETS_BLENDER_THREADS`) |
| bake, final contact sheet | card 05 (`ASSETS_BLENDER_CARD`) | up to `ASSETS_BLENDER_PER_CARD` (3) Blender jobs at once; `ASSETS_BLENDER_SPARES` (2) warm Blender processes wait there (Cycles already on the card; one job each, then they exit) |
| preview | the iGPU (its lock, when idle), else card 05 | keep the iGPU light: it draws the desktop and hosts the image model |
| UE import + stills | one resident editor on card 65 (`ASSETS_UE_CARD`) | loaded on the first final job, kept for `ASSETS_UE_IDLE_S` (2400) without work or until `--stop-editor`; restarted between jobs when `ue/*.py` changed |
| UE, second editor | off (`ASSETS_UE_EDITORS`=1) | UE is the bottleneck with one editor (8 serial UE jobs ~65 s of a 72 s batch), but a second editor on card 05 made it worse on this box: each UE job took 18-37 s instead of 6-9 s and previews doubled (two editors tick and import at once). `ASSETS_UE_EDITORS=2` still starts one when the UE backlog reaches `ASSETS_UE_SECOND_AT` (2); it is safe for the project because every job imports its own asset id into its own folder, one id never runs in both, autosave is off and the stage level is never saved |

Nothing is ever baked or rendered on the CPU: with its home busy, a stage waits in the queue (`--device cpu`
exists only as a manual debugging switch). The harness refuses to fall back to the CPU when its GPU is missing.

**Locks.** All locks are `flock -n`, polled, and never held while waiting for another. While asset work is going
on the scheduler holds a visible tenancy on both cards: card 65 for the editor's life (`/tmp/gpu-65.tenant` names
it), card 05 while Blender work runs or is queued and `ASSETS_BLENDER_LINGER_S` (120) after. It places work on a
card only under that card's lock and when sysfs `mem_info_vram_used` leaves room. The editor's lock is kept by a
small holder process (`assetd.py --hold`), so the helpers UE forks (zenserver, UnrealTraceServer, which outlive
the editor) never inherit a card lock. **Copy this pattern in any tool that starts UE (or anything that forks
daemons) under a card lock:** do not pass the lock fd to the editor; pass it to a holder process that starts the
editor with `close_fds=True` and waits for it (`python3 tools/assets/assetd.py --hold -- <editor command...>`
does exactly that, forwarding SIGTERM). An editor that inherits the fd hands it to zenserver, which can outlive it
and keep the card locked. The project lock (`<project>/Saved/AssetServer/project.lock`) is held while
the editor lives, so a `--local` run never starts a second editor writing the same project.

**Asset work and video work alternate; they do not overlap.** The video model needs both cards. A video runner
either calls `run_asset.py --release-cards` first (stages in flight finish, the queue holds, the editor stops,
both locks are freed; it returns when both cards are free) or simply polls `/tmp/gpu-05.lock` and
`/tmp/gpu-65.lock` with `flock -n`: the scheduler treats an outside holder of the UE card for
`ASSETS_OUTSIDE_RELEASE_S` (20), or of the Blender card alone for `ASSETS_OUTSIDE_RELEASE_BL_S` (120) while
leaving it idle (mean busy under 10%: a runner holding one card while it waits for the other), as the same
release (it never fights for a lock). An outside job that works on the Blender card does not cause a release. Meanwhile, with the Blender card held from outside for
`ASSETS_BAKE_SPILL_S` (20), bakes spill to the UE card beside the editor when it has `ASSETS_SPILL_MIN_FREE_GB`
(8) free, at most `ASSETS_SPILL_JOBS` (2) at once; each spill is logged and counted in `--status`. While released, asset jobs wait
(previews may still run on the iGPU); nothing runs on the CPU. Asset mode returns by itself when both locks have
been free for `ASSETS_RECLAIM_FREE_S` (5) with asset work queued (when the release came from an outside holder
of the UE card, the UE card alone being free is enough: bakes then spill beside the editor while the Blender card
stays held) (after `ASSETS_RELEASE_GRACE_S`, 120, for a
manual release), or at `--reclaim-cards`; the cost is the editor's reload. `--release-card 05` gives up only the
Blender card for an outside job that needs one card.

**Other UE work goes through the same editor (`--ue-job`).** Tools that need a UE editor (the set builder, the
character and MetaHuman tools) should not start private editors: those hold a card the scheduler cannot plan
around. They submit a job instead:

```
python tools/assets/run_asset.py --ue-job JOB.json
# JOB.json: {"project": ".../SetLab/SetLab.uproject", "script": "editor-side .py", "job": {...},
#            "outputs": [files to wait for], "timeout_s": 1800, "id": "name", "log_dir": "folder"}
```

The script runs inside the resident editor with `JOB` (the "job" dict) in its globals. If it defines
`run(job)`, that is called: a dict return is the result; a generator is advanced one step per editor tick (for
anything that needs rendered frames: yield between steps) and its return value is the result. The client prints
`{"rc", "result", "result_file", "seconds", "error"}` as JSON and exits 0 when the script succeeded and every
listed output exists. A result dict with `"ok": false` or an `"error"`, or an exception, fails the job at once;
the outputs are checked once when the script returns (within `ASSETS_UE_JOB_OUTPUT_GRACE_S`, 10 s). The time
limit is `ASSETS_UE_JOB_TIMEOUT_S` (300 s) unless the job sets `timeout_s`. A killed client, or
`run_asset.py --cancel <job>` (numbers and ids are in `--status`), stops a generator script at its next tick; a
script that blocks in one call cannot be interrupted until it returns. The job's stage record is `<log_dir>/logs/sched.json`.

One editor serves one project at a time: when the oldest queued UE work is for another project and the current
project has nothing queued (or that work has waited `ASSETS_UE_SWAP_AFTER_S`, 60 s), the editor quits and
restarts on that project. A swap measured 12.1-12.9 s from quit to ready (AssetLab <-> CharLab, warm caches). A project keeps its own start-up shim (`Content/Python/init_unreal.py`): the job file is passed
as both `ASSETS_UE_JOB` and `MH_JOB`, so the asset chain's shim and the character tools' shim both start the
scheduler's server loop (`ue/ue_render.py`); a project without a shim gets the chain's. Other projects are not
given the stage level, ray-tracing settings or mannequin.

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

- GPU work takes the card's lock (`/tmp/gpu-<bus>.lock`, the iGPU's `/tmp/gpu-igpu.lock`) with a non-blocking
  flock (see "Locks" above; `--local` runs poll the locks themselves and fail after `ASSETS_LOCK_WAIT_S` rather
  than use the CPU). `DRI_PRIME=pci-0000_<bus>_00_0` puts that card first in Vulkan's list and the editor takes
  adapter 0; the run records the card's load to show which one worked.
- A guard (the scheduler, a thread or a detached watchdog in `--local` runs) stops the editor by its own PID when
  MemAvailable falls under `ASSETS_MIN_FREE_GB`; a render waits to start until 10 GiB more than that is free. Shader-compile threads are capped in the project's ini.

## Model scripts

**UVs.** The default Smart UV Project (60 degrees) leaves seams along swept round parts and aliases on very thin
faces. A model script that needs better UVs makes its own (one UV map, non-overlapping, inside 0-1; e.g. a
cylinder unwrapped by `bpy.ops.uv.cylinder_project` or by seams + `bpy.ops.uv.unwrap`) and sets `"keep_uvs": true`
in the spec. The degenerate-face check names the part (object of the model script) and the world positions of up
to three zero-area faces.

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
