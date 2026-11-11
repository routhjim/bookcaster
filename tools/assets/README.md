# Asset chain: Blender model script -> checked asset -> UE5 still (M1)

One command takes an asset spec and a Blender model script through the whole chain, unattended:

```
python tools/assets/run_asset.py tools/assets/examples/anvil/anvil.json tools/assets/examples/anvil/anvil.py
python tools/assets/run_asset.py SPEC.json MODEL.py --preview      # modelling loop: shape + checks + quick sheet
python tools/assets/run_asset.py --status                            # the scheduler: devices, queue, recent jobs
python tools/assets/run_asset.py --stop-editor                       # unload the resident UE editor
```

Stdlib Python 3 only on the calling side. Runs go through one serial line with short queues (see "The line").

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
averages into lumps; macro frames do not hold. Run by the line, paint.py uses the image station's device only
(card 05 by default; the line hands it over, see "The line"); run outside the line it stays off the cards while the
scheduler lives. Its Blender steps on the iGPU (guides, projection) leave one CU for the desktop as the drawer does.

Previews and sheets are not denoised (Blender ran OIDN on the CPU here: 0.4-4.4 s per tile, a 12 spp preview took
12-15 s and 30 s of CPU); they path-trace more samples on the card instead (preview 32 spp: 3.8 s wall, sheet
1.7 s; `--denoise` restores 12 spp + OIDN).
Other flags: `--preview-px`, `--preview-samples` (preview sheet size and samples); a final pass makes no Blender
sheet when UE stills follow (`--sheet` = 320 px, 16 spp; `--full-sheet` = 512 px, 32 spp; `--sheet-px`,
`--sheet-samples`; it ran 10-13 s per asset on a shared card), `--skip-render` (Blender + import only, through the editor), `--source-sheet` (also render the unbaked model),
`--formats glb,fbx,usd`, `--device auto|xtx|igpu` (`cpu`: manual debugging only), `--hiprt 0|1` (Cycles HIP RT, default on), `--warm-frames`, `--settle-frames`,
`--warm-s`, `--settle-s`, `--ue-timeout`.

## The line: stations, the work list, batches and the power guard

All GPU work on this box goes through ONE door: the asset line (`assetd.py`, logic in `line.py`; `run_asset.py` is
its client). Nothing outside the line takes a card or iGPU lock.

```
python tools/assets/run_asset.py SPEC MODEL [--detach]                      # an asset (bake -> engine -> manifest)
python tools/assets/run_asset.py SPEC MODEL --ue-only --detach              # engine stage only (baked before)
python tools/assets/run_asset.py --ue-job JOB.json [--detach]               # an editor script (sets, characters, scenes)
python tools/assets/run_asset.py --line-cmd igpu refs --id X [--log F] [--detach] -- CMD...   # an existing tool at a station
python tools/assets/run_asset.py --feed LIST       # one submitter for a backlog (one argument line per entry, in order)
python tools/assets/run_asset.py --status          # the one-screen view
python tools/assets/run_asset.py --start           # start the line: it resumes the work list on disk
```

**Stations, one job each.** `card05` (Blender bakes, sheets and previews; image-to-3D; and the image model:
reference drawing and paint), `igpu` (cloth drape; the desktop), `card65` (the one resident UE editor; a `--line-cmd engine KIND -- CMD` command
borrows this card while the engine has no work: engine jobs waiting or running go first, an idle editor is asked to
quit, the command runs once it is gone, and the next engine job loads the editor again, ~12-25 s). The image
model's station is `ASSETS_IMAGE_STATION` (default the Blender card; `igpu` puts it back): measured 11-11 with the
same jobs and settings, card 05 drew a 1536 px reference sheet in 14.7 s v 66.2 s on the iGPU (close-up 5.7 v 31.7 s)
and painted a view in 8-11.5 s v 49-52 s, with the model's warm load ~30 s on both (32 v 53 s for the drawer). The
paint stage runs there, and `--line-cmd image KIND -- CMD` puts a drawing or paint command there; the line hands the
card over (stops its warm Blender spares, lets go of its hold) and tells the tool its device in
`ASSETS_IMAGE_DEVICE` (`ref_draw.py` and `paint.py` use it). A job waits for its station; nothing runs on CPU
cores as a fallback. `video` is a booked window: when a video entry waits, the card stations finish what runs,
both cards are released, every waiting clip runs, then the cards are reclaimed (the editor reloads). Generic
`--line-cmd` entries run existing tools unchanged: the tool takes the station's lock itself as it does today; the
line starts it only when the station is free, lets go of its own hold on that card first, and does not count the
tool's process session as an outside user. Engine work is never a `--line-cmd` (use `--ue-job`).

**The engine card: editor work and borrows.** Borrow commands (`--line-cmd engine ...`) wait in their own line
(`card65:borrow` in the status) beside the editor's imports, stills and scripts (`card65`), so a line full of
borrows never holds a bake whose next stage is the editor, and the other way round. The card alternates: a borrow
takes it when no editor work can start (an asset still baking does not count), or once the editor has run
`ASSETS_LINE_BATCH` (8) jobs while the borrow waited, or after `ASSETS_LINE_BORROW_MAX_WAIT_S` (600 s); after a
borrow, the editor work that arrived meanwhile goes first. A turn given to a borrow stands until it starts, and no
editor is loaded for an asset that is still baking while a borrow waits (no unload / reload ping-pong).

**Image-to-3D on either card.** `img3d.py` runs on whichever card its station names: `--line-cmd blender shape
...` runs it on card 05 as an ordinary command, `--line-cmd engine shape ...` borrows card 65. Choose by what waits:
send it to `blender` (card 05) when card 05's line is short and no bakes, drawings or paint passes are queued there
(it then runs at once and the editor keeps card 65); send it to `engine` (card 65) when card 05 is busy with bakes,
the image model or paint, or when the editor has little work (each borrow costs the editor an unload and a ~12 s
reload). Look at `--status` first: the shorter line with the less valuable work in it is the right one. Never send
the same shape to both (a second entry with the same id at one station is refused anyway).

**The work list and restarts.** Every submission is an entry file under `$ASSETS_SCHED_DIR/line/entries/` (done
ones move to `line/done/`; per-entry logs in `line/logs/`). A client that goes away no longer cancels its job
(`--cancel` does); `--detach` returns once the entry is on the line. Starting the line again (`--start`, or any
client) reloads every unfinished entry in list order: a finished bake is kept, whatever was in flight runs again
(a Blender left running by the dead scheduler is stopped first; its orphan editor is asked to quit). Commands are
started through a small runner (`line.py --run RECORD -- CMD`) that writes the exit code to the entry's record
file the moment the command ends, and the line writes the stage's end to the entry (fsynced) before anything else:
after a restart, a command that ended while no line watched it is finished from its record and never runs again,
and one still running is adopted (watched to its end, not killed). A process counts as the entry's only if its pid
and its start time both match (pids are reused). Entries written by older code (no start time, no record) still get
the old treatment: a live process is stopped and the entry runs again.

**Reloading onto newer code.** Every 2 s the line looks at its own code (`assetd.py`, `line.py`, `config.py`,
`run_asset.py`). When it changed, the line starts nothing new, lets the RUNNING jobs finish (a job in flight is
never killed for a reload), unloads the idle editor and warm Blender, checks that the new code imports (if not, it
stays on the old code and says so), and re-executes itself: same pid, same socket and daemon lock, the work list
resumed from disk. Clients that connect meanwhile wait in the socket's backlog; a client following its entry (not
`--detach`) re-attaches by itself and gets the result. Measured in the dry-run test: the reload follows the
running job's end within a second. `ASSETS_LINE_RELOAD=0` turns it off; `--status` shows `RELOADING` and what it
waits for.

**Short lines.** At most `ASSETS_LINE_MAX` (3) jobs wait in front of a station; a submission past that gets
`LINE FULL ... retry later` and exit code 75, and nothing is accepted. A station also holds a job whose next
station's line is full, so the line never grows upstream. A flood of 40 submissions puts 4 on the line (one
running, three waiting) and refuses 36. `--feed` is the way to put a backlog through: it submits one entry at a
time, waits for room (leaving one place free in each line for other builders) and records progress beside the list,
so re-running it skips what was fed. A station holds a job for a full next line only while that line moves (its
station works on it, its editor loads, or it went idle less than `ASSETS_LINE_HOLD_GRACE_S`, 30 s, ago); a next line
that does not move holds at twice its length, so it never idles the station upstream.

**One job, one entry.** A submission that repeats a job already waiting or running is refused with `ALREADY
QUEUED: already queued as entry N` (exit code 76; `--feed` counts it as fed; a client without `--detach` follows
the entry that is there): a command with the same `--id` at the same station, an asset with the same id and mode, an
engine job with the same id and the same job file content. Once the entry has finished, the same job may be
submitted again.

**Batches.** Each station runs up to `ASSETS_LINE_BATCH` (8) jobs of one kind before another waiting kind gets its
turn (oldest first), and keeps going with the same kind while no other kind waits: the engine takes waiting
imports/stills (`ue_asset`) and editor scripts (`ue_script`) in turns of 8; Blender stays warm (one warm spare
process); at the image station a batch of drawings runs before the paint passes. Load times that
decide whether batching pays (measured): UE editor start 11.8 s (then ~4 s per warm import + stills); Blender
start ~1-2 s (a bake of a small prop is ~4.5 s in all); the image model ~30 s warm on card 05 (73-157 s
on the iGPU from a cold page cache), i.e. two to three drawings; image-to-3D 21 s load v 49 s per shape; cloth solver ~5 s. The image
model's batch only saves its load when the tool takes several items per run (refs/ref_draw.py's batch mode does).

**Power and hardware guard.** The battery (`/sys/class/power_supply/BAT0`) is watched and logged with every job
start and end (percent, status, energy_now; `line/power.jsonl` has the energy used per job and kind). At or below
`ASSETS_POWER_PAUSE_PCT` (20%) nothing new starts, running jobs finish, the list's state is written to
`line/state.json` and the status says "paused: battery N%"; work resumes by itself at `ASSETS_POWER_RESUME_PCT`
(40%). A lost charger (AC0 offline, or no USB-C source online) or a card missing from the PCI bus is logged with
its time, shown in the status, and holds card jobs until both are back. Card jobs are refused while
`/sys/module/pcie_aspm/parameters/policy` does not read `[performance]` (a reboot resets it), and while
`$ASSETS_UE_HOME/GPU_HOLD` exists. While the list has work the line holds a `systemd-inhibit --what=idle:sleep`
block (also while paused at the floor), released when the list is empty; it ends by itself if the line dies.

**One engine project.** Props, sets, characters and scenes share one project (`ASSETS_UE_PROJECT`, default
`CharLab`): no project swaps. Props import to `/Game/Assets/Props/<id>` there as before; the stage level was copied
from AssetLab (which stays on disk unchanged); the start-up shim runs a job named by `ASSETS_UE_JOB` or `MH_JOB`.

`--status` shows, per station, what runs and for how long, what waits and the oldest wait, the batch kind; the
battery, load, memory, why work is paused or held; and the jobs finished in the last hour. A line with waiting
entries that starts none says why on the line under it (`why not running (idle Ns): ...`): held for the next line
(and which one, how full), waiting for the editor to unload, editor work goes first, the guard (battery floor,
charger, card, ASPM, the hold file), a video window, a reload in progress, the card's lock or VRAM, or an entry that
waits for its earlier stage. A station idle 60 s with ready work logs the same reasons (`ASSETS_LINE_STALL_REPORT_S`).

**Stages.** An asset job is a chain of stages: `blender_preview`; or `blender_final` (bake, export, then a small
sheet) -> `paint` (with `--paint`, iGPU station) -> `ue` (import + stills in the editor) -> `finalize` (manifest).
The UE stage may start as soon as Blender has written the glTF. One asset id never runs twice at once. Every job
writes `logs/sched.json` (per stage: device, seconds waited, seconds run) and asset.json gets `sched`.

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

**Asset work and video work alternate; they do not overlap.** The video model needs both cards. Put video runs on
the line as `--line-cmd video video -- CMD...` entries: they get the window described above. The older ways still
work for a runner outside the line: `run_asset.py --release-cards` (stages in flight finish, the queue holds, the
editor stops, both locks are freed; it returns when both cards are free), or an outside holder of the UE card for
`ASSETS_OUTSIDE_RELEASE_S` (20) (of the Blender card alone, idle, for `ASSETS_OUTSIDE_RELEASE_BL_S`, 120), which
counts as the same release. While released, card jobs wait; nothing runs on the CPU. Asset mode returns when both
locks have been free for `ASSETS_RECLAIM_FREE_S` (5) with card work queued and no video entry left (after
`ASSETS_RELEASE_GRACE_S`, 120, for a manual release), or at `--reclaim-cards`; the cost is the editor's reload.

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
`run_asset.py --cancel <job>` (numbers and ids are in `--status`; a client that goes away no longer cancels), stops a generator script at its next tick; a
script that blocks in one call cannot be interrupted until it returns. The job's stage record is `<log_dir>/logs/sched.json`.

All engine work uses the one project (above), so the editor does not swap. A job that names another project still
works: the editor quits and restarts on it when that project's oldest job has waited `ASSETS_UE_SWAP_AFTER_S` (60 s)
(a swap costs 12-18 s), so do not. A project keeps its own start-up shim (`Content/Python/init_unreal.py`): the job file is passed
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
