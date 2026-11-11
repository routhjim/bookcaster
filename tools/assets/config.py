"""Locations and machine settings for the asset chain (tools/assets).

Every location comes from an environment variable; the defaults below are the documented layout:

    ASSETS_BLENDER      Blender binary          default $BLENDER, else ~/opt/blender/current/blender (4.5 LTS)
    ASSETS_UE_HOME      UE root folder          default ~/ue
    ASSETS_UE_ENGINE    engine folder           default $ASSETS_UE_HOME/UE_5.8.1
    ASSETS_UE_PROJECT   the one engine project  default $ASSETS_UE_HOME/proj/CharLab/CharLab.uproject (props, sets,
                                                characters and scenes share it: no project swaps; AssetLab is kept
                                                on disk as it was)
    ASSETS_LIBRARY      asset library root      default $ASSETS_UE_HOME/library
    ASSETS_LOGS         log folder              default $ASSETS_UE_HOME/logs
    ASSETS_GPU_CARDS    cards UE may use        default "65,05" (PCI bus ids, tried in order; lock files /tmp/gpu-<bus>.lock)
    ASSETS_UE_CARDS     UE card order           default "65,05" (UE on one card, Blender on the other)
    ASSETS_BLENDER_CARDS  Blender card order    default "05,65"
    ASSETS_UE_IDLE_S    the resident editor quits after this long without work; default 2400 (or --stop-editor)
    ASSETS_LOCK_WAIT_S  give up on a busy card  default 1200
    ASSETS_MIN_FREE_GB  memory guard            default 6 (stop the editor under this MemAvailable)
    ASSETS_THREADS      Blender CPU threads     default 16
    ASSETS_BLENDER_DEVICE  bake/sheet device    default "auto" (a GPU: the home card / iGPU; waits for one, never
                                                the CPU); "xtx", "igpu"; "cpu" only as a manual debugging switch
    ASSETS_IGPU_BUS     iGPU PCI bus            default "c4" (Blender only; never UE)
    ASSETS_IGPU_MAX_BUSY  iGPU busy % to accept default 10
    ASSETS_UE_RAYTRACING  hardware ray tracing  default 1: r.RayTracing + Lumen hardware tracing in the project's
                                                DefaultEngine.ini (written before an editor starts); 0 = software
                                                Lumen (the next editor start recompiles shaders either way)
    ASSETS_HIPRT        Cycles HIP RT default   default 1 (run_asset.py --hiprt 0 turns it off for one run)
    ASSETS_BAKE_WAIT_S  GPU wait before the CPU default 45 (Blender polls the cards, a card's bake slot beside the
                                                warm editor, and the iGPU every 2 s)
    ASSETS_BAKE_VRAM_GB VRAM for a bake beside  default 8
                        the warm editor
    ASSETS_IGPU_LOCK    iGPU lock name          default "igpu" (/tmp/gpu-igpu.lock)

  Scheduler (assetd.py; fixed device homes, GPU only, see README "Scheduler"):
    ASSETS_SCHED        1 (default): run_asset.py goes through the scheduler; 0 = the in-process path (--local)
    ASSETS_SCHED_DIR    socket, state           default $ASSETS_UE_HOME/sched
    ASSETS_SCHED_IDLE_S scheduler exits after   default 60 s with no job and no editor
    ASSETS_UE_CARD      the UE editor's home    default "65"
    ASSETS_BLENDER_CARD Blender's home card     default "05" (bakes, sheets; previews too when the iGPU is taken)
    ASSETS_IMAGE_STATION the image model's station (reference drawing, paint): default the Blender card ("05");
                         "igpu" puts it back on the iGPU (cloth and the desktop stay there either way)
    ASSETS_BLENDER_THREADS   Blender CPU threads per job      default 0 = all cores (the CPU-by-nature parts)
    ASSETS_BLENDER_LINGER_S  keep the Blender card after the last job   default 120
    ASSETS_OUTSIDE_RELEASE_S  an outside holder of the UE card for this long = a release (video run); default 20
    ASSETS_OUTSIDE_RELEASE_BL_S  the same for the Blender card alone; default 120 (bakes spill meanwhile; a video
                         runner that needs both cards should call --release-cards rather than wait)
    ASSETS_RECLAIM_FREE_S  after a release, both locks free this long (and asset work queued) = reclaim; default 5
    ASSETS_RELEASE_GRACE_S  after --release-card the card is not retaken for this long   default 120
    ASSETS_UE_VRAM_GB   free VRAM to start the editor           default 8
    ASSETS_UE_SWAP_AFTER_S  the editor stays on its project while it has work, unless another project's oldest UE
                         job has waited this long; default 60 (a swap costs ~12 s)
    ASSETS_UE_STAY_S    minimum time an editor serves a project before another project's waiting job can force a
                         swap; default 150 (keeps two busy projects from swapping on every job)
    ASSETS_UE_JOB_TIMEOUT_S  --ue-job time limit unless the job sets timeout_s   default 300
    ASSETS_UE_JOB_OUTPUT_GRACE_S  after a --ue-job script returns ok, its outputs must exist within this; default 10
    ASSETS_BLENDER_SPARES  warm Blender processes kept on the Blender card while it is in use; default 1 (0 = off;
                         a spare is a loaded idle Blender, not a second job: the card runs one job at a time)

  The line (line.py): stations with one job each, short lines, batches, the power and hardware guard; see there.

Stdlib only: imported by the orchestrator, never inside Blender or UE.
"""

import os
from pathlib import Path


def _p(var: str, default) -> Path:
    v = os.environ.get(var)
    return Path(os.path.expanduser(v)) if v else Path(os.path.expanduser(str(default)))


BLENDER = _p("ASSETS_BLENDER", os.environ.get("BLENDER") or "~/opt/blender/current/blender")
UE_HOME = _p("ASSETS_UE_HOME", "~/ue")
UE_ENGINE = _p("ASSETS_UE_ENGINE", UE_HOME / "UE_5.8.1")
UE_PROJECT = _p("ASSETS_UE_PROJECT", UE_HOME / "proj" / "CharLab" / "CharLab.uproject")
LIBRARY = _p("ASSETS_LIBRARY", UE_HOME / "library")
LOGS = _p("ASSETS_LOGS", UE_HOME / "logs")
GPU_CARDS = [c.strip() for c in os.environ.get("ASSETS_GPU_CARDS", "65,05").split(",") if c.strip()]
LOCK_WAIT_S = float(os.environ.get("ASSETS_LOCK_WAIT_S", "1200"))
MIN_FREE_GB = float(os.environ.get("ASSETS_MIN_FREE_GB", "6"))
THREADS = int(os.environ.get("ASSETS_THREADS", "16"))
BLENDER_DEVICE = os.environ.get("ASSETS_BLENDER_DEVICE", "auto")   # GPU first: XTX, then iGPU, then CPU
IGPU_BUS = os.environ.get("ASSETS_IGPU_BUS", "c4")
IGPU_MAX_BUSY = int(os.environ.get("ASSETS_IGPU_MAX_BUSY", "10"))

UE_CARDS = [c.strip() for c in os.environ.get("ASSETS_UE_CARDS", "65,05").split(",") if c.strip()]
BLENDER_CARDS = [c.strip() for c in os.environ.get("ASSETS_BLENDER_CARDS", "05,65").split(",") if c.strip()]
UE_IDLE_S = float(os.environ.get("ASSETS_UE_IDLE_S", "2400"))     # the resident editor: long idle, --stop-editor
UE_IDLE2_S = float(os.environ.get("ASSETS_UE_IDLE2_S", "600"))   # the second editor (Blender card)
UE_EDITORS = 1               # one engine on its card (a second on the Blender card measured 3x slower, 11-10)
UE_SECOND_AT = int(os.environ.get("ASSETS_UE_SECOND_AT", "2"))
SCHED_DIR = _p("ASSETS_SCHED_DIR", UE_HOME / "sched")
SCHED_IDLE_S = float(os.environ.get("ASSETS_SCHED_IDLE_S", "60"))
UE_CARD = os.environ.get("ASSETS_UE_CARD", "65")
BLENDER_CARD = os.environ.get("ASSETS_BLENDER_CARD", "05")
# the image model's station (reference drawing, paint): measured 11-11 on one drawing and one 3-view paint pass,
# card 05 drew 4.5-6x faster than the iGPU (sheet 14.7 v 66.2 s, paint view ~9 v ~51 s; load ~30 s on both warm)
_img = os.environ.get("ASSETS_IMAGE_STATION", BLENDER_CARD).strip()
IMAGE_STATION = "igpu" if _img == "igpu" else "card" + (_img[4:] if _img.startswith("card") else _img)
BLENDER_PER_CARD = 1         # one job per station
BLENDER_THREADS = int(os.environ.get("ASSETS_BLENDER_THREADS", "0"))   # 0 = every core (owner, 11-11: no thread limits)
BLENDER_LINGER_S = float(os.environ.get("ASSETS_BLENDER_LINGER_S", "120"))
OUTSIDE_RELEASE_BL_S = float(os.environ.get("ASSETS_OUTSIDE_RELEASE_BL_S", "120"))
OUTSIDE_RELEASE_S = float(os.environ.get("ASSETS_OUTSIDE_RELEASE_S", "20"))
RECLAIM_FREE_S = float(os.environ.get("ASSETS_RECLAIM_FREE_S", "5"))
RELEASE_GRACE_S = float(os.environ.get("ASSETS_RELEASE_GRACE_S", "120"))
UE_VRAM_GB = float(os.environ.get("ASSETS_UE_VRAM_GB", "8"))
UE_JOB_TIMEOUT_S = float(os.environ.get("ASSETS_UE_JOB_TIMEOUT_S", "300"))   # --ue-job default; set timeout_s for more
UE_JOB_OUTPUT_GRACE_S = float(os.environ.get("ASSETS_UE_JOB_OUTPUT_GRACE_S", "10"))
UE_SWAP_AFTER_S = float(os.environ.get("ASSETS_UE_SWAP_AFTER_S", "60"))   # serve another project after this wait
UE_STAY_S = float(os.environ.get("ASSETS_UE_STAY_S", "150"))   # minimum stay on a project before a forced swap
UE_IDLE_FPS = int(os.environ.get("ASSETS_UE_IDLE_FPS", "10"))   # idle editor frame cap (t.MaxFPS)
BLENDER_SPARES = int(os.environ.get("ASSETS_BLENDER_SPARES", "1"))
BAKE_WAIT_S = float(os.environ.get("ASSETS_BAKE_WAIT_S", "45"))   # poll for a GPU this long before the CPU
BAKE_VRAM_GB = float(os.environ.get("ASSETS_BAKE_VRAM_GB", "8"))  # free VRAM to bake beside the warm editor
IGPU_LOCK_NAME = os.environ.get("ASSETS_IGPU_LOCK", "igpu")        # /tmp/gpu-<name>.lock
UE_RAYTRACING = os.environ.get("ASSETS_UE_RAYTRACING", "1") == "1"
HIPRT = int(os.environ.get("ASSETS_HIPRT", "1"))
# Renderer settings the project needs for hardware ray tracing (measured on both XTX cards, 11-10: initialises,
# first start ~67 s of shader compile, later starts ~24 s v ~15 s software). Shadows stay virtual shadow maps.
UE_RT_SETTINGS = {"r.RayTracing": "True", "r.RayTracing.Shadows": "False", "r.Lumen.HardwareRayTracing": "True",
                  "r.SkinCache.CompileShaders": "True"}

UE_BIN = UE_ENGINE / "Engine" / "Binaries" / "Linux"
UE_EDITOR = UE_BIN / "UnrealEditor"
UE_CMD = UE_BIN / "UnrealEditor-Cmd"

# Content paths inside the UE project.
UE_STAGE_MAP = "/Game/AssetStage/L_AssetStage"
UE_ASSET_ROOT = "/Game/Assets"
STAGE_EXPOSURE_EV = float(os.environ.get("ASSETS_STAGE_EXPOSURE_EV", "0"))   # manual exposure compensation

# Tier budgets (see CONTRACT.md).
TIERS = {
    "hero": {"max_tris": 250_000, "tex": 2048},
    "mid": {"max_tris": 60_000, "tex": 1024},
    "background": {"max_tris": 8_000, "tex": 512},
}
