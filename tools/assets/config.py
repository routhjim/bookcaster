"""Locations and machine settings for the asset chain (tools/assets).

Every location comes from an environment variable; the defaults below are the documented layout:

    ASSETS_BLENDER      Blender binary          default $BLENDER, else ~/opt/blender/current/blender (4.5 LTS)
    ASSETS_UE_HOME      UE root folder          default ~/ue
    ASSETS_UE_ENGINE    engine folder           default $ASSETS_UE_HOME/UE_5.8.1
    ASSETS_UE_PROJECT   .uproject to import to  default $ASSETS_UE_HOME/proj/AssetLab/AssetLab.uproject
    ASSETS_LIBRARY      asset library root      default $ASSETS_UE_HOME/library
    ASSETS_LOGS         log folder              default $ASSETS_UE_HOME/logs
    ASSETS_GPU_CARDS    cards UE may use        default "65,05" (PCI bus ids, tried in order; lock files /tmp/gpu-<bus>.lock)
    ASSETS_UE_CARDS     UE card order           default "65,05" (UE on one card, Blender on the other)
    ASSETS_BLENDER_CARDS  Blender card order    default "05,65"
    ASSETS_UE_IDLE_S    long-lived editor quits after this long without work; default 300
    ASSETS_LOCK_WAIT_S  give up on a busy card  default 1200
    ASSETS_MIN_FREE_GB  memory guard            default 6 (stop the editor under this MemAvailable)
    ASSETS_THREADS      Blender CPU threads     default 16
    ASSETS_BLENDER_DEVICE  bake/sheet device    default "auto" (free XTX, else idle iGPU, else CPU); "xtx",
                                                "igpu", "cpu"
    ASSETS_IGPU_BUS     iGPU PCI bus            default "c4" (Blender only; never UE)
    ASSETS_IGPU_MAX_BUSY  iGPU busy % to accept default 10

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
UE_PROJECT = _p("ASSETS_UE_PROJECT", UE_HOME / "proj" / "AssetLab" / "AssetLab.uproject")
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
UE_IDLE_S = float(os.environ.get("ASSETS_UE_IDLE_S", "300"))

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
