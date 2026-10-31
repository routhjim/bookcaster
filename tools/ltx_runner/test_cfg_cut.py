"""10-31: install_cfg_cut turns the guidance passes off from round(cut * n) on, per lane (CFG_TL), and leaves a
lane with cut 1.0 fully guided. CPU, no model: run with the LTX venv python."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import ltx_pipelines.utils.denoisers as DN
from ltx_core.components.guiders import MultiModalGuider, MultiModalGuiderParams
seen = []
DN._guided_denoise = lambda **k: seen.append((k["step_index"], k["video_guider"].params.cfg_scale,
                                             k["audio_guider"].params.stg_scale))
import ltx_split_next as R
R.install_cfg_cut()
g = MultiModalGuider(params=MultiModalGuiderParams(cfg_scale=3.0, stg_scale=1.0))
fails = []
for cut, want_on in ((0.6, 12), (1.0, 20)):
    seen.clear()
    R.CFG_TL.n, R.CFG_TL.cut = 20, cut
    for i in range(20):
        DN._guided_denoise(step_index=i, video_guider=g, audio_guider=g)
    on = sum(1 for _, c, _ in seen if c == 3.0)
    if on != want_on or any(c != 1.0 or s != 0.0 for i, c, s in seen if i >= want_on):
        fails.append(f"cut {cut}: {on} guided steps, want {want_on}")
print({"fails": fails})
sys.exit(1 if fails else 0)
