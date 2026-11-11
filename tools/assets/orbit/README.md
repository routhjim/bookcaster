# Orbit skin: draw once, orbit with the video model, build the asset from the frames (experiment, 11-10)

A creature or prop is drawn once by the image model; the video model orbits the camera around it (one first clip
each way from the drawing: chained clips drift in look); the frames give the shape (multi-view image-to-3D) and
the skin (the frames projected back onto the mesh). The result goes through the normal chain (`run_asset.py`).
Everything that computes runs on a GPU; each step takes its card with `card.sh` (flock -n polling, GPU_HOLD).

| Step | Script | Device | What |
|---|---|---|---|
| 1 frames | ffmpeg | - | every 6th frame of the left-going clip, every 10th of the right-going one, as `NNN_<clip tag><frame>.png`, one path round the subject |
| 2 cameras | `../../../` lab `recon.py` (WorldMirror, `RECON_SIZE=672`, 36 frames, 15 GB) | XTX | per-frame cameras + depth in one world |
| 3 masks | `masks_cams.py` | XTX / iGPU | colour key against a smooth 2-D background, AND above the reconstructed floor (drops shadows), dark interior patches filled where the geometry says subject; azimuth per frame |
| 4 views | `pick_views.py` | - | front = most mirror-symmetric frame of the front clip, sides = front +- 90, back = front + 180, RGBA cut-outs; both left/right assignments as runs |
| 5 shape | `shape_mv.py` (Hunyuan3D-2mv via `tools/blockout_shape.py`) | XTX | one load, several runs; marching cubes and decimation are CPU inside the model code (measured below) |
| 6 fit | `fit.py` | XTX | mesh -> recon world by soft point-splat silhouettes (yaw grid, then Adam; one global focal aspect; then per-frame camera corrections); writes the mesh and the cameras in the asset frame |
| 7 UV + texels | `bl_prep.py` | XTX (Cycles) | smart UV, texel position / normal bakes |
| 8 view positions | `bl_views.py` | XTX (Cycles) | per camera, the surface point at every pixel (1 spp, no filter) |
| 9 skin | `project.py` | XTX | per texel: visible views, weights (facing, density, distance from silhouettes and depth steps), each clip skins its own half (`sides`, blended over `band_m` at the mid-plane), low/high bands, per-view gain match, unseen texels from the nearest seen surface, detail normal + roughness from the high-pass |
| 10 chain | `run_asset.py SPEC model_orbit_skin.py` | chain | spec keys `source_blend` (step 7's blend), `skin_dir` (step 9's maps), `keep_uvs: true` |
| rig | `rig_fit.py`, `bl_rig.py`, `ue_rig_poses.py` | XTX / UE editor | skeleton from a Reeb graph of the surface (geodesic bands from the snout tip), weights per graph node, test poses as one action, FBX into UE through `run_asset.py --ue-job` |

The scripts take every location as an argument; nothing story- or user-specific is stored here.
