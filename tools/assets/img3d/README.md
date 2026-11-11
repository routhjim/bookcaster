# img3d: picture to 3D (TRELLIS.2) for props, creatures and people

One drawing in, a textured mesh out, through the line. TRELLIS.2 (MIT) runs in the trellis.cpp port (MIT), installed
under `$ASSETS_UE_HOME/tools/trellis` (bin-rocm/, bin-vulkan/; weights in `~/models/trellis2-gguf`).

| File | What |
|---|---|
| `img3d.py` | drawing -> `.glb` (+ base-colour atlas, cut-out, per-stage times in `.json`) on the line's station device |
| `model_img3d.py` | asset-chain model script: the `.glb` with its own texture, turned, scaled, decimated to the tier budget; bump and roughness derived from the colour so the harness bakes normal and ORM maps on its GPU |
| `bl_view.py` | Cycles views of a `.glb` / `.blend` on a GPU: ring, ortho, aimed close-ups (`"target": "top:0.13"` = a head) |
| `bl_combo.py` | fit a generated figure over the MetaHuman body: `mode outfit` (its clothes on the MetaHuman head and hands) or `mode statue` (the whole figure rigged to the body's skeleton) |
| `stage.py`, `ue_stage_figure.py` | a plain-mesh figure (a statue) on a set mark with the scene tool's cameras and depth guides |
| `grid_sheet.py` | captioned picture grids |

## Speed (11-11, dragon drawing, seed 42)

| Device | Build | 1024 | 1536 |
|---|---|---|---|
| card 65 (XTX) | ROCm | 154 s | 352 s |
| card 65 (XTX) | Vulkan | 271 s | 607 s |
| iGPU | Vulkan | 738 s | ~31 min (and ~50 GiB of system RAM) |

About 30 s of every run is CPU inside the port (xatlas UV unwrap ~26 s, mesh clean-up). Both resolutions end at
~300k faces, so 1536 adds surface detail, not shape. Default: ROCm, 1024, card 65.

```
ASSETS_TRELLIS_OMP=<folder with libomp.so> \
python tools/assets/run_asset.py --line-cmd engine shape --id NAME --detach -- \
    python tools/assets/img3d/img3d.py DRAWING.png OUT.glb --res 1024 --build rocm
```

`--line-cmd engine` borrows the engine card while the engine has no work (engine jobs go first; the editor unloads
and reloads for the next engine job, ~13 s). Submit one drawing per entry: a long batch in one entry holds the
engine away from everyone else.

## A character from a sheet: the combo route (generated clothes, MetaHuman head, skeleton and hands)

Needs: the character sheet (a drawing), a MetaHuman character of that person in the project (face, groom,
skeleton), and that character's body and head exported as FBX (the character tools' `export_geometry` job writes
them; the first figure used `MHC_<name>_Body.fbx` and `_Head.fbx`).

1. **Spread-pose drawing** (image station, edit model, ~40 s a picture): the sheet as Picture 1 and a prompt asking
   for the same person, same clothes, full length, straight from the front, arms held out and down about 45
   degrees, hands open, legs slightly apart, coat hanging open and clear of the arms, plain light grey background.
   Two seeds, photographic look (the ink look generates too, the photographic one gave the cleaner surface).
   Jobs file format and runner: `tools/assets/qwen_edit.py` (run it through `run_asset.py --line-cmd image refs --
   bash tools/assets/orbit/card.sh 05 <python> -I tools/assets/qwen_edit.py jobs.json`).
2. **Generate**: `img3d.py` on the drawing (above). Look at it with `bl_view.py` (front, back, head) before going on.
3. **Fit, cut, weights, export** (`bl_combo.py`, Blender on the iGPU station, ~20-35 s, part of it CPU):
   ```
   JOB.json {"body_fbx": ".../<char>_Body.fbx", "head_fbx": ".../<char>_Head.fbx",
             "figure_glb": ".../figure.glb", "out": ".../combo", "skirt_pelvis": 1.0}
   python tools/assets/run_asset.py --line-cmd igpu combo --id NAME --detach -- \
       flock /tmp/gpu-igpu.lock <blender> -b --factory-startup -t 0 --python tools/assets/img3d/bl_combo.py -- JOB.json
   ```
   What it does: scales the figure so its shirt collar meets the body's neck (a drawn head is often large); centres
   the body's chest between the figure's chest front and back; turns the body's upper arms and thighs onto the
   figure's wrists and shoes (found from the texture colours: skin, dark shoes); stretches the sleeves to the
   body's wrists; cuts the head with a tilted collar plane and the hands at the wrists; pushes everything out of the
   body and head by 1.4 cm; copies the posed body's weights and then applies the region rules (below); keeps the
   figure's hair as a cap on the back and sides of the head (`hair_cap`); takes the body's own hands into the
   outfit as slot `M_skin`; makes that pose the bind pose and exports `outfit.fbx` (centimetres) with
   `textures/T_outfit_{BaseColor,Normal,RMA}.png`. `report.json` has every number.
4. **Import** (engine, ~15 s): a character-tools job (`tools/assets/characters/char_submit.py`, from the worktree
   that has the scene tool you use) with
   `"import_outfit": {"outfit_fbx": ".../outfit.fbx", "textures": ".../textures", "dest": "/Game/Char/Outfit<X>",
   "cloth_slots": false}` and a character `{"path": <the MetaHuman Blueprint>, "garments": ["@outfit"],
   "hide_body": true}`. Shots with hands checks need the M_skin slot (the outfit mode has it).
5. **character.json** in `<library>/character/<id>/`: a copy of the MetaHuman character's entry with
   `"garments": ["/Game/Char/Outfit<X>/Outfit/outfit"]`, `"hide_body": true`, `"body_swap": null`. The scene tool
   then poses it like any library character.

Region rules (in `bl_combo.py`, general; the numbers are job keys):
- only sleeves follow the arms: a vertex keeps arm weights when it is nearer the arm's surface than the torso's,
  clear of the torso (`sleeve_off_torso_m`), or at the armhole; everything else takes the weights of the body
  without its arms;
- cloth below the waist that is off the legs (`skirt_off_leg_m`, 5 cm) is skirt: it follows the pelvis by
  `skirt_pelvis` and the nearer thigh by the rest. **A long dress or a frock coat: `skirt_pelvis` 1.0** (hangs from
  the pelvis, does not split with the legs; it will pass through the legs on a big step, and a seated pose needs the
  skirt to drape, which this does not do). Trousers stay on the legs;
- shoes (under 11 cm) are rigid on the foot bones; weights are smoothed over the mesh (8 rounds, 25 at the
  armholes);
- mode statue: the head (above the collar plane) is rigid on the head bone and scaled to enclose the hidden
  MetaHuman head (else its face shows through); the hands past the wrists are rigid on the hand bones.

Specific to the first figure (a frock coat, cravat, black hair) and to check on the next one: the colour tests
(skin, white collar, dark hair, dark shoes) in `is_skin` and the collar/hair thresholds; the cravat smoothing
(light unsaturated cloth in front of the neck); the shoulder pull-in (`shoulder_gap_m`). A woman in a long dress:
set `skirt_pelvis` 1.0, check that the collar is found (a high dress collar is not white: pass `"match": "height"`
to scale by height instead), and expect the dress to hang straight in a seated pose.
