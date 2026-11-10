# MetaHuman lab (UE 5.8.1, Linux): scripted characters and stills

Tested 2026-11-10 on a box that has **no Epic sign-in**. All runs go through `mh_run.py`: a no-GPU commandlet
(`cmdlet`) or a full offscreen editor on one locked XTX (`render`). Settings come from environment variables (see the
docstring of `mh_run.py`); the lab project needs the plugins PythonScriptPlugin, EditorScriptingUtilities,
MetaHumanCharacter, MetaHuman, MetaHumanSDK, ControlRig, RigLogic, HairStrands, LiveLinkControlRig and
`r.SkinCache.CompileShaders=True`.

| Script | Where | What |
|---|---|---|
| `mh_introspect.py` | cmdlet | dumps every `unreal.*MetaHuman*` class, its members and docstrings, and the shipped presets |
| `mh_probe.py` | cmdlet | makes default, archetype-rigged and preset-copy characters and records the build gate's reason |
| `mh_make.py` | cmdlet | makes characters from a spec, offline: base (default or preset copy), archetype rig, skin, grooms, garment |
| `mh_render.py` | render | spawns the editor preview actor of each character in a lit level; close / full stills; face and body poses |
| `mh_build.py` | cmdlet | the signed-in route: auto-rig and texture download (cloud), then `build_meta_human` (assembly) |

```
python tools/assets/metahuman/mh_run.py cmdlet tools/assets/metahuman/mh_make.py --job make.json
python tools/assets/metahuman/mh_run.py render render.json          # job["script"] = mh_render.py
```

## What works without sign-in

- **Character asset**: `AssetTools.create_asset(..., unreal.MetaHumanCharacter, MetaHumanCharacterFactoryNew)` or
  `EditorAssetLibrary.duplicate_asset` of one of the 29 presets in `/MetaHumanCharacter/Optional/Presets`
  (scan that path with the asset registry first in a commandlet).
- **Skin**: textures are synthesized locally at 1k (the texture-synthesis model ships in the plugin).
  `commit_skin_settings`, `commit_eyes_settings`, `commit_makeup_settings`; set `preview_material_type = EDITABLE`.
- **Grooms and garment**: 95 groom wardrobe items (`/MetaHumanCharacter/Optional/Grooms/Bindings/<Slot dir>/WI_*`)
  and one outfit (`/MetaHumanCharacter/Optional/Clothing/WI_DefaultGarment`, a plain T-shirt/tank and shorts) via
  `internal_collection.try_add_item_from_wardrobe_item` + `default_instance.set_single_slot_selection`.
- **A face rig, but only the archetype face**: `import_from_face_dna(c, <MetaHumanCoreTech>/ArchetypeDNA/SKM_Face.dna,
  ImportFromDNAParams(import_whole_rig=True))` makes the character "rigged" locally. Its face shape is then the
  archetype's; skin, eyes, hair and body still vary.
- **Render**: `try_add_object_to_edit` -> `spawn_meta_human_actor(c, True)` -> `assemble_for_preview(c)` (needed, or
  grooms and outfit do not show) gives the editor preview actor (body + face skeletal meshes, strand grooms). It
  renders in an ordinary level with `take_high_res_screenshot`. Preset faces render but do not animate (no rig).
- **Face and body poses** (rigged characters only): a one-frame AnimSequence on the face skeleton with
  `CTRL_expressions_*` curves (e.g. `jawOpen`, `mouthCornerPullL/R`, `eyeLookLeftL/R`) or a shipped pose
  (`/MetaHumanCharacter/Optional/Animation/TemplateAnimations/Facial_Poses/*`), and a body AnimSequence on the
  body component (the shipped body range-of-motion clip, sampled at a scanned time), each applied as a paused
  single-node animation (`set_animation_mode(SINGLE_NODE)`, `set_animation`, `set_position`, re-set every shot:
  `override_animation_data` only took the first time).
- **Wrinkle/colour maps**: offline the full animated maps are white placeholders and bleach the face in any
  expression. `mh_render.py` (`"no_animated_maps": true`) points them at the neutral synthesized maps.

## What needs the cloud (and so a signed-in editor)

- `request_auto_rigging` (a rig fitted to a non-archetype face: presets, sculpted or conformed faces).
- `request_texture_sources` (high-resolution textures incl. animated maps). `build_meta_human` refuses without both:
  "Character is not rigged." / "The Character is missing textures, use Download Texture Sources to create them
  before assembling". So no assembled Blueprint offline.

Unauthenticated, a cloud request gets HTTP 401; the plugin then tries the stored EOS login and, failing that, opens
Epic's account-portal sign-in in a browser (`-NoMetaHumanAccountPortalLoginFallback` turns that into a plain
failure). There is no separate sign-in button: the first auto-rig triggers it. After one interactive sign-in EOS keeps
a persistent login that later (unattended) runs reuse (`EOS_LCT_PersistentAuth`). These two calls were not made here.
