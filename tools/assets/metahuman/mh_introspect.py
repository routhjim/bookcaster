"""UE side, no GPU: list the MetaHuman Python surface this build exposes and the state of the shipped presets.

    UnrealEditor-Cmd <project> -run=pythonscript -script=tools/assets/metahuman/mh_introspect.py -nullrhi
    (env MH_OUT = output JSON path)
"""

import json
import os
import traceback

import unreal

out = {"classes": {}, "presets": {}, "errors": []}


def members(obj):
    return sorted(n for n in dir(obj) if not n.startswith("_"))


base = set(dir(unreal.Object))
for name in sorted(n for n in dir(unreal) if "MetaHuman" in n or n in ("ImportFromDNAParams", "ImportErrorCode",
                                                                       "DNAAsset", "RigLogicLibrary")):
    cls = getattr(unreal, name)
    try:
        if isinstance(cls, type) and issubclass(cls, unreal.EnumBase):
            out["classes"][name] = {"enum": [m for m in members(cls) if m.isupper()]}
        elif isinstance(cls, type):
            own = [m for m in members(cls) if m not in base]
            doc = {}
            for m in own:
                d = getattr(getattr(cls, m), "__doc__", None)
                if d and callable(getattr(cls, m)):
                    doc[m] = d.strip().split("\n")[0][:300]
            out["classes"][name] = {"members": own, "doc": doc}
    except Exception as e:
        out["errors"].append(f"{name}: {e}")

ar = unreal.AssetRegistryHelpers.get_asset_registry()
ar.scan_paths_synchronous(["/MetaHumanCharacter/Optional/Presets"], True)
for p in sorted(str(a.package_name) for a in ar.get_assets_by_path("/MetaHumanCharacter/Optional/Presets", recursive=True)):
    try:
        c = unreal.load_asset(p)
        info = {"class": c.get_class().get_name()}
        for prop in ("has_high_resolution_textures", "preview_material_type"):
            try:
                info[prop] = str(c.get_editor_property(prop))
            except Exception as e:
                info[prop] = f"ERR {e}"
        for fn in ("has_face_dna", "has_body_dna", "has_synthesized_textures", "is_character_valid"):
            if hasattr(c, fn):
                try:
                    info[fn] = getattr(c, fn)()
                except Exception as e:
                    info[fn] = f"ERR {e}"
        out["presets"][p] = info
    except Exception:
        out["presets"][p] = {"error": traceback.format_exc()[-400:]}

json.dump(out, open(os.environ["MH_OUT"], "w"), indent=1)
unreal.log_warning(f"MH introspect: {len(out['classes'])} classes, {len(out['presets'])} presets")
