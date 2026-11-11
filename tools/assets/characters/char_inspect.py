"""Editor-side (scheduler --ue-job): spawn an assembled MetaHuman Blueprint and list its components, the meshes they
hold (now and after some ticks), and the Python methods for adding components, so the render job can find them.

JOB = {"bp": "/Game/MetaHumans/X/BP_X", "result": path, "ticks": 30}
"""

import json
import traceback

import unreal

EAL = unreal.EditorAssetLibrary
act = unreal.get_editor_subsystem(unreal.EditorActorSubsystem)


def describe(a):
    rows = []
    for c in a.get_components_by_class(unreal.ActorComponent):
        row = {"name": c.get_name(), "class": c.get_class().get_name()}
        if isinstance(c, unreal.SkinnedMeshComponent):
            m = c.get_skinned_asset()
            row["mesh"] = m.get_path_name() if m else None
            row["visible"] = c.is_visible()
        rows.append(row)
    return rows


def run(job):
    r = {}
    try:
        bp = EAL.load_asset(job["bp"])
        r["bp_class"] = bp.get_class().get_name()
        try:
            r["parent"] = str(bp.get_editor_property("parent_class"))
        except Exception as e:
            r["parent"] = f"ERR {e}"
        try:
            sub = unreal.get_engine_subsystem(unreal.SubobjectDataSubsystem)
            hs = sub.k2_gather_subobject_data_for_blueprint(bp)
            L = unreal.SubobjectDataBlueprintFunctionLibrary
            r["scs"] = []
            for h in hs:
                o = L.get_object(L.get_data(h))
                r["scs"].append([o.get_name() if o else None, o.get_class().get_name() if o else None])
        except Exception as e:
            r["scs_err"] = str(e)
        try:
            unreal.BlueprintEditorLibrary.compile_blueprint(bp)
            r["compiled"] = True
        except Exception as e:
            r["compiled"] = f"ERR {e}"
        r["folder"] = EAL.list_assets(job["bp"].rsplit("/", 1)[0], recursive=False, include_folder=False)
        a = act.spawn_actor_from_object(bp, unreal.Vector(0, 0, 0), unreal.Rotator(0, 0, 0))
        r["actor_class"] = a.get_class().get_name()
        r["gen_class"] = str(bp.generated_class())
        r["at_spawn"] = describe(a)
        for _ in range(int(job.get("ticks", 30))):
            yield
        r["after_ticks"] = describe(a)
        r["skel_by_class"] = [[c.get_name(), (c.get_skinned_asset().get_path_name() if c.get_skinned_asset() else None)]
                              for c in a.get_components_by_class(unreal.SkeletalMeshComponent)]
        props = {}
        for nm in ("Body", "Face", "body", "face", "root_component"):
            try:
                v = a.get_editor_property(nm)
                props[nm] = [v.get_name(), v.get_class().get_name()] if v else None
            except Exception as e:
                props[nm] = f"ERR {str(e)[:80]}"
        r["props"] = props
        r["root"] = str(a.root_component)
        try:
            r["children"] = [str(c) for c in a.root_component.get_children_components(True)]
        except Exception as e:
            r["children"] = str(e)
        r["add_methods"] = [m for m in dir(a) if "component" in m.lower() and ("add" in m.lower() or "create" in m.lower())]
        r["bp_props"] = [m for m in dir(a) if not m.startswith("_")][:400]
        act.destroy_actor(a)
    except Exception:
        r["error"] = traceback.format_exc()
    json.dump(r, open(job["result"], "w"), indent=1, default=str)
    return {"ok": "error" not in r}
