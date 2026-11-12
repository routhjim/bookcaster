"""UE side: set body constraints on an existing MetaHuman Character asset (active), save, and report them before and
after. Run through the line as an editor script (run_asset.py --ue-job), then rebuild with char_render's "cloud"
and "export_geometry" steps.

    job: {"path": "/Game/Char/<name>", "body": {"Neck Length": 6.0, "Neck": 37.0}, "result": path}
"""
import json

import unreal

EAL = unreal.EditorAssetLibrary
mhs = unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)


def run(job):
    c = EAL.load_asset(job["path"])
    added = (not mhs.is_object_added_for_editing(c)) and mhs.try_add_object_to_edit(c)
    try:
        before = {str(b.name): [bool(b.is_active), round(float(b.target_measurement), 2)]
                  for b in mhs.get_body_constraints(c)}
        new = []
        for b in mhs.get_body_constraints(c):          # the array holds copies: rebuild it
            b = b.copy()
            if str(b.name) in job["body"]:
                b.target_measurement = float(job["body"][str(b.name)])
                b.is_active = True
            new.append(b)
        mhs.set_body_constraints(c, new)
        mhs.commit_body_state(c)
        after = {str(b.name): [bool(b.is_active), round(float(b.target_measurement), 2)]
                 for b in mhs.get_body_constraints(c)}
    finally:
        if added:
            mhs.remove_object_to_edit(c)
    EAL.save_asset(job["path"], False)
    r = {"ok": True, "before": {k: before[k] for k in job["body"] if k in before},
         "after": {k: after[k] for k in job["body"] if k in after}}
    json.dump(r, open(job["result"], "w"), indent=1)
    return r
