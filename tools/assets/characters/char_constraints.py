"""UE side: set body constraints on an existing MetaHuman Character asset (active), save, and report them before and
after. Run through the line as an editor script (run_asset.py --ue-job), then rebuild with char_render's "cloud"
and "export_geometry" steps.

    job: {"path": "/Game/Char/<name>", "body": {"Neck Length": 6.0, "Neck": 37.0}, "result": path,
          "add": {"Neck Length": -2.0}, "min": {"Neck Length": 4.5},   (add: relative to the current value)
          "copy_from": "/Game/Char/<other>"}                            (the changes go on a fresh copy of it)
"""
import json

import unreal

EAL = unreal.EditorAssetLibrary
mhs = unreal.get_editor_subsystem(unreal.MetaHumanCharacterEditorSubsystem)


def run(job):
    if job.get("copy_from"):
        # the changes go on a copy (a MetaHuman another character also uses is never changed)
        if EAL.does_asset_exist(job["path"]):
            EAL.delete_asset(job["path"])
        if not EAL.duplicate_asset(job["copy_from"], job["path"]):
            raise RuntimeError(f"copy of {job['copy_from']} failed")
    c = EAL.load_asset(job["path"])
    added = (not mhs.is_object_added_for_editing(c)) and mhs.try_add_object_to_edit(c)
    try:
        before = {str(b.name): [bool(b.is_active), round(float(b.target_measurement), 2)]
                  for b in mhs.get_body_constraints(c)}
        new, clamped = [], {}
        for b in mhs.get_body_constraints(c):          # the array holds copies: rebuild it
            b = b.copy()
            n = str(b.name)
            want = None
            if n in job.get("body", {}):
                want = float(job["body"][n])
            elif n in job.get("add", {}):
                want = max(float(b.target_measurement) + float(job["add"][n]), float(job.get("min", {}).get(n, -1e9)))
            if want is not None:
                # inside the constraint's own range: a target past it (Neck 30.0 under its 30.08) left the solved
                # body as it was, with no error (11-11)
                lo, hi = float(b.min_measurement) + 0.05, float(b.max_measurement) - 0.05
                if not lo <= want <= hi:
                    clamped[n] = [round(want, 2), round(min(max(want, lo), hi), 2)]
                b.target_measurement = min(max(want, lo), hi)
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
    keys = list(job.get("body", {})) + list(job.get("add", {}))
    r = {"ok": True, "clamped": clamped, "before": {k: before[k] for k in keys if k in before}, "after": {k: after[k] for k in keys if k in after}}
    json.dump(r, open(job["result"], "w"), indent=1)
    return r
