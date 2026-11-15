"""Model script (run_asset.py) for an orbit-skinned asset: the UV'd mesh from bl_prep.py and the maps from
project.py, already in the asset frame (metres, front -Y, base on Z = 0, origin at the floor centre).

Spec keys: "source_blend" (bl_prep.py's <id>_uv.blend), "skin_dir" (project.py's textures folder), "keep_uvs": true.
The harness bakes this material through to its own maps on the same UVs.
"""
import os

import bpy

src = SPEC["source_blend"]
skin = SPEC["skin_dir"]
aid = SPEC["id"]
with bpy.data.libraries.load(src, link=False) as (df, dt):
    dt.objects = [n for n in df.objects if n == f"SM_{aid}"]
ob = dt.objects[0]
bpy.context.scene.collection.objects.link(ob)
ob.name = f"orbit_{aid}"
m = bpy.data.materials.new(f"skin_{aid}")
m.use_nodes = True
nt = m.node_tree
pr = nt.nodes["Principled BSDF"]
uv = nt.nodes.new("ShaderNodeUVMap")
uv.uv_map = ob.data.uv_layers[0].name


def tex(kind, colour):
    n = nt.nodes.new("ShaderNodeTexImage")
    n.image = bpy.data.images.load(os.path.join(skin, f"T_{aid}_{kind}.png"))
    n.image.colorspace_settings.name = "sRGB" if colour else "Non-Color"
    nt.links.new(uv.outputs["UV"], n.inputs["Vector"])
    return n


bc = tex("BaseColor", True)
nt.links.new(bc.outputs["Color"], pr.inputs["Base Color"])
orm = tex("ORM", False)
sep = nt.nodes.new("ShaderNodeSeparateColor")
nt.links.new(orm.outputs["Color"], sep.inputs["Color"])
nt.links.new(sep.outputs["Green"], pr.inputs["Roughness"])
nt.links.new(sep.outputs["Blue"], pr.inputs["Metallic"])
nm = tex("Normal", False)
nmap = nt.nodes.new("ShaderNodeNormalMap")
nmap.uv_map = uv.uv_map
nt.links.new(nm.outputs["Color"], nmap.inputs["Color"])
nt.links.new(nmap.outputs["Normal"], pr.inputs["Normal"])
pr.inputs["Specular IOR Level"].default_value = 0.4
ob.data.materials.clear()
ob.data.materials.append(m)
