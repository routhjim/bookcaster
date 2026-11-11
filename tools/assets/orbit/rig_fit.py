"""Fit a creature skeleton and skin weights to a static mesh from its own shape (torch, on a GPU).

    HIP_VISIBLE_DEVICES=<card> <python with torch + trimesh> -I rig_fit.py MESH.ply OUT.json [--band 0.02]

The mesh is in the asset frame (metres, Z up, front -Y, base on Z = 0). Method (a Reeb graph of the surface):
1. area-weighted surface samples, a k-nearest-neighbour graph (edges across a thin gap whose normals face each
   other are dropped, so a wing lying on the flank is not glued to it);
2. geodesic distance from the snout tip (Bellman-Ford relaxation on the GPU), cut into bands of --band x the
   creature's length; each band's connected pieces are graph nodes at their centroids (a ring round a limb ->
   the limb's centre line); nodes in neighbouring bands that touch are linked, giving a tree rooted at the head;
3. leaves are labelled from where they end: back-most = tail (the snout -> tail path is the main line), near the floor = feet (grouped
   into up to four, left/right by x, front/back by y), high and wide = wing tips (per side; the farthest is the
   wing, the next ones fingers); the junctions where those paths leave each other place shoulders and hips on the main line;
4. bones are cut along those centre-line paths: spine (hip -> torso -> shoulder), neck, head, jaw, tail, per leg
   upper / lower / foot / toes, per wing arm / forearm / fingers; joint positions are path points, never fixed offsets;
5. every sample belongs to a node, every node to one path (side twigs such as horns or claws to their nearest
   path ancestor), so each sample gets one bone, blended linearly over a window at each joint; vertices take
   their nearest sample's weights, then a few rounds of smoothing over mesh edges.
OUT.json: bones [{name, parent, head, tail}], weights (per vertex: up to 4 [bone index, w]), labels, timings.
"""
import argparse
import json
import time

import numpy as np
import torch
import trimesh

ap = argparse.ArgumentParser()
ap.add_argument("mesh")
ap.add_argument("out")
ap.add_argument("--samples", type=int, default=60000)
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--band", type=float, default=0.02, help="geodesic band, fraction of the creature's length")
ap.add_argument("--blend", type=float, default=0.35, help="joint blend window, fraction of the shorter bone")
ap.add_argument("--neck-bones", type=int, default=4)
ap.add_argument("--tail-bones", type=int, default=6)
ap.add_argument("--spine-bones", type=int, default=3)
ap.add_argument("--fingers", type=int, default=3)
a = ap.parse_args()
t0 = time.time()
T = {}
dev = "cuda"
tm = trimesh.load(a.mesh, force="mesh")
V = torch.tensor(np.asarray(tm.vertices), dtype=torch.float32, device=dev)
Fc = torch.tensor(np.asarray(tm.faces), dtype=torch.long, device=dev)
lo, hi = V.min(0).values, V.max(0).values
L = float((hi - lo)[1])                       # length along the front axis
Hh = float(hi[2])
halfw = float(torch.maximum(hi[0].abs(), lo[0].abs()))
g = torch.Generator(device=dev).manual_seed(0)
tri = V[Fc]
cr = torch.linalg.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
area = cr.norm(dim=1)
fn = cr / area[:, None].clamp(min=1e-12)
fi = torch.multinomial(area, a.samples, replacement=True, generator=g)
r1 = torch.rand(a.samples, 1, device=dev, generator=g).sqrt()
r2 = torch.rand(a.samples, 1, device=dev, generator=g)
P = (1 - r1) * tri[fi, 0] + r1 * (1 - r2) * tri[fi, 1] + r1 * r2 * tri[fi, 2]
N = fn[fi]
n = len(P)

# ---- kNN graph (chunked), dropping edges across thin gaps (normals facing each other)
k = a.k
nbr = torch.empty(n, k, dtype=torch.long, device=dev)
nd = torch.empty(n, k, device=dev)
for s in range(0, n, 4096):
    d = torch.cdist(P[s:s + 4096], P)
    dv, di = d.topk(k + 1, largest=False)
    nbr[s:s + 4096], nd[s:s + 4096] = di[:, 1:], dv[:, 1:]
facing = (N[:, None, :] * N[nbr]).sum(-1)
dirv = P[nbr] - P[:, None, :]
across = ((N[:, None, :] * dirv).sum(-1) < 0) & (facing < -0.3)        # neighbour behind my surface, facing me
good = ~across & (nd < 4 * nd[:, :3].mean())
T["knn"] = round(time.time() - t0, 2)

# ---- geodesic distance from the torso
# source = the snout tip (front-most point near the mid-plane): geodesic bands from an extremity are cross-sections
# of the body, so limbs split off where they join it (a source on the belly gives discs, not rings)
inner = (P[:, 0].abs() < 0.15 * halfw)
root = int((P[:, 1] + (~inner).float() * 1e3).argmin())
gd = torch.full((n,), float("inf"), device=dev)
gd[root] = 0
w = torch.where(good, nd, torch.full_like(nd, float("inf")))
for it in range(5000):
    cand = (gd[nbr] + w).min(1).values
    new = torch.minimum(gd, cand)
    # symmetric relaxation: push along reverse edges too
    rev = torch.full((n,), float("inf"), device=dev).scatter_reduce(0, nbr.view(-1), (gd[:, None] + w).view(-1),
                                                                    "amin")
    new = torch.minimum(new, rev)
    if torch.equal(new, gd):
        break
    gd = new
reach = torch.isfinite(gd)
T["geodesic"] = round(time.time() - t0, 2)
T["geodesic_iters"] = it

# ---- bands, components, nodes
h = a.band * L
band = torch.where(reach, (gd / h).floor(), torch.full_like(gd, -1)).long()
lab = torch.arange(n, device=dev)
same = (band[:, None] == band[nbr]) & good & reach[:, None]
src = torch.arange(n, device=dev)[:, None].expand(-1, k)[same]
dst = nbr[same]
for _ in range(10000):
    m1 = lab.scatter_reduce(0, src, lab[dst], "amin")
    m1 = m1.scatter_reduce(0, dst, lab[src], "amin")
    if torch.equal(m1, lab):
        break
    lab = m1
lab = torch.where(reach, lab, torch.full_like(lab, -1))
ul, comp = torch.unique(lab, return_inverse=True)
nn_ = len(ul)
cnt = torch.zeros(nn_, device=dev).index_add_(0, comp, torch.ones(n, device=dev))
cen = torch.zeros(nn_, 3, device=dev).index_add_(0, comp, P) / cnt[:, None]
nband = torch.zeros(nn_, dtype=torch.long, device=dev).scatter_reduce(0, comp, band, "amax")
# node links between neighbouring bands
e_s, e_d = comp[:, None].expand(-1, k)[good], comp[nbr][good]
ok = (nband[e_d] == nband[e_s] - 1) & (ul[e_s] >= 0) & (ul[e_d] >= 0)
pairs = torch.stack([e_s[ok], e_d[ok]], 1)
up_, upc = torch.unique(pairs, dim=0, return_counts=True)
T["nodes"] = round(time.time() - t0, 2)
C = cen.cpu().numpy()
NB = nband.cpu().numpy()
cnt_np = cnt.cpu().numpy()
valid = (ul >= 0).cpu().numpy()
parent = -np.ones(nn_, int)
best = np.zeros(nn_)
for (c_, p_), m in zip(up_.cpu().numpy(), upc.cpu().numpy()):
    if m > best[c_]:
        best[c_], parent[c_] = m, p_
rootn = int(comp[root])
children = {i: [] for i in range(nn_)}
for c_ in range(nn_):
    if parent[c_] >= 0:
        children[parent[c_]].append(c_)
# nodes reachable from the root node through parent links
inroot = np.zeros(nn_, bool)
stack = [rootn]
while stack:
    x = stack.pop()
    inroot[x] = True
    stack += children[x]
leaves = [i for i in range(nn_) if inroot[i] and not children[i] and i != rootn]


def path(i):
    out = [i]
    while out[-1] != rootn and parent[out[-1]] >= 0:
        out.append(parent[out[-1]])
    return out[::-1]


# ---- label leaves
Lp = {i: path(i) for i in leaves}
glen = {i: NB[i] * h for i in leaves}
head = -1                                                               # the root node is the head
tail = max(leaves, key=lambda i: C[i, 1])
feet_c = [i for i in leaves if C[i, 2] < 0.12 * Hh and i not in (tail, head)]
# a tail lying on the floor runs back from the hips: drop "feet" that lie on the tail's path region (behind hips)
groups = []
for i in sorted(feet_c, key=lambda i: -glen[i]):
    for gq in groups:
        if np.linalg.norm(C[i, :2] - C[gq[0], :2]) < 0.12 * L:
            gq.append(i)
            break
    else:
        groups.append([i])
groups = [gq for gq in groups if len(gq) and abs(C[gq[0], 1] - C[tail, 1]) > 0.2 * L]
groups = sorted(groups, key=lambda gq: -sum(cnt_np[x] for x in gq))[:4]
feet = {}
tips = [max(gq, key=lambda i: glen[i]) for gq in groups]
ymid = float(np.median([C[i, 1] for i in tips])) if tips else 0.0
for lf in tips:
    side = "l" if C[lf, 0] > 0 else "r"                    # creature's left = +X (front -Y, up +Z)
    end = "front" if C[lf, 1] < ymid else "back"
    feet[f"leg_{end}_{side}"] = lf
wing_c = [i for i in leaves if C[i, 2] > 0.35 * Hh and abs(C[i, 0]) > 0.3 * halfw and i not in (head, tail)]
wings = {}
for side, sgn in (("l", 1), ("r", -1)):
    cands = sorted([i for i in wing_c if np.sign(C[i, 0]) == sgn], key=lambda i: -glen[i])
    if cands:
        wings[side] = cands
T["labels"] = round(time.time() - t0, 2)


def diverge(p1, p2):
    """index in p1 of the last node shared with p2"""
    j = 0
    while j < min(len(p1), len(p2)) and p1[j] == p2[j]:
        j += 1
    return j - 1


def pts(p):
    return C[p]


def arclen(xyz):
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xyz, axis=0), axis=1))]
    return d


def at(xyz, s):
    d = arclen(xyz)
    s = np.clip(s, 0, d[-1])
    j = int(np.clip(np.searchsorted(d, s) - 1, 0, len(d) - 2))
    t = (s - d[j]) / max(d[j + 1] - d[j], 1e-9)
    return xyz[j] * (1 - t) + xyz[j + 1] * t


bones = []          # name, parent, head, tail
assign = {}         # node -> (bone chain list of (name, s0, s1)), s along path
chains = []         # (node path, [(bone name, s0, s1)], xyz)


def add_chain(nodes, names, fracs, parent_name, start_xyz=None):
    xyz = pts(nodes)
    if start_xyz is not None:
        xyz = np.vstack([start_xyz, xyz])
        nodes = [nodes[0]] + list(nodes)
    d = arclen(xyz)
    cuts = [f * d[-1] for f in fracs]
    segs = []
    pn = parent_name
    for b, (s0, s1) in enumerate(zip(cuts[:-1], cuts[1:])):
        bones.append({"name": names[b], "parent": pn, "head": at(xyz, s0).tolist(), "tail": at(xyz, s1).tolist()})
        segs.append((names[b], s0, s1))
        pn = names[b]
    chains.append((nodes, segs, xyz))
    return pn


main = Lp[tail]                                                        # snout -> tail tip
# shoulder: where the front feet and the wings leave the main path; hip: where the back feet leave it
front_div = [diverge(main, Lp[f]) for k_, f in feet.items() if "front" in k_] + \
            [diverge(main, Lp[w[0]]) for w in wings.values()]
back_div = [diverge(main, Lp[f]) for k_, f in feet.items() if "back" in k_]
js = int(np.median(front_div)) if front_div else len(main) // 4
jh = int(np.median(back_div)) if back_div else len(main) // 2
js, jh = max(1, min(js, jh - 2)), max(js + 2, jh)
spine_nodes = main[js:jh + 1][::-1]                                    # hip -> shoulder
nb = a.spine_bones
cs = C[main[(js + jh) // 2]]
bones.append({"name": "root", "parent": None, "head": [float(cs[0]), float(cs[1]), 0.0],
              "tail": [float(cs[0]), float(cs[1]), float(cs[2]) * 0.5]})
add_chain(spine_nodes, [f"spine_{i}" for i in range(nb)], np.linspace(0, 1, nb + 1), "root")
spine_top = f"spine_{nb - 1}"
# neck + head: the head is the last ~12 % of the creature's length along the path to the snout
neck_head = main[:js + 1][::-1]
xyz = pts(neck_head)
dl = arclen(xyz)[-1]
hf = min(0.12 * L / max(dl, 1e-6), 0.5)
nk = a.neck_bones
fr = list(np.linspace(0, 1 - hf, nk + 1)) + [1.0]
last = add_chain(neck_head, [f"neck_{i}" for i in range(nk)] + ["head"], fr, spine_top)
hb = bones[-1]
hh, ht = np.array(hb["head"]), np.array(hb["tail"])
# jaw: from the head's base, under the head centre line, to just behind the snout tip
jaw_head = hh + (ht - hh) * 0.15 - np.array([0, 0, 0.18 * np.linalg.norm(ht - hh)])
jaw_tail = hh + (ht - hh) * 0.9 - np.array([0, 0, 0.25 * np.linalg.norm(ht - hh)])
bones.append({"name": "jaw", "parent": "head", "head": jaw_head.tolist(), "tail": jaw_tail.tolist()})
# tail
add_chain(main[jh:], [f"tail_{i}" for i in range(a.tail_bones)], np.linspace(0, 1, a.tail_bones + 1), "spine_0")
# legs
for name, lf in feet.items():
    p = Lp[lf]
    j = diverge(main, p)
    par = spine_top if "front" in name else "spine_0"
    add_chain(p[j:], [f"{name}_upper", f"{name}_lower", f"{name}_foot", f"{name}_toes"], [0, 0.4, 0.75, 0.9, 1.0],
              par)
# wings: arm / forearm to the wrist, fingers from there to the other tips
for side, cands in wings.items():
    wp = Lp[cands[0]]
    j = diverge(main, wp)
    wn = f"wing_{side}"
    add_chain(wp[j:], [f"{wn}_arm", f"{wn}_forearm", f"{wn}_finger_0"], [0, 0.33, 0.62, 1.0], spine_top)
    wrist_s = 0.62
    xyzw = pts(wp[j:])
    wrist = at(xyzw, wrist_s * arclen(xyzw)[-1])
    used = 1
    for c_ in cands[1:]:
        if used >= a.fingers:
            break
        fp = Lp[c_]
        jj = diverge(wp, fp)
        if jj < j + 1 or glen[c_] < 0.5 * glen[cands[0]]:
            continue
        add_chain(fp[jj:], [f"{wn}_finger_{used}"], [0, 1.0], f"{wn}_forearm", start_xyz=wrist)
        used += 1
T["bones"] = round(time.time() - t0, 2)

# ---- weights: node -> chain (deepest path claiming it), then bone by arc length with blending at joints
bidx = {b["name"]: i for i, b in enumerate(bones)}
node_bones = {}
for nodes, segs, xyz in chains:
    d = arclen(xyz)
    for j_, nd_ in enumerate(nodes):
        s = d[j_]
        ws = {}
        for b, (nm, s0, s1) in enumerate(segs):
            ln = s1 - s0
            win = a.blend * ln
            if s0 - win / 2 <= s <= s1 + win / 2:
                lo_w = 1.0 if (b == 0 or s >= s0 + win / 2) else (s - (s0 - win / 2)) / max(win, 1e-9)
                hi_w = 1.0 if (b == len(segs) - 1 or s <= s1 - win / 2) else ((s1 + win / 2) - s) / max(win, 1e-9)
                ws[nm] = max(0.0, min(lo_w, hi_w))
        if not ws:
            ws[segs[0][0] if s < segs[0][1] else segs[-1][0]] = 1.0
        node_bones.setdefault(nd_, []).append(ws)
# a node on several chains (shared prefix): the spine/parent chain was added first and is kept
nodew = {}
for nd_, lst in node_bones.items():
    nodew[nd_] = lst[0]
# nodes on no chain: nearest ancestor that has weights
for i in range(nn_):
    if i in nodew or not inroot[i]:
        continue
    x = i
    while x >= 0 and x not in nodew:
        x = parent[x]
    nodew[i] = nodew.get(x, {"root": 1.0})
nb_ = len(bones)
Wn = np.zeros((nn_, nb_), np.float32)
for i, ws in nodew.items():
    s = sum(ws.values()) or 1.0
    for nm, v in ws.items():
        Wn[i, bidx[nm]] = v / s
Wn[~inroot] = 0
Wn[~inroot, bidx["root"]] = 1
# jaw: head-bone samples below the head line and ahead of the jaw pivot
Ws = torch.tensor(Wn, device=dev)[comp]                               # samples
# a band ring that still holds the torso and the top of a limb (fused leg/wing roots) gives its samples to the
# body chain; let each sample move to a bone adjacent to its node's bones (parent / child) when that bone's
# segment is nearer: distance-based within the graph neighbourhood only, so a wing never takes a leg's samples
par_i = [bidx[b["parent"]] if b["parent"] else -1 for b in bones]
adj = np.eye(nb_, dtype=bool)
for i, p_ in enumerate(par_i):
    if p_ >= 0:
        adj[i, p_] = adj[p_, i] = True
adj_t = torch.tensor(adj, device=dev)
allowed = (Ws > 0).float() @ adj_t.float() > 0                         # samples x bones
allowed[:, bidx["root"]] = Ws[:, bidx["root"]] > 0.5                   # the root bone only keeps its own
bh = torch.tensor([b["head"] for b in bones], dtype=torch.float32, device=dev)
bt = torch.tensor([b["tail"] for b in bones], dtype=torch.float32, device=dev)
seg = bt - bh
tt_ = (((P[:, None, :] - bh[None]) * seg[None]).sum(-1) / (seg * seg).sum(-1).clamp(min=1e-9)[None]).clamp(0, 1)
dseg = (P[:, None, :] - (bh[None] + tt_[..., None] * seg[None])).norm(dim=-1)
dseg = torch.where(allowed, dseg, torch.full_like(dseg, 1e6))
dmin = dseg.min(1, keepdim=True).values
wd = torch.exp(-((dseg - dmin) / (0.04 * L)) ** 2) * allowed.float()
wd = wd / wd.sum(1, keepdim=True).clamp(min=1e-9)
Ws = 0.35 * Ws + 0.65 * wd
hd = bidx["head"]
hv = torch.tensor(ht - hh, dtype=torch.float32, device=dev)
rel = P - torch.tensor(jaw_head, dtype=torch.float32, device=dev)
along = (rel @ hv) / (hv @ hv)
below = P[:, 2] < torch.tensor(hh, dtype=torch.float32, device=dev)[2] + (along.clamp(0, 1) * hv[2]) - 0.05 * hv.norm()
jm = (Ws[:, hd] > 0.5) & (along > 0.05) & below
Ws[jm, bidx["jaw"]] = Ws[jm, hd]
Ws[jm, hd] = 0
# vertices: nearest sample, then smoothing over mesh edges
nv = len(V)
vi = torch.empty(nv, dtype=torch.long, device=dev)
for s in range(0, nv, 8192):
    vi[s:s + 8192] = torch.cdist(V[s:s + 8192], P).argmin(1)
Wv = Ws[vi]
E = torch.cat([Fc[:, [0, 1]], Fc[:, [1, 2]], Fc[:, [2, 0]]])
E = torch.cat([E, E.flip(1)])
deg = torch.zeros(nv, device=dev).index_add_(0, E[:, 0], torch.ones(len(E), device=dev))
for _ in range(6):
    acc = torch.zeros_like(Wv).index_add_(0, E[:, 0], Wv[E[:, 1]])
    Wv = 0.5 * Wv + 0.5 * acc / deg.clamp(min=1)[:, None]
top = Wv.topk(4, dim=1)
tw = top.values / top.values.sum(1, keepdim=True).clamp(min=1e-9)
T["weights"] = round(time.time() - t0, 2)
out = {"bones": bones, "labels": {"head": int(head), "tail": int(tail), "feet": {k_: int(v) for k_, v in feet.items()},
                                  "wings": {k_: [int(x) for x in v[:4]] for k_, v in wings.items()}},
       "nodes": int(nn_), "leaves": len(leaves), "samples": n, "reached": float(reach.float().mean()),
       "verts": nv, "verts_xyz": np.round(V.cpu().numpy(), 4).tolist(), "weights_idx": top.indices.cpu().numpy().astype(int).tolist(),
       "weights_w": np.round(tw.cpu().numpy(), 4).tolist(),
       "node_xyz": C[inroot].round(3).tolist(), "times_s": T}
json.dump(out, open(a.out, "w"))
T["total"] = round(time.time() - t0, 2)
print(json.dumps({"bones": [b["name"] for b in bones], "feet": list(feet), "wings": {k_: len(v) for k_, v in wings.items()},
                  "nodes": nn_, "leaves": len(leaves), "reached": out["reached"], "times_s": T}))
