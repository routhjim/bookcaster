"""Fit a creature skeleton and skin weights to a static mesh from its own shape (torch, on a GPU).

    HIP_VISIBLE_DEVICES=<card> <python with torch + trimesh> -I rig_fit.py MESH.ply OUT.json [--band 0.02]

The mesh is in the asset frame (metres, Z up, front -Y, base on Z = 0). Method (a Reeb graph of the surface):
0. the left/right mirror plane is found from the shape (the x whose mirror image of the surface best overlaps the
   surface), not from the bounding box, which a tail swept to one side pulls off the body; left/right is taken
   from that plane;
1. area-weighted surface samples, a k-nearest-neighbour graph (edges across a thin gap whose normals face each
   other are dropped, so a wing lying on the flank is not glued to it);
2. geodesic distance from the snout tip (Bellman-Ford relaxation on the GPU), cut into bands of --band x the
   creature's length; each band's connected pieces are graph nodes at their centroids (a ring round a limb ->
   the limb's centre line); nodes in neighbouring bands that touch are linked, giving a tree rooted at the head;
3. leaves are labelled from where they end: back-most = tail (the snout -> tail path is the main line), near the floor = feet (grouped
   into up to four, left/right by x, front/back by y), high and wide = wing tips (per side; the farthest is the
   wing, the next ones fingers); the junctions where those paths leave each other place shoulders and hips on the main line;
4. the neck starts where the main line's rings, walked from the shoulder junction toward the head past the chest's
   largest ring, shrink to --neck-ring x that ring (the chest belongs to the spine, not the neck);
5. each leg is walked again from its own foot (geodesic bands from the foot tip = the leg's cross-sections): its
   root (hip / shoulder joint, inside the thigh or upper arm) is the last ring before one larger than --limb-root
   x the torso's largest ring; the snout's bands cut a hanging leg lengthwise and split it off only below the belly;
6. bones are cut along those centre lines: spine (hip -> chest -> neck base), neck, head, jaw, tail, per leg
   upper / lower / foot / toes (hip ring -> foot tip), per wing arm / forearm / fingers; joint positions are path
   points, never fixed offsets;
7. every sample belongs to a node, every node to one path (side twigs such as horns or claws to their nearest
   path ancestor), so each sample gets one bone, blended linearly over a window at each joint; a leg's samples (the
   surface joined to its foot below the root ring) take the leg bones by their distance from the foot; vertices
   take their nearest sample's weights, then a few rounds of smoothing over mesh edges.
OUT.json: bones [{name, parent, head, tail}], weights (per vertex: up to 4 [bone index, w]), labels, timings, and
for checking: leaf_info, main_xyz / main_rad (the main line's ring centres and sizes), junctions, limb_rings.
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
ap.add_argument("--neck-ring", type=float, default=0.55,
                help="the neck starts at the first ring (from the chest) down to this x the chest's largest ring")
ap.add_argument("--limb-root", type=float, default=0.5,
                help="a leg ends at its first cross-section larger than this x the torso's (see the legs step)")
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

# ---- the left/right mirror plane x = x0: a tail swept to one side or a raised wing moves the bounding box's centre
# off the body's mid-plane (a mid-plane taken from the box can put the snout search on a front foot). x0 is the
# plane whose mirror image of the surface lies closest to the surface itself (median distance, so the asymmetric
# parts do not decide it): a coarse grid over the box, then a fine one round the best
sub = P[torch.randperm(n, generator=g, device=dev)[:4000]]


def mirror_cost(x0):
    m_ = sub.clone()
    m_[:, 0] = 2 * x0 - m_[:, 0]
    return float(torch.cdist(m_, sub).min(1).values.median())


xs = np.linspace(float(lo[0]), float(hi[0]), 61)
x0 = float(min(xs, key=mirror_cost))
step = xs[1] - xs[0]
x0 = float(min(np.linspace(x0 - step, x0 + step, 21), key=mirror_cost))
V = V - torch.tensor([x0, 0.0, 0.0], device=dev)
P = P - torch.tensor([x0, 0.0, 0.0], device=dev)
lo, hi = V.min(0).values, V.max(0).values
halfw = float(torch.maximum(hi[0].abs(), lo[0].abs()))
T["mid_x"] = round(x0, 4)

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
w = torch.where(good, nd, torch.full_like(nd, float("inf")))


def geodesic(src_i):
    """graph distance from sample src_i (Bellman-Ford relaxation over the kNN graph, both edge directions)"""
    gd_ = torch.full((n,), float("inf"), device=dev)
    gd_[src_i] = 0
    for it_ in range(5000):
        cand = (gd_[nbr] + w).min(1).values
        new = torch.minimum(gd_, cand)
        rev = torch.full((n,), float("inf"), device=dev).scatter_reduce(0, nbr.view(-1), (gd_[:, None] + w).view(-1),
                                                                        "amin")
        new = torch.minimum(new, rev)
        if torch.equal(new, gd_):
            break
        gd_ = new
    return gd_, it_


def band_parts(gd_, h_):
    """(band per sample, connected-piece label per sample within its band; -1 where unreached)"""
    reach_ = torch.isfinite(gd_)
    band_ = torch.where(reach_, (gd_ / h_).floor(), torch.full_like(gd_, -1)).long()
    lab_ = torch.arange(n, device=dev)
    same = (band_[:, None] == band_[nbr]) & good & reach_[:, None]
    src_ = torch.arange(n, device=dev)[:, None].expand(-1, k)[same]
    dst_ = nbr[same]
    for _ in range(10000):
        m1 = lab_.scatter_reduce(0, src_, lab_[dst_], "amin")
        m1 = m1.scatter_reduce(0, dst_, lab_[src_], "amin")
        if torch.equal(m1, lab_):
            break
        lab_ = m1
    return band_, torch.where(reach_, lab_, torch.full_like(lab_, -1))


inner = (P[:, 0].abs() < 0.15 * halfw)
root = int((P[:, 1] + (~inner).float() * 1e3).argmin())
gd, it = geodesic(root)
reach = torch.isfinite(gd)
T["geodesic"] = round(time.time() - t0, 2)
T["geodesic_iters"] = it

# ---- bands, components, nodes
h = a.band * L
band, lab = band_parts(gd, h)
ul, comp = torch.unique(lab, return_inverse=True)
nn_ = len(ul)
cnt = torch.zeros(nn_, device=dev).index_add_(0, comp, torch.ones(n, device=dev))
cen = torch.zeros(nn_, 3, device=dev).index_add_(0, comp, P) / cnt[:, None]
nband = torch.zeros(nn_, dtype=torch.long, device=dev).scatter_reduce(0, comp, band, "amax")
# ring size: mean distance of a node's samples from its centre (a ring round a limb is small, round the torso large)
rad = (torch.zeros(nn_, device=dev).index_add_(0, comp, (P - cen[comp]).norm(dim=1)) / cnt).cpu().numpy()
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


def limb_rings(leaf, reach_frac=0.6):
    """Cross-sections of a limb walked from its tip: geodesic bands from the tip sample (the leaf's sample farthest
    from the snout); each ring is the band's pieces touching the previous ring. Returns per ring: sample mask
    (bool, on the GPU), centre, size (mean distance from the centre), geodesic distance from the tip."""
    m = comp == leaf
    tip = int(torch.where(m, gd, torch.full_like(gd, -1.0)).argmax())
    g_, _ = geodesic(tip)
    b_, l_ = band_parts(g_, h)
    cur = l_ == l_[tip]
    rings = []
    for kb in range(int(reach_frac * L / h) + 1):
        if not bool(cur.any()):
            break
        pc = P[cur]
        c_ = pc.mean(0)
        rings.append((cur, c_.cpu().numpy(), float((pc - c_).norm(dim=1).mean()), kb * h))
        e_ok = cur[:, None] & good & (b_[nbr] == kb + 1)
        nxt = torch.unique(l_[nbr][e_ok])
        cur = torch.isin(l_, nxt) & (b_ == kb + 1)
    return rings, g_, tip


def connected_from(seed, keep):
    """samples joined to sample `seed` through kNN edges whose both ends are in `keep`"""
    lab_ = torch.where(keep, torch.arange(n, device=dev), torch.full((n,), n, device=dev))
    e_ok = keep[:, None] & keep[nbr] & good
    src_ = torch.arange(n, device=dev)[:, None].expand(-1, k)[e_ok]
    dst_ = nbr[e_ok]
    for _ in range(10000):
        m1 = lab_.scatter_reduce(0, src_, lab_[dst_], "amin")
        m1 = m1.scatter_reduce(0, dst_, lab_[src_], "amin")
        if torch.equal(m1, lab_):
            break
        lab_ = m1
    return keep & (lab_ == lab_[seed])


limb_dbg = {}


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
# neck base: from the shoulder junction toward the head the rings first grow (the limbs' roots are still inside the
# ring there) up to the chest's largest ring, then narrow into the neck; the neck starts at the first ring down to
# --neck-ring x that chest ring, never inside the head (the first ~12 % of the length from the snout)
xyz_m = pts(main)
arc_m = arclen(xyz_m)
rad_m = np.convolve(np.pad(rad[main], 1, mode="edge"), np.ones(3) / 3, mode="valid")
i_head = int(np.searchsorted(arc_m, 0.12 * L + h))
nbx = js
while nbx - 1 > i_head and rad_m[nbx - 1] >= rad_m[nbx]:
    nbx -= 1
chest = rad_m[nbx]
while nbx - 1 > i_head and rad_m[nbx] > a.neck_ring * chest:
    nbx -= 1
nbx = max(min(nbx, js), min(i_head, js))
# torso ring size: the trunk's largest cross-section (neck base to hip)
R_t = float(rad_m[nbx:jh + 1].max())
spine_nodes = main[nbx:jh + 1][::-1]                                   # hip -> chest -> neck base
nb = a.spine_bones
cs = C[main[(js + jh) // 2]]
bones.append({"name": "root", "parent": None, "head": [float(cs[0]), float(cs[1]), 0.0],
              "tail": [float(cs[0]), float(cs[1]), float(cs[2]) * 0.5]})
add_chain(spine_nodes, [f"spine_{i}" for i in range(nb)], np.linspace(0, 1, nb + 1), "root")
spine_top = f"spine_{nb - 1}"


def nearest_bone(pt, names):
    """the bone among `names` whose segment passes nearest to pt"""
    best_, bn_ = 1e9, names[0]
    for b_ in bones:
        if b_["name"] in names:
            h_, t_ = np.array(b_["head"]), np.array(b_["tail"])
            sg = t_ - h_
            u = np.clip(np.dot(pt - h_, sg) / max(np.dot(sg, sg), 1e-12), 0, 1)
            dd = np.linalg.norm(pt - (h_ + u * sg))
            if dd < best_:
                best_, bn_ = dd, b_["name"]
    return bn_


spine_names = [f"spine_{i}" for i in range(nb)]
spine_sh = nearest_bone(C[main[js]], spine_names)                    # the bone the wings and front legs hang from
# neck + head: the head is the last ~12 % of the creature's length along the path to the snout
neck_head = main[:nbx + 1][::-1]
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
# legs: a leg hanging under the torso is cut by the snout's bands lengthwise, so the graph splits it off only below
# the belly. Its own cross-sections (bands from its foot) find where it really joins the body: the first ring
# larger than --limb-root x the torso ring; the hip / shoulder joint is the ring before it, inside the thigh or
# upper arm. The leg's samples are the surface joined to the foot below that ring; each gets its place along the
# leg from its distance to the foot. A leg whose rings never reach that size keeps the graph path.
leg_w = []                                                            # (sample mask, arc position, segments)
leg_fr = [0, 0.4, 0.75, 0.9, 1.0]
for name, lf in feet.items():
    lnames = [f"{name}_upper", f"{name}_lower", f"{name}_foot", f"{name}_toes"]
    rg, g_, tip = limb_rings(lf)
    sizes = [r_[2] for r_ in rg]
    kr = next((kk for kk in range(3, len(rg)) if sizes[kk] > a.limb_root * R_t), None)
    limb_dbg[name] = {"rings": [[round(r_[2], 3)] + np.round(r_[1], 2).tolist() for r_ in rg], "root_ring": kr}
    if kr is None:
        p = Lp[lf]
        j = diverge(main, p)
        add_chain(p[j:], lnames, leg_fr, spine_sh if "front" in name else "spine_0")
        continue
    xyz = np.vstack([np.array([rg[kk][1] for kk in range(kr)])[::-1], P[tip].cpu().numpy()[None]])
    d = arclen(xyz)
    cuts = [f * d[-1] for f in leg_fr]
    pn = nearest_bone(xyz[0], spine_names)
    segs = []
    for b, (s0, s1) in enumerate(zip(cuts[:-1], cuts[1:])):
        bones.append({"name": lnames[b], "parent": pn, "head": at(xyz, s0).tolist(), "tail": at(xyz, s1).tolist()})
        segs.append((lnames[b], s0, s1))
        pn = lnames[b]
    member = connected_from(tip, g_ < kr * h)
    gm = g_[member].cpu().numpy()
    # ring kk (distances [kk h, (kk + 1) h)) sits at xyz[kr - 1 - kk]; the foot tip (distance 0) at the end
    gk = np.r_[0.0, (np.arange(kr) + 0.5) * h]
    sk = np.r_[d[-1], d[kr - 1 - np.arange(kr)]]
    leg_w.append((member, np.interp(gm, gk, sk), segs))
# wings: arm / forearm to the wrist, fingers from there to the other tips
for side, cands in wings.items():
    wp = Lp[cands[0]]
    j = diverge(main, wp)
    wn = f"wing_{side}"
    add_chain(wp[j:], [f"{wn}_arm", f"{wn}_forearm", f"{wn}_finger_0"], [0, 0.33, 0.62, 1.0], spine_sh)
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
# legs found from their own rings: their samples take the leg bones by place along the leg, blended at the joints
for member, sl, segs in leg_w:
    Wl = np.zeros((len(sl), len(segs)), np.float32)
    for b, (nm, s0, s1) in enumerate(segs):
        win = a.blend * (s1 - s0)
        lo_w = np.ones_like(sl) if b == 0 else np.clip((sl - (s0 - win / 2)) / max(win, 1e-9), 0, 1)
        hi_w = np.ones_like(sl) if b == len(segs) - 1 else np.clip(((s1 + win / 2) - sl) / max(win, 1e-9), 0, 1)
        Wl[:, b] = np.minimum(lo_w, hi_w)
    Wl /= np.maximum(Wl.sum(1, keepdims=True), 1e-9)
    mi = torch.where(member)[0]
    Ws[mi] = 0
    for b, (nm, _, _) in enumerate(segs):
        Ws[mi, bidx[nm]] = torch.tensor(Wl[:, b], device=dev)
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
# back to the mesh's own frame (everything above ran with the mirror plane at x = 0)
for b in bones:
    b["head"][0] += x0
    b["tail"][0] += x0
V[:, 0] += x0
C[:, 0] += x0
out = {"bones": bones, "labels": {"head": int(head), "tail": int(tail), "feet": {k_: int(v) for k_, v in feet.items()},
                                  "wings": {k_: [int(x) for x in v[:4]] for k_, v in wings.items()}},
       "nodes": int(nn_), "leaves": len(leaves), "samples": n, "reached": float(reach.float().mean()),
       "verts": nv, "verts_xyz": np.round(V.cpu().numpy(), 4).tolist(), "weights_idx": top.indices.cpu().numpy().astype(int).tolist(),
       "weights_w": np.round(tw.cpu().numpy(), 4).tolist(),
       "node_xyz": C[inroot].round(3).tolist(), "times_s": T,
       "leaf_info": [{"id": int(i), "xyz": C[i].round(3).tolist(), "geo": round(float(glen[i]), 3),
                      "split": int(diverge(main, Lp[i])), "n": int(cnt_np[i])} for i in leaves],
       "main_xyz": C[main].round(3).tolist(), "main_rad": [round(float(rad[i]), 3) for i in main],
       "junctions": {"neck_base": int(nbx), "shoulder": int(js), "hip": int(jh), "torso_ring": round(R_t, 3)},
       "limb_rings": limb_dbg}
json.dump(out, open(a.out, "w"))
T["total"] = round(time.time() - t0, 2)
print(json.dumps({"bones": [b["name"] for b in bones], "feet": list(feet), "wings": {k_: len(v) for k_, v in wings.items()},
                  "nodes": nn_, "leaves": len(leaves), "reached": out["reached"], "times_s": T}))
