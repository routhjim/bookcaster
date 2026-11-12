"""A small ONNX graph runner on torch, so an ONNX-only model runs on a card (onnxruntime here has only the CPU).

    python tools/assets/redraw/onnx_torch.py export MODEL.onnx OUT.pt     (needs the `onnx` package; parsing only)
    from onnx_torch import Graph; g = Graph("OUT.pt", "cuda"); outs = g({"input": x})

`export` writes the graph (nodes, attributes) and its weights as a torch file once; `Graph` needs only torch. The
operator set is what the whole-body pose model (RTMW, opset 11) uses; an unknown operator raises.
"""
from __future__ import annotations

import sys

import torch
import torch.nn.functional as F


def export(onnx_path: str, out: str) -> None:
    import onnx
    from onnx import helper, numpy_helper
    m = onnx.load(onnx_path)
    w = {i.name: torch.from_numpy(numpy_helper.to_array(i).copy()) for i in m.graph.initializer}
    nodes = []
    for n in m.graph.node:
        at = {}
        for a in n.attribute:
            v = helper.get_attribute_value(a)
            if a.name == "value":
                v = torch.from_numpy(numpy_helper.to_array(v).copy())
            elif isinstance(v, bytes):
                v = v.decode()
            elif isinstance(v, list):
                v = list(v)
            at[a.name] = v
        nodes.append({"op": n.op_type, "in": list(n.input), "out": list(n.output), "at": at})
    torch.save({"weights": w, "nodes": nodes, "inputs": [i.name for i in m.graph.input],
                "outputs": [o.name for o in m.graph.output]}, out)


class Graph:
    def __init__(self, path: str, device: str = "cuda", dtype=torch.float32):
        d = torch.load(path, map_location="cpu", weights_only=False)
        self.dev = torch.device(device)
        self.w = {k: (v.to(self.dev, dtype) if v.is_floating_point() else v) for k, v in d["weights"].items()}
        self.nodes, self.outputs = d["nodes"], d["outputs"]

    @torch.no_grad()
    def __call__(self, feeds: dict) -> list:
        env = dict(self.w)
        env.update(feeds)
        for n in self.nodes:
            ins = [env[i] if i else None for i in n["in"]]
            res = self.op(n["op"], ins, n["at"])
            if not isinstance(res, (list, tuple)):
                res = [res]
            for k, v in zip(n["out"], res):
                env[k] = v
        return [env[o] for o in self.outputs]

    def op(self, op, x, a):
        if op == "Conv":
            p = a.get("pads", [0, 0, 0, 0])
            if p[0] != p[2] or p[1] != p[3]:
                raise NotImplementedError("asymmetric Conv pads")
            return F.conv2d(x[0], x[1], x[2] if len(x) > 2 else None, a.get("strides", [1, 1]), (p[0], p[1]),
                            a.get("dilations", [1, 1]), a.get("group", 1))
        if op == "Sigmoid":
            return torch.sigmoid(x[0])
        if op == "Relu":
            return torch.relu(x[0])
        if op == "HardSigmoid":
            return (x[0] * a.get("alpha", 0.2) + a.get("beta", 0.5)).clamp(0, 1)
        if op == "Mul":
            return x[0] * x[1]
        if op == "Add":
            return x[0] + x[1]
        if op == "Sub":
            return x[0] - x[1]
        if op == "Div":
            return x[0] / x[1]
        if op == "Sqrt":
            return torch.sqrt(x[0])
        if op == "MatMul":
            return torch.matmul(x[0], x[1])
        if op == "Concat":
            if not x[0].is_floating_point():
                return torch.cat([t.cpu() for t in x], a["axis"])
            return torch.cat(x, a["axis"])
        if op == "GlobalAveragePool":
            return x[0].mean((2, 3), keepdim=True)
        if op == "MaxPool":
            p = a.get("pads", [0, 0, 0, 0])
            return F.max_pool2d(x[0], a["kernel_shape"], a.get("strides", [1, 1]), (p[0], p[1]),
                                ceil_mode=bool(a.get("ceil_mode", 0)))
        if op == "Constant":
            v = a["value"]
            return v.to(self.dev, torch.float32) if v.is_floating_point() else v
        if op == "Resize":
            if a.get("mode") != "nearest":
                raise NotImplementedError("Resize mode " + str(a.get("mode")))
            scales = x[2] if len(x) > 2 and x[2] is not None and x[2].numel() else None
            if scales is not None:
                sc = [float(v) for v in scales.flatten()[2:]]
                return F.interpolate(x[0], scale_factor=sc, mode="nearest")
            sz = [int(v) for v in x[3].flatten()[2:]]
            return F.interpolate(x[0], size=sz, mode="nearest")
        if op == "Shape":
            return torch.tensor(list(x[0].shape), dtype=torch.long)
        if op == "Slice":
            t = x[0]
            st, en = x[1].tolist(), x[2].tolist()
            ax = x[3].tolist() if len(x) > 3 and x[3] is not None else list(range(len(st)))
            stp = x[4].tolist() if len(x) > 4 and x[4] is not None else [1] * len(st)
            idx = [slice(None)] * t.dim()
            for s_, e_, a_, p_ in zip(st, en, ax, stp):
                idx[a_] = slice(s_, min(e_, t.shape[a_]) if e_ < 2 ** 62 else None, p_)
            return t[tuple(idx)]
        if op == "Reshape":
            sh = [int(v) for v in x[1].tolist()]
            sh = [x[0].shape[i] if v == 0 else v for i, v in enumerate(sh)]
            return x[0].reshape(sh)
        if op == "ReduceSum":
            return x[0].sum(a["axes"], keepdim=bool(a.get("keepdims", 1)))
        if op == "Clip":
            lo = x[1] if len(x) > 1 and x[1] is not None else None
            hi = x[2] if len(x) > 2 and x[2] is not None else None
            return x[0].clamp(min=None if lo is None else float(lo), max=None if hi is None else float(hi))
        if op == "Split":
            return list(torch.split(x[0], a["split"], a.get("axis", 0)))
        if op == "Squeeze":
            t = x[0]
            for ax in sorted(a["axes"], reverse=True):
                t = t.squeeze(ax)
            return t
        if op == "Unsqueeze":
            t = x[0]
            for ax in sorted(a["axes"]):
                t = t.unsqueeze(ax)
            return t
        if op == "Transpose":
            return x[0].permute(a["perm"])
        if op == "DepthToSpace":
            if a.get("mode", "DCR") != "CRD":
                raise NotImplementedError("DepthToSpace DCR")
            return F.pixel_shuffle(x[0], a["blocksize"])
        raise NotImplementedError(op)


if __name__ == "__main__":
    if sys.argv[1] == "export":
        export(sys.argv[2], sys.argv[3])
        print("wrote", sys.argv[3])
