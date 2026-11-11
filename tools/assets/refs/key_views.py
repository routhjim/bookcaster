"""Key the plain background out of reference drawings (on a GPU, in torch): an RGBA picture per input, for the
image-to-3D model, whose own corner-colour key leaves textured paper backgrounds in and builds them as slabs.

    python key_views.py OUT_DIR IN.png[:TOL[:shadow]] ...   (HIP_VISIBLE_DEVICES picks the card; never the CPU)

TOL (default 0.075) is the colour distance per channel still counted as background; lower it for a pale object
on a pale ground (an opal globe). "shadow" also counts greyish pixels up to ~45% darker than the ground as
background (a soft contact shadow, which the shape model would build as a slab on the floor).

Background = pixels close to the border's colour (after a small blur, so paper grain does not break it) that are
connected to the picture's edge; found by a masked dilation from the border, run on the GPU until it stops
growing. Holes inside the object (between a lantern's straps) stay object unless they reach the edge, which is
what the shape model wants (it cannot see through glass either). The edge is feathered by one blur step.
"""

import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


def key(path, dev, tol=0.075, shadow=False):
    a = torch.from_numpy(np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0).to(dev)
    img = a.permute(2, 0, 1)[None]
    blur = F.avg_pool2d(F.pad(img, (3, 3, 3, 3), mode="replicate"), 7, stride=1)
    h, w = a.shape[:2]
    border = torch.cat([blur[0, :, :4, :].reshape(3, -1), blur[0, :, -4:, :].reshape(3, -1),
                        blur[0, :, :, :4].reshape(3, -1), blur[0, :, :, -4:].reshape(3, -1)], 1)
    bg = border.median(1).values.view(1, 3, 1, 1)
    # the background may shade across the picture: allow the tolerance plus the border's own spread
    spread = (border - bg.view(3, 1)).abs().sum(0).quantile(0.95).item()
    near = ((blur - bg).abs().sum(1, keepdim=True) < tol * 3 + spread).float()
    if shadow:                                         # grey and a little darker than the ground: its shadow
        chroma = blur.max(1, keepdim=True).values - blur.min(1, keepdim=True).values
        ratio = blur.mean(1, keepdim=True) / bg.mean()
        near = torch.maximum(near, ((chroma < 0.09) & (ratio > 0.55) & (ratio <= 1.02)).float())
    seed = torch.zeros_like(near)
    seed[..., 0, :] = seed[..., -1, :] = 1
    seed[..., :, 0] = seed[..., :, -1] = 1
    seed = seed * near
    for _ in range(4 * max(h, w)):
        grown = F.max_pool2d(seed, 3, stride=1, padding=1) * near
        if torch.equal(grown, seed):
            break
        seed = grown
    alpha = 1.0 - F.avg_pool2d(F.pad(seed, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    rgba = torch.cat([img, alpha], 1)[0].permute(1, 2, 0).clamp(0, 1).mul(255).byte().cpu().numpy()
    return Image.fromarray(rgba, "RGBA"), float(alpha.mean())


def main():
    if not torch.cuda.is_available():
        raise SystemExit("no GPU visible: never key on the CPU")
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda")
    for arg in sys.argv[2:]:
        p, *opt = arg.split(":")
        im, cover = key(p, dev, float(opt[0]) if opt and opt[0] else 0.075, len(opt) > 1 and opt[1] == "shadow")
        im.save(out / (Path(p).stem + ".png"))
        print(f"{Path(p).name}: object {cover:.0%} of the picture", flush=True)


if __name__ == "__main__":
    main()
