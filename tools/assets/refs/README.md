# Reference pictures for asset modelling

`ref_draw.py` draws a reference set for an asset (front, side, top, three-quarter, a 2x2 `sheet.png`, an optional
`material.png` close-up, and `notes.md`) with the pipeline's Qwen-Image drawer (`tools/qwen_image.py --serve`,
unchanged) on the integrated GPU, holding the iGPU lock file while the drawer lives. See the module docstring for
the command line and the environment variables.

## Recipe comparison (one starfighter description, seed 11, 2026-11-10)

| recipe | same design in all views? | true front / side / top? | cost on the iGPU |
|---|---|---|---|
| separate (4 prompts, same seed) | no: fin count, nacelle placement and colour blocks differ per view | yes | 4 x ~34 s |
| hero (3/4 view, then 3 views with it as a reference input) | yes: colours, fins, nacelles, rails carried over | no: "front" came out as an overhead rear view, "side" slightly angled | ~29 s + 3 x ~38 s |
| sheet (one 2x2 turnaround picture, cut) | yes: one picture holds one design | yes for side and top; front is a slightly raised front view | ~86 s for all four |

The sheet wins: it is the only recipe whose views are both the same object and real elevations, and it is the
cheapest. Its views are ~700 px each (the 1536 px sheet cut in four) and the cuts follow the emptiest row/columns,
since a view often crosses the midline. The edit model (Qwen-Image-Edit-2509) was not tried: the sheet already gives
one design per picture.

Known limits: the views are drawn, not projected, so proportions agree only roughly (check lengths against the
spec's `size_m`, not the pictures); light glow or small details can differ between views; a sheet whose layout does
not come out as 2x2 needs another seed (`--seed`).
