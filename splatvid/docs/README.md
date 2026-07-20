# splatvid documentation

`splatvid` reconstructs a 3D gaussian-splat scene from an ordinary video.
These documents explain how — both the theory and the specific choices made
in this codebase. Everything described here is implemented from scratch in
this repository (no COLMAP, no prebuilt splatting kernels), so every formula
below maps to code you can read.

## Reading order

| Doc | Covers | Code |
|---|---|---|
| [pipeline-overview.md](pipeline-overview.md) | The big picture: why each stage exists, data flow, coordinate conventions, module structure & dependency graph | `cli.py` |
| [structure-from-motion.md](structure-from-motion.md) | Frames → camera poses + sparse point cloud: features, epipolar geometry, PnP, triangulation, bundle adjustment | `video.py`, `features.py`, `sfm.py`, `ba.py`, `geometry.py` |
| [gaussian-splatting.md](gaussian-splatting.md) | Point cloud → photorealistic model: the 3D gaussian representation, EWA projection, the differentiable tile rasterizer, training, densification | `model.py`, `render.py`, `losses.py`, `train.py` |
| [artifix.md](artifix.md) | Repairing incomplete/artifacted splats: the ArtiFixer-derived pipeline (opacity-gated confidence, floater vote + anchor-verified pruning, plane-sweep hole filling, hole seeding, pseudo-supervised fine-tune) | `artifix.py` |
| [formats-and-viewer.md](formats-and-viewer.md) | Byte-level `.ply` / `.splat` layouts and how the bundled WebGL2 viewer works | `export.py`, `viewer.html` |
| [performance-and-roadmap.md](performance-and-roadmap.md) | Apple Silicon (MPS) usage, benchmarking, where the time goes, and the improvement roadmap | `cli.py`, `render.py`, `scripts/bench_render.py` |
| [testing.md](testing.md) | How the test suite validates each layer of the architecture, and how to run it | `tests/`, `synthetic.py` |

**[explainer.html](explainer.html)** is a self-contained illustrated
explainer of the whole pipeline — open it directly in a browser (no server,
no network needed). It covers the same ground as these documents at a more
visual, less exhaustive level, including an interactive alpha-compositing
demo.

## One-paragraph summary

A video is a stack of 2D projections of one 3D scene. Because the camera
moves, the same scene point lands at different pixels in different frames —
that shift (parallax) encodes depth. Stage 1 picks sharp frames; stage 2
(structure from motion) finds pixel-accurate correspondences between frames
and solves jointly for where the cameras were and where the matched points
sit in 3D, producing a sparse, colored point cloud plus calibrated cameras.
Stage 3 upgrades that sparse skeleton into a dense, photorealistic model: it
places a 3D gaussian on every point and optimizes position, shape, color,
and opacity of all gaussians by rendering them through a differentiable
rasterizer and descending the gradient of the photometric error against the
real frames. Stage 4 writes the result in standard formats and ships a
WebGL viewer to orbit it.
