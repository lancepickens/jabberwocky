# Pipeline overview

```
video.mp4
   │
   ▼
[1] frame extraction        video.py      sharp, evenly spaced frames
   │
   ▼
[2] structure from motion   features.py   SIFT + matching + tracks
   │                        sfm.py        init pair → PnP → triangulate
   │                        ba.py         sparse bundle adjustment
   │
   ├──► camera poses (R, t per frame) + shared focal length
   └──► sparse colored point cloud
   │
   ▼
[3] gaussian splat training model.py      gaussians seeded on the cloud
   │                        render.py     differentiable rasterizer
   │                        train.py      L1 + SSIM descent, densify/prune
   │
   ▼
[4] export + viewing        export.py     scene.ply, scene.splat
                            viewer.html   WebGL2 orbit viewer
```

## Why this decomposition

The end goal is a set of 3D gaussians that, when rendered from any
viewpoint, reproduce the scene. Training such a model needs two things the
video does not directly provide:

1. **Where each frame was taken from.** The photometric loss compares a
   rendering *from the camera's pose* against the real frame; without poses
   there is nothing to render. Poses cannot be assumed (a phone video has
   no useful pose metadata), so they must be recovered from the pixels —
   that is structure from motion (SfM).
2. **A starting point for the geometry.** Gradient descent on gaussians is
   a local method: a gaussian can only drift toward a better position if it
   already overlaps the right part of the image. Random initialization in a
   general scene fails; the sparse SfM point cloud puts a gaussian near
   every piece of well-observed surface from the start.

SfM conveniently produces both at once, because camera poses and 3D point
positions are two halves of the same geometric constraint system.

## Coordinate conventions

Consistent conventions are half the battle in multi-view geometry. The
whole codebase uses one set (`geometry.py` docstring is the authoritative
statement):

- **World → camera:** `x_cam = R · x_world + t` (OpenCV convention). The
  camera center in world coordinates is `C = −Rᵀ t`.
- **Camera frame:** the camera looks down **+z**; a point is visible only
  if its camera-space z is positive. y points down the image, x to the
  right.
- **Intrinsics:** a single pinhole matrix `K = [[f, 0, cx], [0, f, cy],
  [0, 0, 1]]` shared by all frames — one focal length `f` in pixels, square
  pixels, principal point fixed at the image center, no lens distortion.
  `f` is refined by bundle adjustment; the rest is assumed. This is a
  deliberate simplification that holds well for typical phone/camera video
  (see limitations in the main README).
- **Quaternions** are `(w, x, y, z)`, normalized.
- The **web viewer** is the one exception: WebGL cameras conventionally
  look down −z with y up, so `viewer.html` flips the z row when it reuses
  the projection math (documented inline there).

There is one global **gauge freedom** worth knowing about: a monocular
reconstruction has no absolute scale, position, or orientation. splatvid
pins position/orientation by fixing the first registered camera at the
identity pose, and leaves scale free — the reconstruction is correct up to
an unknown global scale factor, which is irrelevant for viewing.

## Data flow between stages

- Stage 1 → 2: a list of BGR frames, all the same size
  (`video.FrameSet`).
- Stage 2 → 3: a `sfm.Reconstruction`: focal/cx/cy, per-frame `(R, t)`,
  an `(N, 3)` point array with per-point RGB colors and reprojection
  errors, and the list of registered frame indices. Also saved to
  `cameras.npz` for reuse.
- Stage 3 → 4: a `model.GaussianModel` holding the optimized parameter
  tensors.
- Stage 4: `scene.ply` (interchange), `scene.splat` + `index.html`
  (bundled viewer), optional `turntable.mp4` (rendered orbit).

## Module structure

The package is deliberately layered: a set of dependency-free **foundation**
modules (each a self-contained algorithm with no intra-package imports), a
thin layer of **stage orchestrators** that compose them, and `cli.py` on top
wiring the four stages together. Nothing imports "sideways" within a layer,
so any foundation module can be read, tested, or reused in isolation.

| Module | Layer | Responsibility | Imports (intra-package) |
|---|---|---|---|
| `geometry.py` | foundation | NumPy rotation / projection / DLT-triangulation primitives; the authoritative coordinate conventions | — |
| `features.py` | foundation | SIFT detection, ratio+cross-check matching, epipolar verification, union-find tracks | — |
| `video.py` | foundation | frame extraction + sharpness selection | — |
| `render.py` | foundation | pure-PyTorch differentiable EWA rasterizer | — |
| `model.py` | foundation | `GaussianModel`: parameters, activations, densify/prune | — |
| `losses.py` | foundation | L1 + differentiable SSIM + PSNR | — |
| `ba.py` | stage | sparse bundle adjustment (SciPy least-squares) | `geometry` |
| `sfm.py` | stage | incremental SfM driver (init → PnP → triangulate → BA) | `features`, `geometry`, `ba` |
| `train.py` | stage | optimization loop, learning-rate schedule, turntable render | `model`, `render`, `losses`, `sfm` |
| `export.py` | stage | `.ply` / `.splat` writers and `.splat` reader | `model` |
| `synthetic.py` | test/demo | procedural scene + orbit-video generator (known ground truth) | `render` |
| `cli.py` | top | argument parsing, device selection, 4-stage orchestration | `video`, `sfm`, `train`, `export` |
| `viewer.html` | top | standalone WebGL2 splat viewer | — (no Python) |

Dependency graph (an arrow means "imports"):

```
cli.py
├─ video.py            (foundation, leaf)
├─ sfm.py ─── features.py            (foundation, leaf)
│         └── ba.py ──┐
│         └── geometry.py ◄┘         (foundation, leaf)
├─ train.py ── sfm.py (above)
│           ├─ model.py              (foundation, leaf)
│           ├─ render.py             (foundation, leaf)
│           └─ losses.py             (foundation, leaf)
└─ export.py ── model.py

synthetic.py ── render.py            (test/demo only)
viewer.html                          (standalone, no imports)
```

Two facts this structure encodes:

- **`geometry.py` is the shared foundation of the SfM half**; the splatting
  half rests instead on `render.py` + `model.py` + `losses.py`. The two
  halves meet only inside `train.py`, which reads an `sfm.Reconstruction`
  and seeds a `model.GaussianModel` from it.
- **Quaternion→matrix math exists twice on purpose.** `geometry.py` has a
  NumPy version used by the CPU-side SfM/BA; `render.py` has an
  autograd-friendly `quat_to_rotmat_torch` used by the differentiable
  renderer (and mirrored again in GLSL inside `viewer.html`). They implement
  the same `(w, x, y, z)` convention; the three tests in `test_geometry.py`
  and `test_render.py` pin them to it.

## Where the time goes

On CPU, roughly: SIFT matching and the rasterizer dominate. Matching is
`O(pairs × features²)` in brute-force descriptor distance; the pipeline
bounds `pairs` by matching only a sliding temporal window (default 6) plus
sparse long-range loop-closure pairs. The rasterizer costs per iteration
roughly (gaussian-tile overlaps) × (pixels per tile); the trainer bounds it
with a small default training resolution (`--train-size 320`), a gaussian
budget (`--max-gaussians`), and a per-tile cap. On a CUDA GPU the same code
runs unmodified and the budgets can be raised substantially.
