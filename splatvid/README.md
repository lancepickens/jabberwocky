# splatvid — video → 3D gaussian splat, from scratch

`splatvid` turns a video of a scene (walk around an object, orbit a room
corner, …) into a 3D gaussian-splat model you can orbit in the browser.
Every stage is implemented from scratch in this repo — no COLMAP, no
prebuilt splatting kernels:

1. **Frame extraction** (`splatvid/video.py`) — samples frames evenly
   across the video and keeps the sharpest frame per window
   (Laplacian-variance scoring) to skip motion blur.
2. **Structure from motion** (`splatvid/features.py`, `sfm.py`, `ba.py`) —
   SIFT features, ratio-test + cross-check matching, fundamental-matrix
   RANSAC verification, track building via union-find, essential-matrix
   initialization of the best-parallax pair, incremental PnP registration,
   DLT triangulation, and sparse bundle adjustment (SciPy least-squares
   with an explicit Jacobian sparsity pattern) that also refines the
   focal length. Output: camera poses + a colored sparse point cloud.
3. **Gaussian splat optimization** (`splatvid/model.py`, `render.py`,
   `train.py`) — a differentiable EWA splatting rasterizer written in
   pure PyTorch (perspective Jacobian projection of 3D covariances,
   depth-sorted per-tile front-to-back alpha compositing), trained with
   L1 + SSIM loss, with gradient-driven densification (clone/split) and
   opacity pruning, seeded from the SfM point cloud.
4. **Export + viewer** (`splatvid/export.py`, `viewer.html`) — writes the
   de-facto standard 3DGS `.ply` (loads in SuperSplat & friends) and the
   compact 32-byte `.splat` format, plus a self-contained WebGL2 viewer
   (instanced quads, counting-sort back-to-front blending) with orbit /
   pan / zoom controls.

## Install

```bash
cd splatvid
pip install -e .          # or: pip install -e '.[dev]' for tests
```

Requires Python ≥ 3.10. CPU-only PyTorch is fine; the best available
device is picked automatically (CUDA → Apple-Silicon MPS → CPU) and makes
training much faster. On M-series Macs, see
[docs/performance-and-roadmap.md](docs/performance-and-roadmap.md) for
benchmarking (`scripts/bench_render.py`) and recommended settings.

## Use

```bash
# Reconstruct a scene from a video
splatvid reconstruct my_video.mp4 -o out/

# Open the interactive viewer (serves out/ on localhost:8000)
splatvid view out/
```

Output directory contents:

| file | what |
|---|---|
| `scene.ply` | standard 3D gaussian splatting PLY (works in external viewers) |
| `scene.splat` | compact binary splat file used by the bundled viewer |
| `index.html` | self-contained WebGL2 viewer (drag = orbit, shift-drag = pan, wheel = zoom) |
| `cameras.npz` | recovered camera poses, intrinsics, and the sparse SfM cloud |

Useful knobs (see `splatvid reconstruct --help`):

- `--iterations 6000 --train-size 640 --max-gaussians 200000` for quality
  (recommended on a GPU; the CPU defaults are deliberately modest).
- `--max-frames` / `--features` trade SfM robustness against runtime.
- `--turntable` also renders an orbit video of the reconstruction.

## What kind of video works

- **Move the camera**, don't just rotate it: parallax is what makes 3D
  recovery possible. Orbit the subject in a slow arc.
- Well-lit, textured, **static** scenes. Avoid mirrors, glass, and large
  featureless walls.
- Steady motion; motion blur hurts feature matching (the sharpness-aware
  frame picker helps, but only so much).

## Demo without a camera

```bash
python scripts/make_demo.py --out demo_out
splatvid view demo_out
```

This renders a synthetic orbit video of a procedurally generated scene
with the package's own rasterizer, then reconstructs it from pixels alone
— the same path a real video takes.

## Documentation

Extensive documentation lives in [`docs/`](docs/README.md):

- [`docs/explainer.html`](docs/explainer.html) — a self-contained, illustrated
  explainer of the whole pipeline (open directly in a browser; includes an
  interactive alpha-compositing demo).
- [`docs/pipeline-overview.md`](docs/pipeline-overview.md) — architecture,
  data flow, coordinate conventions, and the module dependency graph.
- [`docs/structure-from-motion.md`](docs/structure-from-motion.md) — features,
  epipolar geometry, PnP, triangulation, and bundle adjustment in depth.
- [`docs/gaussian-splatting.md`](docs/gaussian-splatting.md) — the gaussian
  representation, EWA projection, the differentiable tile rasterizer,
  training, and densification.
- [`docs/formats-and-viewer.md`](docs/formats-and-viewer.md) — byte-level
  `.ply`/`.splat` layouts and how the WebGL viewer works.
- [`docs/performance-and-roadmap.md`](docs/performance-and-roadmap.md) —
  Apple Silicon / device selection, benchmarking, and planned improvements.
- [`docs/testing.md`](docs/testing.md) — how the test suite validates each
  layer of the architecture, and how to run it.

## Tests

```bash
python -m pytest              # unit tests + slow end-to-end pipeline test
python -m pytest -m 'not slow'  # fast tests only (~seconds)
```

The end-to-end test renders a synthetic orbit video, recovers the camera
ring with the from-scratch SfM (checking the cameras form a planar ring),
trains a splat model, verifies PSNR against the input frames, and
round-trips both export formats.

## Layout

```
splatvid/
  video.py      frame extraction & sharpness selection
  features.py   SIFT, matching, geometric verification, tracks
  sfm.py        incremental SfM (init pair, PnP, triangulation)
  ba.py         sparse bundle adjustment
  geometry.py   rotations, projection, triangulation primitives
  model.py      gaussian parameters, densify/prune
  render.py     differentiable tile rasterizer (pure PyTorch)
  losses.py     L1 + SSIM
  train.py      optimization loop, turntable rendering
  export.py     .ply / .splat writers
  viewer.html   self-contained WebGL2 splat viewer
  cli.py        `splatvid reconstruct` / `splatvid view`
  synthetic.py  procedural scene + video generator (tests/demo)
```

## Limitations

- SfM assumes a single shared pinhole camera with no lens distortion;
  heavily fisheye footage won't converge well.
- The pure-PyTorch rasterizer is exact but not fast: CPU training uses
  small render resolutions by default. On CUDA you can raise
  `--train-size` and `--iterations` substantially.
- Only degree-0 spherical harmonics (flat color per gaussian) are
  modeled, so strongly view-dependent effects (specularities) are baked
  into averages.
