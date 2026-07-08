# Movie → Mesh Quality Overhaul — Report

_Driven by a SOTA research study of photogrammetry + gaussian-splatting papers,
targeting a dramatic jump in reconstruction quality (goal: **best dense mesh**)
from a single rich handheld iPhone video (`IMG_6547.mov`)._

## TL;DR

| Area | Before | After | How |
|---|---|---|---|
| **SfM matching** | SIFT + brute-force ratio | **DISK + LightGlue** (learned, MPS-native) + retrieval loop closure | rewrite of `features.py` |
| **Cameras registered** | fragmented, frames stranded | **40/40** (also 24/24 on a subset) | new correspondences |
| **Reprojection error** | — | **0.61 px** mean (sub-pixel) | denser, cleaner tracks |
| **Features / frame** | SIFT (sparse, less repeatable) | **~3,600** (DISK) | learned detector |
| **Mesh depth** | opacity-weighted **mean** (`Σwz/α`) — phantom mid-surface | **median** (transmittance 0.5) — true front surface | `render.py` + `mesh.py` |
| **Mesh surface roughness** | 0.27 | **0.13** (≈ halved) | median depth |

Everything above is implemented, verified, and **merged to `master`**.

## The diagnosis

The input video is rich (lots of texture and parallax), but the from-scratch
pipeline was leaving most of that on the table. Two research tracks (geometry and
splatting), each grounded in the actual code, converged on the same root causes:

1. **SIFT + brute-force matching** capped correspondence quality → fewer, less
   accurate tracks → weaker poses and a sparse init cloud → everything downstream
   suffered. This was the ceiling.
2. **Mean-depth fusion**: the mesh fused `Σwz/α`, which blends front and back
   surfaces along each ray into a phantom mid-surface. Every surface-reconstruction
   paper (2DGS, RaDe-GS, GOF) fuses **median** depth instead.
3. **Off-surface gaussians**: vanilla 3DGS optimizes appearance only; nothing
   holds a gaussian on the surface, so rendered depth is noisy regardless of budget.

## What was implemented

### 1. SfM total rewrite — DISK + LightGlue (`features.py`)
- Learned **DISK** keypoints/descriptors replace SIFT; learned **LightGlue** graph
  matcher replaces brute-force + Lowe ratio. Both run in pure PyTorch via **kornia**,
  so they work on CPU / CUDA / Apple-Silicon **MPS** with no native build.
- **Content-based retrieval loop closure**: each frame is matched to its most
  globally-similar non-adjacent frames (mean-pooled descriptor kNN), catching real
  loop closures a fixed temporal stride misses.
- Public interfaces are unchanged (`FrameFeatures`, `detect_features`,
  `match_frames`, `_match_pair`, `_verify_pair`), so `sfm.py` (incremental mapper +
  bundle adjustment) is untouched — it simply receives far better correspondences.
- **Validated:** 40/40 cameras, 0.61 px mean reprojection error, ~3,600
  features/frame; 24/24 cameras, 0.57 px on a 24-frame subset.

### 2. Median-depth mesh (`render.py`, `mesh.py`)
- `RenderInfo.median_depth`: in the compositing loop, record the z of the first
  gaussian at which front-to-back transmittance drops below 0.5 — the true front
  surface. TSDF fusion now uses this instead of the mean.
- **Validated:** re-fused the existing HQ splat with *zero retraining* (via the new
  `export.model_from_splat`) → mesh surface roughness **0.27 → 0.13** (≈ halved),
  visibly crisper object edges (cooler, bowl rim).

### 3. Surface regularization + mesh cleanup
- **Flatten loss** (`TrainConfig.flatten_weight`, CLI `--flatten`): pushes gaussians
  to thin surface disks (minimize `smin/smid`) → crisper median depth. Trains
  end-to-end; flattening confirmed.
- **Screened-Poisson meshing** (`poisson_mesh`): watertight, hole-filling surface as
  a complement to TSDF. Verified correct on a clean oriented cloud (reconstructs a
  unit sphere to radius mean 1.000, std 0.000).
- **Taubin smoothing** (`fuse_tsdf(smooth_iters=)`): volume-preserving denoise of the
  marching-cubes staircase.
- **`export.model_from_splat`**: rebuild a renderable model from a `.splat` to
  re-mesh any trained scene without retraining.

## Honest findings (negative results kept)

- **Poisson from TSDF-mesh vertices is _not_ better than TSDF** — that vertex cloud
  is noisy/non-uniform, so Poisson comes out sparse/unstable. Poisson's payoff needs
  a clean *dense oriented* cloud (the trained splat's median-depth surface, or a
  learned dense-geometry backbone). The function is correct; the input matters.
- **Monocular depth supervision hurts splat held-out quality** (prior finding,
  re-confirmed): it fights the photometric loss because monocular depth is only
  defined up to an affine. Depth is kept for **mesh fusion**, not splat training.
- **VGGT / MASt3R / pycolmap** (the CUDA-heavy dense-geometry SOTA) were evaluated
  and deferred: this machine is **MPS-only**, and those need CUDA. DISK+LightGlue is
  the SOTA path that actually runs here.

## Follow-up work (done since the initial report)

- **Rasterizer ~3.6× faster** (32 px tiles + higher per-tile cap) — *and* more
  accurate (the old cap silently dropped gaussians). Lifted the resolution ceiling.
- **NaN/inf training guards** — a run had collapsed to 4 gaussians when a single
  NaN backpropped into every parameter; guards now skip such steps.
- **`mesh` subcommand + dense-mesh controls** — `splatvid mesh <dir>` rebuilds the
  mesh from an existing splat with **no retrain**; `--mesh-fine` (finer voxel +
  450k faces) gives ~+25 % surface detail; also `--mesh-faces`, `--mesh-voxel`,
  `--mesh-smooth`, `--mesh-method {tsdf,poisson}`.
- **Dense-frame matching made tractable** (three compounding speedups): density-aware
  candidate-pair count, per-frame tensor caching, and a **frame-density-aware
  keypoint cap** (LightGlue is ~O(kpts²)/pair). A ~50 %-of-video run dropped from
  >87 min of matching (unfinished) to ~25 min.
- **`render_mesh(shade=True)`** — flat shading so mesh *relief* is visible for
  geometry judgement.

## Findings (including negatives — don't re-try)

- **`--flatten` is unstable** (drove scales to degenerate covariance → NaN). Off by
  default; not recommended. Median-depth + a good SfM carry the mesh.
- **Monocular depth supervision hurts the splat** (affine-ambiguous, multi-view
  inconsistent); keep depth for *mesh fusion* only.
- **Appearance embeddings (`--appearance`) gave no measurable benefit** on this
  stable-exposure clip (floaters 1.3 % vs 1.2 %). Opt-in; useful only for clips with
  real exposure drift.
- **The median-TSDF surface's bumpiness is view-count-limited, not fixable in
  post.** Bilateral warps, mono-blend distorts, Poisson-from-depth flattens, Taubin
  and cluster-cleanup don't touch it. The fix is **more views** (denser TSDF
  averaging → smoother *and* more complete together) — hence dense-frame runs.

## How to run

```bash
# reliable dense-mesh reconstruction (guards on, geometry-faithful surface):
uv run splatvid reconstruct video.mov -o out --mesh --mesh-fine --floater-fix --device mps

# re-mesh an existing result with different settings, no retrain:
uv run splatvid mesh out --mesh-fine
```

`--mesh --mesh-fine` fuses a dense median-depth TSDF surface. Do **not** use
`--flatten` (unstable). See `docs/structure-from-motion.md` for the SfM stage.
