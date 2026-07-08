# Structure from motion: video → cameras + point cloud

> **Note (overhaul):** the feature/matching stage was rewritten from SIFT +
> brute-force to the learned **DISK + LightGlue** stack (kornia) with retrieval
> loop closure — see [`overhaul-report.md`](overhaul-report.md). The mapper /
> bundle-adjustment stages below are unchanged; sections describing SIFT
> descriptors now refer to DISK descriptors, and Lowe-ratio matching to
> LightGlue. Result on IMG_6547: 40/40 cameras, 0.61 px reprojection error.

SfM answers two questions simultaneously from pixels alone: *where was the
camera for each frame?* and *where are the observed points in 3D?* This
document walks the exact algorithm in `video.py`, `features.py`, `sfm.py`,
`ba.py`, in execution order, with the underlying geometry.

## 0. Frame selection (`video.py`)

Videos are redundant (30+ fps) and often blurry. Feeding every frame to SfM
wastes quadratic matching time and pollutes it with motion blur, so:

- The video timeline is split into `max_frames` equal windows (default 60).
- Within each window up to 3 probe frames are decoded and scored by
  **variance of the Laplacian** — a cheap focus measure: blur suppresses
  high spatial frequencies, and the Laplacian responds only to those, so a
  sharp frame has much higher response variance.
- The sharpest probe per window is kept and downscaled so its long side is
  ≤ `--frame-size` (default 960 px). SIFT is scale-invariant per octave,
  and going beyond ~1000 px mostly adds cost, not correspondences.

Result: up to 60 sharp frames, evenly spread, all the same size.

## 1. Features and matching (`features.py`)

### Detection

Each frame gets up to 4000 **SIFT** keypoints with 128-D descriptors.
SIFT is used because it is invariant to scale and rotation and tolerant of
moderate viewpoint/illumination change — exactly what consecutive video
frames exhibit — and its descriptors are distinctive enough for the strict
ratio test below. The pixel color at each keypoint is sampled so that
triangulated points end up with real colors (later used to initialize
gaussian colors).

### Pair selection

Matching all `n·(n−1)/2` pairs is wasteful for temporally ordered frames.
The pipeline matches:

- every pair within a **sliding window** of 6 frames (the workhorse
  correspondences), and
- a sparse grid of **long-range pairs** (every 15th frame against every
  other 15th) — these are loop-closure candidates: if the camera orbits an
  object, first and last frames see the same surface, and a match between
  them locks the loop and prevents accumulated drift.

### Match filtering — three layers

Raw nearest-neighbor descriptor matches are full of lies; three
independent filters remove them:

1. **Lowe ratio test** (0.75): accept a match only if the best descriptor
   distance is < 0.75× the second-best. If two candidates look nearly
   equally good, the feature is ambiguous (repetitive texture) and is
   dropped.
2. **Mutual cross-check:** the match must be each keypoint's best match in
   *both* directions.
3. **Geometric verification:** all surviving matches of a pair are fit
   with a RANSAC **fundamental matrix** (2 px threshold). The fundamental
   matrix encodes the epipolar constraint — for any rigid scene and any
   two camera positions, a correct match `(x, x′)` must satisfy
   `x′ᵀ F x = 0`, i.e. `x′` lies on the epipolar *line* of `x`. Matches
   violating this cannot come from a static rigid scene (they are mismatched
   or on moving objects) and are discarded. Pairs with < 30 verified
   matches are dropped entirely.

### Tracks

Pairwise matches are merged into **tracks** with union-find: keypoint
(frame i, index a) matched to (frame j, index b) joins them into one
connected component = one physical scene point observed in many frames.
A track in which one frame appears twice with different keypoints is
internally contradictory (some match was wrong) and is discarded whole.
Tracks are the unit of triangulation: one track ↦ one 3D point.

## 2. Initialization (`sfm.py::_init_pair`)

Everything starts from one two-view reconstruction, so the seed pair must
be chosen well. Each verified pair is scored by:

- **essential-matrix inlier count** (more support = better), penalized by
- the **homography ratio**: if nearly as many matches fit a homography as
  fit the essential matrix, the pair is degenerate for initialization —
  a homography explains image pairs related by pure rotation or a single
  plane, both of which give no or ill-conditioned parallax. High
  `#H-inliers / #E-inliers` ⇒ score penalty.

The score ranks candidates, but the seed pair is chosen by the *realized*
seed cloud: each top-ranked pair is actually triangulated and the one that
yields the most well-conditioned points wins. Ranking by match count alone
favours temporally adjacent frames, whose short baseline triangulates
poorly — so this is what lets slow, real handheld video initialize at all
(`sfm.py::_init_pair`).

For the winning pair:

1. The **essential matrix** `E` is estimated with RANSAC from the matches
   and the initial intrinsics guess `K` (focal = 1.2 × max(width, height),
   a reliable prior for consumer cameras — later refined). `E = [t]× R`
   packs the relative pose; unlike `F`, it lives in calibrated coordinates
   so the rotation and translation direction can be extracted from it.
2. `recoverPose` decomposes `E` into the four candidate `(R, t)` solutions
   and picks the one that puts the matched points *in front of both
   cameras* (the cheirality test).
3. Camera 1 is fixed at identity — this pins the global position and
   orientation (gauge). ‖t‖ is scaleless; whatever recoverPose returns
   defines the global scale unit.
4. All inlier matches are **triangulated** (below) into the seed cloud.

## 3. Triangulation (`geometry.py::triangulate_point`)

Given a track observed at pixels `uv_i` by cameras `P_i = K[R_i | t_i]`,
each observation contributes two linear equations in the homogeneous point
`X` (the DLT construction: `u · P₃ − P₁` and `v · P₃ − P₂` rows). Stacking
≥ 2 observations gives `A X = 0`, solved by SVD (smallest singular
vector). A candidate point is accepted only if (`sfm.py::_accept_point`):

- **cheirality:** positive depth in every observing camera;
- **reprojection:** max error over observations ≤ 4 px;
- **parallax:** the triangulation angle (angle at `X` between two camera
  centers) is ≥ 1°. Rays that are nearly parallel intersect
  ill-conditionedly — a 0.1° triangulation can move meters under one pixel
  of noise, so such points are refused rather than kept badly.

## 4. Incremental registration (`sfm.py::_register_next`)

With a seed cloud in hand, remaining frames are added one at a time:

1. Pick the unregistered frame that observes the most already-triangulated
   tracks (most 2D–3D correspondences = most constrained).
2. Solve **PnP** (perspective-n-point) with RANSAC: find `(R, t)` that
   projects the known 3D points onto their observed 2D keypoints (4 px
   inlier threshold, ≥ 10 inliers required). PnP needs only *one* new
   frame's observations because the 3D side is already known.
3. Triangulate every track that just gained its second (or later)
   registered view — the cloud grows as the camera set grows.
4. Every 5 registrations, run bundle adjustment (below) and re-filter
   points whose mean reprojection error exceeds 3 px.

Candidates are tried best-supported first; a failed PnP falls through to
the next rather than aborting, and because registration is retried after
every success (each adds more points), a frame too weak now gets another
chance once its neighbours fill in. Registration stops only when no
remaining frame has even a handful of correspondences or every PnP fails.
A final double round of BA + outlier filtering polishes the result.

**Fragmented video.** A phone clip often breaks into segments with little
overlap, so the pairwise-match graph splits into disconnected components
and incremental SfM can only grow one at a time. Two mitigations run before
reconstruction: extra cross-component matching (`_bridge_components`) tries
to reconnect segments the windowed/loop matcher missed, and reconstruction
then runs on the **largest** remaining connected component
(`_connected_components`) rather than whichever island the seed pair landed
in.

## 5. Bundle adjustment (`ba.py`)

Every stage above makes local, greedy decisions; errors accumulate.
Bundle adjustment is the global refinement that makes SfM accurate: it
minimizes total squared **reprojection error** over *all* poses, *all*
points, and the shared focal length simultaneously:

```
min over {f, R_c, t_c, X_p}   Σ_observations ρ( ‖π(f, R_c, t_c, X_p) − uv_observed‖² )
```

where `π` is the pinhole projection and `ρ` is a robust **soft-L1** loss
(residuals beyond ~2 px get down-weighted, so a few surviving bad matches
cannot drag the solution).

Implementation choices that matter:

- **Parameterization:** rotations as 3-vector axis-angle (Rodrigues) — a
  minimal, unconstrained parameterization suited to local optimization;
  focal as `log f` (keeps it positive, makes the step size scale-free).
  Parameter vector: `[log f | rvec, tvec per free camera | xyz per point]`.
- **Gauge fixing:** the first camera is excluded from the parameters and
  held at its pose, removing the 6 degenerate do-nothing directions
  (global rotation/translation). Global scale remains free — acceptable
  gauge slack for a monocular pipeline.
- **Sparsity:** the Jacobian of this problem is enormous but almost empty
  — each residual depends only on *its* camera's 6 parameters, *its*
  point's 3 coordinates, and `f`. `scipy.optimize.least_squares` is given
  this exact sparsity pattern (`jac_sparsity`), which lets its trust-region
  solver estimate the Jacobian with a handful of function evaluations via
  column grouping instead of one per parameter. Residuals are interleaved
  `(u₀, v₀, u₁, v₁, …)` so rows `2i, 2i+1` line up with the pattern —
  a mismatch here silently degrades the Jacobian and stalls convergence
  (this exact bug was caught by `tests/test_sfm_synthetic.py`, which
  verifies BA drives a perturbed problem from ~9 px RMS back below 1 px).

## 6. Output

`sfm.Reconstruction` carries the refined focal length, per-frame `(R, t)`,
and the triangulated points with mean colors and reprojection errors.
On the synthetic end-to-end test (a rendered orbit whose ground truth is
known), this pipeline registers 40/40 cameras with 0.19 px mean
reprojection error and recovers the focal length within 0.1 % — the
camera ring and the scene emerge from pixels alone.

## Failure modes to know

- **Pure rotation** (panning in place): no parallax → no triangulation →
  init-pair search fails by design (the homography penalty). The camera
  must *translate*.
- **Textureless / reflective / moving scenes:** few or geometrically
  inconsistent matches; the epipolar filter drops them and the
  reconstruction stays too small.
- **Rolling shutter, heavy fisheye:** violate the pinhole model; poses
  will absorb some error but quality degrades.
