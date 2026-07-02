# Gaussian splatting: point cloud → photorealistic model

This document explains the representation, the differentiable renderer,
and the optimization loop (`model.py`, `render.py`, `losses.py`,
`train.py`). It follows the 3D Gaussian Splatting formulation (Kerbl et
al. 2023), implemented here from scratch in pure PyTorch.

## 1. Why gaussians?

We need a scene representation that is:

- **renderable in closed form** from any viewpoint (no ray marching),
- **differentiable** in all parameters, so photometric error can be
  back-propagated into geometry and appearance,
- **adaptive** — capacity should concentrate where the scene has detail.

An anisotropic 3D gaussian is a soft ellipsoid: under perspective
projection it maps (to first order) to a 2D gaussian on screen, which can
be composited analytically. A scene is just an unordered set of them —
adding/removing capacity is trivial, and every parameter has a smooth
effect on the image. It is, in effect, a *learnable, renderable point
cloud with volume*.

## 2. The representation (`model.py`)

Each gaussian `i` carries, stored in **raw (pre-activation)** form —
matching the on-disk `.ply` convention:

| Parameter | Raw storage | Effective value | Why |
|---|---|---|---|
| position `μ` | 3 floats | as-is | — |
| scale `s` | `log s` (3) | `exp(·)` | positivity, multiplicative steps |
| rotation `q` | quaternion (4) | normalized | minimal smooth rotation param |
| color | SH degree-0 coeffs (3) | `0.5 + C₀·c`, clamped | matches 3DGS format; `C₀ = 0.28209…` |
| opacity `o` | logit (1) | `sigmoid(·)` | keeps `o ∈ (0,1)` |

The 3D covariance is composed as `Σ = R S Sᵀ Rᵀ` with `R` from the
quaternion and `S = diag(s)` — guaranteed symmetric positive
semi-definite by construction, which a directly-learned 3×3 matrix would
not be.

**Initialization from SfM** (`train.py::init_model`): one gaussian per SfM
point (worst 2 % distance-outliers dropped), color from the track's mean
observed color, opacity 0.1 (translucent start lets wrongly-placed
gaussians fade away cheaply), and isotropic scale set to the mean distance
to the 3 nearest neighbors — dense regions get small gaussians, sparse
regions get large ones that blanket the gap until densification refines
them.

## 3. Projecting a gaussian to the screen (`render.py`)

For a camera `(R, t)` with focal `f`:

1. **Center:** `p = R μ + t`, pixel position `u = f·pₓ/p_z + cx`,
   `v = f·p_y/p_z + cy`. Gaussians with `p_z ≤ 0.05` are culled.
2. **Covariance (EWA splatting):** perspective projection is nonlinear,
   so the 3D gaussian is pushed through its **first-order Taylor
   expansion** at the center. With the projection Jacobian

   ```
   J = [ f/p_z    0      −f·pₓ/p_z² ]
       [ 0       f/p_z   −f·p_y/p_z² ]
   ```

   the screen-space covariance is `Σ₂D = (J R) Σ (J R)ᵀ + 0.3·I`.
   The `+0.3 I` dilation is a low-pass filter: it guarantees every splat
   spans at least about a pixel, preventing sub-pixel gaussians from
   aliasing or vanishing (and killing their own gradients).
3. **Extent:** the splat's screen radius is taken as 3σ of the larger
   eigenvalue of `Σ₂D`; anything whose 3σ disk misses the image is culled.
   The inverse `Σ₂D⁻¹` (the *conic*) is what per-pixel evaluation uses.

## 4. Compositing (`render.py::render`)

Each pixel composites the depth-sorted splats front-to-back with alpha
blending. Splat `i` contributes at pixel `x`:

```
α_i(x) = min(0.99, o_i · exp(−½ (x−μ_i)ᵀ Σ₂D⁻¹ (x−μ_i)))

C(x)   = Σ_i  c_i · α_i(x) · T_i(x),    T_i(x) = Π_{j<i} (1 − α_j(x))
```

`T_i` is the **transmittance** — the fraction of light still unblocked
when splat `i` is reached. The final `T` times the background color is
added at the end. Two clamps stabilize training: α is capped at 0.99 (a
fully opaque splat would zero the transmittance — and the gradients — of
everything behind it), and α < 1/255 is zeroed (invisible contributions
would otherwise waste compute and add noise gradients).

### The tile trick

Evaluating every gaussian at every pixel is `O(N·H·W)` — hopeless. But a
gaussian only influences pixels within its 3σ ellipse, so the screen is
cut into **16×16 tiles** and each gaussian is binned into the tiles its
screen bounding box touches. Per tile, the overlapping gaussians (already
globally depth-sorted; capped at 1024 per tile) are evaluated against the
tile's 256 pixels as one vectorized tensor op, with the transmittance
product computed as an exclusive cumulative product along the sorted
gaussian axis. Cost drops to `O(Σ overlaps × 256)`.

### Differentiability — the whole point

The forward pass is ordinary PyTorch tensor arithmetic end to end
(projection, conic, exp, cumprod, weighted sum), so **autograd provides
the backward pass for free**: the photometric loss differentiates through
compositing → α → conic → covariance → quaternion/scale, and → projected
center → 3D position. There are only two deliberate non-differentiable
choices, both standard: the depth *ordering* is treated as constant
(sorting is discontinuous), and cull/binning decisions are made on
detached values (they select *which* computation runs, not its values).
`tests/test_render.py` asserts finite nonzero gradients reach every
parameter, and that compositing order and camera motion behave correctly.

## 5. The training loop (`train.py`)

Standard stochastic optimization, one random training view per iteration:

- **Loss:** `0.8·L1 + 0.2·(1 − SSIM)`. L1 anchors absolute color; SSIM
  (computed with an 11×11 gaussian window) compares local mean/variance/
  covariance structure, pushing edges and texture to be *structurally*
  right where plain L1 would happily blur.
- **Optimizer:** Adam with per-parameter-group learning rates (positions
  1.6e-4 × scene extent with exponential decay to 1 %, scales 5e-3,
  rotations 1e-3, colors 2.5e-3, opacities 2.5e-2). Positions are scaled
  by scene extent so the same config works for a desk scene and a
  courtyard; the decay lets geometry settle while appearance keeps
  refining.
- Frames are downscaled to `--train-size` (default 320 px on CPU) with
  intrinsics scaled to match.

## 6. Adaptive densification (`model.py::densify_and_prune`)

The SfM seed is sparse; fine detail needs more gaussians *in the right
places*. The signal used is the accumulated **screen-space positional
gradient** per gaussian: a gaussian that the loss keeps trying to drag
around is one gaussian trying to explain something it cannot fit — an
under-reconstructed region.

Every 150 iterations (between iteration 300 and 70 % of training), for
gaussians whose mean screen gradient exceeds 2e-4 (NDC units):

- **Split** (if the gaussian is large, > 1 % of scene extent): replace by
  two children sampled inside the parent, scales divided by 1.6 — adds
  resolution where one blob spans too much geometry.
- **Clone** (if small): duplicate in place — adds coverage where detail
  is denser than the gaussian count.

Simultaneously **prune** gaussians with opacity < 0.005 (optimization has
declared them useless) or footprints > half the scene (floaters that fog
the view). A hard `--max-gaussians` budget keeps memory bounded; the
hottest-gradient candidates win the remaining slots. After any parameter
surgery the Adam optimizer is rebuilt (its per-tensor moment state no
longer matches the new tensor shapes).

## 7. What comes out

Training ends with a transparent-splat prune and hands a `GaussianModel`
to the exporters (see `formats-and-viewer.md`). On the synthetic
end-to-end test the short CPU schedule reaches ≈ 18.5 dB PSNR in 250
iterations; real quality needs the longer defaults (or a GPU and raised
budgets — the code is device-agnostic).

## Design deviations from reference 3DGS

Kept deliberately simple in this from-scratch implementation:

- **SH degree 0 only** (flat color per gaussian): no view-dependent
  color; specular highlights get averaged. Degree ≥ 1 is a
  straightforward extension of the color activation.
- **No opacity reset schedule** (the reference periodically clamps all
  opacities down to fight floaters); pruning + the shorter schedules used
  here made it unnecessary.
- **Python-loop tiling** instead of a fused CUDA kernel: identical math,
  orders of magnitude slower — the price of a dependency-free, readable,
  autograd-checked renderer.
