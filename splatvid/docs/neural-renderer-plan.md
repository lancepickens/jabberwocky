# Deferred Neural Renderer — design & implementation plan

Status: **implemented (M0–M4)** — mechanisms landed and unit/smoke-tested;
per-scene quality tuning (long neural runs, a real diffusion prior for M4) is
future work. Built on PR #5 (SfM + viewer fixes).

### Decisions (locked)

1. **Feature width `C = 16`.**
2. **Geometry: freeze early, unfreeze late** — fixed during shader learning,
   then fine-tuned at low LR near the end.
3. **Render resolution: full-res first**, half-res-splat + learned upsampling
   added as a dedicated speed milestone (M3), not before.
4. **Temporal pairs: both** real temporally-adjacent frames *and* synthesized
   nearby-camera perturbations.
5. **No adversarial loss** — rely on LPIPS + temporal. Revisit only if results
   are too soft after M2.

## Goal

Eliminate the two artifacts that make splat renders look broken —
**popping** (discrete depth-sort order flipping as the camera moves) and
**jagged/spiky gaussians** (needle primitives, aliasing, hard alpha edges) —
while allowing **plausible hallucination** of detail the video never
captured. We are explicitly trading faithfulness for smoothness + plausibility
(see [gaussian-splatting.md](gaussian-splatting.md) for the current faithful
pipeline this builds on).

The chosen architecture keeps the gaussian geometry as a *controllable 3D
proxy* and moves final color generation into a learned image-space renderer.
The network is where smoothing (kills jaggies), temporal coherence (kills
popping), and hallucination (fills gaps) happen.

## Architecture at a glance

```
video ─▶ SfM ─▶ gaussians (xyz, scale, quat, opacity, FEATURE∈R^C)
                     │
                     ▼   (existing EWA splatting, C channels instead of 3)
              feature rasterizer ─▶ F (H×W×C) + alpha (H×W) + depth (H×W)
                     │
                     ▼   (new)
                U-Net neural shader ─▶ RGB (H×W×3)
                     │
                     ▼
     losses:  L1 + SSIM  +  LPIPS  +  temporal-warp  +  (adversarial)
                     │
                     ▼   backprop into shader AND features AND geometry
```

Deferred neural rendering / neural point-based graphics lineage: Thies
*Deferred Neural Rendering* (2019), Aliev *Neural Point-Based Graphics*
(2020), Rückert *ADOP* (2022), and feature-splat variants (Feature-3DGS).

## Component 1 — feature-carrying gaussians (`model.py`)

Add a learned per-gaussian feature vector alongside the existing raw params:

| Param | Shape | Notes |
|---|---|---|
| `xyz`, `log_scale`, `quat`, `opacity` | unchanged | geometry proxy |
| `feature` | `(N, C)` | `C = 16` (locked) |
| `color` (SH deg-0) | `(N, 3)` | **keep** — see warm start |

- Keep the SH-0 `color` so the pipeline still has a valid direct-color path
  (fallback, warm start, and the fast web-viewer preview all rely on it).
- Initialize `feature[:, :3] = (SH color)` and the rest to small noise, so the
  feature buffer starts as a valid-ish image and the shader can learn an
  identity map first.
- `get_feature()` returns raw features (no activation; the shader learns the
  mapping). Densify/prune must carry `feature` through `_rebuild` exactly like
  the other params.

## Component 2 — feature rasterizer (`render.py`)

The EWA splatting math is **channel-agnostic**: today it alpha-composites a
`(N, 3)` rgb tensor; generalize the "color" argument to `(N, C)`. Additions:

- Output a **feature map** `F ∈ R^{H×W×C}` (the current `tile_rgb` einsum, with
  `C` channels).
- Also output **alpha** `A ∈ R^{H×W×1}` (`1 - T_final`, already computed) and
  **depth** `D ∈ R^{H×W×1}` (weighted sum of gaussian depths — one extra
  einsum against the same weights `w`). Put these on `RenderInfo`.
- Everything stays autograd-friendly; gradients flow to `feature` and geometry
  unchanged. No change to the tiling/sorting.

Speed lever (ADOP-style): splat at **half resolution** and let the U-Net
upsample. Splatting cost drops ~4×; the shader recovers detail. Make the
render scale a config knob.

## Component 3 — the neural shader (`shader.py`, new)

A small U-Net mapping the splatted buffers to RGB:

- **Input:** concat of `F` (C) + `A` (1) + normalized `D` (1) + optional
  view-direction encoding (a few channels of `sin/cos` of the ray dir) →
  `(H×W×(C+2+k))`.
- **Body:** 3–4 down/up levels, skip connections, GroupNorm + GELU. Target
  **~1–5M params** for near-real-time. Final `sigmoid` → RGB in `[0,1]`.
- **Background:** the shader sees `A`, so it learns where geometry is; composite
  over a fixed or learned background color.
- **Why it fixes jaggies:** a learned decoder outputs natural-image statistics;
  it cannot emit sub-pixel needles, and it band-limits hard splat edges.

## Component 4 — loss stack (`losses.py`)

| Loss | Purpose | Phase |
|---|---|---|
| L1 + SSIM (existing) | anchor absolute color/structure | M1 |
| **LPIPS** (VGG/Alex) | context-aware perceptual match — the "compare to video" signal | M1 |
| **Temporal-warp** | **anti-popping** — coherence across nearby views | M2 |

- **Temporal-warp (the key anti-popping piece):** take a pair of nearby views,
  render both, reproject one into the other using rendered depth `D`, and
  penalize photometric difference in the co-visible region. This *forces* the
  shader to be view-consistent, which is exactly what removes popping. Without
  it a per-frame CNN can *introduce* flicker — so this loss is mandatory for the
  goal, not optional. **Pairs come from both** sources (decision 4): real
  temporally-adjacent registered frames (with GT on both sides), and
  synthesized small camera perturbations of a training view (self-supervised
  consistency between the two renders, covering unseen/turntable directions).
- **LPIPS** needs a pretrained VGG (torchvision or the `lpips` package) — an
  optional dependency.
- **No adversarial loss** (decision 5): LPIPS + temporal are expected to clear
  the "no popping / no jaggies" bar. A PatchGAN would add sharpness but also
  instability; only revisit if M2 results look too soft.

## Component 5 — training (`train.py`)

Two-stage, warm-started:

1. **Geometry stage:** existing gaussian training (direct color) → good proxy
   geometry. Unchanged.
2. **Neural stage:** train the shader + `feature` with geometry **frozen**,
   then **unfreeze geometry at a low LR near the end** (decision 2) to let it
   correct residual errors the shader can't paper over. Curriculum:
   L1+SSIM+LPIPS first, add temporal at M2.

- **Held-out views:** reserve N frames, never supervised, to measure novel-view
  quality (PSNR/LPIPS) — the guard against hallucination degrading unseen
  angles. This is the metric that tells us the network generalizes vs. memorizes.
- Separate optimizer/LR for shader (Adam ~1e-3) vs. geometry (existing LRs).

## Component 6 — generative hallucination (optional, phase 3)

For genuinely unobserved regions (e.g. the 68° gap in the cooler-bag clip):
generate **pseudo-views** along the orbit with a diffusion prior
(Difix3D+/ReconFusion/CAT3D style), conditioned on the real frames, and feed
them as extra supervision. Faithful regions stay supervised by real frames;
the prior only fills gaps. Heavy pretrained dependency — keep fully optional.

## Web-viewer implications (explicit tradeoff)

The bundled WebGL viewer renders raw gaussians directly and cannot run the
U-Net the same way. Plan:

- **Keep the raw-gaussian path as the interactive preview** (uses the retained
  SH `color`) — fast, orbit-able, but without neural polish.
- **Neural quality is offline/turntable-first:** render the shaded orbit
  server-side into a video (the "hero" output).
- Later, optionally export the shader to WebGPU/ONNX for in-browser neural
  shading (ambitious; separate effort).

This tension (real-time interactivity vs. neural quality) is inherent and
should be surfaced to users, not hidden.

## Milestones

| # | Deliverable | Success check |
|---|---|---|
| ✅ **M0** | feature gaussians + feature rasterizer + identity shader | reproduces current RGB pipeline exactly (plumbing proven) |
| ✅ **M1** | U-Net shader + L1+SSIM+perceptual (geometry frozen) | shape/gradient + two-stage smoke tests pass |
| ✅ **M2** | temporal-warp loss (real + synthesized pairs) + late geometry unfreeze | identity warp → 0; trains with temporal on |
| ✅ **M3** | half-res splat + learned upsampling | full-size output from half-res splat; trains |
| ✅ **M4** | pseudo-view supervision + pluggable ViewPrior hook | mechanism runs; awaits a real diffusion prior |

(No adversarial milestone — decision 5. Reintroduce only if M2 looks too soft.)

## Risks & mitigations

- **Overfitting / bad hallucination on novel views** → held-out metric +
  temporal regularization.
- **Per-frame flicker** → temporal-warp loss is mandatory, not optional.
- **Results too soft without adversarial** → accepted risk (decision 5);
  reintroduce a PatchGAN only if M2 held-out results look mushy.
- **Viewer can't run the shader** → keep raw-gaussian preview; neural is
  offline-first.
- **Dependency weight** (VGG/LPIPS, diffusion) → optional extras in
  `pyproject.toml`.

## Testing

- Extend `test_render.py`: feature rasterizer returns `(H,W,C)` + alpha + depth;
  finite gradients reach `feature` and geometry.
- **M0 sanity test:** identity shader on 3-channel features ≈ current RGB
  render (locks the plumbing).
- Regression: held-out PSNR/LPIPS on the synthetic scene ≥ raw-gaussian baseline.
- **Popping metric:** depth-warp consistency across an orbit, measured
  before/after the temporal loss (quantifies the headline goal).

## Code touch-points (summary)

| File | Change |
|---|---|
| `model.py` | `feature` param; carry through densify/prune; keep SH color |
| `render.py` | C-channel compositing; return feature + alpha + depth buffers |
| `shader.py` (new) | U-Net module |
| `losses.py` | `lpips()`, `temporal_warp_loss()` |
| `train.py` | two-stage loop (freeze → late-unfreeze), shader optimizer, held-out eval |
| `cli.py` | `--neural`, `--render-scale` (feature width fixed at 16) |
| `pyproject.toml` | optional `neural` extra (torchvision/lpips) |
| `viewer.html` | unchanged (stays the raw-gaussian preview) |

## Open questions — resolved

All five settled; see **Decisions (locked)** at the top. Summary: `C=16`;
geometry frozen then unfrozen late; full-res first (half-res in M3); temporal
pairs from both real and synthesized views; no adversarial loss.

Next actionable step: implement **M0** (feature gaussians + feature rasterizer
+ identity shader + sanity test) on a dedicated branch.
