# Deferred Neural Renderer — design & implementation plan

Status: **proposal** (not implemented). Target: a follow-up phase after
PR #5 (SfM + viewer fixes) merges.

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
| `feature` | `(N, C)` | `C = 16` (start), up to 32 |
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
| **Adversarial** (PatchGAN) | plausibility/sharpness, hallucination | M3 |

- **Temporal-warp (the key anti-popping piece):** sample a pair of nearby
  cameras, render both, reproject one into the other using rendered depth `D`,
  and penalize photometric difference in the co-visible region. This *forces*
  the shader to be view-consistent, which is exactly what removes popping.
  Without it, a per-frame CNN can *introduce* flicker — so this loss is
  mandatory for the goal, not optional.
- **LPIPS** needs a pretrained VGG (torchvision or the `lpips` package) — an
  optional dependency.
- **Adversarial** (a PatchGAN discriminator = the "context-aware comparator")
  adds realism but is unstable; gate behind a flag and enable only in M3.

## Component 5 — training (`train.py`)

Two-stage, warm-started:

1. **Geometry stage:** existing gaussian training (direct color) → good proxy
   geometry. Unchanged.
2. **Neural stage:** train the shader + `feature` (optionally fine-tune
   geometry jointly at a low LR) with the loss stack. Curriculum: L1+SSIM+LPIPS
   first, add temporal at M2, adversarial at M3.

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
| **M0** | feature gaussians + feature rasterizer + trivial 1×1-conv "shader" | reproduces current RGB pipeline within tolerance (plumbing proven) |
| **M1** | U-Net shader + L1+SSIM+LPIPS | held-out LPIPS improves vs. baseline; visibly denoised |
| **M2** | temporal-warp loss | flow-warp error across a smooth orbit drops; popping visibly reduced |
| **M3** | adversarial loss | sharper/plausible; held-out PSNR not tanked |
| **M4** | half-res splat + upsample | ≥2× faster at equal quality |
| **M5** (opt) | diffusion pseudo-views | unobserved regions filled plausibly |

## Risks & mitigations

- **Overfitting / bad hallucination on novel views** → held-out metric +
  temporal + adversarial regularization.
- **GAN instability** → flagged off by default; M3 only, after M1/M2 are solid.
- **Per-frame flicker** → temporal-warp loss is mandatory, not optional.
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
| `losses.py` | `lpips()`, `temporal_warp_loss()`, PatchGAN + GAN losses |
| `train.py` | two-stage loop, shader optimizer, held-out eval |
| `cli.py` | `--neural`, `--neural-features C`, `--adversarial`, `--render-scale` |
| `pyproject.toml` | optional `neural` extra (torchvision/lpips) |
| `viewer.html` | unchanged (stays the raw-gaussian preview) |

## Open questions to settle before coding

1. Feature width `C` (16 vs 32) — quality vs. splat memory/bandwidth.
2. Joint vs. frozen geometry in the neural stage.
3. Half-res splat from the start, or add in M4.
4. Temporal pairs from real neighbor frames vs. synthesized nearby cameras.
5. Do we need the adversarial loss at all if LPIPS + temporal already hit the
   "no popping / no jaggies" bar? (Defer; decide after M2.)
