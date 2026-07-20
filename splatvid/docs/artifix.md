# Artifix: repairing incomplete and artifacted splats

`splatvid/artifix.py` is a post-training repair pass for gaussian-splat
scenes: it removes floaters that only novel viewpoints reveal, fills
under-observed holes with multi-view-verified content, and fine-tunes the
model on the repaired views. Run it as part of a reconstruction
(`splatvid reconstruct … --artifix`) or on an already-exported scene
(`splatvid artifix out/ --video original.mp4`).

## Where the design comes from

The pipeline structure is that of **ArtiFixer** (NVIDIA Research, SIGGRAPH
2026): *"ArtiFixer: Enhancing and Extending 3D Reconstruction with
Auto-Regressive Diffusion Models"* — [arXiv:2603.00492](https://arxiv.org/abs/2603.00492),
[project page](https://research.nvidia.com/labs/sil/projects/artifixer/).
Its observation: a 3DGS scene looks right from the captured views but breaks
— holes, floaters, punch-throughs — from viewpoints the camera never
visited, because nothing ever supervised those views. ArtiFixer's recipe:

1. **Render novel views** along trajectories that extend past the capture,
   and read the rasterizer's accumulated **opacity map** as per-pixel
   confidence. Their *opacity mixing* strategy encodes the degraded render
   into a video-diffusion model's latent space and injects noise where
   opacity is low — trusted content is preserved, holes are left for the
   generator to fill.
2. **Generate the repaired frames auto-regressively** (a bidirectional
   teacher distilled into a causal, camera-aware student via Self-Forcing
   DMD), so each repaired frame conditions the next and content stays
   consistent along the trajectory.
3. **Fine-tune the underlying 3D scene** on the repaired frames as
   pseudo-supervision (they report 1–3 dB PSNR over prior repair methods).

ArtiFixer's generator is a 14-billion-parameter video diffusion model
(finetuned Wan 2.1). splatvid ships no pretrained checkpoints and everything
must run from scratch on Apple Silicon, so the generator is replaced by a
**deterministic, geometry-aware filler** with the same information flow —
and deliberately *without* hallucination:

| ArtiFixer | splatvid artifix |
|---|---|
| Opacity map gates latent noise injection | Opacity map → confidence; artifact masks (floater + punch-through) revoke it |
| Video diffusion inpaints low-opacity regions | Plane-sweep warp of nearby captured frames; content accepted only where ≥2 sources photometrically agree on a depth |
| Auto-regressive frame conditioning | Fixed frames join the warp-source pool for subsequent views along the trajectory |
| Generated frames as pseudo-supervision | Fixed frames supervise fine-tuning, weighted per pixel by `(1-confidence) × verified-coverage` |
| Diffusion prior removes artifacts | Novel-view floater vote (ring test) + anchor-verified pruning |
| — | Hole seeding: verified fill pixels are unprojected into new gaussians (densification alone cannot create geometry in empty space) |

A learned generative prior can later replace the deterministic filler
through the `SplatRepairPrior` adapter (see `view_prior.py`, which was
always the designated seam for a Difix3D+/CAT3D/ArtiFixer-class model).

## The stages, in code order

1. **Extended trajectory** (`extended_trajectory`) — fits the captured
   camera ring (same math as the turntable) and sweeps a full orbit with
   elevation and radius oscillating out of the captured plane: exactly the
   under-observed directions. Poses are azimuth-ordered so consecutive
   views overlap — the auto-regressive walk needs that.
2. **Floater pruning** (`prune_floaters`) — from each novel pose, every
   opaque gaussian is tested against the rendered surface *around* it: depth
   is probed on a ring just outside its projected footprint (a blob would
   mask its own detection if probed underneath). A gaussian hanging clearly
   in front of a surrounding surface (*embedded* evidence), or a small blob
   with nothing around it at all (*isolated* evidence, held to a stricter
   vote), accumulates votes across poses. Candidates are then **verified**:
   muted (opacity → 0) and re-rendered against the captured anchors — a true
   floater is geometry the captured views never needed, so anchor PSNR must
   not drop. Groups that fail are bisected; whatever cannot be exonerated
   stays in the model. On a clean scene this typically prunes nothing — by
   design.
3. **Auto-regressive fixing walk** (`fix_view`, `sweep_fill`,
   `warp_source`) — each novel render's confidence map is
   `opacity-confidence × (1 - artifact masks)`; a floater *undercuts* the
   local surface depth, a punch-through (missing wall) *overshoots* it —
   both are re-opened for filling. The surface depth of trusted pixels is
   extended across the view (pull-push), then a small plane sweep warps the
   nearest captured frames (plus the last fixed frames — the AR context) at
   candidate depths around that continuation and keeps, per pixel, the depth
   where independent sources agree. Verified content replaces distrusted
   pixels (`fixed = lerp(render, merged, (1-conf)·cover)`); everything
   unverified passes through unchanged with zero supervision weight. The
   fixer imports real observations; it never invents content.
4. **Hole seeding** (`collect_seeds`, `GaussianModel.append_gaussians`) —
   pixels the splat could not explain but stereo verified are unprojected at
   the agreed depth into new gaussians (~2 px footprint, voxel-deduplicated
   across views, budget-capped). Densification only clones or splits
   existing gaussians, so without seeds an empty region can never grow
   geometry, no matter how it is supervised.
5. **Fine-tune** (`finetune`) — the trainer's loop shape (Adam groups, xyz
   LR decay, non-finite guards, densify + optimizer rebuild) over two
   pools: captured anchors with the standard L1+SSIM, and fixed novel views
   with per-pixel-weighted L1 (default 25 % of steps). A final verified
   prune sweeps up stragglers densified into the fill regions.

Without the original video, `splatvid artifix out/` anchors the fine-tune to
the model's own renders at the captured poses — holes still fill and
floaters still die, but captured views cannot get sharper than they already
are. Pass `--video` for real anchors when you have the footage.

## What to expect (measured)

On the repo's synthetic benchmark (sparse 4-view capture over a 135° arc,
camouflaged floaters injected on captured rays — the messy-capture case;
see `tests/test_artifix.py::test_artifix_end_to_end_repairs_scene`):

- held-out novel-view PSNR: **+0.5 dB** over the broken model, with the
  injected floaters removed and anchor-view PSNR *improved* (~+1 dB);
- novel-view opacity coverage rises (holes acquire geometry);
- on a clean, well-covered capture the pass is roughly neutral: the
  verified prune removes ~nothing, and fills are limited to what multiple
  frames corroborate.

Honest caveats. If your capture is dense and artifact-free, spending the
same iterations on plain training (`--iterations`) helps interpolated views
slightly more than artifix does — the fixer's value is *repair*
(artifacts, holes, extrapolated views), not extra sharpening. And geometry
that **no** frame observed cannot be recovered by a deterministic filler —
filling that plausibly requires a generative prior, which is what the
`SplatRepairPrior` seam is reserved for.

## Apple Silicon

Everything runs on the MPS backend (`--device auto` picks it on M-series):

- float32 throughout — MPS has no float64 (all tensors are created
  explicitly `dtype=torch.float32`, matching the rest of splatvid);
- built from ops Metal implements: `conv2d`/`avg_pool2d` (pull-push,
  median-filter unfold), `grid_sample` (warping), `cumprod` (compositing via
  the existing rasterizer), gather/argmin (plane sweep, ring probe);
- per-tile bookkeeping stays on the CPU exactly as in `render.py`, so the
  pass adds no extra device syncs beyond the renders it performs;
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set by the CLI as usual, so a torch
  build lacking any op falls back to CPU instead of aborting.

Cost scales with `--artifix-views` × render resolution plus the fine-tune
iterations; at the recommended M-series settings (`--train-size 640`) the
pass is a fraction of the original training run. The same
`scripts/bench_render.py` guidance applies: benchmark CPU vs MPS on your
machine at your sizes.
