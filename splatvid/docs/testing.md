# Test architecture

The test suite is the executable specification of everything the other docs
claim. It is layered the same way the code is (see the module structure in
[pipeline-overview.md](pipeline-overview.md)): fast unit tests pin each
foundation module to its contract, and one slow end-to-end test proves the
four stages compose into a working reconstruction. Every quantitative claim
in these docs (0.19 px reprojection error, ≈18.5 dB PSNR, focal within
0.1 %) comes from this suite, not from a one-off measurement.

```bash
python -m pytest                # everything, incl. the slow end-to-end run
python -m pytest -m 'not slow'  # fast unit tests only (~seconds)
```

`tests/conftest.py` puts the repo root on `sys.path` (so tests run without
installing the package) and registers the `slow` marker.

## What each file guards

| File | Speed | Layer under test | Key assertions |
|---|---|---|---|
| `test_geometry.py` | fast | `geometry.py` | quaternion / Rodrigues / rotation-matrix round-trips agree; batched matches single; `project_points` ∘ `triangulate_point` recovers a known 3D point; camera center and triangulation angle are correct |
| `test_render.py` | fast | `render.py` | `Σ = R S Sᵀ Rᵀ` is isotropic for equal scales; projected center and depth match the pinhole formula; a single splat peaks at its center; **front-to-back compositing order** is respected; **finite nonzero gradients reach every parameter** (xyz, scale, quat, color, opacity); behind-camera splats render empty; moving the camera moves the image |
| `test_export.py` | fast | `export.py`, `model.py` | `.splat` write→read round-trips xyz/count; `.ply` header lists the 17 properties in order and the file is the expected size; model activations (sigmoid/exp/SH) match; `densify_and_prune` respects the `max_gaussians` budget |
| `test_sfm_synthetic.py` | fast | `ba.py` | bundle adjustment drives a perturbed camera/point problem from ~9 px RMS back below 1 px; the gauge-fixed camera stays fixed (no images needed — pure synthetic geometry) |
| `test_end_to_end.py` | **slow** | full pipeline | renders a synthetic orbit, runs SfM → training → export; asserts ≥70 % cameras registered, the recovered camera centers form a **coplanar equidistant ring**, PSNR clears a floor (>14 dB on the short CPU schedule), and both export formats round-trip |

## Why this shape

- **The renderer is tested for gradients, not just pixels.** `render.py` is
  useless to the trainer unless autograd flows through it, so
  `test_render.py::test_render_gradients_flow` asserts a finite, nonzero
  gradient reaches each of the five learnable tensors — the property that
  makes the whole splatting stage trainable.
- **BA is tested against ground truth without images.** `synthetic.py` lets
  `test_sfm_synthetic.py` construct a scene whose true poses and points are
  known, perturb them, and confirm bundle adjustment reconverges — isolating
  the optimizer from the noisier feature-matching front end. (The
  interleaved-residual / Jacobian-sparsity bug described in
  [structure-from-motion.md](structure-from-motion.md) is exactly what this
  test catches.)
- **The end-to-end test checks geometry, not just error metrics.** A low
  reprojection error can hide a wrong-shaped reconstruction, so
  `test_sfm_recovers_orbit` verifies the recovered cameras are actually a
  planar ring (small third singular value, low radius variance) — the shape
  the synthetic camera path really had, up to the unrecoverable global
  scale.

The synthetic scene (`synthetic.py::make_scene`) is a textured cube shell
plus a ground disc built from thousands of small high-contrast gaussians —
feature-rich enough for SIFT — rendered with the package's own rasterizer.
Running the pipeline on it exercises every stage against known ground truth,
which is also what `scripts/make_demo.py` does for an interactive demo.
