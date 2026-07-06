# Performance, Apple Silicon, and the improvement roadmap

## Running on Apple Silicon (M-series, incl. M5)

splatvid auto-selects the best torch device: **CUDA → MPS → CPU**
(`cli.pick_device`). On an M-series Mac (M1–M5), `--device auto` (the
default) uses **MPS** — PyTorch's Metal backend — for the training stage;
the SfM stage is NumPy/SciPy/OpenCV and runs on the CPU's NEON/AMX units
via Accelerate, which is already fast on Apple Silicon. There is no
per-chip PyTorch build — Metal abstracts the GPU, so the same wheel that
runs on an M1 runs on an M5.

**First thing to run on a new Mac:**

```bash
pip install -e .          # in splatvid/
splatvid doctor           # or: splatvid doctor --device mps
```

`splatvid doctor` prints your torch build and device info, then renders a
tiny scene forward *and* backward on the target device and checks the
image and gradients match the CPU reference to within float32 tolerance.
It exercises exactly the operators that are the historically fragile parts
of the MPS backend — `cumprod` (transmittance), grouped `conv2d` (SSIM),
`argsort`, `nonzero`, and indexed slice assignment — so a green **PASS**
means the rasterizer is numerically correct on your machine, not just that
it runs. If it FAILs, upgrade to the current PyTorch release
(`pip install --upgrade torch`) and re-run; MPS operator coverage and
correctness fixes land continuously, so a recent torch matters most on the
newest chips. The same check runs automatically under `pytest` (the MPS
case skips on non-Mac machines).

Practical notes:

- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set automatically, so any operator
  a given torch version lacks on MPS falls back to CPU instead of
  aborting.
- MPS is float32-only; splatvid's tensors are float32 throughout, so
  nothing needs configuring.
- **Benchmark before committing to a device.** The rasterizer is a
  Python-orchestrated tile loop; each tile launches small kernels, and
  small-kernel launch overhead on MPS can eat the GPU advantage for small
  scenes/resolutions. The break-even moves with gaussian count and render
  size:

  ```bash
  python scripts/bench_render.py --device cpu
  python scripts/bench_render.py --device mps
  # larger, GPU-favoring workload:
  python scripts/bench_render.py --device mps --gaussians 80000 --size 640
  ```

  Rule of thumb: at the CPU-sized defaults (`--train-size 320`, ≤ 60k
  gaussians) CPU and MPS are comparable; MPS pulls ahead as you raise
  `--train-size` and `--max-gaussians`.
- Tile binning bookkeeping is kept on the CPU deliberately
  (`render.py`): the per-tile `any()` checks would otherwise force a
  blocking GPU sync per tile. Only the per-tile gather indices are moved
  to the device.
- Recommended M-series settings once you've confirmed MPS wins on your
  machine:

  ```bash
  splatvid reconstruct video.mp4 -o out/ \
      --device mps --train-size 640 --iterations 6000 --max-gaussians 150000
  ```

## Where the time goes today

| Stage | Cost driver | Bound by |
|---|---|---|
| Frame extraction | video decode | CPU, I/O |
| SIFT + matching | `O(pairs × features²)` descriptor distances | CPU (OpenCV, NEON) |
| Bundle adjustment | sparse trust-region least squares | CPU (SciPy) |
| Training | gaussian–tile overlaps × pixels, per iteration | torch device |

## Improvement roadmap

Ordered roughly by value-per-effort. Items marked **(perf)** speed things
up; **(quality)** improve output fidelity; **(robust)** widen the set of
videos that reconstruct.

### Rasterizer / training

1. **(perf) Batched tile rasterization.** Replace the Python per-tile
   loop with a padded gather: bucket gaussians by tile into one
   `(n_tiles, max_per_tile)` index tensor and composite all tiles in a
   single batched einsum. Removes per-tile kernel-launch overhead — the
   main thing holding MPS back — at the cost of padding waste. Biggest
   single speedup available without leaving pure PyTorch.
2. **(perf) Custom Metal/CUDA kernel** (or adopting `gsplat`'s kernels as
   an optional accelerator behind the same `render()` signature) for
   another order of magnitude. Keep the pure-PyTorch path as the readable
   reference implementation that tests validate against.
3. **(quality) Spherical harmonics degree 1–2** for view-dependent color:
   extend the color activation and exporters (the `.ply` layout already
   reserves `f_rest_*`). Cheap to add; visibly better on glossy scenes.
4. **(quality) Opacity-reset schedule** (periodically clamp opacities to
   ~0.01 and let them re-earn their keep) — the reference method's
   floater killer, worthwhile for longer training schedules.
5. **(quality) Progressive resolution:** train early iterations at half
   resolution, doubling later — faster convergence and better large-scale
   geometry before fine detail.

### SfM

6. **(perf) Approximate descriptor matching** (FLANN or PCA-reduced
   descriptors) — matching is the CPU bottleneck for long videos;
   brute-force L2 over 4000² descriptors per pair is the naive part of
   the current implementation.
7. **(robust) Radial distortion (k1, k2)** as bundle-adjustment
   parameters — action-camera and wide phone lenses currently violate the
   pinhole assumption.
8. **(robust) Optical-flow-based keyframe selection:** pick frames by
   accumulated flow (≈ constant parallax per keyframe) instead of uniform
   time windows; improves both SfM conditioning and coverage for
   variable-speed footage.
9. **(robust) Motion masking:** reject features on detected moving
   objects (epipolar filtering already drops most, but a person walking
   through half the frames still hurts).

### Pipeline / UX

10. **(quality) Per-frame exposure compensation** (a learned scalar gain
    per view) — phone auto-exposure otherwise bakes brightness seams into
    the model.
11. **(perf) Cache SfM results** (`cameras.npz` is already written;
    a `--resume` flag skipping straight to training is trivial plumbing).
12. **(quality) Held-out validation views** during training with PSNR
    reporting, to detect overfitting to the training frames.
13. **(UX) Live preview**: stream intermediate `.splat` snapshots to the
    bundled viewer over the `splatvid view` server while training runs.

## What was measured here

The claims in these docs come from the repository's own test loop
(synthetic orbit with known ground truth — see `tests/test_end_to_end.py`):
40/40 cameras registered at 0.19 px mean reprojection error, focal length
recovered within 0.1 %, and ≈18.5 dB PSNR after a deliberately short
250-iteration CPU schedule. MPS-specific numbers depend on your torch
build and chip — run `scripts/bench_render.py` on your machine rather
than trusting anyone's table.
