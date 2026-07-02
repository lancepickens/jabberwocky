# Output formats and the WebGL viewer

`export.py` writes two formats; `viewer.html` renders the compact one in
any modern browser with no dependencies or network access.

## `scene.ply` — the interchange format

The de-facto standard 3D Gaussian Splatting PLY layout, as produced by the
reference implementation and read by SuperSplat, gsplat tooling, and most
splat editors. Binary little-endian, one vertex element per gaussian,
17 float properties in this exact order:

| Property | Count | Content |
|---|---|---|
| `x y z` | 3 | position |
| `nx ny nz` | 3 | normals — always zero, present for layout compatibility |
| `f_dc_0..2` | 3 | color as SH degree-0 coefficients (`rgb = 0.5 + 0.28209·f_dc`) |
| `opacity` | 1 | **logit** (apply sigmoid) |
| `scale_0..2` | 3 | **log** scales (apply exp) |
| `rot_0..3` | 4 | quaternion `(w, x, y, z)` |

Note the convention: the file stores *raw* optimization-space values
(logits, logs), not activated ones. Loaders are expected to apply the
activations — this is what makes the file round-trip exactly and is why
external tools can resume editing/training from it.

## `scene.splat` — the compact viewer format

The 32-byte-per-gaussian format introduced by antimatter15's web viewer
and now common for web delivery. Little-endian, no header, records
back-to-back:

| Bytes | Type | Content |
|---|---|---|
| 0–11 | 3 × float32 | position |
| 12–23 | 3 × float32 | scale (linear, activated) |
| 24–27 | 4 × uint8 | R, G, B, A (activated color; A = opacity × 255) |
| 28–31 | 4 × uint8 | quaternion `(w,x,y,z)`, each `round(q·128 + 128)` |

Records are sorted by `opacity × volume^(1/3)` descending, so a viewer
that streams the file progressively shows the most visually important
splats first. Quantization to uint8 costs ~0.4 % color/rotation precision
— invisible in practice — and shrinks scenes ~4× vs the PLY.
`export.load_splat` reads the format back for tests and tooling.

## `viewer.html` — how the browser renders splats

A single self-contained WebGL2 page (~350 lines, no libraries). The
interesting parts:

### Data path

The `.splat` buffer is unpacked into one RGBA32F **texture**, 4 texels per
gaussian (position+opacity / scale / quaternion / color). Textures rather
than vertex attributes because the draw order changes every time the
camera moves: the only per-instance vertex attribute is a single float
`sortedIndex`, and the vertex shader fetches the actual gaussian data with
`texelFetch`. Re-sorting therefore rewrites only a 4-byte-per-splat index
buffer, not the 64-byte payload.

### Draw

One instanced draw call: a 4-vertex triangle-strip quad × N instances.
Per instance the vertex shader:

1. rebuilds `Σ = R S Sᵀ Rᵀ` from quaternion + scale,
2. projects it with the same EWA Jacobian math as the Python renderer
   (one sign flip documented inline: WebGL cameras look down −z),
3. eigendecomposes the 2×2 screen covariance into major/minor axes, and
   stretches the unit quad to the **2σ ellipse** in NDC.

The fragment shader evaluates `α = opacity · exp(−½ d²)` from the
interpolated ellipse coordinate and outputs premultiplied alpha.

### Ordering and blending

Alpha compositing needs depth order, but per-pixel sorting is impossible
in the raster pipeline. The standard splat solution, used here:

- CPU-side **counting sort** of all splats by their depth along the
  current view axis — O(N) with 65 536 buckets, a few ms for hundreds of
  thousands of splats;
- draw **back-to-front** with `blendFunc(ONE, ONE_MINUS_SRC_ALPHA)`
  (premultiplied "over" operator), depth test off;
- re-sort only when the view direction has rotated meaningfully
  (dot < 0.999) — a global depth order is exact for one direction and a
  good approximation nearby.

### Controls & loading

Orbit (drag), pan (shift/right-drag), zoom (wheel), pointer-capture based;
the camera auto-frames the scene bounding sphere on load. The page tries
`fetch("scene.splat")` (the layout `splatvid reconstruct` writes), and
also accepts drag-and-drop or a file picker, so the single HTML file works
standalone with any `.splat` you throw at it. `splatvid view <dir|file>`
serves a directory with the viewer over localhost.

## Why two formats

PLY is lossless and editable but heavy and needs activation-aware loaders;
`.splat` is small, streamable, trivially parseable (fixed 32-byte stride)
and matches what web viewers want. Writing both costs nothing and covers
both workflows: keep `scene.ply` as the master, ship `scene.splat`.
