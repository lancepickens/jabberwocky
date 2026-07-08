# SpatialScan

Reconstruct a **metric scene mesh** from an **Apple spatial video**.

Apple spatial video (iPhone 15 Pro, Vision Pro) is a QuickTime `.mov` whose
video track is **MV-HEVC**: a calibrated horizontal stereo pair — two eyes with
a known baseline — carried in one stream. That calibration is what makes this
different from photogrammetry / SfM: depth comes straight out of stereo
disparity in real-world **metres**, with none of the scale ambiguity of
single-camera reconstruction.

```
spatial video ─▶ stereo depth per frame ─▶ RGB-D odometry ─▶ TSDF fusion ─▶ mesh
   (MV-HEVC)        Z = fx·B / disparity      pose per frame    surface
```

## Install

```bash
pip install -e '.[mesh]'      # open3d for TSDF meshing (recommended)
pip install -e .              # core only: still produces a fused point cloud
```

Decoding real `.mov` clips needs **ffmpeg ≥ 7.1** (multi-view HEVC). Without it
you can still feed a side-by-side export or two separate eye files.

## Use

```bash
# Inspect the stereo geometry stored in the container.
spatialscan info clip.mov
# -> MV-HEVC=True baseline=19.20mm hfov=63.00deg boxes=blin,hfov,stri

# Build a scene mesh from a spatial video.
spatialscan build clip.mov -o scene.ply

# No .mov handy? Render a synthetic spatial video and mesh it.
spatialscan demo -o demo.ply
```

Common options for `build`:

| flag | meaning |
|------|---------|
| `--mode {auto,mvhevc,sbs,dual}` | input layout (auto-detected by extension) |
| `--baseline-mm`, `--hfov` | override the container geometry |
| `--max-frames`, `--stride` | how much of the clip to use |
| `--voxel-mm` | TSDF voxel size (smaller = finer, slower) |
| `--target-faces`, `--smooth` | decimate / Taubin-smooth the mesh |

### Library

```python
from spatialscan.spatial import SpatialVideo
from spatialscan.scene import build_scene_mesh

video = SpatialVideo.open("clip.mov")          # or mode="sbs" / "dual"
result = build_scene_mesh(video, "scene.ply")
print(result.fusion.n_faces, "faces")
```

## Input modes

| mode | what it is | geometry source |
|------|-----------|-----------------|
| `mvhevc` | a real Apple `.mov` | read from the container (`vexu` atoms) |
| `sbs` | side-by-side `[left\|right]` video | you supply `--baseline-mm/--hfov` |
| `dual` | separate `left`/`right` files | you supply `--baseline-mm/--hfov` |

## How it works

1. **Container parse** (`quicktime.py`) — walk the ISO-BMFF atom tree and pull
   the stereo baseline (`vexu > eyes > cams > blin`, micrometres) and
   horizontal FOV (`hfov`, milli-degrees). See [`docs/format.md`](docs/format.md).
2. **Stereo depth** (`stereo.py`) — SGBM disparity with a left-right
   consistency check, then `Z = fx·baseline / disparity` for metric depth.
3. **RGB-D odometry** (`odometry.py`) — hybrid photometric+geometric
   registration of consecutive frames into a global trajectory. Metric depth
   means no scale drift.
4. **TSDF fusion** (`fusion.py`) — integrate every posed RGB-D frame into a
   voxel-hashed signed-distance volume and march cubes to a triangle mesh;
   without Open3D it falls back to a fused, voxel-downsampled point cloud.

## Tests

```bash
pytest
```

The suite runs the whole pipeline on a synthetic spatial video (a rendered room
with known geometry), so stereo depth, container parsing, and mesh scale are all
checked against ground truth without needing a real clip or ffmpeg.

## Layout

```
spatialscan/
  quicktime.py   ISO-BMFF atom reader + spatial-metadata extraction
  spatial.py     SpatialVideo loader (mvhevc / sbs / dual)
  stereo.py      disparity -> metric depth
  odometry.py    RGB-D frame-to-frame trajectory
  fusion.py      TSDF mesh (+ point-cloud fallback)
  geometry.py    intrinsics, back-projection, PLY I/O
  scene.py       high-level build_scene_mesh()
  synthetic.py   synthetic spatial video + container for tests/demo
  cli.py         `spatialscan build | info | demo`
```
