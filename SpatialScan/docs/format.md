# Apple spatial video, and how SpatialScan reads it

Apple *spatial video* is captured by the iPhone 15 Pro / Pro Max and the Vision
Pro. On disk it is an ordinary QuickTime `.mov`, but two things make it special:

1. The video track is **MV-HEVC** (Multiview HEVC): one HEVC **base layer**
   holding one eye plus a **dependent layer** holding the other eye, coded
   together in a single elementary stream.
2. The **stereo geometry** needed to interpret those two eyes metrically lives
   in the container as extension atoms — not in the pixels.

SpatialScan only needs the geometry from the container; the pixels are decoded
by ffmpeg (or supplied as a side-by-side / dual-file export).

## The atoms we read

QuickTime / ISO-BMFF files are a tree of *boxes* (a.k.a. atoms), each
`[4-byte size][4-byte type][payload]`. The stereo description sits inside the
video sample entry:

```
moov
└─ trak
   └─ mdia
      └─ minf
         └─ stbl
            └─ stsd
               └─ hvc1                     ← HEVC visual sample entry
                  └─ vexu                  ← Video Extended Usage (stereo)
                     ├─ eyes
                     │  ├─ stri            ← stereo view info (eye layout)
                     │  └─ cams
                     │     └─ blin         ← baseline (µm)
                     └─ proj
                        └─ hfov            ← horizontal FOV (milli-degrees)
```

`spatialscan/quicktime.py` walks this tree (`parse_atoms`) and pulls out:

| atom | field | units | used for |
|------|-------|-------|----------|
| `blin` | baseline | micrometres | metric scale: `Z = fx·B / disparity` |
| `hfov` | horizontal FOV | milli-degrees | intrinsics `fx = w / (2·tan(hfov/2))` |
| `stri` | eye flags | bitfield | which eyes are present; L/R reversed |
| `vexu`/`lhvC`/`hvcE` | — | — | detect that this *is* MV-HEVC |

### Robustness

The outer box structure (sizes, nesting, the 78-byte `VisualSampleEntry`
header) is fully specified by ISO-BMFF, so the tree walk is exact. The scalar
*leaf* layouts are read best-effort with sanity clamps:

- baseline is accepted only in `0.5 mm … 500 mm`,
- hFOV only in `20° … 160°`,

and the reader probes a couple of field offsets to tolerate an optional
version/flags prefix. If a field is missing or implausible, the loader falls
back to a supplied value (`--baseline-mm` / `--hfov`) or an iPhone-15-Pro
default (`baseline ≈ 19.2 mm`, `hFOV ≈ 63°`). A wrong guess is therefore never
silently trusted — it is clamped away.

## Decoding the two eyes

Splitting MV-HEVC into two ordinary videos needs a decoder that understands the
dependent layer. `spatial.py` shells out to **ffmpeg ≥ 7.1** to write a
left-eye and right-eye stream, then samples frames from each. If ffmpeg is
unavailable (or too old), export the clip to a **side-by-side** file from your
editor and load it with `mode="sbs"`, supplying the baseline and hFOV manually.

## Why stereo, not SfM

Structure-from-motion recovers geometry only up to an unknown global scale — a
reconstruction could be a dollhouse or a cathedral. Spatial video ships a
*calibrated* stereo rig, so every disparity converts to a depth in real metres,
and metric depth in turn keeps RGB-D odometry free of scale drift. The result is
a mesh you can measure.
