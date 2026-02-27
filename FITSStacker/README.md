# FITSStacker

A native **Apple Silicon** macOS application that stacks astronomical FITS light frames from the **ZWO Seestar S50** (and compatible) using GPU-accelerated debayering and registration.

---

## Requirements

| Requirement | Minimum |
|---|---|
| macOS | **14 Sonoma** |
| Chip | **Apple Silicon** (M1 / M2 / M3 / M4) |
| Xcode | **15+** |
| Swift | **5.9+** |

---

## Building

```bash
# Open in Xcode (recommended – gets full app bundle + code signing)
open FITSStacker/Package.swift

# Or build from command line (requires Xcode Command Line Tools on macOS)
swift build -c release --arch arm64
```

---

## Directory Structure

The app expects a **home directory** with the following layout:

```
~/MySessions/
├── lights/        ← Put your Seestar S50 .fits files here  (required)
├── process/       ← Scratch folder for intermediate outputs (auto-created)
└── output/        ← Final stacked images are written here  (auto-created)
```

---

## Workflow

1. Launch **FITSStacker**
2. Click **Choose…** and select your session home directory
3. Place all `.fits` / `.fit` / `.fts` light frames in `lights/`
4. Press **⌘↵** (or click **Stack Frames**)

The pipeline runs fully automatically:

| Step | What happens | Acceleration |
|---|---|---|
| **Verify** | Checks lights/ exists, finds FITS files | – |
| **Load** | Parses FITS headers + pixel data, normalises to [0,1] | – |
| **Debayer** | RGGB→RGB bilinear demosaicing | **Metal GPU** |
| **Register** | Star centroid detection, median translation per frame | **Accelerate (vDSP)** |
| **Stack** | GPU-accumulate with per-frame translation offsets, divide | **Metal GPU** |
| **Output** | 16-bit linear TIFF + 32-bit float FITS | CoreGraphics |

### Output files (`output/`)

- `stacked_YYYYMMDD_HHMMSS.tiff` — 16-bit RGB TIFF, linear, import directly into PixInsight / Siril / APP
- `stacked_YYYYMMDD_HHMMSS.fits` — 32-bit float FITS with 3 image planes (R/G/B)

---

## Apple Silicon Acceleration

| Technology | Used for |
|---|---|
| **Metal** (GPU) | Bayer demosaicing kernel, frame accumulation, normalisation |
| **Accelerate / vDSP** | SIMD float arithmetic, Gaussian blur, luminance conversion |
| **Accelerate / vImage** | Convolution for star detection |
| **BLAS (cblas)** | Channel de-interleaving |
| **Swift Concurrency** | Parallel per-frame registration on all P-cores |
| **Unified Memory** | Zero-copy CPU↔GPU texture access on Apple Silicon |

---

## Seestar S50 Notes

- The Seestar S50 saves FITS files with `BITPIX=16`, `BZERO=32768`, `BSCALE=1` — this app handles those defaults automatically.
- The Bayer pattern (`RGGB`) is read from the `BAYERPAT` or `BAYER` FITS keyword.  If absent, RGGB is assumed.
- Exposure time (`EXPTIME`) and gain (`GAIN`) are parsed from the header for logging.

---

## License

MIT — see `LICENSE` file.
