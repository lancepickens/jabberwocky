"""Load an Apple spatial video into calibrated stereo frame pairs.

The container carries two things we need: the *pixels* of the two eyes
(MV-HEVC layers) and the *geometry* (baseline + hFOV, see ``quicktime.py``).
This module produces a :class:`SpatialVideo` — an iterable of rectified
left/right frames plus the :class:`Intrinsics` and baseline that make the
stereo metric.

Input modes, in order of fidelity to the real format:

* ``mode="mvhevc"`` — a true Apple ``.mov``. The two eyes are demuxed with
  ffmpeg (>= 7.1, which can decode both MV-HEVC layers) and geometry is read
  from the container metadata.
* ``mode="sbs"`` — a side-by-side video (``[left|right]`` per frame), the
  common lossy export. Geometry must be supplied (baseline/hFOV).
* ``mode="dual"`` — two separate left/right video files.
* Constructed directly from in-memory frame lists (used by the synthetic
  generator and tests).

Frames are returned as OpenCV BGR ``uint8`` of identical size. Apple frames are
already rectified (parallel, row-aligned), so no rectification is applied by
default; ``rectify=`` accepts a calibration if you feed raw pairs.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import tempfile
from dataclasses import dataclass
from typing import Iterator

import cv2
import numpy as np

from .geometry import Intrinsics
from .quicktime import SpatialMetadata, extract_spatial_metadata

log = logging.getLogger("spatialscan")

# iPhone 15 Pro spatial-video defaults, used only when the container omits them.
DEFAULT_BASELINE_M = 0.0192   # ~19.2 mm between the wide and ultra-wide cameras
DEFAULT_HFOV_DEG = 63.0       # horizontal field of view per eye


@dataclass
class StereoFrame:
    left: np.ndarray   # BGR uint8
    right: np.ndarray  # BGR uint8
    index: int


class SpatialVideo:
    """A calibrated sequence of stereo pairs from a spatial video."""

    def __init__(self, left: list[np.ndarray], right: list[np.ndarray],
                 intrinsics: Intrinsics, baseline_m: float,
                 metadata: SpatialMetadata | None = None):
        if len(left) != len(right):
            raise ValueError("left/right frame counts differ")
        if not left:
            raise ValueError("no frames")
        self.left = left
        self.right = right
        self.intrinsics = intrinsics
        self.baseline_m = float(baseline_m)
        self.metadata = metadata or SpatialMetadata()

    def __len__(self) -> int:
        return len(self.left)

    def __iter__(self) -> Iterator[StereoFrame]:
        for i, (l, r) in enumerate(zip(self.left, self.right)):
            yield StereoFrame(l, r, i)

    # -- constructors --------------------------------------------------------

    @classmethod
    def open(cls, path: str, *, mode: str = "auto", max_frames: int = 60,
             stride: int = 1, max_dim: int | None = 1024,
             baseline_m: float | None = None, hfov_deg: float | None = None
             ) -> "SpatialVideo":
        """Open a spatial video from disk.

        ``baseline_m`` / ``hfov_deg`` override whatever the container reports
        (and supply it for SBS/dual inputs, which carry no geometry).
        """
        if mode == "auto":
            mode = _guess_mode(path)
        log.info("Opening spatial video %s (mode=%s)", path, mode)

        meta = SpatialMetadata()
        if mode == "mvhevc":
            # The stereo atoms live in moov (at the end of an iPhone .mov); read
            # just that box, seeking past the huge mdat rather than slurping the
            # whole clip.
            meta = extract_spatial_metadata(read_moov_bytes(path))
            log.info("Container metadata: %s", meta.describe())
            left, right = _demux_mvhevc(path, max_frames, stride, meta)
        elif mode == "sbs":
            left, right = _read_sbs(path, max_frames, stride)
        elif mode == "dual":
            left, right = _read_dual(path, max_frames, stride)
        else:
            raise ValueError(f"unknown mode {mode!r}")

        # mvhevc orientation is resolved empirically in _demux_mvhevc; for the
        # SBS/dual paths honour the container's reversed-eyes flag if present.
        if mode != "mvhevc" and meta.eyes_reversed:
            left, right = right, left

        left = [_downscale(im, max_dim) for im in left]
        right = [_downscale(im, max_dim) for im in right]
        h, w = left[0].shape[:2]

        base = baseline_m or meta.baseline_m or DEFAULT_BASELINE_M
        fov = hfov_deg or meta.hfov_deg or DEFAULT_HFOV_DEG
        K = Intrinsics.from_fov(w, h, fov)
        log.info("%d stereo pairs at %dx%d, baseline=%.2fmm, hfov=%.1f deg",
                 len(left), w, h, base * 1000, fov)
        return cls(left, right, K, base, meta)


def _guess_mode(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".mov", ".mv-hevc", ".heic"):
        return "mvhevc"
    return "sbs"


def _downscale(im: np.ndarray, max_dim: int | None) -> np.ndarray:
    if not max_dim:
        return im
    h, w = im.shape[:2]
    s = max_dim / float(max(h, w))
    if s >= 1.0:
        return im
    return cv2.resize(im, (int(round(w * s)), int(round(h * s))),
                      interpolation=cv2.INTER_AREA)


def read_moov_bytes(path: str) -> bytes:
    """Return the ``moov`` box bytes, walking top-level boxes by header only.

    QuickTime stores the sample tables (and the spatial atoms) in ``moov``,
    which on an iPhone clip sits *after* a multi-GB ``mdat``. We read each
    top-level box header and seek past its payload, so we touch only headers
    until we reach ``moov`` — never loading ``mdat``. Falls back to the whole
    file if the walk can't locate it.
    """
    import struct as _struct
    with open(path, "rb") as f:
        f.seek(0, os.SEEK_END)
        size = f.tell()
        f.seek(0)
        pos = 0
        while pos + 8 <= size:
            f.seek(pos)
            hdr = f.read(16)
            if len(hdr) < 8:
                break
            bsize = _struct.unpack(">I", hdr[:4])[0]
            btype = hdr[4:8]
            header = 8
            if bsize == 1:
                bsize = _struct.unpack(">Q", hdr[8:16])[0]
                header = 16
            elif bsize == 0:
                bsize = size - pos
            if btype == b"moov":
                f.seek(pos)
                return f.read(bsize)
            if bsize < header:
                break
            pos += bsize
        f.seek(0)  # no moov found via header walk — hand back everything
        return f.read()


def _sample_indices(n: int, max_frames: int, stride: int) -> list[int]:
    idx = list(range(0, n, max(1, stride)))
    if len(idx) > max_frames:  # thin evenly to the budget
        keep = np.linspace(0, len(idx) - 1, max_frames).round().astype(int)
        idx = [idx[k] for k in keep]
    return idx


def _read_video_frames(path: str, max_frames: int, stride: int) -> list[np.ndarray]:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"could not open video {path!r}")
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or (max_frames * stride)
    want = set(_sample_indices(total, max_frames, stride))
    out, i = [], 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if i in want:
            out.append(frame)
        i += 1
        if len(out) >= max_frames:
            break
    cap.release()
    if not out:
        raise ValueError(f"no frames decoded from {path!r}")
    return out


def _read_sbs(path: str, max_frames: int, stride: int):
    frames = _read_video_frames(path, max_frames, stride)
    left, right = [], []
    for fr in frames:
        w = fr.shape[1] // 2
        left.append(np.ascontiguousarray(fr[:, :w]))
        right.append(np.ascontiguousarray(fr[:, w:2 * w]))
    return left, right


def _read_dual(path: str, max_frames: int, stride: int):
    # ``path`` points at the left file; the right file is the sibling with
    # "left"->"right" (or "_L"->"_R") swapped.
    right_path = None
    for a, b in (("left", "right"), ("_L", "_R"), ("_l", "_r")):
        if a in path:
            right_path = path.replace(a, b)
            break
    if right_path is None or not os.path.exists(right_path):
        raise FileNotFoundError(
            "dual mode needs a matching right-eye file (left/right or _L/_R)")
    return (_read_video_frames(path, max_frames, stride),
            _read_video_frames(right_path, max_frames, stride))


def _demux_mvhevc(path: str, max_frames: int, stride: int,
                  meta: SpatialMetadata) -> tuple[list, list]:
    """Split an MV-HEVC ``.mov`` into (left, right) frame lists.

    Prefers PyAV (which bundles a modern ffmpeg that decodes both MV-HEVC
    views); falls back to an ffmpeg CLI (>= 7.1) if PyAV is unavailable.
    """
    try:
        import av  # noqa: F401
        return _demux_mvhevc_pyav(path, max_frames, stride)
    except ImportError:
        pass
    return _demux_mvhevc_ffmpeg(path, max_frames, stride)


def _demux_mvhevc_pyav(path: str, max_frames: int, stride: int) -> tuple[list, list]:
    """Decode both MV-HEVC views via PyAV; view 0 and view 1 alternate.

    Requesting ``view_ids=all`` makes the HEVC decoder emit both eyes, paired by
    presentation time (``[v0,v1, v0,v1, ...]``). We keep only the pairs we need
    (even spread up to ``max_frames``) and return them as BGR ``uint8``.
    """
    import av

    container = av.open(path)
    vstream = container.streams.video[0]
    vstream.codec_context.options = {"view_ids": "all"}
    n_pairs = vstream.frames or (max_frames * max(1, stride))
    want = set(_sample_indices(n_pairs, max_frames, stride))

    view0, view1, pair = [], [], 0
    pending = None
    for frame in container.decode(vstream):
        if pending is None:
            pending = frame  # first eye of this timestamp
            continue
        if pair in want:
            # ``to_ndarray('bgr24')`` avoids touching frame.side_data, which
            # trips a PyAV enum bug on MV-HEVC's view side-data type.
            view0.append(pending.to_ndarray(format="bgr24"))
            view1.append(frame.to_ndarray(format="bgr24"))
        pending = None
        pair += 1
        if len(view0) >= max_frames:
            break
    container.close()
    if not view0:
        raise RuntimeError("PyAV decoded no MV-HEVC view pairs")
    log.info("Demuxed %d MV-HEVC view pairs via PyAV", len(view0))
    return _orient_stereo(view0, view1)


def _orient_stereo(view_a: list, view_b: list) -> tuple[list, list]:
    """Return (left, right) by picking the ordering with positive disparity.

    MV-HEVC's base view (view 0) is usually the left/hero eye, but rather than
    trust that we measure it: for correctly-ordered rectified stereo,
    ``SGBM(left, right)`` yields predominantly *positive* disparity. We test a
    mid-clip pair both ways and keep whichever agrees.
    """
    from .stereo import StereoConfig, compute_disparity

    mid = len(view_a) // 2
    a = cv2.cvtColor(_downscale(view_a[mid], 512), cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(_downscale(view_b[mid], 512), cv2.COLOR_BGR2GRAY)
    cfg = StereoConfig(num_disparities=96)
    ab = compute_disparity(a, b, cfg)
    ba = compute_disparity(b, a, cfg)
    score_ab = int((ab > 0).sum())
    score_ba = int((ba > 0).sum())
    if score_ba > score_ab:
        log.info("Auto-oriented eyes: view 1 is left (swapped, %d vs %d valid px)",
                 score_ba, score_ab)
        return view_b, view_a
    log.info("Auto-oriented eyes: view 0 is left (%d vs %d valid px)",
             score_ab, score_ba)
    return view_a, view_b


def _demux_mvhevc_ffmpeg(path: str, max_frames: int, stride: int) -> tuple[list, list]:
    """Fallback: split the MV-HEVC views with an ffmpeg CLI (>= 7.1)."""
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError(
            "no MV-HEVC decoder: install PyAV ('pip install av') or ffmpeg >= 7.1, "
            "or export the clip to side-by-side and use mode='sbs'.")

    with tempfile.TemporaryDirectory(prefix="spatialscan_") as tmp:
        left_mp4 = os.path.join(tmp, "left.mp4")
        right_mp4 = os.path.join(tmp, "right.mp4")
        cmd = [
            ffmpeg, "-y", "-loglevel", "error", "-i", path,
            "-map", "0:v:0", left_mp4,
            "-map", "0:v:0", "-view_ids", "1", right_mp4,
        ]
        try:
            subprocess.run(cmd, check=True, capture_output=True, timeout=1800)
            left = _read_video_frames(left_mp4, max_frames, stride)
            right = _read_video_frames(right_mp4, max_frames, stride)
            return _orient_stereo(left, right)
        except (subprocess.CalledProcessError, FileNotFoundError, ValueError) as e:
            raise RuntimeError(
                "ffmpeg could not split the MV-HEVC views (needs multi-view HEVC "
                "support, ffmpeg >= 7.1). Export to side-by-side and use "
                f"mode='sbs'. Underlying error: {e}") from e
