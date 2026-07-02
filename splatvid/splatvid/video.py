"""Frame extraction: pick a spread of sharp frames from an input video."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class FrameSet:
    """Extracted frames plus bookkeeping."""

    images: list[np.ndarray]  # BGR uint8, all the same size
    frame_indices: list[int]  # source frame index in the video
    fps: float
    source: str

    @property
    def size(self) -> tuple[int, int]:
        """(width, height) of the extracted frames."""
        h, w = self.images[0].shape[:2]
        return w, h


def _sharpness(gray: np.ndarray) -> float:
    """Variance of the Laplacian; higher means sharper."""
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def _resize_max_dim(img: np.ndarray, max_dim: int) -> np.ndarray:
    h, w = img.shape[:2]
    scale = max_dim / max(h, w)
    if scale >= 1.0:
        return img
    return cv2.resize(
        img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA
    )


def extract_frames(
    video_path: str,
    max_frames: int = 80,
    max_dim: int = 960,
    sharpness_window: int = 3,
) -> FrameSet:
    """Extract up to ``max_frames`` frames, spread evenly over the video.

    The video is divided into ``max_frames`` equal windows; within each
    window the frame with the highest Laplacian-variance sharpness (among
    up to ``sharpness_window`` probed frames) is kept, which skips most
    motion-blurred frames without decoding everything at full cost.
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(video_path)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    if total <= 0:
        # Some containers do not report frame counts; decode everything.
        frames_all: list[np.ndarray] = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames_all.append(frame)
        cap.release()
        if not frames_all:
            raise ValueError(f"No decodable frames in {video_path}")
        total = len(frames_all)
        reader = lambda i: frames_all[i]  # noqa: E731
    else:
        def reader(i: int) -> np.ndarray | None:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ok, frame = cap.read()
            return frame if ok else None

    n_windows = min(max_frames, total)
    bounds = np.linspace(0, total, n_windows + 1).astype(int)

    images: list[np.ndarray] = []
    indices: list[int] = []
    for wi in range(n_windows):
        lo, hi = bounds[wi], bounds[wi + 1]
        if hi <= lo:
            continue
        probes = np.unique(
            np.linspace(lo, hi - 1, min(sharpness_window, hi - lo)).astype(int)
        )
        best, best_idx, best_score = None, -1, -1.0
        for pi in probes:
            frame = reader(int(pi))
            if frame is None:
                continue
            small = _resize_max_dim(frame, max_dim)
            gray = cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
            score = _sharpness(gray)
            if score > best_score:
                best, best_idx, best_score = small, int(pi), score
        if best is not None:
            images.append(best)
            indices.append(best_idx)

    cap.release()

    if len(images) < 2:
        raise ValueError(
            f"Extracted only {len(images)} usable frame(s) from {video_path}; "
            "need at least 2 for reconstruction."
        )

    # Guard against mixed sizes (some streams change resolution mid-file).
    ref = images[0].shape
    keep = [i for i, im in enumerate(images) if im.shape == ref]
    images = [images[i] for i in keep]
    indices = [indices[i] for i in keep]

    log.info(
        "Extracted %d frames (of %d) from %s at %dx%d",
        len(images), total, os.path.basename(video_path),
        images[0].shape[1], images[0].shape[0],
    )
    return FrameSet(images=images, frame_indices=indices, fps=float(fps), source=video_path)
