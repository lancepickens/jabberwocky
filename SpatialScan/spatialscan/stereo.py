"""Stereo matching: a rectified L/R pair -> a *metric* depth map.

Apple spatial video gives us a calibrated horizontal stereo pair, so depth is
recovered directly from disparity without any of the scale ambiguity that
plagues monocular / SfM reconstruction:

    Z = fx * baseline / disparity          (metres, because baseline is metric)

We use OpenCV's Semi-Global Block Matching (``StereoSGBM``), a left-right
consistency check to drop occluded/mismatched pixels, and a light speckle
filter. The left eye is the reference; the returned depth is in the left
camera's frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import Intrinsics


@dataclass
class StereoConfig:
    """Tunables for SGBM + post-filtering."""

    num_disparities: int = 128   # must be a multiple of 16; search range
    min_disparity: int = 0
    block_size: int = 5          # odd, 3..11
    uniqueness_ratio: int = 10
    speckle_window: int = 100
    speckle_range: int = 2
    lr_consistency_px: float = 1.0  # max left/right disparity disagreement
    max_depth_m: float = 12.0    # clamp far/degenerate matches (indoor scenes)

    def build(self, channels: int) -> cv2.StereoSGBM:
        nd = int(np.ceil(self.num_disparities / 16.0) * 16)
        c = max(1, channels)
        return cv2.StereoSGBM_create(
            minDisparity=self.min_disparity,
            numDisparities=nd,
            blockSize=self.block_size,
            P1=8 * c * self.block_size ** 2,
            P2=32 * c * self.block_size ** 2,
            disp12MaxDiff=int(round(self.lr_consistency_px)),
            uniquenessRatio=self.uniqueness_ratio,
            speckleWindowSize=self.speckle_window,
            speckleRange=self.speckle_range,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY,
        )


def compute_disparity(left: np.ndarray, right: np.ndarray,
                      cfg: StereoConfig | None = None) -> np.ndarray:
    """Sub-pixel disparity (float32, pixels) of the left image; <=0 = invalid.

    SGBM's ``disp12MaxDiff`` performs an internal left-right consistency check
    (occluded / mismatched pixels are returned below ``minDisparity`` and get
    masked to -1 here), the standard defence against textureless regions
    producing confident nonsense. When ``opencv-contrib`` is present we also run
    a dedicated right matcher for a stricter agreement test.
    """
    cfg = cfg or StereoConfig()
    ch = 1 if left.ndim == 2 else left.shape[2]
    matcher = cfg.build(ch)

    disp_l = matcher.compute(left, right).astype(np.float32) / 16.0

    if hasattr(cv2, "ximgproc"):
        right_matcher = cv2.ximgproc.createRightMatcher(matcher)
        disp_r = right_matcher.compute(right, left).astype(np.float32) / 16.0
        keep = _lr_consistent(disp_l, disp_r, cfg.lr_consistency_px)
        disp_l[~keep] = -1.0

    disp_l[disp_l < cfg.min_disparity] = -1.0
    return disp_l


def _lr_consistent(disp_l: np.ndarray, disp_r: np.ndarray, tol: float) -> np.ndarray:
    """Mask where left disparity agrees with the right map it points to.

    ``disp_r`` from OpenCV's right matcher is stored as a negative disparity in
    the left-image layout, so the matching right pixel value is ``-disp_r``.
    """
    h, w = disp_l.shape
    valid = disp_l > 0
    xs = np.arange(w)[None, :].repeat(h, axis=0)
    xr = np.rint(xs - disp_l).astype(np.int64)  # matching right pixel column
    inb = (xr >= 0) & (xr < w) & valid
    ys = np.arange(h)[:, None].repeat(w, axis=1)
    dr = np.zeros_like(disp_l)
    dr[inb] = -disp_r[ys[inb], xr[inb]]
    agree = np.abs(disp_l - dr) <= tol
    return valid & inb & agree


def disparity_to_depth(disp: np.ndarray, fx: float, baseline_m: float,
                       max_depth_m: float = 12.0) -> np.ndarray:
    """Convert a disparity map to a metric depth map (metres); 0 = no data."""
    depth = np.zeros_like(disp, dtype=np.float32)
    good = disp > 0
    depth[good] = (fx * baseline_m) / disp[good]
    depth[(depth <= 0) | (depth > max_depth_m)] = 0.0
    return depth


def stereo_depth(left: np.ndarray, right: np.ndarray, K: Intrinsics,
                 baseline_m: float, cfg: StereoConfig | None = None) -> np.ndarray:
    """End-to-end: rectified L/R pair + calibration -> metric depth map."""
    cfg = cfg or StereoConfig()
    disp = compute_disparity(left, right, cfg)
    return disparity_to_depth(disp, K.fx, baseline_m, cfg.max_depth_m)
