"""High-level pipeline: spatial video path -> scene mesh on disk.

Ties the stages together::

    spatial video ─▶ stereo depth per frame ─▶ RGB-D odometry ─▶ TSDF fusion ─▶ mesh

Use :func:`build_scene_mesh` for the whole thing, or drive :class:`SpatialVideo`
and the stage functions directly for finer control.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import numpy as np

from .fusion import FusionConfig, FusionResult, fuse
from .geometry import Intrinsics
from .odometry import estimate_trajectory
from .spatial import SpatialVideo
from .stereo import StereoConfig, stereo_depth

log = logging.getLogger("spatialscan")


@dataclass
class SceneResult:
    fusion: FusionResult
    intrinsics: Intrinsics
    baseline_m: float
    n_frames: int
    cam_to_world: list[np.ndarray] = field(default_factory=list)
    depths: list[np.ndarray] = field(default_factory=list)
    seconds: float = 0.0


def compute_depths(video: SpatialVideo, stereo_cfg: StereoConfig | None = None
                   ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-frame (color BGR, metric depth) for every stereo pair."""
    stereo_cfg = stereo_cfg or StereoConfig()
    colors, depths = [], []
    for frame in video:
        depth = stereo_depth(frame.left, frame.right, video.intrinsics,
                             video.baseline_m, stereo_cfg)
        colors.append(frame.left)
        depths.append(depth)
        cov = float((depth > 0).mean())
        log.debug("frame %d: depth coverage %.1f%%", frame.index, 100 * cov)
    return colors, depths


def build_scene_mesh(video: SpatialVideo, out_path: str, *,
                     stereo_cfg: StereoConfig | None = None,
                     fusion_cfg: FusionConfig | None = None) -> SceneResult:
    """Run the full pipeline on an opened :class:`SpatialVideo`."""
    t0 = time.time()
    log.info("[1/3] Stereo depth for %d frames", len(video))
    colors, depths = compute_depths(video, stereo_cfg)

    log.info("[2/3] RGB-D odometry")
    cam_to_world = estimate_trajectory(
        colors, depths, video.intrinsics,
        depth_trunc=(fusion_cfg or FusionConfig()).depth_trunc_m)

    log.info("[3/3] Fusing scene mesh -> %s", out_path)
    result = fuse(colors, depths, cam_to_world, video.intrinsics, fusion_cfg, out_path)

    return SceneResult(
        fusion=result, intrinsics=video.intrinsics, baseline_m=video.baseline_m,
        n_frames=len(video), cam_to_world=cam_to_world, depths=depths,
        seconds=time.time() - t0)
