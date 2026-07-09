"""SpatialScan: a scene mesh from an Apple spatial video (MV-HEVC stereo).

Pipeline: parse the container's stereo geometry, stereo-match each MV-HEVC
frame pair to *metric* depth, register frames with RGB-D odometry, and fuse
them into a triangle mesh (TSDF).

Quick start::

    from spatialscan.spatial import SpatialVideo
    from spatialscan.scene import build_scene_mesh

    video = SpatialVideo.open("clip.mov")
    result = build_scene_mesh(video, "scene.ply")
"""

from __future__ import annotations

from .geometry import Intrinsics
from .quicktime import SpatialMetadata, extract_spatial_metadata
from .spatial import SpatialVideo, StereoFrame
from .stereo import StereoConfig, stereo_depth
from .fusion import FusionConfig, FusionResult, fuse
from .scene import SceneResult, build_scene_mesh

__version__ = "0.1.0"

__all__ = [
    "Intrinsics",
    "SpatialMetadata",
    "extract_spatial_metadata",
    "SpatialVideo",
    "StereoFrame",
    "StereoConfig",
    "stereo_depth",
    "FusionConfig",
    "FusionResult",
    "fuse",
    "SceneResult",
    "build_scene_mesh",
]
