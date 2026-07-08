"""Frame-to-frame RGB-D visual odometry -> a global camera trajectory.

A spatial video pans around a scene, so before we can fuse per-frame depth into
one surface we need each frame's pose. Because stereo already gives us *metric*
depth, we can register consecutive RGB-D frames directly (no scale drift from
monocular tracking): Open3D's hybrid photometric+geometric odometry aligns
frame *i* to frame *i-1*, and we chain the relative motions into world poses.

Poses are expressed two ways, both indexed by frame:

* ``cam_to_world`` — 4x4, places camera points into the world (== first frame).
* ``extrinsics``   — 4x4 world->camera, what the TSDF integrator wants.

Without Open3D installed the trajectory degrades to identity (a static-camera
assumption); the caller is warned and the point-cloud fallback still runs.
"""

from __future__ import annotations

import logging

import numpy as np

from .geometry import Intrinsics

log = logging.getLogger("spatialscan")


def _o3d():
    try:
        import open3d as o3d
        return o3d
    except Exception:  # pragma: no cover - optional dep
        return None


def _to_o3d_rgbd(o3d, color_bgr: np.ndarray, depth_m: np.ndarray, depth_trunc: float):
    import cv2
    rgb = cv2.cvtColor(color_bgr, cv2.COLOR_BGR2RGB)
    color = o3d.geometry.Image(np.ascontiguousarray(rgb))
    depth = o3d.geometry.Image(np.ascontiguousarray(depth_m.astype(np.float32)))
    return o3d.geometry.RGBDImage.create_from_color_and_depth(
        color, depth, depth_scale=1.0, depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False)


def _o3d_intrinsic(o3d, K: Intrinsics):
    return o3d.camera.PinholeCameraIntrinsic(K.width, K.height, K.fx, K.fy, K.cx, K.cy)


def estimate_trajectory(colors: list[np.ndarray], depths: list[np.ndarray],
                        K: Intrinsics, *, depth_trunc: float = 12.0
                        ) -> list[np.ndarray]:
    """Camera-to-world 4x4 poses, one per frame; frame 0 anchors the world.

    Each step aligns frame *i* to *i-1* with hybrid RGB-D odometry; a failed
    step falls back to constant velocity (reuse the previous relative motion),
    which keeps the trajectory continuous through a hard-to-track frame.
    """
    n = len(colors)
    cam_to_world = [np.eye(4) for _ in range(n)]
    o3d = _o3d()
    if o3d is None or n < 2:
        if o3d is None:
            log.warning("open3d unavailable: assuming a static camera (identity poses)")
        return cam_to_world

    intr = _o3d_intrinsic(o3d, K)
    option = o3d.pipelines.odometry.OdometryOption()
    prev_rel = np.eye(4)
    for i in range(1, n):
        src = _to_o3d_rgbd(o3d, colors[i], depths[i], depth_trunc)
        tgt = _to_o3d_rgbd(o3d, colors[i - 1], depths[i - 1], depth_trunc)
        ok, trans, _ = o3d.pipelines.odometry.compute_rgbd_odometry(
            src, tgt, intr, np.eye(4),
            o3d.pipelines.odometry.RGBDOdometryJacobianFromHybridTerm(), option)
        if not ok or not np.all(np.isfinite(trans)):
            log.debug("odometry failed at frame %d; using constant velocity", i)
            trans = prev_rel
        else:
            prev_rel = trans
        # ``trans`` maps frame-i camera points into frame-(i-1) camera coords.
        cam_to_world[i] = cam_to_world[i - 1] @ trans
    return cam_to_world


def extrinsics_from_trajectory(cam_to_world: list[np.ndarray]) -> list[np.ndarray]:
    """World->camera matrices (what TSDF integration expects)."""
    return [np.linalg.inv(T) for T in cam_to_world]
