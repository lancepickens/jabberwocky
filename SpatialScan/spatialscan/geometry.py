"""Core camera geometry: intrinsics, back-projection, poses, and PLY I/O.

Conventions (shared with the ``splatvid`` sibling project):

* World-to-camera transform is OpenCV style: ``x_cam = R @ x_world + t``.
* Cameras look down **+z** in their own frame; a point is in front of the
  camera when its camera-space z is positive.
* Pinhole intrinsics ``K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]`` in pixels.
* The *left* eye of a stereo pair is the reference camera; depth and every
  back-projected point are expressed in the left camera's frame.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Intrinsics:
    """Pinhole intrinsics for a single (rectified) view, in pixels."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    @classmethod
    def from_fov(cls, width: int, height: int, hfov_deg: float) -> "Intrinsics":
        """Build square-pixel intrinsics from a horizontal field of view.

        Apple stores a horizontal FOV per eye in the container metadata; a
        centered principal point and equal fx/fy are the right default for the
        rectified views we feed the stereo matcher.
        """
        fx = 0.5 * width / np.tan(0.5 * np.radians(hfov_deg))
        return cls(fx=fx, fy=fx, cx=(width - 1) / 2.0, cy=(height - 1) / 2.0,
                   width=int(width), height=int(height))

    @property
    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0.0, self.cx],
                         [0.0, self.fy, self.cy],
                         [0.0, 0.0, 1.0]], dtype=np.float64)

    def scaled(self, sx: float, sy: float) -> "Intrinsics":
        """Intrinsics for an image resized by (sx, sy) in (width, height)."""
        return Intrinsics(self.fx * sx, self.fy * sy,
                          (self.cx + 0.5) * sx - 0.5, (self.cy + 0.5) * sy - 0.5,
                          int(round(self.width * sx)), int(round(self.height * sy)))


def backproject(depth: np.ndarray, K: Intrinsics) -> np.ndarray:
    """Lift a depth map to a camera-frame point grid ``(H, W, 3)``.

    Pixels with non-positive (missing) depth become NaN so callers can mask
    them out before fusing.
    """
    h, w = depth.shape
    u, v = np.meshgrid(np.arange(w, dtype=np.float64),
                       np.arange(h, dtype=np.float64))
    z = depth.astype(np.float64)
    x = (u - K.cx) * z / K.fx
    y = (v - K.cy) * z / K.fy
    pts = np.stack([x, y, z], axis=-1)
    pts[z <= 0] = np.nan
    return pts


def transform_points(pts: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Apply ``x' = R @ x + t`` to an ``(..., 3)`` array of points."""
    flat = pts.reshape(-1, 3)
    out = flat @ np.asarray(R, dtype=np.float64).T + np.asarray(t, dtype=np.float64)
    return out.reshape(pts.shape)


def camera_to_world(pts_cam: np.ndarray, R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Inverse of the world->camera transform: ``x_world = R^T (x_cam - t)``."""
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64)
    flat = pts_cam.reshape(-1, 3)
    out = (flat - t) @ R
    return out.reshape(pts_cam.shape)


def pose_to_extrinsic(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Pack ``(R, t)`` into a 4x4 world-to-camera matrix."""
    Rt = np.eye(4, dtype=np.float64)
    Rt[:3, :3] = R
    Rt[:3, 3] = t
    return Rt


def extrinsic_to_pose(Rt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split a 4x4 world-to-camera matrix into ``(R, t)``."""
    Rt = np.asarray(Rt, dtype=np.float64)
    return Rt[:3, :3].copy(), Rt[:3, 3].copy()


def save_ply_pointcloud(path: str, points: np.ndarray,
                        colors: np.ndarray | None = None) -> int:
    """Write a colored point cloud as a binary little-endian PLY.

    ``points`` is ``(N, 3)`` float; ``colors`` is ``(N, 3)`` in ``[0, 1]`` or
    ``uint8``. NaN/inf points are dropped. Returns the number written. This is
    the fallback surface product when Open3D (the mesher) is unavailable.
    """
    pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    finite = np.isfinite(pts).all(axis=1)
    pts = pts[finite]
    if colors is None:
        rgb = np.full((pts.shape[0], 3), 200, dtype=np.uint8)
    else:
        col = np.asarray(colors).reshape(-1, 3)[finite]
        if col.dtype != np.uint8:
            col = np.clip(col * 255.0, 0, 255).astype(np.uint8)
        rgb = col
    n = pts.shape[0]
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    with open(path, "wb") as f:
        f.write(header)
        buf = bytearray()
        for (x, y, z), (r, g, b) in zip(pts.astype(np.float32), rgb):
            buf += struct.pack("<fffBBB", x, y, z, int(r), int(g), int(b))
        f.write(buf)
    return n
