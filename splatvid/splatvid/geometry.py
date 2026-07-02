"""Core 3D geometry helpers shared by the SfM and splatting stages.

Conventions used throughout splatvid:

* World-to-camera transforms: ``x_cam = R @ x_world + t`` (OpenCV style).
* Cameras look down +z in their own frame; a point is in front of the
  camera when its camera-space z is positive.
* Intrinsics ``K = [[f, 0, cx], [0, f, cy], [0, 0, 1]]`` with a single
  shared focal length in pixels.
* Quaternions are ``(w, x, y, z)`` and normalized.
"""

from __future__ import annotations

import numpy as np


# ---------------------------------------------------------------------------
# Rotations
# ---------------------------------------------------------------------------

def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Convert quaternion(s) (w, x, y, z) to rotation matrix/matrices.

    Accepts shape (4,) -> (3, 3) or (N, 4) -> (N, 3, 3).
    """
    q = np.asarray(q, dtype=np.float64)
    single = q.ndim == 1
    if single:
        q = q[None]
    q = q / np.linalg.norm(q, axis=-1, keepdims=True)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    R = np.empty((q.shape[0], 3, 3), dtype=np.float64)
    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - w * z)
    R[:, 0, 2] = 2 * (x * z + w * y)
    R[:, 1, 0] = 2 * (x * y + w * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - w * x)
    R[:, 2, 0] = 2 * (x * z - w * y)
    R[:, 2, 1] = 2 * (y * z + w * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R[0] if single else R


def rotmat_to_quat(R: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a quaternion (w, x, y, z)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z])
    if q[0] < 0:
        q = -q
    return q / np.linalg.norm(q)


def rodrigues_to_rotmat(rvec: np.ndarray) -> np.ndarray:
    """Axis-angle vector(s) -> rotation matrix/matrices ((3,) or (N, 3))."""
    rvec = np.asarray(rvec, dtype=np.float64)
    single = rvec.ndim == 1
    if single:
        rvec = rvec[None]
    theta = np.linalg.norm(rvec, axis=-1, keepdims=True)
    small = theta[:, 0] < 1e-12
    axis = np.where(theta > 1e-12, rvec / np.maximum(theta, 1e-12), 0.0)
    kx, ky, kz = axis[:, 0], axis[:, 1], axis[:, 2]
    zero = np.zeros_like(kx)
    K = np.stack(
        [zero, -kz, ky, kz, zero, -kx, -ky, kx, zero], axis=-1
    ).reshape(-1, 3, 3)
    th = theta[:, :, None]
    R = np.eye(3)[None] + np.sin(th) * K + (1 - np.cos(th)) * (K @ K)
    R[small] = np.eye(3)
    return R[0] if single else R


def rotmat_to_rodrigues(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> axis-angle vector."""
    q = rotmat_to_quat(R)
    w = np.clip(q[0], -1.0, 1.0)
    theta = 2.0 * np.arccos(w)
    s = np.sqrt(max(1.0 - w * w, 0.0))
    if s < 1e-12:
        return np.zeros(3)
    return theta * q[1:] / s


# ---------------------------------------------------------------------------
# Projection / triangulation
# ---------------------------------------------------------------------------

def make_K(focal: float, cx: float, cy: float) -> np.ndarray:
    return np.array([[focal, 0, cx], [0, focal, cy], [0, 0, 1]], dtype=np.float64)


def project_points(
    points: np.ndarray, R: np.ndarray, t: np.ndarray, K: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Project world points (N, 3) into pixels.

    Returns (uv (N, 2), z (N,)) where z is camera-space depth.
    """
    pc = points @ R.T + t[None]
    z = pc[:, 2]
    zs = np.where(np.abs(z) < 1e-12, 1e-12, z)
    u = K[0, 0] * pc[:, 0] / zs + K[0, 2]
    v = K[1, 1] * pc[:, 1] / zs + K[1, 2]
    return np.stack([u, v], axis=-1), z


def triangulate_point(
    obs: list[tuple[np.ndarray, np.ndarray, np.ndarray]], K: np.ndarray
) -> np.ndarray | None:
    """DLT triangulation of one point from >= 2 observations.

    ``obs`` is a list of (uv, R, t). Returns the world point or None if the
    system is degenerate.
    """
    A = []
    for uv, R, t in obs:
        P = K @ np.hstack([R, t.reshape(3, 1)])
        u, v = uv
        A.append(u * P[2] - P[0])
        A.append(v * P[2] - P[1])
    A = np.asarray(A)
    _, s, vt = np.linalg.svd(A)
    if s[-2] < 1e-12:
        return None
    X = vt[-1]
    if abs(X[3]) < 1e-12:
        return None
    return X[:3] / X[3]


def triangulation_angle_deg(
    X: np.ndarray, C1: np.ndarray, C2: np.ndarray
) -> float:
    """Angle at point X subtended by camera centers C1 and C2, in degrees."""
    d1 = C1 - X
    d2 = C2 - X
    n1 = np.linalg.norm(d1)
    n2 = np.linalg.norm(d2)
    if n1 < 1e-12 or n2 < 1e-12:
        return 0.0
    c = np.clip(np.dot(d1, d2) / (n1 * n2), -1.0, 1.0)
    return float(np.degrees(np.arccos(c)))


def camera_center(R: np.ndarray, t: np.ndarray) -> np.ndarray:
    """World-space camera center for x_cam = R x_world + t."""
    return -R.T @ t
