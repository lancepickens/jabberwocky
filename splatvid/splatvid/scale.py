"""Metric-scale anchoring: turn the up-to-scale reconstruction into real units.

Monocular SfM recovers geometry only up to an unknown global scale. To pin it
to metres we measure a known object's extent in reconstruction units and divide
its real-world dimension by it, then scale the whole reconstruction (points and
camera translations) and mesh by that factor.

The reference object is identified visually and its real dimension looked up on
the web (e.g. the YETI Hopper Flip cooler in IMG_6547 is ~0.30 m tall). Its
extent in the reconstruction is measured by reprojecting the sparse SfM points,
keeping those that land in the object's image bbox and near depth-cluster
(rejecting background), and spanning them along the chosen axis.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .geometry import project_points


@dataclass
class ScaleSpec:
    frame_index: int  # a registered frame where the object is visible
    bbox: tuple[float, float, float, float]  # x0, y0, x1, y1 in pixels
    real_dim_m: float  # real-world extent of the object along `axis`, in metres
    object_name: str = ""
    axis: str = "up"  # "up" (gravity/ring-normal), "x" | "y" | "z", or "diag"


def up_direction(rec) -> np.ndarray:
    """Approximate world-space vertical: the normal of the camera-orbit plane.

    Cameras orbiting an object sweep a roughly horizontal ring, so the ring's
    plane normal is a good gravity/up estimate — the axis the object's *height*
    lies along.
    """
    C = rec.camera_centers()
    d = C - C.mean(axis=0)
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    normal = vt[2]
    # Sign it to agree with the cameras' up (world dir of camera -y).
    cam_up = np.mean(
        [rec.poses[i][0].T @ np.array([0.0, -1.0, 0.0]) for i in rec.registered],
        axis=0,
    )
    return normal if np.dot(normal, cam_up) >= 0 else -normal


def measure_object_extent(
    rec, frame_index: int, bbox, *, axis: str = "up", depth_keep_frac: float = 0.6
) -> tuple[float, np.ndarray]:
    """Reconstruction-unit extent of an object + the point indices used.

    Selects sparse points whose reprojection into ``frame_index`` lands in
    ``bbox`` and in the nearest ``depth_keep_frac`` depth cluster (drops the
    background wall behind the object), then measures their span along ``axis``.
    """
    R, t = rec.poses[frame_index]
    uv, z = project_points(rec.points, R, t, rec.K)
    x0, y0, x1, y1 = bbox
    in_box = (
        (z > 1e-6)
        & (uv[:, 0] >= x0) & (uv[:, 0] <= x1)
        & (uv[:, 1] >= y0) & (uv[:, 1] <= y1)
    )
    idx = np.nonzero(in_box)[0]
    if idx.size < 4:
        raise ValueError(f"only {idx.size} points reproject into the object bbox")
    # Background rejection: keep the nearest fraction by camera depth.
    order = np.argsort(z[idx])
    keep_n = max(4, int(round(idx.size * depth_keep_frac)))
    idx = idx[order[:keep_n]]
    pts = rec.points[idx]

    if axis == "up":
        u = up_direction(rec)
        proj = pts @ u
        extent = float(proj.max() - proj.min())
    elif axis in ("x", "y", "z"):
        k = {"x": 0, "y": 1, "z": 2}[axis]
        extent = float(pts[:, k].max() - pts[:, k].min())
    elif axis == "diag":  # largest principal-axis span
        d = pts - pts.mean(0)
        _, _, vt = np.linalg.svd(d, full_matrices=False)
        proj = d @ vt[0]
        extent = float(proj.max() - proj.min())
    else:
        raise ValueError(f"unknown axis {axis!r}")
    if extent <= 1e-9:
        raise ValueError("object extent is ~0; check the bbox / axis")
    return extent, idx


def compute_scale(real_dim_m: float, recon_extent: float) -> float:
    """Metres-per-reconstruction-unit."""
    return float(real_dim_m) / float(recon_extent)


def scale_from_spec(rec, spec: ScaleSpec) -> float:
    extent, _ = measure_object_extent(rec, spec.frame_index, spec.bbox, axis=spec.axis)
    return compute_scale(spec.real_dim_m, extent)


def apply_scale(rec, s: float):
    """New Reconstruction scaled by ``s`` (points·s, poses' t·s; R/K unchanged).

    Reprojection is invariant: u = f·(s·x)/(s·z)+cx is unchanged, so the scaled
    reconstruction is geometrically identical, just in metric units.
    """
    from .sfm import Reconstruction

    poses = {i: (R, t * s) for i, (R, t) in rec.poses.items()}
    return Reconstruction(
        focal=rec.focal, cx=rec.cx, cy=rec.cy, width=rec.width, height=rec.height,
        poses=poses, points=rec.points * s, point_colors=rec.point_colors,
        point_errors=rec.point_errors, registered=list(rec.registered),
    )


def apply_scale_mesh(mesh, s: float):
    """Scale a numpy ``MeshData``'s vertices by ``s`` (colors/faces unchanged)."""
    from .mesh import MeshData

    return MeshData(verts=mesh.verts * s, faces=mesh.faces, vert_colors=mesh.vert_colors)
