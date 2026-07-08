"""Fuse posed RGB-D frames into a single scene mesh (TSDF) or point cloud.

Every frame contributes a metric depth + colour image and a pose. We integrate
them into a truncated signed-distance volume (Open3D's ``ScalableTSDFVolume``,
a voxel-hashed volume that only allocates near surfaces) and march cubes to a
watertight-ish triangle mesh, then drop tiny disconnected components and,
optionally, decimate.

Open3D is an optional dependency. Without it, :func:`fuse` falls back to a
fused, voxel-downsampled **colored point cloud** so the pipeline still yields a
usable 3D product — just points instead of triangles.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import cv2
import numpy as np

from .geometry import Intrinsics, backproject, camera_to_world, save_ply_pointcloud

log = logging.getLogger("spatialscan")


@dataclass
class FusionConfig:
    voxel_size_m: float = 0.01      # TSDF voxel edge (1 cm)
    sdf_trunc_m: float = 0.04       # truncation band (~4 voxels)
    depth_trunc_m: float = 12.0     # ignore depth beyond this
    min_component_frac: float = 0.02  # drop mesh islands < this fraction of faces
    target_faces: int = 0           # 0 = no decimation
    smooth_iters: int = 0           # Taubin smoothing passes


@dataclass
class FusionResult:
    kind: str                       # "mesh" or "pointcloud"
    n_vertices: int
    n_faces: int
    path: str | None = None


def _o3d():
    try:
        import open3d as o3d
        return o3d
    except Exception:  # pragma: no cover - optional dep
        return None


def fuse_tsdf(colors, depths, extrinsics, K: Intrinsics, cfg: FusionConfig):
    """Integrate frames into a TSDF volume and extract an Open3D mesh."""
    o3d = _o3d()
    if o3d is None:
        raise RuntimeError("open3d is required for TSDF meshing")
    from .odometry import _o3d_intrinsic, _to_o3d_rgbd

    volume = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=cfg.voxel_size_m,
        sdf_trunc=cfg.sdf_trunc_m,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8)
    intr = _o3d_intrinsic(o3d, K)
    for color, depth, extr in zip(colors, depths, extrinsics):
        rgbd = _to_o3d_rgbd(o3d, color, depth, cfg.depth_trunc_m)
        volume.integrate(rgbd, intr, extr)

    mesh = volume.extract_triangle_mesh()
    mesh.compute_vertex_normals()
    mesh = _clean_mesh(o3d, mesh, cfg)
    return mesh


def _clean_mesh(o3d, mesh, cfg: FusionConfig):
    if cfg.min_component_frac > 0 and len(mesh.triangles) > 0:
        idx, counts, _ = mesh.cluster_connected_triangles()
        idx = np.asarray(idx)
        counts = np.asarray(counts)
        if counts.size:
            keep_min = cfg.min_component_frac * counts.max()
            small = np.where(counts < keep_min)[0]
            remove = np.isin(idx, small)
            mesh.remove_triangles_by_mask(remove)
            mesh.remove_unreferenced_vertices()
    if cfg.target_faces and len(mesh.triangles) > cfg.target_faces:
        mesh = mesh.simplify_quadric_decimation(cfg.target_faces)
    if cfg.smooth_iters:
        mesh = mesh.filter_smooth_taubin(number_of_iterations=cfg.smooth_iters)
    mesh.compute_vertex_normals()
    return mesh


def fuse_pointcloud(colors, depths, cam_to_world, K: Intrinsics,
                    cfg: FusionConfig) -> tuple[np.ndarray, np.ndarray]:
    """Backproject every frame to world points and voxel-downsample (numpy)."""
    all_pts, all_rgb = [], []
    for color, depth, T in zip(colors, depths, cam_to_world):
        pts_cam = backproject(depth, K).reshape(-1, 3)
        rgb = cv2.cvtColor(color, cv2.COLOR_BGR2RGB).reshape(-1, 3)
        good = np.isfinite(pts_cam).all(axis=1)
        pts_cam, rgb = pts_cam[good], rgb[good]
        R, t = T[:3, :3], T[:3, 3]
        # cam_to_world maps camera points into world directly.
        pts_w = pts_cam @ R.T + t
        all_pts.append(pts_w)
        all_rgb.append(rgb)
    pts = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3))
    rgb = np.concatenate(all_rgb, axis=0) if all_rgb else np.zeros((0, 3), np.uint8)
    return _voxel_downsample(pts, rgb, cfg.voxel_size_m)


def _voxel_downsample(pts: np.ndarray, rgb: np.ndarray, voxel: float):
    if pts.shape[0] == 0:
        return pts, rgb
    keys = np.floor(pts / voxel).astype(np.int64)
    _, order = np.unique(keys, axis=0, return_index=True)
    order.sort()
    return pts[order], rgb[order]


def fuse(colors, depths, cam_to_world, K: Intrinsics, cfg: FusionConfig | None = None,
         out_path: str | None = None) -> FusionResult:
    """Fuse frames to a mesh (Open3D) or a point cloud (fallback) and save."""
    cfg = cfg or FusionConfig()
    o3d = _o3d()
    if o3d is not None:
        from .odometry import extrinsics_from_trajectory
        extr = extrinsics_from_trajectory(cam_to_world)
        mesh = fuse_tsdf(colors, depths, extr, K, cfg)
        nv, nf = len(mesh.vertices), len(mesh.triangles)
        if out_path:
            o3d.io.write_triangle_mesh(out_path, mesh)
        log.info("TSDF mesh: %d vertices, %d faces", nv, nf)
        return FusionResult("mesh", nv, nf, out_path)

    log.warning("open3d unavailable: exporting a fused point cloud instead of a mesh")
    pts, rgb = fuse_pointcloud(colors, depths, cam_to_world, K, cfg)
    n = pts.shape[0]
    if out_path:
        if not out_path.lower().endswith(".ply"):
            out_path = out_path.rsplit(".", 1)[0] + ".ply"
        save_ply_pointcloud(out_path, pts, rgb)
    log.info("Point cloud: %d points", n)
    return FusionResult("pointcloud", n, 0, out_path)
