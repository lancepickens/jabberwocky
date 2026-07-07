"""Mesh from photogrammetry: TSDF-fuse splat-rendered depth into a surface.

The trained gaussians are a dense, renderable proxy of the scene, so we get a
mesh by rendering a depth + colour map from every recovered camera and fusing
them in a truncated signed-distance volume (Open3D), then marching-cubes.

Open3D is an optional dependency (``pip install 'splatvid[mesh]'``) imported
lazily inside the fusion / IO functions. ``MeshData`` and ``render_mesh`` are
pure NumPy, so metric scaling, depth supervision, and rendering the mesh from
arbitrary cameras all work without Open3D installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .render import render_model


def render_depth_color(
    model,
    R: np.ndarray,
    t: np.ndarray,
    focal: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    *,
    bg=None,
) -> tuple[np.ndarray, np.ndarray]:
    """Render one camera to ``(depth (H,W) float32, color (H,W,3) uint8 RGB)``.

    ``depth`` is camera-space z of the front surface, de-biased by dividing the
    opacity-weighted depth by the accumulated alpha and zeroed where the splat
    is see-through (``alpha < 0.5``) so downstream TSDF treats those pixels as
    "no measurement" rather than integrating a phantom near-zero surface.
    """
    dev = model.xyz.device
    Rt = torch.as_tensor(R, dtype=torch.float32, device=dev)
    tt = torch.as_tensor(t, dtype=torch.float32, device=dev)
    with torch.no_grad():
        img, info = render_model(
            model, Rt, tt, float(focal), float(cx), float(cy),
            int(width), int(height), bg=bg, return_aux=True,
        )
        alpha = info.alpha
        depth = info.depth / alpha.clamp(min=1e-3)
        depth = torch.where(alpha >= 0.5, depth, torch.zeros_like(depth))
        color = (img.clamp(0.0, 1.0) * 255.0).to(torch.uint8)
    return (
        depth.cpu().numpy().astype(np.float32),
        color.cpu().numpy(),  # (H, W, 3) RGB uint8
    )


def fuse_tsdf(
    model,
    rec,
    *,
    voxel_length: float | None = None,
    sdf_trunc: float | None = None,
    depth_trunc: float | None = None,
    target_faces: int | None = 200_000,
    max_render_dim: int = 480,
    clean: bool = True,
    min_cluster_frac: float = 0.001,  # drop clusters < 0.1% of total triangles
    depth_maps=None,
    images=None,
    bg=None,
):
    """TSDF-fuse depth+colour from every registered camera → Open3D mesh.

    Depth source: the trained splat by default, or — for a much cleaner mesh —
    pass ``depth_maps`` (per-registered-view depth in reconstruction units, e.g.
    aligned monocular depth) together with ``images`` (the real frames, used as
    colour). Monocular depth is dense and smooth and independent of the splat's
    floaters, so the fused surface is markedly better.

    Defaults scale with ``rec.scene_extent()``: voxel = extent/256, sdf_trunc =
    4·voxel (Open3D's canonical ratio), depth_trunc = 3·extent (drops far
    background). The raw marching-cubes mesh is decimated to ``target_faces``
    (quadric decimation) so the pure-NumPy rasterizer stays tractable and the
    mesh is practical for per-view supervision; pass ``target_faces=None`` to
    keep it full. Returns a coloured, normal-computed ``o3d.geometry.TriangleMesh``.
    """
    o3d = _import_o3d()
    extent = rec.scene_extent()
    voxel_length = float(voxel_length or extent / 256.0)
    sdf_trunc = float(sdf_trunc or 4.0 * voxel_length)
    depth_trunc = float(depth_trunc or 3.0 * extent)

    # Render depth/colour at reduced resolution — TSDF integration doesn't need
    # full res, and the pure-Python rasterizer is ~1/s^2 cheaper.
    s = min(1.0, max_render_dim / max(rec.width, rec.height))
    rw, rh = max(1, round(rec.width * s)), max(1, round(rec.height * s))
    rf, rcx, rcy = rec.focal * s, rec.cx * s, rec.cy * s

    vol = o3d.pipelines.integration.ScalableTSDFVolume(
        voxel_length=voxel_length,
        sdf_trunc=sdf_trunc,
        color_type=o3d.pipelines.integration.TSDFVolumeColorType.RGB8,
    )
    intr = o3d.camera.PinholeCameraIntrinsic(rw, rh, rf, rf, rcx, rcy)
    n = 0
    for i, fi in enumerate(rec.registered):
        R, t = rec.poses[fi]
        if depth_maps is not None:
            depth = depth_maps[i].astype(np.float32)
            color = images[fi][:, :, ::-1]  # BGR frame -> RGB
            if (depth.shape[0], depth.shape[1]) != (rh, rw):
                depth = cv2.resize(depth, (rw, rh), interpolation=cv2.INTER_NEAREST)
                color = cv2.resize(color, (rw, rh), interpolation=cv2.INTER_AREA)
            color = np.ascontiguousarray(color.astype(np.uint8))
        else:
            depth, color = render_depth_color(model, R, t, rf, rcx, rcy, rw, rh, bg=bg)
        rgbd = o3d.geometry.RGBDImage.create_from_color_and_depth(
            o3d.geometry.Image(np.ascontiguousarray(color)),
            o3d.geometry.Image(np.ascontiguousarray(depth)),
            depth_scale=1.0,  # depth already in reconstruction units
            depth_trunc=depth_trunc,
            convert_rgb_to_intensity=False,
        )
        extrinsic = np.eye(4)
        extrinsic[:3, :3] = np.asarray(R)
        extrinsic[:3, 3] = np.asarray(t).ravel()
        vol.integrate(rgbd, intr, extrinsic)
        n += 1
    mesh = vol.extract_triangle_mesh()
    if clean:
        # Drop small disconnected floater blobs so the mesh is a cleaner surface
        # than the splat it came from — important because depth-supervising the
        # splat against a *self-derived* mesh is otherwise circular (a mesh that
        # still contains the splat's floaters can't pull them in).
        clusters, n_tri, _ = mesh.cluster_connected_triangles()
        clusters = np.asarray(clusters)
        n_tri = np.asarray(n_tri)
        if n_tri.size:
            # Threshold on a fraction of TOTAL triangles (not the largest
            # cluster) so separate real objects survive and only tiny specks go.
            thresh = max(int(n_tri.sum() * min_cluster_frac), 20)
            keep = n_tri >= thresh
            mesh.remove_triangles_by_mask(~keep[clusters])
    if target_faces is not None and len(mesh.triangles) > target_faces:
        mesh = mesh.simplify_quadric_decimation(int(target_faces))
    mesh.remove_degenerate_triangles()
    mesh.remove_unreferenced_vertices()
    mesh.compute_vertex_normals()
    return mesh


# ---------------------------------------------------------------------------
# Pure-NumPy mesh container + rasterizer (no Open3D)
# ---------------------------------------------------------------------------

@dataclass
class MeshData:
    """A lightweight, Open3D-free triangle mesh."""

    verts: np.ndarray  # (V, 3)
    faces: np.ndarray  # (F, 3) int
    vert_colors: np.ndarray  # (V, 3) float in [0, 1]


def mesh_to_data(o3d_mesh) -> MeshData:
    """Convert an Open3D TriangleMesh to a numpy ``MeshData`` (no I/O)."""
    verts = np.asarray(o3d_mesh.vertices, dtype=np.float64)
    faces = np.asarray(o3d_mesh.triangles, dtype=np.int64)
    cols = np.asarray(o3d_mesh.vertex_colors, dtype=np.float64)
    if cols.shape[0] != verts.shape[0]:
        cols = np.full((verts.shape[0], 3), 0.7)
    return MeshData(verts=verts, faces=faces, vert_colors=cols)


def save_mesh(mesh, path: str) -> None:
    """Write an Open3D mesh (``.ply`` / ``.obj``)."""
    _import_o3d().io.write_triangle_mesh(path, mesh)


def load_mesh(path: str) -> MeshData:
    """Read a mesh file into a numpy ``MeshData``."""
    return mesh_to_data(_import_o3d().io.read_triangle_mesh(path))


def render_mesh(
    mesh: MeshData,
    R: np.ndarray,
    t: np.ndarray,
    focal: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    *,
    bg=(0.0, 0.0, 0.0),
    near: float = 1e-4,
) -> tuple[np.ndarray, np.ndarray]:
    """Rasterize the mesh from a camera → ``(color (H,W,3) [0,1], depth (H,W))``.

    A plain NumPy z-buffer triangle rasterizer with perspective-correct depth and
    barycentric per-vertex colour. No OpenGL, so it runs anywhere (the default
    arbitrary-camera renderer for depth supervision and MeshViewPrior). Depth is
    camera-space z; 0 means "no surface" (background).
    """
    R = np.asarray(R, dtype=np.float64)
    t = np.asarray(t, dtype=np.float64).ravel()
    V = mesh.verts @ R.T + t[None]  # camera space
    z = V[:, 2]
    zc = np.where(z > near, z, 1.0)
    u = focal * V[:, 0] / zc + cx
    v = focal * V[:, 1] / zc + cy
    vc = mesh.vert_colors

    color = np.empty((height, width, 3), np.float32)
    color[:] = np.asarray(bg, np.float32)
    depth = np.zeros((height, width), np.float32)
    zbuf = np.full((height, width), np.inf, np.float64)

    for f in mesh.faces:
        i0, i1, i2 = int(f[0]), int(f[1]), int(f[2])
        if z[i0] <= near or z[i1] <= near or z[i2] <= near:
            continue  # crude near-plane clip (drop triangles behind camera)
        x0, y0 = u[i0], v[i0]
        x1, y1 = u[i1], v[i1]
        x2, y2 = u[i2], v[i2]
        minx = max(int(np.floor(min(x0, x1, x2))), 0)
        maxx = min(int(np.ceil(max(x0, x1, x2))), width - 1)
        miny = max(int(np.floor(min(y0, y1, y2))), 0)
        maxy = min(int(np.ceil(max(y0, y1, y2))), height - 1)
        if minx > maxx or miny > maxy:
            continue
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-12:
            continue
        gx, gy = np.meshgrid(
            np.arange(minx, maxx + 1) + 0.5, np.arange(miny, maxy + 1) + 0.5
        )
        l0 = ((y1 - y2) * (gx - x2) + (x2 - x1) * (gy - y2)) / denom
        l1 = ((y2 - y0) * (gx - x2) + (x0 - x2) * (gy - y2)) / denom
        l2 = 1.0 - l0 - l1
        inside = (l0 >= 0) & (l1 >= 0) & (l2 >= 0)
        if not inside.any():
            continue
        # Perspective-correct depth: interpolate 1/z, invert.
        inv_z = l0 / z[i0] + l1 / z[i1] + l2 / z[i2]
        pz = 1.0 / np.where(inv_z != 0, inv_z, np.inf)
        cc = l0[..., None] * vc[i0] + l1[..., None] * vc[i1] + l2[..., None] * vc[i2]

        zsub = zbuf[miny:maxy + 1, minx:maxx + 1]
        upd = inside & (pz < zsub)
        if not upd.any():
            continue
        zsub[upd] = pz[upd]
        color[miny:maxy + 1, minx:maxx + 1][upd] = cc[upd].astype(np.float32)
        depth[miny:maxy + 1, minx:maxx + 1][upd] = pz[upd].astype(np.float32)
    return color, depth


def _import_o3d():
    try:
        import open3d as o3d  # noqa: PLC0415
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Open3D is required for mesh fusion/IO. Install it with "
            "`uv pip install 'open3d>=0.17'` or `pip install 'splatvid[mesh]'`."
        ) from e
    return o3d
