"""Mesh (TSDF fusion + numpy rasterizer) tests."""

import math

import numpy as np
import pytest
import torch
from torch import nn

from splatvid.mesh import MeshData, render_mesh


def _opaque_model_from_scene(n=1200, seed=7):
    """A GaussianModel whose params match a synthetic opaque scene."""
    from splatvid.model import GaussianModel
    from splatvid.synthetic import make_scene

    scene = make_scene(n=n, seed=seed)
    xyz = scene["xyz"].numpy()
    rgb = scene["rgb"].numpy()
    m = GaussianModel(xyz, rgb, device="cpu")
    to = lambda a: nn.Parameter(torch.tensor(np.asarray(a), dtype=torch.float32))  # noqa: E731
    m.xyz = to(xyz)
    m.log_scale = to(np.log(scene["scale"].numpy()))
    m.quat = to(scene["quat"].numpy())
    m.color = to((np.clip(rgb, 1e-4, 1 - 1e-4) - 0.5) / GaussianModel.SH_C0)
    m.opacity = to(np.full((n, 1), np.log(0.99 / 0.01), np.float32))
    return m, xyz, rgb


def _orbit_reconstruction(xyz, rgb, n_cam=8, w=64, h=48):
    from splatvid.sfm import Reconstruction
    from splatvid.synthetic import orbit_pose

    poses = {}
    for i in range(n_cam):
        R, t = orbit_pose(2 * math.pi * i / n_cam, radius=2.6)
        poses[i] = (R, t)
    return Reconstruction(
        focal=1.1 * w, cx=w / 2, cy=h / 2, width=w, height=h, poses=poses,
        points=xyz.astype(np.float64), point_colors=rgb.astype(np.float64),
        point_errors=np.zeros(len(xyz)), registered=list(range(n_cam)),
    )


def test_render_mesh_zbuffer():
    # A single front-facing triangle: rasterizer fills it with finite depth.
    verts = np.array([[-0.5, -0.5, 2.0], [0.5, -0.5, 2.0], [0.0, 0.5, 2.0]])
    faces = np.array([[0, 1, 2]])
    colors = np.array([[1.0, 0, 0], [0, 1.0, 0], [0, 0, 1.0]])
    mesh = MeshData(verts, faces, colors)
    color, depth = render_mesh(mesh, np.eye(3), np.zeros(3), 60.0, 32, 24, 64, 48)
    assert color.shape == (48, 64, 3) and depth.shape == (48, 64)
    covered = depth > 0
    assert covered.sum() > 20
    assert np.allclose(depth[covered], 2.0, atol=1e-3)  # planar tri at z=2


def test_fuse_tsdf_and_render():
    pytest.importorskip("open3d")
    from splatvid.mesh import fuse_tsdf, mesh_to_data

    model, xyz, rgb = _opaque_model_from_scene()
    rec = _orbit_reconstruction(xyz, rgb)
    o3d_mesh = fuse_tsdf(model, rec, target_faces=4000)
    assert len(o3d_mesh.triangles) > 50
    md = mesh_to_data(o3d_mesh)
    assert md.verts.shape[1] == 3 and md.faces.shape[1] == 3
    # Render it from a registered camera: the object should cover many pixels.
    R, t = rec.poses[0]
    color, depth = render_mesh(md, R, t, rec.focal, rec.cx, rec.cy, rec.width, rec.height)
    assert (depth > 0).sum() > 100
    assert np.isfinite(depth[depth > 0]).all()
