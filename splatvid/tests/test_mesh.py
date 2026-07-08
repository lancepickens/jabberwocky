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


def test_depth_supervision_smoke():
    # train() with mesh depth supervision runs end to end (geometry stage).
    pytest.importorskip("open3d")
    from splatvid.mesh import fuse_tsdf, mesh_to_data
    from splatvid.train import TrainConfig, train

    model0, xyz, rgb = _opaque_model_from_scene(n=800)
    rec = _orbit_reconstruction(xyz, rgb, w=64, h=48)
    mesh = mesh_to_data(fuse_tsdf(model0, rec, target_faces=3000))
    images = [np.random.default_rng(i).integers(0, 255, (48, 64, 3), dtype=np.uint8)
              for i in rec.registered]
    cfg = TrainConfig(iterations=4, train_size=48, densify_from=100,
                      depth_weight=1.0, device="cpu")
    m = train(rec, images, cfg, mesh=mesh)
    assert m.num_gaussians > 0


def test_fuse_from_depth_maps():
    # Fuse from provided depth maps + colour frames (the monocular-depth path).
    pytest.importorskip("open3d")
    from splatvid.mesh import fuse_tsdf, render_depth_color

    model, xyz, rgb = _opaque_model_from_scene(n=800)
    rec = _orbit_reconstruction(xyz, rgb, w=64, h=48)
    dmaps, images = [], []
    for fi in rec.registered:
        R, t = rec.poses[fi]
        d, c = render_depth_color(model, R, t, rec.focal, rec.cx, rec.cy, rec.width, rec.height)
        dmaps.append(d)
        images.append(np.ascontiguousarray(c[:, :, ::-1]))  # RGB -> BGR frame
    mesh = fuse_tsdf(None, rec, depth_maps=dmaps, images=images, target_faces=3000)
    assert len(mesh.triangles) > 50


def test_poisson_mesh_reconstructs_sphere():
    # Screened Poisson on a clean oriented point cloud recovers the surface.
    pytest.importorskip("open3d")
    import open3d as o3d

    from splatvid.mesh import poisson_mesh

    rng = np.random.default_rng(0)
    p = rng.normal(size=(20000, 3))
    p /= np.linalg.norm(p, axis=1, keepdims=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(p)
    pcd.normals = o3d.utility.Vector3dVector(p)  # outward normals = positions
    m = poisson_mesh(pcd, depth=8, density_quantile=0.01, target_faces=None)
    assert len(m.triangles) > 1000
    r = np.linalg.norm(np.asarray(m.vertices), axis=1)
    assert abs(r.mean() - 1.0) < 0.05 and r.std() < 0.05  # a unit sphere


def test_save_mesh_draco(tmp_path):
    import os

    dracopy = pytest.importorskip("DracoPy")
    from splatvid.mesh import MeshData, save_mesh_draco

    rng = np.random.default_rng(0)
    v = rng.uniform(0, 1, (120, 3))
    f = np.array([[i, (i + 1) % 120, (i + 2) % 120] for i in range(80)])
    md = MeshData(v, f, np.ones((120, 3)))
    p = str(tmp_path / "m.drc")
    n = save_mesh_draco(md, p)
    assert n > 0 and os.path.getsize(p) == n
    dec = dracopy.decode(open(p, "rb").read())  # round-trips
    assert len(dec.points) > 0 and len(dec.faces) > 0


def test_mesh_view_prior():
    from splatvid.view_prior import MeshViewPrior, NoopViewPrior

    img = torch.rand(8, 8, 3)
    assert torch.allclose(NoopViewPrior()(img, cam=None), img.detach())
    # A big quad at z=2 fills the frame; MeshViewPrior renders it from the cam.
    quad = MeshData(
        verts=np.array([[-2, -2, 2.0], [2, -2, 2.0], [2, 2, 2.0], [-2, 2, 2.0]]),
        faces=np.array([[0, 1, 2], [0, 2, 3]]),
        vert_colors=np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0.0]]),
    )
    prior = MeshViewPrior(quad, 40.0, 32, 24, 64, 48)
    out = prior(torch.zeros(48, 64, 3), cam=(np.eye(3), np.zeros(3), 40.0, 32, 24))
    assert out.shape == (48, 64, 3)
    assert float(out.abs().sum()) > 0  # the quad was rendered


def test_train_neural_mesh_supervision_smoke():
    pytest.importorskip("open3d")
    from splatvid.mesh import fuse_tsdf, mesh_to_data
    from splatvid.train import TrainConfig, train_neural
    from splatvid.view_prior import MeshViewPrior

    model0, xyz, rgb = _opaque_model_from_scene(n=800)
    rec = _orbit_reconstruction(xyz, rgb, w=64, h=48)
    mesh = mesh_to_data(fuse_tsdf(model0, rec, target_faces=3000))
    images = [np.random.default_rng(i).integers(0, 255, (48, 64, 3), dtype=np.uint8)
              for i in rec.registered]
    ts = 48
    s = min(1.0, ts / max(rec.width, rec.height))
    prior = MeshViewPrior(
        mesh, rec.focal * s, rec.cx * s, rec.cy * s,
        round(rec.width * s), round(rec.height * s),
    )
    cfg = TrainConfig(
        iterations=2, neural_iters=3, train_size=ts, feature_dim=8,
        densify_from=100, holdout_every=2, log_every=3, perceptual_weight=0.0,
        temporal_weight=0.0, pseudo_weight=0.5, depth_weight=0.5, device="cpu",
    )
    m, sh = train_neural(rec, images, cfg, view_prior=prior, mesh=mesh)
    assert m.num_gaussians > 0
