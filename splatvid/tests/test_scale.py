"""Metric-scale tests: reprojection invariance + object measurement."""

import math

import numpy as np

from splatvid.geometry import project_points
from splatvid.scale import apply_scale, apply_scale_mesh, measure_object_extent


def _bar_reconstruction(n=400, bar_height=1.0, w=128, h=128, n_cam=8, seed=0):
    """Reconstruction whose points form a vertical bar of known height."""
    from splatvid.sfm import Reconstruction
    from splatvid.synthetic import orbit_pose

    rng = np.random.default_rng(seed)
    y = rng.uniform(-bar_height / 2, bar_height / 2, n)
    x = rng.uniform(-0.08, 0.08, n)
    z = rng.uniform(-0.08, 0.08, n)
    pts = np.stack([x, y, z], axis=1)
    poses = {i: orbit_pose(2 * math.pi * i / n_cam, radius=2.6, height=0.0)
             for i in range(n_cam)}
    return Reconstruction(
        focal=1.2 * w, cx=w / 2, cy=h / 2, width=w, height=h, poses=poses,
        points=pts, point_colors=np.full((n, 3), 0.5),
        point_errors=np.zeros(n), registered=list(range(n_cam)),
    )


def test_apply_scale_reprojection_invariant():
    rec = _bar_reconstruction()
    s = 2.7
    rec2 = apply_scale(rec, s)
    for fi in rec.registered:
        R, t = rec.poses[fi]
        R2, t2 = rec2.poses[fi]
        uv1, z1 = project_points(rec.points, R, t, rec.K)
        uv2, z2 = project_points(rec2.points, R2, t2, rec2.K)
        assert np.allclose(uv1, uv2, atol=1e-9)   # pixels unchanged
        assert np.allclose(z2, z1 * s, atol=1e-9)  # depths scale by s


def test_measure_object_extent_up():
    # A 1.0-tall bar measured along the ring-normal ("up") axis recovers ~1.0.
    rec = _bar_reconstruction(bar_height=1.0)
    ext, idx = measure_object_extent(
        rec, 0, (0, 0, rec.width, rec.height), axis="up", depth_keep_frac=1.0
    )
    assert abs(ext - 1.0) < 0.12
    assert idx.size > 50


def test_apply_scale_mesh():
    from splatvid.mesh import MeshData

    m = MeshData(
        verts=np.array([[0.0, 0, 0], [1.0, 0, 0], [0, 2.0, 0]]),
        faces=np.array([[0, 1, 2]]),
        vert_colors=np.ones((3, 3)),
    )
    m2 = apply_scale_mesh(m, 3.0)
    assert np.allclose(m2.verts, m.verts * 3.0)
    assert np.array_equal(m2.faces, m.faces)
