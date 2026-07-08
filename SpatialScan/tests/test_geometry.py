"""Camera geometry: intrinsics, back-projection round-trips, PLY I/O."""

import numpy as np

from spatialscan.geometry import (
    Intrinsics, backproject, camera_to_world, save_ply_pointcloud, transform_points,
)


def test_intrinsics_from_fov():
    K = Intrinsics.from_fov(200, 100, 90.0)
    # 90 deg hfov on 200px => fx = 100.
    assert abs(K.fx - 100.0) < 1e-6
    assert abs(K.cx - 99.5) < 1e-6


def test_backproject_recovers_depth():
    K = Intrinsics.from_fov(64, 48, 60.0)
    depth = np.full((48, 64), 2.0, np.float32)
    pts = backproject(depth, K)
    assert np.allclose(pts[..., 2], 2.0)
    # Principal-point pixel maps to (0, 0, Z).
    cy, cx = int(round(K.cy)), int(round(K.cx))
    assert abs(pts[cy, cx, 0]) < 0.05 and abs(pts[cy, cx, 1]) < 0.05


def test_backproject_masks_missing():
    K = Intrinsics.from_fov(8, 8, 60.0)
    depth = np.zeros((8, 8), np.float32)
    depth[4, 4] = 1.0
    pts = backproject(depth, K)
    assert np.isnan(pts[0, 0]).all()
    assert np.isfinite(pts[4, 4]).all()


def test_camera_to_world_inverts_transform():
    rng = np.random.default_rng(0)
    R, _ = np.linalg.qr(rng.standard_normal((3, 3)))
    t = rng.standard_normal(3)
    x = rng.standard_normal((10, 3))
    x_cam = transform_points(x, R, t)
    back = camera_to_world(x_cam, R, t)
    assert np.allclose(back, x, atol=1e-9)


def test_save_ply_pointcloud(tmp_path):
    pts = np.array([[0, 0, 1], [1, 2, 3], [np.nan, 0, 0]], float)
    cols = np.array([[1.0, 0, 0], [0, 1, 0], [0, 0, 1]])
    n = save_ply_pointcloud(str(tmp_path / "p.ply"), pts, cols)
    assert n == 2  # the NaN point is dropped
    data = (tmp_path / "p.ply").read_bytes()
    assert data.startswith(b"ply")
    assert b"element vertex 2" in data
