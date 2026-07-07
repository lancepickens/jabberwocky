"""Monocular depth prior tests (alignment; model itself is optional)."""

import numpy as np

from splatvid.depth_prior import align_disparity_to_recon, depth_available
from splatvid.geometry import project_points


def test_depth_available_is_bool():
    assert isinstance(depth_available(), bool)


def _single_cam_reconstruction(n=400, w=200, h=200, seed=0):
    from splatvid.sfm import Reconstruction

    rng = np.random.default_rng(seed)
    pts = rng.uniform(-0.4, 0.4, (n, 3))
    pts[:, 2] += 4.0  # in front of the camera, depth ~4
    return Reconstruction(
        focal=1.2 * w, cx=w / 2, cy=h / 2, width=w, height=h,
        poses={0: (np.eye(3), np.zeros(3))},
        points=pts, point_colors=np.full((n, 3), 0.5),
        point_errors=np.zeros(n), registered=[0],
    )


def test_align_disparity_recovers_depth():
    # Build a disparity map that is a known affine of true inverse-depth at the
    # sparse points: disp = 2*(1/z) + 0.1. Alignment must invert it back to z.
    rec = _single_cam_reconstruction()
    R, t = rec.poses[0]
    uv, z = project_points(rec.points, R, t, rec.K)
    w, h = rec.width, rec.height
    disp = np.zeros((h, w), np.float32)
    px = np.clip(uv[:, 0].round().astype(int), 0, w - 1)
    py = np.clip(uv[:, 1].round().astype(int), 0, h - 1)
    disp[py, px] = (2.0 / z + 0.1).astype(np.float32)

    depth = align_disparity_to_recon(disp, rec, 0)
    assert depth is not None
    rec_z = depth[py, px]
    ok = rec_z > 0
    # Recovered depth at the sparse points matches their true depth.
    assert ok.mean() > 0.9
    assert np.median(np.abs(rec_z[ok] - z[ok]) / z[ok]) < 0.02


def test_align_too_few_points_returns_none():
    rec = _single_cam_reconstruction(n=5)
    disp = np.ones((rec.height, rec.width), np.float32)
    assert align_disparity_to_recon(disp, rec, 0, min_points=20) is None
