"""SfM component tests on synthetic geometry (no images needed)."""

import numpy as np

from splatvid.ba import bundle_adjust
from splatvid.geometry import make_K, project_points
from splatvid.synthetic import orbit_pose


def _make_problem(rng, n_cams=6, n_pts=120, noise_px=0.4):
    focal, cx, cy = 400.0, 320.0, 240.0
    K = make_K(focal, cx, cy)
    pts = rng.uniform(-0.8, 0.8, (n_pts, 3))
    poses = {}
    obs = []
    for c in range(n_cams):
        R, t = orbit_pose(2 * np.pi * c / (n_cams * 3), radius=3.0)
        poses[c] = (R, t)
        uv, z = project_points(pts, R, t, K)
        assert (z > 0).all()
        for p in range(n_pts):
            u, v = uv[p] + rng.normal(0, noise_px, 2)
            obs.append((c, p, u, v))
    return focal, cx, cy, K, poses, pts, obs


def _reproj_rms(focal, cx, cy, poses, pts, obs):
    K = make_K(focal, cx, cy)
    errs = []
    for c, p, u, v in obs:
        uv, _ = project_points(pts[p][None], *poses[c], K)
        errs.append(((uv[0, 0] - u) ** 2 + (uv[0, 1] - v) ** 2))
    return float(np.sqrt(np.mean(errs)))


def test_bundle_adjustment_recovers_perturbation():
    rng = np.random.default_rng(11)
    focal, cx, cy, K, poses, pts, obs = _make_problem(rng)

    # Perturb everything except camera 0 (the gauge anchor).
    bad_poses = {0: poses[0]}
    for c in range(1, len(poses)):
        R, t = poses[c]
        dR = np.eye(3) + np.cross(np.eye(3), rng.normal(0, 0.01, 3))
        u, _, vt = np.linalg.svd(dR @ R)
        bad_poses[c] = (u @ vt, t + rng.normal(0, 0.03, 3))
    bad_pts = pts + rng.normal(0, 0.03, pts.shape)
    bad_focal = focal * 1.06

    rms_before = _reproj_rms(bad_focal, cx, cy, bad_poses, bad_pts, obs)
    new_focal, new_poses, new_pts = bundle_adjust(
        bad_focal, cx, cy, bad_poses, bad_pts, obs, refine_focal=True, max_nfev=60
    )
    rms_after = _reproj_rms(new_focal, cx, cy, new_poses, new_pts, obs)

    assert rms_before > 5.0
    assert rms_after < 1.0  # down to roughly the injected pixel noise
    assert abs(new_focal - focal) / focal < 0.03


def test_bundle_adjustment_keeps_fixed_camera():
    rng = np.random.default_rng(12)
    focal, cx, cy, K, poses, pts, obs = _make_problem(rng, n_cams=4)
    R0, t0 = poses[0]
    _, new_poses, _ = bundle_adjust(
        focal, cx, cy, poses, pts.copy(), obs, refine_focal=False, max_nfev=10
    )
    assert np.allclose(new_poses[0][0], R0)
    assert np.allclose(new_poses[0][1], t0)
