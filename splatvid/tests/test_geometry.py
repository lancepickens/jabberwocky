import numpy as np
import pytest

from splatvid.geometry import (
    camera_center,
    make_K,
    project_points,
    quat_to_rotmat,
    rodrigues_to_rotmat,
    rotmat_to_quat,
    rotmat_to_rodrigues,
    triangulate_point,
    triangulation_angle_deg,
)


def random_rotation(rng):
    q = rng.normal(size=4)
    return quat_to_rotmat(q / np.linalg.norm(q))


def test_quat_rotmat_roundtrip():
    rng = np.random.default_rng(0)
    for _ in range(50):
        R = random_rotation(rng)
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-10)
        assert np.isclose(np.linalg.det(R), 1.0)
        q = rotmat_to_quat(R)
        R2 = quat_to_rotmat(q)
        assert np.allclose(R, R2, atol=1e-8)


def test_rodrigues_roundtrip():
    rng = np.random.default_rng(1)
    for _ in range(50):
        r = rng.normal(size=3)
        R = rodrigues_to_rotmat(r)
        r2 = rotmat_to_rodrigues(R)
        R2 = rodrigues_to_rotmat(r2)
        assert np.allclose(R, R2, atol=1e-8)
    assert np.allclose(rodrigues_to_rotmat(np.zeros(3)), np.eye(3))


def test_rodrigues_batched_matches_single():
    rng = np.random.default_rng(2)
    rs = rng.normal(size=(10, 3))
    batch = rodrigues_to_rotmat(rs)
    for i in range(10):
        assert np.allclose(batch[i], rodrigues_to_rotmat(rs[i]), atol=1e-12)


def test_project_and_triangulate():
    rng = np.random.default_rng(3)
    K = make_K(500.0, 320.0, 240.0)
    X = rng.normal(size=(20, 3)) * 0.5 + np.array([0, 0, 5.0])

    R1, t1 = np.eye(3), np.zeros(3)
    R2 = quat_to_rotmat(np.array([0.98, 0.0, 0.2, 0.0]))
    t2 = np.array([-1.0, 0.1, 0.2])

    uv1, z1 = project_points(X, R1, t1, K)
    uv2, z2 = project_points(X, R2, t2, K)
    assert (z1 > 0).all() and (z2 > 0).all()

    for i in range(20):
        Xr = triangulate_point([(uv1[i], R1, t1), (uv2[i], R2, t2)], K)
        assert Xr is not None
        assert np.allclose(Xr, X[i], atol=1e-6)


def test_camera_center_and_angle():
    R = np.eye(3)
    t = np.array([0.0, 0.0, -4.0])
    C = camera_center(R, t)
    assert np.allclose(C, [0, 0, 4])
    # Two cameras at +-1 in x looking at origin from z=4: small angle.
    ang = triangulation_angle_deg(
        np.zeros(3), np.array([1.0, 0, 4.0]), np.array([-1.0, 0, 4.0])
    )
    expected = 2 * np.degrees(np.arctan2(1.0, 4.0))
    assert ang == pytest.approx(expected, abs=0.5)
