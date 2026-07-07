import math

import numpy as np
import torch

from splatvid.render import compute_cov3d, project_gaussians, render


def _single_gaussian(opacity=0.9, scale=0.1, z=5.0):
    xyz = torch.tensor([[0.0, 0.0, z]])
    scales = torch.full((1, 3), scale)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    rgb = torch.tensor([[1.0, 0.25, 0.0]])
    op = torch.tensor([[opacity]])
    return xyz, scales, quat, rgb, op


def _identity_cam():
    return torch.eye(3), torch.zeros(3)


def test_cov3d_isotropic():
    scale = torch.full((1, 3), 0.2)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    cov = compute_cov3d(scale, quat)
    assert torch.allclose(cov[0], 0.04 * torch.eye(3), atol=1e-6)
    # Rotation must not change an isotropic covariance.
    q = torch.tensor([[0.9, 0.1, 0.3, 0.2]])
    q = q / q.norm()
    cov_r = compute_cov3d(scale, q)
    assert torch.allclose(cov_r[0], 0.04 * torch.eye(3), atol=1e-5)


def test_projection_center_and_depth():
    xyz, scales, quat, _, _ = _single_gaussian(z=4.0)
    cov3d = compute_cov3d(scales, quat)
    R, t = _identity_cam()
    means2d, cov2d, depth, in_front = project_gaussians(
        xyz, cov3d, R, t, focal=300.0, cx=160.0, cy=120.0
    )
    assert in_front.all()
    assert torch.allclose(means2d[0], torch.tensor([160.0, 120.0]))
    assert torch.allclose(depth[0], torch.tensor(4.0))
    # sigma_px ~ focal * sigma / z = 300 * 0.1 / 4 = 7.5 -> var 56.25 (+0.3)
    assert abs(cov2d[0, 0, 0].item() - 56.55) < 0.5


def test_render_single_gaussian_peak():
    xyz, scales, quat, rgb, op = _single_gaussian(opacity=0.8, scale=0.15, z=5.0)
    R, t = _identity_cam()
    img, info = render(
        xyz, scales, quat, rgb, op, R, t,
        focal=250.0, cx=64.0, cy=48.0, width=128, height=96,
    )
    assert img.shape == (96, 128, 3)
    assert info.visible.all()
    # Peak should be at the projected center with alpha ~ opacity.
    peak = img[48, 64]
    assert torch.allclose(peak, 0.8 * rgb[0], atol=0.03)
    # Far corner is background (black).
    assert img[0, 0].abs().max() < 1e-4


def test_render_compositing_order():
    # A red gaussian in front of a green one, same line of sight.
    xyz = torch.tensor([[0.0, 0.0, 3.0], [0.0, 0.0, 6.0]])
    scales = torch.full((2, 3), 0.2)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]] * 2)
    rgb = torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    op = torch.tensor([[0.6], [0.9]])
    R, t = _identity_cam()
    img, _ = render(
        xyz, scales, quat, rgb, op, R, t,
        focal=200.0, cx=32.0, cy=32.0, width=64, height=64,
    )
    center = img[32, 32]
    # Front red contributes 0.6; green behind contributes 0.9 * (1 - 0.6).
    assert abs(center[0].item() - 0.6) < 0.05
    assert abs(center[1].item() - 0.36) < 0.05
    # Swapping declaration order must not change the result (depth sort).
    img2, _ = render(
        xyz.flip(0), scales, quat, rgb.flip(0), op.flip(0), R, t,
        focal=200.0, cx=32.0, cy=32.0, width=64, height=64,
    )
    assert torch.allclose(img, img2, atol=1e-5)


def test_render_gradients_flow():
    xyz, scales, quat, rgb, op = _single_gaussian()
    params = [p.clone().requires_grad_(True) for p in (xyz, scales, quat, rgb, op)]
    R, t = _identity_cam()
    img, info = render(
        params[0], params[1], params[2], params[3], params[4], R, t,
        focal=200.0, cx=32.0, cy=32.0, width=64, height=64,
    )
    target = torch.zeros_like(img)
    loss = ((img - target) ** 2).mean()
    loss.backward()
    for name, p in zip(("xyz", "scale", "quat", "rgb", "op"), params):
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    assert params[0].grad.abs().sum() > 0
    assert info.means2d.grad is not None
    assert torch.isfinite(info.means2d.grad).all()


def test_render_behind_camera_is_empty():
    xyz, scales, quat, rgb, op = _single_gaussian(z=-3.0)
    R, t = _identity_cam()
    bg = torch.tensor([0.2, 0.3, 0.4])
    img, info = render(
        xyz, scales, quat, rgb, op, R, t,
        focal=200.0, cx=32.0, cy=32.0, width=64, height=64, bg=bg,
    )
    assert not info.visible.any()
    assert torch.allclose(img, bg.expand(64, 64, 3))


def test_render_moved_camera():
    # Gaussian at origin, camera pulled back and rotated 90 deg about y:
    # camera at (5, 0, 0) looking toward origin must still see it centered.
    xyz = torch.tensor([[0.0, 0.0, 0.0]])
    scales = torch.full((1, 3), 0.1)
    quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    rgb = torch.tensor([[0.0, 0.5, 1.0]])
    op = torch.tensor([[0.9]])
    ang = -math.pi / 2
    R = torch.tensor(
        [
            [math.cos(ang), 0.0, -math.sin(ang)],
            [0.0, 1.0, 0.0],
            [math.sin(ang), 0.0, math.cos(ang)],
        ]
    )
    eye = torch.tensor([5.0, 0.0, 0.0])
    t = -R @ eye
    img, info = render(
        xyz, scales, quat, rgb, op, R, t,
        focal=200.0, cx=32.0, cy=32.0, width=64, height=64,
    )
    assert info.visible.all()
    assert img[32, 32, 2] > 0.5  # blue peak at image center


def _feature_scene(seed=0, n=300):
    from splatvid.model import GaussianModel

    rng = np.random.default_rng(seed)
    xyz = rng.normal(0.0, 0.4, (n, 3)).astype(np.float32)
    xyz[:, 2] += 4.0
    rgb = rng.uniform(0.0, 1.0, (n, 3)).astype(np.float32)
    return GaussianModel(xyz, rgb, feature_dim=16)


def test_feature_render_matches_direct_color():
    # M0 plumbing: a feature render (feature[:, :3] == colour) decoded by an
    # identity shader must reproduce the direct-colour rasteriser exactly.
    from splatvid.render import render_features, render_model
    from splatvid.shader import IdentityShader

    model = _feature_scene(seed=0)
    R, t = _identity_cam()
    args = dict(focal=200.0, cx=48.0, cy=36.0, width=96, height=72)

    rgb_img, _ = render_model(model, R, t, **args)
    shaded, info = render_features(model, IdentityShader(16), R, t, **args)

    assert shaded.shape == (72, 96, 3)
    assert torch.allclose(rgb_img, shaded, atol=1e-5)
    # Auxiliary buffers exist and are sane.
    assert info.alpha is not None and info.depth is not None
    assert info.alpha.shape == (72, 96) and info.depth.shape == (72, 96)
    a = info.alpha.detach()
    assert float(a.min()) >= 0.0 and float(a.max()) <= 1.0 + 1e-4


def test_feature_render_gradients_flow():
    # Gradients must reach the per-gaussian features through the shader.
    from splatvid.render import render_features
    from splatvid.shader import IdentityShader

    model = _feature_scene(seed=1, n=200)
    R, t = _identity_cam()
    out, _ = render_features(
        model, IdentityShader(16), R, t,
        focal=200.0, cx=48.0, cy=36.0, width=96, height=72,
    )
    out.sum().backward()
    assert model.feature.grad is not None
    assert torch.isfinite(model.feature.grad).all()
    assert float(model.feature.grad.abs().sum()) > 0.0
