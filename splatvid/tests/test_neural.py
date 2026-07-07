"""Neural-renderer (M1) tests: U-Net shader + two-stage training."""

import numpy as np
import torch

from splatvid.shader import UNetShader


def test_unet_shader_shapes_and_grads():
    # Odd, non-power-of-two size to exercise the bilinear up/down handling.
    h, w, c = 50, 70, 16
    shader = UNetShader(c)
    feat = torch.rand(h, w, c, requires_grad=True)
    alpha = torch.rand(h, w)
    depth = torch.rand(h, w) * 5.0
    out = shader(feat, alpha, depth)
    assert out.shape == (h, w, 3)
    od = out.detach()
    assert float(od.min()) >= 0.0 and float(od.max()) <= 1.0
    out.sum().backward()
    # Gradients reach both the shader weights and the incoming feature map.
    assert feat.grad is not None and torch.isfinite(feat.grad).all()
    p = next(shader.parameters())
    assert p.grad is not None and float(p.grad.abs().sum()) > 0.0


def _tiny_reconstruction(n_cam=3, n_pts=300, w=64, h=48, seed=0):
    from splatvid.sfm import Reconstruction
    from splatvid.synthetic import orbit_pose

    rng = np.random.default_rng(seed)
    pts = rng.normal(0.0, 0.3, (n_pts, 3))
    cols = rng.uniform(0.0, 1.0, (n_pts, 3))
    poses = {}
    for i in range(n_cam):
        R, t = orbit_pose(0.3 * i - 0.3, radius=3.0)
        poses[i] = (R, t)
    images = [rng.uniform(0, 255, (h, w, 3)).astype(np.uint8) for _ in range(n_cam)]
    rec = Reconstruction(
        focal=60.0, cx=w / 2, cy=h / 2, width=w, height=h,
        poses=poses, points=pts, point_colors=cols,
        point_errors=np.zeros(n_pts), registered=list(range(n_cam)),
    )
    return rec, images


def test_temporal_warp_identity_is_zero():
    from splatvid.losses import temporal_warp_loss

    torch.manual_seed(0)
    h, w = 24, 32
    img = torch.rand(h, w, 3)
    depth = torch.full((h, w), 3.0)
    cam = (torch.eye(3), torch.zeros(3), 40.0, w / 2, h / 2)
    # Same image + same camera: the warp is the identity, so loss ~ 0.
    loss0 = temporal_warp_loss(img, depth, cam, img, cam)
    assert float(loss0) < 1e-4
    # A different image at the same camera is penalised.
    loss1 = temporal_warp_loss(img, depth, cam, torch.rand(h, w, 3), cam)
    assert float(loss1) > 1e-2


def test_temporal_warp_gradients():
    from splatvid.losses import temporal_warp_loss

    h, w = 24, 32
    a = torch.rand(h, w, 3, requires_grad=True)
    b = torch.rand(h, w, 3, requires_grad=True)
    depth = torch.full((h, w), 3.0)
    cam = (torch.eye(3), torch.zeros(3), 40.0, w / 2, h / 2)
    temporal_warp_loss(a, depth, cam, b, cam).backward()
    assert a.grad is not None and torch.isfinite(a.grad).all()
    assert b.grad is not None and torch.isfinite(b.grad).all()


def test_train_neural_smoke():
    # End-to-end plumbing of the two-stage neural loop on a tiny scene.
    # perceptual/temporal off so this isolates the M1 path (no torchvision).
    from splatvid.train import TrainConfig, train_neural

    rec, images = _tiny_reconstruction()
    cfg = TrainConfig(
        iterations=3, neural_iters=4, train_size=48, feature_dim=8,
        densify_from=100, holdout_every=2, log_every=2,
        perceptual_weight=0.0, temporal_weight=0.0, device="cpu",
    )
    model, shader = train_neural(rec, images, cfg)
    assert model.get_feature() is not None
    assert model.get_feature().shape[1] == 8
    assert isinstance(shader, UNetShader)
    assert model.num_gaussians > 0


def test_train_neural_temporal_smoke():
    # The two-stage loop runs with the temporal (anti-popping) loss enabled.
    from splatvid.train import TrainConfig, train_neural

    rec, images = _tiny_reconstruction()
    cfg = TrainConfig(
        iterations=2, neural_iters=4, train_size=48, feature_dim=8,
        densify_from=100, holdout_every=2, log_every=4,
        perceptual_weight=0.0, temporal_weight=0.5, device="cpu",
    )
    model, shader = train_neural(rec, images, cfg)
    assert isinstance(shader, UNetShader)


def test_render_scale_upsamples_to_full():
    # Splat at half resolution; the shader must return the full output size.
    from splatvid.model import GaussianModel
    from splatvid.render import render_features

    rng = np.random.default_rng(0)
    xyz = rng.normal(0.0, 0.4, (200, 3)).astype(np.float32)
    xyz[:, 2] += 4.0
    rgb = rng.uniform(0.0, 1.0, (200, 3)).astype(np.float32)
    model = GaussianModel(xyz, rgb, feature_dim=16)
    out, info = render_features(
        model, UNetShader(16), torch.eye(3), torch.zeros(3),
        200.0, 48, 36, 96, 72, render_scale=0.5,
    )
    assert out.shape == (72, 96, 3)
    assert info.alpha.shape == (72, 96) and info.depth.shape == (72, 96)


def test_train_neural_half_res_smoke():
    # Half-res splatting + temporal loss together, end to end.
    from splatvid.train import TrainConfig, train_neural

    rec, images = _tiny_reconstruction()
    cfg = TrainConfig(
        iterations=2, neural_iters=3, train_size=48, feature_dim=8,
        densify_from=100, holdout_every=2, log_every=3,
        perceptual_weight=0.0, temporal_weight=0.5, render_scale=0.5, device="cpu",
    )
    model, shader = train_neural(rec, images, cfg)
    assert isinstance(shader, UNetShader)


def test_view_prior_noop_and_stub():
    from splatvid.view_prior import NoopViewPrior

    img = torch.rand(8, 8, 3, requires_grad=True)
    # No-op prior: target is the detached render, so the pseudo loss is zero.
    tgt = NoopViewPrior()(img)
    assert torch.allclose(tgt, img.detach())
    assert float((img.detach() - tgt).abs().mean()) == 0.0

    # A real prior (stub here) yields a nonzero, differentiable pull.
    class _Stub:
        def __call__(self, x):
            return torch.zeros_like(x)

    loss = (img - _Stub()(img)).abs().mean()
    loss.backward()
    assert img.grad is not None and float(img.grad.abs().sum()) > 0.0


def test_train_neural_pseudo_smoke():
    # The pseudo-view (M4) mechanism runs end to end with the default prior.
    from splatvid.train import TrainConfig, train_neural
    from splatvid.view_prior import NoopViewPrior

    rec, images = _tiny_reconstruction()
    cfg = TrainConfig(
        iterations=2, neural_iters=3, train_size=48, feature_dim=8,
        densify_from=100, holdout_every=2, log_every=3,
        perceptual_weight=0.0, temporal_weight=0.0, pseudo_weight=0.5, device="cpu",
    )
    model, shader = train_neural(rec, images, cfg, view_prior=NoopViewPrior())
    assert isinstance(shader, UNetShader)


def test_perceptual_loss_optional():
    # perceptual_available() must be a bool; if present, it returns a scalar.
    from splatvid.losses import perceptual_available, perceptual_loss

    assert isinstance(perceptual_available(), bool)
    if perceptual_available():
        a = torch.rand(16, 16, 3)
        b = torch.rand(16, 16, 3)
        v = perceptual_loss(a, b)
        assert v.ndim == 0 and float(v) >= 0.0
