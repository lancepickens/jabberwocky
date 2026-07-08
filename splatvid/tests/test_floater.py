"""Floater-fix tests: opacity reset + far-gaussian pruning."""

import numpy as np
import torch

from splatvid.model import GaussianModel


def _model(n=200, seed=0):
    rng = np.random.default_rng(seed)
    xyz = rng.normal(0.0, 1.0, (n, 3)).astype(np.float32)
    xyz[:5] = 20.0  # 5 far floaters
    rgb = rng.uniform(0.0, 1.0, (n, 3)).astype(np.float32)
    return GaussianModel(xyz, rgb)


def test_reset_opacity():
    m = _model()
    m.opacity.data.fill_(5.0)  # high opacity
    m.reset_opacity(0.01)
    op = m.get_opacity()
    assert torch.allclose(op, torch.full_like(op, 0.01), atol=1e-3)
    # reset only clamps down: an already-low opacity is untouched.
    m.opacity.data.fill_(-10.0)
    m.reset_opacity(0.01)
    assert float(m.get_opacity().max()) < 0.01


def test_far_pruning():
    m = _model()
    # Neutralise the other prune criteria so only far-pruning acts.
    m.log_scale.data.fill_(float(np.log(0.01)))
    m.opacity.data.fill_(5.0)
    m.max_grad_accum = torch.zeros(m.num_gaussians)
    m.grad_count = torch.ones(m.num_gaussians)
    n0 = m.num_gaussians
    m.densify_and_prune(
        grad_threshold=1e9, scene_extent=1e9, min_opacity=0.0,
        prune_center=np.zeros(3), prune_radius=10.0,
    )
    assert m.num_gaussians == n0 - 5  # the 5 far floaters removed
    d = (m.xyz.detach() - torch.zeros(3)).norm(dim=-1)
    assert float(d.max()) <= 10.0


def test_nan_guard_preserves_model():
    """A non-finite loss must not wipe the model (regression: run collapsed to 4)."""
    import math

    import splatvid.train as T
    from splatvid.sfm import Reconstruction
    from splatvid.synthetic import make_scene, orbit_pose
    from splatvid.train import TrainConfig, train

    sc = make_scene(n=400, seed=1)
    xyz, rgb = sc["xyz"].numpy(), sc["rgb"].numpy()
    poses = {i: orbit_pose(2 * math.pi * i / 8, radius=2.6) for i in range(8)}
    rec = Reconstruction(
        focal=1.1 * 64, cx=32, cy=24, width=64, height=48, poses=poses,
        points=xyz.astype(np.float64), point_colors=rgb.astype(np.float64),
        point_errors=np.zeros(len(xyz)), registered=list(range(8)),
    )
    imgs = [np.random.default_rng(i).integers(0, 255, (48, 64, 3), np.uint8) for i in range(8)]

    orig, calls = T.image_loss, {"n": 0}

    def poisoned(pred, tgt, w):  # every 3rd iteration returns NaN
        calls["n"] += 1
        loss = orig(pred, tgt, w)
        return loss * float("nan") if calls["n"] % 3 == 0 else loss

    T.image_loss = poisoned
    try:
        m = train(rec, imgs, TrainConfig(iterations=30, train_size=48,
                                         densify_from=100, flatten_weight=0.1, device="cpu"))
    finally:
        T.image_loss = orig
    assert m.num_gaussians > 100  # not collapsed
    for p in (m.xyz, m.log_scale, m.quat, m.opacity, m.color):
        assert torch.isfinite(p).all()
