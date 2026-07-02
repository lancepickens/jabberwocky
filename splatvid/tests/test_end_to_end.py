"""Full-pipeline test: synthetic orbit video -> SfM -> training -> export.

Marked slow: takes a few minutes on CPU. Run with `pytest -m slow` or
plain `pytest` (it is included by default).
"""

import os

import numpy as np
import pytest
import torch

from splatvid.export import load_splat, save_ply, save_splat
from splatvid.losses import psnr
from splatvid.render import render_model
from splatvid.sfm import run_sfm
from splatvid.synthetic import make_synthetic_video
from splatvid.train import TrainConfig, build_views, train
from splatvid.video import extract_frames


@pytest.fixture(scope="module")
def video_path(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("video") / "orbit.mp4")
    make_synthetic_video(path, n_frames=48, width=480, height=360, n_gaussians=4500)
    assert os.path.getsize(path) > 10_000
    return path


@pytest.fixture(scope="module")
def frames(video_path):
    fs = extract_frames(video_path, max_frames=40, max_dim=480)
    assert len(fs.images) >= 30
    assert fs.size == (480, 360)
    return fs


@pytest.fixture(scope="module")
def reconstruction(frames):
    rec = run_sfm(frames.images, n_features=3000)
    return rec


@pytest.mark.slow
def test_sfm_recovers_orbit(frames, reconstruction):
    rec = reconstruction
    # Most cameras registered, healthy point cloud, low reprojection error.
    assert len(rec.registered) >= 0.7 * len(frames.images)
    assert rec.points.shape[0] >= 200
    assert float(np.median(rec.point_errors)) < 2.0

    # Ground truth is a ring of radius 2.6 around the scene: recovered
    # camera centers must be roughly coplanar and equidistant from their
    # centroid (up to global scale).
    C = rec.camera_centers()
    centroid = C.mean(axis=0)
    d = np.linalg.norm(C - centroid, axis=1)
    assert d.std() / d.mean() < 0.15  # ring, not a blob

    # Coplanarity: smallest principal axis much smaller than the others.
    s = np.linalg.svd(C - centroid, compute_uv=False)
    assert s[2] < 0.15 * s[0]


@pytest.mark.slow
def test_training_improves_and_exports(frames, reconstruction, tmp_path):
    rec = reconstruction
    cfg = TrainConfig(
        iterations=250,
        train_size=160,
        densify_from=60,
        densify_every=80,
        max_gaussians=8000,
        log_every=50,
    )
    views = build_views(rec, frames.images, cfg.train_size, cfg.device)

    model = train(rec, frames.images, cfg)
    assert model.num_gaussians > 100

    with torch.no_grad():
        vals = []
        for view in views[:: max(1, len(views) // 6)]:
            pred, _ = render_model(
                model, view.R, view.t, view.focal, view.cx, view.cy,
                view.width, view.height,
            )
            vals.append(psnr(pred, view.image))
    mean_psnr = float(np.mean(vals))
    # Even a short CPU run must beat a flat gray image (~11 dB on this scene).
    assert mean_psnr > 14.0, f"PSNR too low: {mean_psnr:.2f} dB"

    ply = str(tmp_path / "scene.ply")
    splat = str(tmp_path / "scene.splat")
    save_ply(model, ply)
    save_splat(model, splat)
    assert os.path.getsize(ply) > 1000
    back = load_splat(splat)
    assert back["xyz"].shape[0] == model.num_gaussians
