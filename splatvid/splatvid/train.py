"""Optimize a GaussianModel against the extracted frames and SfM cameras."""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import cv2
import numpy as np
import torch

from .losses import image_loss, psnr
from .model import GaussianModel
from .render import render_model
from .sfm import Reconstruction

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    iterations: int = 2000
    train_size: int = 320  # max image dimension used during optimization
    ssim_weight: float = 0.2
    lr_xyz: float = 1.6e-4  # scaled by scene extent
    lr_scale: float = 5e-3
    lr_quat: float = 1e-3
    lr_color: float = 2.5e-3
    lr_opacity: float = 2.5e-2
    densify_from: int = 300
    densify_until_frac: float = 0.7  # stop densifying at this fraction of training
    densify_every: int = 150
    densify_grad_threshold: float = 2e-4
    max_gaussians: int = 60_000
    max_per_tile: int = 1024
    log_every: int = 50
    seed: int = 0
    device: str = "cpu"


@dataclass
class TrainView:
    image: torch.Tensor  # (H, W, 3) float in [0, 1], RGB
    R: torch.Tensor
    t: torch.Tensor
    focal: float
    cx: float
    cy: float
    width: int
    height: int


def build_views(
    rec: Reconstruction,
    images: list[np.ndarray],
    train_size: int,
    device: str,
) -> list[TrainView]:
    """Downscale frames and intrinsics to the training resolution."""
    views = []
    for fi in rec.registered:
        img = images[fi]
        h, w = img.shape[:2]
        s = min(1.0, train_size / max(h, w))
        tw, th = int(round(w * s)), int(round(h * s))
        small = cv2.resize(img, (tw, th), interpolation=cv2.INTER_AREA)
        rgb = torch.tensor(
            small[:, :, ::-1].astype(np.float32) / 255.0, device=device
        )
        R, t = rec.poses[fi]
        views.append(
            TrainView(
                image=rgb,
                R=torch.tensor(R, dtype=torch.float32, device=device),
                t=torch.tensor(t, dtype=torch.float32, device=device),
                focal=rec.focal * s,
                cx=rec.cx * s,
                cy=rec.cy * s,
                width=tw,
                height=th,
            )
        )
    return views


def init_model(rec: Reconstruction, cfg: TrainConfig) -> GaussianModel:
    """Seed gaussians from the SfM point cloud (dropping the worst outliers)."""
    pts = rec.points
    cols = rec.point_colors
    center = np.median(pts, axis=0)
    d = np.linalg.norm(pts - center, axis=1)
    keep = d < np.percentile(d, 98) * 1.5  # drop far fliers that wreck scale init
    model = GaussianModel(
        pts[keep].astype(np.float32), cols[keep].astype(np.float32),
        device=cfg.device,
    )
    log.info("Initialized %d gaussians from SfM points", model.num_gaussians)
    return model


def make_optimizer(model: GaussianModel, cfg: TrainConfig, extent: float) -> torch.optim.Adam:
    return torch.optim.Adam(
        [
            {"params": [model.xyz], "lr": cfg.lr_xyz * extent, "name": "xyz"},
            {"params": [model.log_scale], "lr": cfg.lr_scale, "name": "scale"},
            {"params": [model.quat], "lr": cfg.lr_quat, "name": "quat"},
            {"params": [model.color], "lr": cfg.lr_color, "name": "color"},
            {"params": [model.opacity], "lr": cfg.lr_opacity, "name": "opacity"},
        ],
        eps=1e-15,
    )


def train(
    rec: Reconstruction,
    images: list[np.ndarray],
    cfg: TrainConfig | None = None,
    progress_cb=None,
) -> GaussianModel:
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    views = build_views(rec, images, cfg.train_size, cfg.device)
    model = init_model(rec, cfg)
    extent = rec.scene_extent()
    opt = make_optimizer(model, cfg, extent)

    xyz_lr_final_factor = 0.01
    densify_until = int(cfg.densify_until_frac * cfg.iterations)
    bg = torch.zeros(3, device=cfg.device)

    recent_psnr: list[float] = []
    for it in range(1, cfg.iterations + 1):
        view = views[int(rng.integers(len(views)))]
        pred, info = render_model(
            model, view.R, view.t, view.focal, view.cx, view.cy,
            view.width, view.height, bg=bg, max_per_tile=cfg.max_per_tile,
        )
        loss = image_loss(pred, view.image, cfg.ssim_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()

        # Screen-space gradient statistics drive densification (NDC units).
        if info.means2d.grad is not None:
            g = info.means2d.grad
            ndc = torch.stack(
                [g[:, 0] * (view.width / 2.0), g[:, 1] * (view.height / 2.0)], dim=-1
            )
            model.accumulate_grads(ndc.norm(dim=-1).detach(), info.visible)

        opt.step()

        # Exponential xyz learning-rate decay, as in the reference trainer.
        decay = xyz_lr_final_factor ** (it / cfg.iterations)
        opt.param_groups[0]["lr"] = cfg.lr_xyz * extent * decay

        with torch.no_grad():
            recent_psnr.append(psnr(pred, view.image))
            if len(recent_psnr) > 50:
                recent_psnr.pop(0)

        if (
            cfg.densify_from <= it <= densify_until
            and it % cfg.densify_every == 0
        ):
            model.densify_and_prune(
                cfg.densify_grad_threshold, extent, max_gaussians=cfg.max_gaussians
            )
            opt = make_optimizer(model, cfg, extent)

        if it % cfg.log_every == 0 or it == cfg.iterations:
            msg = (
                f"iter {it}/{cfg.iterations}  loss {float(loss):.4f}  "
                f"psnr {np.mean(recent_psnr):.2f} dB  gaussians {model.num_gaussians}"
            )
            log.info(msg)
            if progress_cb:
                progress_cb(it, cfg.iterations, float(np.mean(recent_psnr)), model.num_gaussians)

    model.prune_transparent()
    log.info("Training done: %d gaussians", model.num_gaussians)
    return model


def evaluate(model: GaussianModel, views: list[TrainView], max_views: int = 10) -> float:
    """Mean PSNR over a subset of training views."""
    step = max(1, len(views) // max_views)
    vals = []
    with torch.no_grad():
        for view in views[::step]:
            pred, _ = render_model(
                model, view.R, view.t, view.focal, view.cx, view.cy,
                view.width, view.height,
            )
            vals.append(psnr(pred, view.image))
    return float(np.mean(vals))


def render_turntable(
    model: GaussianModel,
    rec: Reconstruction,
    out_path: str,
    n_frames: int = 60,
    size: int = 480,
) -> None:
    """Render an orbit around the scene to a video file (sanity-check output)."""
    import cv2 as _cv2

    C = rec.camera_centers()
    center_pts = np.median(rec.points, axis=0)
    rig_center = C.mean(axis=0)
    radius = float(np.linalg.norm(C - rig_center, axis=1).mean())
    up_guess = np.array([0.0, -1.0, 0.0])

    s = size / max(rec.width, rec.height)
    w, h = int(rec.width * s), int(rec.height * s)
    vw = _cv2.VideoWriter(
        out_path, _cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h)
    )
    with torch.no_grad():
        for k in range(n_frames):
            ang = 2 * math.pi * k / n_frames
            offset = np.array([math.cos(ang), 0.0, math.sin(ang)]) * max(radius, 1e-3)
            eye = rig_center + offset
            fwd = center_pts - eye
            fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
            right = np.cross(fwd, up_guess)
            right /= np.linalg.norm(right) + 1e-12
            up = np.cross(fwd, right)
            R = np.stack([right, up, fwd])  # world-to-cam rows
            t = -R @ eye
            img, _ = render_model(
                model,
                torch.tensor(R, dtype=torch.float32),
                torch.tensor(t, dtype=torch.float32),
                rec.focal * s, rec.cx * s, rec.cy * s, w, h,
            )
            frame = (img.numpy() * 255).clip(0, 255).astype(np.uint8)
            vw.write(frame[:, :, ::-1])
    vw.release()
    log.info("Wrote turntable video: %s", out_path)
