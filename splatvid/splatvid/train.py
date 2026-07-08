"""Optimize a GaussianModel against the extracted frames and SfM cameras."""

from __future__ import annotations

import logging
import math
import os
from dataclasses import dataclass, replace

import cv2
import numpy as np
import torch

from .geometry import rodrigues_to_rotmat
from .losses import image_loss, neural_image_loss, psnr, temporal_warp_loss
from .model import GaussianModel
from .render import render_features, render_model
from .sfm import Reconstruction
from .shader import UNetShader
from .view_prior import NoopViewPrior

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
    opacity_reset_every: int = 0  # >0: opacity-reset schedule (floater fix)
    prune_far_factor: float = 0.0  # >0: prune gaussians beyond factor*point-cloud radius
    flatten_weight: float = 0.0  # >0: flatten gaussians into surface disks (better mesh depth)
    appearance: bool = False  # per-view gain/bias to absorb exposure/WB drift (handheld video)
    antialias: bool = False  # Mip-Splatting energy-preserving dilation (sharper splat)
    max_per_tile: int = 1024
    log_every: int = 50
    seed: int = 0
    device: str = "cpu"
    # -- neural renderer (M1+) ---------------------------------------------
    feature_dim: int = 0  # >0 enables per-gaussian features (neural stage)
    neural_iters: int = 1500
    neural_unfreeze_frac: float = 0.7  # unfreeze geometry at this fraction
    perceptual_weight: float = 0.5
    temporal_weight: float = 0.5  # anti-popping view-consistency loss
    temporal_perturb: float = 0.04  # rad, synthesized nearby-camera rotation
    holdout_every: int = 8  # hold out every Nth view for validation
    neural_lr_shader: float = 1e-3
    neural_lr_geom: float = 1e-4
    render_scale: float = 1.0  # <1 splats at reduced res; shader upsamples
    pseudo_weight: float = 0.0  # generative pseudo-view supervision (M4; needs a prior)
    pseudo_perturb: float = 0.12  # rad, novel-camera offset for pseudo-views
    pseudo_per_view: int = 2  # precomputed mesh novel-view targets per train view
    depth_weight: float = 0.0  # mesh depth supervision on the geometry stage


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
        device=cfg.device, feature_dim=cfg.feature_dim,
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
    mesh=None,
    depth_targets=None,
) -> GaussianModel:
    cfg = cfg or TrainConfig()
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)

    views = build_views(rec, images, cfg.train_size, cfg.device)
    model = init_model(rec, cfg)
    extent = rec.scene_extent()
    opt = make_optimizer(model, cfg, extent)

    # Per-view appearance correction (gain + bias) that absorbs the exposure /
    # white-balance drift of handheld phone video, so the optimizer doesn't
    # spawn floaters to explain photometric inconsistency between frames. A
    # training-only nuisance variable — discarded at export/turntable (the
    # gaussian model itself is unchanged). Its own persistent optimizer so its
    # Adam state survives the densification-driven model-optimizer rebuilds.
    app_gain = app_bias = app_opt = None
    if cfg.appearance:
        nv = len(views)
        app_gain = torch.zeros(nv, 3, device=cfg.device, requires_grad=True)  # log-gain
        app_bias = torch.zeros(nv, 3, device=cfg.device, requires_grad=True)
        app_opt = torch.optim.Adam([app_gain, app_bias], lr=1e-3, eps=1e-15)

    # Far-floater pruning reference: the SfM point cloud marks where the real
    # scene is; gaussians that drift well beyond it are floaters.
    prune_center = prune_radius = None
    if cfg.prune_far_factor > 0:
        prune_center = np.median(rec.points, axis=0)
        r = np.linalg.norm(rec.points - prune_center, axis=1)
        prune_radius = float(cfg.prune_far_factor * np.percentile(r, 95))

    xyz_lr_final_factor = 0.01
    densify_until = int(cfg.densify_until_frac * cfg.iterations)
    bg = torch.zeros(3, device=cfg.device)

    # Depth supervision (grounds geometry / kills floaters). Source is either
    # precomputed per-view depth maps (e.g. aligned monocular depth — the
    # independent, non-circular source) or a mesh rendered per view. Computed
    # once up front; it doesn't change as the gaussians optimise.
    mesh_depths = None
    if cfg.depth_weight > 0 and (depth_targets is not None or mesh is not None):
        mesh_depths = []
        for i, v in enumerate(views):
            if depth_targets is not None:
                md = depth_targets[i]
                if md.shape != (v.height, v.width):
                    md = cv2.resize(md, (v.width, v.height), interpolation=cv2.INTER_NEAREST)
            else:
                from .mesh import render_mesh
                _, md = render_mesh(
                    mesh, v.R.detach().cpu().numpy(), v.t.detach().cpu().numpy(),
                    v.focal, v.cx, v.cy, v.width, v.height,
                )
            mesh_depths.append(torch.as_tensor(md, dtype=torch.float32, device=cfg.device))
        src = "monocular" if depth_targets is not None else "mesh"
        log.info("Depth supervision: %s depth for %d views", src, len(views))

    recent_psnr: list[float] = []
    n_nonfinite = 0
    for it in range(1, cfg.iterations + 1):
        vi = int(rng.integers(len(views)))
        view = views[vi]
        pred, info = render_model(
            model, view.R, view.t, view.focal, view.cx, view.cy,
            view.width, view.height, bg=bg, max_per_tile=cfg.max_per_tile,
            return_aux=(mesh_depths is not None), mip=cfg.antialias,
        )
        if cfg.appearance:
            # Map the render through this view's exposure/WB before comparing,
            # so photometric drift is explained here, not by extra gaussians.
            pred = (pred * torch.exp(app_gain[vi]) + app_bias[vi]).clamp(0.0, 1.0)
        loss = image_loss(pred, view.image, cfg.ssim_weight)
        if cfg.appearance:
            # Keep corrections small (identity gain, zero bias) as a prior.
            loss = loss + 0.01 * (app_gain[vi].pow(2).sum() + app_bias[vi].pow(2).sum())
        if cfg.flatten_weight > 0:
            # Flatten each gaussian toward a thin surface disk: minimise the
            # smallest scale relative to the middle one (smin/smid → 0 gives a
            # disk, not a needle). Surface-aligned disks make the rendered
            # median depth crisp, so the fused mesh follows the true surface
            # instead of a fuzzy volumetric shell (RaDe-GS / 2DGS insight).
            s_sorted = model.get_scale().sort(dim=-1).values  # smin, smid, smax
            # Clamp the denominator so the ratio can't blow up as gaussians
            # flatten (a tiny smid otherwise produces huge gradients → NaN).
            loss = loss + cfg.flatten_weight * (
                s_sorted[:, 0] / s_sorted[:, 1].clamp(min=1e-6)
            ).mean()
        if mesh_depths is not None:
            md = mesh_depths[vi]
            # Normalise the opacity-weighted depth (Σwz) by accumulated alpha to
            # get the splat's *surface* depth, comparable to the target; only
            # supervise confidently-covered pixels (alpha>0.5).
            surf = info.depth / info.alpha.clamp(min=1e-3)
            valid = (md > 0) & (info.alpha > 0.5)
            if bool(valid.any()):
                loss = loss + cfg.depth_weight * (surf[valid] - md[valid]).abs().mean()

        # NaN/inf guard: a single non-finite loss, left unchecked, backprops NaN
        # into every parameter and (via the end-of-training transparency prune)
        # can wipe the whole model. Skip the step instead so one bad iteration
        # can't destroy a long run.
        if not torch.isfinite(loss):
            n_nonfinite += 1
            if n_nonfinite <= 3 or n_nonfinite % 100 == 0:
                log.warning(
                    "Non-finite loss at iter %d; skipping step (%d skipped so far)",
                    it, n_nonfinite,
                )
            opt.zero_grad(set_to_none=True)
            if app_opt is not None:
                app_opt.zero_grad(set_to_none=True)
            continue

        opt.zero_grad(set_to_none=True)
        if app_opt is not None:
            app_opt.zero_grad(set_to_none=True)
        loss.backward()

        # Screen-space gradient statistics drive densification (NDC units).
        if info.means2d.grad is not None:
            g = info.means2d.grad
            ndc = torch.stack(
                [g[:, 0] * (view.width / 2.0), g[:, 1] * (view.height / 2.0)], dim=-1
            )
            model.accumulate_grads(ndc.norm(dim=-1).detach(), info.visible)

        # Gradients can be non-finite even when the loss is finite (sqrt/division
        # backward at degenerate gaussians); stepping on them corrupts params.
        if not all(
            torch.isfinite(p.grad).all()
            for p in model.parameters() if p.grad is not None
        ):
            n_nonfinite += 1
            if n_nonfinite <= 3 or n_nonfinite % 100 == 0:
                log.warning("Non-finite gradient at iter %d; skipping step", it)
            opt.zero_grad(set_to_none=True)
            if app_opt is not None:
                app_opt.zero_grad(set_to_none=True)
            continue

        opt.step()
        if app_opt is not None:
            app_opt.step()

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
                cfg.densify_grad_threshold, extent, max_gaussians=cfg.max_gaussians,
                prune_center=prune_center, prune_radius=prune_radius,
            )
            opt = make_optimizer(model, cfg, extent)

        # Opacity reset (the 3DGS floater killer): periodically clamp opacities
        # down so unneeded floaters fade and get pruned next densify pass.
        if (
            cfg.opacity_reset_every > 0
            and cfg.densify_from < it <= densify_until
            and it % cfg.opacity_reset_every == 0
        ):
            model.reset_opacity()
            opt = make_optimizer(model, cfg, extent)

        if it % cfg.log_every == 0 or it == cfg.iterations:
            msg = (
                f"iter {it}/{cfg.iterations}  loss {float(loss.detach()):.4f}  "
                f"psnr {np.mean(recent_psnr):.2f} dB  gaussians {model.num_gaussians}"
            )
            log.info(msg)
            if progress_cb:
                progress_cb(it, cfg.iterations, float(np.mean(recent_psnr)), model.num_gaussians)

    model.prune_transparent()
    log.info("Training done: %d gaussians", model.num_gaussians)
    return model


_GEOM_PARAMS = ("xyz", "log_scale", "quat", "color", "opacity")


def _cam_center(view: TrainView) -> np.ndarray:
    R = view.R.detach().cpu().numpy()
    t = view.t.detach().cpu().numpy()
    return -R.T @ t


def _nearest_neighbours(centers: np.ndarray) -> list[int | None]:
    n = len(centers)
    if n < 2:
        return [None] * n
    out: list[int | None] = []
    for i in range(n):
        d = np.linalg.norm(centers - centers[i], axis=1)
        d[i] = np.inf
        out.append(int(np.argmin(d)))
    return out


def _perturb_cam(view: TrainView, mag: float, rng) -> tuple[torch.Tensor, torch.Tensor]:
    """A synthesized nearby camera: small random rotation about the same centre."""
    dev = view.R.device
    axis = rng.normal(size=3)
    axis = axis / (np.linalg.norm(axis) + 1e-9)
    dR = torch.tensor(rodrigues_to_rotmat(axis * mag), dtype=torch.float32, device=dev)
    Rb = dR @ view.R
    center = -(view.R.T @ view.t)  # camera centre stays put
    tb = -(Rb @ center)
    return Rb, tb


def _temporal_term(model, shader, view, pred_a, info_a, vi, train_views, nn_idx, bg, cfg, rng):
    """Anti-popping loss vs. a real neighbour view (50%) or a synthesized one."""
    use_real = len(train_views) >= 2 and nn_idx[vi] is not None and rng.random() < 0.5
    if use_real:
        vb = train_views[nn_idx[vi]]
        Rb, tb, fb, cxb, cyb, wb, hb = vb.R, vb.t, vb.focal, vb.cx, vb.cy, vb.width, vb.height
    else:
        Rb, tb = _perturb_cam(view, cfg.temporal_perturb, rng)
        fb, cxb, cyb, wb, hb = view.focal, view.cx, view.cy, view.width, view.height
    pred_b, _ = render_features(
        model, shader, Rb, tb, fb, cxb, cyb, wb, hb, bg=bg,
        max_per_tile=cfg.max_per_tile, render_scale=cfg.render_scale,
    )
    cam_a = (view.R, view.t, view.focal, view.cx, view.cy)
    cam_b = (Rb, tb, fb, cxb, cyb)
    return temporal_warp_loss(pred_a, info_a.depth, cam_a, pred_b, cam_b)


def _pseudo_term(model, shader, view, prior, bg, cfg, rng):
    """Generative pseudo-view supervision (M4): render a novel nearby camera and
    pull it toward the prior's cleaned/hallucinated version of that render."""
    Rb, tb = _perturb_cam(view, cfg.pseudo_perturb, rng)
    pred_b, _ = render_features(
        model, shader, Rb, tb, view.focal, view.cx, view.cy, view.width, view.height,
        bg=bg, max_per_tile=cfg.max_per_tile, render_scale=cfg.render_scale,
    )
    cam = (Rb, tb, view.focal, view.cx, view.cy)
    target = prior(pred_b.detach(), cam=cam)
    return (pred_b - target).abs().mean()


def _pseudo_term_banked(model, shader, bank, bg, cfg, rng):
    """Pseudo-view loss against a precomputed mesh target (no per-iter mesh render)."""
    R, t, focal, cx, cy, target = bank[int(rng.integers(len(bank)))]
    h, w = target.shape[0], target.shape[1]
    pred, _ = render_features(
        model, shader, R, t, focal, cx, cy, w, h, bg=bg,
        max_per_tile=cfg.max_per_tile, render_scale=cfg.render_scale,
    )
    return (pred - target).abs().mean()


def train_neural(
    rec: Reconstruction,
    images: list[np.ndarray],
    cfg: TrainConfig | None = None,
    progress_cb=None,
    view_prior=None,
    mesh=None,
    depth_targets=None,
) -> tuple[GaussianModel, UNetShader]:
    """Two-stage neural render: geometry (direct colour) then a U-Net shader.

    Stage 1 is the ordinary gaussian optimisation (features ride along, unused).
    Stage 2 freezes the geometry, trains the U-Net shader + per-gaussian
    features on L1+SSIM+perceptual, then unfreezes geometry at a low LR near the
    end (decision 2). Held-out views (every ``holdout_every``-th) measure
    novel-view quality — the guard against the shader just memorising. Returns
    (model, shader).
    """
    cfg = cfg or TrainConfig()
    if cfg.feature_dim <= 0:
        cfg = replace(cfg, feature_dim=16)

    log.info("[neural] stage 1/2: geometry (%d iters)", cfg.iterations)
    model = train(rec, images, cfg, progress_cb, mesh=mesh, depth_targets=depth_targets)

    dev = cfg.device
    shader = UNetShader(cfg.feature_dim).to(dev)
    geom = [getattr(model, n) for n in _GEOM_PARAMS]
    for p in geom:
        p.requires_grad_(False)

    views = build_views(rec, images, cfg.train_size, dev)
    val_i = set(range(0, len(views), cfg.holdout_every)) if len(views) > cfg.holdout_every else set()
    train_views = [v for i, v in enumerate(views) if i not in val_i]
    val_views = [views[i] for i in sorted(val_i)]

    bg = torch.zeros(cfg.feature_dim, device=dev)
    rng = np.random.default_rng(cfg.seed + 1)
    opt = torch.optim.Adam(
        list(shader.parameters()) + [model.feature], lr=cfg.neural_lr_shader
    )
    unfreeze_at = int(cfg.neural_unfreeze_frac * cfg.neural_iters)

    # Nearest-neighbour train view (by camera centre) for the real temporal pair.
    centers = np.stack([_cam_center(v) for v in train_views]) if train_views else np.zeros((0, 3))
    nn_idx = _nearest_neighbours(centers)
    prior = view_prior if view_prior is not None else NoopViewPrior()

    # Precompute mesh novel-view targets once, so the (slow) numpy mesh render
    # doesn't run every pseudo-view iteration — the loop just resamples the bank.
    pseudo_bank = None
    if cfg.pseudo_weight > 0 and hasattr(prior, "render"):
        pseudo_bank = []
        for v in train_views:
            for _ in range(max(1, cfg.pseudo_per_view)):
                Rb, tb = _perturb_cam(v, cfg.pseudo_perturb, rng)
                tgt = prior.render(Rb, tb, device=dev)
                pseudo_bank.append((Rb, tb, v.focal, v.cx, v.cy, tgt))
        log.info("[neural] precomputed %d mesh pseudo-view targets", len(pseudo_bank))
    log.info(
        "[neural] stage 2/2: shader (%d iters, %d train / %d val views)",
        cfg.neural_iters, len(train_views), len(val_views),
    )

    for it in range(1, cfg.neural_iters + 1):
        vi = int(rng.integers(len(train_views)))
        view = train_views[vi]
        pred, info_a = render_features(
            model, shader, view.R, view.t, view.focal, view.cx, view.cy,
            view.width, view.height, bg=bg, max_per_tile=cfg.max_per_tile,
            render_scale=cfg.render_scale,
        )
        loss = neural_image_loss(
            pred, view.image, cfg.ssim_weight, cfg.perceptual_weight
        )
        if cfg.temporal_weight > 0:
            loss = loss + cfg.temporal_weight * _temporal_term(
                model, shader, view, pred, info_a, vi, train_views, nn_idx, bg, cfg, rng
            )
        if cfg.pseudo_weight > 0:
            if pseudo_bank is not None:
                loss = loss + cfg.pseudo_weight * _pseudo_term_banked(
                    model, shader, pseudo_bank, bg, cfg, rng
                )
            else:
                loss = loss + cfg.pseudo_weight * _pseudo_term(
                    model, shader, view, prior, bg, cfg, rng
                )
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

        if it == unfreeze_at and geom:
            for p in geom:
                p.requires_grad_(True)
            opt.add_param_group({"params": geom, "lr": cfg.neural_lr_geom})
            log.info("[neural] unfroze geometry at iter %d", it)

        if it % cfg.log_every == 0 or it == cfg.neural_iters:
            vp = _neural_val_psnr(model, shader, val_views, bg, cfg.render_scale)
            log.info(
                "[neural] iter %d/%d  loss %.4f  val_psnr %.2f dB",
                it, cfg.neural_iters, float(loss.detach()), vp,
            )
            if progress_cb:
                progress_cb(it, cfg.neural_iters, vp, model.num_gaussians)

    return model, shader


def _neural_val_psnr(model, shader, val_views, bg, render_scale=1.0) -> float:
    if not val_views:
        return float("nan")
    with torch.no_grad():
        vals = []
        for v in val_views:
            pred, _ = render_features(
                model, shader, v.R, v.t, v.focal, v.cx, v.cy, v.width, v.height,
                bg=bg, render_scale=render_scale,
            )
            vals.append(psnr(pred, v.image))
    return float(np.mean(vals))


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
    shader: UNetShader | None = None,
    render_scale: float = 1.0,
) -> None:
    """Render an orbit around the scene to a video file (sanity-check output).

    The orbit follows the plane the cameras actually swept (fitted from the
    recovered camera centres), so it reproduces captured-like viewpoints. A
    world-axis circle instead shows the scene from directions that were never
    filmed — which looks like garbage even when the model is fine.
    """
    dev = model.xyz.device
    C = rec.camera_centers()
    rig_center = C.mean(axis=0)
    d = C - rig_center
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    e0, e1, normal = vt[0], vt[1], vt[2]  # ring plane axes + normal
    radius = float(np.linalg.norm(d, axis=1).mean())
    # Aim at the object, not the whole room: robust centre of the nearer points.
    pc = rec.points - np.median(rec.points, axis=0)
    r = np.linalg.norm(pc, axis=1)
    target = np.median(rec.points[r <= np.percentile(r, 80)], axis=0)
    # Sign the ring normal to agree with the cameras' up axis (world up is the
    # camera's -y direction mapped back to world).
    cam_up = np.mean(
        [rec.poses[i][0].T @ np.array([0.0, -1.0, 0.0]) for i in rec.registered],
        axis=0,
    )
    if np.dot(normal, cam_up) < 0:
        normal = -normal

    s = size / max(rec.width, rec.height)
    w, h = int(rec.width * s), int(rec.height * s)
    vw = cv2.VideoWriter(
        out_path, cv2.VideoWriter_fourcc(*"mp4v"), 24, (w, h)
    )
    # Headless OpenCV builds sometimes cannot open the mp4 encoder and fail
    # silently (producing an empty file). Fall back to a PNG frame sequence
    # so the render is never lost.
    use_writer = vw.isOpened()
    frame_dir = os.path.splitext(out_path)[0] + "_frames"
    if not use_writer:
        vw.release()
        os.makedirs(frame_dir, exist_ok=True)
        log.warning(
            "VideoWriter could not open %s; writing PNG frames to %s instead",
            out_path, frame_dir,
        )
    with torch.no_grad():
        for k in range(n_frames):
            ang = 2 * math.pi * k / n_frames
            eye = rig_center + max(radius, 1e-3) * (
                math.cos(ang) * e0 + math.sin(ang) * e1
            )
            fwd = target - eye
            fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
            right = np.cross(normal, fwd)
            right /= np.linalg.norm(right) + 1e-12
            down = np.cross(fwd, right)
            R = np.stack([right, down, fwd])  # world-to-cam rows
            t = -R @ eye
            R_t = torch.tensor(R, dtype=torch.float32, device=dev)
            t_t = torch.tensor(t, dtype=torch.float32, device=dev)
            fx, cxs, cys = rec.focal * s, rec.cx * s, rec.cy * s
            if shader is not None:
                fbg = torch.zeros(model.get_feature().shape[1], device=dev)
                img, _ = render_features(
                    model, shader, R_t, t_t, fx, cxs, cys, w, h, bg=fbg,
                    render_scale=render_scale,
                )
            else:
                img, _ = render_model(model, R_t, t_t, fx, cxs, cys, w, h)
            frame = (img.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
            bgr = frame[:, :, ::-1]
            if use_writer:
                vw.write(bgr)
            else:
                cv2.imwrite(os.path.join(frame_dir, f"frame_{k:03d}.png"), bgr)
    if use_writer:
        vw.release()
        log.info("Wrote turntable video: %s", out_path)
    else:
        log.info("Wrote %d turntable frames: %s", n_frames, frame_dir)
