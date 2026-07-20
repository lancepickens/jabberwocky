"""Artifact repair + scene completion for trained splats (ArtiFixer-style).

Implements the pipeline of NVIDIA's ArtiFixer (de Lutio et al., SIGGRAPH
2026, arXiv:2603.00492): a trained 3DGS scene looks right from the captured
views but collapses into holes and floaters from viewpoints the camera never
visited. ArtiFixer's recipe, which this module follows stage for stage:

1. Render novel views along an *extended* camera trajectory and read the
   rasterizer's accumulated **opacity map** as per-pixel confidence in the
   reconstruction (their "opacity mixing": trusted content is kept, low-
   opacity holes are handed to a generator to fill).
2. Repair each novel render **auto-regressively** along the trajectory —
   every fixed frame becomes conditioning context for the next one, so
   filled-in content stays consistent instead of flickering view to view.
3. Feed the fixed frames back as **pseudo-supervision** and fine-tune the
   gaussians, which lifts quality in the under-observed regions (the paper
   reports 1-3 dB PSNR over prior repair methods).

ArtiFixer's generator is a 14B-parameter camera-aware video diffusion model
(Wan 2.1, distilled into a causal auto-regressive student). Shipping that
contradicts this repo's from-scratch, no-heavy-checkpoint rule — and it does
not run on our Apple-Silicon-only fleet as a dependency of a CPU/MPS
pipeline. So the *generator* is replaced with a deterministic, geometry-aware
filler with the same information flow: holes are filled by warping nearby
captured frames (and previously fixed frames — the auto-regressive context)
into the novel view, plane-sweeping a few candidate depths around the
pull-push-completed surface and keeping the one where sources photometrically
agree. Unlike the diffusion generator it never hallucinates: pixels no view
has observed keep the render and are excluded from supervision. The
opacity-gated mixing, causal trajectory ordering, floater removal, and
confidence-weighted pseudo-supervised fine-tune are exactly ArtiFixer's
structure; a learned generative prior can later slot into
``SplatRepairPrior`` unchanged.

Everything is pure PyTorch float32 (MPS has no float64), built from ops the
Metal backend implements (conv/pool, grid_sample, unfold, median), with the
same CPU-side bookkeeping conventions as ``render.py`` — so the whole pass
runs on Apple Silicon under ``--device mps`` (or anywhere else).
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .losses import image_loss, psnr
from .model import GaussianModel
from .render import render_model
from .sfm import Reconstruction
from .train import TrainConfig, TrainView, build_views, make_optimizer
from .view_prior import ViewPrior

log = logging.getLogger(__name__)


@dataclass
class ArtifixConfig:
    """Knobs for the repair pass. Defaults are sized for CPU/MPS training."""

    n_novel: int = 24  # novel views along the extended trajectory
    elev_amp: float = 0.35  # out-of-plane sweep, as a fraction of orbit radius
    radius_amp: float = 0.18  # in/out sweep, fraction of orbit radius
    conf_floor: float = 0.15  # alpha below this is a hole (confidence 0)
    conf_ceil: float = 0.85  # alpha above this is trusted (confidence 1)
    floater_ratio: float = 0.72  # depth < ratio * local median depth => floater
    floater_kernel: int = 9  # local-median window for floater detection (px)
    floater_vote_frac: float = 0.55  # flag gaussians voted floater this often
    floater_min_views: int = 2  # ... with at least this much testimony
    floater_verify_tol: float = 0.02  # max anchor-PSNR loss (dB) a prune may cost
    floater_verify_depth: int = 4  # bisection levels when verification fails
    prune_floaters: bool = True
    n_sources: int = 3  # nearest captured frames warped into each novel view
    n_ar_context: int = 2  # previously fixed frames used as warp context
    occl_tol: float = 0.05  # relative depth tolerance for surface correspondence
    sweep_mults: tuple = (0.6, 0.75, 0.9, 1.0, 1.15, 1.35, 1.6)  # depth candidates
    sweep_sigma: float = 0.2  # colour-dispersion scale for source agreement
    sweep_prior: float = 0.03  # tie-break preference for the continuation depth
    seeds_per_view: int = 300  # hole seeds unprojected per fixed view (0 disables)
    seed_conf_max: float = 0.3  # seed only where the splat had little to say
    seed_stereo_min: float = 0.6  # ... and >=2 warped sources agree on the depth
    seed_opacity: float = 0.3
    max_seed_frac: float = 0.5  # total seed budget: fraction of current count
    finetune_iters: int = 600
    pseudo_frac: float = 0.25  # fraction of fine-tune steps on fixed novel views
    densify_every: int = 100  # 0 disables densification during fine-tune
    densify_grad_threshold: float = 2e-4
    max_gaussians_headroom: float = 1.3  # fine-tune budget: current count x this
    max_per_tile: int = 1024
    ssim_weight: float = 0.2
    train_size: int = 320  # max image dimension for anchors/novel views
    log_every: int = 50
    seed: int = 0
    device: str = "cpu"


# -- extended trajectory ------------------------------------------------------


def fit_orbit(rec: Reconstruction) -> dict:
    """Fit the plane the captured cameras swept (same math as the turntable).

    Returns the ring centre, in-plane axes, signed normal, mean radius, and a
    robust look-at target inside the point cloud.
    """
    C = rec.camera_centers()
    center = C.mean(axis=0)
    d = C - center
    _, _, vt = np.linalg.svd(d, full_matrices=False)
    e0, e1, normal = vt[0], vt[1], vt[2]
    radius = float(np.linalg.norm(d, axis=1).mean())
    pc = rec.points - np.median(rec.points, axis=0)
    r = np.linalg.norm(pc, axis=1)
    target = np.median(rec.points[r <= np.percentile(r, 80)], axis=0)
    cam_up = np.mean(
        [rec.poses[i][0].T @ np.array([0.0, -1.0, 0.0]) for i in rec.registered],
        axis=0,
    )
    if np.dot(normal, cam_up) < 0:
        normal = -normal
    return {
        "center": center, "e0": e0, "e1": e1, "normal": normal,
        "radius": max(radius, 1e-3), "target": target,
    }


def look_at_pose(eye: np.ndarray, target: np.ndarray, up_hint: np.ndarray):
    """World-to-camera (R, t) looking from ``eye`` at ``target`` (+z forward)."""
    fwd = target - eye
    fwd = fwd / (np.linalg.norm(fwd) + 1e-12)
    right = np.cross(up_hint, fwd)
    if np.linalg.norm(right) < 1e-6:  # looking straight along the up axis
        right = np.cross(up_hint + np.array([0.31, 0.17, 0.23]), fwd)
    right = right / (np.linalg.norm(right) + 1e-12)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    return R, -R @ eye


def extended_trajectory(
    rec: Reconstruction, cfg: ArtifixConfig
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Novel poses sweeping the full orbit *beyond* the captured ring.

    Azimuth covers the whole circle (including arcs the video skipped) while
    elevation and radius oscillate out of the captured plane — precisely the
    directions 3DGS under-observes. Poses are returned in azimuth order: the
    auto-regressive fixer walks them causally, so consecutive views overlap
    and repaired content propagates along the trajectory.
    """
    orbit = fit_orbit(rec)
    poses = []
    for k in range(cfg.n_novel):
        az = 2 * math.pi * k / cfg.n_novel
        # Two elevation periods + one radius period per lap: each azimuth gets
        # visited off-plane, and start/end stay continuous for the AR walk.
        elev = cfg.elev_amp * orbit["radius"] * math.sin(2 * az)
        rad = orbit["radius"] * (1.0 + cfg.radius_amp * math.cos(az))
        eye = (
            orbit["center"]
            + rad * (math.cos(az) * orbit["e0"] + math.sin(az) * orbit["e1"])
            + elev * orbit["normal"]
        )
        poses.append(look_at_pose(eye, orbit["target"], -orbit["normal"]))
    return poses


# -- image-space building blocks ---------------------------------------------


def median_filter(x: torch.Tensor, k: int) -> torch.Tensor:
    """(H, W) median filter with replicate padding (zero-pad would drag the
    median toward 0 at the border, exactly where depth maps end)."""
    if k <= 1:
        return x
    p = k // 2
    xp = F.pad(x[None, None], (p, p, p, p), mode="replicate")
    win = F.unfold(xp, kernel_size=k)  # (1, k*k, H*W)
    return win.median(dim=1).values.reshape(x.shape)


def pull_push(img: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Fill low-weight pixels of ``img`` from coarser levels (pull-push).

    Classic scattered-data interpolation: average valid pixels down to a
    1-pixel pyramid tip, then walk back up, keeping measured content where
    weight is high and coarse fill where it is low. Smooth, seam-free, and
    built only from avg_pool/interpolate — MPS-safe and cheap.
    """
    h, w = img.shape[:2]
    x = (img * weight[..., None]).permute(2, 0, 1)[None]  # (1, C, H, W)
    a = weight[None, None]  # (1, 1, H, W)
    stack = []
    while x.shape[-1] > 1 or x.shape[-2] > 1:
        stack.append((x, a))
        x = F.avg_pool2d(x, 2, ceil_mode=True)
        a = F.avg_pool2d(a, 2, ceil_mode=True)
    fill = x / a.clamp(min=1e-8)
    for x, a in reversed(stack):
        up = F.interpolate(fill, size=x.shape[-2:], mode="bilinear", align_corners=False)
        aw = a.clamp(max=1.0)
        fill = x / a.clamp(min=1e-8) * aw + up * (1.0 - aw)
    return fill[0].permute(1, 2, 0).reshape(h, w, -1).squeeze(-1)


def confidence_from_alpha(alpha: torch.Tensor, cfg: ArtifixConfig) -> torch.Tensor:
    """Map accumulated opacity to [0, 1] trust — ArtiFixer's opacity gate.

    Below ``conf_floor`` the splat has essentially nothing there (a hole);
    above ``conf_ceil`` the reconstruction is well covered. The ramp between
    lets half-covered fringes blend rather than seam.
    """
    return ((alpha - cfg.conf_floor) / (cfg.conf_ceil - cfg.conf_floor)).clamp(0.0, 1.0)


def floater_mask(
    median_depth: torch.Tensor, alpha: torch.Tensor, cfg: ArtifixConfig
) -> torch.Tensor:
    """Pixels whose surface sits far in front of the local neighbourhood.

    A floater is a small opaque blob hanging between the camera and the real
    surface: its median depth undercuts the local median sharply. Returns a
    float mask (1 = floater) used to *revoke* confidence so the filler
    overwrites those pixels — turning artifacts back into fillable holes.
    """
    local = median_filter(median_depth, cfg.floater_kernel)
    m = (median_depth > 0) & (local > 0) & (median_depth < cfg.floater_ratio * local)
    return (m & (alpha > 0.3)).float()


def punchthrough_mask(
    median_depth: torch.Tensor, cfg: ArtifixConfig
) -> torch.Tensor:
    """Pixels seeing far *behind* the local surface — holes in that surface.

    The dual of the floater test, and the signature of an incomplete capture:
    where a wall is missing, rays punch through to whatever lies beyond (the
    shell's far side, background clutter), which still accumulates opacity —
    so the opacity gate alone would happily trust it. A depth that jumps well
    behind the local median marks those pixels so their confidence is revoked
    and the filler replaces them with warped real content.
    """
    local = median_filter(median_depth, cfg.floater_kernel)
    m = (local > 0) & (median_depth > local / cfg.floater_ratio)
    return m.float()


# -- cross-view warping (the geometric "generator") ---------------------------


def warp_source(
    src_rgb: torch.Tensor,
    src_depth: torch.Tensor,
    src_conf: torch.Tensor,
    cam_src: tuple,
    cam_dst: tuple,
    dst_depth: torch.Tensor,
    occl_tol: float,
    near: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reproject ``src_rgb`` into the destination view via its depth.

    Back-projects every destination pixel with the (completed) destination
    depth, projects into the source camera, and samples colour where the
    source actually saw that 3D point: in front of the camera, on screen, and
    — the occlusion test — at a source depth agreeing with the reprojected
    one (otherwise the source is looking at a different, nearer surface).
    Returns (warped RGB, per-pixel weight in [0, 1]). Same conventions as
    ``losses.temporal_warp_loss``: pixel centres at +0.5, align_corners=False.
    """
    Rd, td, fd, cxd, cyd = cam_dst
    Rs, ts, fs, cxs, cys = cam_src
    h, w = dst_depth.shape
    dev = dst_depth.device
    ys, xs = torch.meshgrid(
        torch.arange(h, device=dev, dtype=torch.float32),
        torch.arange(w, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    z = dst_depth
    xd = torch.stack([(xs + 0.5 - cxd) / fd * z, (ys + 0.5 - cyd) / fd * z, z], dim=-1)
    xw = (xd - td) @ Rd  # row-vector R^T (x - t)
    xc = xw @ Rs.T + ts
    zs = xc[..., 2]
    us = fs * xc[..., 0] / zs.clamp(min=1e-6) + cxs
    vs = fs * xc[..., 1] / zs.clamp(min=1e-6) + cys
    sh, sw = src_depth.shape
    valid = (z > near) & (zs > near) & (us >= 0) & (us <= sw) & (vs >= 0) & (vs <= sh)

    grid = torch.stack([2 * us / sw - 1, 2 * vs / sh - 1], dim=-1)[None]
    samp = lambda m, mode: F.grid_sample(  # noqa: E731
        m[None], grid, mode=mode, align_corners=False, padding_mode="zeros"
    )[0]
    d_src = samp(src_depth[None], "nearest")[0]
    rgb = samp(src_rgb.permute(2, 0, 1), "bilinear").permute(1, 2, 0)
    conf = samp(src_conf[None], "bilinear")[0]
    # Correspondence test: the colour of a source pixel belongs to the surface
    # the source actually observed (its depth), so the sample is valid only
    # when the reprojected point lies ON that surface. Accepting points in
    # front of it would smear surface colours onto empty space; points behind
    # it are occluded. Both directions must reject.
    same_surface = (d_src > 0) & ((zs - d_src).abs() < occl_tol * d_src.clamp(min=near))
    weight = valid.float() * same_surface.float() * conf
    return rgb, weight


def _complete_depth(median_depth: torch.Tensor, trust: torch.Tensor) -> torch.Tensor:
    """Extend the surface depth of *trusted* pixels across the rest (pull-push).

    Warping needs some depth everywhere; the smooth continuation of the
    trusted surface is the geometrically plausible proxy. Artifact pixels
    (floaters, punch-throughs) must not contribute — their depth is exactly
    what is wrong with them — so ``trust`` is the post-revocation confidence.
    """
    wt = (median_depth > 0).float() * trust
    if float(wt.sum()) < 1.0:  # nothing trustworthy rendered — nothing to extend
        return median_depth
    return pull_push(median_depth[..., None], wt).clamp(min=0.0)


def sweep_fill(
    d0: torch.Tensor,
    cam_dst: tuple,
    sources: list[dict],
    cfg: ArtifixConfig,
) -> dict:
    """Plane-sweep the sources over candidate depths around the continuation.

    The completed depth is only a guess inside holes; warping with a wrong
    depth samples the wrong source pixels and fills holes with misplaced
    content. So, per pixel, try several multiples of the continuation depth
    and keep the one where independently warped sources photometrically agree
    — one-pixel multi-view stereo, which both corrects the fill colour and
    yields a *verified* depth to unproject hole seeds at. Returns per-pixel
    ``rgb`` / ``cover`` (fill trust) / ``depth`` (best candidate) / ``stereo``
    (multi-source agreement — the seeding gate).
    """
    dev = d0.device
    h, w = d0.shape
    inf = torch.full((h, w), float("inf"), device=dev)
    best = {
        "score": inf, "rgb": torch.zeros(h, w, 3, device=dev),
        "n_eff": torch.zeros(h, w, device=dev), "depth": d0.clone(),
        "disp": torch.zeros(h, w, device=dev),
    }
    for mult in cfg.sweep_mults:
        dk = d0 * mult
        warped = [
            warp_source(s["rgb"], s["depth"], s["conf"], s["cam"], cam_dst, dk, cfg.occl_tol)
            for s in sources
        ]
        n_eff = torch.stack([wgt for _, wgt in warped]).sum(dim=0)
        mu = torch.stack([rgb * wgt[..., None] for rgb, wgt in warped]).sum(dim=0)
        mu = mu / n_eff.clamp(min=1e-8)[..., None]
        disp = torch.stack(
            [(rgb - mu).abs().mean(dim=-1) * wgt for rgb, wgt in warped]
        ).sum(dim=0) / n_eff.clamp(min=1e-8)
        score = torch.where(
            n_eff > 0.1, disp + cfg.sweep_prior * abs(math.log(mult)), inf
        )
        take = score < best["score"]
        best["score"] = torch.where(take, score, best["score"])
        best["rgb"] = torch.where(take[..., None], mu, best["rgb"])
        best["n_eff"] = torch.where(take, n_eff, best["n_eff"])
        best["depth"] = torch.where(take, dk, best["depth"])
        best["disp"] = torch.where(take, disp, best["disp"])
    agree = torch.exp(-best["disp"] / cfg.sweep_sigma)
    # A single source trivially "agrees" with itself — and single-source
    # correspondences include every near-grazing accidental hit (a ray passing
    # within tolerance of some surface it doesn't belong to). Only content two
    # independent sources verify is worth importing, so coverage itself is
    # gated at n_eff > 1.
    stereo = (best["n_eff"] - 1.0).clamp(0.0, 1.0) * agree
    return {
        "rgb": best["rgb"],
        "cover": stereo,
        "depth": best["depth"],
        "stereo": stereo,
    }


# -- per-view fixing ----------------------------------------------------------


def fix_view(
    render_rgb: torch.Tensor,
    alpha: torch.Tensor,
    median_depth: torch.Tensor,
    cam_dst: tuple,
    sources: list[dict],
    cfg: ArtifixConfig,
) -> dict:
    """Repair one novel render: opacity-gated mix of trusted content and fill.

    ``sources`` are dicts with keys rgb/depth/conf/cam — nearby captured
    frames plus the auto-regressive context of previously fixed frames.
    Mirrors ArtiFixer's opacity mixing: where the splat is confident the
    render passes through untouched; where it is not, warped real content
    replaces it (and where nothing is known, the render also passes through,
    with zero supervision weight). Returns a dict: ``fixed`` image, per-pixel
    supervision ``weight``, post-revocation ``conf``, warp ``cover``, the
    warp ``depth``, and the ``stereo`` depth-verification map.
    """
    artifact = torch.maximum(
        floater_mask(median_depth, alpha, cfg), punchthrough_mask(median_depth, cfg)
    )
    conf = confidence_from_alpha(alpha, cfg) * (1.0 - artifact)
    dst_depth = _complete_depth(median_depth, conf.clamp(min=0.0))

    sweep = sweep_fill(dst_depth, cam_dst, sources, cfg)
    cover = sweep["cover"]
    merged = sweep["rgb"]

    # Replace exactly where distrust meets verified imported content, and
    # nowhere else: ``sup = (1-conf) * cover`` is both the blend factor and
    # the supervision weight. Where the splat was already confident the
    # target equals its own (pre-fine-tune) render — supervising there would
    # only drag the model back toward a stale snapshot of itself (measured to
    # cost more novel-view PSNR than it gains). Where nothing is known from
    # anywhere (cover ~ 0: true background / never-observed), the render
    # passes through and the zero weight keeps those pixels out of the loss —
    # the fixer imports real observations; it does not hallucinate.
    sup = (1.0 - conf) * cover
    fixed = render_rgb * (1.0 - sup)[..., None] + merged * sup[..., None]
    fixed = fixed.clamp(0.0, 1.0)
    # Where the sweep verified a depth, carry it (it corrects the continuation
    # guess); elsewhere keep the continuation.
    depth = torch.where(sweep["stereo"] > 0.25, sweep["depth"], dst_depth)
    return {
        "fixed": fixed, "weight": sup, "conf": conf,
        "cover": cover, "depth": depth, "stereo": sweep["stereo"],
    }


# -- hole seeding --------------------------------------------------------------


def collect_seeds(fix: dict, cam: tuple, cfg: ArtifixConfig, extent: float) -> dict:
    """Unproject hole pixels of a fixed view into world-space gaussian seeds.

    Densification cannot grow geometry where none exists (it only clones and
    splits), so pixels the splat could not explain — but the warp filled with
    real content — are lifted to 3D at the completed depth and later appended
    as new gaussians for the fine-tune to refine. Returns xyz/rgb/radius
    tensors (possibly empty).
    """
    R, t, f, cx, cy = cam
    conf, stereo, depth = fix["conf"], fix["stereo"], fix["depth"]
    h, w = depth.shape
    dev = depth.device
    mask = (
        (conf < cfg.seed_conf_max) & (stereo > cfg.seed_stereo_min)
        & (depth > 0.05) & (depth < 4.0 * extent)
    )
    idx = torch.nonzero(mask)
    empty = {
        "xyz": torch.zeros(0, 3, device=dev), "rgb": torch.zeros(0, 3, device=dev),
        "radius": torch.zeros(0, device=dev),
    }
    if idx.shape[0] == 0 or cfg.seeds_per_view <= 0:
        return empty
    stride = max(1, idx.shape[0] // cfg.seeds_per_view)
    idx = idx[::stride]
    ys, xs = idx[:, 0].float(), idx[:, 1].float()
    z = depth[idx[:, 0], idx[:, 1]]
    xc = torch.stack([(xs + 0.5 - cx) / f * z, (ys + 0.5 - cy) / f * z, z], dim=-1)
    xyz = (xc - t) @ R  # row-vector R^T (x - t)
    return {
        "xyz": xyz,
        "rgb": fix["fixed"][idx[:, 0], idx[:, 1]],
        # ~2-pixel world footprint at that depth: visible but not blobby.
        "radius": 2.0 * z / f,
    }


def _dedup_seeds(
    seeds: list[dict], model_xyz: torch.Tensor, cell: float, budget: int
) -> dict:
    """Voxel-hash dedup of per-view seeds against the model and each other.

    Consecutive fixed views overlap heavily; without this, the same missing
    wall gets seeded a dozen times. First seed to claim a cell wins; cells
    already containing a model gaussian are off limits.
    """
    dev = model_xyz.device
    occupied = {tuple(v) for v in torch.floor(model_xyz / cell).long().cpu().numpy()}
    out_xyz, out_rgb, out_rad = [], [], []
    n = 0
    for s in seeds:
        if n >= budget or s["xyz"].shape[0] == 0:
            continue
        cells = torch.floor(s["xyz"] / cell).long().cpu().numpy()
        for i, c in enumerate(map(tuple, cells)):
            if n >= budget or c in occupied:
                continue
            occupied.add(c)
            out_xyz.append(s["xyz"][i])
            out_rgb.append(s["rgb"][i])
            out_rad.append(s["radius"][i])
            n += 1
    if not out_xyz:
        return {
            "xyz": torch.zeros(0, 3, device=dev), "rgb": torch.zeros(0, 3, device=dev),
            "radius": torch.zeros(0, device=dev),
        }
    return {
        "xyz": torch.stack(out_xyz), "rgb": torch.stack(out_rgb),
        "radius": torch.stack(out_rad),
    }


# -- floater pruning (novel-view depth-consistency vote) ----------------------


def _muted_anchor_psnr(
    model: GaussianModel, anchors: list[TrainView], mute: torch.Tensor | None
) -> float:
    """Anchor PSNR with the ``mute``d gaussians rendered fully transparent."""
    saved = None
    if mute is not None and int(mute.sum()):
        saved = model.opacity.data.clone()
        model.opacity.data[mute] = -12.0  # sigmoid ~ 6e-6: invisible
    try:
        return _anchor_psnr(model, anchors, max_views=6)
    finally:
        if saved is not None:
            model.opacity.data = saved


def _verify_prune(
    model: GaussianModel,
    anchors: list[TrainView],
    candidates: torch.Tensor,
    cfg: ArtifixConfig,
) -> torch.Tensor:
    """Keep only candidates whose removal does not cost captured-view quality.

    A floater is, by definition, geometry the captured views do not need —
    so muting it must leave anchor PSNR intact (usually it improves). The
    novel-view vote alone cannot promise that: on sparse scenes it also
    flags legitimate isolated bits. Candidates are muted as a group; if the
    anchors object, the group is bisected and each half retried (cumulatively
    against everything already accepted), so a few bad apples cannot spoil a
    mostly-correct prune. Returns the verified boolean mask.
    """
    accepted = torch.zeros_like(candidates)
    idx = torch.nonzero(candidates).squeeze(1)
    if idx.numel() == 0:
        return accepted
    base = _muted_anchor_psnr(model, anchors, None)
    queue = [(idx, 0)]
    with torch.no_grad():
        while queue:
            group, depth = queue.pop()
            trial = accepted.clone()
            trial[group] = True
            p = _muted_anchor_psnr(model, anchors, trial)
            if p >= base - cfg.floater_verify_tol:
                accepted = trial
                base = max(base, p)  # removing junk may raise the bar
            elif depth < cfg.floater_verify_depth and group.numel() > 1:
                half = group.numel() // 2
                queue.append((group[:half], depth + 1))
                queue.append((group[half:], depth + 1))
    return accepted


def prune_floaters(
    model: GaussianModel,
    novel_poses: list[tuple[np.ndarray, np.ndarray]],
    focal: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    cfg: ArtifixConfig,
    anchors: list[TrainView] | None = None,
) -> int:
    """Remove gaussians that novel views expose as floating junk.

    From each novel pose, every opaque gaussian is tested against the depth
    on a ring just outside its own projected footprint (see the inline
    comment for why probing *underneath* it would let it mask itself); a
    gaussian hanging in front of the surrounding surface, or sitting alone
    in the void, accumulates votes across poses. The captured views cannot
    cast these votes — from them, floaters sit *on* a surface (that is why
    they exist); only novel viewpoints reveal the disagreement, which is the
    ArtiFixer observation driving this whole module. When ``anchors`` are
    given, the vote's verdict is verified against them before anything is
    deleted (see ``_verify_prune``).
    """
    from .render import compute_cov3d, project_gaussians

    dev = model.xyz.device
    votes = torch.zeros(model.num_gaussians, device=dev)
    votes_emb = torch.zeros(model.num_gaussians, device=dev)
    seen = torch.zeros(model.num_gaussians, device=dev)
    n_dirs = 8
    ang = torch.arange(n_dirs, device=dev, dtype=torch.float32) * (2 * math.pi / n_dirs)
    ring_dir = torch.stack([torch.cos(ang), torch.sin(ang)])  # (2, D)
    with torch.no_grad():
        cov3d = compute_cov3d(model.get_scale(), model.get_quat())
        opaque = model.get_opacity()[:, 0] > 0.3
        for R, t in novel_poses:
            Rt = torch.tensor(R, dtype=torch.float32, device=dev)
            tt = torch.tensor(t, dtype=torch.float32, device=dev)
            _, info = render_model(
                model, Rt, tt, focal, cx, cy, width, height,
                max_per_tile=cfg.max_per_tile, return_aux=True,
            )
            surf = median_filter(info.median_depth, 3)
            means2d, cov2d, z, in_front = project_gaussians(
                model.xyz, cov3d, Rt, tt, focal, cx, cy
            )
            # A gaussian would mask its own detection: right under an opaque
            # blob the rendered surface IS the blob. So probe the depth on a
            # ring just outside its projected footprint — a floater hangs in
            # front of consistently farther surface all the way around, while
            # a gaussian ON a surface (or at a silhouette) has ring samples
            # at its own depth and stays unflagged.
            a, b, c = cov2d[:, 0, 0], cov2d[:, 0, 1], cov2d[:, 1, 1]
            mid = 0.5 * (a + c)
            det = (a * c - b * b).clamp(min=1e-12)
            eig = mid + torch.sqrt((mid * mid - det).clamp(min=1e-12))
            rad = 3.0 * torch.sqrt(eig) + 3.0  # footprint + margin, pixels
            ru = (means2d[:, 0, None] + rad[:, None] * ring_dir[0][None]).round()
            rv = (means2d[:, 1, None] + rad[:, None] * ring_dir[1][None]).round()
            inb = (ru >= 0) & (ru < width) & (rv >= 0) & (rv < height)
            ring = surf[
                rv.clamp(0, height - 1).long(), ru.clamp(0, width - 1).long()
            ]  # (N, D)
            nz = inb & (ring > 0)
            k_surf = nz.sum(dim=1)
            k_ring = inb.sum(dim=1)
            ring_min = torch.where(nz, ring, torch.full_like(ring, float("inf")))
            ring_min = ring_min.min(dim=1).values
            on = (
                in_front & opaque
                & (means2d[:, 0] >= 0) & (means2d[:, 0] < width)
                & (means2d[:, 1] >= 0) & (means2d[:, 1] < height)
            )
            # Two ways a view can testify: *embedded* — the ring lands on
            # rendered surface, and this gaussian undercuts it (hangs in
            # front); or *isolated* — a small blob with (near-)nothing around
            # it, the classic bit-of-junk-in-the-void. Views where the ring is
            # part surface / part void (silhouettes) abstain.
            embedded = on & (k_surf >= n_dirs - 2)
            isolated = (
                on & (k_ring >= n_dirs - 1) & (k_surf <= 2)
                & (rad < 0.12 * width)
            )
            flag_emb = embedded & (z < cfg.floater_ratio * ring_min)
            seen += (embedded | isolated).float()
            votes += (flag_emb | isolated).float()
            votes_emb += flag_emb.float()
        frac = votes / seen.clamp(min=1.0)
        # Depth-undercut (embedded) evidence is specific, so its vote frac
        # suffices; isolation alone is circumstantial (sparse scenes have
        # legitimately isolated bits) and must be near-unanimous.
        drop = (seen >= cfg.floater_min_views) & (
            ((votes_emb > 0) & (frac >= cfg.floater_vote_frac)) | (frac >= 0.8)
        )
        n_cand = int(drop.sum())
        if anchors is not None and n_cand:
            drop = _verify_prune(model, anchors, drop, cfg)
        removed = model.prune_by_mask(~drop)
    if removed or n_cand:
        log.info(
            "Artifix: pruned %d floater gaussians (%d candidates, %d total)",
            removed, n_cand, int(seen.numel()),
        )
    return removed


# -- pseudo-supervised fine-tune ----------------------------------------------


def _render_anchor_views(
    model: GaussianModel, rec: Reconstruction, train_size: int, device: str
) -> list[TrainView]:
    """Self-anchors when the captured frames are unavailable (standalone repair
    of an exported scene): the model's own renders at the captured poses pin
    down everything the video did observe while the pseudo-views fill the rest.
    """
    s = min(1.0, train_size / max(rec.width, rec.height))
    w, h = int(round(rec.width * s)), int(round(rec.height * s))
    f, cx, cy = rec.focal * s, rec.cx * s, rec.cy * s
    views = []
    with torch.no_grad():
        for fi in rec.registered:
            R, t = rec.poses[fi]
            Rt = torch.tensor(R, dtype=torch.float32, device=device)
            tt = torch.tensor(t, dtype=torch.float32, device=device)
            img, _ = render_model(model, Rt, tt, f, cx, cy, w, h)
            views.append(TrainView(img.detach(), Rt, tt, f, cx, cy, w, h))
    return views


def finetune(
    model: GaussianModel,
    anchor_views: list[TrainView],
    pseudo_views: list[dict],
    extent: float,
    cfg: ArtifixConfig,
    progress_cb=None,
) -> None:
    """Optimize the gaussians on captured anchors + confidence-weighted fixes.

    The same loop shape as ``train.train`` (Adam per parameter group, xyz LR
    decay, non-finite guards, periodic densify with optimizer rebuild) but
    over a two-pool sampler. Anchors keep the well-observed scene pinned with
    the standard L1+SSIM; pseudo views pull holes toward their fixed content
    with per-pixel-weighted L1, so hallucinated pixels whisper while
    geometry-backed ones speak at full volume.
    """
    lr = TrainConfig(device=cfg.device)  # trainer's learning rates
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    opt = make_optimizer(model, lr, extent)
    max_g = int(model.num_gaussians * cfg.max_gaussians_headroom)
    densify_until = int(0.7 * cfg.finetune_iters)
    bg = torch.zeros(3, device=cfg.device)
    recent: list[float] = []
    n_bad = 0

    for it in range(1, cfg.finetune_iters + 1):
        use_pseudo = pseudo_views and rng.random() < cfg.pseudo_frac
        if use_pseudo:
            pv = pseudo_views[int(rng.integers(len(pseudo_views)))]
            R, t, f, cx, cy = pv["cam"]
            h, w = pv["rgb"].shape[:2]
            pred, info = render_model(
                model, R, t, f, cx, cy, w, h, bg=bg, max_per_tile=cfg.max_per_tile
            )
            wgt = pv["weight"][..., None]
            loss = ((pred - pv["rgb"]).abs() * wgt).sum() / (wgt.sum() * 3 + 1e-8)
        else:
            view = anchor_views[int(rng.integers(len(anchor_views)))]
            pred, info = render_model(
                model, view.R, view.t, view.focal, view.cx, view.cy,
                view.width, view.height, bg=bg, max_per_tile=cfg.max_per_tile,
            )
            loss = image_loss(pred, view.image, cfg.ssim_weight)

        if not torch.isfinite(loss):
            n_bad += 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if info.means2d.grad is not None:
            g = info.means2d.grad
            hw = pred.shape[:2]
            ndc = torch.stack([g[:, 0] * (hw[1] / 2.0), g[:, 1] * (hw[0] / 2.0)], dim=-1)
            model.accumulate_grads(ndc.norm(dim=-1).detach(), info.visible)
        if not all(
            torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None
        ):
            n_bad += 1
            opt.zero_grad(set_to_none=True)
            continue
        opt.step()
        opt.param_groups[0]["lr"] = lr.lr_xyz * extent * (0.01 ** (it / cfg.finetune_iters))

        with torch.no_grad():
            if not use_pseudo:
                recent.append(psnr(pred, view.image))
                if len(recent) > 50:
                    recent.pop(0)

        if cfg.densify_every > 0 and it <= densify_until and it % cfg.densify_every == 0:
            model.densify_and_prune(
                cfg.densify_grad_threshold, extent, max_gaussians=max_g
            )
            opt = make_optimizer(model, lr, extent)

        if it % cfg.log_every == 0 or it == cfg.finetune_iters:
            log.info(
                "Artifix fine-tune %d/%d  loss %.4f  anchor-psnr %.2f dB  gaussians %d",
                it, cfg.finetune_iters, float(loss.detach()),
                float(np.mean(recent)) if recent else float("nan"), model.num_gaussians,
            )
            if progress_cb:
                progress_cb(it, cfg.finetune_iters,
                            float(np.mean(recent)) if recent else 0.0, model.num_gaussians)
    if n_bad:
        log.warning("Artifix fine-tune skipped %d non-finite steps", n_bad)
    model.prune_transparent(0.005)


# -- the full pass -------------------------------------------------------------


def artifix(
    model: GaussianModel,
    rec: Reconstruction,
    images: list[np.ndarray] | None = None,
    cfg: ArtifixConfig | None = None,
    progress_cb=None,
) -> dict:
    """Repair an incomplete/artifacted splat in place; returns a report dict.

    The ArtiFixer loop: prune floaters exposed by novel viewpoints, walk an
    extended trajectory auto-regressively fixing each render via opacity-
    gated mixing, then fine-tune the gaussians on captured anchors + the
    fixed views. ``images`` are the extracted video frames (BGR uint8, indexed
    like ``rec.registered``); without them the model's own renders anchor the
    fine-tune, which still completes holes but cannot sharpen captured views.
    """
    cfg = cfg or ArtifixConfig()
    dev = cfg.device
    train_size = cfg.train_size
    if images is not None:
        anchors = build_views(rec, images, train_size, dev)
    else:
        anchors = _render_anchor_views(model, rec, train_size, dev)
    if not anchors:
        raise ValueError("artifix needs at least one registered camera")
    f, cx, cy = anchors[0].focal, anchors[0].cx, anchors[0].cy
    w, h = anchors[0].width, anchors[0].height
    extent = rec.scene_extent()
    n0 = model.num_gaussians

    novel = extended_trajectory(rec, cfg)

    # 1) Kill floaters first so the novel renders we are about to trust (and
    #    their depth maps, which drive the warps) are as clean as possible.
    removed = (
        prune_floaters(model, novel, f, cx, cy, w, h, cfg, anchors=anchors)
        if cfg.prune_floaters
        else 0
    )

    # 2) Per-anchor depth/conf for the warp sources (rendered once, no grad).
    src_bank = []
    anchor_centers = []
    with torch.no_grad():
        for v in anchors:
            _, info = render_model(
                model, v.R, v.t, v.focal, v.cx, v.cy, v.width, v.height,
                max_per_tile=cfg.max_per_tile, return_aux=True,
            )
            src_bank.append({
                "rgb": v.image, "depth": info.median_depth,
                "conf": confidence_from_alpha(info.alpha, cfg),
                "cam": (v.R, v.t, v.focal, v.cx, v.cy),
            })
            anchor_centers.append((-v.R.T @ v.t).cpu().numpy())
    anchor_centers = np.stack(anchor_centers)

    # 3) Auto-regressive fixing walk along the trajectory: each fixed frame
    #    joins the warp context for the next, and its unexplained-but-filled
    #    pixels are banked as 3D seeds.
    pseudo_views: list[dict] = []
    ar_context: list[dict] = []
    seed_bank: list[dict] = []
    alpha_before = []
    with torch.no_grad():
        for R, t in novel:
            Rt = torch.tensor(R, dtype=torch.float32, device=dev)
            tt = torch.tensor(t, dtype=torch.float32, device=dev)
            cam = (Rt, tt, f, cx, cy)
            rgb, info = render_model(
                model, Rt, tt, f, cx, cy, w, h,
                max_per_tile=cfg.max_per_tile, return_aux=True,
            )
            alpha_before.append(float(info.alpha.mean()))
            center = -R.T @ t
            near_idx = np.argsort(np.linalg.norm(anchor_centers - center, axis=1))
            sources = [src_bank[i] for i in near_idx[: cfg.n_sources]]
            sources += ar_context[-cfg.n_ar_context:] if cfg.n_ar_context else []
            fix = fix_view(rgb, info.alpha, info.median_depth, cam, sources, cfg)
            # Views the fixer barely touched carry no training signal — keep
            # them out of the bank so fine-tune steps aren't spent on no-ops.
            if float(fix["weight"].mean()) > 0.005:
                pseudo_views.append({"cam": cam, "rgb": fix["fixed"], "weight": fix["weight"]})
            ar_context.append({
                "rgb": fix["fixed"], "depth": fix["depth"], "conf": fix["weight"],
                "cam": cam,
            })
            seed_bank.append(collect_seeds(fix, cam, cfg, extent))

    # Plant deduplicated seeds so the fine-tune has geometry to refine inside
    # the holes (densification alone cannot create it there).
    seeds = _dedup_seeds(
        seed_bank, model.xyz.data, cell=extent / 64.0,
        budget=int(cfg.max_seed_frac * model.num_gaussians),
    )
    n_seeded = seeds["xyz"].shape[0]
    if n_seeded:
        model.append_gaussians(
            seeds["xyz"], seeds["rgb"], seeds["radius"], init_opacity=cfg.seed_opacity
        )
    log.info(
        "Artifix: fixed %d novel views (mean coverage %.2f), planted %d hole seeds",
        len(pseudo_views), float(np.mean(alpha_before)), n_seeded,
    )

    # 4) Fine-tune on anchors + fixed views.
    psnr_before = _anchor_psnr(model, anchors)
    finetune(model, anchors, pseudo_views, extent, cfg, progress_cb)

    # 5) Sweep up after ourselves: densification into fill regions and badly
    #    placed seeds can leave fresh stragglers — one more verified prune.
    removed_post = (
        prune_floaters(model, novel, f, cx, cy, w, h, cfg, anchors=anchors)
        if cfg.prune_floaters
        else 0
    )
    psnr_after = _anchor_psnr(model, anchors)

    alpha_after = []
    with torch.no_grad():
        for R, t in novel:
            Rt = torch.tensor(R, dtype=torch.float32, device=dev)
            tt = torch.tensor(t, dtype=torch.float32, device=dev)
            _, info = render_model(
                model, Rt, tt, f, cx, cy, w, h,
                max_per_tile=cfg.max_per_tile, return_aux=True,
            )
            alpha_after.append(float(info.alpha.mean()))

    report = {
        "floaters_pruned": removed + removed_post,
        "hole_seeds": n_seeded,
        "novel_views": len(pseudo_views),
        "coverage_before": float(np.mean(alpha_before)),
        "coverage_after": float(np.mean(alpha_after)),
        "anchor_psnr_before": psnr_before,
        "anchor_psnr_after": psnr_after,
        "gaussians_before": n0,
        "gaussians_after": model.num_gaussians,
    }
    log.info(
        "Artifix done: novel-view coverage %.2f -> %.2f, anchor PSNR %.2f -> %.2f dB, "
        "%d -> %d gaussians",
        report["coverage_before"], report["coverage_after"],
        psnr_before, psnr_after, n0, model.num_gaussians,
    )
    return report


def _anchor_psnr(model: GaussianModel, views: list[TrainView], max_views: int = 8) -> float:
    step = max(1, len(views) // max_views)
    vals = []
    with torch.no_grad():
        for v in views[::step]:
            pred, _ = render_model(
                model, v.R, v.t, v.focal, v.cx, v.cy, v.width, v.height
            )
            vals.append(psnr(pred, v.image))
    return float(np.mean(vals))


# -- view-prior adapter --------------------------------------------------------


class SplatRepairPrior(ViewPrior):
    """The artifix fixer behind the ``ViewPrior`` interface (see view_prior.py).

    Lets ``train_neural``'s pseudo-view machinery use opacity-gated multi-view
    repair as its generative prior — and marks the exact seam where a learned
    generator (ArtiFixer's video diffusion model, or any successor) would plug
    in: replace ``__call__``, keep the contract.
    """

    def __init__(self, model: GaussianModel, sources: list[dict], cfg: ArtifixConfig):
        self.model = model
        self.sources = sources
        self.cfg = cfg

    def __call__(self, image: torch.Tensor, cam=None) -> torch.Tensor:
        if cam is None:
            return image.detach()
        R, t, f, cx, cy = cam
        h, w = image.shape[:2]
        with torch.no_grad():
            _, info = render_model(
                self.model, R, t, f, cx, cy, w, h,
                max_per_tile=self.cfg.max_per_tile, return_aux=True,
            )
            fix = fix_view(
                image.detach(), info.alpha, info.median_depth, cam, self.sources, self.cfg
            )
        return fix["fixed"]
