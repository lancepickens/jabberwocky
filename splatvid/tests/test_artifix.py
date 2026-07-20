"""Artifix tests: opacity-gated repair, floater pruning, hole filling.

The synthetic scene is its own ground truth: every "captured frame" and every
target a test compares against is rendered from the known gaussian scene, so
repairs are measured against exact GT rather than eyeballed.
"""

import math

import numpy as np
import pytest
import torch

from splatvid import artifix as ax
from splatvid.losses import psnr
from splatvid.model import GaussianModel
from splatvid.render import render, render_model
from splatvid.sfm import Reconstruction
from splatvid.synthetic import make_scene, orbit_pose

W, H = 96, 72
F = 1.1 * W
WEDGE0, WEDGE1 = 0.3, 0.3 + 100 * math.pi / 180


def _gt_scene(n=900, seed=3):
    return make_scene(n=n, seed=seed)


def _gt_render(sc, R, t, aux=False):
    img, info = render(
        sc["xyz"], sc["scale"], sc["quat"], sc["rgb"], sc["opacity"],
        torch.tensor(R, dtype=torch.float32), torch.tensor(t, dtype=torch.float32),
        F, W / 2, H / 2, W, H, return_aux=aux,
    )
    return (img, info) if aux else img


def _degraded_model(sc):
    """The GT scene minus a 100-degree azimuth wedge — an incomplete capture."""
    xyz = sc["xyz"].numpy()
    az = np.arctan2(xyz[:, 2], xyz[:, 0])
    keep = ~((az > WEDGE0) & (az < WEDGE1))
    m = GaussianModel(xyz[keep], sc["rgb"].numpy()[keep])
    with torch.no_grad():
        m.log_scale.data = torch.log(sc["scale"][keep])
        m.quat.data = sc["quat"][keep]
        m.opacity.data.fill_(6.0)
    return m


def _reconstruction(sc, n_cap=8, arc=(0.0, 2 * math.pi)):
    poses = {
        i: orbit_pose(arc[0] + (arc[1] - arc[0]) * i / n_cap, radius=2.6)
        for i in range(n_cap)
    }
    xyz = sc["xyz"].numpy()
    return Reconstruction(
        focal=F, cx=W / 2, cy=H / 2, width=W, height=H, poses=poses,
        points=xyz, point_colors=sc["rgb"].numpy(),
        point_errors=np.zeros(len(xyz)), registered=list(range(n_cap)),
    )


def _frames(sc, rec):
    out = []
    for i in rec.registered:
        img = _gt_render(sc, *rec.poses[i])
        out.append((img.numpy()[:, :, ::-1] * 255).astype(np.uint8))
    return out


# -- trajectory ----------------------------------------------------------------


def test_extended_trajectory_valid_poses():
    sc = _gt_scene(n=300)
    rec = _reconstruction(sc)
    cfg = ax.ArtifixConfig(n_novel=12)
    poses = ax.extended_trajectory(rec, cfg)
    assert len(poses) == 12
    orbit = ax.fit_orbit(rec)
    prev_c = None
    for R, t in poses:
        assert np.allclose(R @ R.T, np.eye(3), atol=1e-5)
        assert np.linalg.det(R) == pytest.approx(1.0, abs=1e-5)
        # Looks at the target: it projects in front, near the image centre.
        pc = R @ orbit["target"] + t
        assert pc[2] > 0
        u = F * pc[0] / pc[2] + W / 2
        v = F * pc[1] / pc[2] + H / 2
        assert 0 <= u <= W and 0 <= v <= H
        # Causal walk: consecutive cameras stay close (AR context overlaps).
        c = -R.T @ t
        if prev_c is not None:
            assert np.linalg.norm(c - prev_c) < orbit["radius"]
        prev_c = c


# -- masks and confidence ------------------------------------------------------


def test_confidence_from_alpha_ramp():
    cfg = ax.ArtifixConfig()
    alpha = torch.tensor([[0.0, cfg.conf_floor, 0.5, cfg.conf_ceil, 1.0]])
    conf = ax.confidence_from_alpha(alpha, cfg)
    assert conf[0, 0] == 0.0 and conf[0, 1] == 0.0
    assert 0.0 < conf[0, 2] < 1.0
    assert conf[0, 3] == 1.0 and conf[0, 4] == 1.0


def test_floater_and_punchthrough_masks():
    cfg = ax.ArtifixConfig()
    md = torch.full((40, 40), 2.0)
    md[10:14, 10:14] = 0.8  # blob well in front of the local surface
    md[25:29, 25:29] = 5.0  # rays punching far behind it (a hole)
    alpha = torch.ones(40, 40)
    fl = ax.floater_mask(md, alpha, cfg)
    pt = ax.punchthrough_mask(md, cfg)
    assert fl[11, 11] == 1.0 and fl[20, 20] == 0.0
    assert pt[26, 26] == 1.0 and pt[20, 20] == 0.0
    assert fl[26, 26] == 0.0 and pt[11, 11] == 0.0


def test_pull_push_fills_holes():
    img = torch.full((33, 47, 3), 0.7)
    wt = torch.ones(33, 47)
    wt[10:20, 15:30] = 0.0  # hole
    filled = ax.pull_push(img, wt)
    assert torch.allclose(filled, torch.full_like(filled, 0.7), atol=1e-4)
    # Known pixels are preserved even with non-constant content.
    img2 = img.clone()
    img2[:, :10] = 0.2
    filled2 = ax.pull_push(img2, wt)
    assert torch.allclose(filled2[:, :5], img2[:, :5], atol=1e-4)
    assert torch.isfinite(filled2).all()


# -- warping -------------------------------------------------------------------


def test_warp_source_identity():
    sc = _gt_scene(n=500)
    R, t = orbit_pose(1.0, radius=2.6)
    img, info = _gt_render(sc, R, t, aux=True)
    cam = (
        torch.tensor(R, dtype=torch.float32), torch.tensor(t, dtype=torch.float32),
        F, W / 2, H / 2,
    )
    conf = torch.ones(H, W)
    warped, wgt = ax.warp_source(
        img, info.median_depth, conf, cam, cam, info.median_depth, occl_tol=0.08
    )
    covered = info.median_depth > 0
    m = (wgt > 0.5) & covered
    # Identity warp must accept (essentially) every rendered pixel...
    assert float(m.float().sum()) >= 0.99 * float(covered.float().sum()) > 0
    # ... and reproduce it.
    assert float((warped - img).abs().mean(dim=-1)[m].mean()) < 0.02


def test_warp_source_cross_view():
    sc = _gt_scene(n=900)
    Ra, ta = orbit_pose(1.0, radius=2.6)
    Rb, tb = orbit_pose(1.25, radius=2.6)
    img_a, info_a = _gt_render(sc, Ra, ta, aux=True)
    img_b, info_b = _gt_render(sc, Rb, tb, aux=True)
    cam_a = (torch.tensor(Ra, dtype=torch.float32), torch.tensor(ta, dtype=torch.float32), F, W/2, H/2)
    cam_b = (torch.tensor(Rb, dtype=torch.float32), torch.tensor(tb, dtype=torch.float32), F, W/2, H/2)
    warped, wgt = ax.warp_source(
        img_a, info_a.median_depth, torch.ones(H, W), cam_a,
        cam_b, info_b.median_depth, occl_tol=0.08,
    )
    m = wgt > 0.5
    covered_b = float((info_b.median_depth > 0).float().mean())
    assert float(m.float().mean()) > 0.4 * covered_b  # decent co-visible overlap
    # The speckle scene is high-frequency, so per-pixel agreement is loose;
    # it must still beat the trivial baseline (a constant mid-grey) clearly.
    err = float((warped - img_b).abs().mean(dim=-1)[m].mean())
    base = float((img_b - img_b.mean()).abs().mean(dim=-1)[m].mean())
    assert err < 0.8 * base


# -- fixing a broken view ------------------------------------------------------


def _sources_from_gt(sc, angles):
    cfg = ax.ArtifixConfig()
    out = []
    for a in angles:
        R, t = orbit_pose(a, radius=2.6)
        img, info = _gt_render(sc, R, t, aux=True)
        out.append({
            "rgb": img, "depth": info.median_depth,
            "conf": ax.confidence_from_alpha(info.alpha, cfg),
            "cam": (
                torch.tensor(R, dtype=torch.float32),
                torch.tensor(t, dtype=torch.float32), F, W / 2, H / 2,
            ),
        })
    return out


def test_fix_view_improves_hole_view():
    sc = _gt_scene()
    deg = _degraded_model(sc)
    a_mid = 0.5 * (WEDGE0 + WEDGE1)  # looking straight into the wedge
    R, t = orbit_pose(a_mid, radius=2.6)
    gt = _gt_render(sc, R, t)
    with torch.no_grad():
        rgb, info = render_model(
            deg, torch.tensor(R, dtype=torch.float32),
            torch.tensor(t, dtype=torch.float32), F, W / 2, H / 2, W, H,
            return_aux=True,
        )
    cfg = ax.ArtifixConfig()
    cam = (torch.tensor(R, dtype=torch.float32), torch.tensor(t, dtype=torch.float32), F, W/2, H/2)
    sources = _sources_from_gt(sc, [WEDGE0 - 0.25, WEDGE1 + 0.25, WEDGE0 - 0.7])
    fix = ax.fix_view(rgb, info.alpha, info.median_depth, cam, sources, cfg)
    # Closer to GT: the fixer may only import multi-source-verified content,
    # so on this high-frequency speckle scene the gain is real but bounded.
    assert psnr(fix["fixed"], gt) > psnr(rgb, gt) + 0.3
    # Confident pixels pass through (opacity gate keeps trusted content).
    keep = fix["conf"] > 0.95
    if bool(keep.any()):
        assert float((fix["fixed"] - rgb).abs().mean(dim=-1)[keep].max()) < 0.15
    assert torch.isfinite(fix["fixed"]).all() and torch.isfinite(fix["weight"]).all()


# -- floater pruning -----------------------------------------------------------


def _inject_camouflaged_floaters(model, rec, frames, n_per_cam=5, seed=11):
    rng = np.random.default_rng(seed)
    target = np.median(rec.points, axis=0)
    pos, col = [], []
    for i in rec.registered:
        Rc, tc = rec.poses[i]
        C = -Rc.T @ tc
        for _ in range(n_per_cam):
            p = C + rng.uniform(0.35, 0.65) * (target - C) + rng.normal(0, 0.06, 3)
            pos.append(p)
            pc = Rc @ p + tc
            u = int(np.clip(round(F * pc[0] / pc[2] + W / 2), 0, W - 1))
            v = int(np.clip(round(F * pc[1] / pc[2] + H / 2), 0, H - 1))
            col.append(frames[i][v, u, ::-1].astype(np.float64) / 255.0)
    model.append_gaussians(
        torch.tensor(np.array(pos), dtype=torch.float32),
        torch.tensor(np.array(col), dtype=torch.float32),
        torch.full((len(pos),), 0.035), init_opacity=0.85,
    )
    return len(pos)


def test_prune_floaters_removes_junk_keeps_anchors():
    from splatvid.train import build_views

    sc = _gt_scene()
    rec = _reconstruction(sc, n_cap=6)
    frames = _frames(sc, rec)
    # "Perfect" model = the GT scene itself, then junk it up.
    m = GaussianModel(sc["xyz"].numpy(), sc["rgb"].numpy())
    with torch.no_grad():
        m.log_scale.data = torch.log(sc["scale"])
        m.quat.data = sc["quat"]
        m.opacity.data.fill_(6.0)
    n_clean = m.num_gaussians
    nf = _inject_camouflaged_floaters(m, rec, frames)
    cfg = ax.ArtifixConfig(n_novel=10, train_size=W)
    anchors = build_views(rec, frames, W, "cpu")
    before = ax._anchor_psnr(m, anchors)
    novel = ax.extended_trajectory(rec, cfg)
    removed = ax.prune_floaters(m, novel, F, W / 2, H / 2, W, H, cfg, anchors=anchors)
    after = ax._anchor_psnr(m, anchors)
    assert removed >= int(0.5 * nf)  # most injected junk gone
    # Verification guarantee: anchors must not get worse.
    assert after >= before - cfg.floater_verify_tol
    # And the prune must not have gutted the real scene.
    assert m.num_gaussians >= n_clean - int(0.1 * n_clean)


def test_append_gaussians_renders():
    sc = _gt_scene(n=200)
    m = GaussianModel(sc["xyz"].numpy(), sc["rgb"].numpy())
    n0 = m.num_gaussians
    m.append_gaussians(
        torch.tensor([[0.0, 0.0, 0.0], [0.1, 0.1, 0.1]]),
        torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        torch.tensor([0.05, 0.05]),
    )
    assert m.num_gaussians == n0 + 2
    for p in (m.xyz, m.log_scale, m.quat, m.color, m.opacity):
        assert torch.isfinite(p).all()
    R, t = orbit_pose(0.5, radius=2.6)
    img, _ = render_model(
        m, torch.tensor(R, dtype=torch.float32), torch.tensor(t, dtype=torch.float32),
        F, W / 2, H / 2, W, H,
    )
    assert torch.isfinite(img).all()


# -- the full pass -------------------------------------------------------------


@pytest.mark.slow
def test_artifix_end_to_end_repairs_scene():
    """Sparse capture + camouflaged floaters -> artifix must lift held-out
    novel-view PSNR, grow coverage, and keep anchors healthy."""
    from splatvid.train import TrainConfig, train

    sc = _gt_scene(n=900, seed=5)
    arc = (0.0, 0.75 * math.pi)
    rec = _reconstruction(sc, n_cap=4, arc=arc)
    frames = _frames(sc, rec)
    model = train(rec, frames, TrainConfig(
        iterations=250, train_size=W, device="cpu", max_gaussians=4000,
        densify_from=60, densify_every=80, log_every=10_000,
    ))
    _inject_camouflaged_floaters(model, rec, frames, n_per_cam=8)

    def eval_novel():
        vals = []
        with torch.no_grad():
            for k in range(6):
                a = arc[0] + (arc[1] - arc[0]) * (k + 0.5) / 6
                hgt = -1.0 + (0.45 if k % 2 else -0.35)
                R, t = orbit_pose(a, radius=2.6, height=hgt)
                pred, _ = render_model(
                    model, torch.tensor(R, dtype=torch.float32),
                    torch.tensor(t, dtype=torch.float32), F, W / 2, H / 2, W, H,
                )
                vals.append(psnr(pred, _gt_render(sc, R, t)))
        return float(np.mean(vals))

    before = eval_novel()
    cfg = ax.ArtifixConfig(
        n_novel=10, finetune_iters=150, train_size=W, densify_every=50,
        log_every=10_000,
    )
    rep = ax.artifix(model, rec, images=frames, cfg=cfg)
    after = eval_novel()

    assert rep["floaters_pruned"] > 0
    assert after > before + 0.2  # repaired scene is measurably closer to GT
    assert rep["coverage_after"] > rep["coverage_before"]  # holes got filled
    assert rep["anchor_psnr_after"] >= rep["anchor_psnr_before"] - 0.1
    for p in (model.xyz, model.log_scale, model.quat, model.color, model.opacity):
        assert torch.isfinite(p).all()
