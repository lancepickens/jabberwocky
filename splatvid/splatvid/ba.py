"""Sparse bundle adjustment on top of scipy.optimize.least_squares.

Parameter vector layout:

    [log_focal?] + [rvec_c, tvec_c for each free camera] + [xyz_p for each point]

The first registered camera is held fixed to pin the gauge (global
rotation/translation); overall scale remains a free gauge, which is fine
for reconstruction purposes.
"""

from __future__ import annotations

import logging

import numpy as np
from scipy.optimize import least_squares
from scipy.sparse import lil_matrix

from .geometry import rodrigues_to_rotmat, rotmat_to_rodrigues

log = logging.getLogger(__name__)


def bundle_adjust(
    focal: float,
    cx: float,
    cy: float,
    poses: dict[int, tuple[np.ndarray, np.ndarray]],
    points: np.ndarray,
    observations: list[tuple[int, int, float, float]],
    fixed_cams: set[int] | None = None,
    refine_focal: bool = True,
    max_nfev: int = 40,
) -> tuple[float, dict[int, tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Jointly refine focal length, camera poses, and 3D points.

    Args:
        focal: shared focal length in pixels.
        poses: {cam_id: (R, t)} world-to-camera.
        points: (P, 3) world points.
        observations: (cam_id, point_idx, u, v) measured pixels.
        fixed_cams: camera ids to hold fixed (defaults to the lowest id).

    Returns (focal, poses, points) refined.
    """
    cam_ids = sorted(poses)
    if fixed_cams is None:
        fixed_cams = {cam_ids[0]}
    free_cams = [c for c in cam_ids if c not in fixed_cams]
    cam_slot = {c: k for k, c in enumerate(free_cams)}

    n_free = len(free_cams)
    n_pts = points.shape[0]
    nf = 1 if refine_focal else 0

    x0 = np.empty(nf + 6 * n_free + 3 * n_pts)
    if refine_focal:
        x0[0] = np.log(focal)
    for c in free_cams:
        R, t = poses[c]
        k = cam_slot[c]
        x0[nf + 6 * k : nf + 6 * k + 3] = rotmat_to_rodrigues(R)
        x0[nf + 6 * k + 3 : nf + 6 * k + 6] = t
    x0[nf + 6 * n_free :] = points.reshape(-1)

    obs = np.asarray(observations, dtype=np.float64)
    obs_cam = obs[:, 0].astype(int)
    obs_pt = obs[:, 1].astype(int)
    obs_uv = obs[:, 2:4]
    n_obs = obs.shape[0]

    fixed_pose = {c: poses[c] for c in fixed_cams}

    def unpack(x):
        f = np.exp(x[0]) if refine_focal else focal
        rt = x[nf : nf + 6 * n_free].reshape(n_free, 6)
        pts = x[nf + 6 * n_free :].reshape(n_pts, 3)
        return f, rt, pts

    # Precompute per-observation slot index (-1 for fixed cameras).
    obs_slot = np.array([cam_slot.get(c, -1) for c in obs_cam])

    def residuals(x):
        f, rt, pts = unpack(x)
        Rs = np.empty((n_obs, 3, 3))
        ts = np.empty((n_obs, 3))
        free_mask = obs_slot >= 0
        if free_mask.any():
            sl = obs_slot[free_mask]
            Rs[free_mask] = rodrigues_to_rotmat(rt[sl, :3])
            ts[free_mask] = rt[sl, 3:]
        for c, (R, t) in fixed_pose.items():
            m = obs_cam == c
            Rs[m] = R
            ts[m] = t
        P = pts[obs_pt]
        pc = np.einsum("nij,nj->ni", Rs, P) + ts
        z = np.where(np.abs(pc[:, 2]) < 1e-9, 1e-9, pc[:, 2])
        u = f * pc[:, 0] / z + cx
        v = f * pc[:, 1] / z + cy
        # Interleaved (u_i, v_i) residuals to match the sparsity pattern below.
        return np.stack([u - obs_uv[:, 0], v - obs_uv[:, 1]], axis=1).ravel()

    # Jacobian sparsity: each residual pair depends on its camera's 6 params,
    # its point's 3 params, and (optionally) the focal.
    A = lil_matrix((2 * n_obs, x0.size), dtype=int)
    rows = np.arange(n_obs)
    for r in (2 * rows, 2 * rows + 1):
        if refine_focal:
            A[r, 0] = 1
        for d in range(3):
            A[r, nf + 6 * n_free + 3 * obs_pt + d] = 1
    free_rows = np.nonzero(obs_slot >= 0)[0]
    for r0 in free_rows:
        base = nf + 6 * obs_slot[r0]
        A[2 * r0, base : base + 6] = 1
        A[2 * r0 + 1, base : base + 6] = 1

    res = least_squares(
        residuals,
        x0,
        jac_sparsity=A,
        method="trf",
        loss="soft_l1",
        f_scale=2.0,
        max_nfev=max_nfev,
        x_scale="jac",
        verbose=0,
    )

    f, rt, pts = unpack(res.x)
    new_poses = dict(fixed_pose)
    for c in free_cams:
        k = cam_slot[c]
        new_poses[c] = (rodrigues_to_rotmat(rt[k, :3]), rt[k, 3:].copy())
    rms = float(np.sqrt(np.mean(res.fun**2)))
    log.info(
        "BA: %d cams (%d fixed), %d pts, %d obs -> RMS %.3f px, focal %.1f",
        len(cam_ids), len(fixed_cams), n_pts, n_obs, rms, f,
    )
    return float(f), new_poses, pts
