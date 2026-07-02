"""Differentiable gaussian-splat rasterizer, from scratch in pure PyTorch.

EWA splatting: each 3D gaussian is projected to a 2D gaussian via the
local affine (Jacobian) approximation of the perspective projection, then
composited front-to-back per 16x16 pixel tile with alpha blending.
Everything is autograd-friendly; gradients flow to positions, scales,
rotations, colors, and opacities.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

TILE = 16


@dataclass
class RenderInfo:
    """Side outputs used for densification and diagnostics."""

    means2d: torch.Tensor  # (N, 2) projected centers (holds .grad after backward)
    visible: torch.Tensor  # (N,) bool, gaussian touched at least one tile
    radii: torch.Tensor  # (N,) float pixel radii (0 for culled)


def quat_to_rotmat_torch(q: torch.Tensor) -> torch.Tensor:
    """(N, 4) normalized quaternions (w, x, y, z) -> (N, 3, 3)."""
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        [
            1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y),
            2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x),
            2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y),
        ],
        dim=-1,
    ).reshape(-1, 3, 3)


def compute_cov3d(scale: torch.Tensor, quat: torch.Tensor) -> torch.Tensor:
    """Sigma = R S S^T R^T from per-axis scales and rotations."""
    R = quat_to_rotmat_torch(quat)
    M = R * scale[:, None, :]  # R @ diag(scale)
    return M @ M.transpose(1, 2)


def project_gaussians(
    xyz: torch.Tensor,
    cov3d: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    focal: float,
    cx: float,
    cy: float,
    near: float = 0.05,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Project 3D gaussians into screen space.

    Returns (means2d (N,2), cov2d (N,2,2), depth (N,), in_front (N,) bool).
    Culled gaussians keep placeholder values; use the mask.
    """
    p_cam = xyz @ R.T + t[None]
    z = p_cam[:, 2]
    in_front = z > near
    zc = z.clamp(min=near)

    u = focal * p_cam[:, 0] / zc + cx
    v = focal * p_cam[:, 1] / zc + cy
    means2d = torch.stack([u, v], dim=-1)

    # Jacobian of (u, v) wrt camera-space point, evaluated at the center.
    zero = torch.zeros_like(zc)
    J = torch.stack(
        [
            focal / zc, zero, -focal * p_cam[:, 0] / (zc * zc),
            zero, focal / zc, -focal * p_cam[:, 1] / (zc * zc),
        ],
        dim=-1,
    ).reshape(-1, 2, 3)
    JW = J @ R[None]
    cov2d = JW @ cov3d @ JW.transpose(1, 2)
    # Low-pass dilation: every splat covers at least ~a pixel (antialias).
    cov2d = cov2d + 0.3 * torch.eye(2, device=xyz.device, dtype=xyz.dtype)[None]
    return means2d, cov2d, z, in_front


def render(
    xyz: torch.Tensor,
    scale: torch.Tensor,
    quat: torch.Tensor,
    rgb: torch.Tensor,
    opacity: torch.Tensor,
    R: torch.Tensor,
    t: torch.Tensor,
    focal: float,
    cx: float,
    cy: float,
    width: int,
    height: int,
    bg: torch.Tensor | None = None,
    max_per_tile: int = 1024,
) -> tuple[torch.Tensor, RenderInfo]:
    """Render one view. Returns (image (H, W, 3) in [0, 1], RenderInfo)."""
    dev = xyz.device
    if bg is None:
        bg = torch.zeros(3, device=dev)
    n = xyz.shape[0]

    cov3d = compute_cov3d(scale, quat)
    means2d, cov2d, depth, in_front = project_gaussians(
        xyz, cov3d, R, t, focal, cx, cy
    )
    if means2d.requires_grad:
        means2d.retain_grad()

    # Conic (inverse 2D covariance) and screen radius (3 sigma).
    a = cov2d[:, 0, 0]
    b = cov2d[:, 0, 1]
    c = cov2d[:, 1, 1]
    det = (a * c - b * b).clamp(min=1e-12)
    conic = torch.stack([c / det, -b / det, a / det], dim=-1)  # (A, B, C)
    mid = 0.5 * (a + c)
    eig_max = mid + torch.sqrt((mid * mid - det).clamp(min=1e-12))
    radii = 3.0 * torch.sqrt(eig_max)

    u = means2d[:, 0]
    v = means2d[:, 1]
    on_screen = (
        in_front
        & (u + radii > 0) & (u - radii < width)
        & (v + radii > 0) & (v - radii < height)
        & (opacity[:, 0] > 1.0 / 255.0)
    )

    # Start from the background; covered tiles overwrite their region.
    image = torch.zeros(height, width, 3, device=dev) + bg[None, None, :]
    info = RenderInfo(
        means2d=means2d,
        visible=torch.zeros(n, dtype=torch.bool, device=dev),
        radii=torch.where(on_screen, radii, torch.zeros_like(radii)).detach(),
    )
    sel = torch.nonzero(on_screen).squeeze(1)
    if sel.numel() == 0:
        return image, info
    info.visible[sel] = True

    # Depth-sort surviving gaussians front to back.
    order = torch.argsort(depth[sel].detach())
    sel = sel[order]

    g_u = u[sel]
    g_v = v[sel]
    g_conic = conic[sel]
    g_rgb = rgb[sel]
    g_op = opacity[sel, 0]
    g_rad = radii[sel].detach()

    n_tx = (width + TILE - 1) // TILE
    n_ty = (height + TILE - 1) // TILE
    # Tile-index bounds per gaussian (inclusive). Binning is pure
    # bookkeeping, so it runs on the CPU: on GPU/MPS devices this avoids a
    # blocking device sync per tile inside the Python loop below — only the
    # final gather indices are shipped to the device.
    u_c = g_u.detach().cpu()
    v_c = g_v.detach().cpu()
    rad_c = g_rad.cpu()
    tx0 = ((u_c - rad_c) / TILE).floor().clamp(0, n_tx - 1).long()
    tx1 = ((u_c + rad_c) / TILE).floor().clamp(0, n_tx - 1).long()
    ty0 = ((v_c - rad_c) / TILE).floor().clamp(0, n_ty - 1).long()
    ty1 = ((v_c + rad_c) / TILE).floor().clamp(0, n_ty - 1).long()

    ys = torch.arange(height, device=dev, dtype=torch.float32) + 0.5
    xs = torch.arange(width, device=dev, dtype=torch.float32) + 0.5

    for ty in range(n_ty):
        row_mask = (ty >= ty0) & (ty <= ty1)
        if not bool(row_mask.any()):
            continue
        row_idx = torch.nonzero(row_mask).squeeze(1)
        y_lo, y_hi = ty * TILE, min((ty + 1) * TILE, height)
        py = ys[y_lo:y_hi]
        for tx in range(n_tx):
            tmask = (tx >= tx0[row_idx]) & (tx <= tx1[row_idx])
            if not bool(tmask.any()):
                continue
            # Front-most first (pre-sorted); indices to the device lazily.
            idx = row_idx[tmask][:max_per_tile].to(dev)
            x_lo, x_hi = tx * TILE, min((tx + 1) * TILE, width)
            px = xs[x_lo:x_hi]

            dx = px[None, None, :] - g_u[idx][:, None, None]  # (G, 1, W_t)
            dy = py[None, :, None] - g_v[idx][:, None, None]  # (G, H_t, 1)
            A = g_conic[idx, 0][:, None, None]
            B = g_conic[idx, 1][:, None, None]
            C = g_conic[idx, 2][:, None, None]
            power = -0.5 * (A * dx * dx + C * dy * dy) - B * dx * dy
            gauss = torch.exp(power.clamp(max=0.0))
            alpha = (g_op[idx][:, None, None] * gauss).clamp(max=0.99)
            alpha = torch.where(alpha < 1.0 / 255.0, torch.zeros_like(alpha), alpha)

            # Exclusive front-to-back transmittance along the gaussian dim.
            one_minus = 1.0 - alpha
            T = torch.cumprod(one_minus, dim=0)
            T_excl = torch.cat([torch.ones_like(T[:1]), T[:-1]], dim=0)
            w = alpha * T_excl  # (G, H_t, W_t)

            tile_rgb = torch.einsum("ghw,gc->hwc", w, g_rgb[idx])
            T_final = T[-1]  # remaining transmittance
            image[y_lo:y_hi, x_lo:x_hi] = tile_rgb + T_final[:, :, None] * bg[None, None, :]

    return image, info


def render_model(model, R, t, focal, cx, cy, width, height, bg=None, **kw):
    """Convenience wrapper rendering a GaussianModel."""
    return render(
        model.xyz,
        model.get_scale(),
        model.get_quat(),
        model.get_rgb(),
        model.get_opacity(),
        R, t, focal, cx, cy, width, height, bg=bg, **kw,
    )
