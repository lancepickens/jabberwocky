"""Image losses: L1 + differentiable SSIM (the standard 3DGS combination)."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F


def _gaussian_window(size: int, sigma: float, device) -> torch.Tensor:
    xs = torch.arange(size, dtype=torch.float32, device=device) - size // 2
    g = torch.exp(-(xs**2) / (2 * sigma**2))
    g = g / g.sum()
    w2d = g[:, None] * g[None, :]
    return w2d


def ssim(img1: torch.Tensor, img2: torch.Tensor, window_size: int = 11) -> torch.Tensor:
    """SSIM for (H, W, 3) images in [0, 1]. Returns a scalar."""
    device = img1.device
    c = img1.shape[-1]
    w = _gaussian_window(window_size, 1.5, device)
    w = w.expand(c, 1, window_size, window_size)
    x = img1.permute(2, 0, 1)[None]
    y = img2.permute(2, 0, 1)[None]
    pad = window_size // 2

    mu_x = F.conv2d(x, w, padding=pad, groups=c)
    mu_y = F.conv2d(y, w, padding=pad, groups=c)
    mu_x2, mu_y2, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    sig_x = F.conv2d(x * x, w, padding=pad, groups=c) - mu_x2
    sig_y = F.conv2d(y * y, w, padding=pad, groups=c) - mu_y2
    sig_xy = F.conv2d(x * y, w, padding=pad, groups=c) - mu_xy

    C1, C2 = 0.01**2, 0.03**2
    s = ((2 * mu_xy + C1) * (2 * sig_xy + C2)) / (
        (mu_x2 + mu_y2 + C1) * (sig_x + sig_y + C2)
    )
    return s.mean()


def image_loss(
    pred: torch.Tensor, target: torch.Tensor, ssim_weight: float = 0.2
) -> torch.Tensor:
    l1 = (pred - target).abs().mean()
    if ssim_weight <= 0:
        return l1
    return (1 - ssim_weight) * l1 + ssim_weight * (1 - ssim(pred, target))


def psnr(pred: torch.Tensor, target: torch.Tensor) -> float:
    mse = float(((pred - target) ** 2).mean())
    if mse <= 1e-12:
        return 99.0
    return -10.0 * math.log10(mse)


# -- perceptual loss (optional; needs torchvision) --------------------------

_VGG = None


def perceptual_available() -> bool:
    """True if a VGG perceptual loss can be constructed (torchvision present)."""
    try:
        import torchvision  # noqa: F401
    except Exception:
        return False
    return True


def _get_vgg(device):
    """Lazily build a frozen VGG16 feature extractor (first conv blocks)."""
    global _VGG
    if _VGG is None:
        from torchvision.models import VGG16_Weights, vgg16

        net = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features[:16].eval()
        for p in net.parameters():
            p.requires_grad_(False)
        _VGG = net
    return _VGG.to(device)


def perceptual_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """VGG16 feature-space L1 between two (H, W, 3) images in [0, 1].

    A stand-in for LPIPS (same idea: compare deep features, not raw pixels) that
    needs only torchvision's pretrained VGG. Raises if torchvision is missing —
    callers should gate on ``perceptual_available()``.
    """
    vgg = _get_vgg(pred.device)
    mean = torch.tensor([0.485, 0.456, 0.406], device=pred.device)[None, :, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=pred.device)[None, :, None, None]

    def prep(x):
        x = x.permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        return (x - mean) / std

    return (vgg(prep(pred)) - vgg(prep(target))).abs().mean()


def temporal_warp_loss(
    img_a: torch.Tensor,
    depth_a: torch.Tensor,
    cam_a: tuple,
    img_b: torch.Tensor,
    cam_b: tuple,
    near: float = 0.05,
) -> torch.Tensor:
    """View-consistency loss between two nearby renders — the anti-popping term.

    Back-projects view ``a``'s pixels with its rendered depth, reprojects them
    into view ``b``, samples ``img_b`` there, and penalises disagreement with
    ``img_a`` over the co-visible region. A shader that flickers/pops as the
    camera moves cannot satisfy this, so minimising it forces view coherence.
    Each ``cam`` is ``(R, t, focal, cx, cy)`` (world-to-camera).
    """
    Ra, ta, fa, cxa, cya = cam_a
    Rb, tb, fb, cxb, cyb = cam_b
    h, w = depth_a.shape
    dev = img_a.device
    ys, xs = torch.meshgrid(
        torch.arange(h, device=dev, dtype=torch.float32),
        torch.arange(w, device=dev, dtype=torch.float32),
        indexing="ij",
    )
    px, py, z = xs + 0.5, ys + 0.5, depth_a
    # Pixel -> camera-a 3D (z is camera-space depth) -> world.
    xa = torch.stack([(px - cxa) / fa * z, (py - cya) / fa * z, z], dim=-1)
    xw = (xa - ta) @ Ra  # row-vector form of R_a^T (x - t)
    # World -> camera b -> pixel.
    xb = xw @ Rb.T + tb
    zb = xb[..., 2]
    ub = fb * xb[..., 0] / zb.clamp(min=1e-6) + cxb
    vb = fb * xb[..., 1] / zb.clamp(min=1e-6) + cyb
    valid = (
        (z > near) & (zb > near)
        & (ub >= 0) & (ub <= w) & (vb >= 0) & (vb <= h)
    )
    # Pixel centres live at i + 0.5, so normalise by the full extent and use
    # align_corners=False; an identity reprojection then samples exactly.
    gx = 2 * ub / w - 1
    gy = 2 * vb / h - 1
    grid = torch.stack([gx, gy], dim=-1)[None]  # (1, H, W, 2)
    imb = img_b.permute(2, 0, 1)[None]  # (1, 3, H, W)
    warped = torch.nn.functional.grid_sample(
        imb, grid, align_corners=False, padding_mode="border"
    )[0].permute(1, 2, 0)
    diff = (warped - img_a).abs().mean(dim=-1)
    m = valid.float()
    return (diff * m).sum() / (m.sum() + 1e-6)


def neural_image_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    ssim_weight: float = 0.2,
    perceptual_weight: float = 0.5,
) -> torch.Tensor:
    """L1 + SSIM + (optional) VGG perceptual, for the neural-renderer stage.

    Falls back to plain ``image_loss`` when ``perceptual_weight <= 0`` or
    torchvision is unavailable, so training runs either way.
    """
    loss = image_loss(pred, target, ssim_weight)
    if perceptual_weight > 0 and perceptual_available():
        loss = loss + perceptual_weight * perceptual_loss(pred, target)
    return loss
