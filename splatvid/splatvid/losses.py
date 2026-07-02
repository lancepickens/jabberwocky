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
