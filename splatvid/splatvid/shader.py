"""Neural shaders: decode a splatted feature map into an RGB image.

M0 provides only the trivial ``IdentityShader`` (a 1x1 convolution initialised
to copy the first three feature channels), which proves the feature-splat ->
shader path reproduces the direct-colour rasteriser. A real U-Net shader
replaces it in M1 (see docs/neural-renderer-plan.md).

Shaders take ``(feat_map, alpha, depth)`` and return an ``(H, W, 3)`` image:
  * ``feat_map`` (H, W, C) -- the composited per-gaussian features
  * ``alpha``    (H, W)    -- accumulated opacity (1 - transmittance)
  * ``depth``    (H, W)    -- opacity-weighted depth
The identity shader ignores ``alpha``/``depth``; later shaders consume them.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn


class IdentityShader(nn.Module):
    """A 1x1 conv initialised to select feature channels 0-2 as RGB.

    With ``feature[:, :3] == colour`` (the model's default init), decoding a
    feature render with this shader is numerically the direct-colour render --
    the M0 plumbing check. It is a real (trainable) conv layer, so the same
    interface upgrades cleanly to a learned decoder.
    """

    def __init__(self, in_channels: int) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 3, kernel_size=1, bias=True)
        with torch.no_grad():
            self.conv.weight.zero_()
            for c in range(min(3, in_channels)):
                self.conv.weight[c, c, 0, 0] = 1.0
            self.conv.bias.zero_()

    def forward(
        self,
        feat_map: torch.Tensor,
        alpha: torch.Tensor | None = None,
        depth: torch.Tensor | None = None,
        out_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        x = feat_map.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        y = self.conv(x)
        if out_size is not None and out_size != tuple(y.shape[-2:]):
            y = F.interpolate(y, size=out_size, mode="bilinear", align_corners=False)
        return y.squeeze(0).permute(1, 2, 0)  # (H, W, 3)


def _double_conv(cin: int, cout: int) -> nn.Sequential:
    g = min(8, cout)
    return nn.Sequential(
        nn.Conv2d(cin, cout, 3, padding=1),
        nn.GroupNorm(g, cout),
        nn.GELU(),
        nn.Conv2d(cout, cout, 3, padding=1),
        nn.GroupNorm(g, cout),
        nn.GELU(),
    )


class UNetShader(nn.Module):
    """Small U-Net that decodes the splatted feature buffer into RGB (M1+).

    Input is the composited feature map plus the alpha and (normalised) depth
    buffers; a 3-level encoder / 2-level decoder with skip connections outputs
    RGB in [0, 1]. A learned decoder cannot emit sub-pixel needles and
    band-limits hard splat edges, which is what removes jaggedness; bilinear
    up/down handles arbitrary (non-power-of-two) image sizes.
    """

    def __init__(self, feat_channels: int, base: int = 32) -> None:
        super().__init__()
        self.in_channels = feat_channels + 2  # + alpha + depth
        c1, c2, c3 = base, base * 2, base * 4
        self.e1 = _double_conv(self.in_channels, c1)
        self.e2 = _double_conv(c1, c2)
        self.e3 = _double_conv(c2, c3)
        self.pool = nn.MaxPool2d(2)
        self.d2 = _double_conv(c3 + c2, c2)
        self.d1 = _double_conv(c2 + c1, c1)
        self.out = nn.Conv2d(c1, 3, 1)

    def forward(
        self,
        feat_map: torch.Tensor,
        alpha: torch.Tensor,
        depth: torch.Tensor,
        out_size: tuple[int, int] | None = None,
    ) -> torch.Tensor:
        # Normalise depth to ~[0, 1] so its raw scene-unit scale doesn't swamp
        # the (already ~unit) features/alpha.
        d = depth / (depth.detach().max() + 1e-6)
        x = torch.cat([feat_map, alpha[..., None], d[..., None]], dim=-1)
        x = x.permute(2, 0, 1).unsqueeze(0)  # (1, Cin, H, W)

        e1 = self.e1(x)
        e2 = self.e2(self.pool(e1))
        e3 = self.e3(self.pool(e2))
        u2 = F.interpolate(e3, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.d2(torch.cat([u2, e2], dim=1))
        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.d1(torch.cat([u1, e1], dim=1))
        # Learned upsampling: when features were splatted at reduced resolution,
        # decode to the full output size (final conv after the interpolation).
        if out_size is not None and out_size != tuple(d1.shape[-2:]):
            d1 = F.interpolate(d1, size=out_size, mode="bilinear", align_corners=False)
        rgb = torch.sigmoid(self.out(d1))  # (1, 3, H, W)
        return rgb.squeeze(0).permute(1, 2, 0)  # (H, W, 3)
