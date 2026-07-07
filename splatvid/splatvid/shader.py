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
    ) -> torch.Tensor:
        x = feat_map.permute(2, 0, 1).unsqueeze(0)  # (1, C, H, W)
        y = self.conv(x)
        return y.squeeze(0).permute(1, 2, 0)  # (H, W, 3)
