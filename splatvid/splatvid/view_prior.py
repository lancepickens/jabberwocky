"""View priors: turn a rendered novel view into a plausible target (M4).

This is the integration point for a generative prior. A pretrained image or
video diffusion model (Difix3D+, ReconFusion, CAT3D, ...) plugs in here: given
a rendered pseudo-view from a camera the video never captured, it returns a
cleaned / hallucinated version, which ``train_neural`` uses as extra
supervision to denoise floaters and fill unobserved regions plausibly.

No such model ships with splatvid (it would be a heavy pretrained dependency),
so the default ``NoopViewPrior`` returns the render unchanged -- the pseudo-view
machinery runs end to end and contributes nothing until a real prior is
supplied. Implement ``__call__`` to add one.
"""

from __future__ import annotations

import torch


class ViewPrior:
    """Map a rendered view (H, W, 3) in [0, 1] to a plausible target image."""

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError


class NoopViewPrior(ViewPrior):
    """Identity prior: returns the render (detached) unchanged.

    The pseudo-view loss becomes ``|render - render.detach()| == 0``, so the
    mechanism is exercised but has no effect — the safe default until a real
    generative prior is plugged in.
    """

    def __call__(self, image: torch.Tensor) -> torch.Tensor:
        return image.detach()
