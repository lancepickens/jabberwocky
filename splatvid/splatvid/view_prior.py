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
    """Map a rendered view (H, W, 3) in [0, 1] to a plausible target image.

    ``cam=(R, t, focal, cx, cy)`` is the pseudo-view camera; priors that render
    an independent scene (e.g. a mesh) use it, priors that clean the render
    ignore it.
    """

    def __call__(self, image: torch.Tensor, cam=None) -> torch.Tensor:
        raise NotImplementedError


class NoopViewPrior(ViewPrior):
    """Identity prior: returns the render (detached) unchanged.

    The pseudo-view loss becomes ``|render - render.detach()| == 0``, so the
    mechanism is exercised but has no effect — the safe default until a real
    generative prior is plugged in.
    """

    def __call__(self, image: torch.Tensor, cam=None) -> torch.Tensor:
        return image.detach()


class MeshViewPrior(ViewPrior):
    """Render a colored mesh from the pseudo-view camera as the target.

    Turns the metric mesh into a source of synthetic novel views: for a camera
    the video never filmed, the mesh render is a plausible target the splat is
    pulled toward — adding coverage of unobserved angles (the mesh's numpy
    rasterizer is CPU-only, so this trades speed for a dependency-free target).
    """

    def __init__(self, mesh, focal, cx, cy, width, height):
        self.mesh = mesh
        self.focal, self.cx, self.cy = float(focal), float(cx), float(cy)
        self.width, self.height = int(width), int(height)

    def __call__(self, image: torch.Tensor, cam=None) -> torch.Tensor:
        if cam is None:
            return image.detach()
        from .mesh import render_mesh

        R, t = cam[0], cam[1]
        R = R.detach().cpu().numpy() if torch.is_tensor(R) else R
        t = t.detach().cpu().numpy() if torch.is_tensor(t) else t
        color, _ = render_mesh(
            self.mesh, R, t, self.focal, self.cx, self.cy, self.width, self.height
        )
        return torch.as_tensor(color, dtype=image.dtype, device=image.device)
