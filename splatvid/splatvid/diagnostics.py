"""Device self-check: verify the rasterizer is correct on the chosen backend.

The pure-PyTorch rasterizer leans on a handful of operators that have
historically been the shaky parts of PyTorch's Metal (MPS) backend used on
Apple-Silicon Macs: ``cumprod`` (front-to-back transmittance), grouped
``conv2d`` (the SSIM window), ``argsort`` (depth sort), boolean ``nonzero``
masks, and indexed slice assignment into the image. A backend can *run*
these and still return wrong numbers, which shows up only as a model that
never converges.

``run_selfcheck`` renders a tiny scene forward **and** backward on the
target device and compares both the image and the position gradients
against the CPU reference. Same math on both devices, so any real
divergence is a backend bug, not a modelling choice. This is what
``splatvid doctor`` runs, and what the MPS parity test asserts.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass, field

import torch

from .losses import image_loss
from .render import render
from .synthetic import make_scene, orbit_pose

# Max absolute image difference and relative gradient difference we tolerate
# between a device and the CPU reference. float32 accumulation across the two
# backends legitimately differs at the 1e-4 level; a real backend bug (e.g. a
# broken cumprod) is off by orders of magnitude, so this cleanly separates the
# two without being flaky.
IMG_ATOL = 2e-3
GRAD_RTOL = 5e-2


@dataclass
class DeviceReport:
    device: str
    ok: bool
    image_max_diff: float = float("nan")
    grad_rel_diff: float = float("nan")
    finite: bool = True
    error: str | None = None
    notes: list[str] = field(default_factory=list)


def _render_once(device: str, scene: dict, seed: int = 0):
    """Forward+backward on ``device``; return (image_cpu, xyz_grad_cpu)."""
    torch.manual_seed(seed)
    params = {
        k: scene[k].to(device).clone().requires_grad_(True)
        for k in ("xyz", "scale", "quat", "rgb", "opacity")
    }
    w, h = 96, 72
    R, t = orbit_pose(0.7)
    Rt = torch.tensor(R, dtype=torch.float32, device=device)
    tt = torch.tensor(t, dtype=torch.float32, device=device)
    focal = 1.1 * w

    img, _ = render(
        params["xyz"], params["scale"].clamp(max=1.0), params["quat"],
        params["rgb"], params["opacity"], Rt, tt, focal, w / 2, h / 2, w, h,
    )
    # SSIM loss exercises grouped conv2d (a classic MPS fallback op).
    target = torch.zeros_like(img)
    loss = image_loss(img, target, ssim_weight=0.2)
    loss.backward()
    return img.detach().cpu(), params["xyz"].grad.detach().cpu()


def run_selfcheck(device: str) -> DeviceReport:
    """Render a tiny scene on ``device`` and compare to the CPU reference."""
    report = DeviceReport(device=device, ok=False)
    if device == "cpu":
        # Nothing to compare against; just confirm it runs and is finite.
        try:
            img, grad = _render_once("cpu", make_scene(n=1500))
            report.finite = bool(torch.isfinite(img).all() and torch.isfinite(grad).all())
            report.ok = report.finite
            report.image_max_diff = 0.0
            report.grad_rel_diff = 0.0
            if not report.finite:
                report.error = "CPU render produced non-finite values"
        except Exception as e:  # pragma: no cover - defensive
            report.error = f"{type(e).__name__}: {e}"
        return report

    scene = make_scene(n=1500)
    try:
        ref_img, ref_grad = _render_once("cpu", scene)
        dev_img, dev_grad = _render_once(device, scene)
    except Exception as e:
        report.error = f"{type(e).__name__}: {e}"
        return report

    report.finite = bool(
        torch.isfinite(dev_img).all() and torch.isfinite(dev_grad).all()
    )
    report.image_max_diff = float((dev_img - ref_img).abs().max())
    denom = ref_grad.abs().max().clamp(min=1e-8)
    report.grad_rel_diff = float((dev_grad - ref_grad).abs().max() / denom)

    if not report.finite:
        report.error = f"{device} produced NaN/Inf (backend op likely unsupported)"
    elif report.image_max_diff > IMG_ATOL:
        report.error = (
            f"image differs from CPU by {report.image_max_diff:.2e} "
            f"(> {IMG_ATOL:.0e}); a rasterizer op is wrong on {device}"
        )
    elif report.grad_rel_diff > GRAD_RTOL:
        report.error = (
            f"gradients differ from CPU by {report.grad_rel_diff:.1%} "
            f"(> {GRAD_RTOL:.0%}); backprop is wrong on {device}"
        )
    else:
        report.ok = True
    return report


def collect_environment(target_device: str) -> dict:
    """Human-facing facts about the torch build and the chosen device."""
    info: dict[str, object] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "target_device": target_device,
        "cuda_available": torch.cuda.is_available(),
    }
    mps = getattr(torch.backends, "mps", None)
    if mps is not None:
        info["mps_available"] = bool(mps.is_available())
        # is_built(): the wheel has the Metal backend compiled in at all.
        info["mps_built"] = bool(getattr(mps, "is_built", lambda: False)())
    else:
        info["mps_available"] = False
        info["mps_built"] = False
    return info
