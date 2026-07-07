"""Independent monocular depth supervision (DepthAnything v2).

The mesh-depth experiment showed that supervising the splat against a mesh built
*from* the splat is circular and doesn't fix floaters. A monocular depth network
predicts depth from each frame independently of the reconstruction, so it is
genuinely new geometry the splat can be corrected toward.

DepthAnything v2 outputs *relative* inverse-depth (disparity), so we affine-align
it per frame to the sparse SfM points (in inverse-depth space) to get a dense
depth map in reconstruction units. Everything here is optional: it needs the
`transformers` extra (`pip install 'splatvid[depth]'`).
"""

from __future__ import annotations

import logging

import cv2
import numpy as np

from .geometry import project_points

log = logging.getLogger(__name__)

_PIPE = None
_DEFAULT_MODEL = "depth-anything/Depth-Anything-V2-Small-hf"


def depth_available() -> bool:
    try:
        import transformers  # noqa: F401
    except Exception:
        return False
    return True


def _get_pipe(model: str, device: str):
    global _PIPE
    if _PIPE is None:
        from transformers import pipeline

        dev = device if device in ("cuda", "mps") else "cpu"
        _PIPE = pipeline("depth-estimation", model=model, device=dev)
    return _PIPE


def predict_disparities(
    images: list[np.ndarray], *, model: str = _DEFAULT_MODEL, device: str = "cpu"
) -> list[np.ndarray]:
    """Per-frame relative inverse-depth (disparity) maps, one per BGR image.

    Higher value = nearer. Resized to each image's resolution.
    """
    from PIL import Image

    pipe = _get_pipe(model, device)
    out = []
    for img in images:
        rgb = img[:, :, ::-1]  # BGR -> RGB
        res = pipe(Image.fromarray(np.ascontiguousarray(rgb)))
        d = res["predicted_depth"].squeeze().float().cpu().numpy()
        if d.shape != img.shape[:2]:
            d = cv2.resize(d, (img.shape[1], img.shape[0]), interpolation=cv2.INTER_LINEAR)
        out.append(d.astype(np.float32))
    return out


def align_disparity_to_recon(
    disp: np.ndarray, rec, frame_index: int, *, min_points: int = 20
) -> np.ndarray | None:
    """Affine-align a disparity map to the SfM geometry → depth in recon units.

    Fits ``inv_depth ≈ a·disp + b`` on the sparse points that reproject into the
    frame (robustly, dropping outliers), then returns ``depth = 1/(a·disp + b)``
    with 0 where the fit is non-positive/invalid. Returns None if too few points.
    """
    R, t = rec.poses[frame_index]
    uv, z = project_points(rec.points, R, t, rec.K)
    h, w = disp.shape
    inside = (
        (z > 1e-6)
        & (uv[:, 0] >= 0) & (uv[:, 0] < w) & (uv[:, 1] >= 0) & (uv[:, 1] < h)
    )
    if int(inside.sum()) < min_points:
        return None
    px = np.clip(uv[inside, 0].round().astype(int), 0, w - 1)
    py = np.clip(uv[inside, 1].round().astype(int), 0, h - 1)
    d_at = disp[py, px].astype(np.float64)
    inv_depth = 1.0 / z[inside].astype(np.float64)

    def fit(mask):
        A = np.stack([d_at[mask], np.ones(mask.sum())], axis=1)
        ab, *_ = np.linalg.lstsq(A, inv_depth[mask], rcond=None)
        return ab

    mask = np.ones(d_at.shape[0], dtype=bool)
    ab = fit(mask)
    for _ in range(2):  # robustify: drop large residuals, refit
        resid = np.abs((ab[0] * d_at + ab[1]) - inv_depth)
        med = np.median(resid)
        mad = np.median(np.abs(resid - med)) + 1e-9
        mask = resid < med + 3.0 * mad
        if mask.sum() < min_points:
            break
        ab = fit(mask)

    a, b = float(ab[0]), float(ab[1])
    inv = a * disp.astype(np.float64) + b
    depth = np.where(inv > 1e-6, 1.0 / inv, 0.0)
    return depth.astype(np.float32)


def depth_targets(
    rec, images: list[np.ndarray], *, model: str = _DEFAULT_MODEL, device: str = "cpu"
) -> list[np.ndarray]:
    """Aligned monocular depth maps (recon units) for each registered view.

    Returned in ``rec.registered`` order (parallel to ``build_views``); a view
    whose alignment failed gets an all-zero map (ignored by the depth loss).
    """
    reg_images = [images[fi] for fi in rec.registered]
    disps = predict_disparities(reg_images, model=model, device=device)
    out = []
    n_ok = 0
    for disp, fi in zip(disps, rec.registered):
        d = align_disparity_to_recon(disp, rec, fi)
        if d is None:
            d = np.zeros(disp.shape, np.float32)
        else:
            n_ok += 1
        out.append(d)
    log.info("Monocular depth: aligned %d/%d views", n_ok, len(rec.registered))
    return out
