"""Export trained gaussians to the standard 3DGS .ply and compact .splat formats."""

from __future__ import annotations

import logging
import struct

import numpy as np
import torch

from .model import GaussianModel

log = logging.getLogger(__name__)


def _gather(model: GaussianModel) -> dict[str, np.ndarray]:
    with torch.no_grad():
        return {
            "xyz": model.xyz.cpu().numpy(),
            "log_scale": model.log_scale.cpu().numpy(),
            "quat": model.get_quat().cpu().numpy(),
            "sh_dc": model.color.cpu().numpy(),
            "opacity_logit": model.opacity.cpu().numpy(),
            "rgb": model.get_rgb().cpu().numpy(),
            "alpha": model.get_opacity().cpu().numpy(),
        }


def save_ply(model: GaussianModel, path: str) -> None:
    """Write the de-facto standard 3D Gaussian Splatting PLY layout.

    Stores raw (pre-activation) values, so files load in common splat
    viewers/editors (SuperSplat, gsplat viewers, etc.).
    """
    g = _gather(model)
    n = g["xyz"].shape[0]
    props = (
        ["x", "y", "z", "nx", "ny", "nz"]
        + [f"f_dc_{i}" for i in range(3)]
        + ["opacity"]
        + [f"scale_{i}" for i in range(3)]
        + [f"rot_{i}" for i in range(4)]
    )
    header = (
        "ply\nformat binary_little_endian 1.0\n"
        f"element vertex {n}\n"
        + "".join(f"property float {p}\n" for p in props)
        + "end_header\n"
    )
    data = np.concatenate(
        [
            g["xyz"],
            np.zeros((n, 3), np.float32),  # normals, unused but conventional
            g["sh_dc"],
            g["opacity_logit"],
            g["log_scale"],
            g["quat"],
        ],
        axis=1,
    ).astype("<f4")
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(data.tobytes())
    log.info("Wrote %s (%d gaussians)", path, n)


def save_splat(model: GaussianModel, path: str) -> None:
    """Write the compact 32-byte-per-gaussian .splat format.

    Layout per gaussian (antimatter15-style, used by many web viewers):
    float32 x, y, z; float32 sx, sy, sz; uint8 r, g, b, a;
    uint8 quat packed as (q * 128 + 128) in (w, x, y, z) order.
    Gaussians are sorted by opacity-weighted size, largest first.
    """
    g = _gather(model)
    n = g["xyz"].shape[0]
    scales = np.exp(g["log_scale"])
    order = np.argsort(
        -(g["alpha"][:, 0] * scales.prod(axis=1) ** (1 / 3))
    )
    rgba = np.concatenate(
        [
            (g["rgb"] * 255).clip(0, 255),
            (g["alpha"] * 255).clip(0, 255),
        ],
        axis=1,
    ).astype(np.uint8)
    quat_u8 = (g["quat"] * 128 + 128).clip(0, 255).astype(np.uint8)

    buf = bytearray()
    for i in order:
        buf += struct.pack("<3f", *g["xyz"][i])
        buf += struct.pack("<3f", *scales[i])
        buf += bytes(rgba[i])
        buf += bytes(quat_u8[i])
    with open(path, "wb") as f:
        f.write(buf)
    log.info("Wrote %s (%d gaussians, %.1f KB)", path, n, len(buf) / 1024)


def load_splat(path: str) -> dict[str, np.ndarray]:
    """Read a .splat file back (for tests and tooling)."""
    raw = np.fromfile(path, dtype=np.uint8)
    if raw.size % 32:
        raise ValueError(f"{path}: size {raw.size} is not a multiple of 32")
    rec = raw.reshape(-1, 32)
    xyz = rec[:, 0:12].copy().view("<f4").reshape(-1, 3)
    scales = rec[:, 12:24].copy().view("<f4").reshape(-1, 3)
    rgba = rec[:, 24:28].astype(np.float32) / 255.0
    quat = (rec[:, 28:32].astype(np.float32) - 128.0) / 128.0
    return {"xyz": xyz, "scales": scales, "rgba": rgba, "quat": quat}
