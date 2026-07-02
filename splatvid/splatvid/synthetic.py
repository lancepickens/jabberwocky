"""Synthetic scene + video generation, for tests and demos.

Builds a scene out of many small colored gaussians (so it is feature-rich
for SIFT), renders an orbit around it with the splatvid rasterizer, and
encodes the frames as a video. Running the full pipeline on this video
exercises every stage with known ground truth.
"""

from __future__ import annotations

import math

import cv2
import numpy as np
import torch

from .render import render


def make_scene(n: int = 4500, seed: int = 7) -> dict[str, torch.Tensor]:
    """A textured cube shell + ground disc built from small opaque gaussians.

    Gaussians are kept small and high-contrast so the rendered images have
    sharp, distinctive texture that feature detectors can latch onto.
    """
    rng = np.random.default_rng(seed)

    pts = []
    # Cube shell: points on the surface of a unit cube.
    n_cube = n // 2
    face_axis = rng.integers(0, 3, n_cube)
    face_sign = rng.choice([-1.0, 1.0], n_cube)
    uv = rng.uniform(-0.5, 0.5, (n_cube, 2))
    for k in range(n_cube):
        p = np.empty(3)
        others = [a for a in range(3) if a != face_axis[k]]
        p[face_axis[k]] = 0.5 * face_sign[k]
        p[others[0]] = uv[k, 0]
        p[others[1]] = uv[k, 1]
        pts.append(p)
    # Ground disc below the cube.
    n_gnd = n - n_cube
    r = np.sqrt(rng.uniform(0, 1, n_gnd)) * 1.6
    ang = rng.uniform(0, 2 * math.pi, n_gnd)
    for k in range(n_gnd):
        pts.append(np.array([r[k] * math.cos(ang[k]), 0.62, r[k] * math.sin(ang[k])]))
    xyz = np.array(pts, dtype=np.float32)

    # High-contrast speckle: saturated colors, small footprints.
    colors = rng.uniform(0.05, 1.0, (n, 3)).astype(np.float32)
    boost = rng.integers(0, 3, n)
    colors[np.arange(n), boost] = rng.uniform(0.75, 1.0, n)
    scales = np.full((n, 3), 0.012, dtype=np.float32) * rng.uniform(
        0.5, 1.8, (n, 1)
    ).astype(np.float32)
    quat = np.zeros((n, 4), dtype=np.float32)
    quat[:, 0] = 1.0
    opacity = np.full((n, 1), 1.0, dtype=np.float32)

    t = torch.tensor
    return {
        "xyz": t(xyz), "scale": t(scales), "quat": t(quat),
        "rgb": t(colors), "opacity": t(opacity),
    }


def orbit_pose(
    angle: float, radius: float = 2.6, height: float = -1.0,
    target: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """World-to-camera (R, t) for a camera on a circle looking at the target.

    Uses the same +z-forward convention as the rest of splatvid (y down).
    """
    if target is None:
        target = np.zeros(3)
    eye = np.array(
        [radius * math.cos(angle), height, radius * math.sin(angle)]
    )
    fwd = target - eye
    fwd = fwd / np.linalg.norm(fwd)
    up_world = np.array([0.0, 1.0, 0.0])  # y-down camera: world +y is "down"
    right = np.cross(up_world, fwd)
    right = right / np.linalg.norm(right)
    down = np.cross(fwd, right)
    R = np.stack([right, down, fwd])
    t = -R @ eye
    return R, t


def render_views(
    scene: dict[str, torch.Tensor],
    n_frames: int = 36,
    width: int = 320,
    height: int = 240,
    focal: float | None = None,
    arc: float = 2 * math.pi,
    radius: float = 2.6,
) -> tuple[list[np.ndarray], list[tuple[np.ndarray, np.ndarray]]]:
    """Render an orbit; returns (BGR uint8 frames, ground-truth poses)."""
    focal = focal or 1.1 * max(width, height)
    frames: list[np.ndarray] = []
    poses: list[tuple[np.ndarray, np.ndarray]] = []
    bg = torch.tensor([0.06, 0.07, 0.09])
    with torch.no_grad():
        for k in range(n_frames):
            ang = arc * k / n_frames
            R, t = orbit_pose(ang, radius=radius)
            img, _ = render(
                scene["xyz"], scene["scale"], scene["quat"], scene["rgb"],
                scene["opacity"],
                torch.tensor(R, dtype=torch.float32),
                torch.tensor(t, dtype=torch.float32),
                focal, width / 2, height / 2, width, height, bg=bg,
            )
            rgb8 = (img.numpy() * 255).clip(0, 255).astype(np.uint8)
            frames.append(rgb8[:, :, ::-1].copy())  # RGB -> BGR
            poses.append((R, t))
    return frames, poses


def write_video(frames: list[np.ndarray], path: str, fps: int = 12) -> None:
    h, w = frames[0].shape[:2]
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    if not vw.isOpened():
        raise RuntimeError(f"Could not open video writer for {path}")
    for f in frames:
        vw.write(f)
    vw.release()


def make_synthetic_video(
    path: str, n_frames: int = 36, width: int = 320, height: int = 240,
    n_gaussians: int = 1200, seed: int = 7, arc: float = 2 * math.pi,
) -> None:
    scene = make_scene(n=n_gaussians, seed=seed)
    frames, _ = render_views(scene, n_frames=n_frames, width=width, height=height, arc=arc)
    write_video(frames, path)
