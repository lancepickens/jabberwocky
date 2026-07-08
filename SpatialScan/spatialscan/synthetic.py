"""Synthetic spatial video: render a known 3D scene into calibrated stereo.

Real Apple ``.mov`` clips (and ffmpeg) aren't always at hand, so this module
manufactures a ground-truth scene — a textured room with a box in it — and
renders it from a moving stereo rig with the *exact* baseline and intrinsics
the loader would otherwise read from the container. That lets the whole
pipeline (stereo → odometry → fusion) run and be checked against known
geometry.

:func:`make_spatial_video` returns a ready :class:`SpatialVideo`.
:func:`build_spatial_mov_bytes` writes a minimal spatial-video atom tree for
exercising the container parser.
"""

from __future__ import annotations

import struct

import numpy as np

from .geometry import Intrinsics
from .spatial import SpatialVideo


# ---------------------------------------------------------------------------
# Scene + renderer
# ---------------------------------------------------------------------------

def make_room_scene(seed: int = 0, n_per_face: int = 9000):
    """A colored point cloud of a room (walls/floor/ceiling/back) + a box.

    Points span x∈[-1.5,1.5], y∈[-1,1], z∈[0.8,3.2] in world coords; the camera
    starts near the origin looking down +z into the room.
    """
    rng = np.random.default_rng(seed)
    pts, cols = [], []

    def face(fixed_axis, fixed_val, a_rng, b_rng, base_color):
        n = n_per_face
        a = rng.uniform(*a_rng, n)
        b = rng.uniform(*b_rng, n)
        p = np.zeros((n, 3))
        axes = [ax for ax in range(3) if ax != fixed_axis]
        p[:, fixed_axis] = fixed_val
        p[:, axes[0]] = a
        p[:, axes[1]] = b
        # High-frequency texture so the stereo matcher has something to lock to.
        noise = rng.uniform(-40, 40, (n, 3))
        c = np.clip(np.array(base_color) + noise, 0, 255)
        pts.append(p)
        cols.append(c)

    face(0, -1.5, (0.8, 3.2), (-1, 1), (180, 120, 90))   # left wall
    face(0, 1.5, (0.8, 3.2), (-1, 1), (90, 120, 180))    # right wall
    face(1, -1.0, (-1.5, 1.5), (0.8, 3.2), (120, 160, 120))  # floor
    face(1, 1.0, (-1.5, 1.5), (0.8, 3.2), (160, 160, 120))   # ceiling
    face(2, 3.2, (-1.5, 1.5), (-1, 1), (150, 150, 150))  # back wall

    # A box floating in the middle to give the scene depth relief.
    nb = n_per_face
    bx = rng.uniform(-0.4, 0.4, nb) + 0.2
    by = rng.uniform(-0.4, 0.4, nb)
    bz = rng.uniform(1.6, 2.2, nb)
    pts.append(np.stack([bx, by, bz], axis=1))
    cols.append(np.clip(np.array([200, 80, 80]) + rng.uniform(-40, 40, (nb, 3)), 0, 255))

    P = np.concatenate(pts, 0)
    C = np.concatenate(cols, 0).astype(np.uint8)
    return P, C


def render_view(points, colors_rgb, K: Intrinsics, R, t, splat: int = 1):
    """Rasterize colored points from a camera pose -> (BGR image, depth map).

    world->camera is ``x_cam = R @ x_world + t``. Occlusion is resolved with a
    global painter pass (far points drawn first). Returns depth in metres
    (0 where nothing was hit).
    """
    R = np.asarray(R, float)
    t = np.asarray(t, float)
    cam = points @ R.T + t
    z = cam[:, 2]
    front = z > 1e-3
    cam, z = cam[front], z[front]
    col = colors_rgb[front]

    u = (K.fx * cam[:, 0] / z + K.cx)
    v = (K.fy * cam[:, 1] / z + K.cy)

    img = np.zeros((K.height, K.width, 3), np.uint8)
    depth = np.zeros((K.height, K.width), np.float32)

    # Expand each point into a small splat, gather all candidate pixels.
    offs = range(-splat, splat + 1)
    U, V, Z, COL = [], [], [], []
    ui, vi = np.rint(u).astype(np.int64), np.rint(v).astype(np.int64)
    for dy in offs:
        for dx in offs:
            U.append(ui + dx)
            V.append(vi + dy)
            Z.append(z)
            COL.append(col)
    U = np.concatenate(U); V = np.concatenate(V)
    Z = np.concatenate(Z); COL = np.concatenate(COL)
    inb = (U >= 0) & (U < K.width) & (V >= 0) & (V < K.height)
    U, V, Z, COL = U[inb], V[inb], Z[inb], COL[inb]

    order = np.argsort(-Z)  # far -> near; nearer painted last, wins
    U, V, Z, COL = U[order], V[order], Z[order], COL[order]
    img[V, U] = COL[:, ::-1]  # RGB -> BGR
    depth[V, U] = Z.astype(np.float32)
    return img, depth


def camera_trajectory(n: int, sweep: float = 0.35):
    """A gentle lateral pan (world->camera poses) looking into the room."""
    poses = []
    for i in range(n):
        f = 0.0 if n == 1 else (i / (n - 1) - 0.5) * 2.0  # -1..1
        cx = f * sweep                      # slide along world x
        C = np.array([cx, 0.0, 0.0])        # camera center
        R = np.eye(3)                       # keep looking down +z
        t = -R @ C
        poses.append((R, t))
    return poses


def make_spatial_video(n_frames: int = 6, width: int = 240, height: int = 180,
                       hfov_deg: float = 63.0, baseline_m: float = 0.06,
                       seed: int = 0) -> SpatialVideo:
    """Render a synthetic Apple-style spatial video as a :class:`SpatialVideo`.

    ``baseline_m`` defaults larger than a real iPhone (6 cm) so the synthetic
    disparities are comfortably resolvable at test resolutions.
    """
    K = Intrinsics.from_fov(width, height, hfov_deg)
    points, colors = make_room_scene(seed=seed)
    lefts, rights = [], []
    for (R, t) in camera_trajectory(n_frames):
        limg, _ = render_view(points, colors, K, R, t, splat=1)
        t_right = t - np.array([baseline_m, 0.0, 0.0])  # right eye B to the right
        rimg, _ = render_view(points, colors, K, R, t_right, splat=1)
        lefts.append(limg)
        rights.append(rimg)
    from .quicktime import SpatialMetadata
    meta = SpatialMetadata(is_mv_hevc=True, baseline_m=baseline_m, hfov_deg=hfov_deg)
    return SpatialVideo(lefts, rights, K, baseline_m, meta)


# ---------------------------------------------------------------------------
# Minimal spatial-video container for parser tests
# ---------------------------------------------------------------------------

def _box(fourcc: bytes, payload: bytes) -> bytes:
    return struct.pack(">I", 8 + len(payload)) + fourcc + payload


def build_spatial_mov_bytes(baseline_um: int = 19200, hfov_millideg: int = 63000,
                            eyes_reversed: bool = False) -> bytes:
    """A tiny ``moov`` tree carrying the spatial atoms the parser looks for.

    Mirrors the real iPhone layout: ``... > stsd > hvc1 > {vexu > eyes >
    {stri, cams > blin}, hfov}`` — note ``hfov`` is a sibling of ``vexu``
    inside the sample entry, not nested under it.
    """
    stri_flags = (0x01 | 0x02) | (0x08 if eyes_reversed else 0)  # both eyes present
    stri = _box(b"stri", struct.pack(">I", 0) + bytes([stri_flags]))
    blin = _box(b"blin", struct.pack(">I", baseline_um))
    cams = _box(b"cams", blin)
    eyes = _box(b"eyes", stri + cams)
    vexu = _box(b"vexu", eyes)
    hfov = _box(b"hfov", struct.pack(">I", hfov_millideg))

    # hvc1 is a VisualSampleEntry: 78 header bytes then child boxes.
    hvc1 = _box(b"hvc1", b"\x00" * 78 + vexu + hfov)
    stsd = _box(b"stsd", struct.pack(">II", 0, 1) + hvc1)  # fullbox + entry count
    stbl = _box(b"stbl", stsd)
    minf = _box(b"minf", stbl)
    mdia = _box(b"mdia", minf)
    trak = _box(b"trak", mdia)
    moov = _box(b"moov", trak)
    ftyp = _box(b"ftyp", b"qt  " + struct.pack(">I", 0) + b"qt  ")
    return ftyp + moov
