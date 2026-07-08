"""SOTA feature extraction + matching: DISK keypoints + LightGlue matcher.

Replaces the old SIFT + brute-force-ratio pipeline. DISK (learned detector +
descriptor, Tyszkiewicz 2020) and LightGlue (learned graph matcher, Lindenberger
2023) run in pure PyTorch via kornia, so they work on CPU / CUDA / Apple-Silicon
MPS with no native build. They give many more, more accurate, more repeatable
correspondences than SIFT+BF — especially on the repetitive textures and wide
baselines of a handheld orbit — which lifts pose accuracy, track density, and
every downstream stage (init cloud, splat, mesh).

Public surface (unchanged, so ``sfm.py`` is untouched): ``FrameFeatures``,
``PairMatch``, ``Track``, ``detect_features``, ``match_frames``,
``build_tracks``, and the ``_match_pair`` / ``_verify_pair`` helpers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np
import torch

log = logging.getLogger(__name__)

# Lazily-built, cached models keyed by (name, device) — loading DISK/LightGlue
# weights is expensive and they are reused across every frame / pair.
_MODELS: dict[tuple[str, str], object] = {}


def _pick_device(pref: str | None = None) -> str:
    if pref:
        return pref
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _get_disk(device: str):
    key = ("disk", device)
    if key not in _MODELS:
        from kornia.feature import DISK  # noqa: PLC0415

        _MODELS[key] = DISK.from_pretrained("depth").to(device).eval()
        log.info("Loaded DISK detector on %s", device)
    return _MODELS[key]


def _get_lightglue(device: str):
    key = ("lightglue", device)
    if key not in _MODELS:
        from kornia.feature import LightGlueMatcher  # noqa: PLC0415

        _MODELS[key] = LightGlueMatcher("disk").to(device).eval()
        log.info("Loaded LightGlue matcher on %s", device)
    return _MODELS[key]


@dataclass
class FrameFeatures:
    keypoints: np.ndarray  # (N, 2) float32 pixel coords
    descriptors: np.ndarray  # (N, 128) float32 DISK descriptors
    colors: np.ndarray  # (N, 3) float32 RGB in [0, 1], sampled at keypoint
    hw: tuple[int, int] = (0, 0)  # image (H, W) — LightGlue positional encoding


@dataclass
class PairMatch:
    i: int
    j: int
    matches: np.ndarray  # (M, 2) int indices into frame i / frame j keypoints


@dataclass
class Track:
    """One scene point observed in multiple frames: {frame_idx: keypoint_idx}."""

    obs: dict[int, int] = field(default_factory=dict)


def _empty(hw: tuple[int, int]) -> FrameFeatures:
    return FrameFeatures(
        keypoints=np.zeros((0, 2), np.float32),
        descriptors=np.zeros((0, 128), np.float32),
        colors=np.zeros((0, 3), np.float32),
        hw=hw,
    )


def detect_features(
    images: list[np.ndarray], n_features: int = 4096, device: str | None = None
) -> list[FrameFeatures]:
    """Detect DISK keypoints + descriptors for every BGR frame."""
    device = _pick_device(device)
    disk = _get_disk(device)
    out: list[FrameFeatures] = []
    for img in images:
        h, w = img.shape[:2]
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        t = torch.from_numpy(rgb).to(device).permute(2, 0, 1).float()[None] / 255.0
        with torch.no_grad():
            feats = disk(
                t, n=n_features, window_size=5,
                score_threshold=0.0, pad_if_not_divisible=True,
            )[0]
        kp = feats.keypoints.float().cpu().numpy()
        desc = feats.descriptors.float().cpu().numpy()
        if kp.shape[0] == 0:
            out.append(_empty((h, w)))
            continue
        xi = np.clip(kp[:, 0].round().astype(int), 0, w - 1)
        yi = np.clip(kp[:, 1].round().astype(int), 0, h - 1)
        colors = img[yi, xi, ::-1].astype(np.float32) / 255.0  # BGR -> RGB
        out.append(
            FrameFeatures(
                keypoints=kp.astype(np.float32),
                descriptors=desc.astype(np.float32),
                colors=colors,
                hw=(h, w),
            )
        )
    log.info(
        "DISK features: median %d per frame",
        int(np.median([f.keypoints.shape[0] for f in out])) if out else 0,
    )
    return out


def _match_pair(fa: FrameFeatures, fb: FrameFeatures, device: str | None = None) -> np.ndarray:
    """LightGlue match between two frames. Returns (M, 2) keypoint indices."""
    if len(fa.keypoints) < 8 or len(fb.keypoints) < 8:
        return np.zeros((0, 2), dtype=int)
    from kornia.feature import laf_from_center_scale_ori  # noqa: PLC0415

    device = _pick_device(device)
    lg = _get_lightglue(device)
    d0 = torch.from_numpy(fa.descriptors).to(device)
    d1 = torch.from_numpy(fb.descriptors).to(device)
    kp0 = torch.from_numpy(fa.keypoints).to(device)[None]
    kp1 = torch.from_numpy(fb.keypoints).to(device)[None]
    laf0 = laf_from_center_scale_ori(kp0)
    laf1 = laf_from_center_scale_ori(kp1)
    hw0 = torch.tensor(fa.hw, device=device)
    hw1 = torch.tensor(fb.hw, device=device)
    with torch.no_grad():
        _, idxs = lg(d0, d1, laf0, laf1, hw1=hw0, hw2=hw1)
    if idxs is None or idxs.numel() == 0:
        return np.zeros((0, 2), dtype=int)
    return idxs.cpu().numpy().astype(int)


def _verify_pair(
    fa: FrameFeatures, fb: FrameFeatures, matches: np.ndarray
) -> np.ndarray:
    """Keep only matches consistent with a fundamental matrix (RANSAC)."""
    if len(matches) < 12:
        return np.zeros((0, 2), dtype=int)
    pa = fa.keypoints[matches[:, 0]]
    pb = fb.keypoints[matches[:, 1]]
    F, mask = cv2.findFundamentalMat(pa, pb, cv2.FM_RANSAC, 2.0, 0.999)
    if F is None or mask is None:
        return np.zeros((0, 2), dtype=int)
    return matches[mask.ravel().astype(bool)]


def _global_descriptors(features: list[FrameFeatures]) -> np.ndarray:
    """One L2-normalized global vector per frame (mean-pooled descriptors).

    A cheap image-retrieval signature: frames that see the same surface have
    similar mean descriptors, so cosine similarity surfaces loop-closure
    candidates that a fixed temporal stride misses.
    """
    g = np.zeros((len(features), 128), np.float32)
    for i, f in enumerate(features):
        if len(f.descriptors):
            v = f.descriptors.mean(0)
            g[i] = v / (np.linalg.norm(v) + 1e-8)
    return g


def _candidate_pairs(
    features: list[FrameFeatures], window: int, loop_stride: int, retrieval_k: int
) -> set[tuple[int, int]]:
    """Frame pairs to match — temporal window + retrieval loop-closures + strided grid.

    **Frame-density-aware:** as the video is sampled more densely, adjacent frames
    become redundant (tiny baseline), so the temporal window and retrieval-k shrink
    and the loop stride widens with ``n``. This keeps the candidate count ~O(n)
    instead of O(n·window) — without it, a 400-frame capture spends most of its
    matching time on near-identical neighbours (the dense-frame bottleneck). Small
    captures (n ≲ 80) are unchanged.
    """
    n = len(features)
    ew = max(2, round(window * min(1.0, 80.0 / n))) if n else window
    er = max(3, round(retrieval_k * min(1.0, 120.0 / n))) if retrieval_k else 0
    es = max(loop_stride, n // 28)
    pairs: set[tuple[int, int]] = set()
    for i in range(n):
        for d in range(1, ew + 1):
            if i + d < n:
                pairs.add((i, i + d))
    if er > 0 and n > ew + 2:
        g = _global_descriptors(features)
        sim = g @ g.T
        for i in range(n):
            added = 0
            for j in np.argsort(-sim[i]):
                j = int(j)
                if abs(j - i) <= ew or j == i:
                    continue
                lo, hi = (i, j) if i < j else (j, i)
                if (lo, hi) not in pairs:
                    pairs.add((lo, hi))
                    added += 1
                if added >= er:
                    break
    for i in range(0, n, es):
        for j in range(i + ew + 1, n, es):
            pairs.add((i, j))
    if n > 80:
        log.info("Dense capture (%d frames): window=%d retrieval=%d stride=%d", n, ew, er, es)
    return pairs


def match_frames(
    features: list[FrameFeatures],
    window: int = 6,
    loop_stride: int = 15,
    min_matches: int = 30,
    retrieval_k: int = 5,
) -> list[PairMatch]:
    """Match sequential pairs + retrieval-based loop-closure pairs with LightGlue.

    Candidate pairs are (a) every frame to its next ``window`` neighbours
    (video is temporally ordered) and (b) for each frame, its ``retrieval_k``
    most globally-similar non-adjacent frames — real loop closures an orbit
    produces, found by content rather than a blind stride. Each candidate is
    matched with LightGlue then geometrically verified.
    """
    pairs = _candidate_pairs(features, window, loop_stride, retrieval_k)
    out: list[PairMatch] = []
    for i, j in sorted(pairs):
        m = _match_pair(features[i], features[j])
        m = _verify_pair(features[i], features[j], m)
        if len(m) >= min_matches:
            out.append(PairMatch(i=i, j=j, matches=m))
    log.info("Verified %d frame pairs (of %d candidates)", len(out), len(pairs))
    return out


class _UnionFind:
    def __init__(self) -> None:
        self.parent: dict[tuple[int, int], tuple[int, int]] = {}

    def find(self, a: tuple[int, int]) -> tuple[int, int]:
        self.parent.setdefault(a, a)
        root = a
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[a] != root:  # path compression
            self.parent[a], a = root, self.parent[a]
        return root

    def union(self, a: tuple[int, int], b: tuple[int, int]) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


def build_tracks(
    features: list[FrameFeatures], pair_matches: list[PairMatch], min_length: int = 2
) -> list[Track]:
    """Merge pairwise matches into multi-view tracks via union-find.

    Tracks in which one frame contributes two different keypoints are
    inconsistent (a matching error somewhere) and are dropped.
    """
    uf = _UnionFind()
    for pm in pair_matches:
        for a, b in pm.matches:
            uf.union((pm.i, int(a)), (pm.j, int(b)))

    groups: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for node in list(uf.parent):
        groups.setdefault(uf.find(node), []).append(node)

    tracks: list[Track] = []
    for members in groups.values():
        if len(members) < min_length:
            continue
        obs: dict[int, int] = {}
        consistent = True
        for frame_idx, kp_idx in members:
            if frame_idx in obs and obs[frame_idx] != kp_idx:
                consistent = False
                break
            obs[frame_idx] = kp_idx
        if consistent and len(obs) >= min_length:
            tracks.append(Track(obs=obs))
    log.info("Built %d tracks", len(tracks))
    return tracks
