"""SIFT feature extraction, pairwise matching, and multi-view track building."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

log = logging.getLogger(__name__)


@dataclass
class FrameFeatures:
    keypoints: np.ndarray  # (N, 2) float32 pixel coords
    descriptors: np.ndarray  # (N, 128) float32
    colors: np.ndarray  # (N, 3) float32 RGB in [0, 1], sampled at keypoint


@dataclass
class PairMatch:
    i: int
    j: int
    matches: np.ndarray  # (M, 2) int indices into frame i / frame j keypoints


@dataclass
class Track:
    """One scene point observed in multiple frames: {frame_idx: keypoint_idx}."""

    obs: dict[int, int] = field(default_factory=dict)


def detect_features(
    images: list[np.ndarray], n_features: int = 4000
) -> list[FrameFeatures]:
    sift = cv2.SIFT_create(nfeatures=n_features)
    out: list[FrameFeatures] = []
    for img in images:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        kps, desc = sift.detectAndCompute(gray, None)
        if desc is None or len(kps) == 0:
            out.append(
                FrameFeatures(
                    keypoints=np.zeros((0, 2), np.float32),
                    descriptors=np.zeros((0, 128), np.float32),
                    colors=np.zeros((0, 3), np.float32),
                )
            )
            continue
        pts = np.array([kp.pt for kp in kps], dtype=np.float32)
        xi = np.clip(pts[:, 0].round().astype(int), 0, img.shape[1] - 1)
        yi = np.clip(pts[:, 1].round().astype(int), 0, img.shape[0] - 1)
        colors = img[yi, xi, ::-1].astype(np.float32) / 255.0  # BGR -> RGB
        out.append(FrameFeatures(keypoints=pts, descriptors=desc, colors=colors))
    log.info(
        "Detected features: median %d per frame",
        int(np.median([f.keypoints.shape[0] for f in out])),
    )
    return out


def _match_pair(
    fa: FrameFeatures, fb: FrameFeatures, ratio: float = 0.75
) -> np.ndarray:
    """Lowe-ratio kNN matching with mutual cross-check. Returns (M, 2) indices."""
    if len(fa.keypoints) < 8 or len(fb.keypoints) < 8:
        return np.zeros((0, 2), dtype=int)
    matcher = cv2.BFMatcher(cv2.NORM_L2)
    fwd = matcher.knnMatch(fa.descriptors, fb.descriptors, k=2)
    good_fwd = {}
    for pair in fwd:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good_fwd[pair[0].queryIdx] = pair[0].trainIdx
    bwd = matcher.knnMatch(fb.descriptors, fa.descriptors, k=2)
    good_bwd = {}
    for pair in bwd:
        if len(pair) == 2 and pair[0].distance < ratio * pair[1].distance:
            good_bwd[pair[0].queryIdx] = pair[0].trainIdx
    out = [
        (qa, ta)
        for qa, ta in good_fwd.items()
        if good_bwd.get(ta, -1) == qa
    ]
    return np.array(out, dtype=int) if out else np.zeros((0, 2), dtype=int)


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


def match_frames(
    features: list[FrameFeatures],
    window: int = 6,
    loop_stride: int = 15,
    min_matches: int = 30,
) -> list[PairMatch]:
    """Match sequentially-near pairs plus sparse long-range pairs.

    Video frames are temporally ordered, so most useful pairs are within a
    small window; sparse long-range pairs add loop closures for orbits.
    """
    n = len(features)
    pairs: set[tuple[int, int]] = set()
    for i in range(n):
        for d in range(1, window + 1):
            if i + d < n:
                pairs.add((i, i + d))
    for i in range(0, n, loop_stride):
        for j in range(i + window + 1, n, loop_stride):
            pairs.add((i, j))

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
