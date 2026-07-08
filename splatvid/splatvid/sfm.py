"""Incremental structure-from-motion, written from scratch on OpenCV primitives.

Pipeline: pick a well-conditioned initial pair -> essential-matrix
initialization -> alternate PnP registration / triangulation for the
remaining frames -> periodic and final bundle adjustment.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import cv2
import numpy as np

from .ba import bundle_adjust
from .features import (
    FrameFeatures,
    PairMatch,
    Track,
    _match_pair,
    _verify_pair,
    build_tracks,
    detect_features,
    match_frames,
)
from .geometry import (
    camera_center,
    make_K,
    project_points,
    triangulate_point,
    triangulation_angle_deg,
)

log = logging.getLogger(__name__)


@dataclass
class Reconstruction:
    """SfM output: shared intrinsics, per-frame poses, and a sparse cloud."""

    focal: float
    cx: float
    cy: float
    width: int
    height: int
    poses: dict[int, tuple[np.ndarray, np.ndarray]]  # frame -> (R, t), world-to-cam
    points: np.ndarray  # (P, 3)
    point_colors: np.ndarray  # (P, 3) RGB in [0, 1]
    point_errors: np.ndarray  # (P,) mean reprojection error in px
    registered: list[int] = field(default_factory=list)

    @property
    def K(self) -> np.ndarray:
        return make_K(self.focal, self.cx, self.cy)

    def camera_centers(self) -> np.ndarray:
        return np.stack([camera_center(*self.poses[i]) for i in self.registered])

    def scene_extent(self) -> float:
        """Radius of the camera rig, used to scale training hyperparameters."""
        C = self.camera_centers()
        return float(np.linalg.norm(C - C.mean(axis=0), axis=1).max()) or 1.0


class SfMError(RuntimeError):
    pass


def load_reconstruction(path: str) -> Reconstruction:
    """Rebuild a Reconstruction from a ``cameras.npz`` written by the CLI.

    Lets training resume without re-running SfM (see ``splatvid reconstruct
    --resume``). The frame indices in ``registered`` refer to the extracted
    frame order, so resuming requires the same ``--max-frames`` /
    ``--frame-size`` settings that produced the file.
    """
    d = np.load(path)
    registered = [int(i) for i in d["registered"]]
    poses = {
        registered[k]: (d["Rs"][k], d["ts"][k]) for k in range(len(registered))
    }
    n_pts = d["points"].shape[0]
    errors = d["point_errors"] if "point_errors" in d.files else np.zeros(n_pts)
    return Reconstruction(
        focal=float(d["focal"]), cx=float(d["cx"]), cy=float(d["cy"]),
        width=int(d["width"]), height=int(d["height"]),
        poses=poses, points=d["points"], point_colors=d["point_colors"],
        point_errors=errors, registered=registered,
    )


@dataclass
class _State:
    K: np.ndarray
    focal: float
    cx: float
    cy: float
    poses: dict[int, tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)
    # track index -> 3D point (None until triangulated)
    track_pts: dict[int, np.ndarray] = field(default_factory=dict)


def _pair_score(
    fa: FrameFeatures, fb: FrameFeatures, pm: PairMatch, K: np.ndarray
) -> tuple[float, tuple | None]:
    """Score an init pair: many matches AND real parallax (not homography-like)."""
    pa = fa.keypoints[pm.matches[:, 0]].astype(np.float64)
    pb = fb.keypoints[pm.matches[:, 1]].astype(np.float64)
    if len(pa) < 50:
        return -1.0, None
    H, hmask = cv2.findHomography(pa, pb, cv2.RANSAC, 4.0)
    E, emask = cv2.findEssentialMat(pa, pb, K, cv2.RANSAC, 0.999, 1.5)
    if E is None or emask is None or E.shape != (3, 3):
        return -1.0, None
    e_inl = int(emask.sum())
    h_inl = int(hmask.sum()) if hmask is not None else 0
    if e_inl < 40:
        return -1.0, None
    # Homography-dominated pairs (pure rotation / planar) are bad seeds.
    h_ratio = h_inl / max(e_inl, 1)
    score = e_inl * max(0.0, 1.0 - 0.7 * max(0.0, h_ratio - 0.7))
    return score, (E, emask.ravel().astype(bool), pa, pb)


def _seed_from_pair(
    state: _State,
    tracks: list[Track],
    track_of_kp: dict[tuple[int, int], int],
    features: list[FrameFeatures],
    pm: PairMatch,
    E: np.ndarray,
    emask: np.ndarray,
    pa: np.ndarray,
    pb: np.ndarray,
) -> tuple[dict[int, tuple[np.ndarray, np.ndarray]], dict[int, np.ndarray]]:
    """Two-view init from one candidate pair: recover pose, triangulate.

    Returns ``(poses, seed_points)`` without mutating anything the caller
    has not already staged; ``state.poses`` is set to the candidate poses
    for the duration so ``_accept_point`` can gate on them.
    """
    i, j = pm.i, pm.j
    _, R, t, pose_mask = cv2.recoverPose(E, pa[emask], pb[emask], state.K)
    poses = {i: (np.eye(3), np.zeros(3)), j: (R, t.ravel())}
    state.poses = poses  # so _accept_point sees this candidate's geometry

    inl_idx = np.nonzero(emask)[0]
    ok_pose = pose_mask.ravel() > 0
    seed: dict[int, np.ndarray] = {}
    for local_k, m_idx in enumerate(inl_idx):
        if not ok_pose[local_k]:
            continue
        a, b = pm.matches[m_idx]
        ti = track_of_kp.get((i, int(a)))
        if ti is None or ti in seed:
            continue
        X = triangulate_point(
            [(pa[m_idx], *poses[i]), (pb[m_idx], *poses[j])], state.K
        )
        if X is None:
            continue
        if _accept_point(X, ti, tracks[ti], state, features, max_err=4.0, min_angle=1.0):
            seed[ti] = X
    return poses, seed


def _init_pair(
    state: _State,
    tracks: list[Track],
    track_of_kp: dict[tuple[int, int], int],
    features: list[FrameFeatures],
    pair_matches: list[PairMatch],
    min_seed: int = 30,
) -> tuple[int, int]:
    """Choose and build the two-view seed reconstruction.

    Ranking candidate pairs by match count alone favours temporally
    adjacent frames, whose short baseline triangulates poorly (few points
    clear the cheirality/reprojection/parallax gates). Instead we rank by
    the pair score, then pick the pair that yields the largest *realized*
    seed cloud, so baseline/parallax — not raw match count — decides.
    """
    scored = []
    for pm in pair_matches:
        score, aux = _pair_score(features[pm.i], features[pm.j], pm, state.K)
        if aux is not None:
            scored.append((score, pm, aux))
    if not scored:
        raise SfMError("No frame pair suitable for initialization (too little parallax?)")
    scored.sort(key=lambda r: -r[0])

    best: tuple[int, int, int, dict, dict] | None = None
    for _score, pm, (E, emask, pa, pb) in scored:
        poses, seed = _seed_from_pair(
            state, tracks, track_of_kp, features, pm, E, emask, pa, pb
        )
        if best is None or len(seed) > best[0]:
            best = (len(seed), pm.i, pm.j, poses, seed)
        # A comfortably large seed is plenty to bootstrap; stop early.
        if len(seed) >= 3 * min_seed:
            break

    n_seed, i, j, poses, seed = best
    if n_seed < min_seed:
        raise SfMError(
            f"Best init pair ({i},{j}) triangulated only {n_seed} points "
            f"(need {min_seed}); the video likely has too little camera "
            "translation (parallax). Move the camera through the scene, "
            "don't pan in place."
        )
    state.poses = poses
    state.track_pts = seed
    log.info("Init pair (%d, %d): %d seed points", i, j, n_seed)
    return i, j


def _accept_point(
    X: np.ndarray,
    track_idx: int,
    track: Track,
    state: _State,
    features: list[FrameFeatures],
    max_err: float,
    min_angle: float,
) -> bool:
    """Cheirality + reprojection + parallax gate for a candidate 3D point."""
    cams = [c for c in track.obs if c in state.poses]
    if len(cams) < 2:
        return False
    errs = []
    for c in cams:
        R, t = state.poses[c]
        uv, z = project_points(X[None], R, t, state.K)
        if z[0] <= 1e-6:
            return False
        kp = features[c].keypoints[track.obs[c]]
        errs.append(np.linalg.norm(uv[0] - kp))
    if max(errs) > max_err:
        return False
    centers = [camera_center(*state.poses[c]) for c in cams]
    best_angle = 0.0
    for a in range(len(centers)):
        for b in range(a + 1, len(centers)):
            best_angle = max(best_angle, triangulation_angle_deg(X, centers[a], centers[b]))
            if best_angle >= min_angle:
                return True
    return best_angle >= min_angle


def _triangulate_new(
    state: _State,
    tracks: list[Track],
    features: list[FrameFeatures],
    max_err: float = 4.0,
    min_angle: float = 1.0,
) -> int:
    """Triangulate all tracks with >=2 registered views and no 3D point yet."""
    n_new = 0
    for ti, tr in enumerate(tracks):
        if ti in state.track_pts:
            continue
        obs = [
            (features[c].keypoints[k].astype(np.float64), *state.poses[c])
            for c, k in tr.obs.items()
            if c in state.poses
        ]
        if len(obs) < 2:
            continue
        X = triangulate_point(obs, state.K)
        if X is None:
            continue
        if _accept_point(X, ti, tr, state, features, max_err, min_angle):
            state.track_pts[ti] = X
            n_new += 1
    return n_new


def _register_next(
    state: _State,
    tracks: list[Track],
    features: list[FrameFeatures],
    candidates: list[int],
    min_obs: int = 8,
    min_inliers: int = 8,
) -> int | None:
    """Register the next unposed frame via PnP, best-supported first.

    Candidates are tried in order of how many already-triangulated points
    they observe; a PnP failure falls through to the next candidate rather
    than aborting registration. Because the caller re-invokes this after
    every successful registration (each of which triangulates more points),
    a frame that is too weak now gets retried once its neighbours have
    filled in more of the cloud. Registration only stops when no remaining
    frame has even ``min_obs`` correspondences or every PnP fails.
    """
    counts: dict[int, list[int]] = {c: [] for c in candidates}
    for ti in state.track_pts:
        for c, _k in tracks[ti].obs.items():
            if c in counts:
                counts[c].append(ti)
    order = sorted(candidates, key=lambda c: -len(counts[c]))
    for cand in order:
        tids = counts[cand]
        if len(tids) < min_obs:
            break  # sorted desc: everything after this is weaker still
        obj = np.array([state.track_pts[ti] for ti in tids])
        img = np.array(
            [features[cand].keypoints[tracks[ti].obs[cand]] for ti in tids],
            dtype=np.float64,
        )
        ok, rvec, tvec, inl = cv2.solvePnPRansac(
            obj, img, state.K, None,
            reprojectionError=4.0, iterationsCount=300, confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok or inl is None or len(inl) < min_inliers:
            continue  # try the next-best candidate instead of giving up
        R, _ = cv2.Rodrigues(rvec)
        state.poses[cand] = (R, tvec.ravel())
        log.info(
            "Registered frame %d via PnP (%d/%d inliers)", cand, len(inl), len(tids)
        )
        return cand
    return None


def _filter_points(
    state: _State,
    tracks: list[Track],
    features: list[FrameFeatures],
    max_err: float = 3.0,
) -> int:
    """Drop triangulated points that no longer reproject well."""
    drop = []
    for ti, X in state.track_pts.items():
        errs = []
        behind = False
        for c, k in tracks[ti].obs.items():
            if c not in state.poses:
                continue
            R, t = state.poses[c]
            uv, z = project_points(X[None], R, t, state.K)
            if z[0] <= 1e-6:
                behind = True
                break
            errs.append(np.linalg.norm(uv[0] - features[c].keypoints[k]))
        if behind or (errs and np.mean(errs) > max_err):
            drop.append(ti)
    for ti in drop:
        del state.track_pts[ti]
    return len(drop)


def _connected_components(
    n_frames: int, pair_matches: list[PairMatch]
) -> list[set[int]]:
    """Connected components of the view graph, largest first.

    Real handheld video often breaks into segments with little overlap
    (fast motion, blur, or the camera looking at unrelated things), which
    leaves the pairwise-match graph fragmented. Incremental SfM can only
    grow one connected component at a time.
    """
    adj: dict[int, set[int]] = {i: set() for i in range(n_frames)}
    for pm in pair_matches:
        adj[pm.i].add(pm.j)
        adj[pm.j].add(pm.i)
    seen: set[int] = set()
    comps: list[set[int]] = []
    for start in range(n_frames):
        if start in seen:
            continue
        stack = [start]
        comp: set[int] = set()
        while stack:
            x = stack.pop()
            if x in seen:
                continue
            seen.add(x)
            comp.add(x)
            stack.extend(adj[x] - seen)
        comps.append(comp)
    comps.sort(key=len, reverse=True)
    return comps


def _largest_component(n_frames: int, pair_matches: list[PairMatch]) -> set[int]:
    comps = _connected_components(n_frames, pair_matches)
    return comps[0] if comps else set()


def _bridge_components(
    features: list[FrameFeatures],
    components: list[set[int]],
    existing: set[tuple[int, int]],
    sample: int = 8,
    min_matches: int = 30,
) -> list[PairMatch]:
    """Try to connect separate components with extra cross-component matches.

    The default matcher only compares frames in a small temporal window plus
    a sparse loop-closure grid, so two segments that *do* overlap in the
    scene can still land in different components if the overlap fell between
    sampled pairs. Here we match a spread of frames from every smaller
    component against a spread from the largest one; any verified pair merges
    them. Truly non-overlapping segments simply find no matches (correct).
    Bounded to ``sample^2`` matches per component so it stays cheap.
    """
    if len(components) < 2:
        return []

    def spread(comp: set[int]) -> list[int]:
        s = sorted(comp)
        if len(s) <= sample:
            return s
        idx = np.linspace(0, len(s) - 1, sample).astype(int)
        return [s[k] for k in sorted(set(idx))]

    main = spread(components[0])
    new_pairs: list[PairMatch] = []
    for comp in components[1:]:
        for a in spread(comp):
            for b in main:
                lo, hi = (a, b) if a < b else (b, a)
                if (lo, hi) in existing:
                    continue
                m = _match_pair(features[lo], features[hi])
                m = _verify_pair(features[lo], features[hi], m)
                if len(m) >= min_matches:
                    new_pairs.append(PairMatch(i=lo, j=hi, matches=m))
    return new_pairs


def _run_ba(
    state: _State,
    tracks: list[Track],
    features: list[FrameFeatures],
    refine_focal: bool,
    max_nfev: int = 30,
) -> None:
    tids = sorted(state.track_pts)
    if len(tids) < 20:
        return
    pts = np.stack([state.track_pts[ti] for ti in tids])
    observations = []
    for pi, ti in enumerate(tids):
        for c, k in tracks[ti].obs.items():
            if c in state.poses:
                kp = features[c].keypoints[k]
                observations.append((c, pi, float(kp[0]), float(kp[1])))
    focal, poses, pts = bundle_adjust(
        state.focal, state.cx, state.cy, state.poses, pts, observations,
        refine_focal=refine_focal, max_nfev=max_nfev,
    )
    state.focal = focal
    state.K = make_K(focal, state.cx, state.cy)
    state.poses = poses
    for pi, ti in enumerate(tids):
        state.track_pts[ti] = pts[pi]


def run_sfm(
    images: list[np.ndarray],
    n_features: int = 4000,
    match_window: int = 6,
    ba_every: int = 5,
    min_track_len: int = 2,
) -> Reconstruction:
    """Full SfM pipeline from a list of BGR frames."""
    h, w = images[0].shape[:2]
    focal0 = 1.2 * max(w, h)
    cx, cy = w / 2.0, h / 2.0
    state = _State(K=make_K(focal0, cx, cy), focal=focal0, cx=cx, cy=cy)

    # Dense captures don't need thousands of keypoints per frame — more VIEWS
    # carry the information, and LightGlue's cost is ~O(kpts^2) per pair, so a
    # dense run at full keypoints spends enormous time matching. Cap keypoints
    # down with frame count (floored) so ~50%-of-video runs stay tractable.
    n = len(images)
    eff_features = min(n_features, max(1536, round(n_features * 150.0 / n))) if n > 150 else n_features
    if eff_features < n_features:
        log.info(
            "Dense capture (%d frames): capping keypoints %d -> %d for tractable matching",
            n, n_features, eff_features,
        )
    features = detect_features(images, n_features=eff_features)
    pair_matches = match_frames(features, window=match_window)
    if not pair_matches:
        raise SfMError("No verifiable frame pairs found; is the video textured and moving?")

    # Try to merge fragmented segments before committing to a component:
    # match extra cross-component pairs the windowed/loop matcher skipped.
    comps = _connected_components(len(images), pair_matches)
    if len(comps) > 1:
        existing = {(pm.i, pm.j) for pm in pair_matches}
        bridges = _bridge_components(features, comps, existing)
        if bridges:
            pair_matches = pair_matches + bridges
            merged = len(comps) - len(_connected_components(len(images), pair_matches))
            log.info(
                "Bridged fragmented view graph: +%d pairs, %d fewer components",
                len(bridges), merged,
            )

    tracks = build_tracks(features, pair_matches, min_length=min_track_len)
    if len(tracks) < 100:
        raise SfMError(f"Only {len(tracks)} feature tracks; not enough to reconstruct.")

    track_of_kp: dict[tuple[int, int], int] = {}
    for ti, tr in enumerate(tracks):
        for c, k in tr.obs.items():
            track_of_kp[(c, k)] = ti

    # Reconstruct the largest connected component of the view graph; a
    # fragmented handheld video otherwise strands most frames on islands
    # the seed pair cannot reach.
    component = _largest_component(len(images), pair_matches)
    if len(component) < 3:
        raise SfMError(
            f"Largest connected set of overlapping frames is only "
            f"{len(component)}; the video does not have enough overlap "
            "between frames to reconstruct."
        )
    if len(component) < len(images):
        log.info(
            "View graph fragmented; reconstructing largest component "
            "(%d of %d frames)", len(component), len(images),
        )
    init_pairs = [pm for pm in pair_matches if pm.i in component and pm.j in component]

    _init_pair(state, tracks, track_of_kp, features, init_pairs)
    _triangulate_new(state, tracks, features)
    _run_ba(state, tracks, features, refine_focal=False, max_nfev=20)

    since_ba = 0
    while True:
        remaining = [c for c in component if c not in state.poses]
        if not remaining:
            break
        cand = _register_next(state, tracks, features, remaining)
        if cand is None:
            log.info("Stopping registration: %d frames unregistered", len(remaining))
            break
        _triangulate_new(state, tracks, features)
        since_ba += 1
        if since_ba >= ba_every:
            _run_ba(state, tracks, features, refine_focal=True)
            _filter_points(state, tracks, features)
            since_ba = 0

    _run_ba(state, tracks, features, refine_focal=True, max_nfev=60)
    n_drop = _filter_points(state, tracks, features)
    _run_ba(state, tracks, features, refine_focal=True, max_nfev=40)
    log.info("Final filter dropped %d points", n_drop)

    if len(state.poses) < 2 or len(state.track_pts) < 50:
        raise SfMError(
            f"Reconstruction too small: {len(state.poses)} cameras, "
            f"{len(state.track_pts)} points."
        )

    tids = sorted(state.track_pts)
    pts = np.stack([state.track_pts[ti] for ti in tids])
    colors = np.zeros((len(tids), 3), dtype=np.float64)
    errors = np.zeros(len(tids))
    for pi, ti in enumerate(tids):
        cols, errs = [], []
        for c, k in tracks[ti].obs.items():
            if c not in state.poses:
                continue
            cols.append(features[c].colors[k])
            R, t = state.poses[c]
            uv, _ = project_points(pts[pi][None], R, t, state.K)
            errs.append(np.linalg.norm(uv[0] - features[c].keypoints[k]))
        colors[pi] = np.mean(cols, axis=0) if cols else 0.5
        errors[pi] = np.mean(errs) if errs else 0.0

    rec = Reconstruction(
        focal=state.focal, cx=cx, cy=cy, width=w, height=h,
        poses=state.poses, points=pts, point_colors=colors,
        point_errors=errors, registered=sorted(state.poses),
    )
    log.info(
        "SfM done: %d/%d cameras, %d points, focal %.1f px, mean err %.2f px",
        len(rec.registered), len(images), len(pts), rec.focal, errors.mean(),
    )
    return rec
