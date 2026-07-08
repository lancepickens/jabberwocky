"""Stereo matching recovers metric depth from a synthetic pair."""

import numpy as np

from spatialscan.stereo import (
    StereoConfig, disparity_to_depth, stereo_depth,
)
from spatialscan.synthetic import make_spatial_video


def test_disparity_to_depth_formula():
    disp = np.array([[10.0, 20.0, 0.0]], np.float32)
    depth = disparity_to_depth(disp, fx=100.0, baseline_m=0.5, max_depth_m=100.0)
    # Z = fx*B/d = 50/d
    assert abs(depth[0, 0] - 5.0) < 1e-4
    assert abs(depth[0, 1] - 2.5) < 1e-4
    assert depth[0, 2] == 0.0  # invalid disparity -> no depth


def test_stereo_depth_matches_scene_range():
    video = make_spatial_video(n_frames=1, width=240, height=180, baseline_m=0.08)
    frame = next(iter(video))
    depth = stereo_depth(frame.left, frame.right, video.intrinsics,
                         video.baseline_m, StereoConfig(num_disparities=128))
    valid = depth[depth > 0]
    # The synthetic room spans ~0.8m (front box) to ~3.2m (back wall).
    assert valid.size > 0.2 * depth.size, "stereo should cover much of the frame"
    med = float(np.median(valid))
    assert 1.0 < med < 3.5, f"median depth {med:.2f}m outside the room"
    assert valid.min() > 0.4 and valid.max() < 4.5
