"""End-to-end: synthetic spatial video -> scene mesh (or point-cloud fallback)."""

import numpy as np
import pytest

from spatialscan.fusion import FusionConfig
from spatialscan.scene import build_scene_mesh, compute_depths
from spatialscan.stereo import StereoConfig
from spatialscan.synthetic import make_spatial_video


def _has_open3d():
    try:
        import open3d  # noqa: F401
        return True
    except Exception:
        return False


def test_compute_depths_all_frames():
    video = make_spatial_video(n_frames=4, baseline_m=0.08)
    colors, depths = compute_depths(video)
    assert len(colors) == len(depths) == 4
    for d in depths:
        assert (d > 0).mean() > 0.15  # meaningful coverage each frame


def test_build_scene_writes_output(tmp_path):
    video = make_spatial_video(n_frames=5, width=200, height=150, baseline_m=0.08)
    out = tmp_path / "scene.ply"
    res = build_scene_mesh(
        video, str(out),
        stereo_cfg=StereoConfig(num_disparities=128),
        fusion_cfg=FusionConfig(voxel_size_m=0.02, sdf_trunc_m=0.08))

    assert res.n_frames == 5
    assert res.fusion.path is not None
    written = tmp_path / res.fusion.path.split("/")[-1]
    assert written.exists() and written.stat().st_size > 0

    if _has_open3d():
        assert res.fusion.kind == "mesh"
        assert res.fusion.n_faces > 100, "TSDF should produce a real surface"
    else:
        assert res.fusion.kind == "pointcloud"
        assert res.fusion.n_vertices > 100


@pytest.mark.skipif(not _has_open3d(), reason="mesh geometry check needs open3d")
def test_mesh_extent_is_metric(tmp_path):
    import open3d as o3d

    video = make_spatial_video(n_frames=6, width=220, height=165, baseline_m=0.08)
    out = tmp_path / "scene.ply"
    build_scene_mesh(video, str(out),
                     fusion_cfg=FusionConfig(voxel_size_m=0.02, sdf_trunc_m=0.08))
    mesh = o3d.io.read_triangle_mesh(str(out))
    aabb = mesh.get_axis_aligned_bounding_box()
    extent = aabb.get_extent()
    # The room is ~3m wide and ~2m tall; a metric reconstruction should be in
    # that ballpark (loose bounds: odometry + TSDF are approximate).
    assert 1.0 < extent[0] < 6.0, f"x-extent {extent[0]:.2f}m not room-scale"
    assert 0.5 < extent[1] < 5.0, f"y-extent {extent[1]:.2f}m not room-scale"
