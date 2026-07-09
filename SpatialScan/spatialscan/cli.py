"""Command line interface: Apple spatial video in, scene mesh out."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import time

log = logging.getLogger("spatialscan")


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="spatialscan",
        description="Reconstruct a metric scene mesh from an Apple spatial video.")
    sub = p.add_subparsers(dest="command", required=True)

    b = sub.add_parser("build", help="build a scene mesh from a spatial video")
    b.add_argument("video", help="path to spatial video (.mov MV-HEVC, or SBS/dual)")
    b.add_argument("-o", "--output", default="scene.ply",
                   help="output mesh path (.ply/.obj); default scene.ply")
    b.add_argument("--mode", default="auto",
                   choices=["auto", "mvhevc", "sbs", "dual"],
                   help="input layout (default: auto-detect from extension)")
    b.add_argument("--max-frames", type=int, default=60)
    b.add_argument("--stride", type=int, default=1, help="use every Nth frame")
    b.add_argument("--max-dim", type=int, default=1024,
                   help="downscale so the longest side is at most this")
    b.add_argument("--baseline-mm", type=float, default=None,
                   help="override stereo baseline (mm); else read from container")
    b.add_argument("--hfov", type=float, default=None,
                   help="override horizontal FOV (deg); else read from container")
    b.add_argument("--voxel-mm", type=float, default=10.0,
                   help="TSDF voxel size in mm (smaller = finer, slower)")
    b.add_argument("--max-depth", type=float, default=12.0,
                   help="ignore stereo depth beyond this many metres")
    b.add_argument("--target-faces", type=int, default=0,
                   help="decimate the mesh to this many faces (0 = off)")
    b.add_argument("--smooth", type=int, default=0, help="Taubin smoothing passes")
    b.add_argument("-v", "--verbose", action="store_true")

    i = sub.add_parser("info", help="print the spatial metadata in a .mov")
    i.add_argument("video")

    d = sub.add_parser("demo", help="render a synthetic spatial video and mesh it")
    d.add_argument("-o", "--output", default="demo_scene.ply")
    d.add_argument("--frames", type=int, default=6)
    d.add_argument("-v", "--verbose", action="store_true")
    return p


def cmd_info(args) -> int:
    from .quicktime import extract_spatial_metadata
    from .spatial import read_moov_bytes
    # The moov (with the stereo atoms) sits at the end of an iPhone .mov; read
    # just that box instead of the whole clip.
    meta = extract_spatial_metadata(read_moov_bytes(args.video))
    print(meta.describe())
    if not meta.is_mv_hevc:
        print("note: no MV-HEVC / vexu atoms found — may not be a spatial video")
    return 0


def _run_build(video, args) -> int:
    from .fusion import FusionConfig
    from .scene import build_scene_mesh
    from .stereo import StereoConfig

    stereo_cfg = StereoConfig(max_depth_m=args.max_depth)
    fusion_cfg = FusionConfig(
        voxel_size_m=args.voxel_mm / 1000.0,
        sdf_trunc_m=max(args.voxel_mm / 1000.0 * 4, 0.01),
        depth_trunc_m=args.max_depth,
        target_faces=args.target_faces,
        smooth_iters=args.smooth)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    t0 = time.time()
    res = build_scene_mesh(video, args.output, stereo_cfg=stereo_cfg,
                           fusion_cfg=fusion_cfg)
    r = res.fusion
    if r.kind == "mesh":
        print(f"Wrote mesh {r.path}: {r.n_vertices} vertices, {r.n_faces} faces "
              f"({res.n_frames} frames, {time.time() - t0:.1f}s)")
    else:
        print(f"Wrote point cloud {r.path}: {r.n_vertices} points "
              f"(install 'spatialscan[mesh]' for a triangle mesh)")
    return 0


def cmd_build(args) -> int:
    from .spatial import SpatialVideo
    video = SpatialVideo.open(
        args.video, mode=args.mode, max_frames=args.max_frames,
        stride=args.stride, max_dim=args.max_dim,
        baseline_m=(args.baseline_mm / 1000.0) if args.baseline_mm else None,
        hfov_deg=args.hfov)
    return _run_build(video, args)


def cmd_demo(args) -> int:
    from .synthetic import make_spatial_video
    video = make_spatial_video(n_frames=args.frames)
    print(f"Rendered synthetic spatial video: {len(video)} frames, "
          f"baseline={video.baseline_m * 1000:.0f}mm")

    class _A:  # reuse the build path with demo-friendly defaults
        output = args.output
        max_depth = 12.0
        voxel_mm = 15.0
        target_faces = 0
        smooth = 0
    return _run_build(video, _A())


def main(argv=None) -> int:
    args = _build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.INFO,
        format="%(message)s")
    try:
        if args.command == "build":
            return cmd_build(args)
        if args.command == "info":
            return cmd_info(args)
        if args.command == "demo":
            return cmd_demo(args)
    except (RuntimeError, FileNotFoundError, ValueError) as e:
        log.error("error: %s", e)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
