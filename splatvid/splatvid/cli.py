"""Command line interface: video in, gaussian splat scene out."""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
import time

log = logging.getLogger("splatvid")


def pick_device() -> str:
    """Best available torch device: cuda > mps (Apple Silicon) > cpu."""
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def _viewer_src() -> str:
    return os.path.join(os.path.dirname(__file__), "viewer.html")


def cmd_reconstruct(args: argparse.Namespace) -> int:
    import numpy as np

    from .export import save_ply, save_splat
    from .sfm import SfMError, load_reconstruction, run_sfm
    from .train import TrainConfig, train, render_turntable
    from .video import extract_frames

    os.makedirs(args.output, exist_ok=True)
    t0 = time.time()

    log.info("[1/4] Extracting frames from %s", args.video)
    frames = extract_frames(
        args.video, max_frames=args.max_frames, max_dim=args.frame_size
    )

    cam_path = os.path.join(args.output, "cameras.npz")
    if args.resume and os.path.exists(cam_path):
        log.info("[2/4] Resuming: loading cameras from %s (skipping SfM)", cam_path)
        rec = load_reconstruction(cam_path)
        log.info("Loaded %d cameras, %d points", len(rec.registered), rec.points.shape[0])
    else:
        log.info("[2/4] Structure from motion (%d frames)", len(frames.images))
        try:
            rec = run_sfm(frames.images, n_features=args.features)
        except SfMError as e:
            log.error("SfM failed: %s", e)
            log.error(
                "Tips: use a video that orbits a static, well-textured scene with "
                "steady motion; avoid pure rotation, motion blur, and reflective surfaces."
            )
            return 2

        np.savez_compressed(
            cam_path,
            focal=rec.focal, cx=rec.cx, cy=rec.cy,
            width=rec.width, height=rec.height,
            registered=np.array(rec.registered),
            Rs=np.stack([rec.poses[i][0] for i in rec.registered]),
            ts=np.stack([rec.poses[i][1] for i in rec.registered]),
            points=rec.points, point_colors=rec.point_colors,
            point_errors=rec.point_errors,
        )

    log.info("[3/4] Optimizing gaussians (%d iterations)", args.iterations)
    cfg = TrainConfig(
        iterations=args.iterations,
        train_size=args.train_size,
        max_gaussians=args.max_gaussians,
        device=args.device,
        feature_dim=16 if args.neural else 0,
        neural_iters=args.neural_iters,
    )
    shader = None
    if args.neural:
        import torch

        from .train import train_neural
        model, shader = train_neural(rec, frames.images, cfg)
        torch.save(shader.state_dict(), os.path.join(args.output, "shader.pt"))
        log.info("Saved neural shader -> shader.pt")
    else:
        model = train(rec, frames.images, cfg)

    log.info("[4/4] Exporting")
    save_ply(model, os.path.join(args.output, "scene.ply"))
    save_splat(model, os.path.join(args.output, "scene.splat"))
    shutil.copyfile(_viewer_src(), os.path.join(args.output, "index.html"))
    if args.turntable:
        render_turntable(
            model, rec, os.path.join(args.output, "turntable.mp4"),
            size=args.train_size, shader=shader,
        )

    dt = time.time() - t0
    log.info("Done in %.1f min. Output in %s", dt / 60, args.output)
    log.info(
        "View it:  cd %s && python -m http.server 8000  ->  http://localhost:8000/",
        args.output,
    )
    return 0


def cmd_view(args: argparse.Namespace) -> int:
    """Serve an output directory (or a bare .splat) in the bundled viewer."""
    import functools
    import http.server

    if os.path.isdir(args.path):
        directory = args.path
    else:
        directory = os.path.dirname(os.path.abspath(args.path)) or "."
    index = os.path.join(directory, "index.html")
    if not os.path.exists(index):
        shutil.copyfile(_viewer_src(), index)
    if os.path.isfile(args.path) and os.path.basename(args.path) != "scene.splat":
        shutil.copyfile(args.path, os.path.join(directory, "scene.splat"))
    handler = functools.partial(
        http.server.SimpleHTTPRequestHandler, directory=directory
    )
    log.info("Serving %s at http://localhost:%d/", directory, args.port)
    http.server.ThreadingHTTPServer(("127.0.0.1", args.port), handler).serve_forever()
    return 0


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(
        prog="splatvid",
        description="Reconstruct a 3D gaussian-splat scene from a video.",
    )
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("reconstruct", help="video -> gaussian splat scene")
    r.add_argument("video", help="input video file (mp4/mov/avi/...)")
    r.add_argument("-o", "--output", default="splat_out", help="output directory")
    r.add_argument("--max-frames", type=int, default=0,
                   help="frames sampled from the video (default 0 = auto from "
                        "clip length; denser sampling keeps SfM connected)")
    r.add_argument("--frame-size", type=int, default=960,
                   help="max frame dimension for SfM (default 960)")
    r.add_argument("--features", type=int, default=4000,
                   help="SIFT features per frame (default 4000)")
    r.add_argument("--iterations", type=int, default=2000,
                   help="training iterations (default 2000; more = sharper)")
    r.add_argument("--train-size", type=int, default=320,
                   help="max image dimension during optimization (default 320; "
                        "raise on a GPU)")
    r.add_argument("--max-gaussians", type=int, default=60_000)
    r.add_argument("--device", default="auto",
                   help="'cpu', 'cuda', 'mps', or 'auto' (default: best available)")
    r.add_argument("--turntable", action="store_true",
                   help="also render an orbit video of the result")
    r.add_argument("--resume", action="store_true",
                   help="reuse cameras.npz in the output dir and skip SfM "
                        "(same --max-frames/--frame-size as the original run)")
    r.add_argument("--neural", action="store_true",
                   help="deferred neural renderer: train a U-Net shader over "
                        "feature gaussians (experimental; see docs)")
    r.add_argument("--neural-iters", type=int, default=1500,
                   help="neural shader training iterations (with --neural)")
    r.set_defaults(fn=cmd_reconstruct)

    v = sub.add_parser("view", help="serve a result directory in the web viewer")
    v.add_argument("path", help="output directory or .splat file")
    v.add_argument("--port", type=int, default=8000)
    v.set_defaults(fn=cmd_view)

    args = p.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname).1s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )
    if getattr(args, "device", None) in ("auto", "mps"):
        # A rarely-implemented op should fall back to CPU rather than abort;
        # must be set before torch initializes its MPS backend.
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    if getattr(args, "device", None) == "auto":
        args.device = pick_device()
        log.info("Using device: %s", args.device)
    return args.fn(args)


if __name__ == "__main__":
    sys.exit(main())
