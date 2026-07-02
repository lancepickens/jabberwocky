#!/usr/bin/env python3
"""Generate a synthetic orbit video and run the full pipeline on it.

This is the quickest way to see splatvid work end to end without shooting
real footage:

    python scripts/make_demo.py --out demo_out
    splatvid view demo_out
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from splatvid.cli import main as cli_main  # noqa: E402
from splatvid.synthetic import make_synthetic_video  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="demo_out")
    ap.add_argument("--frames", type=int, default=36)
    ap.add_argument("--iterations", type=int, default=800)
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    video = os.path.join(args.out, "synthetic_orbit.mp4")
    print(f"Rendering synthetic orbit video -> {video}")
    make_synthetic_video(video, n_frames=args.frames, width=320, height=240)

    return cli_main(
        [
            "reconstruct", video,
            "-o", args.out,
            "--max-frames", str(args.frames),
            "--frame-size", "320",
            "--iterations", str(args.iterations),
            "--train-size", "240",
            "--max-gaussians", "20000",
            "--turntable",
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
