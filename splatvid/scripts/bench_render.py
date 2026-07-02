#!/usr/bin/env python3
"""Benchmark the differentiable rasterizer on any torch device.

Use this to decide whether MPS/CUDA beats CPU on your machine for your
scene sizes, e.g. on Apple Silicon:

    python scripts/bench_render.py --device cpu
    python scripts/bench_render.py --device mps
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch  # noqa: E402

from splatvid.cli import pick_device  # noqa: E402
from splatvid.losses import image_loss  # noqa: E402
from splatvid.render import render  # noqa: E402
from splatvid.synthetic import make_scene, orbit_pose  # noqa: E402


def sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()
    elif device == "mps":
        torch.mps.synchronize()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--gaussians", type=int, default=20_000)
    ap.add_argument("--size", type=int, default=320)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--backward", action="store_true", default=True,
                    help="include backward pass (training-shaped work)")
    args = ap.parse_args()

    device = pick_device() if args.device == "auto" else args.device
    w, h = args.size, int(args.size * 0.75)
    scene = make_scene(n=args.gaussians)
    params = {}
    for k in ("xyz", "scale", "quat", "rgb", "opacity"):
        params[k] = scene[k].to(device).requires_grad_(True)
    target = torch.rand(h, w, 3, device=device)

    R, t = orbit_pose(0.7)
    Rt = torch.tensor(R, dtype=torch.float32, device=device)
    tt = torch.tensor(t, dtype=torch.float32, device=device)
    focal = 1.1 * w

    def step():
        img, _ = render(
            params["xyz"], params["scale"].exp().clamp(max=1.0),
            params["quat"], params["rgb"], params["opacity"],
            Rt, tt, focal, w / 2, h / 2, w, h,
        )
        if args.backward:
            loss = image_loss(img, target)
            for p in params.values():
                p.grad = None
            loss.backward()

    step()  # warmup (also triggers any MPS shader compilation)
    sync(device)
    t0 = time.perf_counter()
    for _ in range(args.iters):
        step()
    sync(device)
    dt = (time.perf_counter() - t0) / args.iters

    print(
        f"device={device}  gaussians={args.gaussians}  render={w}x{h}  "
        f"{'fwd+bwd' if args.backward else 'fwd'}: {dt*1000:.0f} ms/iter "
        f"({1/dt:.2f} it/s)"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
