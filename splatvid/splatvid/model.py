"""The gaussian splat scene model: parameters, activations, densify/prune."""

from __future__ import annotations

import logging

import numpy as np
import torch
from torch import nn

log = logging.getLogger(__name__)


def _knn_mean_dist(points: np.ndarray, k: int = 3) -> np.ndarray:
    """Mean distance to the k nearest neighbours, chunked to bound memory."""
    pts = points.astype(np.float32)
    n = pts.shape[0]
    out = np.empty(n)
    chunk = max(1, min(512, int(2e8 // max(n * 3 * 4, 1))))
    for s in range(0, n, chunk):
        d = np.linalg.norm(pts[s : s + chunk, None] - pts[None], axis=-1)
        d.partition(min(k, n - 1), axis=1)  # element 0 is self-distance 0
        out[s : s + chunk] = d[:, 1 : k + 1].mean(axis=1)
    return out


class GaussianModel(nn.Module):
    """Learnable 3D gaussians.

    Raw (pre-activation) parameters, matching the storage layout of the
    reference 3DGS `.ply` format:

    * ``xyz``        world positions
    * ``log_scale``  per-axis scales, stored in log space
    * ``quat``       rotations (w, x, y, z), normalized on use
    * ``color``      RGB stored as SH degree-0 coefficients
    * ``opacity``    stored as logits, sigmoid on use
    """

    SH_C0 = 0.28209479177387814  # Y_0^0; color = 0.5 + SH_C0 * coeff

    def __init__(
        self,
        xyz: np.ndarray,
        rgb: np.ndarray,
        init_opacity: float = 0.1,
        device: str = "cpu",
        feature_dim: int = 0,
    ) -> None:
        super().__init__()
        n = xyz.shape[0]
        dist = np.clip(_knn_mean_dist(xyz), 1e-5, None)
        quat = np.zeros((n, 4), dtype=np.float32)
        quat[:, 0] = 1.0
        rgb = np.clip(rgb, 1e-4, 1 - 1e-4)
        sh = (rgb - 0.5) / self.SH_C0
        op = np.log(init_opacity / (1 - init_opacity))

        to = lambda a: torch.tensor(a, dtype=torch.float32, device=device)  # noqa: E731
        self.xyz = nn.Parameter(to(xyz))
        self.log_scale = nn.Parameter(to(np.log(dist))[:, None].repeat(1, 3))
        self.quat = nn.Parameter(to(quat))
        self.color = nn.Parameter(to(sh))
        self.opacity = nn.Parameter(torch.full((n, 1), float(op), device=device))
        # Optional learned per-gaussian feature for the neural renderer. Channels
        # 0-2 start as the base colour so an identity shader reproduces the
        # direct-colour render; the rest start at zero. Off (None) by default so
        # the direct-colour pipeline is unchanged.
        if feature_dim > 0:
            c = min(3, feature_dim)
            feat = np.zeros((n, feature_dim), dtype=np.float32)
            feat[:, :c] = rgb[:, :c]
            self.feature = nn.Parameter(to(feat))
        else:
            self.feature = None
        self.max_grad_accum = torch.zeros(n, device=device)
        self.grad_count = torch.zeros(n, device=device)

    # -- activations --------------------------------------------------------
    @property
    def num_gaussians(self) -> int:
        return self.xyz.shape[0]

    def get_scale(self) -> torch.Tensor:
        return torch.exp(self.log_scale)

    def get_quat(self) -> torch.Tensor:
        return self.quat / (self.quat.norm(dim=-1, keepdim=True) + 1e-12)

    def get_opacity(self) -> torch.Tensor:
        return torch.sigmoid(self.opacity)

    def get_rgb(self) -> torch.Tensor:
        return torch.clamp(0.5 + self.SH_C0 * self.color, 0.0, 1.0)

    def get_feature(self) -> torch.Tensor | None:
        """Raw per-gaussian feature vectors (no activation), or None if off."""
        return self.feature

    # -- densification bookkeeping ------------------------------------------
    def accumulate_grads(self, screen_grad_norm: torch.Tensor, visible: torch.Tensor) -> None:
        """Track per-gaussian screen-space positional gradient magnitudes."""
        self.max_grad_accum[visible] += screen_grad_norm[visible]
        self.grad_count[visible] += 1

    def _reset_grad_accum(self) -> None:
        n = self.num_gaussians
        dev = self.xyz.device
        self.max_grad_accum = torch.zeros(n, device=dev)
        self.grad_count = torch.zeros(n, device=dev)

    def _rebuild(self, keep: torch.Tensor, extra: dict[str, torch.Tensor] | None) -> None:
        """Replace parameters with kept rows plus optional appended rows."""
        def cat(name: str) -> torch.Tensor:
            base = getattr(self, name).data[keep]
            if extra is not None:
                base = torch.cat([base, extra[name]], dim=0)
            return base

        self.xyz = nn.Parameter(cat("xyz"))
        self.log_scale = nn.Parameter(cat("log_scale"))
        self.quat = nn.Parameter(cat("quat"))
        self.color = nn.Parameter(cat("color"))
        self.opacity = nn.Parameter(cat("opacity"))
        if self.feature is not None:
            self.feature = nn.Parameter(cat("feature"))
        self._reset_grad_accum()

    def reset_opacity(self, value: float = 0.01) -> None:
        """Clamp all opacities down to ``value`` (the 3DGS opacity-reset trick).

        Gaussians that matter re-earn their opacity via the photometric loss;
        floaters that don't stay low and get pruned — the direct floater killer
        that rendered-depth supervision can't reach. Caller must rebuild the
        optimizer afterwards (the opacity moment state no longer applies)."""
        with torch.no_grad():
            logit = float(np.log(value / (1.0 - value)))
            self.opacity.data.clamp_(max=logit)

    def densify_and_prune(
        self,
        grad_threshold: float,
        scene_extent: float,
        min_opacity: float = 0.005,
        max_gaussians: int = 200_000,
        prune_center: np.ndarray | None = None,
        prune_radius: float | None = None,
    ) -> None:
        """Split large / clone small high-gradient gaussians; prune weak ones."""
        with torch.no_grad():
            avg_grad = self.max_grad_accum / self.grad_count.clamp(min=1)
            scale = self.get_scale()
            max_scale = scale.max(dim=-1).values

            prune = (self.get_opacity()[:, 0] < min_opacity) | (
                max_scale > 0.5 * scene_extent
            )
            if prune_center is not None and prune_radius is not None:
                # Prune gaussians that have drifted far outside the real scene —
                # the far floaters depth supervision leaves behind.
                center = torch.tensor(prune_center, dtype=self.xyz.dtype, device=self.xyz.device)
                far = (self.xyz.data - center).norm(dim=-1) > float(prune_radius)
                prune = prune | far

            room = max_gaussians - self.num_gaussians
            hot = (avg_grad > grad_threshold) & ~prune
            if room <= 0:
                hot = torch.zeros_like(hot)
            elif int(hot.sum()) > room:
                # Keep the hottest gaussians within budget.
                idx = torch.nonzero(hot).squeeze(1)
                order = torch.argsort(avg_grad[idx], descending=True)
                hot = torch.zeros_like(hot)
                hot[idx[order[:room]]] = True

            split = hot & (max_scale > 0.01 * scene_extent)
            clone = hot & ~split

            # Per-gaussian parameters carried through cloning/splitting. The
            # optional feature rides along like any other (copied for children).
            names = ["xyz", "log_scale", "quat", "color", "opacity"]
            if self.feature is not None:
                names.append("feature")
            extra: dict[str, list[torch.Tensor]] = {k: [] for k in names}

            if int(clone.sum()) > 0:
                for name in names:
                    extra[name].append(getattr(self, name).data[clone])

            if int(split.sum()) > 0:
                # Two children sampled inside the parent, at 1/1.6 scale.
                for _ in range(2):
                    noise = torch.randn_like(self.xyz.data[split]) * scale[split]
                    extra["xyz"].append(self.xyz.data[split] + noise)
                    extra["log_scale"].append(
                        self.log_scale.data[split] - float(np.log(1.6))
                    )
                    for name in names:
                        if name in ("xyz", "log_scale"):
                            continue
                        extra[name].append(getattr(self, name).data[split])

            keep = ~(prune | split)
            packed = (
                {k: torch.cat(v, dim=0) for k, v in extra.items()}
                if extra["xyz"]
                else None
            )
            n_before = self.num_gaussians
            self._rebuild(torch.nonzero(keep).squeeze(1), packed)
            log.info(
                "Densify/prune: %d -> %d gaussians (split %d, clone %d, prune %d)",
                n_before, self.num_gaussians,
                int(split.sum()), int(clone.sum()), int(prune.sum()),
            )

    def prune_transparent(self, min_opacity: float = 0.01) -> None:
        with torch.no_grad():
            keep = self.get_opacity()[:, 0] >= min_opacity
            self._rebuild(torch.nonzero(keep).squeeze(1), None)

    def append_gaussians(
        self,
        xyz: torch.Tensor,
        rgb: torch.Tensor,
        radius: torch.Tensor,
        init_opacity: float = 0.3,
    ) -> None:
        """Append fresh gaussians (e.g. artifix hole seeds) to the model.

        Densification can only clone/split *existing* gaussians, so a truly
        empty region can never grow geometry on its own — repair passes must
        be able to plant it. ``radius`` is a per-point world-space size (the
        isotropic scale init); positions/colours are refined by whatever
        optimization follows. Caller must rebuild any optimizer afterwards.
        """
        with torch.no_grad():
            n = xyz.shape[0]
            dev = self.xyz.device
            to = lambda a: torch.as_tensor(a, dtype=torch.float32, device=dev)  # noqa: E731
            col = to(rgb).clamp(1e-4, 1 - 1e-4)
            quat = torch.zeros(n, 4, device=dev)
            quat[:, 0] = 1.0
            op = float(np.log(init_opacity / (1 - init_opacity)))
            extra = {
                "xyz": to(xyz),
                "log_scale": to(radius).clamp(min=1e-6).log()[:, None].repeat(1, 3),
                "quat": quat,
                "color": (col - 0.5) / self.SH_C0,
                "opacity": torch.full((n, 1), op, device=dev),
            }
            if self.feature is not None:
                feat = torch.zeros(n, self.feature.shape[1], device=dev)
                c = min(3, feat.shape[1])
                feat[:, :c] = col[:, :c]
                extra["feature"] = feat
            self._rebuild(torch.arange(self.num_gaussians, device=dev), extra)

    def prune_by_mask(self, keep: torch.Tensor) -> int:
        """Drop gaussians where ``keep`` is False; returns how many were removed.

        External cleanup passes (e.g. the artifix floater vote) decide *which*
        gaussians are artifacts; this applies the verdict. Caller must rebuild
        any optimizer afterwards (parameter tensors are replaced)."""
        with torch.no_grad():
            n_before = self.num_gaussians
            self._rebuild(torch.nonzero(keep).squeeze(1), None)
            return n_before - self.num_gaussians
