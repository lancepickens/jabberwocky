import numpy as np
import torch

from splatvid.export import load_splat, save_ply, save_splat
from splatvid.model import GaussianModel


def _tiny_model():
    rng = np.random.default_rng(5)
    xyz = rng.normal(size=(50, 3)).astype(np.float32)
    rgb = rng.uniform(0.1, 0.9, (50, 3)).astype(np.float32)
    return GaussianModel(xyz, rgb, init_opacity=0.5)


def test_save_splat_roundtrip(tmp_path):
    m = _tiny_model()
    path = str(tmp_path / "scene.splat")
    save_splat(m, path)
    back = load_splat(path)
    assert back["xyz"].shape == (50, 3)
    # Same point set (order differs: the file is sorted by prominence).
    src = np.sort(m.xyz.detach().numpy().round(4), axis=0)
    dst = np.sort(back["xyz"].round(4), axis=0)
    assert np.allclose(src, dst, atol=1e-3)
    assert np.all(back["rgba"] >= 0) and np.all(back["rgba"] <= 1)
    # Quaternions decode to ~unit length.
    norms = np.linalg.norm(back["quat"], axis=1)
    assert np.all(np.abs(norms - 1.0) < 0.05)
    # Opacity should round-trip through the uint8 quantization.
    assert np.allclose(back["rgba"][:, 3], 0.5, atol=0.01)


def test_save_ply_header_and_size(tmp_path):
    m = _tiny_model()
    path = str(tmp_path / "scene.ply")
    save_ply(m, path)
    with open(path, "rb") as f:
        blob = f.read()
    header_end = blob.index(b"end_header\n") + len(b"end_header\n")
    header = blob[:header_end].decode("ascii")
    assert "element vertex 50" in header
    n_props = header.count("property float")
    assert n_props == 17  # xyz + normals + f_dc + opacity + scales + rot
    assert len(blob) - header_end == 50 * n_props * 4

    # Positions parse back correctly from the binary body.
    body = np.frombuffer(blob[header_end:], dtype="<f4").reshape(50, n_props)
    assert np.allclose(body[:, :3], m.xyz.detach().numpy(), atol=1e-6)


def test_model_activations():
    m = _tiny_model()
    assert torch.allclose(m.get_opacity(), torch.full((50, 1), 0.5), atol=1e-5)
    q = m.get_quat()
    assert torch.allclose(q.norm(dim=-1), torch.ones(50), atol=1e-5)
    rgb = m.get_rgb()
    assert (rgb >= 0).all() and (rgb <= 1).all()


def test_densify_and_prune_budget():
    m = _tiny_model()
    m.max_grad_accum = torch.ones(m.num_gaussians) * 1.0  # everything "hot"
    m.grad_count = torch.ones(m.num_gaussians)
    n0 = m.num_gaussians
    m.densify_and_prune(grad_threshold=0.5, scene_extent=1.0, max_gaussians=60)
    assert m.num_gaussians <= 60 + n0  # split adds 2 per parent, bounded by room
    assert m.xyz.shape[0] == m.opacity.shape[0] == m.quat.shape[0]

    # Prune everything by demanding impossible opacity.
    m.prune_transparent(min_opacity=1.1)
    assert m.num_gaussians == 0
