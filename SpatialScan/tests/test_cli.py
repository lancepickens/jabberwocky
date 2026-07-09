"""CLI smoke tests: info on a synthetic container, demo end-to-end."""

from spatialscan.cli import main
from spatialscan.synthetic import build_spatial_mov_bytes


def test_info_reports_metadata(tmp_path, capsys):
    mov = tmp_path / "clip.mov"
    mov.write_bytes(build_spatial_mov_bytes(baseline_um=19200, hfov_millideg=63000))
    rc = main(["info", str(mov)])
    assert rc == 0
    out = capsys.readouterr().out
    assert "MV-HEVC=True" in out
    assert "baseline=19.20mm" in out


def test_demo_builds_scene(tmp_path, capsys):
    out = tmp_path / "demo.ply"
    rc = main(["demo", "-o", str(out), "--frames", "4"])
    assert rc == 0
    # Either a mesh or the point-cloud fallback, but a file must exist.
    produced = list(tmp_path.glob("demo.*"))
    assert produced and produced[0].stat().st_size > 0


def test_build_missing_file_errors(capsys):
    rc = main(["build", "/no/such/file.mov"])
    assert rc == 1
