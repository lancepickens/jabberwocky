"""Container parsing: recover baseline / hFOV / eye layout from atoms."""

import struct

from spatialscan.quicktime import extract_spatial_metadata, parse_atoms
from spatialscan.synthetic import build_spatial_mov_bytes


def test_parses_baseline_and_hfov():
    blob = build_spatial_mov_bytes(baseline_um=19200, hfov_millideg=63000)
    meta = extract_spatial_metadata(blob)
    assert meta.is_mv_hevc
    assert meta.baseline_m is not None
    assert abs(meta.baseline_m - 0.0192) < 1e-6
    assert abs(meta.hfov_deg - 63.0) < 1e-3
    assert meta.has_left_eye and meta.has_right_eye
    assert not meta.eyes_reversed
    assert "blin" in meta.source_boxes and "hfov" in meta.source_boxes


def test_eyes_reversed_flag():
    blob = build_spatial_mov_bytes(eyes_reversed=True)
    meta = extract_spatial_metadata(blob)
    assert meta.eyes_reversed


def test_atom_tree_descends_into_sample_entry():
    blob = build_spatial_mov_bytes()
    root = parse_atoms(blob)
    assert root.find("vexu") is not None
    assert root.find("blin") is not None
    assert root.find("hvc1") is not None


def test_out_of_range_fields_are_rejected():
    # A 5-metre "baseline" is nonsense and must be ignored, not trusted.
    bad = build_spatial_mov_bytes(baseline_um=5_000_000, hfov_millideg=63000)
    meta = extract_spatial_metadata(bad)
    assert meta.baseline_m is None


def test_non_spatial_blob_is_flagged():
    meta = extract_spatial_metadata(b"\x00\x00\x00\x10ftypmp42" + b"\x00" * 8)
    assert not meta.is_mv_hevc
    assert meta.baseline_m is None
