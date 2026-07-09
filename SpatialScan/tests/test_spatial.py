"""Loader plumbing: locate moov past a large mdat; orient eyes by disparity."""

import struct

import numpy as np

from spatialscan.quicktime import extract_spatial_metadata
from spatialscan.spatial import _orient_stereo, read_moov_bytes
from spatialscan.synthetic import build_spatial_mov_bytes, make_spatial_video


def _fake_mov_with_trailing_moov(tmp_path):
    """ftyp + a big mdat (headers only) + moov carrying the spatial atoms."""
    moov = build_spatial_mov_bytes(baseline_um=19274, hfov_millideg=63400)
    ftyp = struct.pack(">I", 16) + b"ftyp" + b"qt  " + b"\x00" * 4
    mdat_payload = b"\x00" * 4096
    mdat = struct.pack(">I", 8 + len(mdat_payload)) + b"mdat" + mdat_payload
    p = tmp_path / "clip.mov"
    p.write_bytes(ftyp + mdat + moov)
    return p


def test_read_moov_skips_mdat(tmp_path):
    p = _fake_mov_with_trailing_moov(tmp_path)
    moov = read_moov_bytes(str(p))
    assert moov[4:8] == b"moov"
    meta = extract_spatial_metadata(moov)
    assert meta.is_mv_hevc
    assert abs(meta.baseline_m - 0.019274) < 1e-6
    assert abs(meta.hfov_deg - 63.4) < 1e-3


def test_orient_stereo_swaps_when_reversed():
    # Render a synthetic pair, then hand _orient_stereo the eyes backwards and
    # confirm it puts them back (left first => positive disparity).
    video = make_spatial_video(n_frames=1, width=200, height=150, baseline_m=0.08)
    left, right = video.left[0], video.right[0]
    a, b = _orient_stereo([right], [left])  # deliberately reversed
    assert np.array_equal(a[0], left) and np.array_equal(b[0], right)
