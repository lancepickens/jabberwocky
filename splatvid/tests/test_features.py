"""Candidate-pair selection (frame-density-aware matching)."""

import numpy as np

from splatvid.features import FrameFeatures, _candidate_pairs


def _dummy(n, seed=0):
    rng = np.random.default_rng(seed)
    base = rng.normal(size=(n + 8, 128))
    return [
        FrameFeatures(
            np.zeros((10, 2), np.float32),
            (base[i : i + 8].mean(0) + 0.1 * rng.normal(size=128)).astype(np.float32)[None].repeat(10, 0),
            np.zeros((10, 3), np.float32),
            (64, 48),
        )
        for i in range(n)
    ]


def _covered(pairs):
    c = set()
    for a, b in pairs:
        c.add(a)
        c.add(b)
    return c


def test_candidate_pairs_connect_every_frame():
    for n in (12, 40, 100, 418):
        pairs = _candidate_pairs(_dummy(n), window=6, loop_stride=15, retrieval_k=5)
        assert _covered(pairs) == set(range(n)), f"frame stranded at n={n}"
        assert all(a < b for a, b in pairs)  # ordered, no self-pairs


def test_candidate_pairs_scale_sublinearly_when_dense():
    # Dense capture must not blow up: 10x frames must cost far less than 10x pairs.
    p40 = len(_candidate_pairs(_dummy(40), 6, 15, 5))
    p418 = len(_candidate_pairs(_dummy(418), 6, 15, 5))
    assert p418 < 6 * p40  # would be ~12x without density scaling
