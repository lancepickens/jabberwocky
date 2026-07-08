"""Frame-selection tests (exposure gating)."""

import numpy as np

from splatvid.video import _exposure_factor, _sharpness


def test_exposure_factor_gates_bad_frames():
    rng = np.random.default_rng(0)
    normal = rng.integers(40, 210, (120, 160), np.uint8)  # well-exposed, textured
    assert _exposure_factor(normal) == 1.0
    assert _exposure_factor(np.full((120, 160), 250, np.uint8)) == 0.1  # blown
    assert _exposure_factor(np.full((120, 160), 5, np.uint8)) == 0.1  # blacked
    assert _exposure_factor(np.full((120, 160), 128, np.uint8)) == 0.1  # flat


def test_exposure_folds_into_selection_score():
    # A blown-out frame loses to a well-exposed one even if it has some texture.
    rng = np.random.default_rng(1)
    normal = rng.integers(40, 210, (120, 160), np.uint8)
    blown = rng.integers(245, 256, (120, 160)).clip(0, 255).astype(np.uint8)
    assert _sharpness(normal) * _exposure_factor(normal) > \
        _sharpness(blown) * _exposure_factor(blown)
