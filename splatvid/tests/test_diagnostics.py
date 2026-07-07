"""Device self-check tests.

On an Apple-Silicon Mac these exercise the MPS backend end to end and
assert it matches the CPU reference — this is the automated version of
`splatvid doctor`. On CPU-only machines the MPS/CUDA cases skip.
"""

import pytest
import torch

from splatvid.diagnostics import collect_environment, run_selfcheck


def test_cpu_selfcheck_passes():
    report = run_selfcheck("cpu")
    assert report.ok, report.error
    assert report.finite


def test_collect_environment_reports_torch_and_device():
    env = collect_environment("cpu")
    assert env["torch"] == torch.__version__
    assert env["target_device"] == "cpu"
    # These keys back the `splatvid doctor` output; keep them present.
    for key in ("mps_available", "mps_built", "cuda_available", "python"):
        assert key in env


def _mps_ready() -> bool:
    mps = getattr(torch.backends, "mps", None)
    return mps is not None and mps.is_available()


@pytest.mark.skipif(not _mps_ready(), reason="MPS backend not available")
def test_mps_matches_cpu():
    """The whole point of the M5 work: MPS must render like the CPU."""
    report = run_selfcheck("mps")
    assert report.ok, report.error
    assert report.finite


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_matches_cpu():
    report = run_selfcheck("cuda")
    assert report.ok, report.error
    assert report.finite
