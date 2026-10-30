"""pytest configuration for the AtomTrick test-suite.

Run from the repository root::

    python -m pytest                 # whole suite, CPU, a few minutes
    python -m pytest -m "not slow"   # skip the train.py subprocess tests
    python -m pytest tests/test_makefx.py   # CUDA-only; skipped without a GPU

Environment handling (done here, before torch / AtomTrick are imported):

* ``PYTORCH_JIT=0`` -- exactly what ``train.py`` sets.  TorchScript breaks the
  second-order ``make_fx`` trace, and running the rest of the suite without
  TorchScript means every test exercises the code path production uses.
* ``ATOMTRICK_MAKEFX`` / ``ATOMTRICK_MAKEFX_DUCKOFF`` and any old-spelling
  ``ATOMTRICKS_*`` variables inherited from the caller's shell are dropped, so
  the ``Trainer`` built by the tests always starts on the eager path unless a
  test opts in explicitly.
"""
import os
import sys

os.environ.setdefault("PYTORCH_JIT", "0")
for _k in list(os.environ):
    if _k.startswith("ATOMTRICKS_") or _k in ("ATOMTRICK_MAKEFX", "ATOMTRICK_MAKEFX_DUCKOFF"):
        del os.environ[_k]

_TESTS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.dirname(_TESTS_DIR)
for _p in (_REPO_ROOT, _TESTS_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import pytest  # noqa: E402
import torch   # noqa: E402

from common import build_model, make_batch  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "cuda: needs a CUDA GPU (skipped otherwise)")
    config.addinivalue_line(
        "markers", "slow: launches train.py in a subprocess or compiles a graph")


@pytest.fixture(autouse=True)
def _seed_everything():
    """Every test starts from the same global RNG state."""
    torch.manual_seed(0)
    yield


@pytest.fixture
def small_model():
    """Final-configuration model, reduced width/depth, fp64, eval mode."""
    return build_model(dtype=torch.float64, seed=0)


@pytest.fixture
def small_batch():
    """Two random molecules (6 + 5 atoms), complete graphs, fp64."""
    return make_batch((6, 5), dtype=torch.float64, seed=0)

