"""Small helpers shared by scripts/data/prepare_*.py (not a public module)."""
from __future__ import annotations

import importlib
import logging
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def import_data_module(name: str):
    """Import ``AtomTrick.data.<name>`` (e.g. ``rmd17``, ``md22``).

    The normal import runs ``AtomTrick/__init__.py``, which pulls in torch.  The
    raw-data processors themselves only need numpy (and pandas for rMD17), so if
    the full package cannot be imported (e.g. data prep on a login node without
    torch) we load ``AtomTrick/data/base.py`` + ``<name>.py`` directly under a
    stub package.  Both routes execute the very same processor code.
    """
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    try:
        return importlib.import_module(f"AtomTrick.data.{name}")
    except ImportError as exc:
        logging.getLogger(__name__).info(
            "full AtomTrick import failed (%s); loading AtomTrick/data/%s.py standalone", exc, name)
        pkg_name = "_atomtrick_data_standalone"
        if pkg_name not in sys.modules:
            pkg = types.ModuleType(pkg_name)
            pkg.__path__ = [str(REPO_ROOT / "AtomTrick" / "data")]
            sys.modules[pkg_name] = pkg
        return importlib.import_module(f"{pkg_name}.{name}")


def setup_logging() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
