"""``edge_cutoff`` (neighbour-list radius) must equal ``cutoff`` (radial basis).

The messages vanish smoothly at the cutoff only when the graph is cut at the
same radius, so ``Trainer.build_model`` -- the one place every train / eval /
MD / benchmark path builds the model -- rejects a mismatch up front
(:func:`AtomTrick.trainer.check_cutoffs`).  Missing keys default to 5.0, as in
the data loaders; every shipped config sets both to 5.0.  The inference
scripts' ``--cutoff`` flag (MD, latency benchmark) resizes only the graph, so
:func:`AtomTrick.trainer.inference_cutoff` rejects any value but the model's.

All CPU; only the default (small) model is ever built.
"""
import glob
import os

import pytest

from AtomTrick.trainer import Trainer, check_cutoffs, inference_cutoff

from common import REPO_ROOT


@pytest.mark.parametrize("cfg", [
    {},                                          # old args.json: both keys missing
    {"cutoff": 5.0, "edge_cutoff": 5.0},
    {"cutoff": 6, "edge_cutoff": 6.0},           # YAML int vs float
    {"cutoff": 6.0, "edge_cutoff": None},        # graph stored in the NPZ
])
def test_consistent_cutoffs_pass(cfg):
    check_cutoffs(cfg)


@pytest.mark.parametrize("cfg", [
    {"cutoff": 5.0, "edge_cutoff": 4.0},
    {"cutoff": 5.0, "edge_cutoff": 6.0},
    {"cutoff": 6.0},                             # edge_cutoff falls back to 5.0
    {"edge_cutoff": 6.0},                        # cutoff falls back to 5.0
])
def test_mismatched_cutoffs_raise(cfg):
    with pytest.raises(ValueError, match="edge_cutoff"):
        check_cutoffs(cfg)
    with pytest.raises(ValueError, match="edge_cutoff"):
        Trainer.build_model(cfg)


def test_build_model_accepts_matching_cutoffs():
    assert Trainer.build_model({"cutoff": 5.0, "edge_cutoff": 5.0}) is not None


@pytest.mark.parametrize("path", sorted(glob.glob(
    os.path.join(str(REPO_ROOT), "configs", "**", "*.yaml"), recursive=True)))
def test_shipped_configs_have_matching_cutoffs(path):
    yaml = pytest.importorskip("yaml")
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}
    if "cutoff" in cfg or "edge_cutoff" in cfg:   # override files set neither
        assert "cutoff" in cfg and "edge_cutoff" in cfg, path
        check_cutoffs(cfg)


@pytest.mark.parametrize("cfg,override,expected", [
    ({}, None, 5.0),                             # default model, no flag
    ({"cutoff": 6.0, "edge_cutoff": 6.0}, None, 6.0),
    ({"cutoff": 6.0, "edge_cutoff": None}, None, 6.0),
    ({"cutoff": 5.0, "edge_cutoff": 5.0}, 5, 5.0),
])
def test_inference_cutoff_defaults_to_model_cutoff(cfg, override, expected):
    assert inference_cutoff(cfg, override) == expected


@pytest.mark.parametrize("override", [4.0, 6.0])
def test_inference_cutoff_rejects_mismatched_override(override):
    """md/md_openmm.py and bench_atomtrick.py --cutoff resize only the graph."""
    with pytest.raises(ValueError, match="--cutoff"):
        inference_cutoff({"cutoff": 5.0, "edge_cutoff": 5.0}, override)
