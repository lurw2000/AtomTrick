"""Parameter counts of the paper configurations.

Three independent routes must agree:

1. the model constructor with the final-configuration kwargs (in-process);
2. ``train.py --dry_run`` (argparse defaults = final configuration), which is
   what every training run and config file resolves through;
3. the shipped config files under ``configs/`` (skipped if a file is absent).

Logged in the original training runs: final 3,730,368; SO(2) K/V 4,418,496.
The remaining values come from the closed-form count in the architecture audit,
which reproduces both logged numbers exactly.
"""
import os

import pytest
import torch

from common import FINAL_MODEL_KWARGS, REPO_ROOT, build_model, n_params, parse_param_count, run_train

FINAL = 3_730_368

#: name -> (constructor overrides on the full-size final model, expected count)
EXPECTED = {
    "final":      ({}, FINAL),
    "so2_kv":     ({"kv": "so2"}, 4_418_496),
    "o3_kv":      ({"kv": "o3"}, 2_452_416),
    "layers4":    ({"n_layers": 4}, 2_516_544),
    "layers8":    ({"n_layers": 8}, 4_944_192),
    "layers12":   ({"n_layers": 12}, 7_371_840),
    "lmax2":      ({"irreps_hidden": "64x0e+64x1o+64x2e"}, 2_402_432),
    "lmax1":      ({"irreps_hidden": "64x0e+64x1o"}, 1_418_560),
}


@pytest.mark.parametrize("name", list(EXPECTED))
def test_constructor_count(name):
    overrides, expected = EXPECTED[name]
    model = build_model(dtype=torch.float32, small=False, **overrides)
    assert n_params(model) == expected, f"{name}: {n_params(model):,} != {expected:,}"


def test_so2_overhead_is_exactly_the_antisymmetric_blocks():
    """SO(2) adds W_b for every |m|>0 block of K and V: 2 x 6 x sum_{m=1..3} ((4-m)C)^2."""
    C, L, layers = 64, 3, 6
    extra = 2 * layers * sum(((L + 1 - m) * C) ** 2 for m in range(1, L + 1))
    assert extra == 688_128
    assert EXPECTED["so2_kv"][1] - FINAL == extra


# ---------------------------------------------------------------------------
# train.py --dry_run
# ---------------------------------------------------------------------------

@pytest.mark.slow
@pytest.mark.parametrize("flags,expected", [
    ([], FINAL),
    (["--kv_mode", "so2"], 4_418_496),
    (["--kv_mode", "o3"], 2_452_416),
])
def test_dry_run_defaults_are_the_final_configuration(flags, expected, tmp_path):
    res = run_train(["--dry_run", *flags], cwd=str(tmp_path))
    assert res.returncode == 0, res.stdout[-4000:]
    assert parse_param_count(res.stdout) == expected


def _cfg(*parts):
    return os.path.join(REPO_ROOT, "configs", *parts)


#: (config files applied in order, --molecule or None, expected count or None = "just resolves")
CONFIG_CASES = [
    (["rmd17/atomtrick.yaml"], "aspirin", FINAL),
    (["rmd17/atomtrick.yaml", "rmd17/ablations/so2.yaml"], "aspirin", 4_418_496),
    (["rmd17/atomtrick.yaml", "rmd17/ablations/o3_kv.yaml"], "aspirin", 2_452_416),
    (["rmd17/atomtrick.yaml", "rmd17/ablations/softmax.yaml"], "aspirin", FINAL),
    (["rmd17/atomtrick.yaml", "rmd17/ablations/perdegree_ln.yaml"], "aspirin", FINAL),
    (["rmd17/atomtrick.yaml", "rmd17/ablations/layers4.yaml"], "aspirin", 2_516_544),
    (["rmd17/atomtrick.yaml", "rmd17/ablations/layers8.yaml"], "aspirin", 4_944_192),
    (["rmd17/atomtrick.yaml", "rmd17/ablations/layers12.yaml"], "aspirin", 7_371_840),
    (["rmd17/atomtrick.yaml", "rmd17/ablations/lmax2.yaml"], "aspirin", 2_402_432),
    (["rmd17/atomtrick.yaml", "rmd17/ablations/lmax1.yaml"], "aspirin", 1_418_560),
    (["md22/atomtrick.yaml"], "Ac-Ala3-NHMe", FINAL),
    (["md22/atomtrick.yaml"], "double-walled_nanotube", FINAL),
    (["spice/stage1_force.yaml"], None, FINAL),
    (["spice/stage1_force.yaml", "spice/stage2_swa.yaml"], None, FINAL),
]


@pytest.mark.slow
@pytest.mark.parametrize("files,molecule,expected", CONFIG_CASES,
                         ids=[f[-1].replace(".yaml", "") for f, _, _ in CONFIG_CASES])
def test_config_files_resolve_to_expected_counts(files, molecule, expected, tmp_path):
    paths = [_cfg(*f.split("/")) for f in files]
    missing = [p for p in paths if not os.path.exists(p)]
    if missing:
        pytest.skip(f"config file(s) not present: {missing}")
    args = []
    for p in paths:
        args += ["--config", p]
    if molecule is not None:
        args += ["--molecule", molecule]
    res = run_train([*args, "--seed", "42", "--dry_run"], cwd=str(tmp_path))
    assert res.returncode == 0, res.stdout[-4000:]
    n = parse_param_count(res.stdout)
    if expected is not None:
        assert n == expected, f"{files}: {n:,} != {expected:,}"


def test_final_kwargs_match_the_contract():
    """Keep the test helper honest: these are the paper-model settings."""
    kw = FINAL_MODEL_KWARGS
    assert kw["irreps_hidden"] == "64x0e+64x1o+64x2e+64x3o"
    assert (kw["n_layers"], kw["n_heads"], kw["ff_mult"]) == (6, 8, 4)
    assert (kw["attn_nonlin"], kw["layernorm_mode"], kw["ffn_act_mode"]) == ("identity", "shared", "gate")
    assert (kw["k_mode"], kw["v_mode"]) == ("o2", "o2")
    assert (kw["rbf_type"], kw["num_radial_basis"], kw["radial_hidden"]) == ("expnorm", 32, 256)
