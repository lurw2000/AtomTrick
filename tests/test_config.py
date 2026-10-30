"""``train.py`` configuration resolution, exercised through ``--dry_run``.

Contract under test (see README "Configuration"):

* ``--config A.yaml --config B.yaml``: later files override earlier ones;
  explicit CLI flags override every file.
* YAML keys are exactly the argparse ``dest`` names; an unknown key is an error.
* Booleans are YAML ``true``/``false`` (``--flag`` / ``--no-flag`` on the CLI).
* After merging, string values are formatted with ``{molecule}``, ``{seed}``
  and ``{experiment}``.
* Retired keys from old configs/args.json (``manual_backward``, ``cuda_graph``,
  ``compile``, ``energy_correlation``, ``radial_rank`` ...) are accepted and ignored.
* A previous run's ``args.json`` may record options this code version no longer
  has; they are dropped with a warning (the same key in a hand-written file is
  still an error).
* Any ``ATOMTRICKS_*`` (pre-rename spelling) environment variable aborts the
  run -- the old names would be silently ignored by the code (an
  ``ATOMTRICKS_MAKEFX`` set by mistake would turn a compiled run into an eager
  one without any warning).

The parameter count printed by ``--dry_run`` is used as a format-independent
probe of which value won (n_layers changes it); booleans are read back from the
resolved configuration ``--dry_run`` prints.
"""
import json

import pytest

from common import parse_param_count, run_train

pytestmark = pytest.mark.slow

FINAL = 3_730_368
L4, L8, L12 = 2_516_544, 4_944_192, 7_371_840


def _yaml(tmp_path, name, text):
    p = tmp_path / name
    p.write_text(text)
    return str(p)


def _dry(tmp_path, *args, env=None):
    res = run_train([*args, "--dry_run"], cwd=str(tmp_path), env=env)
    return res, res.stdout


def test_later_config_overrides_earlier(tmp_path):
    a = _yaml(tmp_path, "a.yaml", "n_layers: 4\nlr: 0.001\n")
    b = _yaml(tmp_path, "b.yaml", "n_layers: 8\n")
    res, out = _dry(tmp_path, "--config", a, "--config", b)
    assert res.returncode == 0, out[-4000:]
    assert parse_param_count(out) == L8
    res, out = _dry(tmp_path, "--config", b, "--config", a)
    assert res.returncode == 0, out[-4000:]
    assert parse_param_count(out) == L4


def test_cli_flag_overrides_all_configs(tmp_path):
    a = _yaml(tmp_path, "a.yaml", "n_layers: 4\n")
    b = _yaml(tmp_path, "b.yaml", "n_layers: 8\n")
    res, out = _dry(tmp_path, "--config", a, "--config", b, "--n_layers", "12")
    assert res.returncode == 0, out[-4000:]
    assert parse_param_count(out) == L12
    # the CLI flag wins regardless of where it appears on the command line
    res, out = _dry(tmp_path, "--n_layers", "12", "--config", a)
    assert res.returncode == 0, out[-4000:]
    assert parse_param_count(out) == L12


def _resolved(out):
    """The 'Resolved configuration:' JSON block printed by --dry_run."""
    body = out.split("Resolved configuration:", 1)[1]
    return json.loads(body[:body.index("\n}") + 2])


def test_yaml_booleans_and_cli_negation(tmp_path):
    cfg = _yaml(tmp_path, "refs.yaml", "fit_atomic_refs: true\n")
    res, out = _dry(tmp_path, "--config", cfg)
    assert res.returncode == 0, out[-4000:]
    assert _resolved(out)["fit_atomic_refs"] is True
    res, out = _dry(tmp_path, "--config", cfg, "--no-fit_atomic_refs")
    assert res.returncode == 0, out[-4000:]
    assert _resolved(out)["fit_atomic_refs"] is False


def test_placeholders_are_formatted(tmp_path):
    cfg = _yaml(tmp_path, "paths.yaml", (
        'data: "data/rmd17/{molecule}_train_default.npz"\n'
        'val_data: "data/rmd17/{molecule}_val_default.npz"\n'
        'out_dir: "runs/{experiment}/{molecule}/s{seed}"\n'
        'wandb_name: "{experiment}_{molecule}_s{seed}"\n'))
    res, out = _dry(tmp_path, "--config", cfg, "--molecule", "aspirin", "--seed", "7",
                    "--experiment", "myexp")
    assert res.returncode == 0, out[-4000:]
    for resolved in ("data/rmd17/aspirin_train_default.npz", "data/rmd17/aspirin_val_default.npz",
                     "runs/myexp/aspirin/s7", "myexp_aspirin_s7"):
        assert resolved in out, f"{resolved!r} not in the resolved config:\n{out[-4000:]}"


def test_dry_run_needs_no_data_files(tmp_path):
    """The default config points at data/ files that do not exist here."""
    res, out = _dry(tmp_path, "--molecule", "aspirin")
    assert res.returncode == 0, out[-4000:]
    assert parse_param_count(out) == FINAL


def test_unknown_config_key_is_an_error(tmp_path):
    cfg = _yaml(tmp_path, "typo.yaml", "n_layerz: 4\n")
    res, out = _dry(tmp_path, "--config", cfg)
    assert res.returncode != 0, "an unknown YAML key must not be silently ignored"
    assert "n_layerz" in out


def test_retired_keys_are_accepted(tmp_path):
    cfg = _yaml(tmp_path, "old.yaml", (
        "manual_backward: true\ncuda_graph: false\ncompile: false\n"
        "energy_correlation: 2\nradial_rank: 8\n"))
    res, out = _dry(tmp_path, "--config", cfg)
    assert res.returncode == 0, out[-4000:]
    assert parse_param_count(out) == FINAL


def test_run_record_drops_options_this_version_lacks(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    record = run / "args.json"
    record.write_text(json.dumps({"optimizer": "adamw", "n_layers": 6,
                                  "option_from_an_older_version": 0.5,
                                  "_provenance": {"git_sha": "0" * 40}}))
    res, out = _dry(tmp_path, "--config", str(record))
    assert res.returncode == 0, out[-4000:]
    assert "option_from_an_older_version" in out, "the dropped key must be named in a warning"
    assert parse_param_count(out) == FINAL

    hand = tmp_path / "hand.json"
    hand.write_text(json.dumps({"option_from_an_older_version": 0.5}))
    res, out = _dry(tmp_path, "--config", str(hand))
    assert res.returncode != 0, "an unknown key in a hand-written config must stay an error"
    assert "option_from_an_older_version" in out

    # a recorded value that this version rejects is still an error
    record.write_text(json.dumps({"optimizer": "sgd", "_provenance": {}}))
    res, out = _dry(tmp_path, "--config", str(record))
    assert res.returncode != 0, out[-4000:]


@pytest.mark.parametrize("var", ["ATOMTRICKS_MAKEFX", "ATOMTRICKS_MAKEFX_DUCKOFF"])
def test_old_env_var_spelling_is_rejected(tmp_path, var):
    res, out = _dry(tmp_path, env={var: "1"})
    assert res.returncode != 0, f"{var}=1 must abort, not be silently ignored"
    assert "ATOMTRICK" in out
