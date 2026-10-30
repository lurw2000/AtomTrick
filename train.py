"""Training script for AtomTrick (O(3)-equivariant graph transformer).

The argparse defaults ARE the paper's final model and rMD17/MD22 training
recipe (6 layers, 64x0e+64x1o+64x2e+64x3o, identity attention, shared
LayerNorm, scalar readout; 3,730,368 parameters), so a bare run trains the
paper model.  Experiments are described by YAML files in ``configs/``.

Usage
-----
    # paper model on rMD17 aspirin, seed 42
    python train.py --config configs/rmd17/atomtrick.yaml --molecule aspirin --seed 42

    # several configs: later files override earlier ones, CLI flags override all
    python train.py --config configs/rmd17/atomtrick.yaml \\
        --config configs/rmd17/ablations/so2.yaml --molecule aspirin --seed 1

    # plain CLI (no config file)
    python train.py --data data/rmd17/aspirin_train_default.npz \\
        --val_data data/rmd17/aspirin_val_default.npz --out_dir runs/aspirin

    # print the resolved configuration + parameter count; needs no data / GPU
    python train.py --config configs/rmd17/atomtrick.yaml --molecule aspirin --dry_run

Config files
------------
Keys are exactly the argparse ``dest`` names below (``n_layers: 6``,
``attn_nonlin: identity``, ``k_mode: o2``, ...); an unknown key is an error.
Boolean flags take YAML ``true`` / ``false``.  ``kv_mode`` is shorthand for
``k_mode`` + ``v_mode``: setting it in a later file (or on the CLI) overrides
the ``k_mode`` / ``v_mode`` of earlier layers unless that same layer sets them
too.  An ``args.json`` written by a previous run is also accepted (its
``atomic_ref`` / ``_provenance`` entries are recomputed, not read; options it
records that this code version no longer has are dropped with a warning).

After merging, the placeholders ``{molecule}``, ``{seed}`` and ``{experiment}``
are substituted in every string-valued option, e.g.
``data: data/rmd17/{molecule}_train_default.npz`` or
``out_dir: runs/{experiment}/{molecule}/s{seed}``.

Compilation
-----------
``--makefx auto`` (default) traces forward + forces with make_fx and compiles
them with torch.compile whenever CUDA is used; ``--makefx_duckoff auto``
disables make_fx duck-shaping iff every training structure has the same atom
count (rMD17 / MD22), as in the reported runs.  An explicitly exported
``ATOMTRICK_MAKEFX`` / ``ATOMTRICK_MAKEFX_DUCKOFF`` is honoured under ``auto``.
"""

import os

# TorchScript breaks make_fx's second-order trace, so the make_fx path needs
# PYTORCH_JIT=0 -- and it must be in the environment BEFORE torch is imported
# (the AtomTrick cores are @torch.jit.script functions).  With PYTORCH_JIT=0
# they simply run as Python; eager results are unaffected.
os.environ.setdefault("PYTORCH_JIT", "0")

import argparse  # noqa: E402
import difflib  # noqa: E402
import json  # noqa: E402
import logging  # noqa: E402
import re  # noqa: E402
import sys  # noqa: E402

logger = logging.getLogger("train")

# Options that no longer do anything.  They are still ACCEPTED (so archived
# launch scripts and old args.json / YAML files keep working) but ignored with
# a warning and not written to the new args.json.
#   manual_backward / compile / cuda_graph / hc_backward: backward-path switches
#       whose code was removed on 2026-07-03 (forces are always F = -dE/dR by
#       autograd, optionally baked into one graph by make_fx);
#   energy_correlation: order of the removed 'channel_tp' energy head;
#   radial_rank: rank of the removed low-rank radial modes (full/adapt/lora);
#   tensor_subclass: experimental O3Tensor subclass backend (unused);
#   grad_cos_every: energy/force gradient-cosine research probe (removed).
DEPRECATED_KEYS = ("manual_backward", "compile", "cuda_graph", "hc_backward",
                   "energy_correlation", "radial_rank", "tensor_subclass",
                   "grad_cos_every")

# Keys that may appear in a config file (e.g. a previous run's args.json) but
# are not read from it.
_IGNORED_FILE_KEYS = ("config", "dry_run", "atomic_ref")

_PLACEHOLDER_RE = re.compile(r"\{(molecule|seed|experiment)\}")


def build_parser():
    parser = argparse.ArgumentParser(
        description="Train AtomTrick.  Defaults = the paper's final model / "
                    "rMD17 training recipe.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # ---- Experiment / config files -----------------------------------------
    parser.add_argument("--config", action="append", default=None, metavar="YAML",
                        help="YAML (or JSON) file with option values, keys = the "
                             "option names below.  May be repeated: later files "
                             "override earlier ones; explicit CLI flags override "
                             "every file.")
    parser.add_argument("--experiment", default="default",
                        help="Experiment name; substituted for {experiment} in "
                             "string options (out_dir, wandb_name, ...).")
    parser.add_argument("--dry_run", action="store_true",
                        help="Resolve the configuration, build the model, print "
                             "the configuration and 'Model parameters: N', then "
                             "exit.  Needs no data files and no GPU.")

    # Data — two ways to specify:
    # (A) Pre-processed NPZs: --data ... --val_data ... [--test_data ...]
    # (B) Auto-process from raw rMD17 / MD22 downloads:
    #     --raw_data_dir ... --dataset rmd17|md22 --molecule aspirin
    # When (B) is given, --data / --val_data / --test_data are derived
    # automatically by running the dataset's processor; cached NPZs are
    # written to --processed_dir (default data/{dataset}/).
    parser.add_argument("--data", default=None, help="Path to training NPZ")
    parser.add_argument("--val_data", default=None, help="Path to validation NPZ (optional)")
    parser.add_argument("--val_split", type=float, default=0.0,
                        help="Fraction of --data to use as validation (ignored if --val_data is set)")
    parser.add_argument("--test_data", default=None, help="Path to test NPZ (optional)")
    parser.add_argument("--edge_cutoff", type=float, default=5.0,
                        help="Graph cutoff (Å): all pairs closer than this are edges. "
                             "Keep equal to --cutoff (the radial envelope) so the "
                             "energy stays smooth when atoms cross the cutoff.")
    parser.add_argument("--max_neighbors", type=int, default=0,
                        help="Cap neighbours kept per central atom (nearest-K by "
                             "distance). 0 = keep all within edge_cutoff (default). "
                             "Bounds graph degree so a large cutoff does not blow up "
                             "edges/memory (cf. OCP/EquiformerV2).")
    # Auto-processing from raw datasets
    parser.add_argument("--raw_data_dir", default=None,
                        help="Root of a raw rMD17 or MD22 download. If set, "
                             "train.py auto-processes the specified molecule "
                             "and fills in --data / --val_data / --test_data.")
    parser.add_argument("--dataset", default=None, choices=["rmd17", "md22"],
                        help="Dataset type for auto-processing (required with --raw_data_dir). "
                             "Also selects the dataset's canonical atomic reference energies.")
    parser.add_argument("--molecule", default=None,
                        help="Molecule name; substituted for {molecule} in string "
                             "options and used for auto-processing. "
                             "rmd17: aspirin / azobenzene / ... ; "
                             "md22: AT-AT / Ac-Ala3-NHMe / double-walled_nanotube / ...")
    parser.add_argument("--processed_dir", default=None,
                        help="Where to cache processed NPZs. "
                             "Default: data/{dataset}/ under the current working directory.")
    parser.add_argument("--split_id", type=int, default=1,
                        help="rmd17 only: which of the 5 official splits to use (default 1).")

    # ---- Model (defaults = paper model, 3,730,368 parameters) --------------
    parser.add_argument("--irreps", default="64x0e+64x1o+64x2e+64x3o",
                        help="Hidden irreps (natural parity 0e/1o/2e/3o required); "
                             "every multiplicity must be divisible by --n_heads.")
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--ff_mult", type=int, default=4)
    parser.add_argument("--max_z", type=int, default=100)
    parser.add_argument("--num_radial_basis", type=int, default=32)
    parser.add_argument("--cutoff", type=float, default=5.0,
                        help="Radial-basis cutoff (Å) of the attention gates.")
    parser.add_argument("--radial_hidden", type=int, default=256)
    parser.add_argument("--energy_hidden", type=int, default=256)
    parser.add_argument("--radial_mode", type=str, default="gate",
                        choices=["gate"],
                        help="Radial injection mode: 'gate' (learned O2Linear weights, "
                             "per-channel radial gating).  The only supported mode; "
                             "kept as an option for old configs.")
    parser.add_argument("--rbf_type", type=str, default="expnorm",
                        choices=["expnorm"],
                        help="Radial basis: 'expnorm' (exponential-normal smearing with "
                             "learnable centres/widths); the only supported basis")
    parser.add_argument("--kv_mode", type=str, default="o2",
                        choices=["o2", "o3", "so2"],
                        help="K/V projection mode (sets both --k_mode and --v_mode "
                             "unless those are given explicitly): 'o2' (paper), "
                             "'so2' (SO(2) control), 'o3' (rotation-free O3Linear).")
    parser.add_argument("--k_mode", type=str, default=None,
                        choices=["o2", "o3", "so2"],
                        help="K projection mode. Default: --kv_mode")
    parser.add_argument("--v_mode", type=str, default=None,
                        choices=["o2", "o3", "so2"],
                        help="V projection mode. Default: --kv_mode")
    parser.add_argument("--ffn_act_mode", type=str, default="gate",
                        choices=["gate"],
                        help="FFN activation: 'gate' (gated nonlinearity); the only "
                             "supported mode")
    parser.add_argument("--attn_nonlin", type=str, default="identity",
                        choices=["softmax", "identity"],
                        help="Attention weight calculation (score is always the "
                             "scaled inner product <q,k>/sqrt(d)): 'identity' (raw "
                             "scores, unnormalized; paper) or 'softmax' (normalized "
                             "over neighbours; ablation)")
    parser.add_argument("--layernorm_mode", type=str, default="shared",
                        choices=["default", "shared"],
                        help="Layer normalization: 'shared' (center l=0, single global "
                             "RMS across all degrees; paper) or 'default' (per-irrep "
                             "RMS norm; ablation)")
    parser.add_argument("--escn_frame", type=str, default="polefree",
                        choices=["polefree", "euler"],
                        help="Wigner-D evaluation of the eSCN edge frame: 'polefree' "
                             "(exactly equivariant forces for every edge direction; "
                             "default) or 'euler' (the direct Euler-angle evaluation of "
                             "the same frame; same function up to round-off, kept "
                             "for the latency comparison in benchmarks/latency/)")

    # ---- Training (defaults = rMD17 / MD22 recipe) --------------------------
    parser.add_argument("--max_iterations", type=int, default=750000,
                        help="Total number of gradient steps (absolute: a resumed run "
                             "continues up to this step)")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4,
                        help="Batch size for val/test eval. On the make_fx path it is "
                             "clamped to --batch_size (the compiled graph has the "
                             "training molecule count baked in). None/0 = --batch_size.")
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--optimizer", type=str, default="adamw",
                        choices=["adamw"],
                        help="AdamW over all parameters (the only optimizer; kept "
                             "so configs and args.json files can state it)")
    parser.add_argument("--rho_force", type=float, default=0.999,
                        help="Force loss weight; energy weight is 1 - rho_force")
    parser.add_argument("--swa_start", type=int, default=0,
                        help="MACE-style two-stage SWA: step to switch to the "
                             "energy-heavy stage (0 disables). Stage 1 (<swa_start) "
                             "uses --rho_force; stage 2 (>=swa_start) switches to "
                             "--swa_rho_force, resets LR to --swa_lr + the plateau "
                             "scheduler, and restarts weight-averaging as a fresh "
                             "uniform SWA average.")
    parser.add_argument("--swa_rho_force", type=float, default=0.1,
                        help="Stage-2 (energy-heavy) force loss weight for --swa_start.")
    parser.add_argument("--swa_lr", type=float, default=0.0,
                        help="LR for the SWA/energy stage (0 = reuse --lr).")
    parser.add_argument("--fit_atomic_refs", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Pre-fit per-element reference energies via LSQ on "
                             "the training set (SchNet/NequIP style) instead of "
                             "using the tabulated values in constants.py. The "
                             "fitted dict is threaded through to every MLFFDataset "
                             "and to the model's atomic_ref_energy buffer, and "
                             "saved in args.json.")
    parser.add_argument("--grad_clip", type=float, default=10.0)
    parser.add_argument("--eval_every", type=int, default=2500,
                        help="Evaluate val/test every this many steps")
    parser.add_argument("--scheduler_patience", type=int, default=3,
                        help="ReduceLROnPlateau patience (eval rounds)")
    parser.add_argument("--scheduler_factor", type=float, default=0.8,
                        help="ReduceLROnPlateau LR reduction factor")
    parser.add_argument("--scheduler_min_lr", type=float, default=1e-7,
                        help="ReduceLROnPlateau minimum LR")
    parser.add_argument("--warmup_steps", type=int, default=10000,
                        help="Linear LR warmup steps (0 to disable)")
    parser.add_argument("--ema_decay", type=float, default=0.99,
                        help="EMA decay of the evaluated / saved weights (0 disables)")

    # ---- Numerics / compilation ---------------------------------------------
    parser.add_argument("--dtype", default="float32", choices=["float32", "float64"],
                        help="Floating-point precision")
    parser.add_argument("--matmul_precision", default="highest", choices=["highest", "high"],
                        help="torch.set_float32_matmul_precision for float32 runs: "
                             "'highest' = full fp32 matmuls (default); 'high' allows TF32 "
                             "matmuls on Ampere+ GPUs. "
                             "Recorded in args.json; evaluate.py reuses it")
    parser.add_argument("--makefx", default="auto", choices=["auto", "on", "off"],
                        help="make_fx + torch.compile training/eval graph (sets "
                             "ATOMTRICK_MAKEFX). auto = on iff running on CUDA")
    parser.add_argument("--makefx_duckoff", default="auto", choices=["auto", "on", "off"],
                        help="Disable make_fx duck-shaping (sets ATOMTRICK_MAKEFX_DUCKOFF). "
                             "auto = on iff every training structure has the same "
                             "atom count (rMD17/MD22); needed there to avoid baked shapes")

    # ---- I/O ------------------------------------------------------------------
    parser.add_argument("--out_dir", default=None,
                        help="Output directory (args.json + checkpoints). Default: "
                             "runs/{experiment}/{molecule}/s{seed} "
                             "(runs/{experiment}/s{seed} without --molecule)")
    parser.add_argument("--save_every", type=int, default=12500,
                        help="Save a rolling full checkpoint every this many steps")
    parser.add_argument("--num_workers", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None, help="e.g. cuda:0 (auto-detected if omitted)")
    parser.add_argument("--resume", default=None, help="Path to a full checkpoint to resume from")
    parser.add_argument("--log_level", default="INFO", choices=["DEBUG", "INFO", "WARNING"])

    # wandb
    parser.add_argument("--wandb_project", default=None,
                        help="Weights & Biases project name (disabled if omitted)")
    parser.add_argument("--wandb_name", default=None,
                        help="Run name; also the checkpoint file prefix "
                             "(<out_dir>/<wandb_name>_best.pt / _final.pt)")

    # ---- Deprecated (accepted, ignored with a warning; see DEPRECATED_KEYS) ----
    for flag in ("manual_backward", "compile", "cuda_graph", "hc_backward",
                 "tensor_subclass"):
        parser.add_argument(f"--{flag}", action=argparse.BooleanOptionalAction,
                            default=None, help=argparse.SUPPRESS)
    for flag in ("energy_correlation", "radial_rank", "grad_cos_every"):
        parser.add_argument(f"--{flag}", type=int, default=None, help=argparse.SUPPRESS)
    return parser


# ---------------------------------------------------------------------------
# Config files
# ---------------------------------------------------------------------------

def _load_config_file(path):
    """Read one YAML/JSON config file into a flat dict."""
    if not os.path.isfile(path):
        raise SystemExit(f"error: config file not found: {path}")
    with open(path) as f:
        text = f.read()
    if path.endswith(".json"):
        data = json.loads(text)
    else:
        try:
            import yaml
        except ImportError:
            raise SystemExit("error: reading YAML configs needs PyYAML "
                             "(pip install pyyaml)") from None
        data = yaml.safe_load(text)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise SystemExit(f"error: {path}: top level must be a mapping of option: value")
    return data


def is_run_record(path, data):
    """True for an ``args.json`` written by train.py (any code version).

    Such a file records every option of the code version that wrote it, so it
    may name options this version no longer has; those are dropped with a
    warning instead of being an error (hand-written configs stay strict).
    """
    return (os.path.basename(str(path)) == "args.json"
            or any(str(k).startswith("_") for k in data))


def stale_config_keys(data, parser=None):
    """Keys of a config mapping that are not options of this code version."""
    actions = {a.dest for a in (parser or build_parser())._actions if a.dest != "help"}
    return [str(k) for k in data
            if not str(k).startswith("_") and str(k) not in _IGNORED_FILE_KEYS
            and str(k) not in actions]


def _coerce_value(action, key, value, source):
    """Validate / convert one config-file value the way argparse would a CLI string."""
    def bad(msg):
        raise SystemExit(f"error: {source}: {key}: {msg} (got {value!r})")

    if value is None:
        return None
    if isinstance(action, (argparse.BooleanOptionalAction, argparse._StoreTrueAction,
                           argparse._StoreFalseAction)):
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        bad("expected true or false")
    typ = action.type
    if typ is None:                      # string option
        if isinstance(value, bool) and action.choices and "on" in action.choices:
            value = "on" if value else "off"   # YAML 1.1 reads on/off as booleans
        elif isinstance(value, (str, int, float)) and not isinstance(value, bool):
            value = str(value)
        else:
            bad("expected a string")
    elif typ is int:
        if isinstance(value, bool):
            bad("expected an integer")
        if isinstance(value, str):
            try:
                value = float(value) if any(c in value for c in ".eE") else int(value)
            except ValueError:
                bad("expected an integer")
        if isinstance(value, float):
            if not value.is_integer():
                bad("expected an integer")
            value = int(value)
        if not isinstance(value, int):
            bad("expected an integer")
    elif typ is float:
        if isinstance(value, bool):
            bad("expected a number")
        try:
            value = float(value)         # also parses YAML 1.1 strings like '2e-4'
        except (TypeError, ValueError):
            bad("expected a number")
    elif isinstance(value, str):
        value = typ(value)
    if action.choices is not None and value not in action.choices:
        bad(f"must be one of {list(action.choices)}")
    return value


def _merge_config_files(parser, paths):
    """Merge config files in order -> {dest: value}.

    Unknown keys are errors, except in a previous run's args.json
    (is_run_record), where they are dropped with a warning.
    """
    actions = {a.dest: a for a in parser._actions if a.dest != "help"}
    merged = {}
    for path in paths:
        layer = {}
        data = _load_config_file(path)
        stale = (set(stale_config_keys(data, parser))
                 if is_run_record(path, data) else set())
        if stale:
            logger.warning("%s: dropping option(s) this code version does not have "
                           "(recorded by an earlier version): %s",
                           path, ", ".join(sorted(stale)))
        for key, value in data.items():
            key = str(key)
            if key in stale:
                continue
            if key.startswith("_"):              # metadata, e.g. _provenance
                continue
            if key in _IGNORED_FILE_KEYS:
                if key == "config" and value:
                    logger.warning("%s: nested 'config' entries are ignored", path)
                elif key == "dry_run" and value:
                    logger.warning("%s: 'dry_run' is CLI-only; ignored", path)
                elif key == "atomic_ref" and value:
                    logger.info("%s: 'atomic_ref' is recomputed at run time "
                                "(--fit_atomic_refs / --dataset), not read", path)
                continue
            if key not in actions:
                hint = difflib.get_close_matches(key, list(actions), n=3)
                parser.error(f"{path}: unknown option '{key}'"
                             + (f" (did you mean: {', '.join(hint)}?)" if hint else ""))
            layer[key] = _coerce_value(actions[key], key, value, path)
        # kv_mode is shorthand for k_mode + v_mode: a layer that sets it
        # overrides k_mode / v_mode inherited from EARLIER layers unless the
        # layer also sets them itself.
        if "kv_mode" in layer:
            for sub in ("k_mode", "v_mode"):
                if sub not in layer:
                    merged.pop(sub, None)
        merged.update(layer)
    return merged


def _explicit_cli_dests(argv):
    """Names of the options given explicitly on the command line."""
    p = build_parser()
    for a in p._actions:
        if a.dest != "help":
            a.default = argparse.SUPPRESS
    ns, _ = p.parse_known_args(argv)
    return set(vars(ns))


def _substitute_placeholders(args):
    """Fill {molecule} / {seed} / {experiment} in every string option.

    Returns the set of placeholder names that were used but have no value.
    """
    missing = set()
    values = {"molecule": args.molecule, "seed": args.seed, "experiment": None}

    def _sub(text):
        def repl(m):
            val = values[m.group(1)]
            if val is None:
                missing.add(m.group(1))
                return m.group(0)
            return str(val)
        return _PLACEHOLDER_RE.sub(repl, text)

    if isinstance(args.experiment, str):
        args.experiment = _sub(args.experiment)
    values["experiment"] = args.experiment
    for key, val in vars(args).items():
        if isinstance(val, str) and key != "experiment":
            setattr(args, key, _sub(val))
    return missing


def parse_args(argv=None):
    """CLI + config files -> resolved argparse Namespace (see module docstring)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    parser = build_parser()

    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", action="append", default=None)
    pre_args, _ = pre.parse_known_args(argv)
    config_paths = pre_args.config or []

    if config_paths:
        parser.set_defaults(**_merge_config_files(parser, config_paths))
    args = parser.parse_args(argv)
    args.config = config_paths or None

    # kv_mode on the CLI overrides k_mode / v_mode from config files unless
    # those are given on the CLI too; then resolve the shorthand.
    cli = _explicit_cli_dests(argv)
    if "kv_mode" in cli:
        for sub in ("k_mode", "v_mode"):
            if sub not in cli:
                setattr(args, sub, None)
    base = args.kv_mode or "o2"
    if args.k_mode is None:
        args.k_mode = base
    if args.v_mode is None:
        args.v_mode = base

    # Deprecated options: warn, then drop so they do not reach args.json.
    used = [k for k in DEPRECATED_KEYS if getattr(args, k, None) is not None]
    if used:
        logger.warning("Ignoring deprecated option(s) with no effect in this code "
                       "version: %s", ", ".join(used))
    for k in DEPRECATED_KEYS:
        if hasattr(args, k):
            delattr(args, k)

    if args.out_dir is None:
        args.out_dir = ("runs/{experiment}/{molecule}/s{seed}" if args.molecule
                        else "runs/{experiment}/s{seed}")
    if args.eval_batch_size is not None and args.eval_batch_size <= 0:
        args.eval_batch_size = None

    missing = _substitute_placeholders(args)
    if missing:
        users = sorted(k for k, v in vars(args).items()
                       if isinstance(v, str) and _PLACEHOLDER_RE.search(v))
        msg = (f"placeholder(s) {sorted('{' + m + '}' for m in missing)} used in "
               f"{users} but no value is set (pass --molecule)")
        if args.dry_run:
            logger.warning("%s; left unresolved for --dry_run", msg)
        else:
            parser.error(msg)
    return parser, args


# ---------------------------------------------------------------------------
# make_fx switches
# ---------------------------------------------------------------------------

def _all_same_atom_count(path):
    """True/False if every structure in ``path`` has the same atom count,
    None if that cannot be determined (e.g. the file does not exist yet)."""
    if not path or not os.path.isfile(path):
        return None
    try:
        import numpy as np
        with np.load(path, allow_pickle=False) as f:
            n = np.asarray(f["N"])
        return bool(n.size > 0 and (n == n.flat[0]).all())
    except Exception as e:                  # noqa: BLE001
        logger.warning("could not read atom counts from %s (%s)", path, e)
        return None


def _resolve_switch(value, env_name, auto_value, auto_reason):
    """Map an auto/on/off option onto its env var; returns (on, reason).

    ``on``/``off`` always win.  Under ``auto`` an already-exported env var is
    honoured; otherwise ``auto_value`` (None = undecidable -> env left unset,
    i.e. the library default "off").
    """
    if value in ("on", "off"):
        os.environ[env_name] = "1" if value == "on" else "0"
        return value == "on", f"--{value}"
    if env_name in os.environ:
        return os.environ[env_name] == "1", f"auto: {env_name}={os.environ[env_name]} from environment"
    if auto_value is None:
        return None, f"auto: {auto_reason}"
    os.environ[env_name] = "1" if auto_value else "0"
    return auto_value, f"auto: {auto_reason}"


def resolve_makefx(args):
    """Set ATOMTRICK_MAKEFX / ATOMTRICK_MAKEFX_DUCKOFF (before the trainer is imported)."""
    import torch
    on_cuda = torch.cuda.is_available() and (
        args.device is None or str(args.device).startswith("cuda"))
    makefx, why = _resolve_switch(
        args.makefx, "ATOMTRICK_MAKEFX", on_cuda,
        "CUDA device" if on_cuda else "no CUDA device -> eager")
    same = _all_same_atom_count(args.data)
    duck_reason = ("training data not readable yet -> duck-shaping stays on"
                   if same is None else
                   "all training structures have the same atom count" if same else
                   "training structures have varying atom counts")
    duckoff, why_d = _resolve_switch(args.makefx_duckoff, "ATOMTRICK_MAKEFX_DUCKOFF",
                                     same, duck_reason)
    logger.info("make_fx: %s (%s); duck-shaping off: %s (%s)",
                "on" if makefx else "off", why,
                {True: "yes", False: "no", None: "no"}[duckoff], why_d)
    if makefx and os.environ.get("PYTORCH_JIT", "0") != "0":
        logger.warning("PYTORCH_JIT=%s is set but the make_fx path needs "
                       "PYTORCH_JIT=0 (TorchScript breaks its trace).",
                       os.environ["PYTORCH_JIT"])
    return {"makefx": bool(makefx), "makefx_duckoff": duckoff,
            "PYTORCH_JIT": os.environ.get("PYTORCH_JIT")}


def clamp_eval_batch_size(args, makefx):
    """Keep ``eval_batch_size <= batch_size`` when validation runs on the make_fx graph.

    The compiled graph has the TRAINING molecule count baked in (training batches
    are always full), and validation reuses it; a larger eval batch would only
    fail at the first validation, hours into the run.  Smaller eval batches are
    padded, so clamping changes nothing but the validation batching.
    """
    if not makefx:
        return
    eval_bs = args.eval_batch_size or args.batch_size
    if eval_bs > args.batch_size:
        logger.warning("eval_batch_size %d > batch_size %d: the make_fx graph fixes the "
                       "molecule count at the training batch size, so validation uses "
                       "eval_batch_size %d", eval_bs, args.batch_size, args.batch_size)
        args.eval_batch_size = args.batch_size


def check_legacy_env():
    """Refuse to run with pre-rename ATOMTRICKS_* variables (silently ignored otherwise)."""
    stale = sorted(k for k in os.environ if k.startswith("ATOMTRICKS_"))
    if stale:
        raise SystemExit(
            "error: environment variable(s) " + ", ".join(stale) + " use the old "
            "spelling from before the package rename AtomTricks -> AtomTrick.  The "
            "current code ignores them (e.g. ATOMTRICKS_MAKEFX=1 would silently "
            "train on the slow eager path).  Unset them and use --makefx / "
            "--makefx_duckoff (or ATOMTRICK_MAKEFX / ATOMTRICK_MAKEFX_DUCKOFF).")


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def dry_run(args, config):
    """Build the model on CPU and print the resolved configuration."""
    import torch
    from AtomTrick.trainer import Trainer

    torch.manual_seed(args.seed)
    model = Trainer.build_model(config)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    sd = model.state_dict()
    sd_mb = sum(t.numel() * t.element_size() for t in sd.values()) / 2**20
    shown = dict(config)
    if "atomic_ref" not in config:
        shown["atomic_ref"] = ("<fitted on the training data at run time>"
                               if config.get("fit_atomic_refs")
                               else "<constants.ATOMIC_REFERENCE_ENERGIES>")
    print("Resolved configuration:")
    print(json.dumps(shown, indent=2, sort_keys=True, default=str))
    print(f"state_dict: {len(sd)} tensors, {sd_mb:.1f} MB")
    print(f"Model parameters: {n_params:,}")


def main(argv=None):
    check_legacy_env()
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s",
                        datefmt="%H:%M:%S")
    parser, args = parse_args(argv)
    logging.getLogger().setLevel(getattr(logging, args.log_level))

    # ---- Auto-process raw dataset if requested -------------------------
    if args.raw_data_dir is not None:
        if args.dataset is None or args.molecule is None:
            parser.error("--raw_data_dir requires --dataset and --molecule")
        if args.data is not None:
            parser.error("--data and --raw_data_dir are mutually exclusive")
        if args.dry_run:
            logger.info("--dry_run: skipping auto-processing of %s/%s",
                        args.dataset, args.molecule)
        else:
            from AtomTrick.data import process_and_resolve
            proc_kwargs = {}
            if args.dataset == "rmd17":
                proc_kwargs["split_id"] = args.split_id
            paths = process_and_resolve(
                dataset=args.dataset,
                raw_data_dir=args.raw_data_dir,
                molecule=args.molecule,
                processed_dir=args.processed_dir,
                **proc_kwargs,
            )
            args.data = str(paths["train"])
            args.val_data = str(paths["val"])
            args.test_data = str(paths["test"])
            logger.info("Auto-processed %s/%s -> train=%s val=%s test=%s",
                        args.dataset, args.molecule,
                        args.data, args.val_data, args.test_data)
    elif args.data is None and not args.dry_run:
        parser.error("Must provide either --data or --raw_data_dir + --dataset + --molecule")

    # make_fx env vars must be in place before AtomTrick.trainer is imported.
    provenance = resolve_makefx(args)
    provenance["matmul_precision"] = args.matmul_precision
    clamp_eval_batch_size(args, provenance["makefx"])

    config = vars(args).copy()
    config.pop("dry_run", None)
    config["_provenance"] = provenance

    # ---- Atomic-ref precedence --------------------------------------
    # 1. --fit_atomic_refs: least-squares fit on the training set.
    # 2. Per-dataset canonical refs (constants.py) when --dataset is set.
    # 3. MLFFDataset's built-in default (ATOMIC_REFERENCE_ENERGIES).
    if args.fit_atomic_refs:
        if args.dry_run:
            logger.info("--dry_run: atomic reference energies would be fitted on "
                        "%s (skipped; they do not change the architecture)", args.data)
        else:
            from AtomTrick.data import fit_atomic_reference_energies
            fitted = fit_atomic_reference_energies(config["data"])
            logger.info("Fitted atomic_ref_energy on %s: %s",
                        config["data"],
                        {z: round(e, 6) for z, e in fitted.items()})
            config["atomic_ref"] = fitted
    elif args.dataset is not None:
        from AtomTrick.constants import ATOMIC_REFERENCE_ENERGIES_BY_DATASET
        canonical = ATOMIC_REFERENCE_ENERGIES_BY_DATASET.get(args.dataset)
        if canonical is not None:
            config["atomic_ref"] = dict(canonical)
            logger.info("Using canonical %s atomic_ref_energy: %s",
                        args.dataset,
                        {z: round(e, 6) for z, e in canonical.items()})

    if args.dry_run:
        dry_run(args, config)
        return

    from AtomTrick.trainer import Trainer

    trainer = Trainer.from_config(config)

    train_loader, val_loader, test_loader = Trainer.build_loaders(config)

    if args.resume is not None:
        trainer.load_checkpoint(args.resume)

    trainer.fit(args.max_iterations, train_loader, val_loader, test_loader)

    if test_loader is not None:
        trainer.inference(test_loader)


if __name__ == "__main__":
    main()
