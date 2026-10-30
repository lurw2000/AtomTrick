"""OpenMM MD driver for the ATOMTRICK force field: NVE energy conservation and 300 K NVT.

Runs gas-phase MD (Ac-Ala3-NHMe by default) with forces supplied EACH STEP by the PyTorch model,
injected into OpenMM through a CustomExternalForce whose per-particle (fx, fy, fz) parameters
are refreshed from the model after every step.  (This is not openmm-torch: the model runs in
Python and OpenMM only integrates.)

Integrators (both custom):
  * NVE : leapfrog  (v += dt*f/m ; x += dt*v)                  -> energy-drift test
  * NVT : leapfrog Langevin B-O-A (one force eval/step + OU)    -> 300 K stability / sampling
  NVE runs first equilibrate with ``--equil-steps`` of the NVT integrator, then hand off.

RESUMABLE: every ``--ckpt-every`` steps the run atomically checkpoints step + positions +
velocities (portable .npz, no OpenMM binary blob) plus the energy CSV and trajectory.  The
CSV/trajectory are flushed ONLY at checkpoints, so re-running the SAME command after a kill
continues cleanly from the last checkpoint without duplicate rows.

Units: OpenMM internal (nm, ps, dalton, kJ/mol).  Model I/O is Angstrom and eV, eV/A.

Protocol details
----------------
1. Coordinates are centred inside the calculator (``R - R.mean(0)``) before every model call.
   The model only uses interatomic vectors, so this is exact, but in float32 the absolute
   coordinate magnitude sets the representable spacing: a molecule that has drifted far from
   the origin loses precision in every bond vector (measured at ~1100 A, centring cut the force
   error vs an fp64 reference 184x, 3.88e-3 -> 2.11e-5 eV/A).
2. Centre-of-mass momentum is removed at the NVE hand-off, so the molecule does not coast away
   ballistically with the thermalised Langevin velocities.
3. The Langevin noise is seeded from ``--seed`` (``CustomIntegrator.setRandomNumberSeed``),
   with a per-stage / per-resume-step offset.  GPU force evaluation is not bitwise
   deterministic, so trajectories are statistically, not bitwise, reproducible.
4. Temperature degrees of freedom (``--dof auto``): 3N while a Langevin thermostat acts (it
   thermalises the centre-of-mass motion too) and 3N-3 in NVE, whose COM momentum is removed
   at the hand-off.
5. The ATOMTRICK architecture is read from the checkpoint directory's ``args.json`` (or
   ``--config``) and built through ``Trainer.from_config``.
6. ``--compile`` traces the make_fx graph with the create_graph=True force trace (the one the
   trainer uses for inference) and calls it under no_grad.  Forces agree with eager to fp32
   noise.
7. Matmul precision is set explicitly (``--matmul-precision``, default ``highest`` = no TF32).

The reported runs used the full-fp32 MD22 Ac-Ala3-NHMe checkpoint (paper model, seed 42); see
``md/README.md``.
"""
import argparse
import json
import os
import shutil
import sys
import tempfile
import time

import numpy as np

# AtomTrick env (must precede its import)
os.environ.setdefault("PYTORCH_JIT", "0")                 # TorchScript breaks make_fx 2nd-order trace
os.environ.setdefault("ATOMTRICK_MAKEFX_DUCKOFF", "1")    # fixed-size molecule under make_fx (--compile)

import openmm as mm  # noqa: E402
from openmm import unit  # noqa: E402
from openmm.app import element as openmm_element  # noqa: E402

REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

EV_PER_KJMOL = 1.0 / 96.48533212       # kJ/mol -> eV
KJMOLNM_PER_EVA = 96.48533212 * 10.0   # eV/A  -> kJ/mol/nm
KB_KJ = 0.00831446261815324            # kJ/mol/K


# --------------------------------------------------------------------------- #
#  Initial frame
# --------------------------------------------------------------------------- #
def load_init_frame(npz_path, index):
    """Return (z:int (na,), R_A:float64 (na,3) centred) from a processed npz."""
    d = np.load(npz_path)
    nf = len(d["E"]) if "E" in d else len(np.ravel(d["N"]))
    na = int(np.ravel(d["N"])[0])
    R = np.asarray(d["R"], np.float64).reshape(nf, na, 3)[index % nf]
    z = np.asarray(d["Z"], np.int64).reshape(nf, na)[0]
    R = R - R.mean(axis=0)
    return z, R


# --------------------------------------------------------------------------- #
#  Calculators: energy_forces(R_A) -> (E_eV: float, F_eVA: (na,3) np.float64)
# --------------------------------------------------------------------------- #
def _load_config_files(paths):
    """Resolve ATOMTRICK model config files (same rules as benchmarks/latency/bench_atomtrick.py).

    A run's ``args.json`` is used verbatim (``Trainer.build_model``'s fallbacks rebuild exactly
    the model an old run trained); YAML experiment configs are resolved by
    ``train.parse_args`` (argparse defaults = the paper model, later files override earlier).
    """
    if not all(p.endswith(".json") for p in paths):
        try:
            if REPO_ROOT not in sys.path:
                sys.path.insert(0, REPO_ROOT)
            import train as _train
            argv = [a for p in paths for a in ("--config", p)] + ["--dry_run"]
            _, ns = _train.parse_args(argv)
            return {k: v for k, v in vars(ns).items() if not k.startswith("_")}
        except (ImportError, AttributeError) as e:
            print(f"WARNING: train.parse_args unavailable ({e}); merging configs verbatim",
                  file=sys.stderr)
    cfg = {}
    for p in paths:
        with open(p) as f:
            if p.endswith(".json"):
                d = json.load(f)
            else:
                import yaml
                d = yaml.safe_load(f) or {}
        cfg.update({k: v for k, v in d.items() if not k.startswith("_")})
    return cfg


class AtomTrickCalc:
    """ATOMTRICK energy/forces.  The architecture comes from ``config_paths`` or, by default,
    the ``args.json`` written by train.py next to the checkpoint."""

    def __init__(self, ckpt, device, dtype, config_paths=None, cutoff=None, compile=False,
                 matmul_precision="highest"):
        import torch
        from scipy.spatial import cKDTree
        if REPO_ROOT not in sys.path:
            sys.path.insert(0, REPO_ROOT)
        from AtomTrick.trainer import Trainer, inference_cutoff
        self.torch = torch
        self.cKDTree = cKDTree
        self.device = device

        if not config_paths:
            aj = os.path.join(os.path.dirname(os.path.abspath(ckpt)), "args.json")
            if not os.path.exists(aj):
                raise SystemExit(f"no args.json next to {ckpt}; pass the model config "
                                 "with --config (YAML/JSON, train.py key names)")
            config_paths = [aj]
        cfg = _load_config_files(config_paths)
        # Neighbour list = the model's radial cutoff; a different --cutoff is an error.
        self.cutoff = inference_cutoff(cfg, cutoff)
        base = cfg.get("kv_mode") or "o2"
        cfg["k_mode"] = cfg.get("k_mode") or base
        cfg["v_mode"] = cfg.get("v_mode") or base
        tmp = tempfile.mkdtemp(prefix="atomtrick_md_")
        cfg.update({"device": str(device), "dtype": dtype, "wandb_project": None,
                    "_skip_save_args": True, "out_dir": tmp, "seed": 0})
        try:
            trainer = Trainer.from_config(cfg)
            raw = torch.load(ckpt, map_location="cpu", weights_only=False)  # trusted file
            if isinstance(raw, dict) and "model" in raw:
                trainer.load_checkpoint(ckpt)      # full dict: swap in the EMA weights
                trainer._ema_apply()
            else:
                trainer.model.load_state_dict(raw)  # *_best.pt = raw EMA state_dict (strict)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
        # Trainer.from_config may enable TF32; MD forces use the precision asked for here.
        torch.set_float32_matmul_precision(matmul_precision)
        self.model = trainer.model.eval()
        self.tdt = next(self.model.parameters()).dtype
        self.config_paths = config_paths
        self.compile = compile      # make_fx + torch.compile inference (~3-5x faster MD)
        self.compiled = None        # built lazily from the first geometry (dynamic edges)

    def set_atoms(self, z):
        torch = self.torch
        self._z = np.asarray(z, np.int64)
        self._z_t = torch.from_numpy(self._z).to(self.device)
        self._batch = torch.zeros(len(self._z), dtype=torch.long, device=self.device)
        self._N = torch.tensor([len(self._z)], dtype=torch.long, device=self.device)

    def _build_batch(self, R_A):
        torch = self.torch
        pairs = self.cKDTree(R_A).query_pairs(self.cutoff, output_type="ndarray")
        pairs = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
        edge = torch.from_numpy(pairs.T.copy()).long().to(self.device)
        R = torch.tensor(R_A, dtype=self.tdt, device=self.device)
        return {"Z": self._z_t, "R": R, "N": self._N, "edge_index": edge, "batch": self._batch}

    def __call__(self, R_A):
        # Centre (exact: translation invariance); keeps float32 coordinates small (item 1).
        R_A = np.asarray(R_A, np.float64)
        R_A = R_A - R_A.mean(axis=0)
        torch = self.torch
        if self.compile:
            from AtomTrick.backward.makefx_compiled import MakeFXCompiledStep, plainify
            if self.compiled is None:
                # One dynamic graph (edge count changes every step), create_graph=True
                # trace (item 6); outputs may carry a symbolic tag -> plainify.
                with torch.enable_grad():
                    self.compiled = MakeFXCompiledStep(
                        self.model, self._build_batch(R_A), dynamic=True)
            with torch.no_grad():
                out = self.compiled.step(self._build_batch(R_A))
            E = float(plainify(out["E"]).reshape(-1)[0].item())
            F = plainify(out["F"]).to("cpu").numpy().astype(np.float64)
            return E, F
        out = self.model(self._build_batch(R_A))
        E = float(out["E"].reshape(-1)[0].item())
        F = out["F"].detach().to("cpu").numpy().astype(np.float64)
        return E, F


O3AttnCalc = AtomTrickCalc   # pre-rename name, kept for old notebooks


class HarmonicCalc:
    """Fast dummy potential for CPU plumbing tests: isotropic tether to R0.
    E = 0.5 k |R-R0|^2 (eV), F = -k (R-R0) (eV/A).  Not translation invariant -> never centred."""
    def __init__(self, k=2.0):
        self.k = k

    def set_atoms(self, z):
        self._n = len(z)

    def set_ref(self, R0):
        self.R0 = np.asarray(R0, np.float64)

    def __call__(self, R_A):
        d = np.asarray(R_A, np.float64) - self.R0
        return 0.5 * self.k * float((d * d).sum()), -self.k * d


# --------------------------------------------------------------------------- #
#  OpenMM system + integrators
# --------------------------------------------------------------------------- #
def build_system(z):
    system = mm.System()
    masses = []
    for zi in z:
        m = openmm_element.Element.getByAtomicNumber(int(zi)).mass.value_in_unit(unit.dalton)
        system.addParticle(m)
        masses.append(m)
    ext = mm.CustomExternalForce("-(fx*x+fy*y+fz*z)")
    for p in ("fx", "fy", "fz"):
        ext.addPerParticleParameter(p)
    for i in range(len(z)):
        ext.addParticle(i, [0.0, 0.0, 0.0])
    system.addForce(ext)
    return system, ext, np.asarray(masses)


def make_nve_integrator(dt_ps):
    integ = mm.CustomIntegrator(dt_ps)
    integ.addUpdateContextState()
    integ.addComputePerDof("v", "v + dt*f/m")   # leapfrog kick (f = injected ML force)
    integ.addComputePerDof("x", "x + dt*v")     # drift
    return integ


def make_nvt_integrator(dt_ps, temp, friction, noise_seed=None):
    """Leapfrog Langevin (B-O-A): ONE force eval per step, kick-first.

    The MLFF force is injected as a CustomExternalForce whose params are held
    CONSTANT for the whole integ.step(1). A leapfrog (kick-then-drift) uses the
    force exactly once per step, so the constant force is exactly right. A BAOAB
    (B-A-O-A-B) would use the force in TWO half-kicks; the second needs the
    post-drift force but would get the stale start-of-step value, injecting energy
    every step (runaway heating). B-O-A keeps one force eval/step + an OU thermostat;
    a=exp(-gamma*dt), b=sqrt(1-a^2) give <v^2>=kT/m at stationarity.

    ``noise_seed``: seed of the integrator's Gaussian stream (item 3).
    """
    from math import exp, sqrt
    integ = mm.CustomIntegrator(dt_ps)
    integ.addGlobalVariable("a", exp(-friction * dt_ps))
    integ.addGlobalVariable("b", sqrt(1.0 - exp(-2.0 * friction * dt_ps)))
    integ.addGlobalVariable("kT", KB_KJ * temp)
    integ.addUpdateContextState()
    integ.addComputePerDof("v", "v + dt*f/m")                   # B: full kick @ x(t)
    integ.addComputePerDof("v", "a*v + b*sqrt(kT/m)*gaussian")  # O: OU thermostat
    integ.addComputePerDof("x", "x + dt*v")                     # A: full drift
    if noise_seed is not None:
        integ.setRandomNumberSeed(int(noise_seed))
    return integ


def noise_seed_for(seed, stage, start_step):
    """Deterministic, nonzero Langevin seed per (seed, stage, resume step).

    Offsetting by the resume step keeps a resumed run from replaying the noise sequence of
    its first segment (OpenMM treats 0 as "pick a random seed", hence the +1)."""
    stage_off = {"equil": 0, "prod": 500_000_003}[stage]
    return 1 + (int(seed) * 1_000_003 + stage_off + int(start_step)) % (2 ** 31 - 2)


# --------------------------------------------------------------------------- #
#  Positions / force injection / context
# --------------------------------------------------------------------------- #
def positions_A(context):
    st = context.getState(getPositions=True)
    return st.getPositions(asNumpy=True).value_in_unit(unit.nanometer) * 10.0


def inject(ext, context, F_eVA):
    p = F_eVA * KJMOLNM_PER_EVA
    for i in range(F_eVA.shape[0]):
        ext.setParticleParameters(i, i, [float(p[i, 0]), float(p[i, 1]), float(p[i, 2])])
    ext.updateParametersInContext(context)


def make_context(system, integ, R_A, temp, seed, remove_com, velocities=None,
                 platform="CPU"):
    plat = mm.Platform.getPlatformByName(platform)
    props = {"Precision": "double"} if platform == "CUDA" else {}
    context = mm.Context(system, integ, plat, props)
    context.setPositions((R_A / 10.0) * unit.nanometer)
    if velocities is not None:
        context.setVelocities(velocities)
    else:
        context.setVelocitiesToTemperature(temp * unit.kelvin, seed)
    if remove_com:
        st = context.getState(getVelocities=True)
        v = st.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
        m = np.array([system.getParticleMass(i).value_in_unit(unit.dalton)
                      for i in range(system.getNumParticles())])
        vcom = (m[:, None] * v).sum(0) / m.sum()
        v = v - vcom
        context.setVelocities(v * (unit.nanometer / unit.picosecond))
    return context


# --------------------------------------------------------------------------- #
#  Checkpoint / resume  (portable: positions + velocities, not the OpenMM blob)
# --------------------------------------------------------------------------- #
def _state_path(out, stage):
    return f"{out}_{stage}_state.npz"


def _atomic_savez(path, compressed=False, **arrays):
    fn = np.savez_compressed if compressed else np.savez
    fn(path + ".tmp", **arrays)          # numpy appends .npz
    os.replace(path + ".tmp.npz", path)


def save_state(out, stage, step, t0, context, traj, z, traj_every, dt_ps):
    """Atomically persist step + positions + velocities for `stage`, and (if any)
    the trajectory. Portable across processes/nodes."""
    st = context.getState(getPositions=True, getVelocities=True)
    R_nm = st.getPositions(asNumpy=True).value_in_unit(unit.nanometer)
    V_nmps = st.getVelocities(asNumpy=True).value_in_unit(unit.nanometer / unit.picosecond)
    _atomic_savez(_state_path(out, stage), step=step, t0=t0,
                  R_nm=R_nm, V_nmps=V_nmps)
    if traj:
        _atomic_savez(f"{out}_traj.npz", compressed=True, z=z,
                      positions=np.asarray(traj, np.float32),
                      stride_ps=traj_every * dt_ps)


def load_state(out, stage):
    p = _state_path(out, stage)
    if not os.path.exists(p):
        return None
    d = np.load(p)
    return {"step": int(d["step"]), "t0": float(d["t0"]),
            "R_A": np.asarray(d["R_nm"], np.float64) * 10.0,
            "V_nmps": np.asarray(d["V_nmps"], np.float64)}


def load_traj(out):
    p = f"{out}_traj.npz"
    if not os.path.exists(p):
        return []
    return [f for f in np.load(p)["positions"]]


def _vel_quantity(v_nmps):
    return v_nmps * (unit.nanometer / unit.picosecond)


# --------------------------------------------------------------------------- #
#  Simulation loop (resumable)
# --------------------------------------------------------------------------- #
def simulate(context, integ, ext, calc, total_steps, dt_ps, dof, report_every,
             traj_every, out, stage, t0, start_step, ckpt_every, z,
             log_csv=True, tag=""):
    """Run/resume `stage` from start_step (exclusive) to total_steps. Checkpoints
    every ckpt_every steps; CSV + traj are flushed only at checkpoints so they never
    lead the state -> clean, duplicate-free resume."""
    import csv as _csv
    csvf = w = None
    if log_csv:
        csv_path = out + "_energy.csv"
        csvf = open(csv_path, "w" if start_step == 0 else "a", newline="")
        w = _csv.writer(csvf)
        if start_step == 0:
            w.writerow(["tag", "step", "time_ps", "Epot_eV", "Ekin_eV", "Etot_eV", "T_K"])
            csvf.flush()
    traj = load_traj(out) if (traj_every and start_step > 0) else []
    buf = []

    def flush(step):
        if log_csv and buf:
            w.writerows(buf); csvf.flush(); buf.clear()
        save_state(out, stage, step, t0, context, traj, z, traj_every, dt_ps)

    R = positions_A(context)
    E, F = calc(R)
    inject(ext, context, F)
    wall = time.time()
    for step in range(start_step + 1, total_steps + 1):
        integ.step(1)
        R = positions_A(context)
        if not np.isfinite(R).all():
            print(f"[{tag}] NaN positions at step {step} — aborting", flush=True)
            if csvf:
                csvf.close()
            return False
        E, F = calc(R)
        inject(ext, context, F)
        if report_every and step % report_every == 0:
            ke = context.getState(getEnergy=True).getKineticEnergy().value_in_unit(
                unit.kilojoule_per_mole)
            ke_eV = ke * EV_PER_KJMOL
            T = 2.0 * ke / (dof * KB_KJ)
            t_ps = t0 + step * dt_ps
            if log_csv:
                buf.append((tag, step, t_ps, E, ke_eV, E + ke_eV, T))
            rate = (step - start_step) / (time.time() - wall)
            print(f"[{tag}] step {step:>9d}/{total_steps}  t={t_ps:9.2f}ps  "
                  f"Epot={E:12.4f}eV  Etot={E + ke_eV:12.4f}eV  T={T:7.2f}K  "
                  f"({rate:5.1f} it/s)", flush=True)
        if traj_every and step % traj_every == 0:
            traj.append(R.astype(np.float32))
        if ckpt_every and step % ckpt_every == 0:
            flush(step)
    flush(total_steps)
    if csvf:
        csvf.close()
    return True


def main():
    ap = argparse.ArgumentParser(
        description="OpenMM MD with the ATOMTRICK force field (see module docstring).")
    ap.add_argument("--model", choices=["atomtrick", "o3attn", "harmonic"],
                    required=True, help="'o3attn' is the pre-rename alias of 'atomtrick'; "
                                        "'harmonic' is a dummy potential for plumbing tests")
    ap.add_argument("--ckpt", default=None,
                    help="ATOMTRICK checkpoint (*_best.pt / *_final.pt)")
    ap.add_argument("--config", action="append", default=None,
                    help="ATOMTRICK model config(s), YAML/JSON with train.py keys (default: "
                         "args.json next to --ckpt)")
    ap.add_argument("--ensemble", choices=["nve", "nvt"], required=True)
    ap.add_argument("--init-npz", default=os.path.join("data", "md22",
                                                       "Ac-Ala3-NHMe_test_default.npz"),
                    help="processed npz holding the initial frame (default: MD22 "
                         "Ac-Ala3-NHMe test split, relative to the working directory)")
    ap.add_argument("--init-index", type=int, default=0)
    ap.add_argument("--temp", type=float, default=300.0)
    ap.add_argument("--dt-fs", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=5000000)      # 5 ns @ 1 fs
    ap.add_argument("--equil-steps", type=int, default=10000)  # NVT pre-equil for NVE
    ap.add_argument("--friction", type=float, default=1.0)     # 1/ps (Langevin)
    ap.add_argument("--report-every", type=int, default=2000)
    ap.add_argument("--traj-every", type=int, default=2000)
    ap.add_argument("--ckpt-every", type=int, default=10000)   # checkpoint cadence
    ap.add_argument("--dtype", choices=["float32", "float64"], default="float32",
                    help="model precision (reported runs: float32)")
    ap.add_argument("--matmul-precision", choices=["highest", "high"], default="highest",
                    help="ATOMTRICK fp32 matmul precision; 'high' enables TF32 "
                         "(reported runs: highest)")
    ap.add_argument("--cutoff", type=float, default=None,
                    help="ATOMTRICK neighbour-list cutoff in A (default: the model's radial "
                         "cutoff; any other value is rejected)")
    ap.add_argument("--compile", action="store_true",
                    help="ATOMTRICK only: make_fx + torch.compile inference")
    ap.add_argument("--platform", choices=["CUDA", "CPU"], default="CPU",
                    help="OpenMM platform for the integrator (reported runs: CPU, fp64)")
    ap.add_argument("--seed", type=int, default=2024,
                    help="seeds the initial velocities and the Langevin noise")
    ap.add_argument("--device", default="cuda", help="torch device for the model")
    ap.add_argument("--dof", choices=["auto", "3N", "3N-3"], default="auto",
                    help="temperature degrees of freedom; auto = 3N under Langevin, 3N-3 in NVE")
    ap.add_argument("--out", required=True, help="output prefix, e.g. runs/md/atomtrick_nve")
    args = ap.parse_args()
    if args.model == "o3attn":
        args.model = "atomtrick"
    if args.model != "harmonic" and not args.ckpt:
        ap.error("--ckpt is required for --model atomtrick")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    z, R0 = load_init_frame(args.init_npz, args.init_index)
    na = len(z)
    if args.dof != "auto":
        dof_lan = dof_nve = 3 * na - (3 if args.dof == "3N-3" else 0)
    else:
        dof_lan, dof_nve = 3 * na, 3 * na - 3
    dt_ps = args.dt_fs / 1000.0
    ck = args.ckpt_every

    if args.model == "atomtrick":
        calc = AtomTrickCalc(args.ckpt, args.device, args.dtype, config_paths=args.config,
                             cutoff=args.cutoff, compile=args.compile,
                             matmul_precision=args.matmul_precision)
        print(f"ATOMTRICK config: {calc.config_paths}  cutoff={calc.cutoff} A  "
              f"params={sum(p.numel() for p in calc.model.parameters()):,}", flush=True)
    else:
        calc = HarmonicCalc()
    calc.set_atoms(z)
    if args.model == "harmonic":
        calc.set_ref(R0)

    system, ext, masses = build_system(z)
    print(f"=== {args.model} {args.ensemble}  {na} atoms  dt={args.dt_fs}fs  "
          f"steps={args.steps}  T={args.temp}K  dtype={args.dtype}  "
          f"compile={args.compile}  "
          f"dof(langevin/nve)={dof_lan}/{dof_nve}  seed={args.seed}  ckpt_every={ck} ===",
          flush=True)

    # --- force self-check on the real device/dtype (guards a bad calc config).
    # Skipped for the harmonic plumbing potential (F=0 at its own reference R0). ---
    if args.model != "harmonic":
        _E0, _F0 = calc(R0)
        _fmax = float(np.abs(_F0).max())
        _eps = 0.02  # large enough that dE signal beats fp32 energy noise
        _Rp = R0.copy(); _Rp[0, 0] += _eps
        _Rm = R0.copy(); _Rm[0, 0] -= _eps
        _Ep, _ = calc(_Rp); _Em, _ = calc(_Rm)
        _mdE = -(_Ep - _Em) / (2 * _eps)
        _ratio = _F0[0, 0] / _mdE if _mdE != 0 else float("nan")
        print(f"[force-check] E0={_E0:.4f}eV |F|max={_fmax:.4f} mean={float(np.abs(_F0).mean()):.4f} eV/A "
              f"FDratio(a0x)={_ratio:+.4f}  (expect |F|max~5, ratio~1)", flush=True)
        if _fmax > 50.0 or not (0.5 < _ratio < 1.5):
            print(f"[force-check] FAILED (|F|max={_fmax:.2f}, ratio={_ratio:.3f}) — aborting before MD", flush=True)
            raise SystemExit(2)

    if args.ensemble == "nvt":
        st = load_state(args.out, "prod")
        start = st["step"] if st is not None else 0
        integ = make_nvt_integrator(dt_ps, args.temp, args.friction,
                                    noise_seed_for(args.seed, "prod", start))
        if st is not None:
            print(f"[resume] nvt prod from step {st['step']}/{args.steps}", flush=True)
            context = make_context(system, integ, st["R_A"], args.temp, args.seed,
                                   remove_com=False, velocities=_vel_quantity(st["V_nmps"]),
                                   platform=args.platform)
        else:
            context = make_context(system, integ, R0, args.temp, args.seed, remove_com=True,
                                   platform=args.platform)
        ok = simulate(context, integ, ext, calc, args.steps, dt_ps, dof_lan, args.report_every,
                      args.traj_every, args.out, "prod", 0.0, start, ck, z,
                      log_csv=True, tag="nvt")
    else:
        prod_st = load_state(args.out, "prod")
        if prod_st is not None:
            # equil already finished; resume the NVE production directly.
            print(f"[resume] nve prod from step {prod_st['step']}/{args.steps}", flush=True)
            integ = make_nve_integrator(dt_ps)
            context = make_context(system, integ, prod_st["R_A"], args.temp, args.seed,
                                   remove_com=False, velocities=_vel_quantity(prod_st["V_nmps"]),
                                   platform=args.platform)
            ok = simulate(context, integ, ext, calc, args.steps, dt_ps, dof_nve, args.report_every,
                          args.traj_every, args.out, "prod", 0.0, prod_st["step"], ck, z,
                          log_csv=True, tag="nve")
        else:
            # NVE: (resumable) Langevin equilibration, then leapfrog production.
            if args.equil_steps > 0:
                eq_st = load_state(args.out, "equil")
                eq_start = min(eq_st["step"], args.equil_steps) if eq_st is not None else 0
                eq_int = make_nvt_integrator(dt_ps, args.temp, args.friction,
                                             noise_seed_for(args.seed, "equil", eq_start))
                if eq_st is not None:
                    tag = "resume" if eq_st["step"] < args.equil_steps else "done"
                    print(f"[resume] nve equil {tag} at step {eq_st['step']}/{args.equil_steps}",
                          flush=True)
                    eq_ctx = make_context(system, eq_int, eq_st["R_A"], args.temp, args.seed,
                                          remove_com=False, velocities=_vel_quantity(eq_st["V_nmps"]),
                                          platform=args.platform)
                else:
                    eq_ctx = make_context(system, eq_int, R0, args.temp, args.seed,
                                          remove_com=True, platform=args.platform)
                simulate(eq_ctx, eq_int, ext, calc, args.equil_steps, dt_ps, dof_lan,
                         args.report_every * 5, 0, args.out, "equil", 0.0, eq_start, ck, z,
                         log_csv=False, tag="equil")
                est = eq_ctx.getState(getPositions=True, getVelocities=True)
                R_start = est.getPositions(asNumpy=True).value_in_unit(unit.nanometer) * 10.0
                V_start = est.getVelocities()
                del eq_ctx, eq_int
            else:
                R_start, V_start = R0, None
            integ = make_nve_integrator(dt_ps)
            # Item 2: the Langevin hand-off velocities carry thermalised COM momentum; remove
            # it so the molecule does not coast away.
            context = make_context(system, integ, R_start, args.temp, args.seed,
                                   remove_com=True, velocities=V_start,
                                   platform=args.platform)
            ok = simulate(context, integ, ext, calc, args.steps, dt_ps, dof_nve, args.report_every,
                          args.traj_every, args.out, "prod", 0.0, 0, ck, z,
                          log_csv=True, tag="nve")

    print(f"DONE ok={ok}  wrote {args.out}_energy.csv + {args.out}_traj.npz", flush=True)
    if not ok:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
