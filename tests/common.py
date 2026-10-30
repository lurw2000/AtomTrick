"""Shared helpers for the AtomTrick test-suite.

Imported by the test modules as ``from common import ...`` (``conftest.py`` puts
this directory on ``sys.path``).  Everything here is plain torch/numpy -- no
dataset files, no GPU -- so the whole suite runs on a laptop CPU.

Conventions used throughout the tests
-------------------------------------
* A *batch* is the dict the model consumes: ``Z`` (atoms,), ``R`` (atoms, 3),
  ``edge_index`` (2, edges) as ``[src, dst]``, ``batch`` (atoms,) molecule
  index, ``N`` (molecules,) atom counts.  The model uses ``R[dst] - R[src]``.
* A group element ``g`` (3x3, proper or improper) acts on positions as
  ``R -> R @ g.T``; equivariant forces must then satisfy ``F -> F @ g.T``.
"""
from __future__ import annotations

import math
import os
import subprocess
import sys
from typing import Dict, Iterable, Optional, Sequence, Tuple

import pytest
import torch

from AtomTrick.model import AtomTrick

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#: Marker for tests that need a GPU (the production make_fx path is CUDA-only).
requires_cuda = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a CUDA GPU")

# ---------------------------------------------------------------------------
# Model configurations
# ---------------------------------------------------------------------------

#: Constructor kwargs of the FINAL (paper main) model: 6 layers, O(2)-linear K/V
#: with radial gates, identity (unnormalised) attention, shared equivariant RMS
#: LayerNorm, gated FFN, scalar readout -> 3,730,368 parameters.  Mirrors the
#: argparse defaults of train.py / configs/rmd17/atomtrick.yaml.
FINAL_MODEL_KWARGS: Dict = dict(
    irreps_hidden="64x0e+64x1o+64x2e+64x3o",
    n_heads=8,
    n_layers=6,
    ff_mult=4,
    max_z=100,
    num_radial_basis=32,
    cutoff=5.0,
    radial_hidden=256,
    energy_hidden=256,
    radial_mode="gate",
    rbf_type="expnorm",
    k_mode="o2",
    v_mode="o2",
    ffn_act_mode="gate",
    attn_nonlin="identity",
    layernorm_mode="shared",
    escn_frame="polefree",
)

#: Same architecture, shrunk for fast CPU tests.  Lmax stays 3 and every
#: architectural switch stays at its final value; only widths/depth shrink.
#: Two layers are the minimum for the l>0 channels to feed back into the
#: attention scores (layer 1 only sees the l=0 atom embedding).
SMALL_OVERRIDES: Dict = dict(
    irreps_hidden="8x0e+8x1o+8x2e+8x3o",
    n_heads=2,
    n_layers=2,
    num_radial_basis=8,
    radial_hidden=16,
    energy_hidden=16,
    max_z=10,
)

ELEMENTS = (1, 6, 7, 8)   # H C N O -- all have default atomic reference energies


def build_model(*, dtype=torch.float64, seed: int = 0, small: bool = True,
                train: bool = False, device="cpu", **overrides) -> AtomTrick:
    """Final-configuration model (``small=True``: reduced width/depth).

    ``kv=<mode>`` is accepted as a shorthand for ``k_mode=v_mode=<mode>``.
    """
    kw = dict(FINAL_MODEL_KWARGS)
    if small:
        kw.update(SMALL_OVERRIDES)
    kv = overrides.pop("kv", None)
    if kv is not None:
        kw["k_mode"] = kw["v_mode"] = kv
    kw.update(overrides)
    torch.manual_seed(seed)
    model = AtomTrick(**kw).to(device=device, dtype=dtype)
    model.train(train)
    return model


def n_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Synthetic molecules
# ---------------------------------------------------------------------------

def radius_graph(R: torch.Tensor, mol: torch.Tensor, cutoff: float) -> torch.Tensor:
    """All ordered pairs (src, dst) of the same molecule closer than ``cutoff``."""
    d = torch.cdist(R, R)
    same = mol[:, None] == mol[None, :]
    eye = torch.eye(R.shape[0], dtype=torch.bool, device=R.device)
    src, dst = ((d < cutoff) & same & ~eye).nonzero(as_tuple=True)
    return torch.stack([src, dst])


def batch_from(Z: torch.Tensor, R: torch.Tensor, mol: torch.Tensor,
               cutoff: float = 5.0) -> Dict[str, torch.Tensor]:
    n_mol = int(mol.max().item()) + 1
    N = torch.bincount(mol, minlength=n_mol)
    return {"Z": Z, "R": R, "edge_index": radius_graph(R, mol, cutoff),
            "batch": mol, "N": N}


def make_batch(n_atoms: Sequence[int] = (6, 5), *, dtype=torch.float64, seed: int = 0,
               box: float = 2.6, min_dist: float = 0.75, cutoff: float = 5.0,
               elements: Sequence[int] = ELEMENTS) -> Dict[str, torch.Tensor]:
    """Random small molecules, one graph.

    Atoms are rejection-sampled in a ``box``^3 cube with a minimum separation,
    so every intra-molecular distance lies in ``[min_dist, box*sqrt(3)]``.  With
    the defaults that is below the 5 A cutoff: the graph is complete and cannot
    change under small (finite-difference) displacements.
    """
    g = torch.Generator().manual_seed(seed)
    els = torch.tensor(list(elements), dtype=torch.long)
    Zs, Rs, mols = [], [], []
    for m, n in enumerate(n_atoms):
        pos = []
        while len(pos) < n:
            p = torch.rand(3, generator=g, dtype=torch.float64) * box
            if all(float((p - q).norm()) > min_dist for q in pos):
                pos.append(p)
        Rs.append(torch.stack(pos))
        Zs.append(els[torch.randint(len(els), (n,), generator=g)])
        mols.append(torch.full((n,), m, dtype=torch.long))
    return batch_from(torch.cat(Zs), torch.cat(Rs).to(dtype), torch.cat(mols), cutoff)


def write_npz(path, n_frames: int = 12, composition: Sequence[int] = (6, 6, 8, 7, 1, 1, 1),
              seed: int = 0, box: float = 2.6, min_dist: float = 0.75) -> str:
    """Tiny synthetic dataset in the repository's NPZ format.

    Keys ``N, Z, R, F, E``; energies are *total* energies in kcal/mol (float64,
    of realistic magnitude: sum of the default atomic references plus a few eV),
    forces in kcal/mol/A -- exactly what ``scripts/data`` writes for rMD17/MD22.
    """
    import numpy as np
    from ase import units
    from AtomTrick.constants import ATOMIC_REFERENCE_ENERGIES

    rng = np.random.default_rng(seed)
    comp = np.asarray(composition, dtype=np.int64)
    n = len(comp)
    R = []
    for _ in range(n_frames):
        pos = []
        while len(pos) < n:
            p = rng.uniform(0.0, box, size=3)
            if all(np.linalg.norm(p - q) > min_dist for q in pos):
                pos.append(p)
        R.append(np.stack(pos))
    kcal = units.kcal / units.mol
    e_ref = sum(ATOMIC_REFERENCE_ENERGIES[int(z)] for z in comp)
    E = (e_ref + rng.uniform(-2.0, 2.0, size=n_frames)) / kcal
    F = rng.normal(0.0, 20.0, size=(n_frames * n, 3))
    np.savez(path, N=np.full(n_frames, n, dtype=np.int64), Z=np.tile(comp, n_frames),
             R=np.concatenate(R), F=F, E=E)
    return str(path)


def with_positions(batch: Dict[str, torch.Tensor], R: torch.Tensor,
                   cutoff: float = 5.0) -> Dict[str, torch.Tensor]:
    """Copy of ``batch`` with new positions and a rebuilt radius graph."""
    return batch_from(batch["Z"], R, batch["batch"], cutoff)


def to_device(batch: Dict[str, torch.Tensor], device) -> Dict[str, torch.Tensor]:
    return {k: v.to(device) for k, v in batch.items()}


# ---------------------------------------------------------------------------
# Group elements
# ---------------------------------------------------------------------------

def random_rotation(seed: int, dtype=torch.float64) -> torch.Tensor:
    """Haar-random proper rotation (det = +1)."""
    g = torch.Generator().manual_seed(seed)
    A = torch.randn(3, 3, generator=g, dtype=torch.float64)
    Q, Rr = torch.linalg.qr(A)
    Q = Q * torch.sign(torch.diagonal(Rr)).unsqueeze(0)
    if torch.linalg.det(Q) < 0:
        Q[:, 0] = -Q[:, 0]
    return Q.to(dtype)


def improper_elements(dtype=torch.float64) -> Dict[str, torch.Tensor]:
    """Named improper O(3) elements (det = -1)."""
    rot = random_rotation(123, torch.float64)
    return {
        "inversion": -torch.eye(3, dtype=torch.float64).to(dtype),
        "mirror_x": torch.diag(torch.tensor([-1.0, 1.0, 1.0], dtype=torch.float64)).to(dtype),
        "mirror_z_then_rotation": (rot @ torch.diag(
            torch.tensor([1.0, 1.0, -1.0], dtype=torch.float64))).to(dtype),
        "rotoinversion": (-rot).to(dtype),
    }


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def predict(model: AtomTrick, batch: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, torch.Tensor]:
    """(E per molecule, F per atom), detached."""
    out = model(batch)
    return out["E"].detach(), out["F"].detach()


def max_rel_err(a: torch.Tensor, b: torch.Tensor, floor: float = 1e-12) -> float:
    """max|a-b| / max|b| (``floor`` guards an all-zero reference)."""
    return float((a - b).abs().max() / b.abs().max().clamp_min(floor))


def symmetry_errors(model: AtomTrick, batch: Dict[str, torch.Tensor],
                    g: torch.Tensor) -> Tuple[float, float]:
    """Relative (energy, force) violation of ``E(gR)=E(R)``, ``F(gR)=g F(R)``."""
    E, F = predict(model, batch)
    g = g.to(batch["R"].dtype)
    E2, F2 = predict(model, with_positions(batch, batch["R"] @ g.T))
    return max_rel_err(E2, E), max_rel_err(F2, F @ g.T)


# ---------------------------------------------------------------------------
# train.py subprocess helper
# ---------------------------------------------------------------------------

def run_train(args: Iterable[str], *, cwd: Optional[str] = None,
              env: Optional[Dict[str, str]] = None,
              timeout: int = 600) -> subprocess.CompletedProcess:
    """Run ``python <repo>/train.py <args>`` on CPU and capture its output.

    ``CUDA_VISIBLE_DEVICES`` is blanked so ``--dry_run`` resolves exactly as it
    would on a GPU-less machine; stale ``ATOMTRICK(S)_*`` variables from the
    caller's shell are removed unless explicitly passed in ``env``.
    """
    full_env = {k: v for k, v in os.environ.items()
                if not k.startswith(("ATOMTRICK_", "ATOMTRICKS_"))}
    full_env["CUDA_VISIBLE_DEVICES"] = ""
    full_env.setdefault("WANDB_MODE", "disabled")
    if env:
        full_env.update(env)
    cmd = [sys.executable, os.path.join(REPO_ROOT, "train.py"), *map(str, args)]
    return subprocess.run(cmd, cwd=cwd or REPO_ROOT, env=full_env, timeout=timeout,
                          stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def parse_param_count(output: str) -> int:
    """Extract N from the ``Model parameters: N`` line (``3,730,368`` or ``3730368``)."""
    import re
    m = re.findall(r"Model parameters:\s*([\d,_]+)", output)
    if not m:
        raise AssertionError("no 'Model parameters: N' line in train.py output:\n" + output[-4000:])
    return int(m[-1].replace(",", "").replace("_", ""))
