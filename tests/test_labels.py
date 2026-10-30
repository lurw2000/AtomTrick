"""Energy/force label arithmetic in ``MLFFDataset``.

The NPZ files store total energies in kcal/mol (float64).  The loader converts
to eV and subtracts per-element atomic reference energies, so the model trains
on small interaction energies.  Total energies are large (aspirin is about
-4.07e5 kcal/mol = -1.76e4 eV), where the float32 spacing is ~2 meV.

Conversion and subtraction run in float64 and only the small residual is cast
to float32 -> exact to float32 round-off of the residual (~1e-7 eV).  (Casting
to float32 FIRST would give per-frame label errors of order 1 meV.)
"""
import numpy as np
import pytest
import torch
from ase import units

from AtomTrick.constants import ATOMIC_REFERENCE_ENERGIES
from AtomTrick.data import MLFFDataset

KCAL_TO_EV = units.kcal / units.mol
# aspirin composition C9 H8 O4
ASPIRIN_Z = np.array([6] * 9 + [1] * 8 + [8] * 4, dtype=np.int64)
N_FRAMES = 64


@pytest.fixture(scope="module")
def npz_path(tmp_path_factory):
    rng = np.random.default_rng(0)
    n = len(ASPIRIN_Z)
    e_ref = sum(ATOMIC_REFERENCE_ENERGIES[int(z)] for z in ASPIRIN_Z)       # eV
    e_int = rng.uniform(-3.0, 3.0, size=N_FRAMES)                           # eV
    E_kcal = (e_ref + e_int) / KCAL_TO_EV                                    # float64, ~-4.07e5
    R = rng.uniform(0.0, 4.0, size=(N_FRAMES * n, 3))
    F_kcal = rng.uniform(-50.0, 50.0, size=(N_FRAMES * n, 3))
    path = tmp_path_factory.mktemp("labels") / "aspirin_like.npz"
    np.savez(path, N=np.full(N_FRAMES, n, dtype=np.int64), Z=np.tile(ASPIRIN_Z, N_FRAMES),
             R=R, F=F_kcal, E=E_kcal)
    return str(path)


def _truth(path):
    """Labels computed entirely in float64 (numpy)."""
    d = np.load(path)
    N, Z = d["N"], d["Z"]
    mol = np.repeat(np.arange(len(N)), N)
    e_ref = np.zeros(len(N))
    np.add.at(e_ref, mol, np.array([ATOMIC_REFERENCE_ENERGIES[int(z)] for z in Z]))
    return d["E"] * KCAL_TO_EV - e_ref, d["F"] * KCAL_TO_EV


def test_float64_label_path_is_exact(npz_path):
    ds = MLFFDataset(npz_path, edge_cutoff=5.0)
    E_true, F_true = _truth(npz_path)
    assert ds.E.dtype == torch.float32 and ds.F.dtype == torch.float32
    e_err = np.abs(ds.E.double().numpy() - E_true).max()
    assert e_err < 1e-5, f"float64 path: max energy label error {e_err * 1e3:.4f} meV"
    f_err = np.abs(ds.F.double().numpy().reshape(-1, 3) - F_true).max()
    assert f_err < 1e-5 * np.abs(F_true).max()


def test_build_loaders_use_the_same_labels(npz_path):
    """``Trainer.build_loaders`` (used by train.py) builds the same labels as MLFFDataset."""
    from AtomTrick.trainer import Trainer
    cfg = {"data": npz_path, "val_data": npz_path, "edge_cutoff": 5.0, "batch_size": 4,
           "eval_batch_size": 4, "num_workers": 0}
    train_loader, val_loader, _ = Trainer.build_loaders(cfg)
    ref = MLFFDataset(npz_path, edge_cutoff=5.0)
    for loader in (train_loader, val_loader):
        assert torch.equal(loader.dataset.E, ref.E)
