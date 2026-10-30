"""``MLFFDataset`` / ``collate_fn`` / ``build_dataloader`` on synthetic NPZ files.

(Label arithmetic -- kcal/mol -> eV and reference subtraction in float64 -- is
covered in ``test_labels.py``.)
"""
import numpy as np
import pytest
import torch
from ase import units

from AtomTrick.constants import ATOMIC_REFERENCE_ENERGIES
from AtomTrick.data import MLFFDataset, build_dataloader, collate_fn

VALID_Z = sorted(ATOMIC_REFERENCE_ENERGIES)           # [1, 6, 7, 8]


def make_molecules(n_mol=5, seed=0):
    rng = np.random.default_rng(seed)
    mols = []
    for _ in range(n_mol):
        n = int(rng.integers(3, 8))
        mols.append({
            "n": n,
            "Z": rng.choice(VALID_Z, size=n).astype(np.int64),
            "R": rng.uniform(-3.0, 3.0, size=(n, 3)).astype(np.float32),
            "F": rng.uniform(-1.0, 1.0, size=(n, 3)).astype(np.float32),
            "E": float(rng.uniform(-100.0, 0.0)),
        })
    return mols


def save_npz(path, mols, include_edges=False):
    arrays = dict(
        N=np.array([m["n"] for m in mols], dtype=np.int64),
        Z=np.concatenate([m["Z"] for m in mols]),
        R=np.concatenate([m["R"] for m in mols]),
        F=np.concatenate([m["F"] for m in mols]),
        E=np.array([m["E"] for m in mols], dtype=np.float64),
    )
    if include_edges:
        from scipy.spatial import cKDTree
        edges, counts = [], []
        for m in mols:
            pairs = cKDTree(m["R"]).query_pairs(5.0, output_type="ndarray")
            pairs = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
            edges.append(pairs.astype(np.int64))
            counts.append(len(pairs))
        arrays["edge_index"] = np.concatenate(edges, axis=0)
        arrays["num_edges"] = np.array(counts, dtype=np.int64)
    np.savez(path, **arrays)
    return str(path)


@pytest.fixture(scope="module")
def mols():
    return make_molecules()


@pytest.fixture(scope="module")
def plain_npz(tmp_path_factory, mols):
    return save_npz(tmp_path_factory.mktemp("data") / "plain.npz", mols)


@pytest.fixture(scope="module")
def edges_npz(tmp_path_factory, mols):
    return save_npz(tmp_path_factory.mktemp("data") / "edges.npz", mols, include_edges=True)


def test_basic_load_builds_edges_lazily(plain_npz, mols):
    ds = MLFFDataset(plain_npz, edge_cutoff=5.0)
    total = sum(m["n"] for m in mols)
    assert ds.N.shape == (5,) and ds.E.shape == (5,)
    assert ds.Z.shape == (total,) and ds.R.shape == (total, 3) and ds.F.shape == (total, 3)
    assert ds._lazy_edges
    assert ds[0]["edge_index"].shape[0] == 2
    assert len(ds) == 5


def test_load_without_targets(plain_npz):
    ds = MLFFDataset(plain_npz, with_targets=False, edge_cutoff=5.0)
    assert not ds.with_targets
    sample = ds[0]
    assert "E" not in sample and "F" not in sample


def test_precomputed_edges(edges_npz):
    ds = MLFFDataset(edges_npz)                        # no cutoff needed
    assert ds.edge_index.shape[1] == 2
    assert ds.edge_offsets.shape == (6,)


def test_missing_edges_and_cutoff_raises(plain_npz):
    with pytest.raises(ValueError):
        MLFFDataset(plain_npz)


def test_getitem_shapes_and_local_indices(plain_npz, mols):
    ds = MLFFDataset(plain_npz, edge_cutoff=5.0)
    for i, m in enumerate(mols):
        s = ds[i]
        n = m["n"]
        assert s["N"].item() == n
        assert s["Z"].shape == (n,) and s["R"].shape == (n, 3) and s["F"].shape == (n, 3)
        assert s["E"].ndim == 0
        assert s["edge_index"].shape[0] == 2 and s["edge_index"].max().item() < n


def test_lazy_edges_match_a_brute_force_radius_graph(plain_npz, mols):
    ds = MLFFDataset(plain_npz, edge_cutoff=5.0)
    for i, m in enumerate(mols):
        ei = ds[i]["edge_index"]
        got = {(int(a), int(b)) for a, b in ei.t().tolist()}
        d = np.linalg.norm(m["R"][:, None] - m["R"][None], axis=-1)
        want = {(a, b) for a in range(m["n"]) for b in range(m["n"]) if a != b and d[a, b] < 5.0}
        assert got == want


def test_collate_offsets_edges(plain_npz, mols):
    ds = MLFFDataset(plain_npz, edge_cutoff=5.0)
    batch = collate_fn([ds[0], ds[1], ds[2]])
    n = sum(m["n"] for m in mols[:3])
    assert batch["N"].shape == (3,) and batch["E"].shape == (3,)
    assert batch["Z"].shape == (n,) and batch["R"].shape == (n, 3) and batch["F"].shape == (n, 3)
    assert batch["batch"].shape == (n,)
    assert batch["batch"].min().item() == 0 and batch["batch"].max().item() == 2
    assert batch["edge_index"].shape[0] == 2 and batch["edge_index"].max().item() < n
    # every edge stays inside its molecule
    src, dst = batch["edge_index"]
    assert torch.equal(batch["batch"][src], batch["batch"][dst])


def test_dataloader_dtypes(plain_npz):
    ds = MLFFDataset(plain_npz, edge_cutoff=5.0)
    loader = build_dataloader(ds, {"batch_size": 2, "num_workers": 0}, train=False)
    batch = next(iter(loader))
    assert {"N", "Z", "R", "F", "E", "batch", "edge_index"} <= set(batch)
    assert batch["N"].dtype == torch.long and batch["Z"].dtype == torch.long
    assert batch["R"].dtype == torch.float32 and batch["edge_index"].dtype == torch.long


def test_eval_loader_covers_every_structure_once(plain_npz):
    ds = MLFFDataset(plain_npz, edge_cutoff=5.0)
    loader = build_dataloader(ds, {"batch_size": 2, "num_workers": 0}, train=False)
    assert sum(int(b["N"].shape[0]) for b in loader) == len(ds)   # runt batch kept


def test_energy_unit_conversion_and_reference_subtraction(plain_npz):
    ds = MLFFDataset(plain_npz, edge_cutoff=5.0)
    raw = np.load(plain_npz)
    offsets = np.concatenate([[0], np.cumsum(raw["N"])])
    e_ref = np.array([sum(ATOMIC_REFERENCE_ENERGIES[int(z)] for z in raw["Z"][a:b])
                      for a, b in zip(offsets[:-1], offsets[1:])])
    expected = raw["E"] * (units.kcal / units.mol) - e_ref
    # float32 storage of the (here large) residual: relative precision ~6e-8
    assert np.allclose(ds.E.double().numpy(), expected, rtol=1e-6, atol=1e-6)
    assert np.allclose(ds.F.double().numpy(), raw["F"] * (units.kcal / units.mol),
                       rtol=1e-6, atol=1e-7)
