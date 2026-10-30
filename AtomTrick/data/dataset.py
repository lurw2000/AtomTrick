import logging

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from scipy.spatial import cKDTree
from ase.neighborlist import primitive_neighbor_list

from ase import units

from ..constants import ATOMIC_REFERENCE_ENERGIES

logger = logging.getLogger(__name__)


def fit_atomic_reference_energies(path):
    """Fit per-element reference energies by least squares on raw totals.

    Solves ``E_total ≈ Σ_z N_z · E_ref[z]`` on the dataset at ``path`` in
    eV (kcal/mol → eV conversion matches ``MLFFDataset``).  Returns a dict
    ``{int(Z): float(E_ref_eV)}`` for elements that appear in the data.
    The whole fit runs in float64.

    Use this at setup time (e.g. in ``train.py``) to replace the hardcoded
    DFT isolated-atom energies with data-driven refs, then pass the dict to
    every ``MLFFDataset`` (train/val/test) and to ``AtomTrick`` so the
    subtraction and re-addition stay in the same reference frame.
    """
    with np.load(path, allow_pickle=True) as data:
        N = np.asarray(data["N"], dtype=np.int64)          # (n_mol,)
        Z = np.asarray(data["Z"], dtype=np.int64)          # (total_atoms,)
        E_kcal = np.asarray(data["E"], dtype=np.float64)   # (n_mol,)
    E = E_kcal * float(units.kcal / units.mol)             # eV
    n_mol = len(N)
    max_z = int(Z.max()) + 1
    # Composition matrix via scatter-add of one-hot Z, grouped by molecule.
    one_hot = np.eye(max_z, dtype=np.float64)[Z]           # (total_atoms, max_z)
    mol_idx = np.repeat(np.arange(n_mol), N)               # (total_atoms,)
    comp = np.zeros((n_mol, max_z), dtype=np.float64)
    np.add.at(comp, mol_idx, one_hot)
    present_mask = comp.sum(0) > 0
    C = comp[:, present_mask]
    b, *_ = np.linalg.lstsq(C, E, rcond=None)
    zs = np.flatnonzero(present_mask)
    return {int(z): float(e) for z, e in zip(zs, b)}


class MLFFDataset(Dataset):
    def __init__(self, path, with_targets=True, edge_cutoff=None,
                 dtype=torch.float32, atomic_ref=None, max_neighbors=None):
        """
        Parameters
        ----------
        atomic_ref : dict[int, float] | None
            Per-element reference energies (eV) to subtract from raw totals.
            ``None`` ⇒ fall back to ``ATOMIC_REFERENCE_ENERGIES`` from
            ``constants.py`` (hardcoded DFT isolated-atom values).  Pass a
            dict from ``fit_atomic_reference_energies`` to use data-driven
            refs instead.

        The kcal/mol → eV conversion of E and F and the atomic-reference
        subtraction are done in float64; only the resulting residual labels
        are cast to ``dtype``.  (Casting the raw total energies, |E| ~ 1e5 --
        1e6 kcal/mol, to float32 first would quantise them by ~1-10 meV.)
        """

        super().__init__()

        if with_targets:
            self.keys = ["N", "Z", "R", "F", "E"]
        else:
            self.keys = ["N", "Z", "R"]

        self._load_npz(path, self.keys)

        self.dtype = dtype
        self.N = torch.tensor(self.N, dtype=torch.long)                      # shape: (num_molecules,)
        self.offsets = torch.cat([
            torch.zeros(1, dtype=torch.long),
            torch.cumsum(self.N, dim=0)
        ])                                                                    # shape: (num_molecules + 1,)
        self.Z = torch.tensor(self.Z, dtype=torch.long)                      # shape: (total_atoms,)
        self.R = torch.tensor(self.R, dtype=dtype).reshape(-1, 3)            # shape: (total_atoms, 3)

        logger.debug("N: %s", self.N.shape)
        logger.debug("Z: %s", self.Z.shape)
        logger.debug("R: %s", self.R.shape)

        self.with_targets = with_targets
        self.edge_cutoff = edge_cutoff
        # Cap neighbours kept per central atom (nearest-K by distance).  None/0 =
        # keep all within edge_cutoff (original behaviour).  Bounds graph degree so
        # a large cutoff does not blow up edges/activation-memory — the force
        # double-backward memory scales with edge count (cf. OCP/EquiformerV2
        # max_neighbors, which is why they stay light at cutoff 12).
        self.max_neighbors = max_neighbors if max_neighbors else None

        if self.with_targets:
            # Label arithmetic in float64; the residuals are cast to ``dtype`` at the end.
            label_dtype = torch.float64
            self.F = torch.tensor(self.F, dtype=label_dtype).reshape(-1, 3)  # shape: (total_atoms, 3)
            self.E = torch.tensor(self.E, dtype=label_dtype)                  # shape: (num_molecules,)

            # Convert forces and energies from kcal/mol to eV
            self.F = self.F * units.kcal / units.mol  # kcal/mol/Å -> eV/Å
            self.E = self.E * units.kcal / units.mol  # kcal/mol   -> eV

            # Subtract atomic reference energies so the model learns
            # interaction (formation) energies instead of total energies.
            # Forces are unaffected since reference energies are constants.
            ref_source = atomic_ref if atomic_ref is not None else ATOMIC_REFERENCE_ENERGIES
            # JSON round-trips dict keys as strings; accept either int or str.
            ref_source = {int(k): float(v) for k, v in ref_source.items()}
            unique_z = set(self.Z.tolist())
            missing = unique_z - set(ref_source.keys())
            if missing:
                raise ValueError(
                    f"No atomic reference energy for element(s) Z={missing}. "
                    f"Pass them via atomic_ref= or add to ATOMIC_REFERENCE_ENERGIES."
                )
            max_z_ref = max(ref_source.keys()) + 1
            ref_table = torch.zeros(max_z_ref, dtype=label_dtype)
            for z, e in ref_source.items():
                ref_table[z] = e
            atom_ref = ref_table[self.Z]                          # (total_atoms,)
            mol_idx = torch.repeat_interleave(
                torch.arange(len(self.N)), self.N
            )
            E_ref = torch.zeros(len(self.N), dtype=label_dtype)
            E_ref.scatter_add_(0, mol_idx, atom_ref)
            self.E = self.E - E_ref

            # Only the (small) residual labels are rounded to the model dtype.
            self.F = self.F.to(dtype)
            self.E = self.E.to(dtype)

            logger.debug("F: %s", self.F.shape)
            logger.debug("E: %s", self.E.shape)
            logger.debug("Subtracted atomic reference energies (source=%s, elements: %s, "
                         "label arithmetic: %s)",
                         "fitted" if atomic_ref is not None else "hardcoded",
                         sorted(ref_source.keys()), label_dtype)

        # ---- Periodic mode (OC20-style) -------------------------------------
        # If the NPZ carries per-frame ``cell`` (n_mol,3,3) + ``pbc`` (n_mol,3),
        # neighbours are periodic and are built ON THE FLY in __getitem__:
        # storing all edges for 200k frames would be ~18 GB, and the model needs
        # per-edge periodic shift vectors (edge_shift = S @ cell) so that
        # diff = R[dst] - R[src] + edge_shift.  Non-periodic datasets omit
        # ``cell`` and fall through to the edge_index/cutoff precompute below,
        # unchanged.
        self.periodic = False
        self._lazy_edges = False
        try:
            self._load_npz(path, ["cell", "pbc"])
            self.cell = torch.tensor(np.asarray(self.cell), dtype=dtype).reshape(-1, 3, 3)
            self.pbc = torch.tensor(np.asarray(self.pbc, dtype=bool)).reshape(-1, 3)
            self.periodic = True
            if edge_cutoff is None:
                raise ValueError(
                    "Periodic dataset (cell present) requires edge_cutoff for "
                    "on-the-fly periodic neighbour construction.")
            logger.debug("Periodic mode: cell %s pbc %s (on-the-fly neighbours, cutoff=%s)",
                         self.cell.shape, self.pbc.shape, edge_cutoff)
        except KeyError:
            pass

        if not self.periodic:
            # Try to load edge connectivity from file (edge_index + num_edges,
            # analogous to Z + N)
            try:
                self._load_npz(path, ["edge_index", "num_edges"])
                self.edge_index = torch.tensor(self.edge_index, dtype=torch.long)   # (total_edges, 2), local
                num_edges = torch.tensor(self.num_edges, dtype=torch.long)           # (num_molecules,)
                self.edge_offsets = torch.cat([
                    torch.zeros(1, dtype=torch.long),
                    torch.cumsum(num_edges, dim=0)
                ])                                                                    # (num_molecules + 1,)
                logger.debug("edge_index: %s  (loaded from file)", self.edge_index.shape)
            except KeyError:
                if edge_cutoff is None:
                    raise ValueError(
                        "No edge_index/num_edges found in the dataset and edge_cutoff is not set. "
                        "Provide edge connectivity in the npz file or pass a float edge_cutoff."
                    )
                # Compute non-periodic edges LAZILY per-item in __getitem__ (parallelised
                # across DataLoader workers) instead of precomputing here: for a large
                # dataset (e.g. SPICE ~1M conformers) an upfront single-threaded loop would
                # be a multi-minute __init__ and hold a ~GB-scale edge_index in RAM.
                self._lazy_edges = True
                logger.debug("non-periodic edges: lazy per-item (cutoff=%s)", edge_cutoff)

    def _load_npz(self, path, keys):
        """Load the keys from the file and set as attributes.

        Parameters
        ----------
        path: str
            Absolute path of the dataset (in npz-format).
        keys: list
            Contains keys in the dataset to load and set as attributes.

        Returns
        -------
        None
        """
        with np.load(path, allow_pickle=True) as data:
            for key in keys:
                if key not in data:
                    raise KeyError(f"Can not find key {key} in the dataset.")
                setattr(self, key, data[key])
    
 
    @staticmethod
    def _knn_mask(center, dist, k):
        """Boolean keep-mask selecting, for each ``center`` atom, its ``k`` nearest
        edges by ``dist``.  Atoms with fewer than ``k`` neighbours keep all of them.
        Vectorised: sort by (center, dist), then rank within each center group."""
        n = len(center)
        if n == 0:
            return np.ones(0, dtype=bool)
        order = np.lexsort((dist, center))        # primary center, secondary dist asc
        c_sorted = center[order]
        change = np.empty(n, dtype=bool)
        change[0] = True
        change[1:] = c_sorted[1:] != c_sorted[:-1]
        group_start = np.flatnonzero(change)                  # start of each group
        sizes = np.diff(np.append(group_start, n))
        starts_per_elem = np.repeat(group_start, sizes)
        within = np.arange(n) - starts_per_elem               # 0-based rank within group
        keep_sorted = within < k
        mask = np.zeros(n, dtype=bool)
        mask[order[keep_sorted]] = True
        return mask

    def _compute_edges(self, R):
        """Compute edges for a single molecule by distance cutoff.

        Parameters
        ----------
        R : Tensor, shape (n_atoms, 3)

        Returns
        -------
        edge_index : Tensor, shape (2, num_edges)
            Local atom indices, no self-loops.
        """
        tree = cKDTree(R.numpy())
        pairs = tree.query_pairs(self.edge_cutoff, output_type="ndarray")  # shape: (num_edges, 2), upper triangle
        # Make directed (both directions)
        pairs = np.concatenate([pairs, pairs[:, ::-1]], axis=0)
        if self.max_neighbors is not None and len(pairs) > 0:
            Rn = R.numpy()
            d = Rn[pairs[:, 1]] - Rn[pairs[:, 0]]
            dist = np.sqrt((d * d).sum(axis=1))
            keep = self._knn_mask(pairs[:, 0], dist, self.max_neighbors)   # per central atom (col 0)
            pairs = pairs[keep]
        return torch.from_numpy(pairs).t().contiguous()   # shape: (2, num_edges)

    def _compute_periodic_edges(self, R, cell, pbc):
        """Periodic neighbour list for one frame via ASE.

        Returns
        -------
        edge_index : LongTensor (2, num_edges)   local atom indices (src, dst)
        edge_shift : Tensor (num_edges, 3)        cartesian periodic shift S @ cell
            so that  diff = R[dst] - R[src] + edge_shift  is the true
            minimum-image (periodic) displacement for each directed edge.
        """
        pos = R.numpy().astype(np.float64)
        cell_np = cell.numpy().astype(np.float64)
        pbc_np = pbc.numpy().astype(bool)
        i, j, S = primitive_neighbor_list(
            "ijS", pbc_np, cell_np, pos, float(self.edge_cutoff),
            self_interaction=False)
        shift = S.astype(np.float64) @ cell_np                            # (E, 3)
        if self.max_neighbors is not None and len(i) > 0:
            # Periodic i->j displacement: d = pos[j] + shift - pos[i]
            # (matches model's diff = R[dst] - R[src] + edge_shift).  Keep each
            # central atom i's nearest max_neighbors edges.
            d = pos[j] + shift - pos[i]
            dist = np.sqrt((d * d).sum(axis=1))
            keep = self._knn_mask(i, dist, self.max_neighbors)
            i, j, shift = i[keep], j[keep], shift[keep]
        edge_index = torch.from_numpy(np.stack([i, j], axis=0)).long()   # (2, E)
        edge_shift = torch.from_numpy(shift).to(self.dtype)
        return edge_index, edge_shift

    def __getitem__(self, idx):
        start = self.offsets[idx].item()
        end = self.offsets[idx + 1].item()

        data = {}
        data["N"] = self.N[idx]
        data["Z"] = self.Z[start:end]   # shape: (n_atoms,)
        data["R"] = self.R[start:end]   # shape: (n_atoms, 3)

        if self.with_targets:
            data["F"] = self.F[start:end]  # shape: (n_atoms, 3)
            data["E"] = self.E[idx]        # shape: scalar

        if self.periodic:
            edge_index, edge_shift = self._compute_periodic_edges(
                data["R"], self.cell[idx], self.pbc[idx])
            data["edge_index"] = edge_index                 # (2, num_edges) local
            data["edge_shift"] = edge_shift                 # (num_edges, 3)
        else:
            if self._lazy_edges:
                data["edge_index"] = self._compute_edges(data["R"])   # (2, num_edges), on the fly
            else:
                e_start = self.edge_offsets[idx].item()
                e_end = self.edge_offsets[idx + 1].item()
                data["edge_index"] = self.edge_index[e_start:e_end].t().contiguous()  # (2, num_edges)

        return data

    def __len__(self):
        return len(self.N)


def collate_fn(batch):
    """Collate variable-size molecules into a flat batch.

    Atom-level tensors (Z, R, F) are concatenated along dim 0.
    A `batch` index tensor maps each atom to its molecule index.
    Molecule-level tensors (N, E) are stacked normally.
    """
    result = {}
    result["N"] = torch.stack([d["N"] for d in batch])           # (batch_size,)
    result["batch"] = torch.repeat_interleave(
        torch.arange(len(batch)), result["N"]
    )                                                              # (total_atoms,)
    result["Z"] = torch.cat([d["Z"] for d in batch])             # (total_atoms,)
    result["R"] = torch.cat([d["R"] for d in batch])             # (total_atoms, 3)

    if "E" in batch[0]:
        result["E"] = torch.stack([d["E"] for d in batch])       # (batch_size,)
        result["F"] = torch.cat([d["F"] for d in batch])         # (total_atoms, 3)

    # Shift local edge indices to global by adding per-molecule atom offsets
    cum_N = torch.cat([torch.zeros(1, dtype=torch.long), result["N"].cumsum(0)[:-1]])
    edge_indices = [d["edge_index"] + off for d, off in zip(batch, cum_N)]
    result["edge_index"] = torch.cat(edge_indices, dim=1)        # shape: (2, total_edges)

    # Periodic per-edge cartesian shift (translation-invariant displacement, so
    # concatenated with NO offset).  Absent for non-periodic datasets.
    if "edge_shift" in batch[0]:
        result["edge_shift"] = torch.cat([d["edge_shift"] for d in batch], dim=0)  # (total_edges, 3)

    return result


def make_padded_collate_fn(max_edges):
    """Return a collate function that pads ``edge_index`` to *max_edges*.

    Padding edges are self-loops to node 0 (src=dst=0).  An ``edge_mask``
    boolean tensor distinguishes real edges (True) from padding (False).
    This makes every batch have identical tensor shapes — a prerequisite
    for CUDA graph capture.
    """
    def _collate(batch):
        result = collate_fn(batch)       # standard collation first
        real_ei = result["edge_index"]   # (2, num_real_edges)
        num_real = real_ei.shape[1]

        if num_real < max_edges:
            pad = torch.zeros(2, max_edges - num_real, dtype=torch.long)
            result["edge_index"] = torch.cat([real_ei, pad], dim=1)
            if "edge_shift" in result:
                spad = torch.zeros(max_edges - num_real, 3, dtype=result["edge_shift"].dtype)
                result["edge_shift"] = torch.cat([result["edge_shift"], spad], dim=0)
        elif num_real > max_edges:
            result["edge_index"] = real_ei[:, :max_edges]
            if "edge_shift" in result:
                result["edge_shift"] = result["edge_shift"][:max_edges]
            num_real = max_edges

        mask = torch.zeros(max_edges, dtype=torch.bool)
        mask[:num_real] = True
        result["edge_mask"] = mask
        return result
    return _collate


def compute_max_edges(dataset, batch_size):
    """Scan the dataset to find the maximum total edge count per batch.

    Uses a quick linear scan — no GPU required.
    """
    from torch.utils.data import DataLoader
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        drop_last=True, collate_fn=collate_fn, num_workers=0)
    max_e = 0
    for batch in loader:
        max_e = max(max_e, batch["edge_index"].shape[1])
    return max_e


def build_dataloader(dataset, config, *, train: bool, max_edges=None,
                     distributed=False, rank=0, world_size=1):
    cfn = make_padded_collate_fn(max_edges) if max_edges else collate_fn
    if distributed and train:
        # DistributedSampler gives each rank a disjoint, shuffled shard.  It
        # owns shuffling/drop_last, so the DataLoader must NOT also set them.
        # fit() calls sampler.set_epoch(epoch) so reshuffles differ per epoch.
        from torch.utils.data.distributed import DistributedSampler
        sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank,
                                     shuffle=True, drop_last=True)
        return DataLoader(
            dataset,
            sampler=sampler,
            shuffle=False,
            drop_last=False,
            collate_fn=cfn,
            pin_memory=True,
            **config,
        )
    return DataLoader(
        dataset,
        shuffle=train,
        drop_last=train,
        collate_fn=cfn,
        pin_memory=True,
        **config,
    )
