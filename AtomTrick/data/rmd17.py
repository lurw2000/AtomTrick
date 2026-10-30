"""Processor for the revised MD17 (rMD17) dataset.

rMD17 layout (as downloaded from Figshare,
https://doi.org/10.6084/m9.figshare.12672038):

    {raw_data_dir}/
        npz_data/
            rmd17_aspirin.npz      (energies, forces, coords, nuclear_charges)
            rmd17_azobenzene.npz
            ...
        splits/
            index_train_01.csv
            index_test_01.csv
            index_train_02.csv
            ...

Each split CSV holds frame indices (one per line); the 5 splits are the
official 950/1000 train/test partitions used by most benchmarks.  We carve a
small validation slice out of the training split deterministically.

Produces three NPZ files per molecule::

    {molecule}_train_default.npz
    {molecule}_val_default.npz
    {molecule}_test_default.npz
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd

from .base import RawDatasetProcessor

logger = logging.getLogger(__name__)


class RMD17Processor(RawDatasetProcessor):
    """Process the revised MD17 dataset using official train/test split indices.

    Parameters
    ----------
    raw_data_dir : str or Path
        Root of the rmd17 download (contains ``npz_data/`` and ``splits/``).
    output_dir : str or Path
        Where to write the processed NPZs.
    seed : int, default 0
        Used only to shuffle the training split before the val slice.
    split_id : int, default 1
        Which of the 5 official splits (1..5) to use.
    val_fraction : float, default 0.05
        Fraction of the training split carved off as validation (default: 50/1000).
    """

    MOLECULES = [
        "aspirin", "azobenzene", "benzene", "ethanol", "malonaldehyde",
        "naphthalene", "paracetamol", "salicylic", "toluene", "uracil",
    ]

    def __init__(
        self,
        raw_data_dir,
        output_dir,
        seed: int = 0,
        split_id: int = 1,
        val_fraction: float = 0.05,
    ):
        super().__init__(raw_data_dir, output_dir, seed)
        if not (1 <= split_id <= 5):
            raise ValueError("split_id must be in [1, 5]")
        if not (0.0 < val_fraction < 1.0):
            raise ValueError("val_fraction must be in (0, 1)")
        self.split_id = split_id
        self.val_fraction = val_fraction

    def _load_raw(self, molecule_name: str):
        raw_path = self.raw_data_dir / "npz_data" / f"rmd17_{molecule_name}.npz"
        if not raw_path.exists():
            raise FileNotFoundError(f"rmd17 raw file not found: {raw_path}")
        return np.load(raw_path)

    def _read_split_indices(self, split_name: str) -> np.ndarray:
        csv_path = self.raw_data_dir / "splits" / f"index_{split_name}_{self.split_id:02d}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(f"rmd17 split file not found: {csv_path}")
        return pd.read_csv(csv_path, header=None).to_numpy().flatten()

    def process_molecule(self, molecule_name: str, **_) -> dict:
        molecule = self._load_raw(molecule_name)
        train_idx = self._read_split_indices("train")
        test_idx = self._read_split_indices("test")

        # Deterministic val carve-out from the shuffled training indices.
        # Use legacy RandomState (MT19937) to match process_rmd17_default.py
        # bit-for-bit — re-processing reproduces existing _default NPZs.
        rng = np.random.RandomState(self.seed)
        rng.shuffle(train_idx)
        n_total = len(train_idx)
        n_val = max(1, int(round(n_total * self.val_fraction)))
        val_idx = train_idx[-n_val:]
        train_idx = train_idx[:-n_val]

        E_full = molecule["energies"]
        F_full = molecule["forces"]
        R_full = molecule["coords"]
        Z_template = molecule["nuclear_charges"]

        splits = {
            "train": self._stack_frames(E_full, F_full, R_full, Z_template, train_idx),
            "val":   self._stack_frames(E_full, F_full, R_full, Z_template, val_idx),
            "test":  self._stack_frames(E_full, F_full, R_full, Z_template, test_idx),
        }

        out_paths = {}
        for name, s in splits.items():
            p = self.output_dir / f"{molecule_name}_{name}_default.npz"
            self._save_split(p, **s)
            out_paths[name] = p
            self._log_split(name, molecule_name, s)

        return {
            "train": out_paths["train"],
            "val":   out_paths["val"],
            "test":  out_paths["test"],
            "n_train": splits["train"]["E"].shape[0],
            "n_val":   splits["val"]["E"].shape[0],
            "n_test":  splits["test"]["E"].shape[0],
            "n_atoms": int(splits["train"]["N"][0]),
        }
