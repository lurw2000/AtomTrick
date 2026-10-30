"""Processor for the MD22 dataset.

MD22 layout (as downloaded from http://www.sgdml.org/):

    {raw_data_dir}/
        md22_Ac-Ala3-NHMe.npz    (E, F, R, z)
        md22_DHA.npz
        md22_stachyose.npz
        md22_AT-AT.npz
        md22_AT-AT-CG-CG.npz
        md22_buckyball-catcher.npz
        md22_double-walled_nanotube.npz

Unlike rMD17, MD22 has no official train/test CSVs: we create random splits
with a fixed seed and per-molecule sizes (taken from the GemNet-OC paper's
convention, which is the standard for MD22 benchmarking).

Produces three NPZ files per molecule::

    {molecule}_train_default.npz
    {molecule}_val_default.npz
    {molecule}_test_default.npz
"""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from .base import RawDatasetProcessor

logger = logging.getLogger(__name__)


# Standard MD22 training-set sizes (per GemNet-OC / the sGDML paper).
# Remaining frames go to test.
DEFAULT_SIZES = {
    "Ac-Ala3-NHMe":            (6000, 500),
    "DHA":                     (8000, 500),
    "stachyose":               (8000, 500),
    "AT-AT":                   (3000, 500),
    "AT-AT-CG-CG":             (2000, 500),
    "buckyball-catcher":       (600, 100),
    "double-walled_nanotube":  (800, 100),
}


class MD22Processor(RawDatasetProcessor):
    """Process the MD22 dataset with random splits at fixed seed.

    Parameters
    ----------
    raw_data_dir : str or Path
        Directory containing ``md22_{molecule}.npz`` files.
    output_dir : str or Path
        Where to write the processed NPZs.
    seed : int, default 1
        RNG seed for the train/val/test shuffle.  (The rmd17 default is 0;
        MD22 defaults to 1 to match the sizes table exactly — change only
        if you want a different split realisation.)
    sizes : dict[str, tuple[int, int]], optional
        Override of ``{molecule: (train_size, val_size)}``.  Defaults to
        :data:`DEFAULT_SIZES` above.
    """

    MOLECULES = list(DEFAULT_SIZES.keys())

    def __init__(
        self,
        raw_data_dir,
        output_dir,
        seed: int = 1,
        sizes=None,
    ):
        super().__init__(raw_data_dir, output_dir, seed)
        self.sizes = dict(DEFAULT_SIZES if sizes is None else sizes)

    def _load_raw(self, molecule_name: str):
        raw_path = self.raw_data_dir / f"md22_{molecule_name}.npz"
        if not raw_path.exists():
            raise FileNotFoundError(f"MD22 raw file not found: {raw_path}")
        return np.load(raw_path)

    def process_molecule(
        self,
        molecule_name: str,
        *,
        train_size: int = None,
        val_size: int = None,
    ) -> dict:
        """Build splits for one molecule.

        If ``train_size`` / ``val_size`` are omitted, uses the entry from
        :data:`DEFAULT_SIZES` (or whatever was passed to ``__init__``).
        """
        if train_size is None or val_size is None:
            if molecule_name not in self.sizes:
                raise KeyError(
                    f"No default sizes for {molecule_name!r}; "
                    f"pass train_size/val_size explicitly or extend DEFAULT_SIZES."
                )
            ts, vs = self.sizes[molecule_name]
            train_size = train_size if train_size is not None else ts
            val_size = val_size if val_size is not None else vs

        molecule = self._load_raw(molecule_name)
        E_full = molecule["E"]
        F_full = molecule["F"]
        R_full = molecule["R"]
        Z_template = molecule["z"]

        total = len(E_full)
        test_size = total - train_size - val_size
        if test_size <= 0:
            raise ValueError(
                f"{molecule_name}: train+val ({train_size}+{val_size}) "
                f"exceeds total frames ({total})"
            )

        # Legacy MT19937 RNG matches process_md22.py exactly — re-processing
        # with the same seed produces the same splits as the existing NPZs.
        rng = np.random.RandomState(self.seed)
        idx = np.arange(total)
        rng.shuffle(idx)
        train_idx = idx[:train_size]
        val_idx = idx[train_size:train_size + val_size]
        test_idx = idx[train_size + val_size:]

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
