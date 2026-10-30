"""Base class for raw-dataset processors."""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


class RawDatasetProcessor(ABC):
    """Abstract base for processing a raw MD dataset into AtomTrick NPZ format.

    Subclasses implement ``process_molecule`` which must produce three NPZ files
    (train / val / test) in ``output_dir`` using :meth:`_save_split`.  Files
    must have the column layout that :class:`AtomTrick.data.MLFFDataset`
    expects: ``E`` (n_mol,), ``F`` (total_atoms, 3), ``R`` (total_atoms, 3),
    ``Z`` (total_atoms,), ``N`` (n_mol,).

    Parameters
    ----------
    raw_data_dir : str or Path
        Directory holding the raw dataset download.
    output_dir : str or Path
        Directory to write processed NPZ files into.
    seed : int, default 0
        Seed used by any shuffling the subclass performs.
    """

    def __init__(self, raw_data_dir, output_dir, seed: int = 0):
        self.raw_data_dir = Path(raw_data_dir).expanduser()
        self.output_dir = Path(output_dir).expanduser()
        self.seed = int(seed)

    # ------------------------------------------------------------------
    # Required subclass hook
    # ------------------------------------------------------------------

    @abstractmethod
    def process_molecule(self, molecule_name: str, **kwargs) -> dict:
        """Process a single molecule.

        Must produce three NPZ files in ``output_dir``; returns a dict with
        paths and split sizes for downstream use::

            {"train": Path, "val": Path, "test": Path,
             "n_train": int, "n_val": int, "n_test": int, "n_atoms": int}
        """

    # ------------------------------------------------------------------
    # Convenience: process every molecule the subclass knows about
    # ------------------------------------------------------------------

    @property
    def molecules(self):
        """Subclass may expose a default list of molecule names via ``MOLECULES``."""
        return list(getattr(self, "MOLECULES", []))

    def process_all(self, molecules=None, **kwargs) -> dict:
        """Process every molecule in ``molecules`` (defaults to ``self.molecules``).

        Returns a ``{molecule: info_dict}`` map.
        """
        molecules = molecules or self.molecules
        if not molecules:
            raise ValueError(
                "No molecules specified and no default MOLECULES list on subclass"
            )
        results = {}
        for m in molecules:
            logger.info("Processing %s", m)
            results[m] = self.process_molecule(m, **kwargs)
        return results

    # ------------------------------------------------------------------
    # Shared helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _save_split(path: Path, *, E, F, R, Z, N, **extras) -> None:
        """Write an AtomTrick-format NPZ to ``path`` (compressed)."""
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = dict(E=E, F=F, R=R, Z=Z, N=N, **extras)
        np.savez_compressed(path, **arrays)

    @staticmethod
    def _stack_frames(E_full, F_full, R_full, Z_template, indices) -> dict:
        """Stack selected frames into the AtomTrick flat-atom layout.

        Parameters
        ----------
        E_full : (T,) or (T, 1)       per-frame energies
        F_full : (T, N, 3)            per-frame forces
        R_full : (T, N, 3)            per-frame coordinates
        Z_template : (N,)             atomic numbers (same for all frames)
        indices : 1-D int array       selected frame indices

        Returns ``dict`` with keys E, F, R, Z, N ready for :meth:`_save_split`.
        """
        indices = np.asarray(indices).flatten()
        n = len(indices)
        N_per_mol = R_full.shape[1]
        E = np.asarray(E_full[indices]).squeeze()
        if E.ndim == 0:  # single-frame edge case
            E = E.reshape(1)
        F = F_full[indices].reshape(n * N_per_mol, 3)
        R = R_full[indices].reshape(n * N_per_mol, 3)
        Z = np.tile(Z_template, (n, 1)).astype("float64").reshape(n * N_per_mol)
        N = (np.ones(n) * N_per_mol).astype(int)
        return dict(E=E, F=F, R=R, Z=Z, N=N)

    @staticmethod
    def _log_split(name: str, molecule_name: str, split: dict) -> None:
        """Log a human-readable summary of one processed split."""
        logger.info(
            "  %s %s: n_frames=%d  n_atoms=%d",
            molecule_name, name,
            split["E"].shape[0], split["N"][0],
        )
