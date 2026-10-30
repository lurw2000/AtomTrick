"""AtomTrick data package.

Contains both the runtime dataset (:class:`MLFFDataset`, collation, loader
helpers — in ``dataset.py``) and the raw-data processors for rMD17 and MD22
(``rmd17.py``, ``md22.py``).  Everything is re-exported here so existing
``from AtomTrick.data import MLFFDataset`` imports keep working after the
``data.py`` → ``data/`` split.
"""
from .dataset import (
    MLFFDataset,
    collate_fn,
    make_padded_collate_fn,
    compute_max_edges,
    build_dataloader,
    fit_atomic_reference_energies,
)
from .base import RawDatasetProcessor
from .rmd17 import RMD17Processor
from .md22 import MD22Processor

__all__ = [
    # Runtime dataset / loader helpers
    "MLFFDataset",
    "collate_fn",
    "make_padded_collate_fn",
    "compute_max_edges",
    "build_dataloader",
    "fit_atomic_reference_energies",
    # Raw-data processors
    "RawDatasetProcessor",
    "RMD17Processor",
    "MD22Processor",
    # Convenience
    "process_and_resolve",
]


def process_and_resolve(dataset, raw_data_dir, molecule,
                        processed_dir=None, seed=None, **kwargs):
    """Process one molecule from raw data if not already cached; return paths.

    Helper used by ``train.py`` to auto-process raw rmd17/md22 into the
    AtomTrick NPZ layout the rest of the pipeline expects.

    Parameters
    ----------
    dataset : {"rmd17", "md22"}
    raw_data_dir : str or Path
        Root of the raw dataset download.
    molecule : str
        Molecule name (e.g. "aspirin", "double-walled_nanotube").
    processed_dir : str or Path, optional
        Where to write the processed NPZs.  Defaults to
        ``data/{dataset}/`` under the current working directory.
    seed : int, optional
        Override the dataset default (0 for rmd17, 1 for md22).
    **kwargs
        Forwarded to the processor constructor (e.g. ``split_id`` for
        rmd17) or to ``process_molecule`` (e.g. ``train_size`` for md22).

    Returns
    -------
    dict
        ``{"train": Path, "val": Path, "test": Path, "n_train": int,
           "n_val": int, "n_test": int, "n_atoms": int}`` — same format
        as :meth:`RawDatasetProcessor.process_molecule`.
    """
    from pathlib import Path
    import logging

    logger = logging.getLogger(__name__)

    if dataset not in {"rmd17", "md22"}:
        raise ValueError(f"dataset must be 'rmd17' or 'md22', got {dataset!r}")

    processed_dir = Path(processed_dir) if processed_dir else Path("data") / dataset

    # Cache check: if the expected train/val/test files already exist,
    # skip re-processing.
    train_p = processed_dir / f"{molecule}_train_default.npz"
    val_p = processed_dir / f"{molecule}_val_default.npz"
    test_p = processed_dir / f"{molecule}_test_default.npz"

    if train_p.exists() and val_p.exists() and test_p.exists():
        logger.info("Processed files already exist at %s — skipping", processed_dir)
        return {"train": train_p, "val": val_p, "test": test_p,
                "n_train": None, "n_val": None, "n_test": None,
                "n_atoms": None}

    # Split kwargs: constructor-time vs process-time
    ctor_kwargs = {}
    proc_kwargs = {}
    if dataset == "rmd17":
        for k in ("split_id", "val_fraction"):
            if k in kwargs:
                ctor_kwargs[k] = kwargs.pop(k)
        if seed is not None:
            ctor_kwargs["seed"] = seed
        proc = RMD17Processor(raw_data_dir, processed_dir, **ctor_kwargs)
    else:  # md22
        if seed is not None:
            ctor_kwargs["seed"] = seed
        if "sizes" in kwargs:
            ctor_kwargs["sizes"] = kwargs.pop("sizes")
        for k in ("train_size", "val_size"):
            if k in kwargs:
                proc_kwargs[k] = kwargs.pop(k)
        proc = MD22Processor(raw_data_dir, processed_dir, **ctor_kwargs)

    if kwargs:
        raise TypeError(f"Unexpected kwargs for {dataset}: {list(kwargs)}")

    logger.info("Processing %s/%s → %s", dataset, molecule, processed_dir)
    return proc.process_molecule(molecule, **proc_kwargs)
