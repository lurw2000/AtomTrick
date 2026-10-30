"""Physical / dataset-specific constants used at data-load time.

Each dataset ships its own isolated-atom reference energies (DFT settings,
basis set, and the specific code used to compute them all vary).  Mixing
a ref table from one dataset with energies from another introduces a
per-molecule bias; always use the canonical table for the dataset.

``ATOMIC_REFERENCE_ENERGIES`` below is the legacy default used for
rMD17 / MD22 training (values originally fit to rmd17 energies); it is
kept unchanged for backward compatibility.

Additional per-dataset tables live in ``ATOMIC_REFERENCE_ENERGIES_BY_DATASET``
and are picked up automatically by train.py when ``--dataset`` is set,
unless ``--fit_atomic_refs`` overrides them.
"""

# Legacy default (rMD17 / MD22).  Kept for existing code paths that import
# this symbol directly (MLFFDataset, data processors).
ATOMIC_REFERENCE_ENERGIES = {
    1: -13.568422498506,
    6: -1027.263868705604,
    7: -1482.111762854364,
    8: -2038.533116588341,
}

ATOMIC_REFERENCE_ENERGIES_BY_DATASET = {
    # Any key not listed here falls back to ATOMIC_REFERENCE_ENERGIES.
    "rmd17": ATOMIC_REFERENCE_ENERGIES,
    "md22":  ATOMIC_REFERENCE_ENERGIES,
}
