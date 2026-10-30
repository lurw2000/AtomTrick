# Datasets

All commands run from the repository root. Raw downloads go to `data/raw/`, the processed NPZs
that `configs/` point to go to `data/<dataset>/`.

| Dataset | Download (size) | Prepare | Output |
|---|---|---|---|
| rMD17 | `bash scripts/data/download_rmd17.sh` (1.1 GB) | `python scripts/data/prepare_rmd17.py` | `data/rmd17/<mol>_{train,val,test}_default.npz` |
| MD22 | `bash scripts/data/download_md22.sh` (565 MB) | `python scripts/data/prepare_md22.py` | `data/md22/<system>_{train,val,test}_default.npz` |
| SPICE (MACE-OFF23 split) | `bash scripts/data/download_spice_maceoff.sh` (1.7 GB) | `python scripts/data/prepare_spice_maceoff.py` | `data/spice_maceoff/spice_maceoff_{train,val,test,test_f15}.npz` |

`python scripts/data/verify_data.py` checks every processed array (dtype, shape and md5 of the
raw bytes, listed in `checksums_arrays.tsv`) against the files we trained on and exits 1 on a
mismatch. `checksums.md5` holds file-level md5s of the raw downloads
(`md5sum -c --ignore-missing scripts/data/checksums.md5`); a regenerated NPZ never matches a
file-level md5 because `numpy.savez_compressed` stores the current time, so use `verify_data.py`
for processed files.

The download scripts verify md5s, skip files that are already present and resume partial
downloads, so they can be re-run. `RMD17_URL` and `MD22_BASE_URL` override the sources. If the
SPICE links move, download the two tarballs from the DOI page into `data/raw/spice_maceoff/` and
re-run the script.

## NPZ format

One file per split, flattened over atoms: `E (n_struct,)` total energy in kcal/mol,
`F (n_atoms_total, 3)` in kcal/mol/Å, `R (n_atoms_total, 3)` in Å, `Z (n_atoms_total,)` atomic
numbers, `N (n_struct,)` atoms per structure; SPICE files add `subset` and `subset_names`. The
training loader converts to eV and subtracts atomic reference energies in float64.

## Splits

**rMD17** (Christensen & von Lilienfeld 2020, figshare
[10.6084/m9.figshare.12672038](https://doi.org/10.6084/m9.figshare.12672038), CC0). Official
split 1: the 1,000 training indices are shuffled with `np.random.RandomState(0)`, the last 50
are validation and the other 950 train; test is the 1,000 official test frames. Validation
selects the checkpoint; results are on the test split.

**MD22** (Chmiela et al. 2023, sGDML; check the sGDML terms of use before redistributing). There
are no official splits: frames are shuffled with `np.random.RandomState(1)`; training sizes
follow Chmiela et al. (6000 / 8000 / 3000 / 8000 / 2000 / 600 / 800 for Ac-Ala3-NHMe, DHA,
AT-AT, stachyose, AT-AT-CG-CG, buckyball-catcher, double-walled_nanotube); validation is 500
frames (100 for the two largest systems); test is every remaining frame.

**SPICE / MACE-OFF23** (Kovács et al., Apollo
[10.17863/CAM.107498](https://doi.org/10.17863/CAM.107498), MIT). MACE-OFF23's own files: train
`train_large_neut_no_bad_clean` (951,005 structures), of which a structure-level random 5 %
(`np.random.default_rng(42)`) is held out for validation (903,455 / 47,550), and test
`test_large_neut_all` (50,195). Labels are the DFT `energy` and `forces` fields.
`spice_maceoff_test_f15.npz` keeps the test structures with max ‖F_atom‖ ≤ 15 eV/Å and
‖Σ F‖ ≤ 0.1 eV/Å (MACE-OFF's SPICE 2 cleaning thresholds, computed in float64 from the DFT
labels); this drops 126 PubChem structures and leaves 50,069. **Results are reported on this
filtered file only**, and the drivers evaluate nothing else. Parsing the 5 GB training xyz is pure
Python (tens of minutes, several GB of RAM); `--skip_train` builds only the test files.
