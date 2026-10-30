# OpenMM molecular dynamics: ATOMTRICK on Ac-Ala3-NHMe

Gas-phase MD of Ac-Ala3-NHMe (MD22, 42 atoms) with the ATOMTRICK force field: NVE energy
conservation, 300 K NVT stability and backbone conformations (the paper's MD appendix).

```
md_openmm.py            the MD driver (NVE / NVT, resumable, --compile)
analysis/
  nve_drift.py          NVE drift / RMS / <T> table (stdlib only)
  plot_nve.py           NVE drift figure
  nvt_stats.py          NVT <T>, sigma_T, T drift (stdlib only)
  nvt_tplot.py          NVT temperature-vs-time figure
  ramachandran_classes.py   backbone phi/psi class populations + C7/C5 H-bond flags
results/reported_1ns/   raw outputs and tables of the reported 1 ns runs
```

Requirements: the ATOMTRICK environment plus `openmm` (the reported runs used 8.5.2) and
matplotlib (`pip install -e .[md]`). The model is the MD22 Ac-Ala3-NHMe checkpoint
(`configs/md22/atomtrick.yaml --molecule Ac-Ala3-NHMe`, seed 42); the driver reads the
architecture from the `args.json` next to it.

## Driver

```bash
python md/md_openmm.py --model atomtrick --ckpt CKPT --ensemble nve|nvt --out runs/md/NAME \
    [--compile] [--steps N] [--equil-steps 10000] [--seed 2024]
```

Defaults are the reported protocol: initial frame 0 of `data/md22/Ac-Ala3-NHMe_test_default.npz`,
300 K, dt 1 fs, Langevin friction 1 ps⁻¹, fp32 model, fp64 OpenMM CPU integrator, forces injected
through a `CustomExternalForce`, coordinates centred before every model call, reports and
trajectory frames every 2000 steps, checkpoints every 10,000 steps (re-run the same command to
resume). NVE = 10 ps Langevin equilibration, centre-of-mass momentum removed at the hand-off,
then production (temperature over 3N−3 degrees of freedom; NVT uses 3N). Details are in the
module docstring.

## Reported runs and re-running them

Both runs: compiled, matmul precision `highest`, 1 ns, frames every 2 ps, about 5 h each on one
L40S. `results/reported_1ns/` holds their raw outputs (`atomtrick_nve_energy.csv`,
`atomtrick_nvt_energy.csv`, `atomtrick_nvt_traj.npz` with 500 frames) and the tables computed
from them (`nve_drift.md`, `nve_drift_per100ps.md`, `nvt_stats.md`, `ramachandran_classes.md`).
GPU force evaluation is not bitwise deterministic, so a re-run reproduces the numbers
statistically.

```bash
A=path/to/md22_Ac-Ala3-NHMe_best.pt          # args.json next to it
# NVE: 10 ps equilibration + 1 ns production
python md/md_openmm.py --model atomtrick --ckpt $A --compile --ensemble nve --steps 1000000 --out runs/md/atomtrick_nve
python md/analysis/nve_drift.py --csv ATOMTRICK=runs/md/atomtrick_nve_energy.csv --windows 100 500 690 1000
# NVT at 300 K, 1 ns, no separate equilibration
python md/md_openmm.py --model atomtrick --ckpt $A --compile --ensemble nvt --equil-steps 0 --steps 1000000 --out runs/md/atomtrick_nvt
python md/analysis/nvt_stats.py --csv ATOMTRICK=runs/md/atomtrick_nvt_energy.csv
python md/analysis/ramachandran_classes.py --traj ATOMTRICK=runs/md/atomtrick_nvt_traj.npz --out runs/md/ramachandran_classes.png
```

Without `--csv` / `--traj` the analysis scripts read `results/reported_1ns/` and recompute the
committed tables; `nve_drift.py --per 100 --t_max 1000` gives the per-100-ps table, and
`plot_nve.py --out nve_drift_1ns.png` and `nvt_tplot.py --out nvt_temperature_1ns.png` draw the
figures.
