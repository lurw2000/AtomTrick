# Inference latency (H100 80GB HBM3, full fp32) — energy + forces

Warmup 10 + 50 timed calls, per-call CUDA-event timing, random-init weights. **ATOMTRICK (compiled)** = make_fx whole-graph capture of forward + force backward, compiled by torch.compile / TorchInductor (no hand-written kernels, no CUDA graphs); **ATOMTRICK (eager)** = plain PyTorch autograd. Baselines run in two modes: **eager** = plain PyTorch (the reported protocol), **`<model> (compiled)`** = the same harness with the whole energy + force callable under `torch.compile(dynamic=True)` / TorchInductor (parameters frozen, `PYTORCH_JIT=0`); of the two compile recipes measured per baseline the table shows the faster one, see "Compiled-baseline variants".

## Median latency (ms), batch size 1

| Dataset | Molecule | Atoms | Equiformer | Equiformer (compiled) | EquiformerV2 | EquiformerV2 (compiled) | GotenNet | GotenNet (compiled) | MACE | MACE (compiled) | ATOMTRICK (eager) | ATOMTRICK (compiled) | fastest eager baseline | ATOMTRICK (compiled) speedup | fastest compiled baseline | ATOMTRICK (compiled) speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rmd17 | ethanol | 9 | 101.0 | 58.3 | 108.3 | 105.0 | 41.7 | 17.6 | 16.3 | 4.3 | 57.0 | 13.5 | MACE | 1.21x | MACE | 0.32x |
| rmd17 | malonaldehyde | 9 | 100.8 | 58.9 | 103.1 | 104.9 | 41.7 | 17.6 | 16.4 | 4.4 | 57.0 | 13.6 | MACE | 1.21x | MACE | 0.32x |
| rmd17 | benzene | 12 | 101.4 | 59.0 | 103.2 | 105.0 | 42.1 | 17.1 | 16.7 | 4.4 | 57.2 | 13.5 | MACE | 1.24x | MACE | 0.32x |
| rmd17 | uracil | 12 | 101.4 | 58.7 | 103.3 | 104.9 | 42.1 | 17.2 | 16.6 | 4.4 | 57.1 | 13.5 | MACE | 1.23x | MACE | 0.32x |
| rmd17 | toluene | 15 | 101.5 | 58.6 | 103.5 | 104.8 | 42.3 | 17.2 | 16.7 | 4.4 | 57.1 | 13.6 | MACE | 1.23x | MACE | 0.32x |
| rmd17 | salicylic | 16 | 95.8 | 58.5 | 103.6 | 105.6 | 42.2 | 17.2 | 16.7 | 4.4 | 57.2 | 13.6 | MACE | 1.23x | MACE | 0.32x |
| rmd17 | naphthalene | 18 | 94.2 | 57.2 | 103.2 | 105.0 | 42.5 | 17.5 | 16.7 | 4.4 | 57.5 | 13.7 | MACE | 1.22x | MACE | 0.32x |
| rmd17 | paracetamol | 20 | 94.5 | 57.1 | 103.3 | 104.8 | 42.5 | 17.5 | 16.7 | 4.4 | 57.5 | 13.7 | MACE | 1.22x | MACE | 0.32x |
| rmd17 | aspirin | 21 | 94.2 | 57.3 | 103.2 | 105.5 | 44.0 | 17.6 | 16.7 | 4.4 | 57.7 | 13.7 | MACE | 1.22x | MACE | 0.32x |
| rmd17 | azobenzene | 24 | 94.2 | 57.2 | 103.6 | 105.8 | 42.5 | 17.5 | 16.7 | 4.4 | 57.8 | 13.7 | MACE | 1.22x | MACE | 0.32x |
| md22 | Ac_Ala3_NHMe | 42 | 118.6 | 58.6 | 110.6 | 107.0 | 29.1 | 12.9 | 18.3 | 4.6 | 57.6 | 13.7 | MACE | 1.34x | MACE | 0.33x |
| md22 | DHA | 56 | 119.3 | 59.0 | 112.9 | 108.6 | 29.6 | 14.0 | 16.8 | 5.1 | 58.1 | 13.8 | MACE | 1.22x | MACE | 0.37x |
| md22 | AT_AT | 60 | 118.8 | 58.7 | 113.2 | 108.5 | 30.1 | 14.4 | 16.8 | 5.3 | 58.1 | 13.8 | MACE | 1.22x | MACE | 0.38x |
| md22 | stachyose | 87 | 118.6 | 58.2 | 147.0 | 131.0 | 37.1 | 18.2 | 16.8 | 7.4 | 59.0 | 14.0 | MACE | 1.20x | MACE | 0.53x |
| md22 | AT_AT_CG_CG | 118 | 119.1 | 58.4 | 175.0 | 153.0 | 44.7 | 22.1 | 16.9 | 8.9 | 60.1 | 13.9 | MACE | 1.22x | MACE | 0.64x |
| md22 | buckyball_catcher | 148 | 109.8 | 58.4 | 210.1 | 182.0 | 54.2 | 26.4 | 18.4 | 11.0 | 58.9 | 14.7 | MACE | 1.25x | MACE | 0.74x |
| md22 | double_walled_nanotube | 370 | 122.0 | 63.3 | 515.6 | 439.4 | 121.2 | 57.8 | 45.0 | 26.3 | 73.7 | 35.8 | MACE | 1.26x | MACE | 0.73x |

## Median latency (ms), batch size 4

| Dataset | Molecule | Atoms | Equiformer | Equiformer (compiled) | EquiformerV2 | EquiformerV2 (compiled) | GotenNet | GotenNet (compiled) | MACE | MACE (compiled) | ATOMTRICK (eager) | ATOMTRICK (compiled) | fastest eager baseline | ATOMTRICK (compiled) speedup | fastest compiled baseline | ATOMTRICK (compiled) speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rmd17 | ethanol | 9 | 100.7 | 58.5 | 104.3 | 105.9 | 42.4 | 17.7 | 16.6 | 4.4 | 57.8 | 14.0 | MACE | 1.18x | MACE | 0.32x |
| rmd17 | malonaldehyde | 9 | 100.6 | 58.2 | 104.0 | 106.1 | 42.4 | 17.7 | 16.6 | 4.4 | 57.8 | 14.0 | MACE | 1.18x | MACE | 0.32x |
| rmd17 | benzene | 12 | 100.6 | 58.2 | 104.1 | 106.1 | 42.8 | 17.8 | 16.9 | 4.5 | 58.1 | 14.1 | MACE | 1.20x | MACE | 0.32x |
| rmd17 | uracil | 12 | 100.7 | 57.8 | 104.2 | 106.0 | 42.7 | 17.8 | 16.8 | 4.5 | 58.0 | 14.1 | MACE | 1.19x | MACE | 0.32x |
| rmd17 | toluene | 15 | 95.9 | 58.8 | 104.7 | 106.7 | 42.9 | 17.6 | 16.9 | 4.5 | 58.4 | 14.1 | MACE | 1.19x | MACE | 0.32x |
| rmd17 | salicylic | 16 | 95.2 | 58.3 | 104.8 | 107.0 | 42.8 | 17.6 | 16.9 | 4.6 | 58.2 | 14.1 | MACE | 1.20x | MACE | 0.33x |
| rmd17 | naphthalene | 18 | 95.2 | 58.1 | 106.0 | 106.4 | 43.0 | 17.8 | 16.8 | 5.1 | 58.3 | 14.2 | MACE | 1.19x | MACE | 0.36x |
| rmd17 | paracetamol | 20 | 95.8 | 57.7 | 106.9 | 107.1 | 44.2 | 17.8 | 16.8 | 5.4 | 58.5 | 14.1 | MACE | 1.19x | MACE | 0.38x |
| rmd17 | aspirin | 21 | 95.6 | 58.1 | 107.5 | 107.8 | 43.2 | 17.7 | 16.8 | 5.7 | 58.4 | 14.2 | MACE | 1.18x | MACE | 0.40x |
| rmd17 | azobenzene | 24 | 95.7 | 58.0 | 107.7 | 107.8 | 43.0 | 17.8 | 16.7 | 6.0 | 58.4 | 14.1 | MACE | 1.18x | MACE | 0.42x |
| md22 | Ac_Ala3_NHMe | 42 | 119.3 | 58.9 | 204.8 | 178.5 | 55.1 | 27.1 | 18.2 | 10.8 | 59.0 | 14.5 | MACE | 1.26x | MACE | 0.74x |
| md22 | DHA | 56 | 119.5 | 58.9 | 258.7 | 223.3 | 69.6 | 34.0 | 23.2 | 13.7 | 58.7 | 16.7 | MACE | 1.39x | MACE | 0.82x |
| md22 | AT_AT | 60 | 119.5 | 58.7 | 259.0 | 223.9 | 71.7 | 34.9 | 23.7 | 14.1 | 58.8 | 16.8 | MACE | 1.42x | MACE | 0.84x |
| md22 | stachyose | 87 | 119.1 | 60.4 | 423.8 | 361.3 | 107.1 | 50.4 | 37.8 | 22.3 | 64.5 | 28.2 | MACE | 1.34x | MACE | 0.79x |
| md22 | AT_AT_CG_CG | 118 | 122.6 | 62.7 | 538.2 | 460.0 | 136.2 | 64.7 | 47.9 | 28.1 | 73.3 | 35.1 | MACE | 1.36x | MACE | 0.80x |
| md22 | buckyball_catcher | 148 | 142.9 | 73.5 | 679.0 | 578.7 | 172.7 | 80.9 | 60.5 | 35.5 | 85.9 | 45.0 | MACE | 1.34x | MACE | 0.79x |
| md22 | double_walled_nanotube | 370 | 359.2 | 168.5 | OOM | OOM | OOM | OOM | 166.5 | 96.4 | 229.6 | 124.7 | MACE | 1.34x | MACE | 0.77x |

## Peak GPU memory (GB), batch size 1

| Dataset | Molecule | Atoms | Equiformer | Equiformer (compiled) | EquiformerV2 | EquiformerV2 (compiled) | GotenNet | GotenNet (compiled) | MACE | MACE (compiled) | ATOMTRICK (eager) | ATOMTRICK (compiled) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rmd17 | ethanol | 9 | 0.11 | 0.06 | 0.79 | 0.62 | 0.42 | 0.46 | 0.34 | 0.33 | 0.28 | 0.26 |
| rmd17 | malonaldehyde | 9 | 0.11 | 0.06 | 0.79 | 0.62 | 0.42 | 0.47 | 0.34 | 0.33 | 0.28 | 0.26 |
| rmd17 | benzene | 12 | 0.13 | 0.06 | 0.99 | 0.76 | 0.65 | 0.73 | 0.35 | 0.35 | 0.30 | 0.28 |
| rmd17 | uracil | 12 | 0.13 | 0.06 | 0.99 | 0.76 | 0.65 | 0.74 | 0.35 | 0.35 | 0.30 | 0.28 |
| rmd17 | toluene | 15 | 0.15 | 0.07 | 1.21 | 0.89 | 0.84 | 1.02 | 0.36 | 0.36 | 0.33 | 0.29 |
| rmd17 | salicylic | 16 | 0.16 | 0.08 | 1.26 | 0.93 | 0.91 | 1.11 | 0.36 | 0.37 | 0.34 | 0.30 |
| rmd17 | naphthalene | 18 | 0.18 | 0.08 | 1.41 | 1.03 | 1.05 | 1.30 | 0.37 | 0.39 | 0.35 | 0.31 |
| rmd17 | paracetamol | 20 | 0.19 | 0.09 | 1.51 | 1.09 | 1.18 | 1.44 | 0.37 | 0.41 | 0.37 | 0.31 |
| rmd17 | aspirin | 21 | 0.20 | 0.09 | 1.61 | 1.16 | 1.29 | 1.59 | 0.39 | 0.42 | 0.38 | 0.32 |
| rmd17 | azobenzene | 24 | 0.21 | 0.10 | 1.68 | 1.20 | 1.38 | 1.68 | 0.40 | 0.44 | 0.39 | 0.33 |
| md22 | Ac_Ala3_NHMe | 42 | 0.51 | 0.22 | 4.09 | 2.81 | 3.19 | 3.91 | 0.67 | 0.75 | 0.71 | 0.50 |
| md22 | DHA | 56 | 0.67 | 0.28 | 5.36 | 3.65 | 4.14 | 5.06 | 0.82 | 0.94 | 0.86 | 0.60 |
| md22 | AT_AT | 60 | 0.67 | 0.28 | 5.41 | 3.67 | 4.29 | 5.24 | 0.83 | 0.97 | 0.86 | 0.60 |
| md22 | stachyose | 87 | 1.18 | 0.48 | 9.41 | 6.35 | 6.91 | 8.54 | 1.26 | 1.48 | 1.35 | 0.89 |
| md22 | AT_AT_CG_CG | 118 | 1.49 | 0.59 | 12.01 | 8.05 | 8.88 | 11.04 | 1.58 | 1.89 | 1.67 | 1.08 |
| md22 | buckyball_catcher | 148 | 1.91 | 0.76 | 15.43 | 10.33 | 11.56 | 14.39 | 1.98 | 2.37 | 2.13 | 1.34 |
| md22 | double_walled_nanotube | 370 | 5.71 | 2.27 | 44.01 | 29.43 | 30.22 | 37.86 | 5.38 | 6.45 | 6.04 | 3.61 |

## Peak GPU memory (GB), batch size 4

| Dataset | Molecule | Atoms | Equiformer | Equiformer (compiled) | EquiformerV2 | EquiformerV2 (compiled) | GotenNet | GotenNet (compiled) | MACE | MACE (compiled) | ATOMTRICK (eager) | ATOMTRICK (compiled) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rmd17 | ethanol | 9 | 0.19 | 0.09 | 1.71 | 1.18 | 1.26 | 1.54 | 0.44 | 0.50 | 0.38 | 0.32 |
| rmd17 | malonaldehyde | 9 | 0.19 | 0.09 | 1.71 | 1.18 | 1.25 | 1.55 | 0.44 | 0.50 | 0.38 | 0.32 |
| rmd17 | benzene | 12 | 0.29 | 0.13 | 2.51 | 1.69 | 2.11 | 2.60 | 0.53 | 0.63 | 0.47 | 0.37 |
| rmd17 | uracil | 12 | 0.29 | 0.13 | 2.49 | 1.68 | 2.12 | 2.62 | 0.53 | 0.63 | 0.47 | 0.37 |
| rmd17 | toluene | 15 | 0.38 | 0.17 | 3.33 | 2.23 | 3.00 | 3.76 | 0.64 | 0.77 | 0.58 | 0.44 |
| rmd17 | salicylic | 16 | 0.41 | 0.18 | 3.57 | 2.38 | 3.25 | 4.10 | 0.67 | 0.81 | 0.60 | 0.45 |
| rmd17 | naphthalene | 18 | 0.50 | 0.21 | 4.22 | 2.81 | 3.89 | 4.93 | 0.75 | 0.91 | 0.69 | 0.50 |
| rmd17 | paracetamol | 20 | 0.52 | 0.22 | 4.54 | 3.01 | 4.20 | 5.33 | 0.80 | 0.98 | 0.74 | 0.52 |
| rmd17 | aspirin | 21 | 0.57 | 0.24 | 4.87 | 3.23 | 4.53 | 5.74 | 0.84 | 1.04 | 0.78 | 0.54 |
| rmd17 | azobenzene | 24 | 0.61 | 0.26 | 5.29 | 3.48 | 4.92 | 6.34 | 0.91 | 1.13 | 0.83 | 0.57 |
| md22 | Ac_Ala3_NHMe | 42 | 1.81 | 0.73 | 14.85 | 9.88 | 11.89 | 14.84 | 1.96 | 2.44 | 2.03 | 1.27 |
| md22 | DHA | 56 | 2.41 | 0.97 | 20.06 | 13.31 | 15.78 | 19.71 | 2.59 | 3.22 | 2.66 | 1.65 |
| md22 | AT_AT | 60 | 2.42 | 0.97 | 20.22 | 13.37 | 16.39 | 20.48 | 2.64 | 3.32 | 2.67 | 1.65 |
| md22 | stachyose | 87 | 4.35 | 1.73 | 36.04 | 23.99 | 26.78 | 33.57 | 4.37 | 5.43 | 4.67 | 2.82 |
| md22 | AT_AT_CG_CG | 118 | 5.58 | 2.21 | 46.47 | 30.83 | 34.71 | 43.48 | 5.63 | 7.05 | 5.97 | 3.56 |
| md22 | buckyball_catcher | 148 | 7.28 | 2.89 | 60.27 | 40.01 | 45.33 | 56.87 | 7.21 | 9.02 | 7.68 | 4.56 |
| md22 | double_walled_nanotube | 370 | 22.50 | 8.89 | OOM | OOM | OOM | OOM | 20.82 | 25.70 | 23.30 | 13.67 |

## Throughput (structures/s), batch size 4 — median-based (mean-based)

Median-based = 4000 / median latency (the paper's definition). Mean-based = 4000 / mean latency; it is pulled down by occasional slow calls.

| Dataset | Molecule | Atoms | Equiformer | Equiformer (compiled) | EquiformerV2 | EquiformerV2 (compiled) | GotenNet | GotenNet (compiled) | MACE | MACE (compiled) | ATOMTRICK (eager) | ATOMTRICK (compiled) |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| rmd17 | ethanol | 9 | 39.7 (39.7) | 68.4 (68.3) | 38.4 (38.2) | 37.8 (37.7) | 94.4 (94.2) | 225.4 (225.0) | 241.2 (240.5) | 901.6 (898.6) | 69.3 (69.1) | 285.0 (285.0) |
| rmd17 | malonaldehyde | 9 | 39.8 (39.8) | 68.7 (68.3) | 38.5 (38.4) | 37.7 (37.7) | 94.3 (94.3) | 225.8 (224.8) | 241.1 (239.8) | 903.6 (901.2) | 69.2 (69.1) | 285.0 (284.9) |
| rmd17 | benzene | 12 | 39.8 (39.7) | 68.8 (68.3) | 38.4 (38.4) | 37.7 (37.6) | 93.5 (93.2) | 225.0 (224.2) | 236.5 (236.5) | 896.4 (894.0) | 68.8 (68.7) | 282.9 (282.9) |
| rmd17 | uracil | 12 | 39.7 (39.7) | 69.2 (69.0) | 38.4 (38.4) | 37.7 (37.6) | 93.7 (93.6) | 225.3 (224.6) | 238.2 (237.1) | 896.4 (893.6) | 69.0 (69.0) | 283.7 (283.8) |
| rmd17 | toluene | 15 | 41.7 (40.5) | 68.0 (68.1) | 38.2 (38.2) | 37.5 (34.4) | 93.2 (85.6) | 226.8 (224.9) | 237.1 (237.0) | 896.1 (892.4) | 68.5 (67.2) | 282.8 (282.6) |
| rmd17 | salicylic | 16 | 42.0 (42.0) | 68.7 (68.3) | 38.2 (38.1) | 37.4 (37.4) | 93.6 (93.5) | 226.9 (225.3) | 236.5 (236.3) | 871.5 (868.6) | 68.7 (68.6) | 283.7 (283.6) |
| rmd17 | naphthalene | 18 | 42.0 (42.0) | 68.9 (68.8) | 37.7 (37.7) | 37.6 (37.5) | 92.9 (92.9) | 225.3 (224.8) | 237.9 (237.9) | 779.6 (777.8) | 68.6 (68.5) | 282.1 (281.9) |
| rmd17 | paracetamol | 20 | 41.8 (41.8) | 69.3 (69.3) | 37.4 (37.4) | 37.4 (37.4) | 90.5 (90.1) | 224.5 (224.0) | 238.6 (238.7) | 738.3 (736.6) | 68.4 (68.4) | 283.3 (283.3) |
| rmd17 | aspirin | 21 | 41.8 (41.8) | 68.8 (68.7) | 37.2 (37.2) | 37.1 (37.1) | 92.7 (85.8) | 225.6 (223.5) | 238.7 (238.4) | 699.8 (698.4) | 68.5 (68.5) | 281.1 (281.3) |
| rmd17 | azobenzene | 24 | 41.8 (41.8) | 69.0 (69.0) | 37.1 (37.2) | 37.1 (37.1) | 93.0 (93.0) | 225.3 (224.6) | 239.5 (239.4) | 671.2 (670.0) | 68.5 (68.5) | 283.5 (283.5) |
| md22 | Ac_Ala3_NHMe | 42 | 33.5 (33.5) | 67.9 (67.4) | 19.5 (19.5) | 22.4 (22.4) | 72.6 (72.5) | 147.6 (147.6) | 219.4 (219.0) | 371.5 (371.1) | 67.8 (67.8) | 276.3 (275.7) |
| md22 | DHA | 56 | 33.5 (33.5) | 67.9 (67.8) | 15.5 (15.5) | 17.9 (17.9) | 57.5 (57.5) | 117.7 (117.5) | 172.3 (172.0) | 292.5 (292.2) | 68.1 (68.0) | 239.7 (238.9) |
| md22 | AT_AT | 60 | 33.5 (33.5) | 68.1 (68.0) | 15.4 (15.4) | 17.9 (17.9) | 55.8 (55.8) | 114.5 (114.4) | 168.6 (168.3) | 284.0 (283.8) | 68.0 (67.8) | 238.7 (238.3) |
| md22 | stachyose | 87 | 33.6 (33.6) | 66.2 (66.0) | 9.4 (9.4) | 11.1 (11.1) | 37.3 (37.3) | 79.4 (79.4) | 105.8 (105.7) | 179.3 (179.2) | 62.0 (62.0) | 141.8 (141.6) |
| md22 | AT_AT_CG_CG | 118 | 32.6 (32.6) | 63.8 (63.8) | 7.4 (7.4) | 8.7 (8.7) | 29.4 (29.4) | 61.8 (61.7) | 83.6 (83.5) | 142.3 (142.2) | 54.6 (54.5) | 114.0 (113.9) |
| md22 | buckyball_catcher | 148 | 28.0 (28.0) | 54.4 (54.4) | 5.9 (5.9) | 6.9 (6.9) | 23.2 (23.2) | 49.5 (49.5) | 66.1 (66.1) | 112.7 (112.7) | 46.6 (46.6) | 88.8 (88.7) |
| md22 | double_walled_nanotube | 370 | 11.1 (11.1) | 23.7 (23.7) | OOM | OOM | OOM | OOM | 24.0 (24.0) | 41.5 (41.5) | 17.4 (17.4) | 32.1 (32.1) |

## Model sizes (parameters, per dataset config)

- **Equiformer**: rMD17: 3,496,001; MD22: 3,496,001
- **Equiformer (compiled)**: rMD17: 3,496,001; MD22: 3,496,001
- **EquiformerV2**: rMD17: 77,182,465; MD22: 77,182,465
- **EquiformerV2 (compiled)**: rMD17: 77,182,465; MD22: 77,182,465
- **GotenNet**: rMD17: 21,069,057; MD22: 42,603,777
- **GotenNet (compiled)**: rMD17: 21,069,057; MD22: 42,603,777
- **MACE**: rMD17: 3,769,616; MD22: 3,769,616
- **MACE (compiled)**: rMD17: 3,769,616; MD22: 3,769,616
- **ATOMTRICK (eager)**: rMD17: 3,730,368; MD22: 3,730,368
- **ATOMTRICK (compiled)**: rMD17: 3,730,368; MD22: 3,730,368

## Compiled-baseline variants

Two torch.compile recipes per baseline (`inductor`: graph break at `autograd.grad`, forces from the compiled backward; `fused`: `autograd.grad` inlined, forward + force backward in one graph; see Measurement conditions). The tables above show the selected variant as `<model> (compiled)`; both are in the long CSV (`selected` column). Selection: `--compiled_variant auto` = lower geometric-mean median latency over the rows where both variants are ok (a single variant with ok rows is taken as is).

| Model | selected | inductor rows ok / oom / compile_error | fused rows ok / oom / compile_error | fused / inductor median latency (geo-mean over common ok rows) | max compile_graphs_warmup (inductor / fused) | rows with compile_graphs_timed > 0 (inductor / fused) |
|---|---|---|---|---|---|---|
| Equiformer | inductor | 34 / 0 / 0 (of 34) | 22 / 12 / 0 (of 34) | 1.235 (n = 22) | 63 / 64 | 0 / 0 |
| EquiformerV2 | inductor | 33 / 1 / 0 (of 34) | 1 / 33 / 0 (of 34) | 1.329 (n = 1) | 35 / 35 | 0 / 0 |
| GotenNet | inductor | 33 / 1 / 0 (of 34) | 33 / 1 / 0 (of 34) | 1.029 (n = 33) | 15 / 16 | 0 / 0 |
| MACE | fused | 34 / 0 / 0 (of 34) | 34 / 0 / 0 (of 34) | 0.948 (n = 34) | 3 / 1 | 0 / 0 |

## Measurement conditions (read before comparing columns)

- **Equiformer**: eager, strict fp32; real frames 0..bs-1 of the raw trajectory; radius graph built inside the timed forward.
- **EquiformerV2**: eager, strict fp32; OC20 N@12_L@6_M@2 config, never trained on these datasets; autograd forces -dE/dR without the direct force head; real frames 0..bs-1 of the raw trajectory; graph built inside the timed forward.
- **GotenNet**: eager, strict fp32; real frames; per-dataset production config; radius graph built inside the timed forward.
- **MACE**: eager (not compiled), strict fp32, cuEquivariance off; real frames; neighbour list built outside the timed region.
- **Compiled baselines** (`<model> (compiled)` columns): the same harness, inputs and precision as the eager column, with the whole energy + force callable wrapped in `torch.compile(dynamic=True)` (TorchInductor, default mode, no CUDA graphs), parameters frozen (`requires_grad_(False)`, inference deployment), `PYTORCH_JIT=0`; Dynamo trace + Inductor compile happen in the 10 warm-up calls and are excluded from the latency. Two recipes were measured per model: `inductor` = Dynamo graph-breaks at `torch.autograd.grad`, the energy forward is compiled and AOTAutograd's compiled backward computes the forces (create_graph=False, as eager); `fused` = `torch.autograd.grad` allowed in the graph, so forward + force backward form ONE compiled graph (MACE's upstream recipe, generalised; create_graph=True inside, traced with grad enabled, outputs detached inside the compiled callable so AOTAutograd compiles an inference graph that runs under no_grad -- the shape of ATOMTRICK's make_fx step). After the warm-up each compiled row's energy and forces were checked against one eager call of the same model (max relative error recorded; a mismatch is a compile_error). The tables show the selected variant (Equiformer: inductor, EquiformerV2: inductor, GotenNet: inductor, MACE: fused).
- **Equiformer (compiled)**: e3nn TorchScript codegen off (`jit_script_fx=False`); radius graph still built inside the timed forward.
- **EquiformerV2 (compiled)**: the harness's model + autograd.grad closure is the compiled callable; graph still built inside the timed forward.
- **GotenNet (compiled)**: radius graph still built inside the timed forward; one random-init model instance reused for all molecules of a dataset (the eager column builds one per molecule; latency is weight-agnostic) so the dynamic-shape graph is shared.
- **MACE (compiled)**: built as MACE's own `prepare()` builds it (e3nn codegen off, registered modules fx-simplified; the upstream torch.compile recipe); one random-init model instance reused for all molecules (the eager column builds one per molecule) so the dynamic-shape graph is shared; run in a separate Python <= 3.12 copy of the MACE environment, because torch 2.5.1 has no torch.compile on Python 3.13 (see the Environments table of MEASUREMENT_CONDITIONS.md).
- **ATOMTRICK**: strict fp32 (matmul precision highest); eager = autograd forces, create_graph=False; compiled = make_fx trace (train_mode) + torch.compile(dynamic=True), run under no_grad; real frames; neighbour list built outside the timed region; one make_fx graph per batch size traced on the first molecule and reused (compile time excluded).
