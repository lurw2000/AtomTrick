# Experiment configurations

One YAML file (or a stack of them) per training experiment in the paper, for `train.py`.

```bash
python train.py --config configs/rmd17/atomtrick.yaml --molecule aspirin --seed 42
python train.py --config configs/rmd17/atomtrick.yaml --config configs/rmd17/ablations/so2.yaml \
    --molecule aspirin --seed 1          # later files override earlier ones; CLI flags override all
python train.py --config configs/rmd17/atomtrick.yaml --molecule aspirin --dry_run   # resolve + count params
```

- **Base files** (`rmd17/atomtrick.yaml`, `md22/atomtrick.yaml`, `spice/stage1_force.yaml`) list
  every model and training key, so a base file alone describes the run.
- **Override files** (`rmd17/ablations/*.yaml`, `spice/stage2_swa.yaml`) contain only the changed
  keys and a new `experiment` name; apply them after their base file.
- Keys are `train.py` option names; `{molecule}`, `{seed}` and `{experiment}` are filled in after
  merging. Runs write to `runs/<experiment>/<molecule>/s<seed>/`: `args.json`, `*_best.pt` (the
  EMA weights at the best validation loss, the checkpoint to evaluate), `*_final.pt` and a
  rolling checkpoint.
- The header of each file states what it reproduces, the parameter count, the hardware and wall
  time of the reported run, the reported result and caveats.

| Result | Config(s) | Seeds | Driver (`scripts/reproduce/`) |
|---|---|---|---|
| rMD17 energy / force MAE and force-L2 tables | `rmd17/atomtrick.yaml` | 42, 1, 2 | `rmd17.sh` |
| rMD17 O(2) vs SO(2) K/V (the main ablation) | + `rmd17/ablations/so2.yaml` | 42, 1, 2 | `rmd17_ablations.sh so2` |
| rMD17 ablation table: softmax attention, per-degree LayerNorm, O(3)-linear K/V, L_max 1 / 2, 4 / 8 / 12 layers | + `rmd17/ablations/{softmax,perdegree_ln,o3_kv,lmax1,lmax2,layers4,layers8,layers12}.yaml` | 42 | `rmd17_ablations.sh <name>` |
| MD22, 7 systems | `md22/atomtrick.yaml` | 42 | `md22.sh` |
| SPICE vs MACE-OFF23, 7 subsets | `spice/stage1_force.yaml` → + `spice/stage2_swa.yaml` | 42 | `spice.sh` |

Every run is full fp32 (`matmul_precision: highest`, float64 labels) and is scored on its test
set. The drivers train every run of a table and then evaluate it; `scripts/reproduce/eval_all.sh`
builds the tables.
