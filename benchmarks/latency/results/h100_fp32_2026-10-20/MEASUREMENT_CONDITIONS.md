# Latency measurement conditions (recorded, not assumed)

Results dir: `benchmarks/latency/results/h100_fp32_2026-10-20`

## Common protocol

- One job, one GPU (NVIDIA H100 80GB HBM3); SLURM job(s) 20320246, 20396545; GPU UUID(s) GPU-215d43ec-3ba2-b8cc-e123-cb7caa323c05, GPU-8d1cc5ff-05d8-1a4c-b7fe-a57ec51415a9. Models run sequentially; driver.log has a WARNING line if any other process was on the GPU.
- Strict fp32 for every model: `torch.set_float32_matmul_precision('highest')`, fp32 weights and inputs (MACE explicitly `--dtype float32`; MACE-OFF/MACE papers train in float64). cuDNN TF32 off for the baselines; bench_atomtrick.py leaves `torch.backends.cudnn.allow_tf32` at the PyTorch default (True), which affects only cuDNN convolutions, and ATOMTRICK has none. Random-init weights (latency does not depend on weight values).
- Inputs: frames 0..bs-1 of the raw rMD17 / MD22 trajectories for every model (real coordinates; no synthetic Gaussian batches).
- One call = energy + forces F = -dE/dR by autograd with `create_graph=False` for every model (Equiformer: upstream hard-codes True, rewritten by `--create-graph false`; GotenNet: output-head attribute; EquiformerV2: built without its direct-force head, `--forces autograd`; MACE `training=False`; ATOMTRICK eager in eval mode). ATOMTRICK compiled: one make_fx graph per batch size (forward + force backward, traced with train_mode=True, the only trace make_fx captures correctly) compiled with torch.compile(dynamic=True) and executed under `torch.no_grad()`, so no second-order graph is built at run time. Compiled baselines, `fused` recipe: `create_graph=True` inside the compiled graph (the traced `autograd.grad` must be differentiable), as recorded in their CSVs; `inductor` recipe: `create_graph=False` as eager.
- 10 warm-up calls (absorb make_fx trace + Inductor compile for ATOMTRICK compiled, and Dynamo trace + Inductor compile for the compiled baselines; see `compile_warmup_s` below) + 50 calls, each bracketed by CUDA events; latency = median; peak memory = `max_memory_allocated` over the timed calls (includes weights).
- Not equalised (architectural/harness facts, disclosed): neighbour list built inside the timed forward for Equiformer, EquiformerV2 and GotenNet (cost measured separately below), outside for ATOMTRICK (CPU cKDTree) and MACE; neighbour caps: Equiformer 1000, EquiformerV2 50, GotenNet 32 (+ self loops), ATOMTRICK/MACE none; EquiformerV2 = OC20 N@12_L@6_M@2 architecture (never trained on these data); GotenNet per-dataset configs; baselines compiled with two recipes each (same harness, inputs and precision as eager; whole energy + force callable under `torch.compile(dynamic=True)`, TorchInductor default mode, no CUDA graphs; parameters frozen with `requires_grad_(False)`; `PYTORCH_JIT=0` before `import torch`; e3nn TorchScript codegen off; Dynamo never reset between molecules; MACE and GotenNet compiled reuse one random-init model instance for all molecules so the dynamic-shape graph is shared): `inductor` = graph break at `torch.autograd.grad`, energy forward compiled, forces from AOTAutograd's compiled backward; `fused` = `torch.autograd.grad` allowed in graph, forward + force backward in ONE compiled graph (MACE's upstream torch.compile recipe, generalised), traced with grad enabled, outputs detached inside the compiled callable so AOTAutograd compiles an inference graph executed under no_grad (the shape of ATOMTRICK's make_fx step); after the warm-up every compiled row's energy and forces were compared with one eager call of the same model (`compile_max_rel_err`, mismatch -> compile_error); the PyG models still build their radius graph inside the timed forward when compiled; MACE compiled runs in its own environment (Python 3.11.17; torch 2.5.1 has no torch.compile on Python 3.13, the interpreter of the original MACE environment); each baseline runs in its own environment (torch versions below).

## Environments

| model | python | torch | CUDA (torch) | key packages | matmul precision while timing |
|---|---|---|---|---|---|
| atomtrick | 3.11.15 | 2.5.1+cu121 | 12.1 | scipy 1.17.1 | highest (bench --tf32 off) |
| equiformer | 3.10.20 | 2.5.1+cu124 | 12.4 | torch_geometric 2.7.0, torch_cluster 1.6.3+pt25cu124, e3nn 0.5.1 | highest |
| equiformer_v2 | 3.10.20 | 2.5.1+cu124 | 12.4 | torch_geometric 2.7.0, torch_cluster 1.6.3+pt25cu124, e3nn 0.5.1 | highest |
| gotennet | 3.10.20 | 2.5.1+cu124 | 12.4 | hydra 1.3.2, torch_geometric 2.7.0, torch_cluster 1.6.3+pt25cu124 | highest |
| mace | 3.11.17 | 2.5.1+cu121 | 12.1 | mace 0.3.15, e3nn 0.4.4, ase 3.29.0 | highest |
| mace_compiled | 3.11.17 | 2.5.1+cu121 | 12.1 | mace 0.3.15, e3nn 0.4.4, ase 3.29.0 | highest |

## Graph sizes (directed edges in the timed graph) and in-forward graph-build cost

| dataset | molecule | atoms | bs | ATOMTRICK | MACE | Equiformer | EquiformerV2 (cap 50) | GotenNet (cap 32, +loops) | graph build ms: Eqf / EqV2 / GotenNet |
|---|---|---|---|---|---|---|---|---|---|
| rmd17 | ethanol | 9 | 1 | 72 | 72 | 72 | 72 | 81 | 0.19 / 0.17 / 0.17 |
| rmd17 | malonaldehyde | 9 | 1 | 72 | 72 | 72 | 72 | 81 | 0.18 / 0.17 / 0.17 |
| rmd17 | benzene | 12 | 1 | 128 | 128 | 128 | 128 | 140 | 0.19 / 0.17 / 0.17 |
| rmd17 | uracil | 12 | 1 | 128 | 128 | 128 | 128 | 140 | 0.19 / 0.17 / 0.17 |
| rmd17 | toluene | 15 | 1 | 192 | 192 | 192 | 192 | 207 | 0.19 / 0.17 / 0.17 |
| rmd17 | salicylic | 16 | 1 | 210 | 210 | 210 | 210 | 226 | 0.19 / 0.17 / 0.17 |
| rmd17 | naphthalene | 18 | 1 | 254 | 254 | 254 | 254 | 272 | 0.18 / 0.18 / 0.17 |
| rmd17 | paracetamol | 20 | 1 | 280 | 280 | 280 | 280 | 300 | 0.19 / 0.17 / 0.17 |
| rmd17 | aspirin | 21 | 1 | 314 | 314 | 314 | 314 | 335 | 0.18 / 0.18 / 0.17 |
| rmd17 | azobenzene | 24 | 1 | 326 | 326 | 326 | 326 | 350 | 0.19 / 0.18 / 0.17 |
| md22 | Ac_Ala3_NHMe | 42 | 1 | 1086 | 1086 | 1086 | 1086 | 1093 | 0.20 / 0.18 / 0.18 |
| md22 | DHA | 56 | 1 | 1474 | 1474 | 1474 | 1474 | 1448 | 0.19 / 0.18 / 0.17 |
| md22 | AT_AT | 60 | 1 | 1476 | 1476 | 1476 | 1476 | 1510 | 0.19 / 0.18 / 0.18 |
| md22 | stachyose | 87 | 1 | 2750 | 2750 | 2750 | 2745 | 2495 | 0.20 / 0.19 / 0.18 |
| md22 | AT_AT_CG_CG | 118 | 1 | 3536 | 3536 | 3536 | 3529 | 3245 | 0.20 / 0.20 / 0.19 |
| md22 | buckyball_catcher | 148 | 1 | 4618 | 4618 | 4618 | 4602 | 4248 | 0.20 / 0.21 / 0.20 |
| md22 | double_walled_nanotube | 370 | 1 | 14480 | 14480 | 14480 | 13614 | 11283 | 0.25 / 0.25 / 0.25 |
| rmd17 | ethanol | 9 | 4 | 288 | 288 | 288 | 288 | 324 | 0.20 / 0.19 / 0.18 |
| rmd17 | malonaldehyde | 9 | 4 | 288 | 288 | 288 | 288 | 324 | 0.20 / 0.19 / 0.18 |
| rmd17 | benzene | 12 | 4 | 518 | 518 | 518 | 518 | 566 | 0.20 / 0.19 / 0.18 |
| rmd17 | uracil | 12 | 4 | 516 | 516 | 516 | 516 | 564 | 0.20 / 0.19 / 0.18 |
| rmd17 | toluene | 15 | 4 | 768 | 768 | 768 | 768 | 828 | 0.20 / 0.19 / 0.18 |
| rmd17 | salicylic | 16 | 4 | 832 | 832 | 832 | 832 | 896 | 0.19 / 0.19 / 0.19 |
| rmd17 | naphthalene | 18 | 4 | 1024 | 1024 | 1024 | 1024 | 1096 | 0.19 / 0.19 / 0.18 |
| rmd17 | paracetamol | 20 | 4 | 1110 | 1110 | 1110 | 1110 | 1190 | 0.20 / 0.19 / 0.18 |
| rmd17 | aspirin | 21 | 4 | 1208 | 1208 | 1208 | 1208 | 1292 | 0.20 / 0.19 / 0.18 |
| rmd17 | azobenzene | 24 | 4 | 1316 | 1316 | 1316 | 1316 | 1412 | 0.20 / 0.20 / 0.19 |
| md22 | Ac_Ala3_NHMe | 42 | 4 | 4334 | 4334 | 4334 | 4334 | 4360 | 0.20 / 0.20 / 0.20 |
| md22 | DHA | 56 | 4 | 5928 | 5928 | 5928 | 5928 | 5822 | 0.21 / 0.21 / 0.21 |
| md22 | AT_AT | 60 | 4 | 5928 | 5928 | 5928 | 5928 | 6060 | 0.21 / 0.21 / 0.21 |
| md22 | stachyose | 87 | 4 | 10984 | 10984 | 10984 | 10964 | 9975 | 0.22 / 0.22 / 0.22 |
| md22 | AT_AT_CG_CG | 118 | 4 | 14128 | 14128 | 14128 | 14105 | 12955 | 0.22 / 0.23 / 0.22 |
| md22 | buckyball_catcher | 148 | 4 | 18480 | 18480 | 18480 | 18413 | 16981 | 0.23 / 0.23 / 0.22 |
| md22 | double_walled_nanotube | 370 | 4 | 57956 | 57956 | 57956 | 54507 | 45169 | 0.28 / 0.26 / 0.26 |

## Compiled baselines

Per model and torch.compile recipe (`inductor`: graph break at `autograd.grad`; `fused`: `autograd.grad` inlined in one graph). compile_warmup_s = wall-clock seconds of the 10 warm-up calls (trace + compile, excluded from the latency); compile_graphs_warmup = unique graphs compiled during a row's warm-up; compile_graph_breaks = graph-break count delta during the warm-up (sum over rows, median per row); max rel. err = largest `compile_max_rel_err` (compiled vs eager energy / forces after the warm-up, max relative error; the harness fails the row above 1e-2); compiled / eager = geometric-mean ratio of the median latency over the (dataset, molecule, bs) rows ok in both. A row that compiled a new graph during the timed calls (compile_graphs_timed > 0), or a recipe whose ok rows compiled no graph at all, is a failed check.

| model | recipe | rows ok / oom / compile_error | compile_warmup_s median / max | max compile_graphs_warmup | compile_graph_breaks sum (median per row) | rows compile_graphs_timed > 0 | rows re-timed | max rel. err vs eager | compiled / eager latency (n) |
|---|---|---|---|---|---|---|---|---|---|
| Equiformer | fused | 22 / 12 / 0 (of 34) | 1.0 / 120.5 | 64 | 84 (0) | 0 | 0 | 1.6e-06 (22 rows) | 0.716 (22) |
| Equiformer | inductor | 34 / 0 / 0 (of 34) | 2.2 / 203.7 | 63 | 84 (0) | 0 | 0 | 4.7e-06 (34 rows) | 0.554 (34) |
| EquiformerV2 | fused | 1 / 33 / 0 (of 34) | 34.4 / 34.4 | 35 | 20 (20) | 0 | 0 | 9.2e-07 (1 rows) | 1.289 (1) |
| EquiformerV2 | inductor | 33 / 1 / 0 (of 34) | 3.5 / 98.9 | 35 | 40 (0) | 0 | 0 | 3.1e-06 (33 rows) | 0.961 (33) |
| GotenNet | fused | 33 / 1 / 0 (of 34) | 0.3 / 45.3 | 16 | 34 (0) | 0 | 0 | 1.1e-05 (33 rows) | 0.450 (33) |
| GotenNet | inductor | 33 / 1 / 0 (of 34) | 0.5 / 95.2 | 15 | 32 (0) | 0 | 0 | 1.2e-05 (33 rows) | 0.437 (33) |
| MACE | fused | 34 / 0 / 0 (of 34) | 0.1 / 102.8 | 1 | 0 (0) | 0 | 0 | 3.5e-06 (34 rows) | 0.350 (34) |
| MACE | inductor | 34 / 0 / 0 (of 34) | 0.1 / 98.4 | 3 | 12 (0) | 0 | 0 | 6.9e-06 (34 rows) | 0.370 (34) |

## Equiformer: fused vs inductor recipe

fused / inductor, median latency: geometric-mean 1.235, range 1.156-1.317 over 22 (molecule, bs) rows.

## EquiformerV2: fused vs inductor recipe

fused / inductor, median latency: geometric-mean 1.329, range 1.329-1.329 over 1 (molecule, bs) rows.

## GotenNet: fused vs inductor recipe

fused / inductor, median latency: geometric-mean 1.029, range 0.999-1.054 over 33 (molecule, bs) rows.

## MACE: fused vs inductor recipe

fused / inductor, median latency: geometric-mean 0.948, range 0.870-1.033 over 34 (molecule, bs) rows.

## Extra: ATOMTRICK pole-free vs Euler edge frame (eager)

polefree (default) / euler (direct Euler-angle evaluation), median latency: geometric-mean 0.901, range 0.816-1.001 over 34 (molecule, bs) rows.

## Extra: ATOMTRICK pole-free vs Euler edge frame (makefx)

polefree (default) / euler (direct Euler-angle evaluation), median latency: geometric-mean 0.950, range 0.748-1.015 over 34 (molecule, bs) rows.

## Consistency checks

- FAIL: results come from 2 jobs / 2 GPUs (['20320246', '20396545'], ['GPU-215d43ec-3ba2-b8cc-e123-cb7caa323c05', 'GPU-8d1cc5ff-05d8-1a4c-b7fe-a57ec51415a9']): not one session -- re-run with FORCE=1
- WARNING: equiformer compiled (fused): 3 ok rows with graph breaks during the warm-up (compile_graph_breaks > 0): the fused recipe did not give ONE graph there (outputs still matched eager): rmd17/ethanol bs1, rmd17/ethanol bs4, md22/Ac_Ala3_NHMe bs1
- WARNING: equiformer_v2 compiled (fused): 1 ok rows with graph breaks during the warm-up (compile_graph_breaks > 0): the fused recipe did not give ONE graph there (outputs still matched eager): rmd17/ethanol bs1
- WARNING: equiformer_v2 compiled (inductor): 4 ok rows compiled no new graph and are SLOWER than the eager row (possible eager fallback for that shape): rmd17/malonaldehyde bs1, rmd17/malonaldehyde bs4, rmd17/uracil bs1, rmd17/uracil bs4
- WARNING: gotennet compiled (fused): 4 ok rows with graph breaks during the warm-up (compile_graph_breaks > 0): the fused recipe did not give ONE graph there (outputs still matched eager): rmd17/ethanol bs1, rmd17/ethanol bs4, md22/Ac_Ala3_NHMe bs1, md22/Ac_Ala3_NHMe bs4

