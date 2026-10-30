# Third-party notices

ATOMTRICK's own code is released under the MIT License (`LICENSE`). Some files in this
repository are adapted from, or are patches against, third-party projects. Each such
component is listed below with its upstream repository, the commit this repository pins or
was tested against, its license, and its copyright and license notice, reproduced as in the
upstream license file. All of them are under the MIT License.

| Component | Files in this repository | Upstream | License |
|---|---|---|---|
| TorchMD-Net | `AtomTrick/radial.py` (`ExpNormalSmearing`, `CosineCutoff`) | https://github.com/torchmd/torchmd-net | MIT |
| MACE | `benchmarks/latency/baselines/mace/` | https://github.com/ACEsuit/mace @ `667eee4e58d23a38ff5a75122109ec2025809649` | MIT |
| Equiformer | `benchmarks/latency/baselines/equiformer/` | https://github.com/atomicarchitects/equiformer @ `64cb7866f48b9aa156e74a9d6a2ef2663b367437` | MIT |
| EquiformerV2 | `benchmarks/latency/baselines/equiformer_v2/` | https://github.com/atomicarchitects/equiformer_v2 @ `d5ad4be729b56f74012ebb7f097f77c5b00a1004` | MIT |
| Open Catalyst Project (OCP) | `benchmarks/latency/baselines/equiformer_v2/ocpmodels_shim/ocpmodels/models/scn/smearing.py` | https://github.com/FAIR-Chem/fairchem (formerly Open-Catalyst-Project/ocp); notice as of `v0.1.0` (`ff25d66a3d24dbf8159611c70188bdbcf12a76e4`) | MIT |
| GotenNet | `benchmarks/latency/baselines/gotennet/` | https://github.com/sarpaykent/GotenNet @ `16c7026b23584d13db3c30bc0c80cecece775c1a` | MIT |

The harnesses and patches under `benchmarks/latency/baselines/` run inside, or are applied
to, a checkout of the upstream project, which is not redistributed here (see the README
there). The patches nevertheless contain upstream source lines as diff context, so the
upstream notices apply to them.

Ideas from other work that are reimplemented rather than copied are credited where they are
used, not here. No GotenNet code is included in the `AtomTrick` package.

## TorchMD-Net

`AtomTrick/radial.py`: `ExpNormalSmearing` and `CosineCutoff` are a close adaptation of the
classes of the same name in `torchmdnet/models/utils.py` (the cutoff_lower = 0 case,
renamed arguments).

Upstream: https://github.com/torchmd/torchmd-net (file `torchmdnet/models/utils.py`;
compared against `833c07708e7d507fc33bdcd835ed083a321cc651`).

```text
MIT License

Copyright (c) 2021-2023 Universitat Pompeu Fabra,  https://www.compscience.org

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## MACE

`benchmarks/latency/baselines/mace/` runs inside a MACE checkout at
`667eee4e58d23a38ff5a75122109ec2025809649` (v0.3.15 + 46 commits).

Upstream: https://github.com/ACEsuit/mace. Notice as in its `LICENSE.md` at that commit
(which ends after the permission notice):

```text
MIT License

Copyright (c) 2022 ACEsuit/mace

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.
```

## Equiformer

`benchmarks/latency/baselines/equiformer/`: latency harness and `nets_init.patch`, for a
checkout of Equiformer.

Upstream: https://github.com/atomicarchitects/equiformer @
`64cb7866f48b9aa156e74a9d6a2ef2663b367437`.

```text
MIT License

Copyright (c) 2023 Yi-Lun Liao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## EquiformerV2

`benchmarks/latency/baselines/equiformer_v2/`: latency harness and the `ocpmodels_shim/`
stub of the `ocpmodels` symbols EquiformerV2 imports, for a checkout of EquiformerV2.

Upstream: https://github.com/atomicarchitects/equiformer_v2 @
`d5ad4be729b56f74012ebb7f097f77c5b00a1004`.

```text
MIT License

Copyright (c) 2023 Yi-Lun Liao

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Open Catalyst Project

`benchmarks/latency/baselines/equiformer_v2/ocpmodels_shim/ocpmodels/models/scn/smearing.py`
is adapted from `ocpmodels/models/scn/smearing.py` of the Open Catalyst Project (same class
names and interface, which EquiformerV2 imports; the code is condensed and not a verbatim
copy) and keeps its license notice in the file. The rest of `ocpmodels_shim/` is
hand-written placeholders.

Upstream: https://github.com/Open-Catalyst-Project/ocp, now https://github.com/FAIR-Chem/fairchem.
No single upstream commit is pinned, because the file is not a copy of one. The notice
below is the repository's `LICENSE.md` at tag `v0.1.0`
(`ff25d66a3d24dbf8159611c70188bdbcf12a76e4`), matching the header of the upstream
`smearing.py`; the current `LICENSE.md` names Meta, Inc. and its affiliates instead.

```text
MIT License

Copyright (c) Facebook, Inc. and its affiliates.

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## GotenNet

`benchmarks/latency/baselines/gotennet/`: latency harness and `gotennet_local.patch`
(`git diff 16c7026 -- gotennet/` of the benchmarked checkout).

Upstream: https://github.com/sarpaykent/GotenNet @ `16c7026b23584d13db3c30bc0c80cecece775c1a`.

```text
MIT License

Copyright (c) 2025 Sarp Aykent

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
```

## Datasets

No dataset is redistributed in this repository; the scripts in `scripts/data/` download
them from their original sources. Each keeps its own terms:

| Dataset | Source | License / terms | Cite |
|---|---|---|---|
| rMD17 | figshare, DOI [10.6084/m9.figshare.12672038](https://doi.org/10.6084/m9.figshare.12672038) | CC0 1.0 | Christensen & von Lilienfeld (2020) |
| MD22 | http://www.sgdml.org/ | no license is stated on sgdml.org | Chmiela et al. (2023), *Sci. Adv.* 9, eadf0873 |
| SPICE | https://github.com/openmm/spice-dataset | data CC0 (the repository's code is MIT) | Eastman et al. (2023) |
| MACE-OFF23 train/test split of SPICE | University of Cambridge Apollo, DOI [10.17863/CAM.107498](https://doi.org/10.17863/CAM.107498) | MIT | Kovács et al. (MACE-OFF23) |
