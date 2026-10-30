#!/usr/bin/env python3
"""Environment probe for the latency driver run_all.sh (CPU-safe; no GPU needed).

usage: python benchmarks/latency/env_probe.py <model> [--repo-root DIR] [--json OUT]
  model in {atomtrick, equiformer, equiformer_v2, gotennet, mace}

Prints (and optionally writes as JSON) python / torch / CUDA / cuDNN versions, the float32
matmul precision and TF32 flags as they are AFTER importing the model code, the default dtype,
and the GPU name if one is visible.  Importing the model code is the smoke test.
"""
import argparse
import json
import os
import platform
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("model")
    ap.add_argument("--repo-root", default=None)
    ap.add_argument("--json", default=None)
    a = ap.parse_args()
    import torch
    info = {"model": a.model, "python": platform.python_version(),
            "torch": torch.__version__, "torch_cuda": torch.version.cuda,
            "cudnn": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
            "host": platform.node()}
    if a.repo_root:
        sys.path.insert(0, os.path.abspath(a.repo_root))
    mods = {}
    try:
        if a.model == "atomtrick":
            os.environ.setdefault("PYTORCH_JIT", "0")
            import AtomTrick  # noqa: F401
            from AtomTrick.backward.makefx_compiled import MakeFXCompiledStep  # noqa: F401
            from AtomTrick.trainer import Trainer  # noqa: F401
            import scipy
            mods["scipy"] = scipy.__version__
        elif a.model == "equiformer":
            from nets.graph_attention_transformer_md17 import GraphAttentionTransformerMD17  # noqa
            import torch_geometric, torch_cluster, e3nn
            mods.update(torch_geometric=torch_geometric.__version__,
                        torch_cluster=torch_cluster.__version__, e3nn=e3nn.__version__)
        elif a.model == "equiformer_v2":
            from nets.equiformer_v2.equiformer_v2_oc20 import EquiformerV2_OC20  # noqa
            import torch_geometric, torch_cluster, e3nn
            mods.update(torch_geometric=torch_geometric.__version__,
                        torch_cluster=torch_cluster.__version__, e3nn=e3nn.__version__)
        elif a.model == "gotennet":
            import hydra, torch_geometric, torch_cluster
            from gotennet.datamodules.components.md22 import MD22  # noqa
            from gotennet.datamodules.components.rmd17 import rMD17  # noqa
            from gotennet.models.components.outputs import AtomwiseV3  # noqa
            mods.update(hydra=hydra.__version__, torch_geometric=torch_geometric.__version__,
                        torch_cluster=torch_cluster.__version__)
        elif a.model == "mace":
            import mace, e3nn, ase
            from mace import modules  # noqa
            mods.update(mace=getattr(mace, "__version__", "?"), e3nn=e3nn.__version__,
                        ase=ase.__version__, mace_file=mace.__file__)
        info["import_ok"] = True
    except Exception as e:  # noqa: BLE001
        info["import_ok"] = False
        info["import_error"] = f"{type(e).__name__}: {e}"
    info["modules"] = mods
    info["float32_matmul_precision_after_import"] = torch.get_float32_matmul_precision()
    info["cuda_matmul_allow_tf32"] = torch.backends.cuda.matmul.allow_tf32
    info["cudnn_allow_tf32"] = torch.backends.cudnn.allow_tf32
    info["default_dtype"] = str(torch.get_default_dtype())
    info["cuda_available"] = torch.cuda.is_available()
    if torch.cuda.is_available():
        info["gpu"] = torch.cuda.get_device_name(0)
    print(json.dumps(info, indent=1))
    if a.json:
        with open(a.json, "w") as f:
            json.dump(info, f, indent=1)
    sys.exit(0 if info["import_ok"] else 1)


if __name__ == "__main__":
    main()
