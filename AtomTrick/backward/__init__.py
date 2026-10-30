"""make_fx compiled training/inference step — the only retained backward path.

The hand-coded manual-backward apparatus (per-op ``_fwd``/``_bwd`` modules, the
HC autograd-free 2nd backward, the compiled-HC inference path, and the
fused-Triton kernels) was removed on 2026-07-03.  Training now uses
``make_fx`` + ``torch.compile`` (:class:`makefx_compiled.MakeFXCompiledStep`)
and inference uses plain eager autograd (:meth:`AtomTrick.model.AtomTrick.forward`).
"""

from .makefx_compiled import MakeFXCompiledStep, MakeFXForwardModule, plainify

__all__ = ["MakeFXCompiledStep", "MakeFXForwardModule", "plainify"]
