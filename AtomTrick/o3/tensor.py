"""O3Tensor — lightweight wrapper pairing a torch.Tensor with O(3) irrep metadata.

``_O3TensorWrapper`` is a plain Python class that stores a reference to a
``torch.Tensor``: transparent to autograd, and unwrapped before anything is
traced by make_fx (the compiled path works on plain tensors).
"""
from __future__ import annotations

from typing import List, Tuple, Union, Iterator, Optional

import torch
from torch import Tensor

from .irreps import Irreps


# ======================================================================
# Wrapper implementation
# ======================================================================

def _rebuild_wrapper(data: Tensor, irreps):
    return _O3TensorWrapper(data, irreps)


class _O3TensorWrapper:
    """Lightweight wrapper pairing a :class:`torch.Tensor` with O(3) irrep metadata.

    The underlying tensor is stored in :attr:`tensor` and is never copied or
    re-wrapped, so autograd graphs are preserved unconditionally.
    """

    __slots__ = ("_tensor", "_irreps", "__dict__")

    def __init__(
        self,
        data: Tensor,
        irreps: Union[str, List, Irreps],
    ):
        if not isinstance(data, torch.Tensor):
            data = torch.as_tensor(data)
        if not isinstance(irreps, Irreps):
            irreps = Irreps(irreps)
        if data.ndim == 0 or data.shape[-1] != irreps.dim:
            raise ValueError(
                f"data last dim ({data.shape[-1] if data.ndim else 'scalar'}) "
                f"!= irreps.dim ({irreps.dim}) for irreps={irreps}"
            )
        self._tensor = data
        self._irreps = irreps

    # ------------------------------------------------------------------
    # Core accessors
    # ------------------------------------------------------------------

    @property
    def tensor(self) -> Tensor:
        """The underlying data tensor (autograd graph intact)."""
        return self._tensor

    @property
    def irreps(self) -> Irreps:
        return self._irreps

    @property
    def shape(self) -> torch.Size:
        return self._tensor.shape

    @property
    def batch_shape(self) -> torch.Size:
        return self._tensor.shape[:-1]

    @property
    def device(self):
        return self._tensor.device

    @property
    def dtype(self):
        return self._tensor.dtype

    def __getitem__(self, idx) -> "_O3TensorWrapper":
        """Index along batch dims, preserving irreps."""
        return _O3TensorWrapper(self._tensor[idx], self._irreps)

    def __add__(self, other: "_O3TensorWrapper") -> "_O3TensorWrapper":
        return _O3TensorWrapper(self._tensor + other._tensor, self._irreps)

    def __sub__(self, other: "_O3TensorWrapper") -> "_O3TensorWrapper":
        return _O3TensorWrapper(self._tensor - other._tensor, self._irreps)

    def __getattr__(self, name):
        # Delegate unknown attributes (abs, max, sum, norm, ...) to the tensor.
        # Guard against recursion when _tensor itself is not yet set.
        if name in ("_tensor", "_irreps"):
            raise AttributeError(name)
        return getattr(self._tensor, name)

    # ------------------------------------------------------------------
    # Irrep-specific accessors
    # ------------------------------------------------------------------

    def get(self, idx: int) -> Tensor:
        """Sub-tensor for entry *idx*, shape ``(*batch_shape, mult, 2l+1)``."""
        mult, irrep = self._irreps._entries[idx]
        chunk = self._tensor[..., self._irreps._slices[idx]]
        return chunk.reshape(*self.shape[:-1], mult, irrep.dim)

    def iter_by_irrep(self) -> Iterator[Tuple[int, int, Tensor]]:
        """Yield ``(l, p, sub_tensor)`` for each entry, in order."""
        raw = self._tensor
        batch = self.shape[:-1]
        slices = self._irreps._slices
        for i, (mult, irrep) in enumerate(self._irreps._entries):
            yield irrep.l, irrep.p, raw[..., slices[i]].reshape(*batch, mult, irrep.dim)

    def norm_by_irrep(self) -> List[Tensor]:
        """Per-entry L2 norm over the 2l+1 components.

        Returns a list where item *i* has shape ``(*batch_shape, mult_i)``.
        """
        raw = self._tensor
        batch = self.shape[:-1]
        slices = self._irreps._slices
        return [
            raw[..., slices[i]].reshape(*batch, mult, irrep.dim).norm(dim=-1)
            for i, (mult, irrep) in enumerate(self._irreps._entries)
        ]

    # ------------------------------------------------------------------
    # Factory class-methods
    # ------------------------------------------------------------------

    @classmethod
    def zeros(cls, irreps, *batch_shape, dtype=None, device=None):
        if not isinstance(irreps, Irreps):
            irreps = Irreps(irreps)
        return cls(torch.zeros(*batch_shape, irreps.dim, dtype=dtype, device=device), irreps)

    @classmethod
    def ones(cls, irreps, *batch_shape, dtype=None, device=None):
        if not isinstance(irreps, Irreps):
            irreps = Irreps(irreps)
        return cls(torch.ones(*batch_shape, irreps.dim, dtype=dtype, device=device), irreps)

    @classmethod
    def randn(cls, irreps, *batch_shape, dtype=None, device=None):
        if not isinstance(irreps, Irreps):
            irreps = Irreps(irreps)
        return cls(torch.randn(*batch_shape, irreps.dim, dtype=dtype, device=device), irreps)

    # ------------------------------------------------------------------
    # Static combinators
    # ------------------------------------------------------------------

    @staticmethod
    def cat(tensors: List["_O3TensorWrapper"], dim: int = 0) -> "_O3TensorWrapper":
        if not tensors:
            raise ValueError("Cannot concatenate an empty list")
        irreps = tensors[0]._irreps
        for t in tensors[1:]:
            if t._irreps != irreps:
                raise ValueError("All O3Tensors must have identical irreps for cat")
        return _O3TensorWrapper(torch.cat([t._tensor for t in tensors], dim=dim), irreps)

    @staticmethod
    def stack(tensors: List["_O3TensorWrapper"], dim: int = 0) -> "_O3TensorWrapper":
        if not tensors:
            raise ValueError("Cannot stack an empty list")
        irreps = tensors[0]._irreps
        for t in tensors[1:]:
            if t._irreps != irreps:
                raise ValueError("All O3Tensors must have identical irreps for stack")
        return _O3TensorWrapper(torch.stack([t._tensor for t in tensors], dim=dim), irreps)

    @staticmethod
    def direct_sum(a: "_O3TensorWrapper", b: "_O3TensorWrapper") -> "_O3TensorWrapper":
        return _O3TensorWrapper(
            torch.cat([a._tensor, b._tensor], dim=-1),
            a._irreps + b._irreps,
        )

    # ------------------------------------------------------------------
    # O(3) transformation
    # ------------------------------------------------------------------

    def transform(self, R: Tensor) -> "_O3TensorWrapper":
        """Transform under a rotation / reflection matrix."""
        from .wigner import rotate_irreps

        R = R.to(dtype=self._tensor.dtype, device=self._tensor.device)
        out = rotate_irreps(self._irreps, R, self._tensor)
        return _O3TensorWrapper(out, self._irreps)

    # ------------------------------------------------------------------
    # repr / pickling
    # ------------------------------------------------------------------

    def __repr__(self) -> str:
        return (
            f"O3Tensor(irreps={self._irreps._repr}, "
            f"batch_shape={tuple(self.shape[:-1])}, "
            f"dtype={self.dtype}, device={self.device})"
        )

    def __reduce_ex__(self, proto):
        return (_rebuild_wrapper, (self._tensor, self._irreps))


# ======================================================================
# Public API
# ======================================================================

O3Tensor = _O3TensorWrapper

# Keep legacy pickle helper working
def _rebuild_o3tensor(data: Tensor, irreps):
    return O3Tensor(data, irreps)
