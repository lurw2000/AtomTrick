"""O(3)-equivariant tensor algebra primitives.

Provides irreducible representations, equivariant linear maps, tensor
products, normalization, activations, and Wigner D-matrix utilities.
"""

from .irreps import Irrep, Irreps
from .tensor import O3Tensor
from .linear import O3Linear, O2Linear, SO2Linear
from .norm import O3LayerNorm, O3SharedLayerNorm
from .activation import O3GatedActivation
from .inner import o3_inner
from .wigner import (
    wigner_D,
    wigner_D_from_matrix,
    precompute_wigner_D_from_matrix,
    wigner_D_block,
    rotate_irreps,
)
from .utils import batched_rotation_to_z

__all__ = [
    "Irrep",
    "Irreps",
    "O3Tensor",
    "O3Linear",
    "O2Linear",
    "SO2Linear",
    "O3LayerNorm",
    "O3SharedLayerNorm",
    "O3GatedActivation",
    "o3_inner",
    "wigner_D",
    "wigner_D_from_matrix",
    "precompute_wigner_D_from_matrix",
    "wigner_D_block",
    "rotate_irreps",
    "batched_rotation_to_z",
]
