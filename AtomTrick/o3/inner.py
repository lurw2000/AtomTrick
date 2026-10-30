import torch
from torch import Tensor
from .tensor import O3Tensor


def o3_inner(x: O3Tensor, y: O3Tensor) -> Tensor:
    """O(3)-invariant inner product of two O3Tensors.

    Computes the dot product along the irrep (last) dimension.
    Batch dimensions are broadcast.

    Parameters
    ----------
    x, y : O3Tensor
        Must have identical irreps. Batch shapes must be broadcastable.

    Returns
    -------
    Tensor
        Shape = broadcast(x.batch_shape, y.batch_shape). A scalar per batch element.
    """
    if x._irreps != y._irreps:
        raise ValueError(
            f"Irreps mismatch: {x._irreps} vs {y._irreps}"
        )
    raw_x = x.tensor
    raw_y = y.tensor
    return (raw_x * raw_y).sum(dim=-1)
