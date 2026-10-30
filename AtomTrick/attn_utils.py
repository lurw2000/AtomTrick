import torch
from torch import Tensor


def _build_head_perm(irreps, n_heads):
    """Precompute a channel permutation that groups channels by head.

    Original layout:  [sector_0, sector_1, ...]
    Permuted layout:  [head_0_all_sectors, head_1_all_sectors, ...]

    Within each sector, multiplicities are split evenly across heads.
    """
    perm = []
    for h in range(n_heads):
        for i, (mult, irrep) in enumerate(irreps):
            sl = irreps._slices[i]
            mh = mult // n_heads
            start = sl.start + h * mh * irrep.dim
            perm.extend(range(start, start + mh * irrep.dim))
    return torch.tensor(perm, dtype=torch.long)


def sparse_softmax(scores, index, num_nodes):
    """Softmax over variable-size groups defined by ``index``.

    Parameters
    ----------
    scores : Tensor, shape (E, H)
    index  : LongTensor, shape (E,)
        Group assignment (destination node index).
    num_nodes : int

    Returns
    -------
    Tensor, shape (E, H)
    """
    idx_exp = index.unsqueeze(-1).expand_as(scores)
    max_vals = torch.full((num_nodes, scores.size(1)), -1e9,
                          device=scores.device, dtype=scores.dtype)
    max_vals = max_vals.scatter_reduce(0, idx_exp, scores,
                                       reduce="amax", include_self=False)
    scores = scores - max_vals[index]

    exp_scores = scores.exp()
    sum_exp = torch.zeros(num_nodes, scores.size(1),
                          device=scores.device, dtype=scores.dtype)
    sum_exp = sum_exp.scatter_add(0, idx_exp, exp_scores)
    return exp_scores / (sum_exp[index] + 1e-8)
